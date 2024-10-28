// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"

#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/lstm.hpp"
#include "intel_gpu/primitives/lstm_cell.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/slice.hpp"

namespace ov {
namespace intel_gpu {
static cldnn::activation_func GetActivationFunc(std::string name) {
    static const std::map<std::string, cldnn::activation_func> name_mapping = {
        {"sigmoid", cldnn::activation_func::logistic},
        {"tanh", cldnn::activation_func::hyperbolic_tan},
        {"relu", cldnn::activation_func::relu},
    };
    auto itr = name_mapping.find(name);
    if (itr != name_mapping.end())
        return itr->second;
    else
        return cldnn::activation_func::none;
}

template <typename T>
void GetLSTMActivationParams(const std::shared_ptr<T>& op,
                             std::vector<cldnn::activation_func>& activations,
                             std::vector<cldnn::activation_additional_params>& activation_params) {
    activations = { cldnn::activation_func::logistic,
                    cldnn::activation_func::hyperbolic_tan,
                    cldnn::activation_func::hyperbolic_tan };
    activation_params = {};
    auto op_activations = op->get_activations();
    if (!op_activations.empty()) {
        if (op_activations.size() != 3)
            OPENVINO_THROW("Wrong number of activations for LSTMCell op ", op->get_friendly_name());
        for (int i = 0; i < 3; i++) {
            auto af = GetActivationFunc(op_activations[i]);
            if (af == cldnn::activation_func::none)
                OPENVINO_THROW("Wrong or unsupported activation type ", op_activations[i], " for LSTMCell op ", op->get_friendly_name());
            activations[i] = af;
        }
    }
    auto op_a = op->get_activations_alpha();
    auto op_b = op->get_activations_beta();
    if (!op_a.empty()) {
        if (op_a.size() != 3 || op_b.size() != 3)
            OPENVINO_THROW("Wrong number of activation parameters for LSTMCell op ", op->get_friendly_name());
        for (int i = 0; i < 3; i++) {
            cldnn::activation_additional_params params = { op_a[i], op_b[i] };
            activation_params.push_back(cldnn::activation_additional_params(params));
        }
    }
}

static void CreateLSTMCellOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::LSTMCell>& op) {
    validate_inputs_count(op, {6});
    int lstm_batch_size, lstm_input_size, lstm_hidden_size;
    auto inputs = p.GetInputInfo(op);

    std::string layerName = layer_type_name_ID(op);
    cldnn::input_info weight = inputs[3];
    cldnn::input_info recurrent = inputs[4];
    cldnn::input_info bias = inputs[5];

    /* check incoming CNN layer and setup required variables */
    {
        const auto in0_pshape = op->get_input_partial_shape(0);
        const auto out0_pshape = op->get_output_partial_shape(0);

        if (in0_pshape[in0_pshape.size() - 1].is_static())
            lstm_input_size = in0_pshape[in0_pshape.size() - 1].get_length();
        else
            lstm_input_size = -1;

        if (in0_pshape[in0_pshape.size() - 2].is_static())
            lstm_batch_size = in0_pshape[in0_pshape.size() - 2].get_length();
        else
            lstm_batch_size = -1;

        if (out0_pshape[out0_pshape.size() - 1].is_static())
            lstm_hidden_size = out0_pshape[out0_pshape.size() - 1].get_length();
        else
            lstm_hidden_size = -1;
    }

    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();

    if (p.use_new_shape_infer()) {
        cldnn::primitive_id input_concatID = layerName + "_inputConcat";
        p.add_primitive(*op, cldnn::concatenation(input_concatID, { inputs[0], inputs[1] }, 1));

        cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected";
        cldnn::primitive_id lstm_elt_id = layerName + "_lstm_elt";
        cldnn::primitive_id wr_concat_id = layerName + "_WRconcat";
        p.add_primitive(*op, cldnn::concatenation(wr_concat_id, { inputs[3], inputs[4] }, 1));
        p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, cldnn::input_info(input_concatID), wr_concat_id, bias.pid));
        p.add_primitive(*op, cldnn::lstm_elt(lstm_elt_id, cldnn::input_info(lstm_fc_id), inputs[2].pid, clip, 0, activations,
                                            activation_params, cldnn::lstm_weights_order::fizo, 0));

        auto outSz = op->get_output_partial_shape(0);
        std::vector<int64_t> outSzPt;
        for (auto pshape : outSz) {
            if (pshape.is_static())
                outSzPt.push_back(pshape.get_length());
            else
                outSzPt.push_back(-1);
        }

        cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::split;
        size_t num_splits = 2;
        cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };

        cldnn::primitive_id outputHiddenCropID = layerName + "_hc";
        cldnn::primitive_id outputHiddenID = layerName + ".out0";
        cldnn::primitive_id outputDataID = layerName + "_data";

        cldnn::layout constLayout = cldnn::layout({}, cldnn::data_types::i64, cldnn::format::bfyx);
        cldnn::memory::ptr data_mem = p.get_engine().allocate_memory(constLayout, false);
        auto& stream = p.get_engine().get_service_stream();
        cldnn::mem_lock<char> lock{data_mem, stream};
        auto buf = lock.data();
        const int64_t axis = 1;
        std::memcpy(&buf[0], &axis, constLayout.bytes_count());
        p.add_primitive(*op,  cldnn::data(outputDataID, data_mem));

        p.add_primitive(*op,
                        cldnn::crop(outputHiddenCropID,
                        {cldnn::input_info(lstm_elt_id), cldnn::input_info(outputDataID)},
                        hiddenSz,
                        cldnn::tensor{0, 0, 0, 0},
                        op_mode, 0, axis, num_splits));
        p.add_primitive(*op, cldnn::reshape(outputHiddenID, cldnn::input_info(outputHiddenCropID),
                        false, outSzPt, op->get_output_partial_shape(0)), {layerName});

        cldnn::primitive_id outputCellCropID = layerName + "_cc";
        cldnn::primitive_id outputCellID = layerName + ".out1";
        p.add_primitive(*op,
                        cldnn::crop(outputCellCropID,
                        {cldnn::input_info(lstm_elt_id), cldnn::input_info(outputDataID)},
                        hiddenSz,
                        cldnn::tensor{0, 1, 0, 0},
                        op_mode, 1, axis, num_splits));
        p.add_primitive(*op, cldnn::reshape(outputCellID, cldnn::input_info(outputCellCropID),
                        false, outSzPt, op->get_output_partial_shape(1)));
    } else {
        //  LSTM primitive works with single precision for all in/out/weights tensors
        auto lstm_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

        cldnn::primitive_id inReshapeID = layerName + "_inReshape";
        cldnn::primitive_id permuteID = layerName + "_inputReorder";
        cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
        cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
        cldnn::primitive_id gemmReshapeID = layerName + "_gemmReshape";
        cldnn::primitive_id gemmReorderID = layerName + "_gemmReorder";
        cldnn::primitive_id input_concatID = layerName + "_inputConcat";

        cldnn::tensor inputShape = { lstm_batch_size, 1, lstm_input_size, 1 };
        cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
        cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
        cldnn::layout hiddenLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inStateShape);
        p.add_primitive(*op, cldnn::reshape(inReshapeID, inputs[0], inputShape));
        p.add_primitive(*op, cldnn::reorder(permuteID, inReshapeID, inputLayout));


        std::string hiddenInResh = inHiddenReshapeID + "_1";
        std::string hiddenInStr = inHiddenReorderID + "_1";
        std::string cellInResh = inHiddenReshapeID + "_2";
        std::string cellInStr = inHiddenReorderID + "_2";
        p.add_primitive(*op, cldnn::reshape(hiddenInResh, inputs[1], inStateShape));
        p.add_primitive(*op, cldnn::reorder(hiddenInStr, cldnn::input_info(hiddenInResh), hiddenLayout));
        p.add_primitive(*op, cldnn::reshape(cellInResh, inputs[2], inStateShape));
        p.add_primitive(*op, cldnn::reorder(cellInStr, cldnn::input_info(cellInResh), hiddenLayout));
        p.add_primitive(*op, cldnn::concatenation(input_concatID,
                                                { permuteID, hiddenInStr },
                                                3));

        cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
        cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
        cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
        cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

        std::string lstm_fc_id = layerName + "_fully_connected";
        std::string lstm_elt_id = layerName + "_lstm_elt";

        cldnn::primitive_id WRconcatID = layerName + "_WRconcat";
        p.add_primitive(*op, cldnn::concatenation(WRconcatID, { weight, recurrent }, 1));

        cldnn::primitive_id FCInputReshapeID = "Reshape_bf_" + lstm_fc_id + "_for_input";
        cldnn::tensor FCInputReshapeSz = { lstm_batch_size, inputShape.spatial[0] + inStateShape.spatial[0], 1, 1 };
        p.add_primitive(*op, cldnn::reshape(FCInputReshapeID, cldnn::input_info(input_concatID), FCInputReshapeSz));

        p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, cldnn::input_info(FCInputReshapeID), WRconcatID, bias.pid));
        p.add_primitive(*op, cldnn::reshape(gemmReshapeID, cldnn::input_info(lstm_fc_id), gemmSz));
        p.add_primitive(*op, cldnn::reorder(gemmReorderID, cldnn::input_info(gemmReshapeID), gemmLayout));
        p.add_primitive(*op, cldnn::lstm_elt(lstm_elt_id, cldnn::input_info(gemmReorderID), cellInStr, clip, 0, activations,
                                            activation_params, cldnn::lstm_weights_order::fizo, 0));


        cldnn::tensor outSz = cldnn::tensor{ lstm_batch_size, lstm_hidden_size, 1, 1 };
        cldnn::primitive_id outputHiddenCropID = layerName + "_hc";
        cldnn::primitive_id outputHiddenID = layerName + ".out0";
        p.add_primitive(*op, cldnn::crop(outputHiddenCropID, cldnn::input_info(lstm_elt_id), hiddenSz, cldnn::tensor{0, 0, 0, 0}));
        p.add_primitive(*op, cldnn::reshape(outputHiddenID, cldnn::input_info(outputHiddenCropID), outSz), {layerName});

        cldnn::primitive_id outputCellCropID = layerName + "_cc";
        cldnn::primitive_id outputCellID = layerName + ".out1";
        p.add_primitive(*op, cldnn::crop(outputCellCropID, cldnn::input_info(lstm_elt_id), hiddenSz, cellCropSz));
        p.add_primitive(*op, cldnn::reshape(outputCellID, cldnn::input_info(outputCellCropID), outSz));
    }
}

static void CreateLSTMSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::LSTMSequence>& op) {
    validate_inputs_count(op, {7});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    if (op->get_input_shape(2).size() != 3 || op->get_input_shape(3).size() != 1 \
        || op->get_input_shape(4).size() != 3 || op->get_input_shape(5).size() != 3 || op->get_input_shape(6).size() != 2)
        OPENVINO_THROW("Wrong input shapes for LSTMSequence op ", op->get_friendly_name());
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    cldnn::primitive_id lstm_seq_id = layerName;
    auto mutable_precision_firstsecond = op->get_output_element_type(1);
    unsigned int direction = op->get_direction() == ov::op::RecurrentSequenceDirection::REVERSE ? 1 : 0;

    if (p.use_new_shape_infer()) {
        cldnn::lstm_seq prim(layerName, inputs[0], inputs[1], \
            inputs[2], inputs[4], inputs[5], inputs[6], inputs[3], "", "", \
            clip, false, activations, activation_params, cldnn::lstm_weights_order::fizo, direction, cldnn::padding(), \
            static_cast<int>(op->get_output_size()));
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
        return;
    }

    cldnn::layout out12Layout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_firstsecond),
                cldnn::format::bfyx,
                tensor_from_dims(op->get_output_shape(1)));

    std::vector<cldnn::memory::ptr> shared_memories;
    shared_memories.push_back(p.get_engine().allocate_memory(out12Layout));
    const cldnn::primitive_id mutable_id_1 = layerName + "_md_write1";
    const cldnn::mutable_data mutable_prim_1{mutable_id_1, shared_memories.front()};
    p.add_primitive(*op, mutable_prim_1);
    shared_memories.push_back(p.get_engine().allocate_memory(out12Layout));
    const cldnn::primitive_id mutable_id_2 = layerName + "_md_write2";
    const cldnn::mutable_data mutable_prim_2{mutable_id_2, shared_memories.back()};
    p.add_primitive(*op, mutable_prim_2);
    cldnn::lstm_seq prim(lstm_seq_id + ".out0", inputs[0], inputs[1], \
        inputs[2], inputs[4], inputs[5], inputs[6], inputs[3], mutable_id_1, mutable_id_2, \
        clip, false, activations, activation_params, cldnn::lstm_weights_order::fizo, direction);
    p.add_primitive(*op, prim);
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out1", {cldnn::input_info(lstm_seq_id + ".out0")}, shared_memories.front()));
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out2", {cldnn::input_info(lstm_seq_id + ".out0")}, shared_memories.back()));
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace intel_gpu
}  // namespace ov
