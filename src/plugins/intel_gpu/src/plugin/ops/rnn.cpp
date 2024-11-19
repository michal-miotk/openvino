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
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    assert(!inputs[5].pid.empty());
    cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected";
    p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, inputs[0], inputs[3].pid, inputs[5].pid));
    if (p.use_new_shape_infer()) {
        auto prim = cldnn::lstm_cell(layerName+".out0", cldnn::input_info(lstm_fc_id), inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], \
                    cldnn::input_info(), "",  layerName + "_md_write.1", clip, false, activations, \
                    activation_params, cldnn::lstm_weights_order::fizo, ov::op::RecurrentSequenceDirection::FORWARD, cldnn::padding(), \
                    static_cast<int>(op->get_output_size()));
        p.add_primitive(*op, prim);
        return;
    }

    auto mutable_precision_first = op->get_output_element_type(1);
    cldnn::layout outLayout = cldnn::layout(
            cldnn::element_type_to_data_type(mutable_precision_first),
            cldnn::format::get_default_format(op->get_output_shape(1).size()),
            tensor_from_dims(op->get_output_shape(1)));

    cldnn::memory::ptr shared_memory = p.get_engine().allocate_memory(outLayout);
    const cldnn::primitive_id mutable_id_1 = layerName + "_md_write.1";
    const cldnn::mutable_data mutable_prim_1{mutable_id_1, shared_memory};
    p.add_primitive(*op, mutable_prim_1);

    p.add_primitive(*op, cldnn::lstm_cell(layerName+".out0", cldnn::input_info(lstm_fc_id), inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], \
                    cldnn::input_info(), "",  layerName + "_md_write.1", clip, false, activations, \
                    activation_params, cldnn::lstm_weights_order::fizo, ov::op::RecurrentSequenceDirection::FORWARD, cldnn::padding(), 1));

    p.add_primitive(*op, cldnn::mutable_data(layerName + ".out1", {cldnn::input_info(layerName + ".out0")}, shared_memory));
}

static void CreateLSTMSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::LSTMSequence>& op) {
    validate_inputs_count(op, {7});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    auto max_seq_len = op->get_input_partial_shape(0)[1];
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    if (max_seq_len.get_max_length() == 1) {
        int lstm_batch_size, lstm_input_size, lstm_hidden_size, lstm_sequence_len;
        cldnn::input_info weight = inputs[4];
        cldnn::input_info recurrent = inputs[5];
        cldnn::input_info bias = inputs[6];
        {
            const auto in_dims0 = op->get_input_shape(0);
            const auto out_dims0 = op->get_output_shape(0);
            if (in_dims0.size() != 3 ||
                op->get_input_shape(1).size() != 3 ||
                op->get_input_shape(2).size() != 3)
                OPENVINO_THROW("Wrong input shapes for LSTMSequence op ", op->get_friendly_name());

            lstm_input_size = static_cast<int>(in_dims0.back());
            lstm_sequence_len = static_cast<int>(in_dims0.at(in_dims0.size() - 2));
            lstm_batch_size = static_cast<int>(in_dims0.at(in_dims0.size() - 3));
            lstm_hidden_size = static_cast<int>(out_dims0.back());
        }

        bool isForward = op->get_direction() == ov::op::RecurrentSequenceDirection::FORWARD;

        //  LSTM primitive works with single precision for all in/out/weights tensors
        auto lstm_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

        cldnn::primitive_id inReshapeID = layerName + "_inReshape";
        cldnn::primitive_id permuteID = layerName + "_inputReorder";
        cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
        cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
        cldnn::primitive_id inHiddenStateID = inHiddenReshapeID + "_1";
        cldnn::primitive_id inCellStateID = inHiddenReshapeID + "_2";

        std::vector<cldnn::input_info> output_ids_offsets;

        cldnn::tensor inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, 1 };
        cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
        cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
        p.add_primitive(*op, cldnn::reshape(inReshapeID, inputs[0], inputShape));
        p.add_primitive(*op, cldnn::reorder(permuteID, cldnn::input_info(inReshapeID), inputLayout));

        p.add_primitive(*op, cldnn::reshape(inHiddenStateID, inputs[1], inStateShape));
        p.add_primitive(*op, cldnn::reshape(inCellStateID, inputs[2], inStateShape));

        cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
        cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
        cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
        cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};
        cldnn::primitive_id hiddenStr = inHiddenReshapeID + "_1";
        cldnn::primitive_id cellStr = inHiddenReshapeID + "_2";
        cldnn::primitive_id inputCropID = layerName + "_inputCrop";

        cldnn::primitive_id wr_concat_id = layerName + "_WRconcat";
        p.add_primitive(*op, cldnn::concatenation(wr_concat_id, { weight, recurrent }, 2));

        std::vector<size_t> WRreshapeSize = { 4 * size_t(lstm_hidden_size), size_t(lstm_input_size + lstm_hidden_size) };
        cldnn::primitive_id WRreshapeID = wr_concat_id + "_reshape";
        auto reshapeInPrim = cldnn::reshape(WRreshapeID, cldnn::input_info(wr_concat_id), tensor_from_dims(WRreshapeSize));
        p.add_primitive(*op, reshapeInPrim);

        for (int i = 0; i < lstm_sequence_len; ++i) {
            const std::string id_str = std::to_string(i);
            cldnn::primitive_id concatID = layerName + "_inputConcat" + id_str;
            cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected" + id_str;
            cldnn::primitive_id fc_input_resh_id = "Reshape_bf_" + lstm_fc_id + "_for_input" + id_str;
            cldnn::primitive_id lstm_fc_resh_id = layerName + "_gemmReshape" + id_str;
            cldnn::primitive_id lstm_fc_reor_id = layerName + "_gemmReorder" + id_str;
            cldnn::primitive_id lstm_elt_id = layerName + "_lstm_elt" + id_str;
            cldnn::primitive_id crop_id = layerName + "_crop" + id_str;

            int seqIdx = isForward ? i : lstm_sequence_len - 1 - i;
            const std::string seqIdx_str = std::to_string(seqIdx);

            cldnn::tensor crop_tensor{ inputShape.batch[0], 1, inputShape.spatial[0], inputShape.spatial[1] };
            cldnn::tensor offset_tensor{ 0, static_cast<cldnn::tensor::value_type>(seqIdx), 0, 0 };
            cldnn::primitive_id inputCrop_id = inputCropID + ":" + seqIdx_str;
            p.add_primitive(*op, cldnn::crop(inputCrop_id, cldnn::input_info(permuteID), crop_tensor, offset_tensor));

            p.add_primitive(*op, cldnn::concatenation(concatID, { cldnn::input_info(inputCrop_id), cldnn::input_info(hiddenStr) }, 3));

            cldnn::tensor fc_input_resh_tensor = { crop_tensor.batch[0], crop_tensor.spatial[0] + inStateShape.spatial[0],
                                                crop_tensor.feature[0], crop_tensor.spatial[1]};
            p.add_primitive(*op, cldnn::reshape(fc_input_resh_id, cldnn::input_info(concatID), fc_input_resh_tensor));

            p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, fc_input_resh_id, WRreshapeID, bias.pid));

            p.add_primitive(*op, cldnn::reshape(lstm_fc_resh_id, cldnn::input_info(lstm_fc_id), gemmSz));
            p.add_primitive(*op, cldnn::reorder(lstm_fc_reor_id, cldnn::input_info(lstm_fc_resh_id), gemmLayout));
            p.add_primitive(*op, cldnn::lstm_elt(lstm_elt_id, cldnn::input_info(lstm_fc_reor_id), cellStr, clip, 0, activations,
                                                activation_params, cldnn::lstm_weights_order::fizo, 0));

            hiddenStr = crop_id + ":hidden";
            cellStr = crop_id + ":cell";
            p.add_primitive(*op, cldnn::crop(hiddenStr, cldnn::input_info(lstm_elt_id), hiddenSz, cldnn::tensor{ 0, 0, 0, 0 }));
            output_ids_offsets.push_back(cldnn::input_info(hiddenStr));

            if (i < lstm_sequence_len - 1) {
                p.add_primitive(*op, cldnn::crop(cellStr, cldnn::input_info(lstm_elt_id), hiddenSz, cellCropSz));
            } else {
                // last hidden state crop (output 2)

                // last cell state crop (output 3)
                p.add_primitive(*op, cldnn::crop(cellStr, cldnn::input_info(lstm_elt_id), hiddenSz, cellCropSz));
            }
        }

        if (!isForward) std::reverse(output_ids_offsets.begin(), output_ids_offsets.end());
        // concatenated hidden state (output 1)
        cldnn::primitive_id concatStr = layerName + ":hiddenConcat";
        p.add_primitive(*op, cldnn::concatenation(concatStr, output_ids_offsets, 1));

        p.add_primitive(*op, cldnn::reshape(layerName + ".out0", concatStr, tensor_from_dims(op->get_output_shape(0))), {layerName});
        p.add_primitive(*op, cldnn::reshape(layerName + ".out1", hiddenStr, tensor_from_dims(op->get_output_shape(1))));
        p.add_primitive(*op, cldnn::reshape(layerName + ".out2", cellStr, tensor_from_dims(op->get_output_shape(2))));
    } else {
        if (op->get_input_shape(2).size() != 3 || op->get_input_shape(3).size() != 1 \
            || op->get_input_shape(4).size() != 3 || op->get_input_shape(5).size() != 3 || op->get_input_shape(6).size() != 2)
            OPENVINO_THROW("Wrong input shapes for LSTMSequence op ", op->get_friendly_name());
        auto mutable_precision_firstsecond = op->get_output_element_type(1);
        auto direction = op->get_direction();

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
        cldnn::lstm_seq prim(layerName + ".out0", inputs[0], inputs[1], \
            inputs[2], inputs[4], inputs[5], inputs[6], inputs[3], mutable_id_1, mutable_id_2, \
            clip, false, activations, activation_params, cldnn::lstm_weights_order::fizo, direction);
        p.add_primitive(*op, prim);
        p.add_primitive(*op, cldnn::mutable_data(layerName + ".out1", {cldnn::input_info(layerName + ".out0")}, shared_memories.front()));
        p.add_primitive(*op, cldnn::mutable_data(layerName + ".out2", {cldnn::input_info(layerName + ".out0")}, shared_memories.back()));
    }
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace intel_gpu
}  // namespace ov
