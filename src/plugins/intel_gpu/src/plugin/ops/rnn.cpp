// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/gru_sequence.hpp"

#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/lstm_cell.hpp"
#include "intel_gpu/primitives/gru_seq.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/slice.hpp"

namespace ov::intel_gpu {
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

template <typename T>
void GetGRUActivationParams(const std::shared_ptr<T>& op,
                             std::vector<cldnn::activation_func>& activations,
                             std::vector<cldnn::activation_additional_params>& activation_params) {
    activations = { cldnn::activation_func::logistic,
                    cldnn::activation_func::hyperbolic_tan };
    activation_params = {};
    auto op_activations = op->get_activations();
    if (!op_activations.empty()) {
        if (op_activations.size() != 2)
            OPENVINO_THROW("Wrong number of activations for GRUSeq op ", op->get_friendly_name());
        for (int i = 0; i < 2; i++) {
            auto af = GetActivationFunc(op_activations[i]);
            if (af == cldnn::activation_func::none)
                OPENVINO_THROW("Wrong or unsupported activation type ", op_activations[i], " for GRUSeq op ", op->get_friendly_name());
            activations[i] = af;
        }
    }
    auto op_a = op->get_activations_alpha();
    auto op_b = op->get_activations_beta();
    if (!op_a.empty()) {
        if (op_a.size() != 2 || op_b.size() != 2)
            OPENVINO_THROW("Wrong number of activation parameters for GRU op ", op->get_friendly_name());
        for (int i = 0; i < 2; i++) {
            cldnn::activation_additional_params params = { op_a[i], op_b[i] };
            activation_params.push_back(cldnn::activation_additional_params(params));
        }
    }
}

static void CreateGRUSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::GRUSequence>& op) {
    validate_inputs_count(op, {6});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    auto max_seq_len = op->get_input_partial_shape(0)[1];
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetGRUActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    if (op->get_input_shape(2).size() != 1 || op->get_input_shape(3).size() != 3 \
            || op->get_input_shape(4).size() != 3 || op->get_input_shape(5).size() != 2)
            OPENVINO_THROW("Wrong input shapes for GRUSequence op ", op->get_friendly_name());
    auto direction = op->get_direction();

    OPENVINO_ASSERT(p.use_new_shape_infer());
    cldnn::gru_seq prim(layerName,  inputs[0], inputs[1], cldnn::input_info(""), inputs[3], inputs[4], inputs[5], inputs[2],
        clip, false, activations, activation_params, cldnn::lstm_weights_order::fizo, direction, static_cast<int>(op->get_output_size()));
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

static void CreateLSTMCellOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::LSTMCell>& op) {
    validate_inputs_count(op, {6});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    OPENVINO_ASSERT(!inputs[5].pid.empty());
    OPENVINO_ASSERT(p.use_new_shape_infer());
    p.add_primitive(*op, cldnn::lstm_cell(layerName, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], cldnn::input_info(),
        clip, false, activations, activation_params, cldnn::lstm_weights_order::fizo, ov::op::RecurrentSequenceDirection::FORWARD,
        static_cast<int>(op->get_output_size())));
}

static void CreateLSTMSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::LSTMSequence>& op) {
    validate_inputs_count(op, {7});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    const float clip = op->get_clip();
    OPENVINO_ASSERT(op->get_input_shape(2).size() == 3 && op->get_input_shape(3).size() == 1 && op->get_input_shape(4).size() == 3 &&
        op->get_input_shape(5).size() == 3 && op->get_input_shape(6).size() == 2, "Wrong input shapes for LSTMSequence op ", op->get_friendly_name());
    auto direction = op->get_direction();

<<<<<<< HEAD
    OPENVINO_ASSERT(p.use_new_shape_infer());
    cldnn::lstm_seq prim(layerName, inputs[0], inputs[1], inputs[2], inputs[4], inputs[5], inputs[6], inputs[3], clip, false, activations,
        activation_params, cldnn::lstm_weights_order::fizo, direction, static_cast<int>(op->get_output_size()));
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
=======
    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstm_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id inHiddenStateID = inHiddenReshapeID + "_1";
    cldnn::primitive_id inCellStateID = inHiddenReshapeID + "_2";

    cldnn::tensor inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, 1 };
    cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
    p.add_primitive(*op, cldnn::reshape(inReshapeID, inputs[0], inputShape));
    p.add_primitive(*op, cldnn::reorder(permuteID, cldnn::input_info(inReshapeID), inputLayout));

    p.add_primitive(*op, cldnn::reshape(inHiddenStateID, inputs[1], inStateShape));
    p.add_primitive(*op, cldnn::reshape(inCellStateID, inputs[2], inStateShape));

    cldnn::primitive_id wr_concat_id = layerName + "_WRconcat";
    p.add_primitive(*op, cldnn::concatenation(wr_concat_id, { weight, recurrent }, 2));

    std::vector<size_t> WRreshapeSize = { 4 * size_t(lstm_hidden_size), size_t(lstm_input_size + lstm_hidden_size) };
    cldnn::primitive_id WRreshapeID = wr_concat_id + "_reshape";
    auto reshapeInPrim = cldnn::reshape(WRreshapeID, cldnn::input_info(wr_concat_id), tensor_from_dims(WRreshapeSize));
    p.add_primitive(*op, reshapeInPrim);

    auto a = op->get_output_shape(1);
    auto b = op->get_output_shape(2);
    auto mutable_precision_first = op->get_output_element_type(1);
    cldnn::layout out1Layout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_first),
                cldnn::format::get_default_format(op->get_output_shape(1).size()),
                tensor_from_dims(op->get_output_shape(1)));
    cldnn::memory::ptr shared_memory1 = p.get_engine().allocate_memory(out1Layout);

    auto mutable_precision_second = op->get_output_element_type(2);
    cldnn::layout out2Layout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_second),
                cldnn::format::get_default_format(op->get_output_shape(2).size()),
                tensor_from_dims(op->get_output_shape(2)));
    cldnn::memory::ptr shared_memory2 = p.get_engine().allocate_memory(out2Layout);

    cldnn::primitive_id lstm_seq_id = layerName;// + "_lstm_seq";
    p.add_primitive(*op, cldnn::lstm_seq(lstm_seq_id + ".out0", cldnn::input_info(permuteID), cldnn::input_info(inHiddenStateID), \
    cldnn::input_info(inCellStateID), cldnn::input_info(WRreshapeID), cldnn::input_info(bias), inCellStateID, clip, 0, activations, \
                                            activation_params, cldnn::lstm_weights_order::fizo, 0));

    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out1", shared_memory1));
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out2", shared_memory2));
>>>>>>> d461e6623f (19jul)
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, GRUSequence);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace ov::intel_gpu
