// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "activation.hpp"
#include <vector>
#include <algorithm>
#include <string>
#include "intel_gpu/graph/serialization/activation_serializer.hpp"

namespace cldnn {

/// @brief Weights orders
/// @details Specifies the order in which the weights are concatenated.
/// e.g. [i, o, f, z] : [input, output, forget, block]
/// ONNX order: iofz
/// Caffe order: ifoz
/// pyTorch order: izof
/// OV order: fizo
enum class lstm_weights_order {
    iofz,
    ifoz,
    izof,
    fizo
};

template <typename PType>
struct RNNParams : public primitive_base<PType> {
    RNNParams() : primitive_base<PType>("", {}) {}
    RNNParams(const RNNParams&) = default;
    RNNParams(const primitive_id& id,
              const input_info& x,
              const input_info& initial_hidden_state,
              const input_info& initial_cell_state,
              const input_info& W,
              const input_info& R,
              const input_info& B,
              const input_info& seq_lenghts,
              const primitive_id& out1_prim_id = "",
              const primitive_id& out2_prim_id = "",
              const float clip = 0,
              bool input_forget = false,
              const std::vector<activation_func>& activations = {activation_func::logistic,
                                                                activation_func::hyperbolic_tan,
                                                                activation_func::hyperbolic_tan},
              const std::vector<activation_additional_params>& activation_params = {},
              const lstm_weights_order& offset_order = lstm_weights_order::iofz,
              const ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
              const padding& output_padding = padding(),
              const int num_outputs = 1)
        : primitive_base<PType>(id, {x}, num_outputs, {optional_data_type()}, {output_padding}),
        id(id),
        x(x),
        initial_hidden_state(initial_hidden_state),
        initial_cell_state(initial_cell_state),
        W(W),
        R(R),
        B(B),
        seq_lenghts(seq_lenghts),
        out1_prim_id(out1_prim_id),
        out2_prim_id(out2_prim_id),
        clip(clip),
        input_forget(input_forget),
        activations(activations),
        activation_params(activation_params),
        offset_order(offset_order),
        direction(direction),
        output_padding(output_padding),
        num_outputs(num_outputs) {
        std::vector<std::string> pids{initial_hidden_state.pid, initial_cell_state.pid, W.pid, R.pid, B.pid, seq_lenghts.pid, out1_prim_id, out2_prim_id};
        assert(direction == ov::op::RecurrentSequenceDirection::FORWARD || direction == ov::op::RecurrentSequenceDirection::REVERSE);
        for (auto pid : pids) {
            if (!pid.empty()) {
                primitive_base<PType>::input.push_back(pid);
            }
        }
    }

    primitive_id id;
    input_info x;
    input_info initial_hidden_state;
    input_info initial_cell_state;
    input_info W;
    input_info R;
    input_info B;
    input_info seq_lenghts;
    primitive_id out1_prim_id;
    primitive_id out2_prim_id;
    /// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
    float clip;
    bool input_forget;
    /// @brief A list of 3 activation functions for the input, output, forget, cell, and hidden.
    std::vector<activation_func> activations;
    /// @brief Optional scaling values used by some activation functions. The values are consumed in the order of activation functions.
    std::vector<activation_additional_params> activation_params;
    /// @brief Weights, recurrent weights, and biases order. [iofz] : ONNX, [ifoz] : Caffe
    lstm_weights_order offset_order;
    /// @brief direction of LSTMSequence - only FORWARD or REVERSE, currently BIDIRECTIONAL not supported
    ov::op::RecurrentSequenceDirection direction;
    padding output_padding;
    int num_outputs;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        seed = hash_combine(seed, x.pid);
        seed = hash_combine(seed, initial_hidden_state.pid);
        seed = hash_combine(seed, initial_cell_state.pid);
        seed = hash_combine(seed, seq_lenghts.pid);
        seed = hash_combine(seed, W.pid);
        seed = hash_combine(seed, R.pid);
        seed = hash_combine(seed, B.pid);
        seed = hash_combine(seed, out1_prim_id);
        seed = hash_combine(seed, out2_prim_id);
        seed = hash_combine(seed, clip);
        seed = hash_range(seed, activations.begin(), activations.end());
        for (auto& act_param : activation_params) {
            seed = hash_combine(seed, act_param.a);
            seed = hash_combine(seed, act_param.b);
        }
        seed = hash_combine(seed, offset_order);
        seed = hash_combine(seed, direction);
        seed = hash_combine(seed, num_outputs);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!primitive::compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const PType>(rhs);
        bool act_params_eq = activation_params.size() == rhs_casted.activation_params.size();
        for (size_t i = 0; i < activation_params.size(); ++i) {
            act_params_eq &= activation_params[i].a == rhs_casted.activation_params[i].a &&
                             activation_params[i].b == rhs_casted.activation_params[i].b;
        }

        #define cmp_fields(name) name == rhs_casted.name
        return act_params_eq &&
               cmp_fields(id) &&
               cmp_fields(x) &&
               cmp_fields(initial_hidden_state) &&
               cmp_fields(initial_cell_state) &&
               cmp_fields(seq_lenghts) &&
               cmp_fields(W) &&
               cmp_fields(R) &&
               cmp_fields(B) &&
               cmp_fields(out1_prim_id) &&
               cmp_fields(out2_prim_id) &&
               cmp_fields(clip) &&
               cmp_fields(activations) &&
               cmp_fields(offset_order) &&
               cmp_fields(direction) &&
               cmp_fields(output_padding) &&
               cmp_fields(num_outputs);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << id;
        ob << x;
        ob << initial_hidden_state;
        ob << initial_cell_state;
        ob << W;
        ob << R;
        ob << B;
        ob << seq_lenghts;
        ob << out1_prim_id;
        ob << out2_prim_id;
        ob << clip;
        ob << activations;
        ob << activation_params;
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
        ob << make_data(&direction, sizeof(ov::op::RecurrentSequenceDirection));
        ob << output_padding;
        ob << num_outputs;
    }

    void load(BinaryInputBuffer& ib) override{
        ib >> id;
        ib >> x;
        ib >> initial_hidden_state;
        ib >> initial_cell_state;
        ib >> W;
        ib >> R;
        ib >> B;
        ib >> seq_lenghts;
        ib >> out1_prim_id;
        ib >> out2_prim_id;
        ib >> clip;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
        ib >> make_data(&direction, sizeof(ov::op::RecurrentSequenceDirection));
        ib >> output_padding;
        ib >> num_outputs;
    }
};

struct lstm_seq : public RNNParams<lstm_seq> {
    CLDNN_DECLARE_PRIMITIVE(lstm_seq)
    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    using RNNParams::RNNParams;
    lstm_seq() : RNNParams() {
        weights = W.pid;
        input = x.pid;
    }
    lstm_seq(const lstm_seq&) = default;
    primitive_id input;
    primitive_id weights;
};
} //namespace cldnn
