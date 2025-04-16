// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gru_seq_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gru_seq)

layout gru_seq_inst::calc_output_layout(gru_seq_node const& node, kernel_impl_params const& impl_param) {
    return gru_seq_inst::calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> gru_seq_inst::calc_output_layouts(gru_seq_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<gru_seq>();

    auto input_layout_x = impl_param.get_input_layout(0);
    auto input_pshape_x = input_layout_x.get_partial_shape();
    auto input_layout_hidden = impl_param.get_input_layout(1);
    auto input_pshape_hidden = input_layout_hidden.get_partial_shape();
    int gru_batch_size, gru_seq_length, gru_hidden_size;
    if (input_pshape_x[0].is_static()) {
        gru_batch_size = input_pshape_x[0].get_length();
    } else {
        gru_batch_size = -1;
    }

    if (input_pshape_x[1].is_static()) {
        gru_seq_length = input_pshape_x[1].get_length();
    } else {
        gru_seq_length = -1;
    }

    if (input_pshape_hidden[2].is_static()) {
        gru_hidden_size = input_pshape_hidden[2].get_length();
    } else {
        gru_hidden_size = -1;
    }
    auto first_out_fmt = cldnn::format::bfyx;
    auto second_out_fmt = input_layout_x.format;
    auto third_out_fmt = input_layout_x.format;
    if (node.permute_inserted) {
        first_out_fmt = node.get_preferred_output_fmt();
        second_out_fmt = node.get_preferred_output_fmt(1);
        third_out_fmt = node.get_preferred_output_fmt(2);
        return {cldnn::layout{ShapeType{gru_seq_length, gru_batch_size, gru_hidden_size, 1}, input_layout_x.data_type, first_out_fmt}, \
            cldnn::layout{ShapeType{gru_batch_size, 1, gru_hidden_size}, input_layout_x.data_type, second_out_fmt}};
    } else {
        return {cldnn::layout{ShapeType{gru_batch_size, 1, gru_seq_length, gru_hidden_size}, input_layout_x.data_type, first_out_fmt}, \
                cldnn::layout{ShapeType{gru_batch_size, 1, gru_hidden_size}, input_layout_x.data_type, second_out_fmt}};
    }
}

template std::vector<layout> gru_seq_inst::calc_output_layouts<ov::PartialShape>(gru_seq_node const& node, const kernel_impl_params& impl_param);

std::string gru_seq_inst::to_string(gru_seq_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite gru_seq_info;
    node_info->add("gru seq info", gru_seq_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gru_seq_inst::typed_primitive_inst(network& network, gru_seq_node const& node) : parent(network, node) {}
}  // namespace cldnn
