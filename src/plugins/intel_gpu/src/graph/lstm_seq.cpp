// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_seq_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm_seq)

layout lstm_seq_inst::calc_output_layout(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
    const auto& desc = impl_param.typed_desc<lstm_seq>();
    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& input_pshape = input_layout.get_partial_shape();
    const auto& input_layout_hidden = impl_param.get_input_layout(1);
    const auto& input_pshape_hidden = input_layout_hidden.get_partial_shape();
    const auto& lstm_batch_size = input_pshape[0];
    const auto& lstm_seq_length = input_pshape[1];
    const auto& lstm_hidden_size = input_pshape_hidden[2];

    auto first_out_fmt = cldnn::format::bfyx;
    if (node.get_preferred_output_fmt() != format::any) {
        first_out_fmt = node.get_preferred_output_fmt();
    }

    return cldnn::layout{ov::PartialShape{lstm_batch_size, desc->num_directions(), lstm_seq_length, lstm_hidden_size}, input_layout.data_type, first_out_fmt};
}

template<typename ShapeType>
std::vector<layout> lstm_seq_inst::calc_output_layouts(lstm_seq_node const& node, kernel_impl_params const& impl_param) {
<<<<<<< HEAD
    const auto& desc = impl_param.typed_desc<lstm_seq>();
    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& input_pshape = input_layout.get_partial_shape();
    const auto& input_layout_hidden = impl_param.get_input_layout(1);
    const auto& input_pshape_hidden = input_layout_hidden.get_partial_shape();
    const auto& lstm_batch_size = input_pshape[0];
    const auto& lstm_seq_length = input_pshape[1];
    const auto& lstm_hidden_size = input_pshape_hidden[2];
=======
    // input partial shape [batch, input_size (= hidden_size * 4)]
    auto input_layout_x = impl_param.get_input_layout(0);
    auto input_pshape_x = input_layout_x.get_partial_shape();
    auto input_layout_hidden = impl_param.get_input_layout(1);
    auto input_pshape_hidden = input_layout_hidden.get_partial_shape();
    if (impl_param.desc->output_data_types.size() > 0) {
        OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false, "Output data type forcing is not supported for lstm_seq_node!");
    }
    OPENVINO_ASSERT(input_pshape_x.rank().get_length() == 4, "input_layout rank should be 4 on dynamic shape.");
>>>>>>> d461e6623f (19jul)

    auto first_out_fmt = cldnn::format::bfyx;
    auto second_out_fmt = input_layout.format;
    auto third_out_fmt = input_layout.format;
    if (node.get_preferred_output_fmt() != format::any) {
        first_out_fmt = node.get_preferred_output_fmt();
        second_out_fmt = node.get_preferred_output_fmt(1);
        third_out_fmt = node.get_preferred_output_fmt(2);
    }
    auto num_directions = desc->num_directions();

    return {cldnn::layout{ShapeType{lstm_batch_size, num_directions, lstm_seq_length, lstm_hidden_size}, input_layout.data_type, first_out_fmt},
            cldnn::layout{ShapeType{lstm_batch_size, num_directions, lstm_hidden_size}, input_layout.data_type, second_out_fmt},
            cldnn::layout{ShapeType{lstm_batch_size, num_directions, lstm_hidden_size}, input_layout.data_type, third_out_fmt}};
}

template std::vector<layout> lstm_seq_inst::calc_output_layouts<ov::PartialShape>(lstm_seq_node const& node, const kernel_impl_params& impl_param);

std::string lstm_seq_inst::to_string(lstm_seq_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite lstm_seq_info;
    node_info->add("lstm seq info", lstm_seq_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_seq_inst::typed_primitive_inst(network& network, lstm_seq_node const& node) : parent(network, node) {}
}  // namespace cldnn
