// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_conv_silu_pair_inst.h"

#include "json_object.h"
#include "primitive_type_base.h"

#include <sstream>
#include <string>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(fused_conv_silu_pair)

namespace {

// Rozmiar wyjscia zalezy wylacznie od geometrii conv1 - conv2 jest 1x1 ze
// stride 1 i bez paddingu, wiec nie zmienia wymiarow przestrzennych.
ov::Dimension spatial_out_dim(const ov::Dimension& in,
                              size_t kernel,
                              size_t stride,
                              size_t dilation,
                              std::ptrdiff_t pad_begin,
                              std::ptrdiff_t pad_end) {
    if (in.is_dynamic())
        return ov::Dimension::dynamic();

    const auto effective_kernel = static_cast<std::ptrdiff_t>((kernel - 1) * dilation + 1);
    const auto padded = in.get_length() + pad_begin + pad_end - effective_kernel;
    if (padded < 0)
        return ov::Dimension::dynamic();

    return ov::Dimension(padded / static_cast<std::ptrdiff_t>(stride) + 1);
}

}  // namespace

template <typename ShapeType>
std::vector<layout> fused_conv_silu_pair_inst::calc_output_layouts(fused_conv_silu_pair_node const& /*node*/,
                                                                   const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<fused_conv_silu_pair>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);

    auto in_shape = input_layout.get<ShapeType>();
    OPENVINO_ASSERT(in_shape.rank().is_dynamic() || in_shape.size() == 4,
                    "[GPU] fused_conv_silu_pair supports only 4D inputs");

    if (in_shape.rank().is_dynamic())
        return {layout(ShapeType::dynamic(4), output_type, input_layout.format)};

    ShapeType out_shape = in_shape;
    out_shape[1] = ov::Dimension(static_cast<int64_t>(desc->out_channels));
    out_shape[2] = spatial_out_dim(in_shape[2],
                                   desc->kernel1[0],
                                   desc->strides[0],
                                   desc->dilations[0],
                                   desc->pads_begin[0],
                                   desc->pads_end[0]);
    out_shape[3] = spatial_out_dim(in_shape[3],
                                   desc->kernel1[1],
                                   desc->strides[1],
                                   desc->dilations[1],
                                   desc->pads_begin[1],
                                   desc->pads_end[1]);

    return {layout(out_shape, output_type, input_layout.format)};
}

template std::vector<layout> fused_conv_silu_pair_inst::calc_output_layouts<ov::PartialShape>(
    fused_conv_silu_pair_node const& node,
    const kernel_impl_params& impl_param);

layout fused_conv_silu_pair_inst::calc_output_layout(fused_conv_silu_pair_node const& node,
                                                     kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

std::string fused_conv_silu_pair_inst::to_string(fused_conv_silu_pair_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    json_composite info;
    info.add("input_id", node.input().id());
    info.add("weights1_id", node.weights1().id());
    info.add("bias1_id", node.bias1().id());
    info.add("weights2_id", node.weights2().id());
    info.add("bias2_id", node.bias2().id());
    info.add("kernel1", desc->kernel1);
    info.add("strides", desc->strides);
    info.add("dilations", desc->dilations);
    info.add("pads_begin", desc->pads_begin);
    info.add("pads_end", desc->pads_end);
    info.add("in_channels", desc->in_channels);
    info.add("mid_channels", desc->mid_channels);
    info.add("out_channels", desc->out_channels);
    info.add("beta1", desc->beta1);
    info.add("beta2", desc->beta2);

    node_info->add("fused_conv_silu_pair_info", info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

fused_conv_silu_pair_inst::typed_primitive_inst(network& network, fused_conv_silu_pair_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
