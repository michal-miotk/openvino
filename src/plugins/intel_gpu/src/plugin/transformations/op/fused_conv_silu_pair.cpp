// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fused_conv_silu_pair.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::intel_gpu::op {

FusedConvSiluPair::FusedConvSiluPair(const ov::Output<Node>& src,
                                     const ov::Output<Node>& weights1,
                                     const ov::Output<Node>& bias1,
                                     const ov::Output<Node>& weights2,
                                     const ov::Output<Node>& bias2,
                                     const ov::Strides& strides,
                                     const ov::Strides& dilations,
                                     const ov::CoordinateDiff& pads_begin,
                                     const ov::CoordinateDiff& pads_end,
                                     const ov::Shape& kernel1,
                                     size_t in_channels,
                                     size_t mid_channels,
                                     size_t out_channels,
                                     float beta1,
                                     float beta2,
                                     const ov::element::Type output_type)
    : Op({src, weights1, bias1, weights2, bias2}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_kernel1(kernel1),
      m_in_channels(in_channels),
      m_mid_channels(mid_channels),
      m_out_channels(out_channels),
      m_beta1(beta1),
      m_beta2(beta2),
      m_output_type(output_type) {
    validate_and_infer_types();
}

bool FusedConvSiluPair::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel1", m_kernel1);
    visitor.on_attribute("in_channels", m_in_channels);
    visitor.on_attribute("mid_channels", m_mid_channels);
    visitor.on_attribute("out_channels", m_out_channels);
    visitor.on_attribute("beta1", m_beta1);
    visitor.on_attribute("beta2", m_beta2);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void FusedConvSiluPair::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 5, "FusedConvSiluPair expects exactly 5 inputs");
    NODE_VALIDATION_CHECK(this,
                          m_strides.size() == 2 && m_dilations.size() == 2 && m_pads_begin.size() == 2 &&
                              m_pads_end.size() == 2 && m_kernel1.size() == 2,
                          "FusedConvSiluPair supports only 2D spatial configuration");

    const auto& src_pshape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          src_pshape.rank().is_dynamic() || src_pshape.rank().get_length() == 4,
                          "FusedConvSiluPair expects a 4D activation input");

    const auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;

    if (src_pshape.rank().is_dynamic()) {
        set_output_type(0, output_type, ov::PartialShape::dynamic(4));
        return;
    }

    // Conv2 jest 1x1 ze stride 1 i bez paddingu, wiec wymiary przestrzenne
    // wynikaja wylacznie z geometrii conv1.
    auto spatial = [&](const ov::Dimension& in, size_t idx) -> ov::Dimension {
        if (in.is_dynamic())
            return ov::Dimension::dynamic();
        const auto effective_kernel = static_cast<int64_t>((m_kernel1[idx] - 1) * m_dilations[idx] + 1);
        const auto padded = in.get_length() + m_pads_begin[idx] + m_pads_end[idx] - effective_kernel;
        if (padded < 0)
            return ov::Dimension::dynamic();
        return ov::Dimension(padded / static_cast<int64_t>(m_strides[idx]) + 1);
    };

    ov::PartialShape out_pshape = src_pshape;
    out_pshape[1] = ov::Dimension(static_cast<int64_t>(m_out_channels));
    out_pshape[2] = spatial(src_pshape[2], 0);
    out_pshape[3] = spatial(src_pshape[3], 1);

    set_output_type(0, output_type, out_pshape);
}

std::shared_ptr<Node> FusedConvSiluPair::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FusedConvSiluPair>(new_args.at(0),
                                               new_args.at(1),
                                               new_args.at(2),
                                               new_args.at(3),
                                               new_args.at(4),
                                               m_strides,
                                               m_dilations,
                                               m_pads_begin,
                                               m_pads_end,
                                               m_kernel1,
                                               m_in_channels,
                                               m_mid_channels,
                                               m_out_channels,
                                               m_beta1,
                                               m_beta2,
                                               m_output_type);
}

}  // namespace ov::intel_gpu::op
