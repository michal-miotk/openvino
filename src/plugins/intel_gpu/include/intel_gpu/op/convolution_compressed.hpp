// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/convolution_base.hpp"

namespace ov::intel_gpu::op {

// Common node for v1::Convolution and v1::GroupConvolution with few extensions
//  - Relaxed type requirements
//  - Bias support
//  - Asymmetric quantization support
class ConvolutionCompressed : public ov::op::util::ConvolutionFwdPropBase {
public:
    OPENVINO_OP("ConvolutionCompressed", "gpu_opset", ov::op::util::ConvolutionFwdPropBase);

    ConvolutionCompressed() = default;

    ConvolutionCompressed(const ov::Output<Node>& data_batch,
                const ov::Output<Node>& filters,
                const ov::Output<Node>& weights_scale,
                const ov::Output<Node>& weights_zero_point,
                const ov::Strides& strides,
                const ov::CoordinateDiff& pads_begin,
                const ov::CoordinateDiff& pads_end,
                const ov::Strides& dilations,
                const int64_t& groups,
                const ov::op::PadType& auto_pad,
                const ov::element::Type& output_type);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_groups() const;
    int64_t get_groups() const;

    bool is_asymmetric() const;

    struct Args {
        static constexpr const size_t INPUT = 0;
        static constexpr const size_t WEIGHTS = 1;
        static constexpr const size_t BIAS = 2;
        static constexpr const size_t SCALE = 3;
        static constexpr const size_t SCALE_ZP = 4;
        static constexpr const size_t AZP = 5;
        static constexpr const size_t WZP = 6;
        static constexpr const size_t COMPENSATION = 7;
    };

protected:
    const ov::Output<Node>& weights_scale;
    const ov::Output<Node>& weights_zero_point;
    int64_t m_groups = -1; // negative value means no groups
    bool m_asymmetric = false;
    ov::element::Type m_output_type = ov::element::undefined;
};

std::vector<ov::PartialShape> shape_infer(const ConvolutionCompressed* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          CoordinateDiff& pads_begin,
                                          CoordinateDiff& pads_end);

}   // namespace ov::intel_gpu::op
