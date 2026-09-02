// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// @brief Zfuzowany blok conv1 -> +bias1 -> SiLU -> conv2(1x1) -> +bias2 -> SiLU.
///
/// Wejscia: (src, weights1, bias1, weights2, bias2).
/// Wagi i biasy sa przekazywane jako plaskie stale, juz przepakowane przez
/// transformacje FuseConvSiluPair do layoutu os_is_yx_isv16_osv16 - dzieki temu
/// prymityw GPU dostaje je jako zwykle bufory i nie musi przechodzic przez
/// sciezke reorderu wag, ktora obsluguje tylko jeden tensor wag na prymityw.
class FusedConvSiluPair : public ov::op::Op {
public:
    OPENVINO_OP("FusedConvSiluPair", "gpu_opset");

    FusedConvSiluPair() = default;

    FusedConvSiluPair(const ov::Output<Node>& src,
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
                      const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const ov::Strides& get_strides() const { return m_strides; }
    const ov::Strides& get_dilations() const { return m_dilations; }
    const ov::CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
    const ov::CoordinateDiff& get_pads_end() const { return m_pads_end; }
    const ov::Shape& get_kernel1() const { return m_kernel1; }
    size_t get_in_channels() const { return m_in_channels; }
    size_t get_mid_channels() const { return m_mid_channels; }
    size_t get_out_channels() const { return m_out_channels; }
    float get_beta1() const { return m_beta1; }
    float get_beta2() const { return m_beta2; }
    ov::element::Type get_output_type() const { return m_output_type; }

private:
    ov::Strides m_strides;
    ov::Strides m_dilations;
    ov::CoordinateDiff m_pads_begin;
    ov::CoordinateDiff m_pads_end;
    ov::Shape m_kernel1;
    size_t m_in_channels = 0;
    size_t m_mid_channels = 0;
    size_t m_out_channels = 0;
    float m_beta1 = 1.0f;
    float m_beta2 = 1.0f;
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
