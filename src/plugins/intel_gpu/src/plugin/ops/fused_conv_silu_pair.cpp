// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fused_conv_silu_pair.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/fused_conv_silu_pair.hpp"

namespace ov {
namespace op {
namespace internal {
using FusedConvSiluPair = ov::intel_gpu::op::FusedConvSiluPair;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateFusedConvSiluPairOp(ProgramBuilder& p,
                                      const std::shared_ptr<ov::intel_gpu::op::FusedConvSiluPair>& op) {
    validate_inputs_count(op, {5});
    auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    auto prim = cldnn::fused_conv_silu_pair(layer_name,
                                            inputs,
                                            op->get_strides(),
                                            op->get_dilations(),
                                            op->get_pads_begin(),
                                            op->get_pads_end(),
                                            op->get_kernel1(),
                                            op->get_in_channels(),
                                            op->get_mid_channels(),
                                            op->get_out_channels(),
                                            op->get_beta1(),
                                            op->get_beta2(),
                                            cldnn::element_type_to_data_type(op->get_output_element_type(0)));

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, FusedConvSiluPair);

}  // namespace ov::intel_gpu
