// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_conv_silu_pair_kernel_selector.h"

#include "fused_conv_silu_pair_kernel_bfyx_f16.h"

namespace kernel_selector {

fused_conv_silu_pair_kernel_selector::fused_conv_silu_pair_kernel_selector() {
    Attach<FusedConvSiluPairKernel_bfyx_f16>();
}

KernelsData fused_conv_silu_pair_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::FUSED_CONV_SILU_PAIR);
}

}  // namespace kernel_selector
