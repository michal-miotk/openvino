// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

class fused_conv_silu_pair_kernel_selector : public kernel_selector_base {
public:
    static fused_conv_silu_pair_kernel_selector& Instance() {
        static fused_conv_silu_pair_kernel_selector instance_;
        return instance_;
    }

    fused_conv_silu_pair_kernel_selector();
    ~fused_conv_silu_pair_kernel_selector() override = default;

    KernelsData GetBestKernels(const Params& params) const override;
};

}  // namespace kernel_selector
