// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fused_conv_silu_pair_kernel_base.h"

#include <string>
#include <vector>

namespace kernel_selector {

// Jedyna na razie implementacja zfuzowanej pary konwolucji - wariant
// blokowy fsv16, zbudowany na tym samym szkielecie co
// ConvolutionKernel_b_fs_yx_fsv16.
class FusedConvSiluPairKernel_bfyx_f16 : public KernelBaseOpenCL {
public:
    FusedConvSiluPairKernel_bfyx_f16() : KernelBaseOpenCL("fused_conv_silu_pair_gpu_bfyx_f16") {}
    ~FusedConvSiluPairKernel_bfyx_f16() override = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;

    // Wyliczone raz i wspoldzielone przez SetDefault() i GetJitConstants(),
    // zeby host i kernel na pewno mialy ten sam podzial pracy.
    struct TuningData {
        size_t sub_group_size = 16;
        size_t feature_block_size = 16;
        size_t block_width = 8;      // OUTPUT_X_BLOCK_SIZE
        size_t num_sub_groups = 1;   // sub-group na work-group
        size_t ic_blocks = 1;
        size_t mid_ic_blocks = 1;
        size_t oc_blocks = 1;
    };

    TuningData GetTuningData(const fused_conv_silu_pair_params& params) const;
    CommonDispatchData SetDefault(const fused_conv_silu_pair_params& params, const TuningData& td) const;
    JitConstants GetJitConstants(const fused_conv_silu_pair_params& params, const TuningData& td) const;
};

}  // namespace kernel_selector
