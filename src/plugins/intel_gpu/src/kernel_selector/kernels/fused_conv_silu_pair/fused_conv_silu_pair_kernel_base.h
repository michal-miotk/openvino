// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fused_conv_silu_pair_params
//
// Parametry zfuzowanego bloku conv -> +bias -> SiLU -> conv(1x1) -> +bias -> SiLU.
//
// Wagi i biasy NIE sa tu tensorami `weights`/`bias` kernel_selectora, tylko
// zwyklymi wejsciami:
//   inputs[0] - aktywacje,
//   inputs[1] - wagi conv1 (przepakowane do os_is_yx_isv16_osv16, plaski 1-D),
//   inputs[2] - bias conv1 (dopelniony zerami do wielokrotnosci 16),
//   inputs[3] - wagi conv2 (przepakowane do os_is_yx_isv16_osv16, plaski 1-D),
//   inputs[4] - bias conv2 (dopelniony zerami do wielokrotnosci 16).
//
// Pakowanie wag robi transformacja (FuseConvSiluPair), dzieki czemu ten
// prymityw nie musi przechodzic przez maszynerie reorderu wag cldnn, ktora
// zaklada dokladnie jeden tensor wag na prymityw.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fused_conv_silu_pair_params : public base_params {
    fused_conv_silu_pair_params() : base_params(KernelType::FUSED_CONV_SILU_PAIR) {}

    // Geometria pierwszej konwolucji. Druga jest z zalozenia 1x1 / stride 1 /
    // bez paddingu i dilation, wiec nie ma wlasnych pol.
    uSize filterSize1 = {1, 1, 1};
    uSize stride = {1, 1, 1};
    uSize dilation = {1, 1, 1};
    uSize padding = {0, 0, 0};

    // Logiczne (niedopelnione) liczby kanalow.
    size_t in_features = 0;
    size_t mid_features = 0;
    size_t out_features = 0;

    // Beta obu aktywacji Swish/SiLU (dla klasycznego SiLU rowne 1.0).
    float swish_beta1 = 1.0f;
    float swish_beta2 = 1.0f;
};

}  // namespace kernel_selector
