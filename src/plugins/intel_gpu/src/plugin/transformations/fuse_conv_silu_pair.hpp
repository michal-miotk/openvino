// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// @brief Skleja wzorzec
///        Convolution -> Add(bias) -> Swish -> Convolution(1x1) -> Add(bias) -> Swish
///        w jeden wewnetrzny op FusedConvSiluPair.
///
/// Ograniczenia: druga konwolucja musi byc 1x1 ze stride 1, bez paddingu i
/// dilation (klasyczny bottleneck), obie konwolucje niegrupowane i 4D, wagi
/// i biasy stale, typ f16 lub f32.
///
/// Transformacja od razu przepakowuje wagi do layoutu os_is_yx_isv16_osv16,
/// bo prymityw GPU dostaje je jako zwykle wejscia i nie przechodzi przez
/// sciezke reorderu wag.
class FuseConvSiluPair : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseConvSiluPair");
    FuseConvSiluPair();
};

}  // namespace ov::intel_gpu
