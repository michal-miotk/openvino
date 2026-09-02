// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

#include <utility>
#include <vector>

namespace cldnn {

/// @brief Zfuzowana para konwolucji: conv1 -> +bias1 -> SiLU -> conv2(1x1) -> +bias2 -> SiLU.
/// @details Tensor posredni nigdy nie trafia do pamieci globalnej - jest
/// stagowany w SLM wewnatrz jednego kernela. Wagi i biasy przychodza jako
/// zwykle wejscia, juz przepakowane przez transformacje do layoutu
/// os_is_yx_isv16_osv16 (plaskie bufory 1-D), dzieki czemu prymityw nie
/// przechodzi przez maszynerie reorderu wag, ktora obsluguje tylko jeden
/// tensor wag na prymityw.
///
/// Kolejnosc wejsc: {input, weights1, bias1, weights2, bias2}.
struct fused_conv_silu_pair : public primitive_base<fused_conv_silu_pair> {
    CLDNN_DECLARE_PRIMITIVE(fused_conv_silu_pair)

    fused_conv_silu_pair() : primitive_base("", {}) {}

    fused_conv_silu_pair(const primitive_id& id,
                         const std::vector<input_info>& inputs,
                         ov::Strides strides,
                         ov::Strides dilations,
                         ov::CoordinateDiff pads_begin,
                         ov::CoordinateDiff pads_end,
                         ov::Shape kernel1,
                         size_t in_channels,
                         size_t mid_channels,
                         size_t out_channels,
                         float beta1,
                         float beta2,
                         data_types output_dt)
        : primitive_base(id, inputs, 1, {optional_data_type{output_dt}}),
          strides(std::move(strides)),
          dilations(std::move(dilations)),
          pads_begin(std::move(pads_begin)),
          pads_end(std::move(pads_end)),
          kernel1(std::move(kernel1)),
          in_channels(in_channels),
          mid_channels(mid_channels),
          out_channels(out_channels),
          beta1(beta1),
          beta2(beta2) {}

    /// @brief Stride pierwszej konwolucji (conv2 jest z definicji stride 1).
    ov::Strides strides;
    /// @brief Dilation pierwszej konwolucji.
    ov::Strides dilations;
    /// @brief Padding pierwszej konwolucji.
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    /// @brief Rozmiar jadra pierwszej konwolucji jako {KY, KX}.
    ov::Shape kernel1;

    /// @brief Logiczne (niedopelnione) liczby kanalow.
    size_t in_channels = 0;
    size_t mid_channels = 0;
    size_t out_channels = 0;

    /// @brief Beta obu aktywacji Swish/SiLU.
    float beta1 = 1.0f;
    float beta2 = 1.0f;

    size_t hash() const override {
        size_t seed = primitive::hash();
        for (auto v : strides)
            seed = hash_combine(seed, v);
        for (auto v : dilations)
            seed = hash_combine(seed, v);
        for (auto v : pads_begin)
            seed = hash_combine(seed, v);
        for (auto v : pads_end)
            seed = hash_combine(seed, v);
        for (auto v : kernel1)
            seed = hash_combine(seed, v);
        seed = hash_combine(seed, in_channels);
        seed = hash_combine(seed, mid_channels);
        seed = hash_combine(seed, out_channels);
        seed = hash_combine(seed, beta1);
        seed = hash_combine(seed, beta2);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const fused_conv_silu_pair>(rhs);
        return strides == rhs_casted.strides && dilations == rhs_casted.dilations &&
               pads_begin == rhs_casted.pads_begin && pads_end == rhs_casted.pads_end &&
               kernel1 == rhs_casted.kernel1 && in_channels == rhs_casted.in_channels &&
               mid_channels == rhs_casted.mid_channels && out_channels == rhs_casted.out_channels &&
               beta1 == rhs_casted.beta1 && beta2 == rhs_casted.beta2;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<fused_conv_silu_pair>::save(ob);
        ob << strides;
        ob << dilations;
        ob << pads_begin;
        ob << pads_end;
        ob << kernel1;
        ob << in_channels;
        ob << mid_channels;
        ob << out_channels;
        ob << beta1;
        ob << beta2;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<fused_conv_silu_pair>::load(ib);
        ib >> strides;
        ib >> dilations;
        ib >> pads_begin;
        ib >> pads_end;
        ib >> kernel1;
        ib >> in_channels;
        ib >> mid_channels;
        ib >> out_channels;
        ib >> beta1;
        ib >> beta2;
    }
};

}  // namespace cldnn
