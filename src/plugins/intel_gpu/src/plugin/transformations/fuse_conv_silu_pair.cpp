// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_conv_silu_pair.hpp"

#include "intel_gpu/op/fused_conv_silu_pair.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace ov::intel_gpu {
namespace {

constexpr size_t fsv = 16;

size_t align_up(size_t v, size_t a) {
    return ((v + a - 1) / a) * a;
}

// Przepakowanie wag [OC, IC, KY, KX] do layoutu os_is_yx_isv16_osv16, czyli
// dokladnie tego, ktorego oczekuje kernel fused_conv_silu_pair_gpu_bfyx_f16.
// Ogony po OC i IC (gdy nie sa wielokrotnoscia 16) sa zerowane, dzieki czemu
// kernel moze bezwarunkowo czytac pelne bloki 16x16.
std::vector<float> pack_weights(const std::vector<float>& src, size_t oc, size_t ic, size_t ky, size_t kx) {
    const size_t ic_blocks = align_up(ic, fsv) / fsv;
    const size_t oc_blocks = align_up(oc, fsv) / fsv;

    const size_t x_pitch = fsv * fsv;
    const size_t y_pitch = x_pitch * kx;
    const size_t is_pitch = y_pitch * ky;
    const size_t os_pitch = is_pitch * ic_blocks;

    std::vector<float> dst(os_pitch * oc_blocks, 0.0f);
    for (size_t o = 0; o < oc; o++) {
        for (size_t i = 0; i < ic; i++) {
            for (size_t y = 0; y < ky; y++) {
                for (size_t x = 0; x < kx; x++) {
                    const size_t off = (o / fsv) * os_pitch + (i / fsv) * is_pitch + y * y_pitch + x * x_pitch +
                                       (i % fsv) * fsv + (o % fsv);
                    dst[off] = src[((o * ic + i) * ky + y) * kx + x];
                }
            }
        }
    }
    return dst;
}

// Bias musi byc odczytywalny pelnym blokiem 16 elementow na slice fsv16.
std::vector<float> pack_bias(const std::vector<float>& src, size_t oc) {
    std::vector<float> dst(align_up(oc, fsv), 0.0f);
    std::copy_n(src.begin(), std::min(src.size(), dst.size()), dst.begin());
    return dst;
}

// Bias konwolucji zapisany jako Add ze stala nadajaca sie na broadcast po
// kanalach. Broadcast numpy wyrownuje ksztalty DO PRAWEJ, wiec dla wyjscia
// [N, C, H, W] kanalowym biasem sa np. [1, C, 1, 1] i [C, 1, 1], ale NIE [C]
// (ten trafilby na os W).
bool is_channel_bias(const std::shared_ptr<ov::op::v0::Constant>& c, size_t channels, size_t out_rank) {
    if (!c)
        return false;

    const auto& shape = c->get_shape();
    if (shape.size() > out_rank)
        return false;
    if (ov::shape_size(shape) != channels)
        return false;

    const size_t offset = out_rank - shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
        const size_t axis = offset + i;
        const size_t expected = (axis == 1) ? channels : 1;
        if (shape[i] != expected)
            return false;
    }
    return true;
}

bool all_equal(const std::vector<size_t>& v, size_t value) {
    return std::all_of(v.begin(), v.end(), [value](size_t x) {
        return x == value;
    });
}

float get_swish_beta(const std::shared_ptr<ov::Node>& swish) {
    if (swish->get_input_size() < 2)
        return 1.0f;
    const auto beta = ov::as_type_ptr<ov::op::v0::Constant>(swish->get_input_node_shared_ptr(1));
    if (!beta || ov::shape_size(beta->get_shape()) != 1)
        return 0.0f;  // niestala beta - wzorzec zostanie odrzucony
    return beta->cast_vector<float>()[0];
}

}  // namespace

FuseConvSiluPair::FuseConvSiluPair() {
    using namespace ov::pass::pattern;

    auto src = any_input();
    auto weights1_m = wrap_type<ov::op::v0::Constant>();
    auto conv1_m = wrap_type<ov::op::v1::Convolution>({src, weights1_m});
    auto bias1_m = wrap_type<ov::op::v0::Constant>();
    auto add1_m = wrap_type<ov::op::v1::Add>({conv1_m, bias1_m});
    // Swish wystepuje w dwoch wariantach: z domyslna beta i z jawnym drugim wejsciem.
    auto silu1_a = wrap_type<ov::op::v4::Swish>({add1_m});
    auto silu1_b = wrap_type<ov::op::v4::Swish>({add1_m, any_input()});
    auto silu1_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{silu1_a, silu1_b});

    auto weights2_m = wrap_type<ov::op::v0::Constant>();
    auto conv2_m = wrap_type<ov::op::v1::Convolution>({silu1_m, weights2_m});
    auto bias2_m = wrap_type<ov::op::v0::Constant>();
    auto add2_m = wrap_type<ov::op::v1::Add>({conv2_m, bias2_m});
    auto silu2_a = wrap_type<ov::op::v4::Swish>({add2_m});
    auto silu2_b = wrap_type<ov::op::v4::Swish>({add2_m, any_input()});
    auto silu2_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{silu2_a, silu2_b});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();

        auto root = m.get_match_root();
        if (transformation_callback(root))
            return false;

        auto conv1 = ov::as_type_ptr<ov::op::v1::Convolution>(pm.at(conv1_m).get_node_shared_ptr());
        auto conv2 = ov::as_type_ptr<ov::op::v1::Convolution>(pm.at(conv2_m).get_node_shared_ptr());
        auto weights1 = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(weights1_m).get_node_shared_ptr());
        auto weights2 = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(weights2_m).get_node_shared_ptr());
        auto bias1 = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(bias1_m).get_node_shared_ptr());
        auto bias2 = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(bias2_m).get_node_shared_ptr());
        if (!conv1 || !conv2 || !weights1 || !weights2 || !bias1 || !bias2)
            return false;

        // Zfuzowany kernel produkuje tylko koncowe wyjscie, wiec kazdy wezel
        // posredni musi miec dokladnie jednego konsumenta.
        for (const auto& n : {pm.at(conv1_m), pm.at(add1_m), pm.at(conv2_m), pm.at(add2_m)}) {
            if (n.get_target_inputs().size() != 1)
                return false;
        }
        auto silu1 = pm.count(silu1_a) ? pm.at(silu1_a).get_node_shared_ptr() : pm.at(silu1_b).get_node_shared_ptr();
        if (silu1->output(0).get_target_inputs().size() != 1)
            return false;
        auto silu2 = pm.count(silu2_a) ? pm.at(silu2_a).get_node_shared_ptr() : pm.at(silu2_b).get_node_shared_ptr();

        // --- Typy ---
        const auto& et = conv1->get_input_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32)
            return false;
        for (const auto& c : {weights1, weights2, bias1, bias2}) {
            if (c->get_element_type() != et)
                return false;
        }

        // --- Ksztalty ---
        const auto& src_pshape = conv1->get_input_partial_shape(0);
        const auto& out_pshape = silu2->get_output_partial_shape(0);
        if (src_pshape.rank().is_dynamic() || src_pshape.rank().get_length() != 4)
            return false;
        if (src_pshape[1].is_dynamic() || out_pshape.is_dynamic())
            return false;

        const auto w1_shape = weights1->get_shape();  // [OC1, IC, KY, KX]
        const auto w2_shape = weights2->get_shape();  // [OC2, OC1, 1, 1]
        if (w1_shape.size() != 4 || w2_shape.size() != 4)
            return false;

        const size_t in_channels = w1_shape[1];
        const size_t mid_channels = w1_shape[0];
        const size_t out_channels = w2_shape[0];
        if (static_cast<size_t>(src_pshape[1].get_length()) != in_channels)
            return false;
        if (w2_shape[1] != mid_channels)
            return false;

        // --- Geometria ---
        // Conv2 musi byc czystym 1x1 ze stride 1 - tylko wtedy kafelek posredni
        // pokrywa sie 1:1 z kafelkiem wyjsciowym i fuzja nie wymaga halo.
        if (w2_shape[2] != 1 || w2_shape[3] != 1)
            return false;
        if (!all_equal(conv2->get_strides(), 1) || !all_equal(conv2->get_dilations(), 1))
            return false;
        const auto& c2_pb = conv2->get_pads_begin();
        const auto& c2_pe = conv2->get_pads_end();
        if (std::any_of(c2_pb.begin(), c2_pb.end(), [](std::ptrdiff_t v) { return v != 0; }) ||
            std::any_of(c2_pe.begin(), c2_pe.end(), [](std::ptrdiff_t v) { return v != 0; }))
            return false;

        if (conv1->get_strides().size() != 2)
            return false;
        const auto& c1_pb = conv1->get_pads_begin();
        const auto& c1_pe = conv1->get_pads_end();
        if (c1_pb.size() != 2 || c1_pe.size() != 2)
            return false;
        // Kernel obsluguje tylko symetryczny, nieujemny padding.
        if (c1_pb[0] < 0 || c1_pb[1] < 0 || c1_pb != c1_pe)
            return false;

        // --- Biasy ---
        if (!is_channel_bias(bias1, mid_channels, 4) || !is_channel_bias(bias2, out_channels, 4))
            return false;

        // --- Aktywacje ---
        const float beta1 = get_swish_beta(silu1);
        const float beta2 = get_swish_beta(silu2);
        if (beta1 == 0.0f || beta2 == 0.0f)
            return false;

        // --- Przepakowanie stalych ---
        const auto packed_w1 = pack_weights(weights1->cast_vector<float>(),
                                            mid_channels,
                                            in_channels,
                                            w1_shape[2],
                                            w1_shape[3]);
        const auto packed_w2 = pack_weights(weights2->cast_vector<float>(), out_channels, mid_channels, 1, 1);
        const auto packed_b1 = pack_bias(bias1->cast_vector<float>(), mid_channels);
        const auto packed_b2 = pack_bias(bias2->cast_vector<float>(), out_channels);

        auto w1_const = ov::op::v0::Constant::create(et, ov::Shape{packed_w1.size()}, packed_w1);
        auto b1_const = ov::op::v0::Constant::create(et, ov::Shape{packed_b1.size()}, packed_b1);
        auto w2_const = ov::op::v0::Constant::create(et, ov::Shape{packed_w2.size()}, packed_w2);
        auto b2_const = ov::op::v0::Constant::create(et, ov::Shape{packed_b2.size()}, packed_b2);

        auto fused = std::make_shared<op::FusedConvSiluPair>(conv1->input_value(0),
                                                             w1_const,
                                                             b1_const,
                                                             w2_const,
                                                             b2_const,
                                                             conv1->get_strides(),
                                                             conv1->get_dilations(),
                                                             conv1->get_pads_begin(),
                                                             conv1->get_pads_end(),
                                                             ov::Shape{w1_shape[2], w1_shape[3]},
                                                             in_channels,
                                                             mid_channels,
                                                             out_channels,
                                                             beta1,
                                                             beta2,
                                                             et);
        fused->set_friendly_name(silu2->get_friendly_name());

        if (fused->get_output_partial_shape(0) != out_pshape)
            return false;

        ov::copy_runtime_info({conv1, conv2, silu1, silu2, root}, fused);
        ov::replace_node(silu2, fused);
        register_new_node(fused);

        return true;
    };

    auto m = std::make_shared<Matcher>(silu2_m, "FuseConvSiluPair");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
