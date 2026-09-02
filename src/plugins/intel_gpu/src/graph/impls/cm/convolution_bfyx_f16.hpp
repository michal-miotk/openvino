// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "convolution_inst.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

// CM (C-for-Metal) implementacja konwolucji dla layoutu b_fs_yx_fsv16 - patrz
// convolution_bfyx_f16.cm po pelny opis kernela. Waski zakres, zgodny z
// ograniczeniami kernela: tylko f16 lub f32 (ten sam typ dla input/output/
// wag/biasu naraz - kernel nie robi konwersji miedzy typami), tylko groups==1, bez kwantyzacji
// (zero-points/compensation), bez deformable conv, tylko statyczny ksztalt.
// Fizyczny padding buforow na osiach przestrzennych (Y/X) jest obslugiwany
// (patrz CONV_*_PAD_* w .cm); padding na batchu/kanalach - nie. Fused
// post-ops sa obslugiwane TYLKO w jednej, waskiej postaci: pojedyncza
// aktywacja SiLU/Swish(beta=1) bez zewnetrznych zaleznosci (fuzowana wprost
// w epilogu kernela) - wszystko inne spada do onednn/ocl (patrz kolejnosc
// kandydatow w registry/convolution_impls.cpp).
struct ConvolutionBfyxF16ImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::conv::bfyx_f16")
    explicit ConvolutionBfyxF16ImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<convolution>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);
        in_fmts[0] = format::b_fs_yx_fsv16;
        out_fmts[0] = format::b_fs_yx_fsv16;
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<convolution>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        if (!check_cm_jit_support(engine, config) || !config.get_use_cm()) {
            return false;
        }

        if (node.is_dynamic()) {
            return false;
        }

        if (node.has_fused_primitives()) {
            const auto& fused_prims = node.get_fused_primitives();
            const bool is_silu_only = fused_prims.size() == 1 &&
                                       fused_prims[0].is_type<activation>() &&
                                       fused_prims[0].deps.empty() &&
                                       fused_prims[0].typed_desc<activation>()->activation_function == activation_func::swish &&
                                       fused_prims[0].typed_desc<activation>()->additional_params.a == 1.0f;
            if (!is_silu_only) {
                return false;
            }
        }

        const auto& conv_node = node.as<convolution>();
        if (conv_node.get_deformable_mode() || conv_node.get_groups() != 1) {
            return false;
        }
        if (conv_node.weights_zero_points_term() || conv_node.activations_zero_points_term() || conv_node.compensation_term()) {
            return false;
        }

        const auto& input_layout = node.get_input_layout(0);
        const auto& weights_layout = conv_node.weights().get_output_layout();
        const auto& output_layout = node.get_output_layout(0);

        if (input_layout.format != format::b_fs_yx_fsv16 || output_layout.format != format::b_fs_yx_fsv16) {
            return false;
        }
        // Wspierane typy danych: f16 lub f32 - ale ten sam typ dla
        // input/output/wag/biasu naraz, bo kernel nie konwertuje miedzy
        // typami (CONV_DT jest jeden, wspolny, patrz get_jit_constants
        // w convolution_bfyx_f16.cpp).
        const auto compute_dt = input_layout.data_type;
        if (compute_dt != data_types::f16 && compute_dt != data_types::f32) {
            return false;
        }
        if (output_layout.data_type != compute_dt || weights_layout.data_type != compute_dt) {
            return false;
        }
        if (conv_node.bias_term() && conv_node.bias().get_output_layout().data_type != compute_dt) {
            return false;
        }

        // Kernel obsluguje fizyczny padding TYLKO na osiach przestrzennych
        // (indeksy 2=Y, 3=X w data_padding - "w tej samej kolejnosci co
        // shape", czyli [batch, feature, y, x]). Padding na batchu/kanalach
        // (indeksy 0,1) nie jest obslugiwany przez ten kernel.
        auto has_non_spatial_padding = [](const cldnn::layout& l) {
            return l.data_padding._lower_size[0] != 0 || l.data_padding._upper_size[0] != 0 ||
                   l.data_padding._lower_size[1] != 0 || l.data_padding._upper_size[1] != 0;
        };
        if (has_non_spatial_padding(input_layout) || has_non_spatial_padding(output_layout)) {
            return false;
        }

        // Tylko filtry 1x1. Zmierzone na yolo26m/480x480 (f32, GPU Xe2), czas
        // konwolucji per ksztalt, ten kernel vs referencyjna sciezka OpenCL:
        //   * 1x1  - CM szybszy praktycznie zawsze, lacznie ok. -380 us na
        //     przebieg (np. 1024x60x60 -> 256x60x60: 581 vs 669 us,
        //     128x60x60 -> 128x60x60: 63 vs 136 us),
        //   * 3x3  - CM WOLNIEJSZY praktycznie zawsze, lacznie ok. +400 us
        //     (np. 512x30x30 -> 64x30x30: 221 vs 123 us, 128x30x30 ->
        //     128x30x30: 821 vs 747 us).
        // Bez tego ograniczenia oba efekty niemal dokladnie sie znosily i
        // caly kernel CM nie dawal zysku na modelu. Dla 3x3 wyspecjalizowany
        // convolution_gpu_bfyx_f16 lepiej wykorzystuje sub-group shuffle do
        // reuzycia linii wejscia miedzy punktami filtra, czego ten
        // jednowatkowy kernel nie ma czym zastapic.
        if (weights_layout.spatial(0) != 1 || weights_layout.spatial(1) != 1) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
