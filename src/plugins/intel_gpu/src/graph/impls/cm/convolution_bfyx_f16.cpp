// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "convolution_bfyx_f16.hpp"

#include <memory>
#include <tuple>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

// Musi sie zgadzac z FEATURE_SLICE_SIZE w convolution_bfyx_f16.cm.
constexpr size_t CONV_FEATURE_SLICE_SIZE = 16;

// Ile pozycji X wyjscia liczy jeden watek (OUTPUT_X_BLOCK_SIZE w .cm) oraz
// ile 16-kanalowych slice'ow wyjscia (CONV_OC_BLOCKS_PER_THREAD w .cm).
// Oba parametry powiekszaja kafelek liczony przez jeden watek, czyli
// poprawiaja stosunek "MAC-ow na bajt odczytu", ale zmniejszaja liczbe
// watkow i zwiekszaja zuzycie rejestrow (akumulator ma OCB*TX*16 floatow).
//
// Zmierzone na yolo26m/480x480: przy sztywnym TX=16 male warstwy (np.
// 768x15x15 -> 512x15x15, gdzie wychodzi tylko ~480 watkow na cale GPU)
// wyraznie odstawaly od duzych - czyste zaglodzenie rownoleglosci, nie
// przepustowosc. Stad dobor per warstwa: bierzemy najwiekszy kafelek, ktory
// wciaz zostawia CONV_MIN_THREADS watkow.
constexpr size_t CONV_MIN_THREADS = 2048;
// TX=32 (i OCB=4) probowane - kompilator CM raportuje wtedy
// "Spill memory used = 896 bytes for kernel ..." na stderr, czyli kafelek
// nie miesci sie juz w domyslnym pliku 128 GRF.
constexpr size_t CONV_MAX_X_BLOCK_SIZE = 16;
constexpr size_t CONV_MIN_X_BLOCK_SIZE = 4;
// Gorna granica OCB: przy TX=16 akumulator zajmuje OCB*16 rejestrow GRF po
// 64B; dla OCB=4 razem z buforem linii wejscia i buforem zapisu przekracza
// to domyslny plik rejestrow 128 GRF - kompilator CM raportuje wtedy wprost
// "Spill memory used = ... bytes for kernel ..." na stderr, a kernel jest
// wolniejszy niz bez tej optymalizacji.
constexpr size_t CONV_MAX_OC_BLOCKS_PER_THREAD = 2;

// RepeatCount instrukcji cm_dpas - ustala liczbe pozycji X liczonych przez
// jeden watek na sciezce XMX. Musi sie zgadzac z DPAS_M w kernelu .cm.
constexpr size_t CONV_DPAS_M = 8;

// Czy uzyc systolicznej sciezki cm_dpas (tf32) zamiast mad-ow na FPU.
// XMX nie ma wariantu f32 x f32, a tf32 to jedyna precyzja zachowujaca pelny
// zakres wykladnika f32 - dla f16 istnieje wydajniejszy CM_PRECISION_HF, ale
// ten kernel dla f16 i tak nie wygrywa z referencyjnym OpenCL, wiec sciezke
// XMX wlaczamy wylacznie dla f32. Filtry sa juz zawezone do 1x1 przez
// validate_impl, ale sprawdzamy to jawnie, bo get_tiling musi byc spojne z
// tym, co trafi do jit-a.
bool use_dpas(const RuntimeParams& params) {
    if (params.get_input_layout(0).data_type != data_types::f32) {
        return false;
    }
    const auto& weights_layout = params.weights_layout.value();
    return weights_layout.spatial(0) == 1 && weights_layout.spatial(1) == 1;
}
// Odrzucony wariant: kafelkowanie takze po Y (kilka wierszy wyjscia na watek),
// zeby raz wczytana waga obsluzyla TY razy wiecej pozycji wyjscia. Zmierzone
// na yolo26m/480x480 - konsekwentnie WOLNIEJSZE (do +25% czasu warstw CM przy
// TY=2 stosowanym szeroko), i to bez spillu rejestrow. Wagi sa najwyrazniej i
// tak dobrze trzymane w L2 (wszystkie rownolegle watki czytaja ten sam slice),
// wiec jedynym efektem bylo obciecie liczby watkow, czyli gorsze ukrywanie
// latencji.

struct ConvTiling {
    size_t x_block;   // OUTPUT_X_BLOCK_SIZE
    size_t oc_blocks_per_thread;  // CONV_OC_BLOCKS_PER_THREAD
};

// Uwaga: musi zalezec WYLACZNIE od output_layout, ktory wchodzi do
// kernel_impl_params::hash(), a wiec do klucza cache'a skompilowanego kernela
// (get_entry_point) - inaczej dwa wezly o tej samej nazwie kernela mogłyby
// dostac rozne rozwiniecia petli przy tym samym skompilowanym kodzie.
ConvTiling get_tiling(const RuntimeParams& params) {
    const auto& output_layout = params.get_output_layout(0);
    const size_t out_x = static_cast<size_t>(output_layout.spatial(0));
    const size_t out_y = static_cast<size_t>(output_layout.spatial(1));
    const size_t batch = static_cast<size_t>(output_layout.batch());
    const size_t oc_blocks =
        (static_cast<size_t>(output_layout.feature()) + CONV_FEATURE_SLICE_SIZE - 1) / CONV_FEATURE_SLICE_SIZE;

    auto threads_for = [&](size_t tx, size_t ocb) {
        return ((out_x + tx - 1) / tx) * out_y * (oc_blocks / ocb) * batch;
    };

    auto pick_oc_blocks = [&](size_t tx) {
        for (size_t cand = CONV_MAX_OC_BLOCKS_PER_THREAD; cand > 1; cand /= 2) {
            if (oc_blocks % cand == 0 && threads_for(tx, cand) >= CONV_MIN_THREADS) {
                return cand;
            }
        }
        return static_cast<size_t>(1);
    };

    // Na sciezce XMX kafelek X jest sztywno rowny RepeatCount instrukcji dpas,
    // wiec adaptacyjne zmniejszanie kafelka jest niedostepne - liczbe watkow
    // reguluje juz tylko OCB.
    if (use_dpas(params)) {
        return {CONV_DPAS_M, pick_oc_blocks(CONV_DPAS_M)};
    }

    size_t tx = CONV_MAX_X_BLOCK_SIZE;
    while (tx > CONV_MIN_X_BLOCK_SIZE && threads_for(tx, 1) < CONV_MIN_THREADS) {
        tx /= 2;
    }

    return {tx, pick_oc_blocks(tx)};
}

// Musi sie zgadzac z CONV_ACTIVATION_* w convolution_bfyx_f16.cm.
constexpr int CONV_ACTIVATION_NONE = 0;
constexpr int CONV_ACTIVATION_SILU = 2;

constexpr auto get_conv_build_options() {
    return " -cmc";
}

class ConvolutionBfyxF16Generator : public KernelGenerator {
public:
    ConvolutionBfyxF16Generator() : KernelGenerator("CMconvolution_bfyx_f16") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_conv_build_options();
    }

    // kernel_impl_params::hash() (i wiec bazowy KernelGenerator::get_entry_point,
    // uzywany przez framework jako REALNY klucz cache'a/nazwa skompilowanego
    // kernela - patrz kd.code->entry_point w impls/cm/utils/kernel_generator.cpp)
    // NIE uwzglednia weights_layout ani bias_layout. Dwa rozne wezly o
    // identycznym input/output layout i tych samych fused_desc (np. ten sam
    // ksztalt konwolucji, ale jeden z biasem a drugi bez) dostawalyby wiec TA
    // SAMA nazwe/hash kernela mimo ze kompilowany tekst (liczba argumentow,
    // CONV_FILTER_SIZE_*) jest inny - realnie zaobserwowane jako "Error set
    // arg 2 ... error code: -50" (kolizja cache'a kernela: framework
    // odtwarzal skompilowany obiekt jednego wezla, ale wiazal argumenty wg
    // sygnatury drugiego). Doklejamy wiec brakujace roznicujace skladniki.
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& weights_layout = params.weights_layout.value();
        const int activation_mode = params.has_fused_primitives() ? CONV_ACTIVATION_SILU : CONV_ACTIVATION_NONE;
        return KernelGenerator::get_entry_point(params) +
                "_b" + std::to_string(params.bias_layout.has_value() ? 1 : 0) +
                "_a" + std::to_string(activation_mode) +
                "_f" + std::to_string(weights_layout.spatial(0)) + "x" + std::to_string(weights_layout.spatial(1));
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        auto desc = params.typed_desc<convolution>();
        const auto& input_layout = params.get_input_layout(0);
        const auto& weights_layout = params.weights_layout.value();
        const auto& output_layout = params.get_output_layout(0);

        uint32_t stride_x, stride_y, stride_z;
        std::tie(stride_x, stride_y, stride_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(desc->stride, 1);
        uint32_t dilation_x, dilation_y, dilation_z;
        std::tie(dilation_x, dilation_y, dilation_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(desc->dilation, 1);
        uint32_t pad_x, pad_y, pad_z;
        std::tie(pad_x, pad_y, pad_z) = ov::intel_gpu::get_xyz<ov::CoordinateDiff, uint32_t>(desc->padding_begin, 0);

        // data_padding jest indeksowany w kolejnosci shape'u [batch,
        // feature, y, x] (patrz validate_impl w convolution_bfyx_f16.hpp) -
        // stad indeksy 2=Y, 3=X.
        const auto& in_pad = input_layout.data_padding;
        const auto& out_pad = output_layout.data_padding;

        // Fuzja aktywacji: jedyny obslugiwany przypadek to pojedynczy,
        // niezalezny SiLU/Swish(beta=1) - validate_impl juz to wymusil,
        // wiec tutaj wystarczy sprawdzic czy fused_desc w ogole istnieje.
        int activation_mode = CONV_ACTIVATION_NONE;
        if (params.has_fused_primitives()) {
            activation_mode = CONV_ACTIVATION_SILU;
        }

        // CONV_DT musi odpowiadac faktycznemu typowi danych input/output/wag
        // (validate_impl w convolution_bfyx_f16.hpp juz wymusza, ze wszystkie
        // trzy - i bias, jesli jest - maja TEN SAM typ, wiec wystarczy
        // sprawdzic input_layout). Wspierane: f16 i f32.
        const std::string conv_dt = input_layout.data_type == data_types::f32 ? "float" : "half";

        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("CONV_DT", conv_dt),
            make_jit_constant("CONV_IFM_NUM", static_cast<size_t>(input_layout.feature())),
            make_jit_constant("CONV_OFM_NUM", static_cast<size_t>(output_layout.feature())),
            make_jit_constant("CONV_IN_SIZE_X", static_cast<size_t>(input_layout.spatial(0))),
            make_jit_constant("CONV_IN_SIZE_Y", static_cast<size_t>(input_layout.spatial(1))),
            make_jit_constant("CONV_OUT_SIZE_X", static_cast<size_t>(output_layout.spatial(0))),
            make_jit_constant("CONV_OUT_SIZE_Y", static_cast<size_t>(output_layout.spatial(1))),
            make_jit_constant("CONV_FILTER_SIZE_X", static_cast<size_t>(weights_layout.spatial(0))),
            make_jit_constant("CONV_FILTER_SIZE_Y", static_cast<size_t>(weights_layout.spatial(1))),
            make_jit_constant("CONV_STRIDE_X", stride_x),
            make_jit_constant("CONV_STRIDE_Y", stride_y),
            make_jit_constant("CONV_DILATION_X", dilation_x),
            make_jit_constant("CONV_DILATION_Y", dilation_y),
            make_jit_constant("CONV_PAD_X", pad_x),
            make_jit_constant("CONV_PAD_Y", pad_y),
            make_jit_constant("CONV_INPUT_PAD_BEFORE_X", static_cast<size_t>(in_pad._lower_size[3])),
            make_jit_constant("CONV_INPUT_PAD_AFTER_X", static_cast<size_t>(in_pad._upper_size[3])),
            make_jit_constant("CONV_INPUT_PAD_BEFORE_Y", static_cast<size_t>(in_pad._lower_size[2])),
            make_jit_constant("CONV_INPUT_PAD_AFTER_Y", static_cast<size_t>(in_pad._upper_size[2])),
            make_jit_constant("CONV_OUTPUT_PAD_BEFORE_X", static_cast<size_t>(out_pad._lower_size[3])),
            make_jit_constant("CONV_OUTPUT_PAD_AFTER_X", static_cast<size_t>(out_pad._upper_size[3])),
            make_jit_constant("CONV_OUTPUT_PAD_BEFORE_Y", static_cast<size_t>(out_pad._lower_size[2])),
            make_jit_constant("CONV_OUTPUT_PAD_AFTER_Y", static_cast<size_t>(out_pad._upper_size[2])),
            make_jit_constant("OUTPUT_X_BLOCK_SIZE", get_tiling(params).x_block),
            make_jit_constant("CONV_OC_BLOCKS_PER_THREAD", get_tiling(params).oc_blocks_per_thread),
            make_jit_constant("CONV_WITH_BIAS", params.bias_layout.has_value() ? 1 : 0),
            make_jit_constant("CONV_ACTIVATION", activation_mode),
            make_jit_constant("CONV_USE_DPAS", use_dpas(params) ? 1 : 0),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::WEIGHTS, 0});
        if (params.bias_layout.has_value()) {
            args.push_back({ArgumentDescriptor::Types::BIAS, 0});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());
            const auto& output_layout = params.get_output_layout(0);

            const size_t ofm_num = static_cast<size_t>(output_layout.feature());
            const size_t out_x = static_cast<size_t>(output_layout.spatial(0));
            const size_t out_y = static_cast<size_t>(output_layout.spatial(1));
            const size_t batch = static_cast<size_t>(output_layout.batch());

            const size_t oc_blocks = (ofm_num + CONV_FEATURE_SLICE_SIZE - 1) / CONV_FEATURE_SLICE_SIZE;
            const auto tiling = get_tiling(params);
            const size_t x_blocks = (out_x + tiling.x_block - 1) / tiling.x_block;

            // Wymiar 0 = grupa slice'ow kanalow wyjscia (patrz komentarz przy
            // cm_group_id w convolution_bfyx_f16.cm): sasiednie watki maja
            // dzielic ten sam kafelek wejscia, zeby czytac go z L1/L2
            // zamiast OC_BLOCKS razy z DRAM.
            auto& wgs = kd.params.workGroups;
            wgs.global = {oc_blocks / tiling.oc_blocks_per_thread, x_blocks * out_y, batch};
            wgs.local = {1, 1, 1};
        }};
    }
};

class ConvolutionBfyxF16Impl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::ConvolutionBfyxF16Impl)

    Stage::Ptr conv = make_stage<ConvolutionBfyxF16Generator>();

    ConvolutionBfyxF16Impl() : PrimitiveImplOCL(ConvolutionBfyxF16ImplementationManager::get_type_info_static()) {}
    ConvolutionBfyxF16Impl(const program_node& node, const RuntimeParams& params) : ConvolutionBfyxF16Impl() {
        add_stage(conv, params);

        // Kernel zaklada wagi w ukladzie os_is_yx_isv16_osv16 (patrz
        // komentarz w convolution_bfyx_f16.cm) - jesli aktualny layout wag
        // jest inny, zglaszamy WeightsReorderParams, zeby wspolny mechanizm
        // frameworku (ten sam co dla onednn/ocl fully_connected) przelozyl
        // wagi jednorazowo na wlasciwy fizyczny uklad przed pierwszym
        // uzyciem tego kernela.
        const auto& current_weights_layout = params.weights_layout.value();
        if (current_weights_layout.format != format::os_is_yx_isv16_osv16) {
            cldnn::layout target_weights_layout = current_weights_layout;
            target_weights_layout.format = format::os_is_yx_isv16_osv16;
            _weights_reorder_params = std::make_shared<WeightsReorderParams>(current_weights_layout, target_weights_layout);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<ConvolutionBfyxF16Impl>(this);
    }

    // Bazowa PrimitiveImplOCL::get_arguments() wypelnia tylko inputs/outputs/
    // fused_op_inputs/shape_info/intermediates - WEIGHTS i BIAS (zadeklarowane
    // w get_arguments_desc() powyzej) trzeba dowiazac tutaj jawnie, tak samo
    // jak robi to impls/ocl/convolution.cpp. Bez tego data.weights zostaje
    // nullptr i kazde wykonanie tego kernela pada na "Error set arg 2 ...
    // error code: -50" (CL_INVALID_ARG_VALUE z set_kernel_arg w ocl_stream.cpp,
    // ktore dla WEIGHTS/BIAS nie ma OPENVINO_ASSERT-a jak INPUT/OUTPUT).
    [[nodiscard]] kernel_arguments_data get_arguments(const primitive_inst& instance) const override {
        kernel_arguments_data args = PrimitiveImplCM::get_arguments(instance);
        const auto& conv_instance = static_cast<const convolution_inst&>(instance);
        args.weights = conv_instance.weights_memory();
        args.bias = conv_instance.bias_term() ? conv_instance.bias_memory() : nullptr;
        return args;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> ConvolutionBfyxF16ImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<convolution>());
    return std::make_unique<ConvolutionBfyxF16Impl>(node, params);
}

}  // namespace ov::intel_gpu::cm

// cldnn::convolution jest juz zarejestrowany w impls/ocl/convolution.cpp -
// rejestrujemy tu tylko nowy typ implementacji.
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::ConvolutionBfyxF16Impl)
