// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include "reorder/reorder_kernel_base.h"
#include <vector>
#include <algorithm>
#include <cstdlib>

namespace kernel_selector {

namespace {
bool post_reorder_fused(const convolution_params& params) {
    if (!params.fused_ops.empty()) {
        if (params.fused_ops.back().GetType() == KernelType::REORDER) {
            return true;
        }
    }

    return false;
}
}  // namespace

ConvolutionKernel_b_fs_yx_fsv16::ConvolutionKernel_b_fs_yx_fsv16() : ConvolutionKernelBase("convolution_gpu_bfyx_f16") {
    std::vector<size_t> outputBlockWidths = { 2, 4, 8 };
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{ w, exeMode });
        }
    }
}

ConvolutionKernel_b_fs_yx_fsv16::AutoTuneOption ConvolutionKernel_b_fs_yx_fsv16::GetAutoTuneOptions(const Params& params,
                                                                                                    int /*autoTuneIndex*/) const {
    const convolution_params& cp = static_cast<const convolution_params&>(params);
    auto x = cp.outputs[0].X().v;
    auto f = cp.outputs[0].Feature().v;

    // Wymuszenie szerokosci kafelka X (OV_GPU_CONV_TX). Kafelek decyduje o
    // rozmiarze akumulatora `dst` i tablicy `line_cache`, czyli wprost o
    // presji rejestrowej, a przez nia o zajetosci EU. Do strojenia.
    static const int forced_tx = []() {
        const char* v = std::getenv("OV_GPU_CONV_TX");
        const int n = (v != nullptr) ? std::atoi(v) : 0;
        return (n == 2 || n == 4 || n == 8) ? n : 0;
    }();
    if (forced_tx != 0)
        return { static_cast<size_t>(forced_tx), EXE_MODE_DEFAULT };

    if (x * f <= 256) {
        if (x <= 8 || x * f <= 128)
            return { 2, EXE_MODE_DEFAULT };
        return {4, EXE_MODE_DEFAULT};
    }
    if (x * f <= 1536) {
        return { 4, EXE_MODE_DEFAULT };
    }
    if (x >= 8 && x < 12 && x * f < 2600)
        return {4, EXE_MODE_DEFAULT};
    if (x < 12 && x * f < 8192)
        return {8, EXE_MODE_DEFAULT};
    return {8, EXE_MODE_AGE_BASED};
}

float ConvolutionKernel_b_fs_yx_fsv16::EstimateOccupancy(const convolution_params& params,
                                                         const ConvolutionTuningData& tuning_data) const {
    auto tuneOptions = GetAutoTuneOptions(params, 0);
    auto blockWidth = tuneOptions.blockWidth;

    auto x = params.outputs[0].X().v;
    auto y = params.outputs[0].Y().v;
    auto f = params.outputs[0].Feature().v;
    auto b = params.outputs[0].Batch().v;

    auto threads = CeilDiv(x, blockWidth) * y * CeilDiv(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor * b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

ConvolutionKernel_b_fs_yx_fsv16::ConvolutionTuningData ConvolutionKernel_b_fs_yx_fsv16::GetTuningParams(const convolution_params& params) const {
    ConvolutionTuningData tuning_data;

    const auto& input = params.inputs[0];

    size_t ic_blocks = CeilDiv(input.Feature().v / params.groups, tuning_data.feature_block_size);

    size_t max_slm_div_factor = params.engineInfo.maxWorkGroupSize / tuning_data.sub_group_size;

    bool slm_exception = params.outputs[0].X().v == 3 && params.outputs[0].Y().v == 3 && params.outputs[0].ElementSize() == 4
                         && params.outputs[0].Feature().v <= 512;

    // Prog zajetosci, ponizej ktorego oplaca sie rozbic redukcje po
    // kanalach wejscia na kilka sub-group liczacych ten sam piksel
    // wyjscia i sumujacych sie przez SLM. Rozbicie kosztuje bariere,
    // zapis/odczyt czesciowych sum przez SLM i dodatkowe sumowanie, wiec
    // "4 fale" to kompromis dostrojony pod duzo starsze iGPU. Da sie go
    // przestawic przez OV_GPU_CONV_OCC, zeby zmierzyc, czy na tym GPU
    // wciaz jest optymalny (0 = calkiem bez rozbijania).
    static const float occupancy_target = []() {
        const char* v = std::getenv("OV_GPU_CONV_OCC");
        return (v != nullptr) ? static_cast<float>(std::atof(v)) : 4.0f;
    }();

    if (params.engineInfo.deviceType == dev_type::integrated_gpu && params.engineInfo.supports_imad && !slm_exception)
        while (ic_blocks % (tuning_data.slm_div_factor * 2) == 0 && (tuning_data.slm_div_factor * 2 <= max_slm_div_factor) &&
               EstimateOccupancy(params, tuning_data) < occupancy_target)
            tuning_data.slm_div_factor *= 2;

    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;

    return tuning_data;
}

ParamsKey ConvolutionKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);

    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    // TODO Add bias per output support to kernel
    // k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_b_fs_yx_fsv16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16::SetDefault(const convolution_params& params,
                                                                                int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    const auto& out = params.outputs[0];

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = autoTune.blockWidth;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = CeilDiv(x, autoTune.blockWidth) * y;
    dispatchData.gws[1] = Align(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor;
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = tuning_data.work_group_size;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    return p.outputs[0].Batch().v == 1 ? FORCE_PRIORITY_2 :  FORCE_PRIORITY_7;
}

bool ConvolutionKernel_b_fs_yx_fsv16::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p) || !ConvolutionCheckInput(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const convolution_params&>(p);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (params.groups > 1) {
        auto outFeaturesPerGroup = output.Feature().v / params.groups;
        auto inFeaturesPerGroup = input.Feature().v / params.groups;
        auto multipleGroupsInputPreload = (tuning_data.feature_block_size % outFeaturesPerGroup == 0) &&
                                          (tuning_data.feature_block_size % inFeaturesPerGroup == 0) &&
                                          (tuning_data.feature_block_size / outFeaturesPerGroup > 1) &&
                                          (tuning_data.feature_block_size / inFeaturesPerGroup > 1) &&
                                          (outFeaturesPerGroup != 1) &&
                                          (inFeaturesPerGroup != 1);
        auto grouped = inFeaturesPerGroup % tuning_data.sub_group_size == 0 &&
                       (outFeaturesPerGroup % tuning_data.sub_group_size == 0 || tuning_data.sub_group_size % outFeaturesPerGroup == 0);

        if (!multipleGroupsInputPreload && !grouped)
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % tuning_data.feature_block_size != 0 || output.Feature().pad.before % tuning_data.feature_block_size != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Not supporting batch padding for different format (reorder-fused case)
    if (input.GetLayout() == DataLayout::b_fs_yx_fsv16 && output.GetLayout() == DataLayout::bfyx) {
        if (output.Batch().pad.before != 0 || output.Batch().pad.after != 0)
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (!params.bias.empty() && params.bias[0].GetDType() != input.GetDType())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16::GetJitConstants(const convolution_params& params,
                                                              const DispatchData& dispatchData) const {
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto jit = Parent::GetJitConstants(params, dispatchData);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    if (post_reorder_fused(params) &&
        input.GetLayout() == DataLayout::b_fs_yx_fsv16 &&
        output.GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("OUTPUT_FORMAT_BFYX", 1));
    }

    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    if (!params.fused_ops.empty()) {
        DataLayout orig_output_layout = output.GetLayout();
        if (post_reorder_fused(params)) {
            orig_output_layout = params.fused_ops.back().GetOpParams<reorder_fuse_params>()->input_layout;
        }
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           {"b", "(feature_block * 16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           blockWidth,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X,
                                           {}, false, "", orig_output_layout };
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "(feature_block * 16)", "y", "(x + i)"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X,
                                              {}, false, "", orig_output_layout };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec, conf_scalar}));
    }

    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1)*params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());

    auto outFeaturesPerGroup = output.Feature().v / params.groups;
    auto inFeaturesPerGroup = input.Feature().v / params.groups;
    auto multipleGroupsInputPreload = (params.groups > 1) &&
                                      (tuning_data.feature_block_size % outFeaturesPerGroup == 0) &&
                                      (tuning_data.feature_block_size % inFeaturesPerGroup == 0) &&
                                      (tuning_data.feature_block_size / outFeaturesPerGroup > 1) &&
                                      (tuning_data.feature_block_size / inFeaturesPerGroup > 1);

    if (multipleGroupsInputPreload)
        jit.AddConstant(MakeJitConstant("MULTIPLE_GROUPS_INPUT_PRELOAD", 1));

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", tuning_data.sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));
    jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(inFeaturesPerGroup, tuning_data.feature_block_size)));

    // Sciezka DPAS/XMX (tf32) w wewnetrznej petli MAC kernela.
    // OV_GPU_CONV_DPAS wybiera wariant, zeby dalo sie zmierzyc kilka
    // wersji w JEDNYM buildzie (przeplot ABBA) - porownywanie
    // bezwzglednych czasow miedzy buildami jest w tym projekcie
    // niewiarygodne przez dwa stany zegara GPU:
    //   0 - wylaczone (referencyjna sciezka mad)
    //   1 - DPAS: lokalny select + shuffle float4 (najlepszy poprawny)
    //   2 - SONDA: DPAS bez przetasowania miedzy lane'ami (wynik zly,
    //       mierzy koszt samej transpozycji operandu `a`)
    //   3 - SONDA: DPAS z jednym akumulatorem (wynik poprawny, mierzy
    //       koszt presji rejestrowej drugiego akumulatora)
    //   5 - SONDA: stara wersja tasujaca caly float8 (wynik ten sam co 1,
    //       ale dwa razy wiecej przetasowan)
    //
    // Warunki wymagane przez rozszerzenie i przez uklad operandow:
    //  - tf32 dziala tylko na danych 32-bitowych (wejscie i wagi f32),
    //  - N w DPAS jest rowne rozmiarowi sub-group i musi wynosic 16,
    //  - M (= OUTPUT_X_BLOCK_SIZE) moze byc 1/2/4/8; obslugujemy 4 i 8,
    //  - slice kanalow musi byc pelnym blokiem 16 (dwa DPAS-y po K=8),
    //  - sciezki grupowe maja inny uklad wag i sa wylaczone.
    static const int dpas_mode = []() {
        const char* v = std::getenv("OV_GPU_CONV_DPAS");
        return (v != nullptr) ? std::atoi(v) : 0;
    }();
    const bool use_dpas_tf32 = dpas_mode != 0 &&
                               params.groups == 1 &&
                               !multipleGroupsInputPreload &&
                               input.GetDType() == Datatype::F32 &&
                               params.weights.GetDType() == WeightsType::F32 &&
                               tuning_data.sub_group_size == 16 &&
                               tuning_data.feature_block_size == 16 &&
                               (blockWidth == 4 || blockWidth == 8);
    if (use_dpas_tf32) {
        jit.AddConstant(MakeJitConstant("CONV_USE_DPAS_TF32", 1));
        if (dpas_mode == 2)
            jit.AddConstant(MakeJitConstant("CONV_DPAS_NO_SHUFFLE", 1));
        if (dpas_mode == 3)
            jit.AddConstant(MakeJitConstant("CONV_DPAS_ONE_ACC", 1));
        // Tryb 6: operand `a` budowany przez inline vISA (deinterleave GRF).
        // Tryb 7: smoke test - sprawdza tylko, czy IGC akceptuje inline asm.
        // Oba wymagaja blockWidth == 8.
        if (dpas_mode == 6 && blockWidth == 8)
            jit.AddConstant(MakeJitConstant("CONV_DPAS_ASM", 1));
        if (dpas_mode == 7 && blockWidth == 8)
            jit.AddConstant(MakeJitConstant("CONV_DPAS_ASM_SMOKE", 1));
        if (dpas_mode != 2 && dpas_mode != 5)
            jit.AddConstant(MakeJitConstant("CONV_DPAS_HALF_SHUFFLE", 1));
    }

    // Sondy ruchu pamieciowego (OV_GPU_CONV_PROBE). WYNIK JEST ZLY -
    // sluza tylko do rozbicia narzutu kernela na skladniki:
    //   1 - odczyty wag przypiete do 2 KB (znika ruch L3 dla wag)
    //   2 - odczyty wejscia przypiete do 8 KB
    //   4 - wypelnianie line_cache ze stalymi indeksami (bez marginesow)
    static const int probe_mode = []() {
        const char* v = std::getenv("OV_GPU_CONV_PROBE");
        return (v != nullptr) ? std::atoi(v) : 0;
    }();
    if (probe_mode & 1)
        jit.AddConstant(MakeJitConstant("CONV_PROBE_WEI_L1", 1));
    if (probe_mode & 2)
        jit.AddConstant(MakeJitConstant("CONV_PROBE_SRC_L1", 1));
    if (probe_mode & 4)
        jit.AddConstant(MakeJitConstant("CONV_PROBE_STATIC_FILL", 1));
    if (probe_mode & 8)
        jit.AddConstant(MakeJitConstant("CONV_PROBE_NO_MAC", 1));

    // Programowy prefetch LSC danych nastepnej iteracji icb. Domyslnie
    // wlaczony; wylaczalny do pomiaru jego realnego zysku.
    static const bool use_prefetch = []() {
        const char* v = std::getenv("OV_GPU_CONV_PREFETCH");
        return (v == nullptr) || (std::atoi(v) != 0);
    }();
    if (use_prefetch)
        jit.AddConstant(MakeJitConstant("CONV_USE_PREFETCH", 1));

    // Liczba niezaleznych akumulatorow na sciezce mad (OV_GPU_CONV_ACC).
    // Domyslnie 1, czyli tak jak bylo. Patrz komentarz w kernelu.
    static const int mad_acc = []() {
        const char* v = std::getenv("OV_GPU_CONV_ACC");
        const int n = (v != nullptr) ? std::atoi(v) : 1;
        return (n == 2 || n == 4) ? n : 1;
    }();
    jit.AddConstant(MakeJitConstant("CONV_MAD_ACC", mad_acc));

    if (params.outputs[0].Feature().v % tuning_data.feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }
    if (inFeaturesPerGroup % tuning_data.feature_block_size != 0 && !multipleGroupsInputPreload) {
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetTunedKernelsDataByIndex(const Params& params,
                                                                        const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetKernelsDataForAutoTune(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector
