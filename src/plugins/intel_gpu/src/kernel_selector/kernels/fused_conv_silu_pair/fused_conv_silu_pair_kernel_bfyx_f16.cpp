// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_conv_silu_pair_kernel_bfyx_f16.h"

#include "common_tools.h"
#include "kernel_selector_utils.h"

#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {
namespace {

constexpr size_t feature_slice_size = 16;

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

}  // namespace

ParamsKey FusedConvSiluPairKernel_bfyx_f16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    // Wagi i biasy sa plaskimi stalymi 1-D, wiec przychodza jako bfyx.
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey FusedConvSiluPairKernel_bfyx_f16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();
    return k;
}

FusedConvSiluPairKernel_bfyx_f16::TuningData
FusedConvSiluPairKernel_bfyx_f16::GetTuningData(const fused_conv_silu_pair_params& params) const {
    TuningData td;

    td.ic_blocks = ceil_div(params.in_features, feature_slice_size);
    td.mid_ic_blocks = ceil_div(params.mid_features, feature_slice_size);
    td.oc_blocks = ceil_div(params.out_features, feature_slice_size);

    // Szerokosc kafelka X. Duzy kafelek lepiej amortyzuje odczyt wag i
    // line_cache, ale liniowo powieksza bufor SLM na tensor posredni, wiec
    // schodzimy w dol, dopoki sie nie zmiesci.
    const size_t elem_size = BytesPerElement(params.inputs[0].GetDType());
    const size_t slm_budget = static_cast<size_t>(params.engineInfo.maxLocalMemSize);

    td.block_width = 8;
    while (td.block_width > 2) {
        const size_t slm_bytes = td.mid_ic_blocks * td.block_width * feature_slice_size * elem_size;
        if (slm_bytes <= slm_budget && td.block_width <= params.outputs[0].X().v)
            break;
        td.block_width /= 2;
    }

    // Work-group musi pokryc wszystkie slice'y kanalow posrednich i
    // wyjsciowych dla jednego kafelka. Uzycie wszystkich slice'ow naraz
    // (num_sub_groups == slices) daje po jednej iteracji na sub-grupe, co
    // pomiarowo wypada gorzej niz dwie: przy jednej iteracji za malo pracy
    // przypada na watek, zeby ukryc opoznienia odczytow, a przy czterech i
    // wiecej rosnie zajetosc SLM na work-item i spada liczba rownoleglych
    // work-groupow na subslice. Stad polowa liczby slice'ow.
    //
    // Dolne ograniczenie 2 jest obowiazkowe dla poprawnosci: przy jednej
    // sub-group na work-group bariera miedzy faza 1 a faza 2 jest zbedna z
    // punktu widzenia modelu wykonania i kompilator ja usuwa, po czym moze
    // przeplesc obie petle - faza 2 czyta wtedy slice'y SLM, ktorych faza 1
    // jeszcze nie zapisala. Dwie sub-group wymuszaja prawdziwa bariere.
    const size_t max_sub_groups = std::max<size_t>(1, params.engineInfo.maxWorkGroupSize / td.sub_group_size);
    const size_t slices = std::max(td.mid_ic_blocks, td.oc_blocks);
    const size_t min_sub_groups = std::min<size_t>(2, max_sub_groups);
    td.num_sub_groups = std::clamp(slices / 2, min_sub_groups, max_sub_groups);

    return td;
}

CommonDispatchData FusedConvSiluPairKernel_bfyx_f16::SetDefault(const fused_conv_silu_pair_params& params,
                                                               const TuningData& td) const {
    CommonDispatchData dispatchData;

    const auto& out = params.outputs[0];
    const size_t x_blocks = ceil_div(out.X().v, td.block_width);
    const size_t work_group_size = td.num_sub_groups * td.sub_group_size;

    dispatchData.gws = {x_blocks * out.Y().v, work_group_size, out.Batch().v};
    dispatchData.lws = {1, work_group_size, 1};

    return dispatchData;
}

JitConstants FusedConvSiluPairKernel_bfyx_f16::GetJitConstants(const fused_conv_silu_pair_params& params,
                                                              const TuningData& td) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& out = params.outputs[0];

    // Ile pozycji wejscia trzeba zbuforowac w line_cache, zeby obsluzyc caly
    // kafelek dla wszystkich pozycji kw filtra conv1.
    // UWAGA: w przeciwienstwie do convolution_kernel_b_fs_yx_fsv16 NIE przycinamy
    // tego do szerokosci wejscia. Tamten kernel moze sobie na to pozwolic, bo
    // prymityw konwolucji zada paddingu tensora wejsciowego (in.X().pad.Total()
    // pokrywa wtedy halo). Tutaj wejscie jest niepaddowane, wiec przyciecie
    // sprawiloby, ze indeks line_cache[kw*DILATION + STRIDE*i] wychodzi poza
    // tablice (np. stride 2, blok 8, filtr 3x3, wejscie 16 -> potrzeba 17 pozycji).
    // Pozycje wypadajace poza obrazem i tak sa jawnie zerowane w kernelu.
    const size_t input_line_size =
        params.stride.x * (td.block_width - 1) + (params.filterSize1.x - 1) * params.dilation.x + 1;

    jit.AddConstants({
        MakeJitConstant("SUB_GROUP_SIZE", td.sub_group_size),
        MakeJitConstant("OUTPUT_X_BLOCK_SIZE", td.block_width),
        MakeJitConstant("X_BLOCKS", ceil_div(out.X().v, td.block_width)),
        MakeJitConstant("INPUT_LINE_SIZE", input_line_size),
        MakeJitConstant("NUM_SUB_GROUPS", td.num_sub_groups),
        MakeJitConstant("WORK_GROUP_SIZE", td.num_sub_groups * td.sub_group_size),

        MakeJitConstant("IC_BLOCKS", td.ic_blocks),
        MakeJitConstant("MID_IC_BLOCKS", td.mid_ic_blocks),
        MakeJitConstant("OC_BLOCKS", td.oc_blocks),
        MakeJitConstant("MID_FEATURE_NUM", params.mid_features),

        MakeJitConstant("FILTER1_SIZE_X", params.filterSize1.x),
        MakeJitConstant("FILTER1_SIZE_Y", params.filterSize1.y),
        MakeJitConstant("STRIDE_SIZE_X", params.stride.x),
        MakeJitConstant("STRIDE_SIZE_Y", params.stride.y),
        MakeJitConstant("DILATION_SIZE_X", params.dilation.x),
        MakeJitConstant("DILATION_SIZE_Y", params.dilation.y),
        MakeJitConstant("PADDING_SIZE_X", params.padding.x),
        MakeJitConstant("PADDING_SIZE_Y", params.padding.y),

        MakeJitConstant("SWISH_BETA_1", params.swish_beta1),
        MakeJitConstant("SWISH_BETA_2", params.swish_beta2),
    });

    // Flagi "czy ostatni slice fsv16 jest niepelny" - wlaczaja w kernelu
    // sciezki skalarne z maskowaniem kanalow paddingowych. Definiowane tylko
    // gdy sa potrzebne, tak jak w convolution_gpu_bfyx_f16.
    if (params.in_features % feature_slice_size != 0)
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    if (params.mid_features % feature_slice_size != 0)
        jit.AddConstant(MakeJitConstant("MID_LEFTOVERS", 1));
    if (params.out_features % feature_slice_size != 0)
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));

    return jit;
}

bool FusedConvSiluPairKernel_bfyx_f16::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (params.GetType() != KernelType::FUSED_CONV_SILU_PAIR)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& p = static_cast<const fused_conv_silu_pair_params&>(params);

    // Kernel jest w pelni statyczny - dispatch i rozmiar SLM sa zapieczone w JIT.
    if (p.is_shape_agnostic || p.has_dynamic_tensors())
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (p.inputs.size() != 5 || p.outputs.size() != 1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Aktywacje i wyjscie musza byc w blokowym fsv16; wagi/biasy sa plaskimi
    // buforami 1-D, wiec ich layout nie jest tu istotny.
    if (p.inputs[0].GetLayout() != DataLayout::b_fs_yx_fsv16 ||
        p.outputs[0].GetLayout() != DataLayout::b_fs_yx_fsv16)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Kernel adresuje wszystkie piec wejsc rodzina makr DT_INPUT_*, ktora jest
    // zwiazana z INPUT0_TYPE - wiec typy musza byc identyczne.
    const auto dt = p.inputs[0].GetDType();
    if (dt != Datatype::F16 && dt != Datatype::F32)
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    for (const auto& input : p.inputs) {
        if (input.GetDType() != dt)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    if (p.outputs[0].GetDType() != dt)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Brak wsparcia dla fused ops - zfuzowany blok i tak konczy sie SiLU.
    if (!p.fused_ops.empty())
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Conv2 jest 1x1/stride1, wiec kafelek posredni i wyjsciowy sa identyczne.
    if (p.inputs[0].Feature().v != p.in_features || p.outputs[0].Feature().v != p.out_features)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (p.in_features == 0 || p.mid_features == 0 || p.out_features == 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto td = GetTuningData(p);
    if (td.block_width < 2)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Tensor posredni musi sie zmiescic w SLM - to jest twarde ograniczenie
    // calego pomyslu na te fuzje.
    const size_t slm_bytes =
        td.mid_ic_blocks * td.block_width * feature_slice_size * BytesPerElement(p.inputs[0].GetDType());
    if (slm_bytes > static_cast<size_t>(p.engineInfo.maxLocalMemSize))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (td.num_sub_groups * td.sub_group_size > p.engineInfo.maxWorkGroupSize)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    return true;
}

KernelsData FusedConvSiluPairKernel_bfyx_f16::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    const auto& prim_params = static_cast<const fused_conv_silu_pair_params&>(params);

    const auto td = GetTuningData(prim_params);
    const auto dispatchData = SetDefault(prim_params, td);

    KernelData kd = KernelData::Default<fused_conv_silu_pair_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params, td);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,          // weights
                     false,          // bias
                     5,              // number_of_inputs
                     0,              // number_of_inputs_for_fused_prims
                     1,              // number_of_outputs
                     false);         // is_dynamic

    return {kd};
}

KernelsPriority FusedConvSiluPairKernel_bfyx_f16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

}  // namespace kernel_selector
