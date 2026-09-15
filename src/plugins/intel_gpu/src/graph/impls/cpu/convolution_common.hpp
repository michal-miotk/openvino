// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "openvino/core/parallel.hpp"

#include <algorithm>
#include <cstdint>

namespace cldnn {
namespace cpu {

// Number of features processed together as a single slice.
// This mirrors FEATURE_SLICE_SIZE / the SUB_GROUP_SIZE (16) used by
// convolution_gpu_bfyx_f16.cl: that kernel assigns one sub-group (16 work
// items) to compute 16 consecutive output-feature-map (OFM) values for a
// given spatial position, while walking the input feature maps (IFM) in the
// same 16-wide slices. We keep the same slice granularity here so the host
// computation follows the exact same blocking scheme as the OpenCL kernel.
constexpr int64_t feature_slice_size = 16;

// Fast per-(b,f,x,y) address calculator for b_fs_yx_fsv16 (blocked-by-16-on-feature, `blocked
// = true`) and plain unblocked 4D formats such as the default weights format oiyx (`blocked =
// false`).
//
// layout::get_linear_offset() is fully general (any format/padding) but is comparatively
// expensive per call (allocates small vectors and does bounds checking), which matters a lot
// here since the reference convolution calls it once per (b, f, x, y) tuple touched - i.e.
// potentially tens of millions of times for a realistically sized tensor.
//
// Instead we measure this layout's per-axis address strides ("pitches") ONCE, via finite
// differences of get_linear_offset() itself (so we never have to hand-derive/hard-code the
// format's internal block/ordering conventions - the already-tested generic implementation
// remains the single source of truth for correctness). After that, addressing every element is
// just a handful of integer multiply-adds, mirroring how the .cl kernel itself only ever does
// pointer/index arithmetic once INPUT0_GET_INDEX-style macros are expanded, instead of looking
// anything up per element.
class fast_offset {
public:
    fast_offset(const layout& l, bool blocked) : blocked_(blocked) {
        const auto zero = tensor(0, 0, 0, 0, 0, 0);
        base_ = static_cast<int64_t>(l.get_linear_offset(zero));

        const auto shape = l.get_shape();  // [B, F, Y, X]
        if (shape[0] > 1)
            pitch_b_ = static_cast<int64_t>(l.get_linear_offset(tensor(1, 0, 0, 0, 0, 0))) - base_;
        if (shape[3] > 1)
            pitch_x_ = static_cast<int64_t>(l.get_linear_offset(tensor(0, 0, 1, 0, 0, 0))) - base_;
        if (shape[2] > 1)
            pitch_y_ = static_cast<int64_t>(l.get_linear_offset(tensor(0, 0, 0, 1, 0, 0))) - base_;
        if (shape[1] > 1)
            pitch_f_ = static_cast<int64_t>(l.get_linear_offset(tensor(0, 1, 0, 0, 0, 0))) - base_;
        // For a b_fs_yx_fsv16 layout, stepping the feature index from 0 to 16 crosses into the
        // next feature-slice: the resulting address jump ("slice_pitch_") is generally different
        // from 16 * pitch_f_ (which only holds *within* a single 16-wide slice, since fsv is the
        // fastest-varying/contiguous sub-dimension of the format). We measure that jump directly
        // too, so no assumption about slice layout beyond "16 contiguous features per slice" is
        // hard-coded here.
        if (blocked_ && shape[1] > feature_slice_size)
            slice_pitch_ = static_cast<int64_t>(l.get_linear_offset(tensor(0, static_cast<tensor::value_type>(feature_slice_size), 0, 0, 0, 0))) - base_;
    }

    inline int64_t offset(int64_t b, int64_t f, int64_t x, int64_t y) const {
        int64_t off = base_ + b * pitch_b_ + x * pitch_x_ + y * pitch_y_;
        if (blocked_)
            off += (f / feature_slice_size) * slice_pitch_ + (f % feature_slice_size) * pitch_f_;
        else
            off += f * pitch_f_;
        return off;
    }

    int64_t pitch_f() const { return pitch_f_; }

private:
    bool blocked_;
    int64_t base_ = 0;
    int64_t pitch_b_ = 0;
    int64_t pitch_f_ = 0;
    int64_t pitch_x_ = 0;
    int64_t pitch_y_ = 0;
    int64_t slice_pitch_ = 0;
};

// Plain scalar convolution, used as the fallback for any data type/ISA combination not covered
// by a specialized vectorized path (see convolution_avx2.cpp for the f32 AVX2 version).
//
// Parallelized over (batch, output feature-slice, output row) - the same granularity the .cl
// kernel dispatches work at (one work-group per (b, feature_block, y-row-block)) - via
// ov::parallel_for, so multiple CPU cores are used the way multiple GPU EUs would be.
template <typename InputT, typename WeightsT, typename OutputT, typename AccT>
void convolve_ref(const InputT* input,
                   const WeightsT* weights,
                   const AccT* bias,
                   OutputT* output,
                   const layout& input_layout,
                   const layout& weights_layout,
                   const layout& output_layout,
                   int64_t groups,
                   const ov::Strides& stride,
                   const ov::Strides& dilation,
                   const ov::CoordinateDiff& pads_begin) {
    const auto in_shape = input_layout.get_shape();    // [B, IFM, Y, X]
    const auto out_shape = output_layout.get_shape();  // [B, OFM, Y, X]
    const auto w_shape = weights_layout.get_shape();   // [OFM, IFM/groups, KY, KX]

    const int64_t batch_num = static_cast<int64_t>(out_shape[0]);
    const int64_t ofm_num = static_cast<int64_t>(out_shape[1]);
    const int64_t out_size_y = static_cast<int64_t>(out_shape[2]);
    const int64_t out_size_x = static_cast<int64_t>(out_shape[3]);
    const int64_t in_size_y = static_cast<int64_t>(in_shape[2]);
    const int64_t in_size_x = static_cast<int64_t>(in_shape[3]);
    const int64_t kernel_size_y = static_cast<int64_t>(w_shape[2]);
    const int64_t kernel_size_x = static_cast<int64_t>(w_shape[3]);

    const int64_t ofm_per_group = ofm_num / groups;
    const int64_t ifm_per_group = static_cast<int64_t>(w_shape[1]);

    const int64_t stride_y = static_cast<int64_t>(stride[0]);
    const int64_t stride_x = static_cast<int64_t>(stride[1]);
    const int64_t dilation_y = static_cast<int64_t>(dilation[0]);
    const int64_t dilation_x = static_cast<int64_t>(dilation[1]);
    const int64_t pad_y = static_cast<int64_t>(pads_begin[0]);
    const int64_t pad_x = static_cast<int64_t>(pads_begin[1]);

    // input/output are required (by the format list this impl is registered with) to be
    // b_fs_yx_fsv16; weights use the default (unblocked) oiyx/goiyx layout.
    const fast_offset in_off(input_layout, /*blocked=*/true);
    const fast_offset out_off(output_layout, /*blocked=*/true);
    const fast_offset w_off(weights_layout, /*blocked=*/false);

    const int64_t num_feature_slices = (ofm_num + feature_slice_size - 1) / feature_slice_size;
    const int64_t total_work_items = batch_num * num_feature_slices * out_size_y;

    // Flatten (b, feature_slice, y) into a single range so ov::parallel_for can distribute rows
    // of independent output work across threads - each unit only ever writes to its own disjoint
    // slice of the output buffer, so no synchronization is required.
    ov::parallel_for(total_work_items, [&](int64_t idx) {
        const int64_t y = idx % out_size_y;
        const int64_t fs_idx = (idx / out_size_y) % num_feature_slices;
        const int64_t b = idx / (out_size_y * num_feature_slices);

        const int64_t fs_begin = fs_idx * feature_slice_size;
        const int64_t fs_end = std::min(fs_begin + feature_slice_size, ofm_num);

        const int64_t iy_base = y * stride_y - pad_y;

        for (int64_t x = 0; x < out_size_x; x++) {
            AccT acc[feature_slice_size] = {};
            for (int64_t of = fs_begin; of < fs_end; of++)
                acc[of - fs_begin] = bias ? bias[of] : AccT(0);

            const int64_t ix_base = x * stride_x - pad_x;

            // Kernel window taps, same nesting as the "for (int kh ...)/for (int kw ...)"
            // loops around the FUNC_CALL(convolve) body in the .cl kernel.
            for (int64_t ky = 0; ky < kernel_size_y; ky++) {
                const int64_t iy = iy_base + ky * dilation_y;
                if (iy < 0 || iy >= in_size_y)
                    continue;
                for (int64_t kx = 0; kx < kernel_size_x; kx++) {
                    const int64_t ix = ix_base + kx * dilation_x;
                    if (ix < 0 || ix >= in_size_x)
                        continue;

                    for (int64_t of = fs_begin; of < fs_end; of++) {
                        // Support grouped convolutions: figure out which group this output
                        // feature belongs to (the .cl kernel derives the same "group"/"my_group"
                        // value from feature_block and sglid under '#if GROUPED').
                        const int64_t group = of / ofm_per_group;
                        const int64_t ifm_begin = group * ifm_per_group;

                        AccT sum = AccT(0);
                        // Input features are consumed in the same 16-wide slices as the kernel's
                        // IN_BUF/input_pack reads, one MAKE_VECTOR_TYPE(..., 16) block at a time.
                        for (int64_t icb = 0; icb < ifm_per_group; icb += feature_slice_size) {
                            const int64_t icb_end = std::min(icb + feature_slice_size, ifm_per_group);
                            for (int64_t iif = icb; iif < icb_end; iif++) {
                                const auto in_offset = in_off.offset(b, ifm_begin + iif, ix, iy);
                                const auto w_offset = w_off.offset(of, iif, kx, ky);
                                sum += static_cast<AccT>(input[in_offset]) * static_cast<AccT>(weights[w_offset]);
                            }
                        }
                        acc[of - fs_begin] += sum;
                    }
                }
            }

            for (int64_t of = fs_begin; of < fs_end; of++) {
                const auto o_offset = out_off.offset(b, of, x, y);
                output[o_offset] = static_cast<OutputT>(acc[of - fs_begin]);
            }
        }
    });
}

// AVX2-vectorized f32 convolution path (implemented in convolution_avx2.cpp, compiled with AVX2
// codegen enabled for that translation unit only). Only called when ov::with_cpu_x86_avx2()
// reports the running CPU actually supports AVX2 - convolve_ref() above remains the always-safe
// fallback.
void convolve_avx2_f32(const float* input,
                        const float* weights,
                        const float* bias,
                        float* output,
                        const layout& input_layout,
                        const layout& weights_layout,
                        const layout& output_layout,
                        int64_t groups,
                        const ov::Strides& stride,
                        const ov::Strides& dilation,
                        const ov::CoordinateDiff& pads_begin);

}  // namespace cpu
}  // namespace cldnn
