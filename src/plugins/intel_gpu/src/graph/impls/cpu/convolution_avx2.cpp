// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// AVX2-vectorized f32 convolution path. This translation unit is compiled with AVX2 code
// generation enabled (see CMakeLists.txt), independent of the optimization level used for the
// rest of the plugin (the unit tests in this session run Debug builds, where the compiler does
// not auto-vectorize anything - so any real SIMD usage here has to come from explicit
// intrinsics rather than from "vectorizer-friendly" loop shapes).
//
// The real convolution_gpu_bfyx_f16.cl kernel keeps weights in a *blocked* layout
// (os_is_yx_isv16_osv16 / g_os_is_yx_isv16_osv16, see ConvolutionKernel_b_fs_yx_fsv16::
// GetPreferredWeightsLayout()) so that all 16 output-feature weights needed for one
// sub-group step are contiguous in memory and can be read with a single vectorized
// DT_FILTER_BLOCK_READ8 (+ DT_FILTER_BLOCK_READ8) pair. This CPU implementation keeps weights in
// the plain (unblocked) oiyx/goiyx layout it is actually handed by the graph, so instead of a
// contiguous vector load we gather the 16 per-slice weights into a small local buffer once per
// (kx, ky, iif) tap and then run the actual multiply-accumulate over all 16 output features as
// two 8-wide AVX2 FMA vector ops, mirroring the kernel's "16 output features computed per
// sub-group / per SIMD lane" structure even though the memory layout differs.
#include "convolution_common.hpp"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

namespace cldnn {
namespace cpu {

#ifdef HAVE_AVX2

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
                        const ov::CoordinateDiff& pads_begin) {
    const auto in_shape = input_layout.get_shape();
    const auto out_shape = output_layout.get_shape();
    const auto w_shape = weights_layout.get_shape();

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

    const fast_offset in_off(input_layout, /*blocked=*/true);
    const fast_offset out_off(output_layout, /*blocked=*/true);
    const fast_offset w_off(weights_layout, /*blocked=*/false);
    // Weights for 16 consecutive output features at a fixed (iif, kx, ky) are `pitch_of` floats
    // apart (equal to w_off's per-"batch"-like axis here is actually the OFM axis stride, i.e.
    // the address delta between of and of+1). We reuse fast_offset's own of-axis pitch (its
    // "pitch_b_", since weights are addressed as (of, iif, ky, kx) through the same 4-arg
    // offset() call, with `of` passed positionally where fast_offset expects `b`).
    const int64_t pitch_of = w_off.offset(1, 0, 0, 0) - w_off.offset(0, 0, 0, 0);

    const int64_t num_feature_slices = (ofm_num + feature_slice_size - 1) / feature_slice_size;
    const int64_t total_work_items = batch_num * num_feature_slices * out_size_y;

    ov::parallel_for(total_work_items, [&](int64_t idx) {
        const int64_t y = idx % out_size_y;
        const int64_t fs_idx = (idx / out_size_y) % num_feature_slices;
        const int64_t b = idx / (out_size_y * num_feature_slices);

        const int64_t fs_begin = fs_idx * feature_slice_size;
        const int64_t fs_end = std::min(fs_begin + feature_slice_size, ofm_num);
        const int64_t fs_width = fs_end - fs_begin;
        const bool full_slice = (fs_width == feature_slice_size);

        const int64_t iy_base = y * stride_y - pad_y;

        alignas(32) float wbuf[feature_slice_size];

        for (int64_t x = 0; x < out_size_x; x++) {
            alignas(32) float acc[feature_slice_size] = {};
            for (int64_t of = fs_begin; of < fs_end; of++)
                acc[of - fs_begin] = bias ? bias[of] : 0.0f;

            const int64_t ix_base = x * stride_x - pad_x;

            for (int64_t ky = 0; ky < kernel_size_y; ky++) {
                const int64_t iy = iy_base + ky * dilation_y;
                if (iy < 0 || iy >= in_size_y)
                    continue;
                for (int64_t kx = 0; kx < kernel_size_x; kx++) {
                    const int64_t ix = ix_base + kx * dilation_x;
                    if (ix < 0 || ix >= in_size_x)
                        continue;

                    // Grouped convolutions still need the group derived per output feature; when
                    // groups > 1 and a feature slice straddles a group boundary we simply fall
                    // back to scalar accumulation for this tap (rare/degenerate case - keeps the
                    // fast vector path simple for the common groups==1 / group-aligned-slices
                    // case, which is what convolution_gpu_bfyx_f16.cl itself targets).
                    const int64_t group_begin = fs_begin / ofm_per_group;
                    const int64_t group_end = (fs_end - 1) / ofm_per_group;
                    const bool single_group = (group_begin == group_end);

                    if (full_slice && single_group) {
                        const int64_t ifm_begin = group_begin * ifm_per_group;
                        __m256 acc_lo = _mm256_loadu_ps(&acc[0]);
                        __m256 acc_hi = _mm256_loadu_ps(&acc[8]);

                        for (int64_t iif = 0; iif < ifm_per_group; iif++) {
                            const auto in_offset = in_off.offset(b, ifm_begin + iif, ix, iy);
                            const __m256 in_bcast = _mm256_set1_ps(input[in_offset]);

                            // Gather the 16 per-output-feature weights for this (iif, kx, ky)
                            // tap. Weights are not contiguous across `of` in the plain oiyx
                            // layout (unlike the blocked os_is_yx_isv16_osv16 layout the real
                            // kernel expects), so this gather-then-vectorize approach is used
                            // instead of a single block read.
                            const int64_t w_base = w_off.offset(fs_begin, iif, kx, ky);
                            for (int64_t k = 0; k < feature_slice_size; k++)
                                wbuf[k] = weights[w_base + k * pitch_of];

                            const __m256 w_lo = _mm256_load_ps(&wbuf[0]);
                            const __m256 w_hi = _mm256_load_ps(&wbuf[8]);
#if defined(__FMA__)
                            acc_lo = _mm256_fmadd_ps(in_bcast, w_lo, acc_lo);
                            acc_hi = _mm256_fmadd_ps(in_bcast, w_hi, acc_hi);
#else
                            acc_lo = _mm256_add_ps(acc_lo, _mm256_mul_ps(in_bcast, w_lo));
                            acc_hi = _mm256_add_ps(acc_hi, _mm256_mul_ps(in_bcast, w_hi));
#endif
                        }

                        _mm256_storeu_ps(&acc[0], acc_lo);
                        _mm256_storeu_ps(&acc[8], acc_hi);
                    } else {
                        // Partial slice (tail) or group-straddling slice: plain scalar fallback.
                        for (int64_t of = fs_begin; of < fs_end; of++) {
                            const int64_t group = of / ofm_per_group;
                            const int64_t ifm_begin = group * ifm_per_group;
                            float sum = 0.0f;
                            for (int64_t iif = 0; iif < ifm_per_group; iif++) {
                                const auto in_offset = in_off.offset(b, ifm_begin + iif, ix, iy);
                                const auto w_offset = w_off.offset(of, iif, kx, ky);
                                sum += input[in_offset] * weights[w_offset];
                            }
                            acc[of - fs_begin] += sum;
                        }
                    }
                }
            }

            for (int64_t of = fs_begin; of < fs_end; of++) {
                const auto o_offset = out_off.offset(b, of, x, y);
                output[o_offset] = acc[of - fs_begin];
            }
        }
    });
}

#else  // !HAVE_AVX2

// This translation unit wasn't compiled with AVX2 code generation enabled (ENABLE_AVX2=OFF, or
// target arch without AVX2 support) - fall back to the portable scalar reference path so
// convolve_avx2_f32() remains callable unconditionally from convolution.cpp; the runtime
// ov::with_cpu_x86_avx2() check on the caller side is what normally prevents this function from
// being invoked at all in that configuration, but keeping a working fallback here avoids ever
// needing an `#ifdef HAVE_AVX2` at the call site too.
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
                        const ov::CoordinateDiff& pads_begin) {
    convolve_ref<float, float, float, float>(input,
                                              weights,
                                              bias,
                                              output,
                                              input_layout,
                                              weights_layout,
                                              output_layout,
                                              groups,
                                              stride,
                                              dilation,
                                              pads_begin);
}

#endif  // HAVE_AVX2

}  // namespace cpu
}  // namespace cldnn
