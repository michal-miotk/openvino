# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_cpu_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    LINK_LIBRARIES openvino::reference
)

if(ENABLE_SSE42)
  ov_sse42_optimization_flags(sse4_2_flags)
  set_source_files_properties(detection_output.cpp PROPERTIES
    COMPILE_FLAGS "${sse4_2_flags}"
    COMPILE_DEFINITIONS "HAVE_SSE")
endif()

if(ENABLE_AVX2)
  ov_avx2_optimization_flags(avx2_flags)
  string(REPLACE ";" " " avx2_flags_str "${avx2_flags}")
  set_source_files_properties(convolution_avx2.cpp PROPERTIES
    COMPILE_FLAGS "${avx2_flags_str}"
    COMPILE_DEFINITIONS "HAVE_AVX2")
endif()

ov_build_target_faster(${TARGET_NAME} PCH PCH_EXCLUDE detection_output.cpp convolution_avx2.cpp)
