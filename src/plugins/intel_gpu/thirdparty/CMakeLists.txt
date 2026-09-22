# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(XETLA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/xetla/include/" CACHE PATH "Path to XeTLA headers")

#
# oneDNN for GPU plugin
#

if(ENABLE_ONEDNN_FOR_GPU)
    # Cached because the installed OpenVINOConfig.cmake resolves this name.
    set(DNNL_GPU_LIBRARY_NAME "openvino_onednn_gpu" CACHE STRING "Name of oneDNN library for Intel GPU Plugin")

    if(GPU_RT_TYPE STREQUAL "COMBINED")
        message(FATAL_ERROR "GPU_RT_TYPE=COMBINED is not supported: oneDNN for GPU is built as a part of the "
                            "OpenVINO build tree and a single build tree cannot host two oneDNN configurations. "
                            "Use GPU_RT_TYPE=OCL, ZE or SYCL.")
    endif()

    function(ov_add_onednn_gpu)
        set(ONEDNN_GPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onednn_gpu")
        set(ONEDNN_GPU_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu")
        # Configuration independent location of the produced archive, so that ONEDNN_GPU_LIB_PATH
        # published below is a plain path which OpenVINOConfig.cmake can consume directly.
        set(ONEDNN_GPU_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_lib")

        if(X86_64)
            set(DNNL_TARGET_ARCH "X64")
        elseif(X86)
            set(DNNL_TARGET_ARCH "X86")
        elseif(RISCV64)
            set(DNNL_TARGET_ARCH "RV64")
        elseif(ARM)
            set(DNNL_TARGET_ARCH "ARM")
        elseif(AARCH64)
            set(DNNL_TARGET_ARCH "AARCH64")
        else()
            message(WARNING "Intel GPU plugin unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
        endif()

        if(GPU_RT_TYPE STREQUAL "ZE")
            # oneDNN is using different version of L0 headers.
            # Disable LTO to avoid build issues.
            # Enable once headers are unified MFDNN-14973
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
        else()
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
        endif()

        # oneDNN specific compile flags
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
            ov_add_compiler_flags(-Wno-undef)
            ov_add_compiler_flags(-Wno-missing-declarations)
            ov_add_compiler_flags(-Wno-ignored-attributes)
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
                ov_add_compiler_flags(-Wno-array-bounds)
                ov_add_compiler_flags(-Wno-stringop-overflow)
                if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
                    ov_add_compiler_flags(-Wno-restrict)
                endif()
            endif()
        endif()

        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
        endif()

        # remove CMAKE_COMPILE_WARNING_AS_ERROR for onednn_gpu
        if(WIN32 AND CMAKE_COMPILE_WARNING_AS_ERROR AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_VERSION VERSION_LESS 3.24)
            ov_add_compiler_flags(/WX-)
        endif()
        set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

        if(NOT BUILD_SHARED_LIBS)
            if(MSVC)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNGEN_LINK_OPENCL")
            else()
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGEN_LINK_OPENCL")
            endif()
        endif()

        # oneDNN picks the OpenCL version it needs on its own, drop the plugin-wide one
        # to avoid a macro redefinition
        remove_definitions(-DCL_TARGET_OPENCL_VERSION=${INTEL_GPU_TARGET_OCL_VERSION})

        # oneDNN configuration. ONEDNN_* are the canonical option names and the DNNL_* aliases are
        # derived from them by oneDNN itself, so the values have to be set with the ONEDNN_ prefix.
        set(ONEDNN_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
        set(ONEDNN_LIBRARY_NAME "${DNNL_GPU_LIBRARY_NAME}" CACHE STRING "" FORCE)
        set(ONEDNN_CPU_RUNTIME "NONE" CACHE STRING "" FORCE)
        set(ONEDNN_GPU_RUNTIME "${GPU_RT_TYPE}" CACHE STRING "" FORCE)
        set(ONEDNN_BLAS_VENDOR "NONE" CACHE STRING "" FORCE)
        set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
        set(ONEDNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(ONEDNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(ONEDNN_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
        set(ONEDNN_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE)
        set(ONEDNN_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)
        set(ONEDNN_ENABLE_JIT_PROFILING ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
        set(ONEDNN_ENABLE_ITT_TASKS ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
        set(ONEDNN_EXPERIMENTAL_PROFILING ON CACHE BOOL "" FORCE)
        set(ONEDNN_EXPERIMENTAL_GROUPED_MEMORY ON CACHE BOOL "" FORCE)
        set(ONEDNN_ENABLE_PRIMITIVE "CONCAT;CONVOLUTION;DECONVOLUTION;GATED_MLP;MATMUL;REORDER;POOLING;REDUCTION;RNN"
            CACHE STRING "" FORCE)
        set(ONEDNN_ENABLE_PRIMITIVE_GPU_ISA "XELP;XEHP;XEHPG;XE2;XE3;XE3P" CACHE STRING "" FORCE)

        add_subdirectory("${ONEDNN_GPU_SOURCE_DIR}" "${ONEDNN_GPU_BINARY_DIR}" EXCLUDE_FROM_ALL)

        # Keep the archive out of the per-configuration subdirectories and without the debug
        # postfix, so that its location is known at configuration time (see ONEDNN_GPU_LIB_PATH
        # below), which is what OpenVINOConfig.cmake resolves for a static build.
        set_target_properties(dnnl PROPERTIES
            DEBUG_POSTFIX ""
            ARCHIVE_OUTPUT_DIRECTORY "${ONEDNN_GPU_LIB_DIR}")
        foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
            string(TOUPPER "${config}" config_upper)
            set_target_properties(dnnl PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${ONEDNN_GPU_LIB_DIR}")
        endforeach()

        # ONEDNN_GPU_LIB_PATH escapes to src/cmake/openvino.cmake (OpenVINOConfig PATH_VARS),
        # so it must be cached.
        set(ONEDNN_GPU_LIB_PATH
            "${ONEDNN_GPU_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${DNNL_GPU_LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}"
            CACHE FILEPATH "Path to oneDNN GPU library" FORCE)

        set(LIB_INCLUDE_DIRS "${ONEDNN_GPU_SOURCE_DIR}/include"
                             "${ONEDNN_GPU_BINARY_DIR}/include"
                             "${ONEDNN_GPU_SOURCE_DIR}/src"
                             "${ONEDNN_GPU_SOURCE_DIR}/src/gpu/intel/jit/ngen"
                             "${ONEDNN_GPU_SOURCE_DIR}/src/gpu/intel/jit/config"
                             "${ONEDNN_GPU_SOURCE_DIR}/src/gpu/intel/gemm/jit/include"
                             "${ONEDNN_GPU_SOURCE_DIR}/third_party/ngen")
        set(LIB_DEFINITIONS ENABLE_ONEDNN_FOR_GPU
                            DNNL_DLL
                            DNNL_DLL_EXPORTS
                            DNNL_ENABLE_CPU_ISA_HINTS
                            DNNL_ENABLE_MAX_CPU_ISA
                            DNNL_X64=1
                            NGEN_CONFIG)

        if(NOT BUILD_SHARED_LIBS)
            list(APPEND LIB_DEFINITIONS NGEN_LINK_OPENCL)
        endif()

        add_library(onednn_gpu_tgt INTERFACE)
        set_target_properties(onednn_gpu_tgt PROPERTIES
            INTERFACE_LINK_LIBRARIES $<BUILD_INTERFACE:dnnl>
            INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${LIB_INCLUDE_DIRS}>"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${LIB_INCLUDE_DIRS}>"
            INTERFACE_COMPILE_DEFINITIONS "${LIB_DEFINITIONS}"
        )

        if(NOT BUILD_SHARED_LIBS)
            # 'dnnl' is installed explicitly, because set_target_properties in OpenVINOConfig.cmake
            # makes 'onednn_gpu_tgt' point to the installation location of this file
            ov_install_static_lib(dnnl ${OV_CPACK_COMP_CORE})
            ov_install_static_lib(onednn_gpu_tgt ${OV_CPACK_COMP_CORE})
        endif()
    endfunction()

    ov_add_onednn_gpu()
endif()
