// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include <filesystem>
#include <fstream>

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/util/mmap_object.hpp"

#include "remote_tensor_tests/helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

TEST(ZeRemoteContext, smoke_CorrectContextType) {
    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);
}

using MmapFileMemoryParams = std::tuple<std::size_t, std::size_t>;

// Mirrors GpuRemoteTensorFromFile::smoke_mmapFileMemoryAsInput from ocl_remote_tensor_tests.cpp, but exercises
// the Level Zero backend's zero-copy mmap-file import path (ze_engine::create_hostbuffer_impl).
class ZeRemoteTensorFromFile : public ::testing::TestWithParam<MmapFileMemoryParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MmapFileMemoryParams>& obj) {
        const auto& [offset, bytes_after_offset] = obj.param;
        return "offset_" + std::to_string(offset) + "_bytes_after_offset_" + std::to_string(bytes_after_offset);
    }

protected:
    std::filesystem::path m_file_path;

    void SetUp() override {
        m_file_path = ov::test::utils::generateTestFilePrefix() + ".bin";
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(m_file_path, ec);
    }

    static void write_data_at_offset(const std::filesystem::path& path,
                                     std::size_t offset,
                                     const std::vector<float>& values) {
        std::ofstream file(path, std::ios::binary);
        if (offset > 0) {
            const std::vector<char> padding(offset, 0);
            file.write(padding.data(), padding.size());
        }
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }

    static std::vector<float> make_values(std::size_t element_count) {
        std::vector<float> values(element_count);
        for (std::size_t i = 0; i < element_count; ++i) {
            values[i] = static_cast<float>(i + 1);
        }
        return values;
    }
};

TEST_P(ZeRemoteTensorFromFile, smoke_mmapFileMemoryAsInput) {
    const auto& [offset, bytes_after_offset] = GetParam();

    ov::Core core;
    std::string target_device = ov::test::utils::DEVICE_GPU;
    const ov::Shape shape{bytes_after_offset / sizeof(float)};
    const size_t element_count = ov::shape_size(shape);
    auto ctx = core.get_default_context(target_device);
    ASSERT_EQ(ctx.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);

    const auto input_values = make_values(element_count);
    write_data_at_offset(m_file_path, offset, input_values);
    ASSERT_EQ(std::filesystem::file_size(m_file_path), offset + bytes_after_offset);

    const ov::AnyMap params = {
        {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::MMAPED_FILE},
        {ov::intel_gpu::file_descriptor.name(), ov::intel_gpu::FileDescriptor{m_file_path, offset, ov::intel_gpu::FileAccess::READ}}};
    auto remote_input_tensor = ctx.create_tensor(ov::element::f32, shape, params);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_tensor(compiled.input(), remote_input_tensor);
    infer_req.infer();

    auto output_tensor = infer_req.get_output_tensor();
    const auto* output_data = output_tensor.data<float>();
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], input_values[i]) << "Mismatch at index " << i;
    }
}

static std::vector<MmapFileMemoryParams> generate_ze_mmap_file_memory_params() {
#ifdef _WIN32
    // Windows maps file views with 64K allocation granularity
    const std::size_t mmap_granularity = 65536;
#else
    // Page size varies per platform: 4K on x86-64, 16K or 64K on some ARM64 Linux distributions.
    const auto mmap_granularity = static_cast<std::size_t>(ov::util::get_system_page_size());
#endif
    const std::vector<std::pair<std::size_t, std::size_t>> layouts{
        // Sizes smaller than / not divisible by the device cacheline size, e.g. f32 with shape {1}.
        {0, sizeof(float)},
        {mmap_granularity, 3 * sizeof(float)},
        {0, 256},
        {0, 4 * mmap_granularity},
        {mmap_granularity, 256},
        {2 * mmap_granularity, 256},
        {mmap_granularity, mmap_granularity},
        {3 * mmap_granularity, 4 * mmap_granularity}};

    std::vector<MmapFileMemoryParams> params;
    params.reserve(layouts.size());
    for (const auto& [offset, bytes_after_offset] : layouts) {
        params.emplace_back(offset, bytes_after_offset);
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_mmapFileMemory,
                         ZeRemoteTensorFromFile,
                         ::testing::ValuesIn(generate_ze_mmap_file_memory_params()),
                         ZeRemoteTensorFromFile::getTestCaseName);

#endif  // OV_GPU_WITH_ZE_RT
