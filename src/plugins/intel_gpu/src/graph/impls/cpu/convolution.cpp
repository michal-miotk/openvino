// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "impls/cpu/convolution_common.hpp"
#include "register.hpp"
#include "convolution_inst.h"
#include "registry/implementation_map.hpp"

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/system_conf.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

namespace cldnn {
namespace cpu {


struct convolution_impl : public typed_primitive_impl<convolution> {
    using parent = typed_primitive_impl<convolution>;
    using parent::parent;

    uint32_t groups = 1;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff padding_begin;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::convolution_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<convolution_impl>(*this);
    }

    convolution_impl() : parent("convolution_cpu_impl") {}

    explicit convolution_impl(const convolution_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<convolution>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<convolution>();
        const auto prim = node.get_primitive();

        OPENVINO_ASSERT(!node.get_deformable_mode(), "[GPU] cpu convolution impl does not support deformable convolutions");
        OPENVINO_ASSERT(!prim->transposed, "[GPU] cpu convolution impl does not support transposed convolutions");
        OPENVINO_ASSERT(!node.weights_zero_points_term() && !node.activations_zero_points_term() && !node.compensation_term(),
                         "[GPU] cpu convolution impl does not support quantized (zero-point) convolutions");
        OPENVINO_ASSERT(node.get_input_layout(0).format == format::b_fs_yx_fsv16,
                         "[GPU] cpu convolution impl only supports b_fs_yx_fsv16 input0 format, got: ",
                         node.get_input_layout(0).format.to_string());

        groups = node.get_groups();
        stride = prim->stride;
        dilation = prim->dilation;
        padding_begin = prim->padding_begin;
        // Make sure spatial rank always covers Y and X, even if the op reported a lower rank.
        if (stride.size() < 2)
            stride.resize(2, 1);
        if (dilation.size() < 2)
            dilation.resize(2, 1);
        if (padding_begin.size() < 2)
            padding_begin.resize(2, 0);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << groups;
        ob << stride;
        ob << dilation;
        ob << padding_begin;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> groups;
        ib >> stride;
        ib >> dilation;
        ib >> padding_begin;
    }

    template <typename InputT, typename WeightsT, typename OutputT>
    void execute_typed(convolution_inst& instance) const {
        auto& stream = instance.get_network().get_stream();
        const auto* params = instance.get_impl_params();

        auto input_mem = instance.input_memory_ptr(0);
        auto weights_mem = instance.weights_memory();
        auto output_mem = instance.output_memory_ptr();

        cldnn::mem_lock<InputT, mem_lock_type::read> input_lock(input_mem, stream);
        cldnn::mem_lock<WeightsT, mem_lock_type::read> weights_lock(weights_mem, stream);
        cldnn::mem_lock<OutputT, mem_lock_type::write> output_lock(output_mem, stream);

        // Bias is accumulated using the same precision as the OpenCL kernel's ACCUMULATOR_TYPE:
        // fp32 for everything but pure fp16 without bias upconversion needs, so we just keep it
        // simple and always accumulate in float.
        std::vector<float> bias_storage;
        const float* bias_ptr = nullptr;
        if (instance.bias_term()) {
            auto bias_mem = instance.bias_memory();
            cldnn::mem_lock<uint8_t, mem_lock_type::read> bias_lock(bias_mem, stream);
            const auto bias_dt = params->bias_layout.value().data_type;
            const auto bias_count = bias_mem->count();
            bias_storage.resize(bias_count);
            if (bias_dt == data_types::f32) {
                const auto* typed = reinterpret_cast<const float*>(bias_lock.data());
                std::copy_n(typed, bias_count, bias_storage.begin());
            } else if (bias_dt == data_types::f16) {
                const auto* typed = reinterpret_cast<const ov::float16*>(bias_lock.data());
                for (size_t i = 0; i < bias_count; i++)
                    bias_storage[i] = static_cast<float>(typed[i]);
            } else {
                OPENVINO_THROW("[GPU] cpu convolution impl: unsupported bias data type");
            }
            bias_ptr = bias_storage.data();
        }

        // For plain f32 on an AVX2-capable host, use the vectorized path (2x8-wide FMA per
        // 16-feature slice); otherwise fall back to the portable scalar reference loop. The
        // AVX2 availability check is a runtime one (ov::with_cpu_x86_avx2()), not a compile-time
        // one, so a binary built on an AVX2 machine still runs correctly (just without the
        // vectorized fast path) on an older host.
        if constexpr (std::is_same<InputT, float>::value && std::is_same<WeightsT, float>::value &&
                      std::is_same<OutputT, float>::value) {
            if (ov::with_cpu_x86_avx2()) {
                convolve_avx2_f32(reinterpret_cast<const float*>(input_lock.data()),
                                   reinterpret_cast<const float*>(weights_lock.data()),
                                   bias_ptr,
                                   reinterpret_cast<float*>(output_lock.data()),
                                   params->input_layouts[0],
                                   params->weights_layout.value(),
                                   params->output_layouts[0],
                                   static_cast<int64_t>(groups),
                                   stride,
                                   dilation,
                                   padding_begin);
                return;
            }
        }

        convolve_ref<InputT, WeightsT, OutputT, float>(input_lock.data(),
                                                        weights_lock.data(),
                                                        bias_ptr,
                                                        output_lock.data(),
                                                        params->input_layouts[0],
                                                        params->weights_layout.value(),
                                                        params->output_layouts[0],
                                                        static_cast<int64_t>(groups),
                                                        stride,
                                                        dilation,
                                                        padding_begin);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, convolution_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "convolution::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();
        if (!pass_through_events)
            stream.wait_for_events(events);

        const auto* params = instance.get_impl_params();
        const auto in_dt = params->input_layouts[0].data_type;
        const auto w_dt = params->weights_layout.value().data_type;
        const auto out_dt = params->output_layouts[0].data_type;

        OPENVINO_ASSERT(params->input_layouts[0].format == format::b_fs_yx_fsv16,
                         "[GPU] cpu convolution impl only supports b_fs_yx_fsv16 input0 format, got: ",
                         params->input_layouts[0].format.to_string());
        OPENVINO_ASSERT(in_dt == w_dt && w_dt == out_dt,
                         "[GPU] cpu convolution impl requires input, weights and output to share the same data type");

        switch (in_dt) {
        case data_types::f32:
            execute_typed<float, float, float>(instance);
            break;
        case data_types::f16:
            execute_typed<ov::float16, ov::float16, ov::float16>(instance);
            break;
        default:
            OPENVINO_THROW("[GPU] cpu convolution impl: unsupported data type");
        }

        if (pass_through_events)
            return stream.group_events(events);

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const convolution_node& arg, const kernel_impl_params& impl_param) {
        OPENVINO_ASSERT(impl_param.fused_desc.empty(), "[GPU] cpu convolution impl does not support fused operations");
        OPENVINO_ASSERT(impl_param.input_layouts[0].format == format::b_fs_yx_fsv16,
                         "[GPU] cpu convolution impl only supports b_fs_yx_fsv16 input0 format, got: ",
                         impl_param.input_layouts[0].format.to_string());
        return std::make_unique<convolution_impl>(arg);
    }
};

namespace detail {

attach_convolution_impl::attach_convolution_impl() {
    auto formats = {
        format::b_fs_yx_fsv16,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
    };

    implementation_map<convolution>::add(impl_types::cpu, shape_types::static_shape, convolution_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::convolution_impl)
