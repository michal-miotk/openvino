// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gru_seq_inst.h"
#include "gru_seq.hpp"
#include "lstm/gru_seq_kernel_selector.h"
#include "lstm/gru_kernel_params.h"
#include "lstm/lstm_kernel_base.h"
#include "openvino/op/gru_sequence.hpp"
#include "impls/registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

struct gru_seq_impl : typed_primitive_impl_ocl<gru_seq> {
    using parent = typed_primitive_impl_ocl<gru_seq>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gru_seq_kernel_selector;
    using kernel_params_t = kernel_selector::gru_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gru_seq_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gru_seq_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<gru_seq>& instance) const override {
        kernel_arguments_data args;
        size_t op_input_size = 6;
        for (size_t i = 0; i < op_input_size; i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }
        for (size_t i = op_input_size; i < instance.inputs_memory_count(); i++) {
            args.outputs.push_back(instance.dep_memory_ptr(i));
        }
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<gru_seq>();
        auto params = get_default_params<kernel_selector::gru_params>(impl_param);
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        if (!primitive->activations.empty()) {
            auto a_sz = primitive->activations.size();
            auto param_sz = primitive->activation_params.size();
            OPENVINO_ASSERT(param_sz == 0|| a_sz == param_sz, "[GPU] Unexpected activation params count in gru_seq impl: ", param_sz);
            for (size_t i = 0; i < a_sz; i++) {
                params.activations.emplace_back(get_kernel_selector_activation_param(primitive->activations[i]),
                                                         param_sz ? primitive->activation_params[i].a : 0.0f,
                                                         param_sz ? primitive->activation_params[i].b : 0.0f);
            }
        }

        if (primitive->clip > 0.0f) {
            params.activations.emplace_back(get_kernel_selector_activation_param(activation_func::clamp), -primitive->clip, primitive->clip);
        }

        params.SetOffsetOrder(static_cast<int32_t>(primitive->offset_order));
        params.clip = primitive->clip;
        params.direction = primitive->direction;
        //Legacy multi-output
        params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));

        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        if (impl_params.get_input_layout().get_partial_shape().size() != 3) {
            return primitive_impl::static_canonicalize_shapes(impl_params);
        }
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }
};

std::unique_ptr<primitive_impl> GRUSeqImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<gru_seq>());
    return typed_primitive_impl_ocl<gru_seq>::create<gru_seq_impl>(static_cast<const gru_seq_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gru_seq_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gru_seq)
