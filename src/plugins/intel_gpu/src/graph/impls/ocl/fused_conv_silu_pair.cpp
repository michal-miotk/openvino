// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_conv_silu_pair_inst.h"
#include "fused_conv_silu_pair/fused_conv_silu_pair_kernel_base.h"
#include "fused_conv_silu_pair/fused_conv_silu_pair_kernel_selector.h"
#include "primitive_base.hpp"

#include <algorithm>
#include <cstddef>

namespace cldnn {
namespace ocl {

struct fused_conv_silu_pair_impl : typed_primitive_impl_ocl<fused_conv_silu_pair> {
    using parent = typed_primitive_impl_ocl<fused_conv_silu_pair>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::fused_conv_silu_pair_kernel_selector;
    using kernel_params_t = kernel_selector::fused_conv_silu_pair_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fused_conv_silu_pair_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<fused_conv_silu_pair_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<fused_conv_silu_pair>();

        // get_default_params() wypelnia inputs[0] i outputs[0]; pozostale cztery
        // wejscia (wagi/biasy obu konwolucji) dokladamy recznie, bo dla tego
        // prymitywu sa zwyklymi buforami, a nie tensorami `weights`/`bias`.
        auto params = get_default_params<kernel_selector::fused_conv_silu_pair_params>(impl_param, is_shape_agnostic);
        for (size_t i = 1; i < impl_param.input_layouts.size(); i++)
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));

        params.filterSize1 = {static_cast<uint32_t>(primitive->kernel1[1]),
                              static_cast<uint32_t>(primitive->kernel1[0]),
                              1};
        params.stride = {static_cast<uint32_t>(primitive->strides[1]),
                         static_cast<uint32_t>(primitive->strides[0]),
                         1};
        params.dilation = {static_cast<uint32_t>(primitive->dilations[1]),
                           static_cast<uint32_t>(primitive->dilations[0]),
                           1};
        params.padding = {static_cast<uint32_t>(std::max<std::ptrdiff_t>(primitive->pads_begin[1], 0)),
                          static_cast<uint32_t>(std::max<std::ptrdiff_t>(primitive->pads_begin[0], 0)),
                          0};

        params.in_features = primitive->in_channels;
        params.mid_features = primitive->mid_channels;
        params.out_features = primitive->out_channels;
        params.swish_beta1 = primitive->beta1;
        params.swish_beta2 = primitive->beta2;

        return params;
    }
};

namespace detail {

attach_fused_conv_silu_pair_impl::attach_fused_conv_silu_pair_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {format::b_fs_yx_fsv16};

    implementation_map<fused_conv_silu_pair>::add(
        impl_types::ocl,
        shape_types::static_shape,
        typed_primitive_impl_ocl<fused_conv_silu_pair>::create<fused_conv_silu_pair_impl>,
        types,
        formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::fused_conv_silu_pair_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::fused_conv_silu_pair)
