// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/onednn/utils.hpp"
#include "lstm_cell_inst.h"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct lstm_onednn : typed_primitive_onednn_impl<lstm_cell, dnnl::lstm_forward::primitive_desc, dnnl::lstm_forward> {
    using parent = typed_primitive_onednn_impl<lstm_cell, dnnl::lstm_forward::primitive_desc, dnnl::lstm_forward>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::reorder_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(lstm_cell_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        int input_idx = DNNL_ARG_FROM;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)));
            args.insert({input_idx++, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        return args;
    }

    static std::shared_ptr<dnnl::lstm_forward::primitive_desc> get_lstm_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                           const dnnl::primitive_attr& attr) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<reorder>();

        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.get_output_layout();

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        OPENVINO_ASSERT(input_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the input memory descriptor of onednn reorder cannot be 'any'.");
        OPENVINO_ASSERT(output_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the output memory descriptor of onednn reorder cannot be 'any'.");

        return std::make_shared<dnnl::lstm_forward::primitive_desc>(
            engine.get_onednn_engine(),
            input_md,
            engine.get_onednn_engine(),
            output_md,
            attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0));
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());

        auto prim_desc = std::make_shared<dnnl::lstm_forward::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            input_md,
            ib.get_engine().get_onednn_engine(),
            output_md,
            *_attrs.get());
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();
        if (prim_cache.size() > 0)
            _prim = dnnl::lstm_forward(_pd, prim_cache);
        else
            _prim = dnnl::lstm_forward(_pd);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const lstm_cell_node& arg, const kernel_impl_params& impl_params) {
            auto& engine = impl_params.prog->get_engine();
            auto& config = impl_params.prog->get_config();
            auto attr = impl_params.attrs_onednn;
            auto prim_desc = get_lstm_primitive_descriptor(impl_params, *attr);
            return cldnn::make_unique<lstm_onednn>(engine, config, attr, *prim_desc);
    }
};

namespace detail {

attach_lstm_onednn::attach_lstm_onednn() {
    implementation_map<lstm_cell>::add(impl_types::onednn, lstm_onednn::create, {});
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::lstm_onednn)
