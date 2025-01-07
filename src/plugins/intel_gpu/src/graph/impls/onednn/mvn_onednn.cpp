// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_onednn_base.h"
#include "mvn_onednn.hpp"
#include "impls/registry/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct mvn_onednn : typed_primitive_onednn_impl<mvn> {
    using parent = typed_primitive_onednn_impl<mvn>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::mvn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mvn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(mvn_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        auto& input = instance.input_memory(0);
        auto offset = onednn::get_offset(instance.get_input_layout(0), _pd.dnnl::primitive_desc_base::src_desc(0));
        auto in_mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0), offset);
        args.insert({DNNL_ARG_SRC_LAYER, in_mem});

        auto& output = instance.output_memory(0);
        offset = onednn::get_offset(instance.get_output_layout(0), _pd.dnnl::primitive_desc_base::dst_desc(0));
        auto out_mem = output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset);
        args.insert({DNNL_ARG_DST_LAYER, out_mem});

        return args;
    }

    static cldnn::layout get_reorder_layout(const kernel_impl_params& impl_params, size_t layout_nr) {
        auto weights_shape = impl_params.get_input_layout(layout_nr).get_shape();
        auto target_weights_layout = impl_params.get_input_layout(layout_nr);
        target_weights_layout.format = cldnn::format::bfzyx;
        return target_weights_layout;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        const auto weights_layout_idx = 0;
        auto source_weights_layout = impl_params.get_input_layout(weights_layout_idx);
        auto target_weights_layout = get_reorder_layout(impl_params, weights_layout_idx);
        auto W_desc = onednn::layout_to_memory_desc(source_weights_layout);
        auto grouped_weights = format::is_grouped(source_weights_layout.format);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            W_desc,
                                                            W_desc,
                                                            false,
                                                            grouped_weights);
    }
    static std::shared_ptr<dnnl::layer_normalization_forward::primitive_desc> get_mvn_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                                           cldnn::engine& engine,
                                                                                                           const dnnl::primitive_attr& attr) {
        auto prim = impl_params.typed_desc<mvn>();
        auto input_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(0), dnnl::memory::format_tag::abc);
        auto output_md = onednn::layout_to_memory_desc(impl_params.get_output_layout(), dnnl::memory::format_tag::abc);

        auto eng = engine.get_onednn_engine();

        return std::make_shared<dnnl::layer_normalization_forward::primitive_desc>(
            eng,
            dnnl::prop_kind::forward_inference,
            input_md,
            output_md,
            prim->epsilon,
            dnnl::normalization_flags::none);
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
        auto prim_desc = std::make_shared<dnnl::layer_normalization_forward::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            input_md,
            output_md,
            0.00001f,
            dnnl::normalization_flags::none);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;
        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const mvn_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_mvn_primitive_descriptor(impl_params, engine, *attr);
        return cldnn::make_unique<mvn_onednn>(engine, config, attr, *prim_desc, get_weights_reorder(impl_params, *prim_desc));
    }
};

std::unique_ptr<primitive_impl> MVNImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const  {
    assert(node.is_type<mvn>());
    return onednn::mvn_onednn::create(static_cast<const mvn_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::mvn_onednn)
