// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/onednn/utils.hpp"
#include "lstm_seq_inst.h"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct lstm_seq_onednn : typed_primitive_onednn_impl<lstm_seq, dnnl::lstm_forward::primitive_desc, dnnl::lstm_forward> {
    using parent = typed_primitive_onednn_impl<lstm_seq, dnnl::lstm_forward::primitive_desc, dnnl::lstm_forward>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::lstm_seq_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_seq_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(lstm_seq_inst& instance) const override {
        std::cout << "getting args" << std::endl;
        std::unordered_map<int, dnnl::memory> args;

        int input_idx = DNNL_ARG_FROM;
        std::cout << instance.inputs_memory_count() << "instance.inputs_memory_count() ff" << std::endl;
        for (size_t i = 0; i < instance.inputs_memory_count()-3; i++) {
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
        {
            auto& output = instance.input_memory(7);
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST+1, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }
        {
            auto& output = instance.input_memory(8);
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST+2, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }
        return args;
    }

    static std::shared_ptr<dnnl::lstm_forward::primitive_desc> get_lstm_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                           const dnnl::primitive_attr& attr) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<lstm_seq>();

        auto input_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(0));
        auto initial_hidden_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(1));
        auto initial_cell_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2));
        auto W_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(3), dnnl::memory::format_tag::any);
        auto R_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(4), dnnl::memory::format_tag::any);
        auto B_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(5));
        auto output_md = onednn::layout_to_memory_desc(impl_params.get_output_layout(), dnnl::memory::format_tag::undef);
        auto output1_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(7));
        auto output2_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(8));

        OPENVINO_ASSERT(input_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the input memory descriptor of onednn lstm_seq cannot be 'any'.");
        OPENVINO_ASSERT(output_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the output memory descriptor of onednn lstm_seq cannot be 'any'.");

        ExecutionConfig config;
        dnnl::memory::desc emptyMemDescriptorForPeephole;
        //engine.create_onednn_engine(config);
        return std::make_shared<dnnl::lstm_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::rnn_direction::bidirectional_concat,
            input_md,
            emptyMemDescriptorForPeephole,
            emptyMemDescriptorForPeephole,
            W_md,
            R_md,
            B_md,
            output_md,
            initial_cell_md,
            initial_cell_md,
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
        auto initial_hidden_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(1));
        auto initial_cell_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(2));
        auto W_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(3));
        auto R_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(4));
        auto B_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(5));
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());
        auto output2_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());
        std::cout << "LOAD IB" << std::endl;
        auto prim_desc = std::make_shared<dnnl::lstm_forward::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::rnn_direction::unidirectional_left2right,
            input_md,
            initial_hidden_md,
            initial_cell_md,
            W_md,
            R_md,
            W_md,
            W_md,
            B_md,
            output_md,
            output_md,
            output2_md,
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

    static std::unique_ptr<primitive_impl> create(const lstm_seq_node& arg, const kernel_impl_params& impl_params) {
            auto& engine = impl_params.prog->get_engine();
            auto& config = impl_params.prog->get_config();
            auto attr = impl_params.attrs_onednn;
            auto prim_desc = get_lstm_primitive_descriptor(impl_params, *attr);
            return cldnn::make_unique<lstm_seq_onednn>(engine, config, attr, *prim_desc);
    }
};

namespace detail {

attach_lstm_seq_onednn::attach_lstm_seq_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::bfzyx,
        format::byxf,
        format::bzyxf,
        format::b_fs_yx_fsv2,
        format::b_fs_zyx_fsv2,
        format::b_fs_yx_fsv4,
        format::b_fs_zyx_fsv4,
        format::b_fs_yx_fsv8,
        format::b_fs_zyx_fsv8,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv8,
        format::bs_fs_yx_bsv16_fsv4,
        format::bs_fs_yx_bsv16_fsv2,
        format::bs_fs_zyx_bsv8_fsv4,
        format::bs_fs_zyx_bsv16_fsv8,
        format::bs_fs_zyx_bsv16_fsv4,
        format::bs_fs_zyx_bsv16_fsv2,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_zyx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
    };
    implementation_map<lstm_seq>::add(impl_types::onednn, lstm_seq_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::lstm_seq_onednn)
