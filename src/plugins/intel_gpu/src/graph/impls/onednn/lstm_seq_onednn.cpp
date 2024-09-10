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

struct rnn_onednn : typed_primitive_onednn_impl<lstm_seq, dnnl::lstm_forward::primitive_desc, dnnl::lstm_forward> {
    using parent = typed_primitive_onednn_impl<lstm_seq, dnnl::lstm_forward::primitive_desc, dnnl::lstm_forward>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::rnn_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rnn_onednn>(*this);
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

    static std::shared_ptr<dnnl::lstm_forward::primitive_desc> get_lstm_primitive_descriptor(const kernel_impl_params& impl_params, cldnn::engine& engine,
                                                                                           const dnnl::primitive_attr& attr) {
        auto i0 = impl_params.get_input_layout(0);
        auto i1 = impl_params.get_input_layout(1);
        auto i2 = impl_params.get_input_layout(2);
        auto i3 = impl_params.get_input_layout(3);
        auto i4 = impl_params.get_input_layout(4);
        auto i5 = impl_params.get_input_layout(5);
        auto i6 = impl_params.get_input_layout(6);
        auto prim = impl_params.typed_desc<lstm_seq>();
        auto input_layout = impl_params.get_input_layout(0);
        auto in_shape = impl_params.get_input_layout(0).get_shape();
        in_shape.pop_back();
        auto input_lay = cldnn::layout{ov::PartialShape{static_cast<long int>(in_shape[1]), static_cast<long int>(in_shape[0]), \
        static_cast<long int>(in_shape[2])}, input_layout.data_type, cldnn::format::bfx};
        auto input_md = onednn::layout_to_memory_desc(input_lay);
        auto initial_hidden_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(1));
        auto initial_cell_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2));
        auto shapeW = impl_params.get_input_layout(4).get_shape();
        auto lay_W = cldnn::layout{ov::PartialShape{static_cast<long int>(shapeW[0]), static_cast<long int>(shapeW[0]), static_cast<long int>(shapeW[2]), 4, \
        static_cast<long int>(shapeW[1]/4)}, input_layout.data_type, cldnn::format::bfzyx};
        auto W_md = onednn::layout_to_memory_desc(lay_W, dnnl::memory::format_tag::any);
        auto shapeR = impl_params.get_input_layout(5).get_shape();
        shapeR[1] = 1;
        shapeR.push_back(shapeR[2]);
        shapeR[3] = 4;
        auto lR = impl_params.get_input_layout(5).clone_with_other_shape(shapeR);
        lR.format = cldnn::format::bfzyx;
        auto R_md = onednn::layout_to_memory_desc(lR, dnnl::memory::format_tag::any);
        auto shapeB = impl_params.get_input_layout(5).get_shape();
        shapeB[3] = shapeB[1]/4;
        shapeB[1] = 1;
        shapeB[2] = 4;
        auto lB = impl_params.get_input_layout(6).clone_with_other_shape(shapeB);
        lB.format = cldnn::format::bfyx;
        auto B_md = onednn::layout_to_memory_desc(lB);
        auto out_shape = impl_params.get_output_layout().get_shape();
        auto output_lay = cldnn::layout{ov::PartialShape{static_cast<long int>(out_shape[0]), static_cast<long int>(out_shape[1]), \
        static_cast<long int>(out_shape[2])}, input_layout.data_type, cldnn::format::bfx};
        auto output_md = onednn::layout_to_memory_desc(output_lay, dnnl::memory::format_tag::abc);
        dnnl::memory::desc emptyMemDescriptorForPeephole;
        //engine.create_onednn_engine(config);
        auto eng = engine.get_onednn_engine();
        return std::make_shared<dnnl::lstm_forward::primitive_desc>(
            eng,
            dnnl::prop_kind::forward_inference,
            dnnl::rnn_direction::unidirectional_left2right,
            input_md,
            emptyMemDescriptorForPeephole,
            emptyMemDescriptorForPeephole,
            W_md,
            R_md,
            B_md,
            output_md,
            emptyMemDescriptorForPeephole,
            emptyMemDescriptorForPeephole);
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
        std::cout << "LOADING" << std::endl;
        std::cout << "2LOADING" << std::endl;
        std::cout << "3LOADING" << std::endl;
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
            auto prim_desc = get_lstm_primitive_descriptor(impl_params, engine, *attr);
            return cldnn::make_unique<rnn_onednn>(engine, config, attr, *prim_desc);
    }
};

namespace detail {

attach_rnn_onednn::attach_rnn_onednn() {
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
    implementation_map<lstm_seq>::add(impl_types::onednn, rnn_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::rnn_onednn)
