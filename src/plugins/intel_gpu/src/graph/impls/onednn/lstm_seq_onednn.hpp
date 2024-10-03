// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "impls/registry/implementation_manager.hpp"

#include <memory>


namespace cldnn {
namespace onednn {

struct LSTMSeqImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::lstm_seq")
    LSTMSeqImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<lstm_seq>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad)
            return false;
        assert(node.is_type<lstm_seq>());
        const auto& lstm_node = node.as<lstm_seq>();

        return lstm_node.clip() == 0.f;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<lstm_seq>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0 ; idx < 3; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            in_fmts[idx] = cldnn::format::bfyx;
        }
        out_fmts[0] = cldnn::format::ybfx;
        out_fmts[1] = cldnn::format::fbyx;
        out_fmts[2] = cldnn::format::fbyx;
        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
