// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "impls/registry/implementation_manager.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>


namespace cldnn {
namespace onednn {

struct LSTMSeqImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::lstm_seq")
    LSTMSeqImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<lstm_seq>());
        auto seq_len_program_node = node.get_dependencies()[6].first;
        bool is_seq_len_constant = seq_len_program_node->is_constant();
        const auto& lstm_node = node.as<lstm_seq>();
        auto activations = lstm_node.activations();
        std::vector<activation_func> desired_act{ activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan};
        return is_seq_len_constant && activations == desired_act && lstm_node.clip() == 0.f;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<lstm_seq>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;
        }
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
