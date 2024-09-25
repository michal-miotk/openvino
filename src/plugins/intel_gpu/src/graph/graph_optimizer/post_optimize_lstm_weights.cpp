// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "impls/registry/implementation_map.hpp"

#include "intel_gpu/runtime/format.hpp"
#include "lstm_seq_inst.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/permute.hpp"
#endif // ENABLE_ONEDNN_FOR_GPU
namespace cldnn {

post_optimize_lstm_weights::post_optimize_lstm_weights(reorder_factory& rf_ref)
    : base_pass("post_optimize_weights"), _rf(rf_ref) {}

template<typename T>
post_optimize_lstm_weights::weights_bias_offset post_optimize_lstm_weights::get_weights_bias_offset(const T& node) {
    return weights_bias_offset(node.get_primitive()->input.size(), 1);
}

// function which prepares given primitive for weights optimization
template<typename T>
void post_optimize_lstm_weights::optimize_lstm_weights(T& node, program& p) {
    //auto offsets = get_weights_bias_offset(node);
    auto impl = node.get_selected_impl();

    // Skip load-time weights reordering if impl is not selected
    if (!impl)
        return;

    if (impl->is_dynamic()) {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->disable_build_time_weight_reorder_for_dynamic_nodes) {
            return;
        }
        // TODO: To relax current limitation w.r.t the future optimization of weight reorder process
        // In dynamic shape, selected weight format can change in runtime. However reordering blocked format to blocked format is not fully verified yet.
        // So we need to enable other primitives such as convolution with verifying reorder b/w the possible layouts
        // Also we skip weight reorder for onednn impl because onednn fully connected layer is using simple format, therefore
        // reordering to cldnn shape_agnostic_kernel's preferred blocked format at build time does not helpful for the performance.
        // This situation might be changed once onednn shape agnostic kernel is used in the future.
        if (p.is_internal_program())
            return;
        if (node.get_preferred_impl_type() == impl_types::onednn)
            return;
        if (node.type() != lstm_seq::type_id())
            return;
    }

    auto output_layout = node.get_output_layout();
    auto weights_reorder_params = impl->get_weights_reorder_params();
    for (auto i = 0; i < 1; i++) {
        program_node& prev_node = node.get_dependency(i);

        if (weights_reorder_params != nullptr) {
            auto x = _rf.get_weights_split(prev_node.id(), weights_reorder_params, p);
            // insert new weights reorder node to topology
            std::cout << x.first->input_size() << std::endl;
            auto& first_split_node = p.get_or_create(x.first);
            auto& last_split_node = p.get_or_create(x.second);
            //p.remove_connection(prev_node, node);
            p.add_connection(prev_node, first_split_node, 0);
            //p.add_intermediate(weights_reorder.first, node, i, !weights_reorder.second);
            // set weights reorder's node output layout and implementation
            p.add_connection(last_split_node, node, 3);
        }
    }
    // set the old output layout and do not invalidate users as change of weights will not affect output layout
    node.set_output_layout(output_layout, false);
}

void post_optimize_lstm_weights::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<lstm_seq>()) {
            optimize_lstm_weights(node->as<lstm_seq>(), p);
        }
    }
}

}  // namespace cldnn
