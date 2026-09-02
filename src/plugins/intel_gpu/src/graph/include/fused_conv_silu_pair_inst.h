// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/fused_conv_silu_pair.hpp"
#include "primitive_inst.h"

#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<fused_conv_silu_pair> : public typed_program_node_base<fused_conv_silu_pair> {
    using parent = typed_program_node_base<fused_conv_silu_pair>;
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    program_node& weights1() const { return get_dependency(1); }
    program_node& bias1() const { return get_dependency(2); }
    program_node& weights2() const { return get_dependency(3); }
    program_node& bias2() const { return get_dependency(4); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using fused_conv_silu_pair_node = typed_program_node<fused_conv_silu_pair>;

template <>
class typed_primitive_inst<fused_conv_silu_pair> : public typed_primitive_inst_base<fused_conv_silu_pair> {
    using parent = typed_primitive_inst_base<fused_conv_silu_pair>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(fused_conv_silu_pair_node const& node,
                                                   const kernel_impl_params& impl_param);
    static layout calc_output_layout(fused_conv_silu_pair_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(fused_conv_silu_pair_node const& node);

    typed_primitive_inst(network& network, fused_conv_silu_pair_node const& node);
};

using fused_conv_silu_pair_inst = typed_primitive_inst<fused_conv_silu_pair>;

}  // namespace cldnn
