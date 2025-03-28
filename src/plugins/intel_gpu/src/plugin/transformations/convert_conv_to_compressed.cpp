// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_conv_to_compressed.hpp"

#include <memory>

#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/convolution_compressed.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

    ConvertConvolutionToConvolutionCompressed::ConvertConvolutionToConvolutionCompressed() {
    using namespace ov::pass::pattern;

    auto compressed_constant = [](const ov::Output<ov::Node>& output) {
        return (output.get_element_type() == ov::element::u8 ||
                output.get_element_type() == ov::element::i8 ||
                output.get_element_type() == ov::element::u4 ||
                output.get_element_type() == ov::element::i4);
    };
    auto weights_m = wrap_type<ov::op::v0::Constant>(compressed_constant);
    auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m});

    auto sub_const_m = any_input();
    auto sub = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});

    __attribute_maybe_unused__ auto mul_const_m = wrap_type<ov::op::v0::Constant>();
    __attribute_maybe_unused__ auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({sub, mul_const_m});

    __attribute_maybe_unused__ auto data_m = any_input();
    auto conv_m = wrap_type<ov::op::v1::Convolution>({data_m, mul_with_sub_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(conv_m));
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(sub_const_m));
        OPENVINO_ASSERT(pattern_map.count(weights_m));
        OPENVINO_ASSERT(pattern_map.count(convert_m));
        auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map.at(conv_m).get_node_shared_ptr());
        if (!conv) {
            return false;
        }

        auto scale_shape = pattern_map.at(mul_const_m).get_shape();
        auto weight_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        const ov::Output<Node>& conv_input_a = pattern_map.at(data_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> conv_input_b = pattern_map.at(weights_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> conv_input_scale =  pattern_map.at(mul_const_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> conv_input_zp = pattern_map.at(sub_const_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};


        int64_t groups = 0;
        if (auto grouped_conv = ov::as_type_ptr<ov::op::v1::GroupConvolution>(conv_m)) {
            auto weights_shape = grouped_conv->get_input_partial_shape(1);
            if (weights_shape[0].is_dynamic())
                return false;
            groups = weights_shape[0].get_length();
        }
        std::shared_ptr<ov::Node> new_conv = nullptr;
        new_conv = std::make_shared<op::ConvolutionCompressed>(conv_input_a,
                                                                conv_input_b,
                                                                conv_input_scale,
                                                                conv_input_zp,
                                                                conv->get_strides(),
                                                                conv->get_pads_begin(),
                                                                conv->get_pads_end(),
                                                                conv->get_dilations(),
                                                                groups,
                                                                conv->get_auto_pad(),
                                                                conv->get_output_element_type(0));
        result_nodes.push_back(new_conv);
        new_conv->set_friendly_name(conv->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_m, "ConvertConvolutionToConvolutionCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
