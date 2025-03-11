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

    auto reshape_3d_to_2d = [](const ov::Output<ov::Node>& output) {
        auto in_ps = output.get_node()->get_input_partial_shape(0);
        auto out_ps = output.get_node()->get_output_partial_shape(0);
        return in_ps.rank().is_static() && out_ps.rank().is_static() && in_ps.size() == 3 && out_ps.size() == 2;
    };

    auto weights_m = wrap_type<ov::op::v0::Constant>(compressed_constant);
    auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m});

    auto sub_const_m = wrap_type<ov::op::v0::Constant>();
    auto sub_convert_const_m = wrap_type<ov::op::v0::Convert>({sub_const_m});
    auto sub_with_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_convert_const_m});
    auto sub_no_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});
    auto subtract_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sub_with_convert_m, sub_no_convert_m});

    auto mul_const_m = wrap_type<ov::op::v0::Constant>();
    auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m});

    auto data_m = any_input();
    auto bias_m = any_input();
    auto weights_input_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{weights_m, mul_with_sub_m});
    auto conv_m = wrap_type<op::Convolution>({data_m, weights_input_m, bias_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(conv_m));
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(weights_m));
        OPENVINO_ASSERT(pattern_map.count(bias_m));
        OPENVINO_ASSERT(pattern_map.count(convert_m));
        auto fc = ov::as_type_ptr<op::Convolution>(pattern_map.at(conv_m).get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        auto scale_shape = pattern_map.at(mul_const_m).get_shape();
        bool grouped = std::count_if(scale_shape.begin(), scale_shape.end(), [](size_t d) { return d > 1; }) > 1;
        bool sub_with_convert = (pattern_map.count(sub_with_convert_m) > 0) ? true : false;

        auto weight_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        bool weight_u8 = false;
        if (weight_ptr->get_element_type() == ov::element::u8 || weight_ptr->get_element_type() == ov::element::i8)
            weight_u8 = true;

        std::shared_ptr<ov::Node> optional_zero_point = nullptr;
        const ov::Output<Node>& fc_input_a = fc->input(0).get_source_output();
        const auto& scale = mul_const_m;
        std::shared_ptr<ov::Node> fc_input_b = weights_m;
        std::shared_ptr<ov::Node> fc_input_scale = scale;
        std::shared_ptr<ov::Node> fc_input_zp = optional_zero_point;
        std::shared_ptr<ov::Node> fc_input_bias = pattern_map.at(bias_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};



        std::shared_ptr<ov::Node> new_fc = nullptr;

        new_fc = std::make_shared<op::ConvolutionCompressed>(fc_input_a,
                                                                fc_input_b,
                                                                fc_input_bias,
                                                                fc_input_scale,
                                                                fc_input_zp,
                                                                fc->get_strides(),
                                                                fc->get_pads_begin(),
                                                                fc->get_pads_end(),
                                                                fc->get_dilations(),
                                                                fc->get_groups(),
                                                                fc->get_auto_pad(),
                                                                fc->get_output_element_type());
        result_nodes.push_back(new_fc);
        new_fc->set_friendly_name(fc->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(fc, new_fc);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_m, "ConvertConvolutionToConvolutionCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
