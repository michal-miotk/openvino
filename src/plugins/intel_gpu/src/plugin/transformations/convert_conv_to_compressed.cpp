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
    std::cout << "oh creating matcher" << std::endl;
    auto weights_m = wrap_type<ov::op::v0::Constant>(compressed_constant);
    auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m});

    auto sub_const_m = any_input();
    auto sub = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});
    /*
    __attribute_maybe_unused__ auto mul_const_m = wrap_type<ov::op::v0::Constant>();
    __attribute_maybe_unused__ auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({sub, mul_const_m});

    __attribute_maybe_unused__ auto data_m = any_input();
    
    //auto bias_m = any_input();
    auto conv_m = wrap_type<op::Convolution>({data_m, mul_with_sub_m});
    */
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        std::cout << "matcher begin" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();
        //OPENVINO_ASSERT(pattern_map.count(conv_m));
        std::cout << "aa" << std::endl;
        //OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        std::cout << "bb" << std::endl;
        OPENVINO_ASSERT(pattern_map.count(weights_m));
        std::cout << "cc" << std::endl;
        //OPENVINO_ASSERT(pattern_map.count(convert_m));
        /*
        //
        std::cout << "dd" << std::endl;
        auto conv = ov::as_type_ptr<op::Convolution>(pattern_map.at(conv_m).get_node_shared_ptr());
        if (!conv || transformation_callback(conv)) {
            std::cout << "it is not match" << std::endl;
            return false;
        }
        std::cout << "is is match" << std::endl;
        auto scale_shape = pattern_map.at(mul_const_m).get_shape();
   
        auto weight_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());


        std::shared_ptr<ov::Node> optional_zero_point = nullptr;
        const ov::Output<Node>& fc_input_a = conv->input(0).get_source_output();
        const auto& scale = mul_const_m;
        std::shared_ptr<ov::Node> fc_input_b = weights_m;
        std::shared_ptr<ov::Node> fc_input_scale = scale;
        std::shared_ptr<ov::Node> fc_input_zp = pattern_map.at(sub_const_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> fc_input_bias = nullptr;//pattern_map.at(bias_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};



        std::shared_ptr<ov::Node> new_fc = nullptr;

        new_fc = std::make_shared<op::ConvolutionCompressed>(fc_input_a,
                                                                fc_input_b,
                                                                fc_input_bias,
                                                                fc_input_scale,
                                                                fc_input_zp,
                                                                conv->get_strides(),
                                                                conv->get_pads_begin(),
                                                                conv->get_pads_end(),
                                                                conv->get_dilations(),
                                                                conv->get_groups(),
                                                                conv->get_auto_pad(),
                                                                conv->get_output_element_type(0));
        result_nodes.push_back(new_fc);
        new_fc->set_friendly_name(conv->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(conv, new_fc);
        */
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sub, "ConvertConvolutionToConvolutionCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
