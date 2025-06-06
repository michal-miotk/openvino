// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/opsets/opset10_decl.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/convert_precision.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/gather.hpp"

using namespace ov;

static std::shared_ptr<ov::Node> make_branch(const std::shared_ptr<ov::op::v0::Convert> zp_convert,
                                             const ov::Shape& lp_const_shape,
                                             const std::vector<size_t>& sub_mul_shape,
                                             bool ref_model = false) {
        // const data path
        const auto lp_const = std::make_shared<opset10::Constant>(element::i8, lp_const_shape, 1);
        const auto data_convert = std::make_shared<opset10::Convert>(lp_const, element::f32);

        const auto zp_target_shape = std::make_shared<opset10::Constant>(ov::element::i64, ov::Shape{sub_mul_shape.size()}, sub_mul_shape);
        const auto zp_reshape = std::make_shared<opset10::Reshape>(zp_convert, zp_target_shape, false);

        const auto subtract = std::make_shared<opset10::Subtract>(data_convert, zp_reshape);

        // scale
        std::shared_ptr<ov::Node> scale_const;
        std::shared_ptr<ov::Node> scale_reshape;
        if (ref_model) {
            scale_const = std::make_shared<opset10::Constant>(element::f32, ov::Shape{sub_mul_shape}, 1);
            scale_reshape = scale_const;
        } else {
            scale_const = std::make_shared<opset10::Constant>(element::f32, ov::Shape{64}, 1);
            auto scale_target_shape = std::make_shared<opset10::Constant>(ov::element::i64, ov::Shape{sub_mul_shape.size()}, sub_mul_shape);
            scale_reshape = std::make_shared<opset10::Reshape>(scale_const, scale_target_shape, false);
        }

        const auto multiply = std::make_shared<opset10::Multiply>(subtract, scale_reshape);

        if (ref_model) {
            mark_as_dequantization_node(subtract);
            mark_as_dequantization_node(multiply);
            disable_constant_folding(data_convert);
            disable_constant_folding(zp_convert);
            enable_keep_const_precision(lp_const);
            disable_keep_const_precision(scale_const);
        }

        return multiply;
}

/* Construct a following graph that will have only Scale constant
   folded during Constant folding as swapping of ZP Reshape & Convert
   is not possible due to different shapes, hence no CF here.

                              ZP Const
                                 │
                                 ▼
         Input                Convert                Input
           │                  │     │                  │
           ▼                  ▼     ▼                  ▼
Scale      Convert      Reshape     Reshape      Convert      Scale
    |            │    (64,1,1,1)   (1,64,1,1)    │            │
    |            │      │                 │      │            |
    ▼            ▼      ▼                 ▼      ▼            ▼
    Reshape      Subtract                 Subtract      Reshape
          |      |                               |      |
          ▼      ▼                               ▼      ▼
          Multiply                               Multiply
*/

TEST_F(TransformationTestsF, KeepConstPrecision2BranchesDiffShapes) {
    {

        // zero points
        const auto zp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64}, 1);
        const auto zp_convert = std::make_shared<opset10::Convert>(zp_const, element::f32);

        const auto right_branch = make_branch(zp_convert, {128,64,2,2}, {1, 64, 1, 1});
        const auto result_right = std::make_shared<opset10::Result>(right_branch);

        const auto left_branch = make_branch(zp_convert, {64,3,3,3}, {64, 1, 1, 1});
        const auto result_left = std::make_shared<opset10::Result>(left_branch);

        model = std::make_shared<Model>(ResultVector{result_right, result_left}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{ov::element::i8, ov::element::u8, ov::element::i4, ov::element::u4});
    manager.register_pass<pass::ConstantFolding>();
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::i8});
    manager.register_pass<pass::ConvertPrecision>(ov::element::u4, ov::element::u8, type_to_fuse_map{}, false, false);

    {
        // zero points
        const auto zp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64}, 1);
        enable_keep_const_precision(zp_const);
        const auto zp_convert = std::make_shared<opset10::Convert>(zp_const, element::f32);

        const auto right_branch = make_branch(zp_convert, {128, 64, 2, 2}, {1, 64, 1, 1}, true);
        const auto result_right = std::make_shared<opset10::Result>(right_branch);

        const auto left_branch = make_branch(zp_convert, {64, 3, 3, 3}, {64, 1, 1, 1}, true);
        const auto result_left = std::make_shared<opset10::Result>(left_branch);

        model_ref = std::make_shared<Model>(ResultVector{result_right, result_left}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}


static std::shared_ptr<ov::Node> make_branch_same(const std::shared_ptr<ov::op::v0::Convert> zp_convert) {
    // const data
    const auto lp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64, 3, 3, 3}, 1);
    const auto data_convert = std::make_shared<opset10::Convert>(lp_const, element::f32);

    const auto subtract = std::make_shared<opset10::Subtract>(data_convert, zp_convert);

    // scale
    const auto scale_const = std::make_shared<opset10::Constant>(element::f32, ov::Shape{64, 1, 1, 1}, 1);
    const auto multiply = std::make_shared<opset10::Multiply>(subtract, scale_const);

    enable_keep_const_precision(lp_const);
    disable_constant_folding(data_convert);
    mark_as_dequantization_node(subtract);
    mark_as_dequantization_node(multiply);
    disable_keep_const_precision(scale_const);

    return multiply;
}

/* Construct the same as graph above, but with shapes being identical, hence
   possible to swap and then Constant fold*/

TEST_F(TransformationTestsF, KeepConstPrecision2BranchesSameShapes) {
    {

        // zero points
        const auto zp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64}, 1);
        const auto zp_convert = std::make_shared<opset10::Convert>(zp_const, element::f32);

        const auto right_branch = make_branch(zp_convert, {64, 3, 3, 3}, {64, 1, 1, 1});
        const auto result_right = std::make_shared<opset10::Result>(right_branch);

        const auto left_branch = make_branch(zp_convert, {64, 3, 3, 3}, {64, 1, 1, 1});
        const auto result_left = std::make_shared<opset10::Result>(left_branch);

        model = std::make_shared<Model>(ResultVector{result_right, result_left}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{ov::element::i8, ov::element::u8, ov::element::i4, ov::element::u4});
    manager.register_pass<pass::ConstantFolding>();
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::i8});
    manager.register_pass<pass::ConvertPrecision>(ov::element::u4, ov::element::u8, type_to_fuse_map{}, false, false);

    {
        // zero points
        const auto zp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64, 1, 1, 1}, 1);
        const auto zp_convert = std::make_shared<opset10::Convert>(zp_const, element::f32);

        const auto multiply_left = make_branch_same(zp_convert);
        const auto result_left = std::make_shared<opset10::Result>(multiply_left);

        const auto multiply_right = make_branch_same(zp_convert);
        const auto result_right = std::make_shared<opset10::Result>(multiply_right);

        model_ref = std::make_shared<Model>(ResultVector{result_right, result_left}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationScaleOnTheLeftBranch) {
    {
        auto lp_const = std::make_shared<opset10::Constant>(element::u4, Shape{27}, 1);
        auto scale = opset10::Constant::create(element::i64, Shape{}, {2});

        auto convert_lp = std::make_shared<opset10::Convert>(lp_const, element::f32);
        auto convert_scale = std::make_shared<opset10::Convert>(scale, element::f32);
        auto multiply = std::make_shared<opset10::Multiply>(convert_scale, convert_lp);
        auto stub_op = std::make_shared<opset10::Relu>(multiply);
        model = std::make_shared<Model>(stub_op, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u4});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto lp_const = std::make_shared<opset10::Constant>(element::u4, Shape{27}, 1);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {2});

        auto convert_lp = std::make_shared<opset10::Convert>(lp_const, element::f32);
        auto multiply = std::make_shared<opset10::Multiply>(scale, convert_lp);
        auto stub_op = std::make_shared<opset10::Relu>(multiply);
        model_ref = std::make_shared<Model>(stub_op, ParameterVector{});

        mark_as_dequantization_node(multiply);
        ov::pass::disable_constant_folding(convert_lp);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepConstPrecision) {
    {
        auto lp_const = std::make_shared<opset10::Constant>(element::u4, Shape{27}, 1);

        const auto target_shape = std::make_shared<opset10::Constant>(ov::element::i64, ov::Shape{3}, 3);
        auto reshape = std::make_shared<opset10::Reshape>(lp_const, target_shape, false);

        auto second_convert = std::make_shared<opset10::Convert>(reshape, element::f32);
        auto zero_point = opset10::Constant::create(element::f32, Shape{}, {127});
        auto subtract = std::make_shared<opset10::Subtract>(second_convert, zero_point);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        auto stub_op = std::make_shared<opset10::Relu>(multiply);
        model = std::make_shared<Model>(stub_op, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u4});
    manager.register_pass<pass::ConstantFolding>();
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::u4});
    manager.register_pass<pass::ConvertPrecision>(ov::element::u4, ov::element::u8, type_to_fuse_map{}, false, false);

    {
        auto lp_const = std::make_shared<opset10::Constant>(element::u4, Shape{3, 3, 3}, 1);
        auto second_convert = std::make_shared<opset10::Convert>(lp_const, element::f32);
        auto zero_point = opset10::Constant::create(element::f32, Shape{}, {127});
        auto subtract = std::make_shared<opset10::Subtract>(second_convert, zero_point);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        auto stub_op = std::make_shared<opset10::Relu>(multiply);
        model_ref = std::make_shared<Model>(stub_op, ParameterVector{});

        mark_as_dequantization_node(subtract);
        mark_as_dequantization_node(multiply);
        enable_keep_const_precision(lp_const);
        ov::pass::disable_constant_folding(second_convert);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationTransformation) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert   Constant      Constant  Constant
    //      |U8        |U8           |I8       |I8
    //      |          |             |         |
    //     Convert  Convert(DCF) Convert(DCF) Convert(DCF)
    //       \FP32    /FP32          |FP32   /F32
    //        \      /               |      /
    //        Subtract  Constant    Subtract  Constant
    //          \FP32   /FP32        |FP32    /FP32
    //           \     /             |       /
    //           Multiply           Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantization all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // All 'Convert(DCF)' nodes from above graph are marked with 'DisableConstantFolding' attribute
    // Weights and zero points are marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        enable_keep_const_precision(weights);
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            enable_keep_const_precision(zero_point);
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationTransformationNoZeroPoint) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert             Constant
    //      |U8                   |I8
    //      |                     |
    //     Convert  Constant   Convert(DCF) Constant
    //       \FP32    /FP32       \FP32    /F32
    //        \      /             \      /
    //         Multiply            Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantization all Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also 'Convert(DCF)' node from above graph is marked with 'DisableConstantFolding' attribute
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        enable_keep_const_precision(weights);
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationTransformationNoZeroPointFP16) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert  Constant   Constant        Constant
    //      |U8       |FP16       |I8         /FP16
    //      |         |           |          /
    //     Convert  Convert    Convert(DCF) Convert
    //       \FP32    /FP32       \FP32    /F32
    //        \      /             \      /
    //         Multiply            Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantization all Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also 'Convert(DCF)' node from above graph is marked with 'DisableConstantFolding' attribute
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale_convert);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale_convert);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::u8, element::i8});

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(second_convert, scale_convert);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        enable_keep_const_precision(weights);
        {
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto scale = opset10::Constant::create(element::f16, Shape{}, {0.2});
            auto scale_convert = std::make_shared<opset10::Convert>(scale, element::f32);
            mark_as_decompression(scale_convert);
            auto multiply = std::make_shared<opset10::Multiply>(convert, scale_convert);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationTransformationNotConstantWeights) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize            Constant
    //      |F32                     |I8
    //      |                        |
    //     Convert   Constant       Clamp    Constant
    //      |U8        |U8           |I8       |I8
    //      |          |             |         |
    //     Convert  Convert(DCF) Convert(DCF) Convert(DCF)
    //       \FP32    /FP32          |FP32   /F32
    //        \      /               |      /
    //        Subtract  Constant    Subtract  Constant
    //          \FP32   /FP32        |FP32    /FP32
    //           \     /             |       /
    //           Multiply           Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    // After MarkDequantization all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also all 'Convert(DCF)' nodes from above graph are marked with 'DisableConstantFolding' attribute
    // Weights and zero point nodes are marked with 'KeepConstPrecision' attribute

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-3});
        enable_keep_const_precision(weights);
        {
            auto clamp = std::make_shared<opset10::Clamp>(weights, -2, 2);
            auto convert = std::make_shared<opset10::Convert>(clamp, element::f32);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            enable_keep_const_precision(zero_point);
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::u8, element::i8});
    manager.register_pass<pass::ConstantFolding>();

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 16, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset10::FakeQuantize>(parameter,
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {20}),
                                                    opset10::Constant::create(element::f32, Shape{}, {0}),
                                                    opset10::Constant::create(element::f32, Shape{}, {254}),
                                                    255);
        {
            auto first_convert = std::make_shared<opset10::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset10::Convert>(first_convert, element::f32);
            auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(second_convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset10::Constant::create(element::i8, Shape{4, 16, 1, 1}, {-2});
        {
            // Clamp was constantfolded
            auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
            pass::disable_constant_folding(convert);
            auto zero_point = opset10::Constant::create(element::i8, Shape{}, {127});
            auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
            pass::disable_constant_folding(convert_on_zero_point);
            auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
            mark_as_dequantization_node(subtract);
            auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
            mark_as_dequantization_node(multiply);
            weights = multiply;
        }

        auto conv = std::make_shared<opset10::Convolution>(activations,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        model_ref = std::make_shared<Model>(conv, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkDequantizationTransformationFoldSubConst) {
    // Input graph:           After transformation:
    //
    // Constant    Constant               Constant
    //    |U8         |U8                    |U8
    //    |           |                      |
    // Convert     Convert                Convert(DCF)  Constant
    //    |FP32   /F32                       |FP32      /FP32
    //    |      /                            \        /
    //   Subtract  Constant                    Subtract   Constant
    //    |FP32    /FP32                          |FP32     /FP32
    //    |       /                                \        /
    //   Multiply                                   Multiply
    //
    // After MarkDequantization all Subtract and Multiply nodes from above graph
    // are marked with 'DequantizationNode' attribute.
    // Also all 'Convert(DCF)' node before weights is marked with 'DisableConstantFolding' attribute
    // but Convert before Dequantization Sub const isn't because fold_subtract_const is set to true
    // Weights node is marked with 'KeepConstPrecision' attribute

    {
        auto weights = opset10::Constant::create(element::u8, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
        auto zero_point = opset10::Constant::create(element::u8, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, element::f32);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        model = std::make_shared<ov::Model>(ov::OutputVector{multiply});
    }

    manager.register_pass<pass::MarkDequantization>(element::TypeVector{element::u8}, true);
    manager.register_pass<pass::KeepConstPrecision>(element::TypeVector{element::u8}, true);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = opset10::Constant::create(element::u8, Shape{4, 16, 1, 1}, {3});
        enable_keep_const_precision(weights);
        auto convert = std::make_shared<opset10::Convert>(weights, element::f32);
        pass::disable_constant_folding(convert);
        auto zero_point = opset10::Constant::create(element::f32, Shape{}, {127});
        auto subtract = std::make_shared<opset10::Subtract>(convert, zero_point);
        mark_as_dequantization_node(subtract);
        auto scale = opset10::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{multiply});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepDequantizationPrecisionTransformationMarkup) {
    // After KeepDequantizationPrecision all Convert, Subtract, and Multiply nodes
    // are marked with the 'disable_fp16_compression' attribute

    auto quantization_dt = element::u16;
    auto dequantization_dt = element::f32;

    {
        auto parameter = std::make_shared<opset10::Parameter>(dequantization_dt, Shape{1});
        auto weights = opset10::Constant::create(quantization_dt, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, dequantization_dt);
        auto zero_point = opset10::Constant::create(quantization_dt, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, dequantization_dt);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = opset10::Constant::create(dequantization_dt, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        auto add = std::make_shared<opset10::Add>(parameter, multiply);
        model = std::make_shared<ov::Model>(ov::OutputVector{add});
    }

    manager.register_pass<pass::KeepDequantizationPrecision>(element::TypeVector{quantization_dt});

    {
        auto parameter = std::make_shared<opset10::Parameter>(dequantization_dt, Shape{1});
        auto weights = opset10::Constant::create(quantization_dt, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, dequantization_dt);
        disable_fp16_compression(convert);
        auto zero_point = opset10::Constant::create(quantization_dt, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, dequantization_dt);
        disable_fp16_compression(convert_on_zero_point);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        disable_fp16_compression(subtract);
        auto scale = opset10::Constant::create(dequantization_dt, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        disable_fp16_compression(multiply);
        auto add = std::make_shared<opset10::Add>(parameter, multiply);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepDequantizationPrecisionTransformationMarkupNonConst) {
    // KeepDequantizationPrecision pass should not match the specified subgraph,
    // as it only targets constant weights, scales, and zero points.

    auto quantization_dt = element::u16;
    auto dequantization_dt = element::f32;

    {
        auto parameter = std::make_shared<opset10::Parameter>(dequantization_dt, Shape{1});
        auto weights = opset10::Constant::create(quantization_dt, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, dequantization_dt);
        auto zero_point = opset10::Constant::create(quantization_dt, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, dequantization_dt);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = std::make_shared<opset10::Parameter>(dequantization_dt, Shape{});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        auto add = std::make_shared<opset10::Add>(parameter, multiply);
        model = std::make_shared<ov::Model>(ov::OutputVector{add});
    }

    manager.register_pass<pass::KeepDequantizationPrecision>(element::TypeVector{quantization_dt});

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTests, KeepDequantizationPrecisionTransformationFolding) {
    // This test uses the legacy TransformationTests class because ConvertPrecision pass
    // internally inserts additional nodes for data type consistency, which causes the
    // check_unique_names() function to fail with a "Tensor name: N is missing in ..." error.
    // Once this issue is fixed, the test should be migrated to the new TransformationTestsF infrastructure.

    // After KeepDequantizationPrecision all Convert, Subtract, and Multiply nodes
    // are marked with the 'disable_fp16_compression' attribute, and dequantization subgraph is folded
    // during Constant Folding (called from ConvertPrecision pass), using f32 data type for constants.

    std::shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto quantization_dt = element::u16;
    auto dequantization_dt = element::f32;

    {
        auto parameter = std::make_shared<opset10::Parameter>(dequantization_dt, Shape{1});
        auto weights = opset10::Constant::create(quantization_dt, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, dequantization_dt);
        auto zero_point = opset10::Constant::create(quantization_dt, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, dequantization_dt);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = opset10::Constant::create(dequantization_dt, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        auto add = std::make_shared<opset10::Add>(parameter, multiply);
        model = std::make_shared<ov::Model>(ov::OutputVector{add});

        const auto keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::KeepDequantizationPrecision>(element::TypeVector{quantization_dt});
        manager.register_pass<pass::ConvertPrecision>(element::f32, element::f16, type_to_fuse_map{}, keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto parameter = std::make_shared<opset10::Parameter>(element::f16, Shape{1});
        auto input_decompress_to_f32 = std::make_shared<opset10::Convert>(parameter, element::f32);
        auto weights = opset10::Constant::create(dequantization_dt, Shape{4, 16, 1, 1}, {3});
        auto add = std::make_shared<opset10::Add>(input_decompress_to_f32, weights);
        auto add_compressed_to_f16 = std::make_shared<opset10::Convert>(add, element::f16);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add_compressed_to_f16});
    }

    auto func_comparator = FunctionsComparator::with_default();
    auto result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST_F(TransformationTests, KeepDequantizationPrecisionTransformationFoldingWithConvert) {
    // This test uses the legacy TransformationTests class because ConvertPrecision pass
    // internally inserts additional nodes for data type consistency, which causes the
    // check_unique_names() function to fail with a "Tensor name: N is missing in ..." error.
    // Once this issue is fixed, the test should be migrated to the new TransformationTestsF infrastructure.

    // After KeepDequantizationPrecision all Convert, Subtract, and Multiply nodes
    // are marked with the 'disable_fp16_compression' attribute, and dequantization subgraph is folded
    // during Constant Folding (called from ConvertPrecision pass), using f32 data type and finally
    // converted to the f16 data type.

    std::shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto quantization_dt = element::u16;
    auto dequantization_dt = element::f32;

    {
        auto parameter = std::make_shared<opset10::Parameter>(dequantization_dt, Shape{1});
        auto weights = opset10::Constant::create(quantization_dt, Shape{4, 16, 1, 1}, {3});
        auto convert = std::make_shared<opset10::Convert>(weights, dequantization_dt);
        auto zero_point = opset10::Constant::create(quantization_dt, Shape{}, {127});
        auto convert_on_zero_point = std::make_shared<opset10::Convert>(zero_point, dequantization_dt);
        auto subtract = std::make_shared<opset10::Subtract>(convert, convert_on_zero_point);
        auto scale = opset10::Constant::create(dequantization_dt, Shape{}, {0.2});
        auto multiply = std::make_shared<opset10::Multiply>(subtract, scale);
        auto add = std::make_shared<opset10::Add>(parameter, multiply);
        model = std::make_shared<ov::Model>(ov::OutputVector{add});

        const auto keep_precision_sensitive_in_fp32 = true;
        const auto add_precision_sensitive_convert = true;
        manager.register_pass<pass::KeepDequantizationPrecision>(element::TypeVector{quantization_dt}, add_precision_sensitive_convert);
        manager.register_pass<pass::ConvertPrecision>(element::f32, element::f16, type_to_fuse_map{}, keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto weights_dt = element::f16;
        auto parameter = std::make_shared<opset10::Parameter>(element::f16, Shape{1});
        auto weights = opset10::Constant::create(weights_dt, Shape{4, 16, 1, 1}, {3});
        auto add = std::make_shared<opset10::Add>(parameter, weights);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add});
    }

    auto func_comparator = FunctionsComparator::with_default();
    auto result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

inline std::shared_ptr<Model> make_gather_model(element::Type data_type,
                                                element::Type indices_type,
                                                bool use_data_convert,
                                                bool use_indices_convert,
                                                bool indices_as_param,
                                                bool reference_model = false) {
    using op::v0::Constant;
    using op::v0::Convert;
    using op::v0::Parameter;
    using op::v8::Gather;

    ParameterVector params;

    std::shared_ptr<Node> data = std::make_shared<Constant>(data_type, Shape{4}, 2);
    if (reference_model)
        enable_keep_const_precision(data);
    if (use_data_convert) {
        data = std::make_shared<Convert>(data, element::f32);
        if (reference_model)
            disable_constant_folding(data);
    }

    std::shared_ptr<Node> indices;
    if (indices_as_param) {
        params.push_back(std::make_shared<Parameter>(indices_type, Shape{3}));
        indices = params.back();
    } else {
        indices = std::make_shared<Constant>(indices_type, Shape{3}, 1);
    }
    if (reference_model)
        enable_keep_const_precision(indices);

    if (use_indices_convert) {
        indices = std::make_shared<Convert>(indices, element::i32);
        if (reference_model)
            disable_constant_folding(indices);
    }

    auto axis = std::make_shared<Constant>(element::i64, Shape{1}, 0);
    auto gather = std::make_shared<Gather>(data, indices, axis);
    if (reference_model)
        disable_constant_folding(gather);

    return std::make_shared<Model>(OutputVector{gather}, params);
}

TEST_F(TransformationTestsF, MarkGatherSubgraph) {
    auto data_type = element::f8e4m3;
    auto indices_type = element::u4;

    model = make_gather_model(data_type, indices_type, true, true, false);
    model_ref = make_gather_model(data_type, indices_type, true, true, false, true);

    manager.register_pass<pass::MarkGatherSubgraph>(
        element::TypeVector{element::f8e4m3, element::f16},
        element::TypeVector{element::u4, element::u8}
    );

    manager.register_pass<pass::ConstantFolding>();
    precisions_map map = {
        {data_type, element::f32},
        {indices_type, element::i32},
    };
    manager.register_pass<pass::ConvertPrecision>(map);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkGatherSubgraph_IndicesAsParam) {
    auto data_type = element::f8e4m3;
    auto indices_type = element::u4;

    model = make_gather_model(data_type, indices_type, true, true, true);
    model_ref = make_gather_model(data_type, indices_type, true, true, true, true);

    manager.register_pass<pass::MarkGatherSubgraph>(
        element::TypeVector{element::f8e4m3, element::f16},
        element::TypeVector{element::u4, element::u8}
    );

    manager.register_pass<pass::ConstantFolding>();
    precisions_map map = {
        {data_type, element::f32},
        {indices_type, element::i32},
    };
    manager.register_pass<pass::ConvertPrecision>(map);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
