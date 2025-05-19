// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "test_utils.h"
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <openvino/runtime/core.hpp>
#include "openvino/op/concat.hpp"
#include "openvino/op/add.hpp"
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>

using namespace cldnn;
using namespace tests;

class test_device_mem_usage_estimation: public ::testing::Test {
public:
    void test_basic(bool is_caching_test) {
        ExecutionConfig cfg = get_test_default_config(get_test_engine());
        cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));

        std::shared_ptr<cldnn::engine> engine1 = create_test_engine();
        if (engine1->get_device_info().supports_immad) {
            // Enable this test for out_of_order queue-type if Onednn supports out_of_order
            return;
        }

        auto input1 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        auto input2 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        topology topology(
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            permute("permute1", input_info("input1"), { 0, 3, 1, 2 }),
            permute("permute2", input_info("input2"), { 0, 2, 1, 3 }),
            eltwise("eltw", { input_info("permute1"), input_info("permute2") }, eltwise_mode::sum, data_types::f16),
            reorder("output", input_info("eltw"), format::bfyx, data_types::f32)
        );

        auto prog = program::build_program(*engine1, topology, cfg);
        std::pair<int64_t, int64_t> estimated_mem_usage = prog->get_estimated_device_mem_usage();

        std::shared_ptr<cldnn::engine> engine2 = create_test_engine();
        auto input3 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        auto input4 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });

        cldnn::network::ptr network = get_network(*engine2, topology, cfg, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input3);
        network->set_input_data("input2", input4);
        ASSERT_EQ(estimated_mem_usage.first + estimated_mem_usage.second, engine2->get_used_device_memory(allocation_type::usm_device));
    }

    std::shared_ptr<ov::Model> make_single_concat_with_constant(ov::Shape input_shape, ov::element::Type type) {
        ov::ParameterVector parameter{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
        parameter[0]->set_friendly_name("Param_1");
        parameter[0]->output(0).get_tensor().set_names({"data"});

        auto init_const = ov::op::v0::Constant::create(type, input_shape, {0});

        std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], init_const};
        auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
        conc->set_friendly_name("concat");
        auto add_const = ov::op::v0::Constant::create(type, {1}, {0});
        auto add = std::make_shared<ov::op::v1::Add>(conc, add_const);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        res->set_friendly_name("result");

        std::shared_ptr<ov::Model> model =
            std::make_shared<ov::Model>(ov::ResultVector({res}), ov::ParameterVector{parameter});
        model->set_friendly_name("SingleConcatWithConstant");
        return model;
    }

    void get_max_batch_size() {
        ov::Core ie;
        uint32_t batch_size = 99;
        uint32_t n_streams = 2;
        std::string target_device = "GPU";
        auto simpleNetwork = make_single_concat_with_constant(ov::Shape({1,1,1,224}), ov::element::Type("i8"));
        auto exec_net1 = ie.compile_model(simpleNetwork, target_device);
        //std::shared_ptr<cldnn::engine> engine = create_test_engine(engine_types::ocl, runtime_types::ocl);
        //std::cout << engine->get_lockable_preferred_memory_allocation_type(lay.format.is_image_2d());
        ov::AnyMap _options = {ov::hint::model(simpleNetwork),
                            ov::num_streams(n_streams)};

        OV_ASSERT_NO_THROW(batch_size = ie.get_property(target_device, ov::max_batch_size.name(), _options).as<unsigned int>());

        std::cout << "batch_size: " << batch_size<< std::endl;
    }
};



TEST_F(test_device_mem_usage_estimation, GetMaxBatchSize) {
    this->get_max_batch_size();
}

TEST_F(test_device_mem_usage_estimation, basic) {
    this->test_basic(false);
}

TEST_F(test_device_mem_usage_estimation, basic_cached) {
    this->test_basic(true);
}
