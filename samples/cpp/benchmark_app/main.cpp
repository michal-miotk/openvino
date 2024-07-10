// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include <vector>
// clang-format off
#include "openvino/openvino.hpp"
#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
// clang-format on
 //set openvino device and device config
        inline ov::AnyMap SetDeviceConfig(const std::string & device){ 
            ov::AnyMap device_config;
            if (device.find("CPU") != std::string::npos) {
                device_config[ov::cache_dir.name()] = "asr-cache";
                device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
                device_config[ov::hint::enable_hyper_threading.name()] = false;
                device_config[ov::hint::enable_cpu_pinning.name()] = true;
                device_config[ov::enable_profiling.name()] = false;
            }
            if (device.find("GPU") != std::string::npos) {
                device_config[ov::cache_dir.name()] = "asr-cache";
                device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
                device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
                device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
                device_config[ov::hint::enable_cpu_pinning.name()] = true;
                device_config[ov::enable_profiling.name()] = false;
            }
            return device_config;
        };

int main(int argc, char* argv[]) {
    try {
        slog::info << "OpenVINO:" << slog::endl;
        slog::info << ov::get_openvino_version();
        std::string device_name = "GPU";
        slog::info << "device_name "<< device_name << slog::endl;
        //Replace it with the model path
        std::string model_path = "/home/gta/resnet-34_kinetics.onnx";
        slog::info << "model_path "<< model_path << slog::endl;

        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model_path, device_name, SetDeviceConfig(device_name));
        auto conf = SetDeviceConfig(device_name);
        conf.emplace(ov::hint::inference_precision("f32"));
        slog::info << "compile_model succeed" << slog::endl;
        ov::InferRequest ireq = compiled_model.create_infer_request();
        slog::info << "create_infer_request succeed" << slog::endl;
        std::vector<ov::Tensor> tensors{{ov::element::Type_t::f32, {1,3,16,112,112}}};
        ov::Tensor out_tensor{ov::element::Type_t::f32, {1,400}};
        for(int i=0;i<3*16*112*112;i++) {
            ((float*)tensors[0].data())[i] = 0.5f;
        }
        ireq.set_input_tensors(0, tensors);
        ireq.set_output_tensor(0, out_tensor);
        ireq.infer();
        auto ten = ireq.get_output_tensor();

        for(int i=0; i<400;i++){
            std::cout << ((float*)ten.data())[i] << std::endl;
        }
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
