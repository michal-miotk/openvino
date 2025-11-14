#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include <fstream>
#include <openvino/core/version.hpp>

 
int main() {
    ov::Core core;
 
    // Ścieżka do modelu
    std::string model_path = "/home/gta/1185113_03092025112025_ic3_v7_1_bgr_b6_h736_w864.onnx";  // lub "model.xml"
 
    // Wczytaj i skompiluj model
   
    // Ustawienie katalogu cache
    std::string cache_dir = "./cache_2025new";
    std::cout <<  OPENVINO_VERSION_MAJOR << "_______" << OPENVINO_VERSION_MINOR << std::endl;
    std::map<std::string, ov::Any> xx;
    xx["CACHE_DIR"] = cache_dir;
    xx[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SIZE;
    core.set_property("GPU", xx);
    std::map<std::string, ov::Any> yy;
    std::cout << ov::cache_mode.name() << " ov::cache_mode.name()" << std::endl;
    auto model = core.read_model(model_path, "", xx);
    auto compiled_model = core.compile_model(model, "GPU", xx);
    
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 5. Przygotuj dane wejściowe
    //ov::Tensor input_tensor = infer_request.get_input_tensor();
    // Załóżmy, że wejście to float32 i rozmiar [1, 3, 224, 224]
    //float* input_data = input_tensor.data<float>();
    // Wypełnij input_data danymi (np. z obrazu)

    // 6. Wykonaj inferencję
    infer_request.infer();
    std::cout << "done" << std::endl;
    return 0;
}