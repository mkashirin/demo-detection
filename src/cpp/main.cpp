#include <argparse/argparse.hpp>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

class YOLO26Detector {
  public:
    YOLO26Detector(const std::string &model_path) {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO26_Inference");
        Ort::SessionOptions session_options;

        // Optimized for RTX 5060 (CUDA)
        try {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (...) {
            std::cout << "CUDA Provider failed, using CPU." << std::endl;
        }

        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );
        session = std::make_unique<Ort::Session>(
            env, model_path.c_str(), session_options
        );

        input_name = "images";
        output_name = "output0";
    }

    void process_frame(cv::Mat &frame, float conf_thresh = 0.35f) {
        const int imgsz = 1280;

        cv::Mat blob = cv::dnn::blobFromImage(
            frame,
            1.0 / 255.0,
            cv::Size(imgsz, imgsz),
            cv::Scalar(0, 0, 0),
            true,
            false
        );

        std::vector<int64_t> input_shape = {1, 3, imgsz, imgsz};
        auto memory_info
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            (float *)blob.data,
            blob.total(),
            input_shape.data(),
            input_shape.size()
        );

        // 3. Inference
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1
        );

        float *raw_output = output_tensors[0].GetTensorMutableData<float>();

        for (int i = 0; i < 300; ++i) {
            float *det = raw_output + (i * 6);
            float score = det[4];
            if (score < conf_thresh)
                continue;

            float x1 = det[0] * frame.cols / imgsz;
            float y1 = det[1] * frame.rows / imgsz;
            float x2 = det[2] * frame.cols / imgsz;
            float y2 = det[3] * frame.rows / imgsz;
            int cls = (int)det[5];

            cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            std::string label = (cls == 0 ? "car" : "truck") + std::string(" ")
                + std::to_string(score).substr(0, 4);
            cv::putText(
                frame,
                label,
                cv::Point(x1, y1 - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(255, 255, 255),
                2
            );
        }
    }

  private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    const char *input_name;
    const char *output_name;
};

int main(int argc, char **argv) {
    argparse::ArgumentParser program("demo_detection", "1.0");

    program.add_argument("--model_path")
        .required()
        .help("Path to the YOLO26 ONNX model file");

    program.add_argument("--input_data")
        .required()
        .help("Path to the input video file (.mp4, .avi)");

    program.add_argument("--output_data")
        .required()
        .help("Path where the annotated video will be saved");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string model_path = program.get<std::string>("--model_path");
    std::string input_path = program.get<std::string>("--input_data");
    std::string output_path = program.get<std::string>("--output_data");

    try {
        YOLO26Detector detector(model_path);
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened())
            throw std::runtime_error("Failed to open input video.");

        cv::VideoWriter writer(
            output_path,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            cap.get(cv::CAP_PROP_FPS),
            cv::Size(
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT)
            )
        );

        cv::Mat frame;
        while (cap.read(frame)) {
            detector.process_frame(frame);
            writer.write(frame);
        }
        std::cout << "Successfully processed video. Output: " << output_path
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Execution Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
