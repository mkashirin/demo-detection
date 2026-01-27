#include <argparse/argparse.hpp>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp> // Необходимо для NMSBoxes
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;

class YOLO26Detector {
  public:
    YOLO26Detector(const std::string &model_path) {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO26_Inference");
        Ort::SessionOptions session_options;

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

    void process_frame(cv::Mat &frame, float conf_thresh, float iou_thresh) {
        const int imgsz = 1280;

        // 1. Препроцессинг
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

        // 2. Инференс
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1
        );

        float *raw_output = output_tensors[0].GetTensorMutableData<float>();

        // Контейнеры для результатов перед NMS
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        // 3. Сбор кандидатов (YOLO26 обычно выдает 300 предсказаний)
        for (int i = 0; i < 300; ++i) {
            float *det = raw_output + (i * 6);
            float score = det[4];

            if (score < conf_thresh)
                continue;

            // Масштабирование координат (исходя из того, что экспорт в пикселях
            // 0-1280)
            float x1 = det[0] * frame.cols / imgsz;
            float y1 = det[1] * frame.rows / imgsz;
            float x2 = det[2] * frame.cols / imgsz;
            float y2 = det[3] * frame.rows / imgsz;

            float w = x2 - x1;
            float h = y2 - y1;

            bboxes.push_back(
                cv::Rect(
                    static_cast<int>(x1),
                    static_cast<int>(y1),
                    static_cast<int>(w),
                    static_cast<int>(h)
                )
            );
            scores.push_back(score);
            class_ids.push_back(static_cast<int>(det[5]));
        }

        // 4. Выполнение NMS для фильтрации перекрывающихся боксов
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, conf_thresh, iou_thresh, indices);

        // 5. Отрисовка отфильтрованных результатов
        for (int idx : indices) {
            cv::Rect box = bboxes[idx];
            float score = scores[idx];
            int cls = class_ids[idx];

            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            std::string label = (cls == 0 ? "car" : "truck") + std::string(" ")
                + std::to_string(score).substr(0, 4);

            int baseLine;
            cv::Size labelSize = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine
            );
            // Фон для текста (опционально, для читаемости)
            cv::rectangle(
                frame,
                cv::Rect(
                    box.x,
                    box.y - labelSize.height - 5,
                    labelSize.width,
                    labelSize.height + 5
                ),
                cv::Scalar(0, 255, 0),
                cv::FILLED
            );

            cv::putText(
                frame,
                label,
                cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 0, 0),
                1
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

    program.add_argument("--model_path").required().help("Path to ONNX model");
    program.add_argument("--input_data").required().help("Path to input video");
    program.add_argument("--output_data")
        .required()
        .help("Path to output video");

    // Добавление параметров порога
    program.add_argument("--conf").default_value(0.5f).scan<'g', float>().help(
        "Confidence threshold"
    );

    program.add_argument("--iou").default_value(0.4f).scan<'g', float>().help(
        "IoU threshold for NMS"
    );

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    string model_path = program.get<string>("--model_path");
    string input_path = program.get<string>("--input_data");
    string output_path = program.get<string>("--output_data");
    float conf_v = program.get<float>("--conf");
    float iou_v = program.get<float>("--iou");

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
            detector.process_frame(frame, conf_v, iou_v);
            writer.write(frame);
        }
        std::cout << "Done! Output saved to: " << output_path << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Execution Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
