#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"

// 参数定义
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11s.hbm", "Model file");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/1.mp4", "Input video");
DEFINE_string(output_video, "detection_result.mp4", "Output saved path");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes.names", "Labels");
DEFINE_double(score_thres, 0.25, "Score thres");
DEFINE_double(nms_thres, 0.7, "NMS thres");

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 1. 初始化模型
    YOLO11 yolo11(FLAGS_model_path);
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    cv::VideoCapture cap(FLAGS_input_video);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << FLAGS_input_video << std::endl;
        return -1;
    }

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(FLAGS_output_video, cv::VideoWriter::fourcc('m','p','4','v'), 
                           fps, cv::Size(frame_w, frame_h));

    std::cout << "Starting Detection (No Tracking)..." << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    auto total_start = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_idx++;

        // 2. 推理与检测
        yolo11.pre_process(frame);
        yolo11.infer();
        // 获取当前帧的所有检测结果
        auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, frame_w, frame_h);

        // 3. 遍历检测结果并绘制
        for (const auto &det : detections) {
            // det.bbox 格式通常为 [x1, y1, x2, y2]
            cv::Rect rect((int)det.bbox[0], (int)det.bbox[1], 
                          (int)(det.bbox[2] - det.bbox[0]), 
                          (int)(det.bbox[3] - det.bbox[1]));

            // 获取标签名称
            std::string label_name = (det.class_id < class_names.size()) ? class_names[det.class_id] : "unknown";
            
            // 设定颜色（可以根据类别固定颜色，或者用绿色）
            cv::Scalar color = cv::Scalar(0, 255, 0); 
            if (label_name.find("under") != std::string::npos) {
                color = cv::Scalar(0, 165, 255); // 橙色表示水下
            }

            // 绘制矩形框
            cv::rectangle(frame, rect, color, 2);

            // 绘制标签和置信度
            std::string info = label_name + " " + std::to_string((int)(det.score * 100)) + "%";
            cv::putText(frame, info, cv::Point(rect.x, rect.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }

        // 4. 写入视频文件
        if (writer.isOpened()) writer.write(frame);

        if (frame_idx % 50 == 0) {
            std::cout << "Processed frames: " << frame_idx << std::endl;
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = total_end - total_start;
    
    std::cout << "\n--- Detection Finished ---" << std::endl;
    std::cout << "Total frames: " << frame_idx << std::endl;
    std::cout << "Average Speed: " << (frame_idx / diff.count()) << " FPS" << std::endl;

    return 0;
}