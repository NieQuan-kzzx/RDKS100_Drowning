#include <iostream>
#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "PlogInitializer.h"
#include "Patchcore.h"
#include "YoloSeg.h"
#include "common_utils.hpp"

DEFINE_string(patchcore_model, "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm", "Patchcore模型路径");
DEFINE_string(seg_model, "/home/sunrise/Desktop/RDKS100_Drowning/models/Water_seg.hbm", "实例分割模型路径");
DEFINE_string(input, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_patchcore.jpg", "输入图像路径");
DEFINE_double(anomaly_threshold, 50.0, "Patchcore异常分数阈值");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_water_seg.names", "分割模型标签文件");
DEFINE_string(water_class, "water", "要匹配的water类别名称");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    PlogInitializer::getInstance().init(plog::verbose);
    PLOGI << "Starting WaterIngress dual-model test...";

    // 1. 读取输入
    cv::Mat frame = cv::imread(FLAGS_input);
    if (frame.empty()) {
        PLOGE << "Cannot read image: " << FLAGS_input;
        return -1;
    }
    cv::Mat display = frame.clone();

    // 2. 初始化 Patchcore
    Inf::Patchcore patchcore;
    if (!patchcore.init(FLAGS_patchcore_model)) {
        PLOGE << "Patchcore init failed!";
        return -1;
    }

    // 3. 初始化实例分割模型
    Inf::YoloSeg yolo_seg;
    if (!yolo_seg.init(FLAGS_seg_model)) {
        PLOGE << "YoloSeg init failed!";
        return -1;
    }
    std::vector<std::string> labels = load_linewise_labels(FLAGS_label_file);
    yolo_seg.setLabels(labels);

    // 4. 查找 water 类别对应的 class_id
    int water_class_id = -1;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == FLAGS_water_class) {
            water_class_id = i;
            break;
        }
    }
    if (water_class_id < 0) {
        PLOGW << "Water class '" << FLAGS_water_class << "' not found in labels, using class_id 0";
        water_class_id = 0;
    }

    // 5. 执行 Patchcore 推理
    PLOGI << "Running Patchcore inference...";
    auto patchcore_results = patchcore.run(display);
    float anomaly_score = patchcore_results.empty() ? 0.0f : patchcore_results[0].score;
    PLOGI << "Patchcore anomaly score: " << anomaly_score;

    // 6. 绘制 Patchcore 热力图
    patchcore.draw(display, patchcore_results);

    // 7. 执行实例分割推理
    PLOGI << "Running instance segmentation...";
    auto seg_results = yolo_seg.run(display);

    // 8. 绘制分割结果
    yolo_seg.draw(display, seg_results);

    // 9. 双重判定逻辑
    bool has_water = false;
    for (const auto& det : seg_results) {
        if (det.class_id == water_class_id) {
            has_water = true;
            cv::rectangle(display, det.rect, cv::Scalar(255, 0, 0), 3);
            std::string label = "WATER detected (ID:" + std::to_string(det.track_id) + ")";
            cv::putText(display, label, cv::Point(det.rect.x, det.rect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        }
    }

    bool water_ingress = (anomaly_score > FLAGS_anomaly_threshold) && has_water;

    // 10. 绘制报警
    if (water_ingress) {
        cv::Mat overlay = display.clone();
        cv::rectangle(overlay, cv::Rect(0, 0, display.cols, 80), cv::Scalar(0, 0, 255), -1);
        cv::addWeighted(overlay, 0.4, display, 0.6, 0, display);

        std::string warn_text = "ALARM: WATER INGRESS DROWNING!";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(warn_text, cv::FONT_HERSHEY_DUPLEX, 1.5, 3, &baseline);
        cv::Point text_org((display.cols - text_size.width) / 2, 55);
        cv::putText(display, warn_text, text_org + cv::Point(2, 2),
                    cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 0), 3);
        cv::putText(display, warn_text, text_org,
                    cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
    }

    // 11. 显示状态信息
    std::string info = "Patchcore Score: " + std::to_string(anomaly_score).substr(0, 6)
                     + " | Threshold: " + std::to_string(FLAGS_anomaly_threshold)
                     + " | Water: " + (has_water ? "YES" : "NO")
                     + " | Alarm: " + (water_ingress ? "TRIGGERED" : "NORMAL");
    cv::putText(display, info, cv::Point(30, display.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // 12. 保存结果
    std::string out_path = "water_ingress_result.jpg";
    cv::imwrite(out_path, display);
    PLOGI << "Result saved to: " << out_path;
    PLOGI << "Anomaly Score: " << anomaly_score;
    PLOGI << "Water detected: " << (has_water ? "YES" : "NO");
    PLOGI << "Water Ingress Alarm: " << (water_ingress ? "TRIGGERED" : "NORMAL");
    PLOGI << "Test finished.";

    return 0;
}
