#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "PlogInitializer.h"
#include "Patchcore.h"
#include "YoloSeg.h"
#include "common_utils.hpp"

DEFINE_string(patchcore_model, "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm", "Patchcore模型路径");
DEFINE_string(seg_model, "/home/sunrise/Desktop/RDKS100_Drowning/models/Water_seg.hbm", "实例分割模型路径");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_water.mp4", "输入视频路径");
DEFINE_string(output_video, "water_ingress_result.mp4", "输出视频路径");
DEFINE_double(anomaly_threshold, 50.0, "Patchcore异常分数阈值");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_water_seg.names", "分割模型标签文件");
DEFINE_string(water_class, "water", "要匹配的water类别名称");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    PlogInitializer::getInstance().init(plog::verbose);
    PLOGI << "Starting WaterIngress video test...";

    // 1. 打开视频
    cv::VideoCapture cap(FLAGS_input_video);
    if (!cap.isOpened()) {
        PLOGE << "Cannot open video: " << FLAGS_input_video;
        return -1;
    }

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    PLOGI << "Video: " << frame_w << "x" << frame_h << " @" << fps << "fps, " << total_frames << " frames";

    cv::VideoWriter writer(FLAGS_output_video,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(frame_w, frame_h));

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

    // 4. 查找 water 类别
    int water_class_id = -1;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == FLAGS_water_class) {
            water_class_id = i;
            break;
        }
    }
    if (water_class_id < 0) {
        PLOGW << "Water class not found in labels, using class_id 0";
        water_class_id = 0;
    }

    // 5. 逐帧处理
    cv::Mat frame;
    int frame_idx = 0;
    auto total_start = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_idx++;

        cv::Mat display = frame.clone();

        // 先完成所有推理（都在干净的 display 上）
        auto patchcore_results = patchcore.run(display);
        auto seg_results = yolo_seg.run(display);

        float anomaly_score = patchcore_results.empty() ? 0.0f : patchcore_results[0].score;

        // 再画图（seg 先画 mask，patchcore 后画热力图 + 文字在顶层）
        yolo_seg.draw(display, seg_results);
        patchcore.draw(display, patchcore_results);

        // 双重判定
        bool has_water = false;
        for (const auto& det : seg_results) {
            if (det.class_id == water_class_id) {
                has_water = true;
                cv::rectangle(display, det.rect, cv::Scalar(255, 0, 0), 3);
                std::string label = "WATER (ID:" + std::to_string(det.track_id) + ")";
                cv::putText(display, label, cv::Point(det.rect.x, det.rect.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
            }
        }

        bool water_ingress = (anomaly_score > FLAGS_anomaly_threshold) && has_water;

        // 绘制报警
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

        // 状态信息
        std::string info = "Frame:" + std::to_string(frame_idx)
                         + " Score:" + std::to_string(anomaly_score).substr(0, 6)
                         + " Water:" + (has_water ? "Y" : "N")
                         + " Alarm:" + (water_ingress ? "YES" : "NO");
        cv::putText(display, info, cv::Point(30, display.rows - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        writer.write(display);

        if (frame_idx % 50 == 0) {
            PLOGI << "Processed: " << frame_idx << "/" << total_frames;
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = total_end - total_start;

    PLOGI << "--- Processing Finished ---";
    PLOGI << "Total frames: " << frame_idx;
    PLOGI << "Time spent: " << diff.count() << " seconds";
    PLOGI << "Average Speed: " << (frame_idx / diff.count()) << " FPS";
    PLOGI << "Result saved to: " << FLAGS_output_video;

    return 0;
}
