#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <deque>
#include <map>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"
#include "BYTETracker.hpp"

// 命令行参数定义
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/s_PC-YOLO_UnderSurface.hbm", "BPU模型文件路径");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/swim_test.mp4", "输入视频路径");
DEFINE_string(output_video, "swim_result.mp4", "输出保存路径");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_swim.names", "标签文件路径");
DEFINE_double(score_thres, 0.25, "置信度阈值");
DEFINE_double(nms_thres, 0.7, "NMS阈值");

int main(int argc, char **argv) {
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 1. 初始化模型与标签
    YOLO11 yolo11 = YOLO11(FLAGS_model_path);
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    // 2. 初始化追踪器与计时器
    BYTETracker tracker;
    std::map<int, std::chrono::steady_clock::time_point> underwater_timers;

    // 3. 视频流准备
    cv::VideoCapture cap(FLAGS_input_video);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << FLAGS_input_video << std::endl;
        return -1;
    }

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 25.0;

    cv::VideoWriter writer;
    if (!FLAGS_output_video.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(FLAGS_output_video, fourcc, video_fps, cv::Size(frame_w, frame_h));
    }

    cv::Mat frame;
    int frame_idx = 0;
    std::deque<double> fps_history;

    std::cout << "开始处理视频..." << std::endl;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_idx++;
        auto frame_start = std::chrono::steady_clock::now();

        // --- A. 目标检测 (BPU 推理) ---
        yolo11.pre_process(frame);
        yolo11.infer();
        auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, frame.cols, frame.rows);

        // --- B. 数据转换 ---
        std::vector<Object> objects;
        for (auto &det : detections) {
            Object o;
            float x1 = det.bbox[0];
            float y1 = det.bbox[1];
            float x2 = det.bbox[2];
            float y2 = det.bbox[3];
            o.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
            o.label = det.class_id; // 0: surface, 1: underwater
            o.prob = det.score;
            objects.push_back(o);
        }

        // --- C. 目标追踪 (ByteTrack) ---
        auto tracks = tracker.update(objects);

        // --- D. 溺水计时与逻辑判断 ---
        bool global_warning = false;
        for (auto &t : tracks) {
            if (!t.is_activated) continue;

            int label_id =-1;
            float max_iou = 0.0;
            for (auto &obj : objects) {
                // 计算两个矩形的交集
                cv::Rect2f track_rect(t.tlwh[0], t.tlwh[1], t.tlwh[2], t.tlwh[3]);
                float intersection_area = (track_rect & obj.rect).area();
                float union_area = track_rect.area() + obj.rect.area() - intersection_area;
                float iou = intersection_area / union_area;

                if (iou > max_iou) {
                    max_iou = iou;
                    label_id = obj.label; // 拿到检测结果里的标签
                }
            }

            if (label_id == -1) {
                label_id =1;
            }

            // 获取坐标并防止越界
            int x = std::max(0, (int)t.tlwh[0]);
            int y = std::max(0, (int)t.tlwh[1]);
            int w = std::min(frame.cols - x, (int)t.tlwh[2]);
            int h = std::min(frame.rows - y, (int)t.tlwh[3]);
            cv::Rect box(x, y, w, h);

            // int label_id = t.label;
            std::string label_text = (label_id == 0) ? "surface" : "underwater";
            cv::Scalar col = (label_id == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255); // 绿色 vs 橙色

            // 核心：计时逻辑
            if (label_id == 1) { // underwater
                if (underwater_timers.find(t.track_id) == underwater_timers.end()) {
                    underwater_timers[t.track_id] = std::chrono::steady_clock::now();
                } else {
                    auto now = std::chrono::steady_clock::now();
                    auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - underwater_timers[t.track_id]).count();
                    if (sec >= 3) {
                        label_text = "DROWNING!";
                        col = cv::Scalar(0, 0, 255); // 红色
                        global_warning = true;
                        // 终端实时输出告警
                        if (frame_idx % 10 == 0) 
                            std::cout << "[WARNING] ID:" << t.track_id << " 溺水超过 " << sec << "s!" << std::endl;
                    }
                }
            } else {
                underwater_timers.erase(t.track_id); // 回到水面则重置
            }

            // 绘制框和标签
            cv::rectangle(frame, box, col, 2);
            std::string display_info = "ID:" + std::to_string(t.track_id) + " " + label_text;
            cv::putText(frame, display_info, cv::Point(x, std::max(20, y - 5)), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
        }

        // --- E. 绘制 HUD 信息 ---
        // 1. 左上角全局警告
        if (global_warning) {
            cv::putText(frame, "WARNING: DROWNING DETECTED", cv::Point(20, 100),
                        cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
        }

        // 2. 右上角计数统计
        std::string count_info = "Now:" + std::to_string(tracks.size()) + " Timers:" + std::to_string(underwater_timers.size());
        cv::putText(frame, count_info, cv::Point(frame.cols - 300, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // 3. FPS 显示
        auto frame_end = std::chrono::steady_clock::now();
        double cur_fps = 1.0 / std::chrono::duration<double>(frame_end - frame_start).count();
        fps_history.push_back(cur_fps);
        if (fps_history.size() > 10) fps_history.pop_front();
        double avg_fps = 0;
        for (double f : fps_history) avg_fps += f;
        avg_fps /= fps_history.size();
        cv::putText(frame, "FPS: " + std::to_string((int)avg_fps), cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 保存视频
        if (writer.isOpened()) writer.write(frame);

        // 内存清理：移除已丢失目标的计时器
        if (frame_idx % 30 == 0) {
            for (auto it = underwater_timers.begin(); it != underwater_timers.end(); ) {
                bool exists = false;
                for (auto &t : tracks) if (t.track_id == it->first) { exists = true; break; }
                if (!exists) it = underwater_timers.erase(it);
                else ++it;
            }
            std::cout << "已处理帧数: " << frame_idx << std::endl;
        }
    }

    std::cout << "处理完成。视频已保存至: " << FLAGS_output_video << std::endl;
    return 0;
}