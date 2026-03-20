#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <deque>
#include <iomanip>
#include <map>
#include <algorithm>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"
#include "BYTETracker.hpp"

// --- 使用独立命名避免与 ByteTrack 冲突 ---
struct DrowningTrackState {
    std::deque<cv::Point2f> history_pos; 
    int underwater_count = 0;           
    bool is_drowned = false;            
};

class DrowningDetector {
public:
    DrowningDetector(float move_thr = 100.0f, int time_thr = 50)
        : move_threshold(move_thr), time_threshold(time_thr) {}

    void update(int track_id, cv::Point2f center, bool is_underwater) {
        auto& state = manager[track_id];
        state.history_pos.push_back(center);
        if (state.history_pos.size() > 50) state.history_pos.pop_front();

        float displacement = 100.0f; 
        if (state.history_pos.size() >= 30) {
            displacement = cv::norm(state.history_pos.back() - state.history_pos.front());
        }

        // 溺水判定逻辑：处于水下且位移极小
        if (is_underwater && displacement < move_threshold) {
            state.underwater_count++;
        } else {
            state.underwater_count = 0; 
            state.is_drowned = false;
        }

        if (state.underwater_count >= time_threshold) {
            state.is_drowned = true;
        }
    }

    bool isDrowned(int track_id) {
        auto it = manager.find(track_id);
        return (it != manager.end()) && it->second.is_drowned;
    }

    void clean(const std::vector<int>& active_ids) {
        for (auto it = manager.begin(); it != manager.end(); ) {
            if (std::find(active_ids.begin(), active_ids.end(), it->first) == active_ids.end()) {
                it = manager.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    std::map<int, DrowningTrackState> manager;
    float move_threshold;
    int time_threshold;
};

// 参数定义
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/yolo11n_detect_nashe_640x640_nv12.hbm", "Model file");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/1.mp4", "Input video");
DEFINE_string(output_video, "drowning_result_2.mp4", "Output saved path");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_swim.names", "Labels");
DEFINE_double(score_thres, 0.25, "Score thres");
DEFINE_double(nms_thres, 0.7, "NMS thres");

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 初始化模型和算法
    YOLO11 yolo11(FLAGS_model_path);
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    BYTETracker tracker;
    DrowningDetector drowning_checker(12.0f, 125); // 约5秒判定(25fps)

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

    std::cout << "Starting processing (No GUI)..." << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    auto total_start = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_idx++;

        // 1. 推理与检测
        yolo11.pre_process(frame);
        yolo11.infer();
        auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, frame_w, frame_h);

        // 2. 准备跟踪输入
        std::vector<Object> objects;
        for (auto &det : detections) {
            Object o;
            o.rect = cv::Rect_<float>(det.bbox[0], det.bbox[1], det.bbox[2]-det.bbox[0], det.bbox[3]-det.bbox[1]);
            o.label = det.class_id;
            o.prob = det.score;
            objects.push_back(o);
        }

        // 3. 更新跟踪器
        auto tracks = tracker.update(objects);
        std::vector<int> active_ids;
        bool global_drowning_alert = false;

        // 4. 处理每一个跟踪对象
        for (auto &t : tracks) {
            if (!t.is_activated) continue;
            active_ids.push_back(t.track_id);

            cv::Rect track_rect((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);
            bool is_underwater = false;
            std::string label_name = "unknown";

            // IoU 匹配回溯 YOLO 原始类别
            for (const auto& obj : objects) {
                cv::Rect det_rect((int)obj.rect.x, (int)obj.rect.y, (int)obj.rect.width, (int)obj.rect.height);
                cv::Rect inter = track_rect & det_rect;
                if (inter.area() > 0) {
                    float iou = (float)inter.area() / (track_rect.area() + det_rect.area() - inter.area());
                    if (iou > 0.6) { // 匹配阈值
                        if (obj.label < class_names.size()) {
                            label_name = class_names[obj.label];
                            if (label_name.find("under") != std::string::npos) is_underwater = true;
                        }
                    }
                }
            }

            // 更新溺水判定状态
            cv::Point2f center(t.tlwh[0] + t.tlwh[2]/2, t.tlwh[1] + t.tlwh[3]/2);
            drowning_checker.update(t.track_id, center, is_underwater);

            // 绘制结果到离线帧
            bool drowned = drowning_checker.isDrowned(t.track_id);
            if (drowned) global_drowning_alert = true;

            cv::Scalar color = drowned ? cv::Scalar(0, 0, 255) : tracker.get_color(t.track_id);
            cv::rectangle(frame, track_rect, color, drowned ? 4 : 2);

            std::string info = "ID:" + std::to_string(t.track_id) + " " + label_name;
            
            cv::putText(frame, info, cv::Point(track_rect.x, track_rect.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        if (global_drowning_alert) {
            cv::Mat overlay = frame.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(frame.cols, 80), cv::Scalar(0, 0, 255), -1);
            cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);

            std::string warn_text = "WARNING: DROWNING DETECTED!";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(warn_text, cv::FONT_HERSHEY_DUPLEX, 1.5, 3, &baseline);
            
            cv::Point text_org((frame.cols - text_size.width) / 2, 55);
            
            cv::putText(frame, warn_text, text_org + cv::Point(2, 2), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 0), 3);
            cv::putText(frame, warn_text, text_org, cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
        }

        // 清理缓存
        drowning_checker.clean(active_ids);

        // 写入视频文件
        if (writer.isOpened()) writer.write(frame);

        if (frame_idx % 50 == 0) {
            std::cout << "Processed frames: " << frame_idx << std::endl;
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = total_end - total_start;
    
    std::cout << "\n--- Processing Finished ---" << std::endl;
    std::cout << "Total frames: " << frame_idx << std::endl;
    std::cout << "Time spent: " << diff.count() << " seconds" << std::endl;
    std::cout << "Average Speed: " << (frame_idx / diff.count()) << " FPS" << std::endl;
    std::cout << "Result saved to: " << FLAGS_output_video << std::endl;

    return 0;
}