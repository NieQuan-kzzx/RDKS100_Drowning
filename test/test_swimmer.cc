#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <deque>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"
#include "BYTETracker.hpp"

// --- 配置参数（宏定义模式） ---
#define MODEL_PATH "/home/sunrise/Desktop/RDKS100_Drowning/models/swimmer.hbm"
#define TEST_VIDEO_PATH "/home/sunrise/Desktop/RDKS100_Drowning/tem/swim_test.mp4"
#define OUTPUT_VIDEO_PATH "swimmer_result.mp4"
#define SCORE_THRESHOLD 0.25  // 置信度阈值
#define NMS_THRESHOLD 0.7     // 非极大值抑制阈值
#define FONT_SIZE 0.6
#define FONT_THICKNESS 2
#define LINE_SIZE 2

// 类别名称定义
std::vector<std::string> object_names = {"swimmer"};

float calculate_iou(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x2 <= x1 || y2 <= y1) return 0.0;
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    return intersection / (area1 + area2 - intersection);
}

int main(int argc, char **argv)
{
    // 1. 初始化模型
    YOLO11 yolo11 = YOLO11(MODEL_PATH);

    // 2. 初始化跟踪器
    BYTETracker tracker;
    
    // 3. 打开视频流
    cv::VideoCapture cap(TEST_VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << TEST_VIDEO_PATH << std::endl;
        return -1;
    }

    // 获取视频参数
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 25.0;

    // 4. 初始化视频写入器
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    writer.open(OUTPUT_VIDEO_PATH, fourcc, video_fps, cv::Size(frame_w, frame_h));

    cv::Mat frame;
    int frame_idx = 0;
    int global_max_id = 0; // 全局计数器，放在循环外
    
    // FPS 平滑处理
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::deque<double> fps_history;
    const int fps_history_size = 10;
    
    std::cout << "Starting processing..." << std::endl;

    while (cap.read(frame))
    {
        frame_idx++;
        if (frame.empty()) break;

        auto frame_start = std::chrono::steady_clock::now();

        // --- 目标检测 ---
        yolo11.pre_process(frame);
        yolo11.infer();
        auto detections = yolo11.post_process(SCORE_THRESHOLD, NMS_THRESHOLD, frame.cols, frame.rows);
        
        // --- 格式转换 (YOLO -> ByteTrack) ---
        std::vector<Object> objects;
        for (auto &det : detections) {
            Object o;
            float x1 = det.bbox[0];
            float y1 = det.bbox[1];
            float x2 = det.bbox[2];
            float y2 = det.bbox[3];
            o.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
            o.label = det.class_id;
            o.prob = det.score;
            objects.push_back(o);
        }
        
        // --- 目标跟踪 ---
        auto tracks = tracker.update(objects);

        // --- FPS 计算 ---
        auto frame_end = std::chrono::steady_clock::now();
        double current_fps = 1.0 / std::chrono::duration<double>(frame_end - frame_start).count();
        fps_history.push_back(current_fps);
        if (fps_history.size() > fps_history_size) fps_history.pop_front();
        double avg_fps = 0;
        for (double f : fps_history) avg_fps += f;
        avg_fps /= fps_history.size();
        
        // --- 绘制结果 ---
        for (auto &t : tracks)
        {
            if (!t.is_activated) continue;

            int recovered_label = 0;
            float max_iou = 0.0;
            cv::Rect track_rect((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);

            for (auto &det : detections) {
                // det.bbox = [x1, y1, x2, y2, score, class_id]
                cv::Rect det_rect((int)det.bbox[0], (int)det.bbox[1], 
                                (int)(det.bbox[2] - det.bbox[0]), (int)(det.bbox[3] - det.bbox[1]));
                
                float iou = calculate_iou(track_rect, det_rect);
                if (iou > max_iou && iou > 0.4) {
                    max_iou = iou;
                    recovered_label = (int)det.class_id; // 拿到检测结果里的标签
                }
            }

            // 获取颜色：从自定义 rdk_colors 中循环取色
            cv::Scalar col = ::rdk_colors[t.track_id % ::rdk_colors.size()];
            
            // 绘制矩形框
            cv::Rect box(cv::Point((int)t.tlwh[0], (int)t.tlwh[1]), cv::Size((int)t.tlwh[2], (int)t.tlwh[3]));
            cv::rectangle(frame, box, col, LINE_SIZE);
            
            // 拼接标签文字: "swimmer: 1"
            std::string label_name = (recovered_label < object_names.size()) 
                             ? object_names[recovered_label] 
                             : "unknown";
            std::string display_txt = label_name + ": " + std::to_string(t.track_id);
            
            // 绘制文字背景以便看清
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(display_txt, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_THICKNESS, &baseline);
            cv::rectangle(frame, cv::Rect(box.x, box.y - text_size.height - 5, text_size.width, text_size.height + 5), col, -1);
            cv::putText(frame, display_txt, cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(255, 255, 255), FONT_THICKNESS);

            // 更新全局计数
            if (t.track_id > global_max_id) global_max_id = t.track_id;
        }

        // --- 绘制 HUD (右上角统计) ---
        std::string count_info = "Now: " + std::to_string(tracks.size()) + "  Total: " + std::to_string(global_max_id);
        cv::putText(frame, count_info, cv::Point(frame.cols - 250, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // 绘制 FPS
        std::string fps_txt = "FPS: " + std::to_string((int)avg_fps);
        cv::putText(frame, fps_txt, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 保存视频
        if (writer.isOpened()) writer.write(frame);

        if (frame_idx % 50 == 0) std::cout << "Processing frame: " << frame_idx << std::endl;
    }

    // --- 结束统计 ---
    auto total_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(total_end - start_time).count();
    
    std::cout << "\n======= Final Report =======" << std::endl;
    std::cout << "Output: " << OUTPUT_VIDEO_PATH << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1) << (frame_idx / total_time) << std::endl;
    std::cout << "Total unique objects: " << global_max_id << std::endl;

    cap.release();
    writer.release();
    return 0;
}