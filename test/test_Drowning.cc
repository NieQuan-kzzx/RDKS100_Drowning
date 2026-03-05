#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <deque>
#include <iomanip>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"
#include "BYTETracker.hpp"

// 参数定义
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/drowning_TwoSelect.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(test_img, "/home/sunrise/Desktop/test_bmp/1.jpg",
              "Path to load the test image.");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/swim_test.mp4",
              "Path to input video file. If set, video mode will be used.");
DEFINE_string(output_video, "result.mp4",
              "Path to save processed output video.");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_swim.names",
              "Path to load ImageNet label mapping file.");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.7, "IoU threshold for Non-Maximum Suppression.");

int main(int argc, char **argv)
{
    // 解析命令行参数
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 导入模型
    YOLO11 yolo11 = YOLO11(FLAGS_model_path);

    // 导入类别名称
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    // 初始化 ByteTrack
    BYTETracker tracker;
    
    // 视频处理模式
    if (!FLAGS_input_video.empty())
    {
        cv::VideoCapture cap(FLAGS_input_video);
        if (!cap.isOpened())
        {
            std::cerr << "Failed to open input video: " << FLAGS_input_video << std::endl;
            return -1;
        }

        int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double video_fps = cap.get(cv::CAP_PROP_FPS);
        if (video_fps <= 0) video_fps = 25.0;

        // 视频写入初始化
        cv::VideoWriter writer;
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        if (!FLAGS_output_video.empty())
            writer.open(FLAGS_output_video, fourcc, video_fps, cv::Size(frame_w, frame_h));

        cv::Mat frame;
        int frame_idx = 0;
        int max_id = 0;
        
        // FPS 相关
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        std::deque<double> fps_history;
        const int fps_history_size = 10;
        
        while (cap.read(frame))
        {
            frame_idx++;
            if (frame.empty()) break;

            auto frame_start = std::chrono::steady_clock::now();

            int img_w = frame.cols;
            int img_h = frame.rows;

            // 1. 目标检测
            yolo11.pre_process(frame);
            yolo11.infer();
            auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, img_w, img_h);
            
            // 2. 准备跟踪输入
            std::vector<Object> objects;
            for (auto &det : detections)
            {
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
            
            // 3. 更新跟踪器
            auto tracks = tracker.update(objects);

            // 4. 计算并平滑 FPS
            auto frame_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> frame_duration = frame_end - frame_start;
            double current_fps = 1.0 / frame_duration.count();
            fps_history.push_back(current_fps);
            if (fps_history.size() > fps_history_size) fps_history.pop_front();
            double avg_fps = 0;
            for (double f : fps_history) avg_fps += f;
            avg_fps /= fps_history.size();
            
            // 5. 绘制结果
            for (auto &t : tracks)
            {
                if (!t.is_activated) continue;

                // 获取跟踪框坐标
                cv::Rect track_rect((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);

                // --- 修复点：通过 IoU 找回 label 名称 ---
                std::string label_name = "unknown";
                int best_class_id = -1;
                float max_iou = 0.0f;

                for (const auto& obj : objects) {
                    cv::Rect inter = track_rect & cv::Rect((int)obj.rect.x, (int)obj.rect.y, (int)obj.rect.width, (int)obj.rect.height);
                    if (inter.area() > 0) {
                        float iou = (float)inter.area() / (track_rect.area() + (int)obj.rect.area() - inter.area());
                        if (iou > max_iou) {
                            max_iou = iou;
                            best_class_id = obj.label;
                        }
                    }
                }

                if (best_class_id >= 0 && best_class_id < (int)class_names.size()) {
                    label_name = class_names[best_class_id];
                }
                // ----------------------------------------

                // 分配颜色并绘制
                cv::Scalar col = tracker.get_color(t.track_id);
                cv::rectangle(frame, track_rect, col, 2);
                
                std::string label_text = "ID:" + std::to_string(t.track_id) + " " + label_name;

                // 绘制背景框和文字
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                cv::rectangle(frame, 
                              cv::Point(track_rect.x, track_rect.y - text_size.height - 10),
                              cv::Point(track_rect.x + text_size.width, track_rect.y),
                              col, -1);

                cv::putText(frame, label_text, cv::Point(track_rect.x, track_rect.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

                if (t.track_id > max_id) max_id = t.track_id;
            }

            // 绘制 HUD 信息
            std::ostringstream fps_ss, count_ss;
            fps_ss << "FPS: " << std::fixed << std::setprecision(1) << avg_fps;
            count_ss << "Now:" << objects.size() << "  Total:" << max_id;

            cv::putText(frame, fps_ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, count_ss.str(), cv::Point(frame.cols - 240, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

            if (writer.isOpened()) writer.write(frame);

            if (frame_idx % 30 == 0) {
                std::cout << "Processed " << frame_idx << " frames..." << std::endl;
            }
        }

        std::cout << "\n======= Processing Complete =======" << std::endl;
        if (writer.isOpened()) std::cout << "Saved video to: " << FLAGS_output_video << std::endl;
        return 0;
    }

    std::cerr << "Error: No input video specified." << std::endl;
    return -1;
}