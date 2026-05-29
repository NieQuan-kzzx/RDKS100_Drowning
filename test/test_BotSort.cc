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
#include "BotSortAdapter.hpp"

// Google Flags 命令行参数定义
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/yolo11s_mot.hbm", "Path to BPU Quantized *.hbm model file");
DEFINE_string(test_img, "/home/sunrise/Desktop/test_bmp/1.jpg", "Path to load the test image.");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_mot.mp4", "Path to input video file. If set, video mode will be used.");
DEFINE_string(output_video, "botsort_mot.mp4", "Path to save processed output video.");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_mot.names", "Path to load ImageNet label mapping file.");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.7, "IoU threshold for Non-Maximum Suppression.");

int main(int argc, char **argv) {
    std::cout << "=== YOLO11 + BotSortAdapter 完整跟踪系统 ===" << std::endl;

    // 解析命令行参数
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    try {
        // 加载模型
        std::cout << "Loading YOLO11 model from: " << FLAGS_model_path << std::endl;
        YOLO11 yolo11(FLAGS_model_path);
        std::cout << "Model loaded successfully!" << std::endl;

        // 加载类别名称
        std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
        std::cout << "Loaded " << class_names.size() << " class names" << std::endl;

        // 初始化 BotSortAdapter 跟踪器
        BotSortAdapter tracker(
            0.45f,   // track_high_thresh
            0.1f,   // track_low_thresh
            0.6f,   // new_track_thresh
            0.75f,   // match_thresh
            30,     // track_buffer
            60      // max_age
        );

        // 视频处理模式
        if (!FLAGS_input_video.empty()) {
            std::cout << "Opening video: " << FLAGS_input_video << std::endl;
            cv::VideoCapture cap(FLAGS_input_video);
            if (!cap.isOpened()) {
                std::cerr << "Failed to open input video: " << FLAGS_input_video << std::endl;
                return -1;
            }

            // 获取视频参数
            int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double video_fps = cap.get(cv::CAP_PROP_FPS);
            if (video_fps <= 0) video_fps = 25.0;

            std::cout << "Video info: " << frame_w << "x" << frame_h << " @ " << video_fps << " FPS" << std::endl;

            // 初始化视频写入器
            cv::VideoWriter writer;
            int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
            if (!FLAGS_output_video.empty()) {
                writer.open(FLAGS_output_video, fourcc, video_fps, cv::Size(frame_w, frame_h));
                if (!writer.isOpened()) {
                    std::cerr << "Warning: could not open VideoWriter for output '" << FLAGS_output_video << "'" << std::endl;
                } else {
                    std::cout << "Output video will be saved to: " << FLAGS_output_video << std::endl;
                }
            }

            cv::Mat frame;
            int frame_idx = 0;

            // 性能统计变量
            std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
            std::deque<double> fps_history;
            const int fps_history_size = 30; // 30帧平均

            std::cout << "\nStarting processing..." << std::endl;
            while (cap.read(frame)) {
                frame_idx++;
                if (frame.empty()) break;

                // 计时开始
                auto frame_start = std::chrono::steady_clock::now();

                int img_w = frame.cols;
                int img_h = frame.rows;

                // YOLO11 目标检测
                yolo11.pre_process(frame);
                yolo11.infer();
                auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, img_w, img_h);

                // BotSortAdapter 目标跟踪 - 传入 frame 用于计算 CMC (相机运动补偿)
                auto tracks = tracker.track(detections, frame);

                // 计算FPS
                auto frame_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> frame_duration = frame_end - frame_start;
                double current_fps = 1.0 / frame_duration.count();

                // FPS平滑
                fps_history.push_back(current_fps);
                if (fps_history.size() > fps_history_size) {
                    fps_history.pop_front();
                }
                double avg_fps = 0.0;
                for (double f : fps_history) avg_fps += f;
                avg_fps /= fps_history.size();

                // 使用适配器绘制结果
                BotSortAdapter::draw_tracks(frame, tracks);

                // 绘制FPS
                std::ostringstream fps_stream;
                fps_stream << "FPS: " << std::fixed << std::setprecision(1) << avg_fps;
                cv::putText(frame, fps_stream.str(), cv::Point(frame.cols - 120, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                // 写入视频
                if (writer.isOpened()) {
                    writer.write(frame);
                }

                // 显示进度
                if (frame_idx % 30 == 0) {
                    std::cout << "Frame: " << frame_idx << " | Active tracks: " << tracks.size() << " | FPS: " << std::fixed << std::setprecision(1) << avg_fps << std::endl;
                }
            }

            // 输出最终统计信息
            auto end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> total_elapsed = end_time - start_time;
            double total_avg_fps = frame_idx / total_elapsed.count();

            std::cout << "\n=== 处理完成 ===" << std::endl;
            std::cout << "总帧数: " << frame_idx << std::endl;
            std::cout << "总耗时: " << std::fixed << std::setprecision(2) << total_elapsed.count() << " 秒" << std::endl;
            std::cout << "平均FPS: " << std::fixed << std::setprecision(1) << total_avg_fps << std::endl;

            if (writer.isOpened()) {
                std::cout << "结果视频已保存到: " << FLAGS_output_video << std::endl;
            }

            cap.release();
            if (writer.isOpened()) writer.release();

            return 0;
        }

        // 单张图像处理模式
        if (!FLAGS_test_img.empty()) {
            std::cout << "Processing single image: " << FLAGS_test_img << std::endl;
            cv::Mat image = cv::imread(FLAGS_test_img);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << FLAGS_test_img << std::endl;
                return -1;
            }

            // YOLO11 检测
            yolo11.pre_process(image);
            yolo11.infer();
            auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, image.cols, image.rows);

            // BotSortAdapter 跟踪 - 单张图同样需要传入 image
            auto tracks = tracker.track(detections, image);

            // 使用适配器绘制结果
            BotSortAdapter::draw_tracks(image, tracks);

            // 保存结果
            std::string output_img_path = "result_image.jpg";
            cv::imwrite(output_img_path, image);
            std::cout << "Result saved to: " << output_img_path << std::endl;

            return 0;
        }

        std::cerr << "Error: Please specify either --input_video or --test_img" << std::endl;
        return -1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
