#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <deque>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"
#include "BYTETracker.hpp"

// Google Flags库，#include "gflags/gflags.h"。
// 这里的 DEFINE_string 是 gflags 库（Google Flags）中定义命令行参数的一个宏。
// 参数分别为：变量名，默认参数，说明文字
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11s.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(test_img, "/home/sunrise/Desktop/test_bmp/1.jpg",
              "Path to load the test image.");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/1.mp4",
              "Path to input video file. If set, video mode will be used.");
DEFINE_string(output_video, "result.mp4",
              "Path to save processed output video.");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes.names",
              "Path to load ImageNet label mapping file.");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.7, "IoU threshold for Non-Maximum Suppression.");

int main(int argc, char **argv)
{
  // Parse command-line arguments
  // 解析命令行参数
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Load model
  // 导入模型
  YOLO11 yolo11 = YOLO11(FLAGS_model_path);

  // Load class names
  // 导入类别名称,也可自行写，替换这里的txt文件
  std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

  // Initialize ByteTrack
  BYTETracker tracker;
  
  // 视频处理模式
  if (!FLAGS_input_video.empty())
  {
    // 读取视频，改为0即为调用摄像头(未尝试，在python下是0)
    cv::VideoCapture cap(FLAGS_input_video);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open input video: " << FLAGS_input_video << std::endl;
        return -1;
    }
    // 获取视频参数
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);

    if (video_fps <= 0) video_fps = 25.0;

    // 视频写入初始化
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');

    if (!FLAGS_output_video.empty())
        writer.open(FLAGS_output_video, fourcc, video_fps, cv::Size(frame_w, frame_h));

    if (!FLAGS_output_video.empty() && !writer.isOpened()) {
        std::cerr << "Warning: could not open VideoWriter for output '" << FLAGS_output_video << "' - output will not be saved\n";
    }

    cv::Mat frame;
    int frame_idx = 0;
    
    // FPS计算相关变量
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end_time;
    double fps = 0.0;
    std::deque<double> fps_history;  // 用于平滑FPS
    const int fps_history_size = 10; // 平均FPS的帧数
    
    while (cap.read(frame))
    {
        frame_idx++;
        if (frame.empty()) break;

        // 开始计时
        auto frame_start = std::chrono::steady_clock::now();

        int img_w = frame.cols;
        int img_h = frame.rows;

        // Run YOLO on this frame
        // 官方封装的预处理函数和推理函数
        ////////////////////////////////////目标检测////////////////////////////////////////
        yolo11.pre_process(frame);
        yolo11.infer();
        auto detections = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, img_w, img_h);
        
        ////////////////////////////////////目标跟踪////////////////////////////////////////
        // 转换检测器输出的bbox格式传给bytetrack处理器
        std::vector<Object> objects;
        for (auto &det : detections)
        {
            Object o;
            // bbox = [x_center, y_center, width, height, confidence, class_id]
            float x1 = det.bbox[0];
            float y1 = det.bbox[1];
            float x2 = det.bbox[2];
            float y2 = det.bbox[3];
            o.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
            o.label = det.class_id;
            o.prob = det.score;
            objects.push_back(o);
        }
        
        // Update tracker
        auto tracks = tracker.update(objects);

        // 计算当前帧处理时间
        auto frame_end = std::chrono::steady_clock::now();
        // 处理思路是：记录下一帧的处理时间，然后用1s / 这个时间 得到帧率
        std::chrono::duration<double> frame_duration = frame_end - frame_start;
        double current_fps = 1.0 / frame_duration.count();
        
        // 平滑FPS（移动平均）
        fps_history.push_back(current_fps);
        if (fps_history.size() > fps_history_size) {
            fps_history.pop_front();
        }
        
        // 计算平均FPS
        double sum_fps = 0.0;
        for (double f : fps_history) {
            sum_fps += f;
        }
        fps = sum_fps / fps_history.size();
        
        ////////////////////////////////////////绘制结果//////////////////////////////
        // Draw tracked boxes and IDs
        int max_id = 0;
        for (auto &t : tracks)
        {
            if (!t.is_activated) continue;
            // tlwh：目标检测框，格式未[x,y,w,h]
            float x = t.tlwh[0];
            float y = t.tlwh[1];
            float w = t.tlwh[2];
            float h = t.tlwh[3];
            cv::Rect box(cv::Point((int)x, (int)y), cv::Size((int)w, (int)h));
            cv::Scalar col = tracker.get_color(t.track_id);
            cv::rectangle(frame, box, col, 2);
            
            std::ostringstream oss;
            oss << t.track_id;
            std::string id_text = oss.str();
            std::string label_text = id_text;
            
            if (t.is_activated && t.track_id > max_id) {
                max_id = t.track_id;
            }
            
            // 绘制计数信息（右上角）
            std::ostringstream count_hud;
            count_hud << "Now:" << objects.size() << "  Total:" << max_id;
            cv::Rect count_bg_rect(frame.cols - 245, 5, 240, 30);
            cv::Mat count_roi = frame(count_bg_rect);
            cv::Mat count_color(count_roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            cv::addWeighted(count_color, 0.5, count_roi, 0.5, 0.0, count_roi);
            cv::putText(frame, count_hud.str(), cv::Point(frame.cols - 240, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);


            cv::putText(frame, label_text, cv::Point((int)x, (int)std::max(0.0f, y-5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
        }

        // 绘制HUD信息
        // FPS信息
        std::ostringstream fps_hub;
        fps_hub << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(frame, fps_hub.str(), cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);



        if (writer.isOpened()) {
            writer.write(frame);
        }

        // 显示处理进度（每30帧）
        if (frame_idx % 30 == 0) {
            end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            double avg_fps = frame_idx / elapsed.count();
            std::cout << "Processed " << frame_idx << " frames, "<< std::endl;
        }
    }

    // 计算并显示总体统计信息
    end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed = end_time - start_time;
    double total_avg_fps = frame_idx / total_elapsed.count();
    
    std::cout << "\n======= Processing Complete =======" << std::endl;
    std::cout << "Total frames processed: " << frame_idx << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_elapsed.count() << " seconds" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1) << total_avg_fps << std::endl;
    
    if (writer.isOpened()) {
        std::cout << "Saved video to: " << FLAGS_output_video << std::endl;
    }
    
    return 0;
  }

  // 如果没有指定视频文件，输出错误信息
  std::cerr << "Error: No input video specified. Please set --input_video parameter." << std::endl;
  std::cerr << "Usage: " << argv[0] << " --input_video=<video_path>" << std::endl;
  return -1;
}