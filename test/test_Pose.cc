#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11_pose.hpp"
#include "common_utils.hpp"

// 命令行参数
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/yolo11n_pose_nashe_640x640_nv12.hbm",
              "BPU量化模型路径 (*.hbm)");
DEFINE_string(video_path, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test.mp4", "视频文件路径或摄像头索引");
DEFINE_string(output_path, "output.mp4", "输出视频路径，为空则不保存");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes.names", "类别标签文件");
DEFINE_double(score_thres, 0.25, "检测置信度阈值");
DEFINE_double(nms_thres, 0.7, "非极大值抑制阈值");
DEFINE_double(kpt_conf_thres, 0.5, "关键点置信度阈值");
DEFINE_bool(show_fps, true, "显示FPS");

// YOLO Pose 17个关键点的连接关系
const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

// 17种不同的颜色，对应17个关键点
const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 0),     // 蓝色
    cv::Scalar(255, 85, 0),    // 橙蓝
    cv::Scalar(255, 170, 0),   // 橙
    cv::Scalar(255, 255, 0),   // 黄色
    cv::Scalar(170, 255, 0),   // 黄绿
    cv::Scalar(85, 255, 0),    // 绿色
    cv::Scalar(0, 255, 0),     // 亮绿
    cv::Scalar(0, 255, 85),    // 青绿
    cv::Scalar(0, 255, 170),   // 青色
    cv::Scalar(0, 255, 255),   // 亮青
    cv::Scalar(0, 170, 255),   // 浅蓝
    cv::Scalar(0, 85, 255),    // 蓝色
    cv::Scalar(0, 0, 255),     // 红色
    cv::Scalar(85, 0, 255),    // 紫色
    cv::Scalar(170, 0, 255),   // 粉紫
    cv::Scalar(255, 0, 255),   // 粉色
    cv::Scalar(255, 0, 170)    // 紫红
};

void draw_keypoints_enhanced(cv::Mat& image,
                            const std::vector<std::vector<Keypoint>>& kpts,
                            float kpt_conf_thresh,
                            int radius_outer = 5,
                            int radius_inner = 3)
{
    // 将sigmoid阈值转换为原始logits阈值（与draw_keypoints函数一致）
    const float kpt_conf_inverse = -std::log(1.0f / kpt_conf_thresh - 1.0f);
    
    for (size_t i = 0; i < kpts.size(); ++i) {
        const auto& instance = kpts[i];
        if (instance.size() < 17) continue;  // 确保有17个关键点
        
        // 1. 先绘制骨骼连接线
        for (const auto& edge : SKELETON) {
            int idx1 = edge.first;
            int idx2 = edge.second;
            
            if (idx1 < instance.size() && idx2 < instance.size()) {
                const Keypoint& kp1 = instance[idx1];
                const Keypoint& kp2 = instance[idx2];
                
                // 检查置信度（使用原始logits比较）
                if (kp1.score >= kpt_conf_inverse && kp2.score >= kpt_conf_inverse) {
                    const int x1 = static_cast<int>(kp1.x);
                    const int y1 = static_cast<int>(kp1.y);
                    const int x2 = static_cast<int>(kp2.x);
                    const int y2 = static_cast<int>(kp2.y);
                    
                    // 使用起始点的颜色绘制连接线
                    cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2),
                            COLORS[idx1 % COLORS.size()], 2, cv::LINE_AA);
                }
            }
        }
        
        // 2. 绘制关键点
        for (size_t j = 0; j < instance.size(); ++j) {
            const Keypoint& kp = instance[j];
            
            // 跳过低置信度的点
            if (kp.score < kpt_conf_inverse) continue;
            
            const int x = static_cast<int>(kp.x);
            const int y = static_cast<int>(kp.y);
            
            // 绘制外圈（白色）
            cv::circle(image, cv::Point(x, y), radius_outer, 
                      cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
            
            // 绘制内圈（彩色）
            cv::circle(image, cv::Point(x, y), radius_inner,
                      COLORS[j % COLORS.size()], -1, cv::LINE_AA);
        }
    }
}

int main(int argc, char **argv)
{
    // 解析命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // 初始化模型
    YOLO11_Pose yolo11_pose(FLAGS_model_path);
    
    // 加载类别标签
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    
    // 打开视频源
    cv::VideoCapture cap;
    if (FLAGS_video_path.size() == 1 && isdigit(FLAGS_video_path[0])) {
        cap.open(std::stoi(FLAGS_video_path));  // 摄像头
    } else {
        cap.open(FLAGS_video_path);  // 视频文件
    }
    
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频源: " << FLAGS_video_path << std::endl;
        return -1;
    }
    
    // 获取视频参数
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // 初始化视频写入器
    cv::VideoWriter writer;
    if (!FLAGS_output_path.empty()) {
        double out_fps = (fps > 0) ? fps : 30.0;
        writer.open(FLAGS_output_path, 
                   cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                   out_fps, cv::Size(width, height));
    }
    
    // 性能统计变量
    int frame_count = 0;
    double total_time = 0.0;
    cv::TickMeter tm;
    
    std::cout << "开始处理视频..." << std::endl;
    
    cv::Mat frame;
    while (cap.read(frame)) {
        frame_count++;
        
        // 记录开始时间
        tm.start();
        
        // 保存原始尺寸
        int img_w = frame.cols;
        int img_h = frame.rows;
        
        // 模型推理流程
        yolo11_pose.pre_process(frame);
        yolo11_pose.infer();
        auto [final_dets, resized_kpts] = yolo11_pose.post_process(
            FLAGS_score_thres, FLAGS_nms_thres, FLAGS_kpt_conf_thres, img_w, img_h);
        
        // 可视化结果
        draw_boxes(frame, final_dets, class_names, rdk_colors);
        
        // 绘制函数
        draw_keypoints_enhanced(frame, resized_kpts, FLAGS_kpt_conf_thres);

        // 计算处理时间
        tm.stop();
        double frame_time = tm.getTimeMilli();
        total_time += frame_time;
        tm.reset();
        
        // 显示FPS
        if (FLAGS_show_fps) {
            double current_fps = 1000.0 / frame_time;
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));
            // 在左上角添加黑色背景的FPS显示
            cv::rectangle(frame, cv::Point(10, 10), cv::Point(150, 50), cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, fps_text, cv::Point(20, 40), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }
        
        // 保存输出视频
        if (writer.isOpened()) {
            writer.write(frame);
        }
        
        // 每10帧打印一次进度
        if (frame_count % 10 == 0) {
            std::cout << "已处理 " << frame_count << " 帧" << std::endl;
        }
    }
    
    // 释放资源
    cap.release();
    if (writer.isOpened()) {
        writer.release();
        std::cout << "输出视频已保存: " << FLAGS_output_path << std::endl;
    }
    cv::destroyAllWindows();
    
    // 输出统计信息
    if (frame_count > 0) {
        std::cout << "\n===== 处理完成 =====" << std::endl;
        std::cout << "总帧数: " << frame_count << std::endl;
        std::cout << "总时间: " << total_time / 1000.0 << " 秒" << std::endl;
        std::cout << "平均每帧时间: " << total_time / frame_count << " 毫秒" << std::endl;
        std::cout << "平均FPS: " << 1000.0 / (total_time / frame_count) << std::endl;
    }
    
    return 0;
}