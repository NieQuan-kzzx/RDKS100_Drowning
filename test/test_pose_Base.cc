// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include "YoloPose.h"
// #include "gflags/gflags.h"

// DEFINE_string(model, "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11n-pose.hbm", "模型路径");
// DEFINE_string(input, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test.mp4", "输入视频路径");

// // --- 步骤 1: 把连线函数放在这里 ---
// void draw_skeleton_logic(cv::Mat& image, const std::vector<Inf::Detection>& results, float kpt_conf_threshold) {
//     // 关键点连接关系 (COCO 17点标准)
//     const std::vector<std::pair<int, int>> SKELETON = {
//         {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
//         {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
//     };

//     // 绘制颜色方案
//     const std::vector<cv::Scalar> COLORS = {
//         cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
//         cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
//         cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
//         cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
//         cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
//         cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170)
//     };

//     for (const auto& det : results) {
//         if (det.keypoints.empty()) continue;

//         // 绘制线条
//         for (const auto& edge : SKELETON) {
//             const auto& kp1 = det.keypoints[edge.first];
//             const auto& kp2 = det.keypoints[edge.second];
//             if (kp1.score >= kpt_conf_threshold && kp2.score >= kpt_conf_threshold) {
//                 cv::line(image, cv::Point(kp1.x, kp1.y), cv::Point(kp2.x, kp2.y), 
//                          COLORS[edge.first % COLORS.size()], 2, cv::LINE_AA);
//             }
//         }
//         // 绘制点
//         for (size_t i = 0; i < det.keypoints.size(); ++i) {
//             const auto& kp = det.keypoints[i];
//             if (kp.score >= kpt_conf_threshold) {
//                 cv::circle(image, cv::Point(kp.x, kp.y), 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
//                 cv::circle(image, cv::Point(kp.x, kp.y), 2, COLORS[i % COLORS.size()], -1, cv::LINE_AA);
//             }
//         }
//     }
// }

// int main(int argc, char** argv) {
//     gflags::ParseCommandLineFlags(&argc, &argv, true);
//     Inf::YoloPose pose_handler;
    
//     if (!pose_handler.init(FLAGS_model)) return -1;

//     cv::VideoCapture cap(FLAGS_input);
//     cv::Mat frame;
//     cv::TickMeter tm;

//     while (cap.read(frame)) {
//         tm.start();
        
//         // --- 步骤 2: 获取结果 (这里面已经包含了你的匹配逻辑和 TrackID) ---
//         std::vector<Inf::Detection> results = pose_handler.run(frame);
        
//         tm.stop();

//         // --- 步骤 3: 绘制结果 ---
//         // 1. 画框和 ID
//         for (const auto& det : results) {
//             cv::rectangle(frame, det.rect, cv::Scalar(0, 255, 0), 2);
//             cv::putText(frame, "ID:" + std::to_string(det.track_id), 
//                         cv::Point(det.rect.x, det.rect.y - 10), 
//                         cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
//         }

//         // 2. 画骨骼连线 (调用刚才定义的函数)
//         draw_skeleton_logic(frame, results, 0.5f);

//         cv::imshow("RDK X5 - YOLO11 Pose", frame);
//         if (cv::waitKey(1) == 27) break;
//         tm.reset();
//     }
//     return 0;
// }


#include <iostream>
#include <opencv2/opencv.hpp>
#include "YoloPose.h"
#include "gflags/gflags.h"

DEFINE_string(model, "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11n-pose.hbm", "模型路径");
DEFINE_string(input, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test.mp4", "输入视频路径");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Inf::YoloPose pose_handler;
    if (!pose_handler.init(FLAGS_model)) {
        return -1;
    }

    cv::VideoCapture cap(FLAGS_input);
    cv::Mat frame;

    while (cap.read(frame)) {
        // 1. 推理并获得结果（含 TrackID 和 关键点）
        auto results = pose_handler.run(frame);
        
        // 2. 直接调用类的 draw 方法绘制所有内容
        pose_handler.draw(frame, results);

        // 3. 显示
        cv::imshow("RDK X5 - YOLO11 Pose", frame);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}