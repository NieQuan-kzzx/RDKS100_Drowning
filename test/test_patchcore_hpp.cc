#include <iostream>
#include <opencv2/opencv.hpp>
#include "Patchcore.h"  // 确保包含你的 Patchcore 头文件
#include "gflags/gflags.h"
#include "PlogInitializer.h"

// 定义命令行参数
DEFINE_string(model, "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm", "Patchcore模型路径");
DEFINE_string(input, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_patchcore.jpg", "测试图像或视频路径");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 1. 初始化日志
    PlogInitializer::getInstance().init(plog::verbose);
    PLOGI << "Starting Patchcore standalone test (Save Image Mode)...";

    // 2. 初始化 Patchcore
    Inf::Patchcore detector; 
    if (!detector.init(FLAGS_model)) {
        PLOGE << "Patchcore init failed!";
        return -1;
    }

    // 3. 读取输入
    cv::Mat frame = cv::imread(FLAGS_input); // 直接读图测试最快
    if (frame.empty()) {
        PLOGE << "Cannot read image: " << FLAGS_input;
        return -1;
    }

    // 4. 执行推理
    auto results = detector.run(frame);
    
    // 5. 绘制结果（内部包含热力图叠加和文字）
    detector.draw(frame, results);

    // 6. 保存图片而不显示
    std::string out_path = "patchcore_result.jpg";
    cv::imwrite(out_path, frame);
    
    PLOGI << "Inference success! Result saved to: " << out_path;
    PLOGI << "Global Score: " << (results.empty() ? 0 : results[0].score);
    PLOGI << "Test finished.";

    return 0;
}