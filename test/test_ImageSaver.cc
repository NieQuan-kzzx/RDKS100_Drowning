#include "ImageSaver.h"
#include <opencv2/opencv.hpp>

int main() {
    // 1. 初始化（会在当前目录下创建 test_captures）
    ImageSaver saver("./test_captures");

    // 2. 模拟产生 5 张图像
    for (int i = 0; i < 5; ++i) {
        // 创建一张纯色背景并写上数字的图
        cv::Mat testImg(720, 1280, CV_8UC3, cv::Scalar(0, 255, 0)); // 绿色背景
        cv::putText(testImg, "Test Frame " + std::to_string(i), 
                    cv::Point(400, 360), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);

        // 调用你的类方法
        saver.addImage(testImg, "frame_" + std::to_string(i));
        std::cout << "Added mock frame " << i << " to saver." << std::endl;
    }

    // 3. 执行写入
    std::cout << "Flushing images to disk..." << std::endl;
    saver.flush();

    std::cout << "Test Done. Please check ./test_captures folder." << std::endl;
    return 0;
}