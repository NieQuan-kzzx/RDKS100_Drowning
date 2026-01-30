#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace Inf { // 增加命名空间
    // 这里是参考地平线官方的common.hpp文件的格式
    // 以后添加的模型也是按照这种结构体的形式
    struct Detection {
        cv::Rect rect;     // 我们的标准：使用 cv::Rect
        float score;
        int class_id;
        int track_id = -1;
    };
    // 这是基类的推理，提供几个公共的接口，便于后续的多态实现
    class BaseInfer {
    public:
        virtual ~BaseInfer() {}
        virtual bool init(const std::string& model_path) = 0; // 模型初始化
        virtual std::vector<Detection> run(cv::Mat& frame) = 0; // 核心推理接口
        virtual void cleanup() = 0; // 资源清理接口
    };
}