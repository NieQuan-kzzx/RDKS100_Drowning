#pragma once

#include <opencv2/opencv.hpp>
#include "BaseInfer.h"

// 业务逻辑基类
class LogicBase {
public:
    virtual ~LogicBase() = default;
    virtual void process(cv::Mat& frame, const std::vector<Inf::Detection>& results) = 0;
};