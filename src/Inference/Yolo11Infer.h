#pragma once
#include "BaseInfer.h"
#include "ultralytics_yolo11.hpp"
#include "BYTETracker.hpp"
#include <memory>

namespace Inf { // 必须包裹

class Yolo11Infer : public BaseInfer {
public:
    Yolo11Infer() = default;
    ~Yolo11Infer() override;

    bool init(const std::string& model_path) override;
    // 这里的 Detection 自动识别为 Inf::Detection
    std::vector<Detection> run(cv::Mat& frame) override; 
    void cleanup() override;

private:
    std::unique_ptr<YOLO11> m_yolo;
    std::unique_ptr<BYTETracker> m_tracker;
};

} // namespace Inf