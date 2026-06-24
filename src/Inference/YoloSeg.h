#pragma once
#include "BaseInfer.h"
#include "ultralytics_yolo11_seg.hpp"
#include "BYTETracker.hpp"
#include <memory>

namespace Inf { // 必须包裹

class YoloSeg : public BaseInfer {
public:
    YoloSeg();
    ~YoloSeg() override;

    bool init(const std::string& model_path) override;
    // 这里的 Detection 自动识别为 Inf::Detection
    std::vector<Detection> run(cv::Mat& frame) override; 
    void cleanup() override;

    void draw(cv::Mat& frame, const std::vector<Detection>& results) override;

    void setLabels(const std::vector<std::string>& labels) { m_labels = labels;}
    std::vector<std::string> getLabels() const override { return m_labels; }

private:
    std::unique_ptr<YOLO11_Seg> m_yolo_seg;
    std::unique_ptr<BYTETracker> m_tracker;

    std::map<int, int> m_track_id_to_class;
    std::vector<std::string> m_labels;
};

} // namespace Inf