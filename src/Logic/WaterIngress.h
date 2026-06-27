#pragma once
#include "LogicBase.h"
#include "Patchcore.h"
#include "YoloSeg.h"
#include <memory>

class WaterIngress : public LogicBase {
public:
    WaterIngress();
    ~WaterIngress() override;

    bool initPatchcore(const std::string& model_path);
    bool initSeg(const std::string& model_path);
    void setPatchcoreThreshold(float thr) { m_patchcore_threshold = thr; }
    void setSegLabels(const std::vector<std::string>& labels);
    void setWaterClassId(int id) { m_water_class_id = id; }

    void process(cv::Mat& frame, const std::vector<Inf::Detection>& results) override;

private:
    std::unique_ptr<Inf::Patchcore> m_patchcore;
    std::unique_ptr<Inf::YoloSeg> m_yolo_seg;
    float m_patchcore_threshold = 50.0f;
    int m_water_class_id = 0;
    bool m_seg_initialized = false;
};
