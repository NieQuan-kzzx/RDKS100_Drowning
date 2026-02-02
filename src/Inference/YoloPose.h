# pragma once 

# include "BaseInfer.h"
# include "ultralytics_yolo11_pose.hpp"
# include "BYTETracker.hpp"
# include <memory>

// 添加命名空间是为了解决与官方自带的命名冲突问题
namespace Inf {

class YoloPose : public BaseInfer {
public:
    YoloPose();
    ~YoloPose() override;

    bool init(const std::string& model_path) override;
    std::vector<Detection> run(cv::Mat& frame) override;
    void cleanup() override;

    void draw(cv::Mat& frame, const std::vector<Detection>& results) override;
private:
    std::unique_ptr<YOLO11_Pose> m_yolo_pose;
    std::unique_ptr<BYTETracker> m_tracker;
};

};