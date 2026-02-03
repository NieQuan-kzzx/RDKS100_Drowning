#pragma once
#include "BaseInfer.h"
#include <memory>
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"

namespace Inf {
class Patchcore : public BaseInfer {
public:
    Patchcore();
    ~Patchcore() override;

    bool init(const std::string &modelPath) override;
    std::vector<Detection> run(cv::Mat& frame) override;
    void draw(cv::Mat& frame, const std::vector<Detection>& detections) override;
    void cleanup() override;
private:
    // 内部处理函数
    void setupTensors();
    void preprocess(const cv::Mat& bgr);

    // BPU 相关句柄
    hbDNNPackedHandle_t packed_handle_ = nullptr;
    hbDNNHandle_t dnn_handle_ = nullptr;
    std::vector<hbDNNTensor> inputs_;
    std::vector<hbDNNTensor> outputs_;

    // 算法参数与缓存
    float threshold_ = 50.0f;
    cv::Mat m_current_amap; // 缓存热力图用于 draw 接口
    
    // 预分配中间变量，避免 run 循环中重复申请内存
    cv::Mat resized_rgb_;
    cv::Mat yuv420p_;

};
} // namespace Inf