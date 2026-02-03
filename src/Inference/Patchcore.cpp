#include "Patchcore.h"
#include "hobot/hb_ucp_sys.h"
#include <iostream>
#include "PlogInitializer.h"

namespace Inf {

Patchcore::Patchcore() {
    PlogInitializer::getInstance().init(plog::verbose);
}

Patchcore::~Patchcore() {
    cleanup();
}

bool Patchcore::init(const std::string& model_path) {
    const char* path = model_path.c_str();
    if (hbDNNInitializeFromFiles(&packed_handle_, &path, 1) != 0) {
        return false;
    }

    const char **name_list;
    int count = 0;
    hbDNNGetModelNameList(&name_list, &count, packed_handle_);
    hbDNNGetModelHandle(&dnn_handle_, packed_handle_, name_list[0]);

    setupTensors();
    return true;
}

void Patchcore::setupTensors() {
    int input_count = 0, output_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle_);
    hbDNNGetOutputCount(&output_count, dnn_handle_);

    inputs_.resize(input_count);
    outputs_.resize(output_count);

    for (int i = 0; i < input_count; i++) {
        hbDNNGetInputTensorProperties(&inputs_[i].properties, dnn_handle_, i);
        auto &props = inputs_[i].properties;

        // 核心修复：显式补全所有 stride 维度，不要漏掉任何一个
        if (i == 0) { // Y 分量
            props.stride[0] = 50176; 
            props.stride[1] = 224;
            props.stride[2] = 1;
            props.stride[3] = 1;
            props.alignedByteSize = 50176;
        } else { // UV 分量
            props.stride[0] = 25088;
            props.stride[1] = 224;
            props.stride[2] = 2;
            props.stride[3] = 1;
            props.alignedByteSize = 25088;
        }
        
        // 确保分配的内存大小与 alignedByteSize 一致
        hbUCPMallocCached(&inputs_[i].sysMem, props.alignedByteSize, 0);
        inputs_[i].sysMem.memSize = props.alignedByteSize; // 别忘了设置 memSize
    }

    // 输出部分同理
    for (int i = 0; i < output_count; i++) {
        hbDNNGetOutputTensorProperties(&outputs_[i].properties, dnn_handle_, i);
        hbUCPMallocCached(&outputs_[i].sysMem, outputs_[i].properties.alignedByteSize, 0);
        outputs_[i].sysMem.memSize = outputs_[i].properties.alignedByteSize;
    }
}

void Patchcore::preprocess(const cv::Mat& bgr) {
    // 1. 颜色空间转换与缩放
    cv::cvtColor(bgr, resized_rgb_, cv::COLOR_BGR2RGB);
    cv::resize(resized_rgb_, resized_rgb_, cv::Size(224, 224));
    cv::cvtColor(resized_rgb_, yuv420p_, cv::COLOR_RGB2YUV_I420);

    // 2. Y 分量拷贝
    std::memcpy(inputs_[0].sysMem.virAddr, yuv420p_.data, 224 * 224);
    
    // 3. UV 分量交错排列 (NV12 构造)
    uint8_t* u_src = yuv420p_.data + (224 * 224);
    uint8_t* v_src = u_src + (224 * 224 / 4);
    uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs_[1].sysMem.virAddr);
    
    for (int i = 0; i < 112; ++i) {
        for (int j = 0; j < 112; ++j) {
            uv_dest[i * 224 + j * 2] = u_src[i * 112 + j];
            uv_dest[i * 224 + j * 2 + 1] = v_src[i * 112 + j];
        }
    }
    
    hbUCPMemFlush(&inputs_[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    hbUCPMemFlush(&inputs_[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
}

std::vector<Detection> Patchcore::run(cv::Mat& frame) {
    preprocess(frame);

    hbUCPTaskHandle_t task = nullptr;
    hbDNNInferV2(&task, outputs_.data(), inputs_.data(), dnn_handle_);
    
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    hbUCPSubmitTask(task, &sched);
    hbUCPWaitTaskDone(task, 0);
    hbUCPReleaseTask(task);

    hbUCPMemFlush(&(outputs_[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(&(outputs_[1].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

    float* amap_ptr = (float*)outputs_[0].sysMem.virAddr;
    float global_score = *(float*)outputs_[1].sysMem.virAddr;

    // 缓存热力图
    m_current_amap = cv::Mat(224, 224, CV_32FC1, amap_ptr).clone();

    Detection d;
    d.score = global_score;
    d.class_id = (global_score > threshold_) ? 1 : 0;
    return {d};
}

void Patchcore::draw(cv::Mat& frame, const std::vector<Detection>& results) {
    if (results.empty() || m_current_amap.empty()) return;

    double minV, maxV;
    cv::minMaxLoc(m_current_amap, &minV, &maxV);
    
    cv::Mat amap_8u;
    m_current_amap.convertTo(amap_8u, CV_8UC1, 255.0/std::max(1e-5, maxV - minV), -minV*255.0/std::max(1e-5, maxV - minV));
    
    cv::Mat color_map;
    cv::applyColorMap(amap_8u, color_map, cv::COLORMAP_JET);
    cv::resize(color_map, color_map, frame.size());
    
    cv::addWeighted(frame, 0.7, color_map, 0.3, 0, frame);

    std::string label = (results[0].score > threshold_) ? "ALARM: LEAK" : "Normal";
    cv::Scalar color = (results[0].score > threshold_) ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
    
    cv::putText(frame, label + " Score:" + std::to_string(results[0].score).substr(0, 5), 
                cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
}

void Patchcore::cleanup() {
    for (auto &in : inputs_) {
        if (in.sysMem.virAddr) hbUCPFree(&in.sysMem);
    }
    for (auto &out : outputs_) {
        if (out.sysMem.virAddr) hbUCPFree(&out.sysMem);
    }
    inputs_.clear();
    outputs_.clear();
    if (packed_handle_) {
        hbDNNRelease(packed_handle_);
        packed_handle_ = nullptr;
    }
}

} // namespace Inf