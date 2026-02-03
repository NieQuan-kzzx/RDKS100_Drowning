#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define CHECK_RET(ret, msg) \
    if ((ret) != 0) { \
        std::cerr << msg << " Error code: " << (ret) << std::endl; \
        return -1; \
    }

/**
 * 预处理函数：严格对齐 ONNX 端的 RGB 预处理逻辑
 * 步骤：BGR -> RGB -> Resize -> YUV420 -> NV12
 */
void prepare_nv12_debug(const cv::Mat& bgr, std::vector<hbDNNTensor>& inputs) {
    cv::Mat rgb, resized, yuv420p;
    
    // 1. 显式转换为 RGB (Patchcore 训练通常基于 RGB)
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    
    // 2. Resize 到模型要求的 224x224
    cv::resize(rgb, resized, cv::Size(224, 224));
    
    // 3. 将 RGB 数据转为 YUV420P
    cv::cvtColor(resized, yuv420p, cv::COLOR_RGB2YUV_I420);

    // 4. 拷贝 Y 分量 (224 * 224)
    std::memcpy(inputs[0].sysMem.virAddr, yuv420p.data, 224 * 224);
    
    // 5. 拼凑 NV12 的 UV 分量 (交错排列)
    uint8_t* u_src = yuv420p.data + (224 * 224);
    uint8_t* v_src = u_src + (224 * 224 / 4);
    uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs[1].sysMem.virAddr);
    
    for (int i = 0; i < 112; ++i) {
        for (int j = 0; j < 112; ++j) {
            uv_dest[i * 224 + j * 2] = u_src[i * 112 + j];
            uv_dest[i * 224 + j * 2 + 1] = v_src[i * 112 + j];
        }
    }
    
    // 刷新 Cache，确保 BPU 硬件能读取到最新数据
    hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
}

int main() {
    const char* model_file = "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm";
    const char* image_file = "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_patchcore.jpg";
    
    // 设定一个初始阈值，量化后的分数通常在 10-100 之间，需通过调试打印确定
    float threshold = 50.0f; 

    // --- 1. 加载并初始化模型 ---
    hbDNNPackedHandle_t packed_dnn_handle;
    CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN failed");
    
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    // --- 2. 准备输入/输出 Tensor ---
    int input_count = 0, output_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);
    
    std::vector<hbDNNTensor> inputs(input_count), outputs(output_count);

    // 输入内存申请 (含 Stride 修复)
    for (int i = 0; i < input_count; i++) {
        hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);
        auto &props = inputs[i].properties;
        if (i == 0) { // Y
            props.stride[0] = 50176; props.stride[1] = 224; props.stride[2] = 1; props.stride[3] = 1;
            props.alignedByteSize = 50176;
        } else { // UV
            props.stride[0] = 25088; props.stride[1] = 224; props.stride[2] = 2; props.stride[3] = 1;
            props.alignedByteSize = 25088;
        }
        hbUCPMallocCached(&inputs[i].sysMem, props.alignedByteSize, 0);
        inputs[i].sysMem.memSize = props.alignedByteSize;
    }

    // 输出内存申请
    for (int i = 0; i < output_count; i++) {
        hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);
        hbUCPMallocCached(&outputs[i].sysMem, outputs[i].properties.alignedByteSize, 0);
        outputs[i].sysMem.memSize = outputs[i].properties.alignedByteSize;
    }

    // --- 3. 读取图像并执行推理 ---
    cv::Mat bgr = cv::imread(image_file);
    if (bgr.empty()) {
        std::cerr << "Cannot find image: " << image_file << std::endl;
        return -1;
    }

    prepare_nv12_debug(bgr, inputs);

    hbUCPTaskHandle_t task_handle = nullptr;
    CHECK_RET(hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle), "Infer failed");
    
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    hbUCPSubmitTask(task_handle, &sched);
    hbUCPWaitTaskDone(task_handle, 0);

    // --- 4. 数据获取与调试监控 ---
    hbUCPMemFlush(&(outputs[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(&(outputs[1].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

    float* amap_ptr = (float*)outputs[0].sysMem.virAddr;
    float global_score = *(float*)outputs[1].sysMem.virAddr;

    // 分析热力图数值
    cv::Mat amap(224, 224, CV_32FC1, amap_ptr);
    double minVal, maxVal;
    cv::minMaxLoc(amap, &minVal, &maxVal);

    std::cout << "\n================ DEBUG LOG ================" << std::endl;
    std::cout << "Global Anomaly Score : " << std::fixed << std::setprecision(4) << global_score << std::endl;
    std::cout << "Heatmap Min Value    : " << minVal << std::endl;
    std::cout << "Heatmap Max Value    : " << maxVal << std::endl;
    std::cout << "Used Threshold       : " << threshold << std::endl;
    std::cout << "===========================================" << std::endl;

    // --- 5. 动态归一化渲染 (解决全图泛红的关键) ---
    cv::Mat amap_norm;
    if (maxVal - minVal > 1e-5) {
        // 将当前帧的数值拉伸到 0-255，确保热力图只显示相对异常区域
        amap.convertTo(amap_norm, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        amap_norm = cv::Mat::zeros(224, 224, CV_8UC1);
    }

    cv::Mat heatmap;
    cv::applyColorMap(amap_norm, heatmap, cv::COLORMAP_JET);
    cv::resize(heatmap, heatmap, bgr.size());

    cv::Mat overlay;
    cv::addWeighted(bgr, 0.7, heatmap, 0.3, 0, overlay);

    float relative_score = global_score - minVal; // 计算相对异常增量
    std::cout << "Relative Anomaly Score: " << relative_score << std::endl;

    if (global_score > threshold) {
        // 只有当全局分数远高于背景基础分(minVal)时才报警
        cv::putText(overlay, "ALARM: LEAKAGE!", cv::Point(50, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    }

    // 状态判定
    if (global_score > threshold) {
        cv::putText(overlay, "ALARM: LEAKAGE!", cv::Point(50, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    } else {
        cv::putText(overlay, "Status: Normal", cv::Point(50, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("debug_output.jpg", overlay);
    std::cout << "Success! Debug result saved to debug_output.jpg" << std::endl;

    // --- 6. 资源释放 ---
    hbUCPReleaseTask(task_handle);
    for (auto &in : inputs) hbUCPFree(&in.sysMem);
    for (auto &out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);

    return 0;
}

// #include <iostream>
// #include <vector>
// #include <cstring>
// #include <chrono>   // 高精度计时
// #include <iomanip>  // 格式化输出
// #include <opencv2/opencv.hpp>

// #include "hobot/dnn/hb_dnn.h"
// #include "hobot/hb_ucp.h"
// #include "hobot/hb_ucp_sys.h"



// // 预处理：将 BGR 图像转换为 224x224 的 NV12 格式并拷贝到 Tensor 内存
// void prepare_nv12(const cv::Mat& bgr, std::vector<hbDNNTensor>& inputs) {
//     cv::Mat resized, yuv420p;
//     cv::resize(bgr, resized, cv::Size(224, 224));
//     // 注意：Patchcore 转换时设定了 input_type_train 为 RGB
//     // 如果转换配置中 input_type_rt 为 nv12，这里转为 YUV420P 再手动拼 NV12
//     cv::cvtColor(resized, yuv420p, cv::COLOR_BGR2YUV_I420);

//     // 拷贝 Y 分量 (224 * 224)
//     std::memcpy(inputs[0].sysMem.virAddr, yuv420p.data, 224 * 224);
    
//     // 手动拼 NV12 的 UV 分量 (UV交错: UVUVUV...)
//     uint8_t* u_src = yuv420p.data + (224 * 224);
//     uint8_t* v_src = u_src + (224 * 224 / 4);
//     uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs[1].sysMem.virAddr);
    
//     for (int i = 0; i < 112; ++i) { // 高度 224/2
//         for (int j = 0; j < 112; ++j) { // 宽度 224/2
//             uv_dest[i * 224 + j * 2] = u_src[i * 112 + j];     // U
//             uv_dest[i * 224 + j * 2 + 1] = v_src[i * 112 + j]; // V
//         }
//     }
    
//     // 刷新 Cache，确保 CPU 写入的数据被 BPU 读到
//     hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
//     hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
// }

// int main() {
//     const char* model_file = "/home/sunrise/Desktop/test_patchcore/patchcore.hbm";
//     const char* video_source = "/home/sunrise/Desktop/test_patchcore/5.mp4"; // 或 "0" 代表板载摄像头
//     float threshold = 0.6f;

//     // --- 1. 模型初始化 ---
//     hbDNNPackedHandle_t packed_dnn_handle;
//     CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN failed");
//     const char **model_name_list;
//     int model_count = 0;
//     hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
//     hbDNNHandle_t dnn_handle;
//     hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

//     // --- 2. 内存申请与 Stride 修复 ---
//     int input_count, output_count;
//     hbDNNGetInputCount(&input_count, dnn_handle);
//     hbDNNGetOutputCount(&output_count, dnn_handle);
//     std::vector<hbDNNTensor> inputs(input_count), outputs(output_count);

//     // 输入 Tensor 设置 (224x224 NV12)
//     for (int i = 0; i < input_count; i++) {
//         hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);
//         auto &p = inputs[i].properties;
//         if (i == 0) { // Y
//             p.stride[0]=50176; p.stride[1]=224; p.stride[2]=1; p.stride[3]=1; p.alignedByteSize=50176;
//         } else { // UV
//             p.stride[0]=25088; p.stride[1]=224; p.stride[2]=2; p.stride[3]=1; p.alignedByteSize=25088;
//         }
//         hbUCPMallocCached(&inputs[i].sysMem, p.alignedByteSize, 0);
//     }
//     // 输出 Tensor 设置
//     for (int i = 0; i < output_count; i++) {
//         hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);
//         hbUCPMallocCached(&outputs[i].sysMem, outputs[i].properties.alignedByteSize, 0);
//     }

//     // --- 3. 视频源准备 ---
//     cv::VideoCapture cap(video_source);
//     if (!cap.isOpened()) { std::cerr << "Cannot open video!" << std::endl; return -1; }
//     int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//     int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//     double video_fps = cap.get(cv::CAP_PROP_FPS);
//     cv::VideoWriter writer("output_rdk.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), video_fps, cv::Size(width, height));

//     std::cout << "Starting RDK BPU Video Processing..." << std::endl;

//     int frame_count = 0;
//     // --- 记录开始时间 ---
//     auto start_time = std::chrono::high_resolution_clock::now();

//     cv::Mat frame;
//     while (cap.read(frame)) {
//         // A. 预处理
//         prepare_nv12(frame, inputs);

//         // B. BPU 推理
//         hbUCPTaskHandle_t task_handle = nullptr;
//         hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle);
//         hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
//         hbUCPSubmitTask(task_handle, &sched);
//         hbUCPWaitTaskDone(task_handle, 0);
//         hbUCPReleaseTask(task_handle);

//         // C. 后处理与渲染
//         hbUCPMemFlush(&(outputs[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
//         hbUCPMemFlush(&(outputs[1].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

//         float* amap_ptr = (float*)outputs[0].sysMem.virAddr; // 224x224 热力图
//         float score = *(float*)outputs[1].sysMem.virAddr;    // 全局得分

//         cv::Mat amap(224, 224, CV_32FC1, amap_ptr);
//         double minV, maxV;
//         cv::minMaxLoc(amap, &minV, &maxV);
        
//         cv::Mat amap_norm, heatmap;
//         amap.convertTo(amap_norm, CV_8UC1, 255.0/(maxV - minV + 1e-6), -minV * 255.0/(maxV - minV + 1e-6));
//         cv::applyColorMap(amap_norm, heatmap, cv::COLORMAP_JET);
//         cv::resize(heatmap, heatmap, frame.size());

//         cv::Mat overlay;
//         cv::addWeighted(frame, 0.7, heatmap, 0.3, 0, overlay);

//         // 绘制界面信息
//         std::string score_str = "Score: " + std::to_string(score).substr(0, 5);
//         cv::putText(overlay, score_str, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
//         if (score > threshold) {
//             cv::putText(overlay, "LEAKAGE ALERT!", cv::Point(30, 120), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
//         }

//         writer.write(overlay);
//         frame_count++;

//         if (frame_count % 50 == 0) {
//             std::cout << "已处理 " << frame_count << " 帧..." << std::endl;
//         }
//     }

//     // --- 记录结束时间并统计性能 ---
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
//     double avg_ms = duration.count() / frame_count;
//     double avg_fps = 1000.0 / avg_ms;

//     std::cout << "\n================ Performance Statistics ================" << std::endl;
//     std::cout << "Total Frames      : " << frame_count << std::endl;
//     std::cout << "Average Latency   : " << std::fixed << std::setprecision(2) << avg_ms << " ms/frame" << std::endl;
//     std::cout << "Average Throughput: " << std::fixed << std::setprecision(2) << avg_fps << " FPS" << std::endl;
//     std::cout << "========================================================" << std::endl;

//     // --- 4. 释放资源 ---
//     for (auto &in : inputs) hbUCPFree(&in.sysMem);
//     for (auto &out : outputs) hbUCPFree(&out.sysMem);
//     hbDNNRelease(packed_dnn_handle);
//     cap.release();
//     writer.release();

//     return 0;
// }