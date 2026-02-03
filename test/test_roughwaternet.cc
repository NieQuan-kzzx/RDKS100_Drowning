#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

// 宏定义：检查函数返回值，如果出错则打印错误码并退出
#define CHECK_RET(ret, msg) \
    if ((ret) != 0) { \
        std::cerr << msg << " Error code: " << (ret) << std::endl; \
        return -1; \
    }

int main() {
    const char* model_file = "/home/sunrise/Desktop/RDKS100_Drowning/models/rough_waternet.hbm"; 
    const char* image_file = "/home/sunrise/Desktop/RDKS100_Drowning/tem/water.jpg";

    // --- 1. 初始化并加载模型 (Init DNN) ---
    hbDNNPackedHandle_t packed_dnn_handle;
    CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN failed");

    // 获取模型名称列表并根据名称获取模型句柄
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    // --- 2. 准备输入 Tensor (Input Tensor Preparation) ---
    int input_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    std::vector<hbDNNTensor> inputs(input_count);

    for (int i = 0; i < input_count; i++) {
        // 获取输入 Tensor 的属性（形状、类型、步长等）
        hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);
        auto &props = inputs[i].properties;
        
        /**
         * 针对 512x512 NV12 模型的输入对齐修复 (Memory Alignment)
         * BPU 硬件要求每一行数据的起始地址必须符合对齐要求（通常是16或32字节）。
         * 这里手动覆盖 Stride 信息以确保与硬件预期的 512 宽度完全匹配。
         */
        if (i == 0) { // Y 分量 (Brightness/Luma)
            props.stride[0] = 262144; // 512 * 512
            props.stride[1] = 512;    // 行步长 (Row Stride)
            props.stride[2] = 1; 
            props.stride[3] = 1;
            props.alignedByteSize = 262144;
        } else { // UV 分量 (Chroma/Color)
            props.stride[0] = 131072; // 512 * 256
            props.stride[1] = 512;    // UV交替存储，一行依然是512字节
            props.stride[2] = 2; 
            props.stride[3] = 1;
            props.alignedByteSize = 131072;
        }
        // 在 BPU Cached 内存中申请空间
        hbUCPMallocCached(&inputs[i].sysMem, props.alignedByteSize, 0);
        inputs[i].sysMem.memSize = props.alignedByteSize;
    }

    // --- 3. 图像预处理 (Preprocessing: BGR to NV12) ---
    cv::Mat bgr = cv::imread(image_file);
    cv::Mat resized;
    if (!bgr.empty()) {
        cv::Mat yuv420p;
        // 缩放到模型要求的 512x512 尺寸
        cv::resize(bgr, resized, cv::Size(512, 512));
        // 将 BGR 转换为 YUV420P (I420: YYYY...UU...VV) 格式
        cv::cvtColor(resized, yuv420p, cv::COLOR_RGB2YUV_I420);
        
        // 拷贝 Y 数据到第一个输入 Tensor
        std::memcpy(inputs[0].sysMem.virAddr, yuv420p.data, 512 * 512);
        
        // 计算 U 和 V 数据在 I420 内存中的起始位置
        uint8_t* u_src = yuv420p.data + (512 * 512);
        uint8_t* v_src = u_src + (512 * 512 / 4);
        uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs[1].sysMem.virAddr);
        
        // 手动将 I420 转换为 NV12 (UV交错存储: UVUVUV...)
        for (int i = 0; i < 256; ++i) { // 高度减半
            for (int j = 0; j < 256; ++j) { // 宽度减半，但每个点存一对 UV
                uv_dest[i * 512 + j * 2] = u_src[i * 256 + j];     // U
                uv_dest[i * 512 + j * 2 + 1] = v_src[i * 256 + j]; // V
            }
        }
        // 刷新 Cache，确保数据从 CPU 同步到 BPU 硬件可见的内存中
        hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // --- 4. 输出 Tensor 准备 (Output Tensor Preparation) ---
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> outputs(output_count);
    hbDNNGetOutputTensorProperties(&outputs[0].properties, dnn_handle, 0);
    // 申请输出内存：3个类别 * 512 * 512 * 4字节 (Float32 类型)
    outputs[0].properties.alignedByteSize = 3145728; 
    hbUCPMallocCached(&outputs[0].sysMem, 3145728, 0);
    outputs[0].sysMem.memSize = 3145728;

    // --- 5. 执行推理 (Inference) ---
    hbUCPTaskHandle_t task_handle = nullptr;
    // 发起推理任务
    CHECK_RET(hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle), "Infer failed");
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    // 提交任务到 BPU 调度并等待执行结束
    hbUCPSubmitTask(task_handle, &sched);
    hbUCPWaitTaskDone(task_handle, 0);

    // --- 6. 后处理：NCHW 寻址与结果解析 (Post-processing) ---
    // 无效化 Cache，确保读取到的是 BPU 写入的最新的物理内存数据
    hbUCPMemFlush(&(outputs[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    float* raw_ptr = (float*)outputs[0].sysMem.virAddr;
    auto &prop = outputs[0].properties;

    int h = prop.validShape.dimensionSize[2]; // 高度: 512
    int w = prop.validShape.dimensionSize[3]; // 宽度: 512
    int c_out = prop.validShape.dimensionSize[1]; // 通道数/类别数: 3

    cv::Mat result(h, w, CV_8UC3);
    // 定义类别对应的颜色调色板 (Class 0, 1, 2)
    std::vector<cv::Vec3b> palette = {{0,0,0}, {0,255,0}, {0,0,255}}; // 黑, 绿, 红
    std::vector<int> class_counts(3, 0);
    float min_val = 1e10f, max_val = -1e10f;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int max_id = 0;
            float current_max = -1e10f;

            // 针对每个像素，在不同通道（类别）中寻找最大概率
            for (int c = 0; c < c_out; c++) {
                /**
                 * 连续 NCHW 寻址公式 (Continuous Memory Access)
                 * 计算公式：通道偏移 + 行偏移 + 列偏移
                 */
                int offset = c * (h * w) + y * w + x; 
                float val = raw_ptr[offset];

                // 记录数值范围以便调试
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;

                // Argmax 逻辑
                if (val > current_max) {
                    current_max = val;
                    max_id = c;
                }
            }
            class_counts[max_id % 3]++;
            // 将预测出的类别 ID 映射到调色板颜色
            result.at<cv::Vec3b>(y, x) = palette[max_id % 3];
        }
    }

    // --- 7. 打印统计信息与结果保存 ---
    std::cout << "--- Inference Debug Info ---" << std::endl;
    std::cout << "Output Value Range: [" << min_val << ", " << max_val << "]" << std::endl;
    for (int k = 0; k < 3; k++) {
        std::cout << "Class " << k << " pixels: " << class_counts[k] 
                  << " (" << (class_counts[k] * 100.0 / (h * w)) << "%)" << std::endl;
    }

 
    // ---  图像叠加 (Image Overlay/Blending) ---
    cv::Mat overlay;
    double alpha = 0.6; // 原图的权重
    double beta = 0.4;  // 掩码图的权重
    double gamma = 0.0; // 亮度偏移量
    
    // 将原图与分割结果按比例融合
    cv::addWeighted(resized, alpha, result, beta, gamma, overlay);

    // 保存叠加后的结果
    cv::imwrite("roughwaternet_blended.png", overlay);
    std::cout << "Overlay result saved to overlay_result.png" << std::endl;

    cv::imwrite("roughwaternet_mask.png", result);

    // --- 8. 释放资源 (Resource Cleanup) ---
    hbUCPReleaseTask(task_handle);
    for (auto &in : inputs) hbUCPFree(&in.sysMem);
    for (auto &out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);

    std::cout << "Done! Result saved to output_water.png" << std::endl;
    return 0;
}