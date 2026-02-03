#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define CHECK_RET(ret, msg) \
    if ((ret) != 0) { \
        std::cerr << "[ERROR] " << msg << " | Code: " << (ret) << std::endl; \
        return -1; \
    }

int main() {
    const char* model_file = "/home/sunrise/Desktop/RDKS100_Drowning/models/fine_waternet.hbm";
    const char* image_file = "/home/sunrise/Desktop/RDKS100_Drowning/tem/water.jpg";

    hbDNNPackedHandle_t packed_dnn_handle;
    CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN");
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    const int MODEL_W = 2160;
    const int MODEL_H = 256;
    const int ALIGNED_W = 2176; 

    // --- 1. 输入 Tensor 准备 (严格匹配硬件 Stride 要求) ---
    int input_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    std::vector<hbDNNTensor> inputs(input_count);

    for (int i = 0; i < input_count; i++) {
        hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);
        auto &prop = inputs[i].properties;

        if (i == 0) { // Y 通道
            uint32_t mem_size = MODEL_H * ALIGNED_W;
            CHECK_RET(hbUCPMallocCached(&inputs[i].sysMem, mem_size, 0), "Malloc Y");
            inputs[i].sysMem.memSize = mem_size;
            prop.stride[0] = mem_size; 
            prop.stride[1] = ALIGNED_W;
            prop.stride[2] = 1; 
            prop.stride[3] = 1;
        } else { // UV 通道
            uint32_t mem_size = (MODEL_H / 2) * ALIGNED_W;
            CHECK_RET(hbUCPMallocCached(&inputs[i].sysMem, mem_size, 0), "Malloc UV");
            inputs[i].sysMem.memSize = mem_size;
            prop.stride[0] = mem_size; 
            prop.stride[1] = ALIGNED_W;
            prop.stride[2] = 2; 
            prop.stride[3] = 1;
        }
    }

    // --- 2. 图像处理 (BGR -> NV12) ---
    cv::Mat bgr = cv::imread(image_file);
    if (bgr.empty()) return -1;
    cv::Mat resized, yuv420p;
    cv::resize(bgr, resized, cv::Size(MODEL_W, MODEL_H)); // resized 用于后续叠加
    cv::cvtColor(resized, yuv420p, cv::COLOR_BGR2YUV_I420);

    uint8_t* y_src = yuv420p.data;
    uint8_t* u_src = y_src + (MODEL_W * MODEL_H);
    uint8_t* v_src = u_src + (MODEL_W * MODEL_H / 4);

    // 拷贝 Y
    uint8_t* y_dest = (uint8_t*)inputs[0].sysMem.virAddr;
    for (int r = 0; r < MODEL_H; r++) {
        std::memcpy(y_dest + r * ALIGNED_W, y_src + r * MODEL_W, MODEL_W);
    }
    // 拷贝 UV (交错模式)
    uint8_t* uv_dest = (uint8_t*)inputs[1].sysMem.virAddr;
    for (int r = 0; r < MODEL_H / 2; r++) {
        for (int c = 0; c < MODEL_W / 2; c++) {
            uv_dest[r * ALIGNED_W + c * 2]     = u_src[r * (MODEL_W / 2) + c];
            uv_dest[r * ALIGNED_W + c * 2 + 1] = v_src[r * (MODEL_W / 2) + c];
        }
    }

    for (int i = 0; i < input_count; i++) hbUCPMemFlush(&inputs[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);

    // --- 3. 输出 Tensor 准备 ---
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> outputs(output_count);
    hbDNNGetOutputTensorProperties(&outputs[0].properties, dnn_handle, 0);

    uint32_t out_stride_w = ALIGNED_W * sizeof(float);
    uint32_t channel_size = MODEL_H * out_stride_w;
    uint32_t out_total_size = 3 * channel_size;

    CHECK_RET(hbUCPMallocCached(&outputs[0].sysMem, out_total_size, 0), "Malloc Output");
    outputs[0].sysMem.memSize = out_total_size;

    outputs[0].properties.stride[0] = channel_size; 
    outputs[0].properties.stride[1] = out_stride_w;
    outputs[0].properties.stride[2] = sizeof(float);
    outputs[0].properties.stride[3] = sizeof(float);

    // --- 4. 推理 ---
    hbUCPTaskHandle_t task_handle = nullptr;
    CHECK_RET(hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle), "Inference");
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    hbUCPSubmitTask(task_handle, &sched);
    hbUCPWaitTaskDone(task_handle, 0);

    // --- 5. 后处理与叠加 ---
    hbUCPMemFlush(&(outputs[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    float* raw_ptr = (float*)outputs[0].sysMem.virAddr;

    int ch_offset = channel_size / sizeof(float);
    int row_offset = out_stride_w / sizeof(float);

    cv::Mat result(MODEL_H, MODEL_W, CV_8UC3);
    std::vector<cv::Vec3b> palette = { {128,128,128}, {255,0,0}, {0,0,255} };

    for (int y = 0; y < MODEL_H; y++) {
        for (int x = 0; x < MODEL_W; x++) {
            float v0 = raw_ptr[0 * ch_offset + y * row_offset + x];
            float v1 = raw_ptr[1 * ch_offset + y * row_offset + x];
            float v2 = raw_ptr[2 * ch_offset + y * row_offset + x];

            int max_id = (v1 > v0) ? (v1 > v2 ? 1 : 2) : (v0 > v2 ? 0 : 2);
            result.at<cv::Vec3b>(y, x) = palette[max_id];
        }
    }

    // --- 叠加原图 ---
    cv::Mat blended;
    // resized 是 BGR 格式的原图，result 是生成的 Mask
    // 0.7 是原图权重，0.3 是掩码权重，透明度可根据需要调整
    cv::addWeighted(resized, 0.7, result, 0.3, 0, blended);

    // 同时输出掩码图和叠加图
    cv::imwrite("finewaternet_mask.png", result);
    cv::imwrite("finewaternet_blended.png", blended);
    
    std::cout << "Success! output_final.png and output_blended.png generated." << std::endl;

    hbUCPReleaseTask(task_handle);
    for (auto &in : inputs) hbUCPFree(&in.sysMem);
    for (auto &out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);

    return 0;
}