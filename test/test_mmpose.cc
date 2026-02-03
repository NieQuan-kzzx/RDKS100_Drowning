#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define CHECK_RET(ret, msg) \
    if ((ret) != 0) { \
        std::cerr << "[ERROR] " << msg << " | Code: " << (ret) << std::endl; \
        return -1; \
    }

// 头部特征点误差较大，其他关节点误差较小，猜测可能是图压缩之后，头部特征全部丢失了，占比像素太小

struct Point { float x, y, score; };

int main() {
    const char* model_file = "/home/sunrise/Desktop/RDKS100_Drowning/models/Hrnet.hbm"; 
    const char* image_file = "/home/sunrise/Desktop/RDKS100_Drowning/tem/pose.png";

    // 1. 初始化
    hbDNNPackedHandle_t packed_dnn_handle = nullptr;
    CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN");
    const char **model_name_list = nullptr;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    // 2. 准备输入 (手动填充 Stride 解决 -100001 报错)
    int input_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    std::vector<hbDNNTensor> inputs(input_count);
    for (int i = 0; i < input_count; i++) {
        std::memset(&inputs[i], 0, sizeof(hbDNNTensor));
        hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);
        uint32_t mem_size = (i == 0) ? 49152 : 24576;
        inputs[i].properties.stride[0] = mem_size;
        inputs[i].properties.stride[1] = 192;
        inputs[i].properties.stride[2] = (i == 0) ? 1 : 2;
        inputs[i].properties.stride[3] = 1;
        CHECK_RET(hbUCPMallocCached(&inputs[i].sysMem, mem_size, 0), "Malloc Input");
        inputs[i].sysMem.memSize = mem_size;
    }

    // 3. 准备输出
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> outputs(output_count);
    for (int i = 0; i < output_count; i++) {
        std::memset(&outputs[i], 0, sizeof(hbDNNTensor));
        hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);
        uint32_t total_out_size = 17 * 64 * 48 * sizeof(float); // 基础大小
        // 如果系统要求的 alignedByteSize 更大，按大的来
        if(outputs[i].properties.alignedByteSize > total_out_size) 
            total_out_size = outputs[i].properties.alignedByteSize;
            
        CHECK_RET(hbUCPMallocCached(&outputs[i].sysMem, total_out_size, 0), "Malloc Output");
        outputs[i].sysMem.memSize = total_out_size;
    }

    // 4. Letterbox 预处理
    cv::Mat ori_img = cv::imread(image_file);
    if (ori_img.empty()) return -1;

    float scale = std::min(192.0f / ori_img.cols, 256.0f / ori_img.rows);
    int nw = (int)(ori_img.cols * scale);
    int nh = (int)(ori_img.rows * scale);
    int dx = (192 - nw) / 2;
    int dy = (256 - nh) / 2;

    std::cout << "[DEBUG] Scale: " << scale << " | NW: " << nw << " NH: " << nh << " | DX: " << dx << " DY: " << dy << std::endl;

    cv::Mat resized, canvas(256, 192, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(ori_img, resized, cv::Size(nw, nh));
    resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

    // 数据拷贝到 NV12
    cv::Mat yuv420p;
    cv::cvtColor(canvas, yuv420p, cv::COLOR_BGR2YUV_I420);
    std::memcpy(inputs[0].sysMem.virAddr, yuv420p.data, 49152);
    uint8_t* u_src = yuv420p.data + 49152;
    uint8_t* v_src = u_src + 12288;
    uint8_t* uv_dest = (uint8_t*)inputs[1].sysMem.virAddr;
    for (int i = 0; i < 12288; i++) { uv_dest[i*2] = u_src[i]; uv_dest[i*2+1] = v_src[i]; }

    for (int i = 0; i < input_count; i++) hbUCPMemFlush(&inputs[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);

    // 5. 推理
    hbUCPTaskHandle_t task_handle = nullptr;
    CHECK_RET(hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle), "Inference");
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    hbUCPSubmitTask(task_handle, &sched);
    hbUCPWaitTaskDone(task_handle, 0);

    // 6. 后处理
    // hbUCPMemFlush(&outputs[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    // float* raw_ptr = (float*)outputs[0].sysMem.virAddr;

    // // 显式定义常量，确保计算使用浮点数
    // const int out_c = 17;
    // const int out_h = 64;
    // const int out_w = 48;
    // const float stride_ratio = 4.0f; // Heatmap到Input的倍率

    // std::vector<Point> pts(out_c);

    // std::cout << "\n[FINAL DEBUG] Optimized Mapping:" << std::endl;
    // std::cout << "------------------------------------------------------" << std::endl;

    // for (int c = 0; c < out_c; c++) {
    //     float max_val = -100.0f;
    //     int max_x = 0, max_y = 0;
        
    //     // 紧凑寻址
    //     float* ch_ptr = raw_ptr + (c * out_h * out_w); 

    //     for (int h = 0; h < out_h; h++) {
    //         for (int w = 0; w < out_w; w++) {
    //             float val = ch_ptr[h * out_w + w]; 
    //             if (val > max_val) {
    //                 max_val = val;
    //                 max_x = w;
    //                 max_y = h;
    //             }
    //         }
    //     }

    //     // --- 亚像素能量重心修正 ---
    //     float final_x = (float)max_x;
    //     float final_y = (float)max_y;

    //     if (max_x > 0 && max_x < (out_w - 1) && max_y > 0 && max_y < (out_h - 1)) {
    //         float l = ch_ptr[max_y * out_w + (max_x - 1)];
    //         float r = ch_ptr[max_y * out_w + (max_x + 1)];
    //         float u = ch_ptr[(max_y - 1) * out_w + max_x];
    //         float d = ch_ptr[(max_y + 1) * out_w + max_x];
            
    //         // 使用更平滑的梯度修正，而不只是固定的 0.25
    //         // 这能解决 ID 0 和 ID 2 重叠的问题，让它们在亚像素层面分开
    //         final_x += (r - l) * 0.25f; 
    //         final_y += (d - u) * 0.25f;
    //     }

    //     // --- 核心坐标变换逻辑 ---
    //     // 1. 在Input空间(192x256)还原坐标
    //     // 注意：去掉 +0.5f，因为 BPU 硬件 resize 通常是 Top-left 对齐
    //     float x_in_input = final_x * stride_ratio;
    //     float y_in_input = final_y * stride_ratio;

    //     // 2. 逆向 Letterbox 映射
    //     // 必须使用 float 类型的 dx/dy 保证除法精度
    //     pts[c].x = (x_in_input - (float)dx) / scale;
    //     pts[c].y = (y_in_input - (float)dy) / scale;
    //     pts[c].score = max_val;

    //     // 打印前5个头部点位，观察它们是否由于亚像素修正而分开了
    //     if (c < 5) {
    //         printf("ID %d | Score: %.2f | Heatmap: (%d,%d) | Sub-pixel: (%.2f,%.2f) | Final: (%.1f, %.1f)\n", 
    //             c, max_val, max_x, max_y, final_x, final_y, pts[c].x, pts[c].y);
    //     }
    // }
    // std::cout << "------------------------------------------------------" << std::endl;

    // hbUCPMemFlush(&outputs[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    // float* raw_ptr = (float*)outputs[0].sysMem.virAddr;

    // int out_c = 17, out_h = 64, out_w = 48;
    // std::vector<Point> pts(out_c);

    // std::cout << "\n[DEBUG] Keypoint Analysis:" << std::endl;
    // std::cout << "------------------------------------------------------" << std::endl;
    // std::cout << "ID | Score | Heatmap(x,y) | Final Ori(x,y) | Offset" << std::endl;

    // for (int c = 0; c < out_c; c++) {
    //     float max_val = -100.0f;
    //     int max_x = 0, max_y = 0;
    //     float* ch_ptr = raw_ptr + (c * out_h * out_w); 

    //     for (int h = 0; h < out_h; h++) {
    //         for (int w = 0; w < out_w; w++) {
    //             float val = ch_ptr[h * out_w + w]; 
    //             if (val > max_val) {
    //                 max_val = val; max_x = w; max_y = h;
    //             }
    //         }
    //     }

    //     // 亚像素修正
    //     float final_x = (float)max_x;
    //     float final_y = (float)max_y;
    //     float sub_x = 0, sub_y = 0;

    //     if (max_x > 0 && max_x < (out_w - 1) && max_y > 0 && max_y < (out_h - 1)) {
    //         float dx_grad = ch_ptr[max_y * out_w + (max_x + 1)] - ch_ptr[max_y * out_w + (max_x - 1)];
    //         float dy_grad = ch_ptr[(max_y + 1) * out_w + max_x] - ch_ptr[(max_y - 1) * out_w + max_x];
            
    //         sub_x = (dx_grad > 0) ? 0.25f : (dx_grad < 0 ? -0.25f : 0.0f);
    //         sub_y = (dy_grad > 0) ? 0.25f : (dy_grad < 0 ? -0.25f : 0.0f);
    //         final_x += sub_x;
    //         final_y += sub_y;
    //     }

    //     // 坐标还原 (加入 0.5 像素中心对齐)
    //     pts[c].x = ((final_x + 0.5f) * 4.0f - dx) / scale;
    //     pts[c].y = ((final_y + 0.5f) * 4.0f - dy) / scale;
    //     pts[c].score = max_val;

    //     // 打印每个点的详细数据
    //     if (c < 5) { // 重点打印头部点：0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳
    //         printf("%2d | %5.2f | (%2d, %2d)    | (%7.1f, %7.1f) | (%.2f, %.2f)\n", 
    //                c, max_val, max_x, max_y, pts[c].x, pts[c].y, sub_x, sub_y);
    //     }
    // }
    // std::cout << "------------------------------------------------------" << std::endl;

    hbUCPMemFlush(&outputs[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    float* raw_ptr = (float*)outputs[0].sysMem.virAddr;

    int out_c = 17, out_h = 64, out_w = 48;
    std::vector<Point> pts(out_c);

    for (int c = 0; c < out_c; c++) {
        float max_val = -100.0f;
        int max_x = 0, max_y = 0;
        
        // 强制使用紧凑寻址：直接按 C * H * W 偏移
        float* ch_ptr = raw_ptr + (c * out_h * out_w); 

        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                // 直接使用 w 寻址，不使用 Stride
                float val = ch_ptr[h * out_w + w]; 
                if (val > max_val) {
                    max_val = val;
                    max_x = w;
                    max_y = h;
                }
            }
        }

        // 坐标映射
        pts[c].x = (max_x * 4.0f - dx) / scale;
        pts[c].y = (max_y * 4.0f - dy) / scale;
        pts[c].score = max_val;
    }

    // 7. 渲染
    for (int i = 0; i < 17; i++) {
        if (pts[i].score > -1.0f) { // HRNet经过量化后阈值通常较低
            cv::circle(ori_img, cv::Point((int)pts[i].x, (int)pts[i].y), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    cv::imwrite("mmpose_result.png", ori_img);

    // 释放资源
    hbUCPReleaseTask(task_handle);
    for (auto &in : inputs) hbUCPFree(&in.sysMem);
    for (auto &out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);
    return 0;
}