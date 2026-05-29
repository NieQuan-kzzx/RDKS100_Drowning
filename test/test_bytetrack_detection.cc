// Attention: This program runs on RDK S100 board.
// D-Robotics S100 *.hbm 模型路径
// Path of D-Robotics S100 *.hbm model.
#define MODEL_PATH "/home/sunrise/Desktop/RDKS100_Drowning/models/bytetrack_s.hbm"
// 推理使用的测试图片路径
// Path of the test image used for inference.
#define TEST_IMG_PATH "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_mot.png"
// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0 
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE
// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "bytetrack_result.jpg"
// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 1
// NMS的阈值, 默认0.7
// Non-Maximum Suppression (NMS) threshold, default is 0.7
#define NMS_THRESHOLD 0.7
// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.15
// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16
// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0
// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0
// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0
// C/C++ Standard Libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
// Third Party Libraries
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
// RDK S100 UCP API
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
            return ret_code;                                                     \
        }                                                                        \
    } while (0);
// COCO Names
std::vector<std::string> object_names = {
    "pedestrian"
    //, "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor",
    // "person at surface", "person underwater"
    // "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    // "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    // "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    // "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    // "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    // "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    // "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    // "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    // "scissors", "teddy bear", "hair drier", "toothbrush"
};
// S100定制颜色
std::vector<cv::Scalar> rdk_colors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
};
// Softmax function for DFL calculation
void softmax(float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    } 
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}
int main()
{
    // Step 0: 加载S100 hbm模型
    // Step 0: Load S100 hbm model
    auto begin_time = std::chrono::system_clock::now();
    hbDNNPackedHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics S100 Quantize model time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 1: 打印基本信息
    // Step 1: Print basic information
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;
    // Step 2: 获取模型句柄
    // Step 2: Get model handle
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");
    if (model_count > 1) {
        std::cout << "This model file have more than 1 model, only use model 0." << std::endl;
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;
    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");
    // Step 3: 检查模型输入
    // Step 3: Check model input
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");
    if (input_count < 1) {
        std::cout << "S100 YOLO model should have at least 1 input, but got " << input_count << std::endl;
        return -1;
    } else if (input_count > 1) {
        std::cout << "S100 YOLO model has " << input_count << " inputs, using first input for inference" << std::endl;
    } 
    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");
    // S100 UCP 模型需要检测输入格式是否支持
    std::cout << "✓ input tensor type: " << input_properties.tensorType << std::endl; 
    // 检测输入格式是否为NV12 (type 3)
    if (input_properties.tensorType != 3) {
        std::cout << "[ERROR] This program only supports NV12 input (type 3), but got type: " << input_properties.tensorType << std::endl;
        return -1;
    }
    // 检测输入tensor布局为NCHW
    if (input_properties.validShape.numDimensions == 4) {
        // NCHW布局，H和W应该在维度1和2位置，且通道数应该为1
        int32_t channels = input_properties.validShape.dimensionSize[3];
        if (channels != 1) {
            std::cout << "[ERROR] This program expects NCHW layout with 1 channel, but got " << channels << " channels" << std::endl;
            return -1;
        }
        std::cout << "✓ input tensor layout: NCHW (verified)" << std::endl;
    } else {
        std::cout << "[ERROR] Expected 4D input tensor for NCHW layout, but got " << input_properties.validShape.numDimensions << "D" << std::endl;
        return -1;
    }
    // 获取输入尺寸
    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4) {
        input_H = input_properties.validShape.dimensionSize[1];
        input_W = input_properties.validShape.dimensionSize[2];
        std::cout << "✓ input tensor valid shape: (" 
                  << input_properties.validShape.dimensionSize[0] << ", "
                  << input_H << ", " << input_W << ", "
                  << input_properties.validShape.dimensionSize[3] << ")" << std::endl;
    } else {
        std::cout << "S100 YOLO model input should be 4D" << std::endl;
        return -1;
    }
    // Step 4: 检查模型输出 - ByteTrack 按照单输出格式导出
    // Step 4: Check model output - ByteTrack should have 1 output
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");
    if (output_count != 1) {
        std::cout << "ByteTrack model should have 1 output, but got " << output_count << std::endl;
        return -1;
    }
    std::cout << "✓ ByteTrack model has 1 output" << std::endl;

    // 打印输出信息
    std::cout << "\033[32m-> output tensors\033[0m" << std::endl;
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        std::cout << "output[" << i << "] valid shape: ("
                << output_properties.validShape.dimensionSize[0] << ", "
                << output_properties.validShape.dimensionSize[1] << ", "
                << output_properties.validShape.dimensionSize[2] << ", "
                << output_properties.validShape.dimensionSize[3] << "), ";
        std::cout << "QuantiType: " << output_properties.quantiType << std::endl;
    }

    // Step 5: 前处理 - 读取图像并转换为YUV420SP
    // Step 5: Preprocessing - Load image and convert to YUV420SP
    std::cout << "\033[32m-> Starting preprocessing\033[0m" << std::endl;
    cv::Mat img = cv::imread(TEST_IMG_PATH);
    if (img.empty()) {
        std::cout << "Failed to load image: " << TEST_IMG_PATH << std::endl;
        return -1;
    }
    std::cout << "✓ img path: " << TEST_IMG_PATH << std::endl;
    std::cout << "✓ img (rows, cols, channels): (" << img.rows << ", " << img.cols << ", " << img.channels() << ")" << std::endl;
    // 前处理参数
    float y_scale = 1.0, x_scale = 1.0;
    int x_shift = 0, y_shift = 0;
    cv::Mat resize_img;
    begin_time = std::chrono::system_clock::now();
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // LetterBox前处理
        float scale = std::min(1.0f * input_H / img.rows, 1.0f * input_W / img.cols);
        int new_w = int(img.cols * scale);
        int new_h = int(img.rows * scale);
        // 确保尺寸为偶数
        new_w = (new_w / 2) * 2;
        new_h = (new_h / 2) * 2;
        // 重新计算实际的缩放因子
        x_scale = 1.0f * new_w / img.cols;
        y_scale = 1.0f * new_h / img.rows;
        x_shift = (input_W - new_w) / 2;
        int x_other = input_W - new_w - x_shift;
        y_shift = (input_H - new_h) / 2;
        int y_other = input_H - new_h - y_shift;
        cv::Size targetSize(new_w, new_h);
        cv::resize(img, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        // Resize前处理
        cv::Size targetSize(input_W, input_H);
        cv::resize(img, resize_img, targetSize);
        y_scale = 1.0 * input_H / img.rows;
        x_scale = 1.0 * input_W / img.cols;
    }
    std::cout << "✓ y_scale = " << y_scale << ", x_scale = " << x_scale << std::endl;
    std::cout << "✓ y_shift = " << y_shift << ", x_shift = " << x_shift << std::endl;
    // BGR转YUV420SP (NV12)
    cv::Mat img_nv12;
    cv::Mat yuv_mat;
    cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int uv_height = input_H / 2;
    int uv_width = input_W / 2;
    int y_size = input_H * input_W;  
    // 复制Y平面
    memcpy(ynv12, yuv, y_size);   
    // 交错UV平面
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    std::cout << "\033[31m pre process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 6: 准备输入tensor
    // Step 6: Prepare input tensor
    std::vector<hbDNNTensor> input_tensors(input_count);
    
    // ByteTrack 模型只有 1 个输出
    int actual_output_count = 1; 
    std::vector<hbDNNTensor> output_tensors(actual_output_count);

    // 分配输入内存 (ByteTrack有2个输入：Y分量和UV分量)
    for (int i = 0; i < input_count; i++) {
        // 复制输入tensor属性
        input_tensors[i].properties = input_properties;
        int data_size;
        if (i == 0) {
            // 第一个输入：Y分量 608x1088x1
            data_size = input_H * input_W;
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = input_H;
            input_tensors[i].properties.validShape.dimensionSize[2] = input_W;
            input_tensors[i].properties.validShape.dimensionSize[3] = 1;

            input_tensors[i].properties.stride[3] = 1;
            input_tensors[i].properties.stride[2] = 1;
            input_tensors[i].properties.stride[1] = input_W;
            input_tensors[i].properties.stride[0] = input_W * input_H;
        } else {
            // 第二个输入：UV分量 304x544x2 (尺寸减半，2通道)
            int uv_h = input_H / 2;
            int uv_w = input_W / 2;
            data_size = uv_h * uv_w * 2;

            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = uv_w;
            input_tensors[i].properties.validShape.dimensionSize[3] = 2;

            input_tensors[i].properties.stride[3] = 1;
            input_tensors[i].properties.stride[2] = 2;
            input_tensors[i].properties.stride[1] = uv_w * 2;
            input_tensors[i].properties.stride[0] = uv_w * uv_h * 2;
        }

        // 分配内存
        hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0);
        std::cout << "✓ Input tensor " << i << " memory allocated: " << data_size << " bytes" << std::endl;

        // 复制数据
        if (i == 0) {
            memcpy(input_tensors[i].sysMem.virAddr, ynv12, input_H * input_W);
            std::cout << "✓ Y component data copied to tensor " << i << std::endl;
        } else {
            uint8_t *uv_src = ynv12 + input_H * input_W;
            memcpy(input_tensors[i].sysMem.virAddr, uv_src, data_size);
            std::cout << "✓ UV component data copied to tensor " << i << std::endl;
        }

        // 刷新内存
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // 分配输出内存 (只有1个输出)
    for (int i = 0; i < actual_output_count; i++) {
        hbDNNTensorProperties output_properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        output_tensors[i].properties = output_properties;
        
        int out_aligned_size = output_properties.alignedByteSize;
        hbUCPSysMem &mem = output_tensors[i].sysMem;
        hbUCPMallocCached(&mem, out_aligned_size, 0);
        std::cout << "✓ Output tensor " << i << " memory allocated: " << out_aligned_size << " bytes" << std::endl;
    }

    // Step 7: 推理
    // Step 7: Inference
    std::cout << "\033[32m-> Starting inference\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    // 生成任务句柄
    hbUCPTaskHandle_t task_handle = nullptr;
    int infer_ret = hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
    if (infer_ret != 0) {
        std::cout << "[ERROR] hbDNNInferV2 failed with error code: " << infer_ret << std::endl;
        return -1;
    }
    if (task_handle == nullptr) {
        std::cout << "[ERROR] task_handle is null after hbDNNInferV2" << std::endl;
        return -1;
    }
    std::cout << "✓ Inference task created successfully" << std::endl;
    // 设置UCP调度参数
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;  // 使用任意BPU核心
    // 提交任务到UCP
    int submit_ret = hbUCPSubmitTask(task_handle, &ctrl_param);
    if (submit_ret != 0) {
        std::cout << "[ERROR] hbUCPSubmitTask failed with error code: " << submit_ret << std::endl;
        return -1;
    }
    std::cout << "✓ Inference task submitted successfully" << std::endl;
    // 等待任务完成，设置合理的超时时间(10秒)
    int wait_ret = hbUCPWaitTaskDone(task_handle, 10000);
    if (wait_ret != 0) {
        std::cout << "[ERROR] hbUCPWaitTaskDone failed with error code: " << wait_ret << std::endl;
        return -1;
    }
    std::cout << "✓ Inference task completed successfully" << std::endl;
    std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
        
    // Step 8: 后处理 - ByteTrack (YOLOX架构) 真正的终局修复版
    // Step 8: Post-processing - The Real Final Fix
    std::cout << "\033[32m-> Starting post-processing\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();

    // 刷新BPU内存
    hbUCPMemFlush(&(output_tensors[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

    hbDNNTensorProperties &out_prop = output_tensors[0].properties;
    auto *raw_data = reinterpret_cast<uint8_t *>(output_tensors[0].sysMem.virAddr);

    std::vector<cv::Rect2d> all_bboxes;
    std::vector<float> all_scores;
    std::vector<int> all_ids;

    // --- 1. 构建 608x1088 输入尺寸下的 YOLOX 网格坐标 ---
    struct GridAndStride {
        int grid_x;
        int grid_y;
        int stride;
    };
    std::vector<GridAndStride> grid_strides;
    int strides[3] = {8, 16, 32};
    for (int s = 0; s < 3; s++) {
        int stride = strides[s];
        int grid_h = 608 / stride;
        int grid_w = 1088 / stride;
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                grid_strides.push_back({w, h, stride});
            }
        }
    }

    int num_preds = 13566;      // 锚点总数

    // --- 2. 利用底层 Stride 精确读取数据 ---
    auto get_value = [&](int i, int c) -> float {
        uint8_t* ptr = raw_data + i * out_prop.stride[1] + c * out_prop.stride[2];
        return *reinterpret_cast<float*>(ptr);
    };

    // 核心修复：BPU 将 6 通道输出填充到了 8 通道！
    // Channel 4 和 5 是 BPU 填充的垃圾数据 (min=0.00)
    // 真正的 obj 和 cls 被挤到了 Channel 6 和 7！
    // 因为 Sigmoid(0) = 0.5，背景框的分数为 0.5 * 0.5 = 0.25。
    // 为了过滤掉这成千上万个背景框，必须将阈值提高到 0.40 以上！
    float HARD_SCORE_THRESHOLD = 0.40f; 

    // --- 3. 遍历解码 ---
    for (int i = 0; i < num_preds; i++) {
        float dx = get_value(i, 0);
        float dy = get_value(i, 1);
        float dw = get_value(i, 2);
        float dh = get_value(i, 3);
        
        float obj_logit = get_value(i, 6);
        float cls_logit = get_value(i, 7);
        
        // 最终分数 = sigmoid(obj) * sigmoid(cls)
        float obj_conf = 1.0f / (1.0f + std::exp(-obj_logit));
        float cls_conf = 1.0f / (1.0f + std::exp(-cls_logit));
        float score = obj_conf * cls_conf;

        if (score >= HARD_SCORE_THRESHOLD) {
            int grid_x = grid_strides[i].grid_x;
            int grid_y = grid_strides[i].grid_y;
            int stride = grid_strides[i].stride;

            // YOLOX 官方标准解码公式
            float cx = (1.0f / (1.0f + std::exp(-dx)) * 2.0f - 0.5f + grid_x) * stride;
            float cy = (1.0f / (1.0f + std::exp(-dy)) * 2.0f - 0.5f + grid_y) * stride;
            float w  = std::exp(dw) * stride;
            float h  = std::exp(dh) * stride;

            // cxcywh 转 xyxy
            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            // 合法性检查
            if (x2 > x1 && y2 > y1 && w > 2.0f && h > 2.0f) {
                all_bboxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                all_scores.push_back(score);
                all_ids.push_back(0); // 单类模型
            }
        }
    }

    // 调试信息：打印前 5 个最高分的框
    std::cout << "\033[33m[DEBUG] Decoded boxes (Top 5 scores):\033[0m" << std::endl;
    std::vector<size_t> indices(all_scores.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return all_scores[a] > all_scores[b]; });
    for (int i = 0; i < std::min((size_t)5, indices.size()); ++i) {
        int idx = indices[i];
        std::cout << "Score: " << all_scores[idx] 
                  << ", Box: " << all_bboxes[idx].x << " " << all_bboxes[idx].y << " " 
                  << all_bboxes[idx].width << " " << all_bboxes[idx].height << std::endl;
    }



    // Step 9: 分类别 NMS 处理 (保持原逻辑不变)
    // Step 9: Class-wise NMS processing
    std::vector<std::vector<int>> nms_indices(CLASSES_NUM);
    int total_detections_after_nms = 0;    
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        std::vector<cv::Rect2d> class_bboxes;
        std::vector<float> class_scores;
        std::vector<int> original_indices;        
        for (size_t i = 0; i < all_bboxes.size(); i++) {
            if (all_ids[i] == cls_id) {
                class_bboxes.push_back(all_bboxes[i]);
                class_scores.push_back(all_scores[i]);
                original_indices.push_back(i);
            }
        }        
        if (!class_bboxes.empty()) {
            std::vector<int> class_nms_indices;
            cv::dnn::NMSBoxes(class_bboxes, class_scores, SCORE_THRESHOLD, NMS_THRESHOLD, class_nms_indices);            
            for (int idx : class_nms_indices) {
                nms_indices[cls_id].push_back(original_indices[idx]);
            }
            total_detections_after_nms += class_nms_indices.size();
        }
    }  
    std::cout << "✓ Total detections after NMS: " << total_detections_after_nms << std::endl;
    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 10: 绘制结果
    // Step 10: Draw results
    std::cout << "\033[32m-> Drawing results\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now(); 
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        for (int global_idx : nms_indices[cls_id]) {
            // 坐标还原到原始图像分辨率 (处理 LetterBox/Resize 偏移)
            float x1 = (all_bboxes[global_idx].x - x_shift) / x_scale;
            float y1 = (all_bboxes[global_idx].y - y_shift) / y_scale;
            float x2 = x1 + all_bboxes[global_idx].width / x_scale;
            float y2 = y1 + all_bboxes[global_idx].height / y_scale;

            // 越界边界检查
            x1 = std::max(0.0f, std::min((float)img.cols - 1, x1));
            y1 = std::max(0.0f, std::min((float)img.rows - 1, y1));
            x2 = std::max(0.0f, std::min((float)img.cols - 1, x2));
            y2 = std::max(0.0f, std::min((float)img.rows - 1, y2)); 

            // 绘制边界框
            cv::Scalar color = rdk_colors[cls_id % rdk_colors.size()];
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, LINE_SIZE); 
            
            // 绘制带有背景板的标签
            std::string label = object_names[cls_id] + " " + std::to_string(int(all_scores[global_idx] * 100)) + "%";
            int baseline;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_THICKNESS, &baseline); 
            
            // 自动调整标签位置防止超出顶部
            int label_y = (y1 - 10 > textSize.height) ? (y1 - 10) : (y1 + textSize.height + 10);
            cv::Point label_pos(x1, label_y);
            
            // // 绘制标签背景
            // cv::rectangle(img, label_pos + cv::Point(0, baseline), 
            //              label_pos + cv::Point(textSize.width, -textSize.height), color, cv::FILLED);
            // // 绘制黑色文字
            // cv::putText(img, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 0), FONT_THICKNESS); 
        }
    }
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) 
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
            << " ms\033[0m" << std::endl;
    // Step 11: 保存结果
    // Step 11: Save result
    cv::imwrite(IMG_SAVE_PATH, img);
    std::cout << "\033[32m✓ saved in path: \"" << IMG_SAVE_PATH << "\"\033[0m" << std::endl;
    // Step 12: 资源释放
    // Step 12: Release resources
    std::cout << "\033[32m-> Cleaning up resources\033[0m" << std::endl;    
    // 释放任务句柄
    hbUCPReleaseTask(task_handle);    
    // 释放输入内存
    for (int i = 0; i < input_count; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    // 释放输出内存
    for (int i = 0; i < output_count; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    // 释放模型
    hbDNNRelease(packed_dnn_handle);   
    std::cout << "\033[32m✓ Program completed successfully\033[0m" << std::endl;
    return 0;
}