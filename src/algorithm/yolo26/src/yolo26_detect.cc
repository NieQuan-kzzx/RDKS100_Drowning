/*
 * Copyright (c) 2025, D-Robotics Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yolo26_detect.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// ============================================================================
// Helper: read a single float value from tensor at given N,H,W,C offset,
// handling both float32 and quantized (int8/int32) tensor types.
// ============================================================================
static inline float read_tensor_val(const hbDNNTensor& tensor, int n, int h, int w, int c)
{
    const hbDNNTensorShape& shape = tensor.properties.validShape;
    const int64_t* stride = tensor.properties.stride;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(tensor.sysMem.virAddr);

    size_t offset = n * stride[0] + h * stride[1] + w * stride[2] + c * stride[3];

    if (tensor.properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
        tensor.properties.quantiType == NONE) {
        return reinterpret_cast<const float*>(base)[offset / sizeof(float)];
    } else if (tensor.properties.tensorType == HB_DNN_TENSOR_TYPE_S32) {
        int32_t raw = reinterpret_cast<const int32_t*>(base)[offset / sizeof(int32_t)];
        return dequant_value(raw, c, tensor.properties);
    } else if (tensor.properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
        int16_t raw = reinterpret_cast<const int16_t*>(base)[offset / sizeof(int16_t)];
        return dequant_value(raw, c, tensor.properties);
    } else {
        // uint8 or int8
        int8_t raw = static_cast<int8_t>(base[offset]);
        return dequant_value(raw, c, tensor.properties);
    }
}

// ============================================================================
// decode_layer: decode a single decoupled head (YOLO26 anchor-free)
// ============================================================================
void decode_layer(const hbDNNTensor& cls_tensor,
                  const hbDNNTensor& box_tensor,
                  float conf_thres_raw,
                  int stride,
                  std::vector<Detection>& detections)
{
    const hbDNNTensorShape& cls_shape = cls_tensor.properties.validShape;
    int H = cls_shape.dimensionSize[1];
    int W = cls_shape.dimensionSize[2];
    int C = cls_shape.dimensionSize[3];

    // Thread-local buffers for OpenMP
    std::vector<std::vector<Detection>> thread_dets(omp_get_max_threads());

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();
            auto& dets_local = thread_dets[tid];

            // 1. Argmax over classes and check logit threshold
            float max_val = -1e30f;
            int max_id = 0;
            for (int c = 0; c < C; ++c) {
                float val = read_tensor_val(cls_tensor, 0, h, w, c);
                if (val > max_val) {
                    max_val = val;
                    max_id = c;
                }
            }

            if (max_val < conf_thres_raw) continue;

            // 2. Decode anchor-free box: [l, t, r, b] from grid center
            float anchor_x = static_cast<float>(w) + 0.5f;
            float anchor_y = static_cast<float>(h) + 0.5f;

            float l = read_tensor_val(box_tensor, 0, h, w, 0);
            float t = read_tensor_val(box_tensor, 0, h, w, 1);
            float r = read_tensor_val(box_tensor, 0, h, w, 2);
            float b = read_tensor_val(box_tensor, 0, h, w, 3);

            Detection det{};
            det.bbox[0] = (anchor_x - l) * stride;  // x1
            det.bbox[1] = (anchor_y - t) * stride;  // y1
            det.bbox[2] = (anchor_x + r) * stride;  // x2
            det.bbox[3] = (anchor_y + b) * stride;  // y2
            det.score = sigmoid(max_val);
            det.class_id = max_id;

            dets_local.push_back(det);
        }
    }

    // Merge thread-local results
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
    }
}

// ============================================================================
// YOLO26Detect constructor
// ============================================================================
YOLO26Detect::YOLO26Detect(const std::string& model_path,
                           int classes_num,
                           int resize_type,
                           float score_thres,
                           float nms_thres)
    : model_count_(0), dnn_handle_(nullptr),
      input_count_(0), output_count_(0),
      input_h_(0), input_w_(0),
      classes_num_(classes_num), resize_type_(resize_type),
      score_thres_(score_thres), nms_thres_(nms_thres),
      strides_({8, 16, 32})
{
    auto modelFileName = model_path.c_str();
    const char** model_name_list = nullptr;

    // Load model
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &modelFileName, 1),
                        "hbDNNInitializeFromFiles failed");

    // Get model name
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");

    // Get model handle
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");

    // Query I/O counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle_),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle_),
                        "hbDNNGetOutputCount failed");

    // Prepare tensor descriptors
    input_tensors_.resize(input_count_);
    output_tensors_.resize(output_count_);

    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache input dimensions (NHWC layout)
    input_h_ = input_tensors_[0].properties.validShape.dimensionSize[1];
    input_w_ = input_tensors_[0].properties.validShape.dimensionSize[2];

    // Allocate tensor memory
    prepare_input_tensor(input_tensors_);
    prepare_output_tensor(output_tensors_);

    printf("[YOLO26] Model loaded: %dx%d, %d outputs, %d classes\n",
           input_w_, input_h_, output_count_, classes_num_);
}

// ============================================================================
// YOLO26Detect destructor
// ============================================================================
YOLO26Detect::~YOLO26Detect()
{
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors_[i].sysMem));
    }
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors_[i].sysMem));
    }
    hbDNNRelease(packed_dnn_handle_);
}

// ============================================================================
// pre_process
// ============================================================================
void YOLO26Detect::pre_process(cv::Mat& bgr_mat)
{
    if (resize_type_ == 1) {
        // Letterbox resize
        cv::Mat resized_mat;
        resized_mat.create(input_h_, input_w_, bgr_mat.type());
        letterbox_resize(bgr_mat, resized_mat);
        bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_);
    } else {
        // Direct resize (stretch)
        cv::Mat resized_mat;
        cv::resize(bgr_mat, resized_mat, cv::Size(input_w_, input_h_));
        bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_);
    }
}

// ============================================================================
// infer
// ============================================================================
void YOLO26Detect::infer()
{
    hbUCPTaskHandle_t task_handle{nullptr};

    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors_.data(),
                                     input_tensors_.data(), dnn_handle_),
                        "hbDNNInferV2 failed");

    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                        "hbUCPSubmitTask failed");

    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    for (int i = 0; i < output_count_; i++) {
        hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
}

// ============================================================================
// post_process: decode decoupled heads, NMS, rescale
// ============================================================================
std::vector<Detection> YOLO26Detect::post_process(float score_thres, float nms_thres,
                                                   int img_w, int img_h)
{
    float conf_thres_raw = -std::log(1.0f / score_thres - 1.0f);

    std::vector<Detection> all_detections;

    // YOLO26 outputs: [Cls_8, Box_8, Cls_16, Box_16, Cls_32, Box_32]
    int num_heads = output_count_ / 2;

    for (int s = 0; s < num_heads; ++s) {
        const hbDNNTensor& cls_tensor  = output_tensors_[2 * s + 0];
        const hbDNNTensor& box_tensor = output_tensors_[2 * s + 1];

        int grid_h = cls_tensor.properties.validShape.dimensionSize[1];
        int grid_w = cls_tensor.properties.validShape.dimensionSize[2];
        int stride_h = input_h_ / grid_h;
        int stride_w = input_w_ / grid_w;
        int stride = (stride_h + stride_w) / 2;

        std::vector<Detection> dets;
        decode_layer(cls_tensor, box_tensor, conf_thres_raw, stride, dets);

        all_detections.insert(all_detections.end(),
                              std::make_move_iterator(dets.begin()),
                              std::make_move_iterator(dets.end()));
    }

    // NMS
    auto results = nms_bboxes(all_detections, nms_thres);

    // Rescale to original image coordinates
    if (resize_type_ == 1) {
        scale_letterbox_bboxes_back(results, img_w, img_h, input_w_, input_h_);
    } else {
        // Direct resize: simple scale
        float scale_x = static_cast<float>(img_w) / input_w_;
        float scale_y = static_cast<float>(img_h) / input_h_;
        for (auto& det : results) {
            det.bbox[0] *= scale_x;
            det.bbox[2] *= scale_x;
            det.bbox[1] *= scale_y;
            det.bbox[3] *= scale_y;
            // Clamp
            det.bbox[0] = std::max(0.0f, std::min(det.bbox[0], static_cast<float>(img_w)));
            det.bbox[2] = std::max(0.0f, std::min(det.bbox[2], static_cast<float>(img_w)));
            det.bbox[1] = std::max(0.0f, std::min(det.bbox[1], static_cast<float>(img_h)));
            det.bbox[3] = std::max(0.0f, std::min(det.bbox[3], static_cast<float>(img_h)));
        }
    }

    return results;
}
