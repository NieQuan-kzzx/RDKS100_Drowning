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

#pragma once

#include "common_utils.hpp"
#include "preprocess_utils.hpp"
#include "postprocess_utils.hpp"
#include <vector>
#include <string>

/**
 * @brief Decode a single decoupled detection head (YOLO26 anchor-free).
 *
 * Filters classification logits by raw logit threshold, applies sigmoid
 * only to valid candidates, and decodes anchor-free LTRB box format.
 *
 * @param cls_tensor   [in]  Classification logits, shape (1, H, W, C).
 * @param box_tensor   [in]  Box distances, shape (1, H, W, 4) as [l, t, r, b].
 * @param conf_thres_raw [in]  Confidence threshold in logit space.
 * @param stride       [in]  Feature map stride.
 * @param detections   [out] Decoded detections appended here.
 */
void decode_layer(const hbDNNTensor& cls_tensor,
                  const hbDNNTensor& box_tensor,
                  float conf_thres_raw,
                  int stride,
                  std::vector<Detection>& detections);

/**
 * @class YOLO26Detect
 * @brief YOLO26 object detection wrapper using Horizon DNN API.
 *
 * Provides a complete inference pipeline: preprocess -> infer -> postprocess.
 * YOLO26 uses decoupled heads (separate Cls and Box) with anchor-free decoding.
 */
class YOLO26Detect
{
private:
    int model_count_;
    hbDNNPackedHandle_t packed_dnn_handle_;
    hbDNNHandle_t dnn_handle_;
    int32_t input_count_;
    int32_t output_count_;
    std::vector<hbDNNTensor> input_tensors_;
    std::vector<hbDNNTensor> output_tensors_;
    int input_h_;
    int input_w_;
    int classes_num_;
    int resize_type_;
    float score_thres_;
    float nms_thres_;
    std::vector<int> strides_;

public:
    /**
     * @brief Construct and load the YOLO26 model.
     * @param model_path   Path to the .hbm model file.
     * @param classes_num  Number of classes (default: 80).
     * @param resize_type  0=stretch, 1=letterbox (default: 1).
     * @param score_thres  Confidence threshold (default: 0.25).
     * @param nms_thres    NMS IoU threshold (default: 0.45).
     */
    YOLO26Detect(const std::string& model_path,
                 int classes_num = 80,
                 int resize_type = 1,
                 float score_thres = 0.25f,
                 float nms_thres = 0.45f);

    ~YOLO26Detect();

    /**
     * @brief Preprocess a BGR image for inference.
     * @param bgr_mat Input image in BGR format.
     */
    void pre_process(cv::Mat& bgr_mat);

    /**
     * @brief Execute model inference.
     */
    void infer();

    /**
     * @brief Postprocess raw outputs into detection results.
     * @param score_thres Confidence threshold override.
     * @param nms_thres   NMS IoU threshold override.
     * @param img_w       Original image width.
     * @param img_h       Original image height.
     * @return Vector of detections after NMS and coordinate rescaling.
     */
    std::vector<Detection> post_process(float score_thres, float nms_thres,
                                        int img_w, int img_h);

    /**
     * @brief Get model input height.
     */
    int getInputH() const { return input_h_; }

    /**
     * @brief Get model input width.
     */
    int getInputW() const { return input_w_; }
};
