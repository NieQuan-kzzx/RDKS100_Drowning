/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
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

/**
 * @class YOLO11_Pose
 * @brief Wrapper class for YOLOv11 pose estimation model inference.
 *
 * This class handles:
 * - Model loading and initialization
 * - Preprocessing input images into model-compatible tensors
 * - Running inference on BPU
 * - Postprocessing outputs into bounding boxes and keypoints
 */
class YOLO11_Pose
{
private:
    int model_count_;                                // Number of models loaded in the packed handle
    hbDNNPackedHandle_t packed_dnn_handle_;          // Packed DNN handle for multi-model usage
    hbDNNHandle_t dnn_handle_;                       // Single model handle
    int32_t input_count_;                            // Number of input tensors
    int32_t output_count_;                           // Number of output tensors
    std::vector<hbDNNTensor> input_tensors_;         // Model input tensors
    std::vector<hbDNNTensor> output_tensors_;        // Model output tensors
    int input_h_;                                    // Model input height
    int input_w_;                                    // Model input width

public:
    /**
     * @brief Construct YOLO11_Pose object and load the model.
     * @param[in] model_path Path to the BPU quantized *.hbm model file.
     */
    YOLO11_Pose(std::string model_path);

    /**
     * @brief Destructor - releases model handles and memory.
     */
    ~YOLO11_Pose();

    /**
     * @brief Preprocess an input image to match model input format (NV12, resize, letterbox, normalize).
     * @param[in] bgr_mat Input image in BGR format.
     */
    void pre_process(cv::Mat& bgr_mat);

    /**
     * @brief Run inference on the preprocessed input tensor.
     */
    void infer();

    /**
     * @brief Postprocess model outputs into detections and keypoints.
     * @param[in] score_thres       Confidence score threshold for filtering boxes.
     * @param[in] nms_thres         IoU threshold for Non-Maximum Suppression.
     * @param[in] kpt_conf_thres    Confidence threshold for keypoint filtering.
     * @param[in] img_w             Original image width.
     * @param[in] img_h             Original image height.
     * @return [out] Pair of:
     *         - std::vector<Detection>: Final bounding boxes with class info.
     *         - std::vector<std::vector<Keypoint>>: List of keypoints per detection.
     */
    std::pair<std::vector<Detection>, std::vector<std::vector<Keypoint>>>
    post_process(float score_thres, float nms_thres, float kpt_conf_thres, int img_w, int img_h);
};
