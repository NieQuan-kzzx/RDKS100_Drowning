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
 * @brief Container for a detection and its associated MCES feature vector.
 */
struct DetectionWithMCES {
    Detection det;              // Detection box + class/score
    std::vector<float> mces;    // MCES feature vector (length 32)
};

/**
 * @class YOLO11_Seg
 * @brief Wrapper for loading, running, and post-processing a YOLOv11 instance-segmentation model
 *        via the D-Robotics DNN API.
 *
 * Responsibilities:
 * - Model loading and initialization
 * - Input preprocessing (resize/letterbox, pixel-format conversion)
 * - Inference execution
 * - Post-processing: decode boxes/masks, score filtering, NMS, rescaling to original image
 */
class YOLO11_Seg
{
    private:
        int model_count_;                             // Number of models in the packed handle
        hbDNNPackedHandle_t packed_dnn_handle_;       // Packed DNN handle for one or more models
        hbDNNHandle_t dnn_handle_;                    // DNN handle for the YOLOv11-Seg model
        int32_t input_count_;                         // Number of model inputs
        int32_t output_count_;                        // Number of model outputs
        std::vector<hbDNNTensor> input_tensors_;      // Input tensor
        std::vector<hbDNNTensor> output_tensors_;     // Output tensor
        int input_h_;                                 // Model expected input height (pixels)
        int input_w_;                                 // Model expected input width  (pixels)

    public:
        /**
         * @brief Construct and initialize the YOLO11_Seg model from file.
         * @param model_path [in] Path to the *.hbm model file.
         */
        YOLO11_Seg(std::string model_path);

        /**
         * @brief Destroy the YOLO11_Seg object and release DNN resources.
         */
        ~YOLO11_Seg();

        /**
         * @brief Preprocess a BGR image into the model's expected input format.
         *
         * Typical steps:
         * - Letterbox resize to (input_w_, input_h_)
         * - Convert to the required pixel format (e.g., NV12)
         * - Normalize/quantize if required
         *
         * @param bgr_mat [in] Input image in OpenCV BGR format.
         */
        void pre_process(cv::Mat& bgr_mat);

        /**
         * @brief Run inference using the prepared input tensor(s).
         *
         * Allocates a task, submits to the scheduler, waits for completion,
         * and makes output tensor data CPU-visible.
         */
        void infer();

        /**
         * @brief Post-process raw model outputs into final detections and instance masks.
         *
         * The returned pair contains:
         * - First: vector of final detections after score thresholding and NMS (boxes, class IDs, scores)
         * - Second: vector of binary/soft masks (cv::Mat), aligned with the detections vector
         *
         * @param score_thres [in] Confidence score threshold for filtering predictions.
         * @param nms_thres   [in] IoU threshold for Non-Maximum Suppression (NMS).
         * @param img_w       [in] Original image width (for rescaling boxes/masks back).
         * @param img_h       [in] Original image height (for rescaling boxes/masks back).
         * @return std::pair<std::vector<Detection>, std::vector<cv::Mat>> [out]
         *         Final detections and their corresponding masks.
         */
        std::pair<std::vector<Detection>, std::vector<cv::Mat>>
        post_process(float score_thres, float nms_thres, int img_w, int img_h);
};
