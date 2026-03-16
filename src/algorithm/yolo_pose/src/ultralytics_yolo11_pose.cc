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

#include "ultralytics_yolo11_pose.hpp"
#include <omp.h>

/**
 * @brief Stride per detection head (from high to low resolution).
 */
static std::vector<int> strides       = {8, 16, 32};

/**
 * @brief Feature-map grid size per detection head (e.g., ~640 input -> 80/40/20).
 */
static std::vector<int> anchor_sizes  = {80, 40, 20};

/**
 * @brief Fixed bin offsets for DFL (0..15).
 */
static std::vector<int> weights_static = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

/**
 * @brief Filter class logits, decode DFL boxes and keypoints on a single head.
 *
 * Workflow per location (n,h,w):
 * 1) Take the confidence from @p cls_tensor (float logit), compare to @p conf_thres_raw.
 * 2) Decode bbox l/t/r/b via DFL softmax expectation on @p bbox_tensor.
 * 3) Decode 17 keypoints from @p kpts_tensor (dx, dy, score) and map to input scale.
 *
 * Parallelized across H×W with OpenMP; merges thread-local buffers at the end.
 *
 * @param cls_tensor    [in]  Classification logits tensor (N,H,W,1), float.
 * @param bbox_tensor   [in]  BBox distribution tensor (N,H,W,64), float (DFL bins).
 * @param kpts_tensor   [in]  Keypoints tensor (N,H,W,51 = 17×(dx,dy,score)), float.
 * @param conf_thres_raw[in]  Confidence threshold in logit space (pre-sigmoid).
 * @param grid_size     [in]  Feature map size for this head (e.g., 80/40/20).
 * @param stride        [in]  Input stride for this head (e.g., 8/16/32).
 * @param weights_static[in]  DFL bin offsets (0..15).
 * @param detections    [out] Decoded detections appended here.
 * @param kpts          [out] Decoded keypoints aligned with @p detections.
 */
void filter_and_decode_detections_kpts(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    const hbDNNTensor& kpts_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    const std::vector<int>& weights_static,
    std::vector<Detection>& detections,
    std::vector<std::vector<Keypoint>>& kpts
) {
    detections.clear();
    kpts.clear();

    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int N = shape.dimensionSize[0];
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3]; // for cls head, typically 1

    const int64_t* stride_cls  = cls_tensor.properties.stride;
    const int64_t* stride_bbox = bbox_tensor.properties.stride;
    const int64_t* stride_kpts = kpts_tensor.properties.stride;

    const uint8_t* data_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* data_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);
    const uint8_t* data_kpts = reinterpret_cast<const uint8_t*>(kpts_tensor.sysMem.virAddr);

    // Thread-local caches to reduce contention
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Detection>>              thread_dets(nthreads);
    std::vector<std::vector<std::vector<Keypoint>>>  thread_kpts(nthreads);

    // Parallel over spatial grid
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();
            auto& dets_local = thread_dets[tid];
            auto& kpts_local = thread_kpts[tid];

            for (int n = 0; n < N; ++n) {
                size_t base_cls_offset  = n * stride_cls[0]  + h * stride_cls[1]  + w * stride_cls[2];
                size_t base_bbox_offset = n * stride_bbox[0] + h * stride_bbox[1] + w * stride_bbox[2];
                size_t base_kpts_offset = n * stride_kpts[0] + h * stride_kpts[1] + w * stride_kpts[2];

                // 1) confidence (float logit); cls head is 1-channel in this variant
                const float* ptr_cls = reinterpret_cast<const float*>(data_cls + base_cls_offset);
                float val = *ptr_cls;
                if (val < conf_thres_raw) continue;

                Detection det{};
                det.score = sigmoid(val);  // convert to probability
                det.class_id = 0;          // single-class pose (person) by default

                // 2) DFL bbox decode (4 sides × 16 bins)
                float anchor_x = 0.5f + w;
                float anchor_y = 0.5f + h;
                float ltrb[4] = {0, 0, 0, 0};

                for (int side = 0; side < 4; ++side) {
                    float bins[16];
                    for (int bin = 0; bin < 16; ++bin) {
                        int channel = side * 16 + bin;
                        const float* ptr_bbox = reinterpret_cast<const float*>(
                            data_bbox + base_bbox_offset + channel * stride_bbox[3]);
                        bins[bin] = *ptr_bbox;
                    }
                    // softmax with max-trick
                    float max_bin = bins[0];
                    for (int i = 1; i < 16; ++i) if (bins[i] > max_bin) max_bin = bins[i];

                    float sum = 0.0f;
                    float probs[16];
                    for (int i = 0; i < 16; ++i) { probs[i] = std::exp(bins[i] - max_bin); sum += probs[i]; }
                    for (int i = 0; i < 16; ++i) { ltrb[side] += probs[i] * weights_static[i] / sum; }
                }

                // (anchor, ltrb) -> (x1,y1,x2,y2) at input scale
                det.bbox[0] = (anchor_x - ltrb[0]) * stride;
                det.bbox[1] = (anchor_y - ltrb[1]) * stride;
                det.bbox[2] = (anchor_x + ltrb[2]) * stride;
                det.bbox[3] = (anchor_y + ltrb[3]) * stride;

                // 3) Keypoints decode: (dx, dy, score) per keypoint
                std::vector<Keypoint> kpts_vec(17);
                for (int k = 0; k < 17; ++k) {
                    const int base_ch = 3 * k; // layout: [dx, dy, score]
                    const float* ptr_k = reinterpret_cast<const float*>(
                        data_kpts + base_kpts_offset + base_ch * stride_kpts[3]);

                    // Offsets are predicted on the grid; map to input-space pixels
                    kpts_vec[k].x     = (ptr_k[0] * 2.0f + (anchor_x - 0.5f)) * stride;
                    kpts_vec[k].y     = (ptr_k[1] * 2.0f + (anchor_y - 0.5f)) * stride;
                    kpts_vec[k].score = ptr_k[2];
                }

                dets_local.push_back(det);
                kpts_local.push_back(std::move(kpts_vec));
            }
        }
    }

    // Merge thread-local results
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
        kpts.insert(kpts.end(),
                    std::make_move_iterator(thread_kpts[t].begin()),
                    std::make_move_iterator(thread_kpts[t].end()));
    }
}

/**
 * @brief Class-wise NMS for detections while keeping keypoints aligned.
 *
 * @param detections [in]  Candidate detections from all heads.
 * @param kpts       [in]  Keypoints aligned with @p detections.
 * @param iou_thresh [in]  IoU threshold for suppression (default 0.7).
 * @return std::pair<std::vector<Detection>, std::vector<std::vector<Keypoint>>> [out]
 *         Kept detections and their keypoints, order-aligned.
 */
std::pair<std::vector<Detection>,  std::vector<std::vector<Keypoint>>>
nms_bboxes_kpts(
    const std::vector<Detection>& detections,
    std::vector<std::vector<Keypoint>>& kpts,
    float iou_thresh = 0.7f)
{
    std::vector<Detection> kept_dets;
    std::vector<std::vector<Keypoint>> kept_kpts;

    std::unordered_map<int, std::vector<size_t>> class_map; // class_id -> indices

    // Group by class (store indices)
    for (size_t i = 0; i < detections.size(); ++i) {
        class_map[detections[i].class_id].push_back(i);
    }

    for (auto& [cls_id, idx_list] : class_map) {
        // Sort by score desc
        std::sort(idx_list.begin(), idx_list.end(),
                  [&](size_t a, size_t b) { return detections[a].score > detections[b].score; });

        std::vector<bool> suppressed(idx_list.size(), false);

        for (size_t i = 0; i < idx_list.size(); ++i) {
            if (suppressed[i]) continue;

            size_t keep_idx = idx_list[i];
            kept_dets.push_back(detections[keep_idx]);
            kept_kpts.push_back(kpts[keep_idx]);

            for (size_t j = i + 1; j < idx_list.size(); ++j) {
                if (suppressed[j]) continue;
                if (iou(detections[keep_idx], detections[idx_list[j]]) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return {kept_dets, kept_kpts};
}

/**
 * @brief Construct and initialize the YOLO11_Pose model from file.
 *
 * Loads model pack, retrieves model handle, queries I/O counts and tensor
 * properties, and allocates memory for all tensors.
 *
 * @param model_path [in] Path to the *.hbm model file.
 */
YOLO11_Pose::YOLO11_Pose(std::string model_path)
{
    auto modelFileName = model_path.c_str();

    const char **model_name_list = nullptr;

    // Initialize model pack
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &modelFileName, 1),
                        "hbDNNInitializeFromFiles failed");
    // Model list
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");
    // Model handle
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");
    // I/O counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle_),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle_),
                        "hbDNNGetOutputCount failed");

    // Tensor descriptors
    input_tensors_.resize(input_count_);
    output_tensors_.resize(output_count_);

    // Input tensor properties
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    // Output tensor properties
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache expected input size (H, W)
    input_h_ = input_tensors_[0].properties.validShape.dimensionSize[1];
    input_w_ = input_tensors_[0].properties.validShape.dimensionSize[2];

    // Allocate memory
    prepare_input_tensor(input_tensors_);
    prepare_output_tensor(output_tensors_);
}

/**
 * @brief Destructor: release tensor memory and model resources.
 */
YOLO11_Pose::~YOLO11_Pose()
{
    // Free input mem
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors_[i].sysMem));
    }
    // Free output mem
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors_[i].sysMem));
    }
    // Release model
    hbDNNRelease(packed_dnn_handle_);
}

/**
 * @brief Preprocess a BGR image to the model's expected input format.
 *
 * Performs letterbox resize to (input_w_, input_h_) and converts to NV12 tensor.
 *
 * @param bgr_mat [in] Input image in OpenCV BGR format.
 */
void YOLO11_Pose::pre_process(cv::Mat& bgr_mat)
{
    // Letterbox resize to model input size
    cv::Mat resized_mat;
    resized_mat.create(input_h_, input_w_, bgr_mat.type());
    letterbox_resize(bgr_mat, resized_mat);

    // BGR -> NV12 tensor copy
    bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_);
}

/**
 * @brief Execute inference on prepared inputs and fill outputs.
 *
 * Creates a task, submits to UCP scheduler, waits for completion, invalidates
 * CPU cache for outputs, and releases the task handle.
 */
void YOLO11_Pose::infer()
{
    hbUCPTaskHandle_t task_handle{nullptr}; // inference task handle

    // Create inference task
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors_.data(), input_tensors_.data(), dnn_handle_),
                        "hbDNNInferV2 failed");

    // Submit task to BPU
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                        "hbUCPSubmitTask failed");

    // Wait for completion
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    // Ensure CPU can read outputs
    for (int i = 0; i < output_count_; i++) {
        hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // Release handle
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
}

/**
 * @brief Post-process outputs: merge heads, NMS, and map boxes/keypoints back to original image.
 *
 * Steps:
 * 1) Convert probability threshold to logit threshold.
 * 2) For each head (8/16/32), decode detections & keypoints.
 * 3) Concatenate all head results, apply class-wise NMS.
 * 4) Rescale boxes and keypoints back (inverse letterbox).
 *
 * @param score_thres     [in]  Confidence threshold (probability domain).
 * @param nms_thres       [in]  IoU threshold for NMS.
 * @param kpt_conf_thres  [in]  Keypoint confidence threshold (may be used by caller).
 * @param img_w           [in]  Original image width (pixels).
 * @param img_h           [in]  Original image height (pixels).
 * @return std::pair<std::vector<Detection>, std::vector<std::vector<Keypoint>>> [out]
 *         Final detections and their keypoints in original image coordinates.
 */
std::pair<std::vector<Detection>, std::vector<std::vector<Keypoint>>>
YOLO11_Pose::post_process(float score_thres, float nms_thres, float kpt_conf_thres, int img_w, int img_h)
{
    // Probability -> logit for comparison with raw logits
    float conf_thres_raw = -std::log(1.0f / score_thres - 1.0f);

    std::vector<Detection> all_detections;
    std::vector<std::vector<Keypoint>> all_kpts_data;

    // Each head contributes [cls, bbox, kpts]
    for (size_t s = 0; s < strides.size(); ++s) {
        const hbDNNTensor& cls_tensor  = output_tensors_[3*s + 0];
        const hbDNNTensor& bbox_tensor = output_tensors_[3*s + 1];
        const hbDNNTensor& kpts_tensor = output_tensors_[3*s + 2];

        std::vector<Detection> dets;
        std::vector<std::vector<Keypoint>> keypoint;

        filter_and_decode_detections_kpts(cls_tensor, bbox_tensor, kpts_tensor,conf_thres_raw,
                                          anchor_sizes[s], strides[s], weights_static, dets, keypoint);

        // Merge results
        all_detections.insert(all_detections.end(),
                              std::make_move_iterator(dets.begin()),
                              std::make_move_iterator(dets.end()));
        all_kpts_data.insert(all_kpts_data.end(),
                             std::make_move_iterator(keypoint.begin()),
                             std::make_move_iterator(keypoint.end()));
    }

    // NMS with keypoints kept aligned
    auto [final_dets, final_kpts] = nms_bboxes_kpts(all_detections, all_kpts_data, nms_thres);

    // Map boxes & keypoints back to original image coordinates (inverse letterbox)
    scale_letterbox_bboxes_back(final_dets, img_w, img_h, input_w_, input_h_);
    scale_keypoints_back_letterbox(final_kpts, img_w, img_h, input_w_, input_h_);

    return {final_dets, final_kpts};
}
