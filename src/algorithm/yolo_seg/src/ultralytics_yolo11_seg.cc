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

#include "ultralytics_yolo11_seg.hpp"
#include <omp.h>

/**
 * @brief Stride per detection head (from high to low resolution).
 */
std::vector<int> strides       = {8, 16, 32};

/**
 * @brief Feature-map grid size per detection head (e.g., for ~640 input: 80/40/20).
 */
std::vector<int> anchor_sizes  = {80, 40, 20};

/**
 * @brief Fixed bin offsets for DFL (0..15).
 */
std::vector<int> weights_static = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

/**
 * @brief Filter class logits and decode DFL boxes + MCES on one head.
 *
 * For each (n,h,w):
 * 1) Take argmax over classes from @p cls_tensor (float logits).
 * 2) Compare with @p conf_thres_raw (logit domain); skip if lower.
 * 3) Decode 4 sides (l,t,r,b) via DFL on @p bbox_tensor (int32, dequantized per-channel).
 * 4) Read 32-dim MCES vector from @p mces_tensor (int32, dequantized per-channel).
 * 5) Convert to (x1,y1,x2,y2) in input scale using @p stride.
 *
 * Parallelized across H×W using OpenMP; thread-local buffers merged at the end.
 *
 * @param cls_tensor   [in]  Classification logits tensor (N,H,W,C), dtype float.
 * @param bbox_tensor  [in]  Bounding-box distribution tensor (N,H,W,64), dtype int32.
 * @param mces_tensor  [in]  MCES tensor (N,H,W,32), dtype int32.
 * @param conf_thres_raw [in] Confidence threshold in logit space (pre-sigmoid).
 * @param grid_size    [in]  Feature map size for this head (e.g., 80/40/20).
 * @param stride       [in]  Input stride for this head (e.g., 8/16/32).
 * @param weights_static [in] DFL bin offsets (0..15).
 * @param detections   [out] Decoded detections appended here.
 * @param all_mces     [out] MCES vectors aligned with @p detections.
 */
void filter_and_decode_detections_mces(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    const hbDNNTensor& mces_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    const std::vector<int>& weights_static,
    std::vector<Detection>& detections,
    std::vector<std::vector<float>>& all_mces
) {
    detections.clear();
    all_mces.clear();

    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int N = shape.dimensionSize[0];
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3];

    const int64_t* stride_cls  = cls_tensor.properties.stride;
    const int64_t* stride_bbox = bbox_tensor.properties.stride;
    const int64_t* stride_mces = mces_tensor.properties.stride;

    const uint8_t* data_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* data_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);
    const uint8_t* data_mces = reinterpret_cast<const uint8_t*>(mces_tensor.sysMem.virAddr);

    // Thread-local caches to avoid contention
    std::vector<std::vector<Detection>>           thread_dets(omp_get_max_threads());
    std::vector<std::vector<std::vector<float>>>  thread_mces(omp_get_max_threads());

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();
            auto& dets_local = thread_dets[tid];
            auto& mces_local = thread_mces[tid];

            for (int n = 0; n < N; ++n) {
                size_t base_cls_offset  = n * stride_cls[0]  + h * stride_cls[1]  + w * stride_cls[2];
                size_t base_bbox_offset = n * stride_bbox[0] + h * stride_bbox[1] + w * stride_bbox[2];
                size_t base_mces_offset = n * stride_mces[0] + h * stride_mces[1] + w * stride_mces[2];

                // 1) Argmax over classes (float logits)
                float max_val = -1e30f;
                int max_id = 0;
                for (int c = 0; c < C; ++c) {
                    const float* ptr_cls = reinterpret_cast<const float*>(
                        data_cls + base_cls_offset + c * stride_cls[3]);
                    float val = *ptr_cls;
                    if (val > max_val) { max_val = val; max_id = c; }
                }
                if (max_val < conf_thres_raw) continue; // threshold in logit domain

                Detection det{};
                det.score = sigmoid(max_val);  // convert to probability
                det.class_id = max_id;

                // 2) Decode bbox via DFL (4 sides × 16 bins)
                float anchor_x = 0.5f + w;
                float anchor_y = 0.5f + h;
                float ltrb[4] = {0, 0, 0, 0};

                for (int side = 0; side < 4; ++side) {
                    float bins[16];
                    for (int bin = 0; bin < 16; ++bin) {
                        int channel = side * 16 + bin;
                        const int32_t* ptr_bbox = reinterpret_cast<const int32_t*>(
                            data_bbox + base_bbox_offset + channel * stride_bbox[3]);
                        bins[bin] = dequant_value(*ptr_bbox, channel, bbox_tensor.properties);
                    }
                    // Softmax over bins (max-trick for numerical stability)
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

                // 3) Read MCES vector (32 channels), dequantize per-channel
                std::vector<float> mces_vec(32);
                for (int d = 0; d < 32; ++d) {
                    const int32_t* ptr_mces = reinterpret_cast<const int32_t*>(
                        data_mces + base_mces_offset + d * stride_mces[3]);
                    mces_vec[d] = dequant_value(*ptr_mces, d, mces_tensor.properties);
                }

                dets_local.push_back(det);
                mces_local.push_back(std::move(mces_vec));
            }
        }
    }

    // Merge thread-local results
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
        all_mces.insert(all_mces.end(),
                        std::make_move_iterator(thread_mces[t].begin()),
                        std::make_move_iterator(thread_mces[t].end()));
    }
}

/**
 * @brief Class-wise NMS that also keeps MCES aligned with detections.
 *
 * @param detections [in]  All candidate detections (mixed classes).
 * @param mces       [in]  MCES vectors aligned with @p detections.
 * @param iou_thresh [in]  IoU threshold for suppression.
 * @return std::pair<std::vector<Detection>, std::vector<std::vector<float>>> [out]
 *         Kept detections and their MCES, order-aligned.
 */
std::pair<std::vector<Detection>, std::vector<std::vector<float>>>
nms_bboxes_mces(const std::vector<Detection>& detections,
                const std::vector<std::vector<float>>& mces,
                float iou_thresh = 0.7f)
{
    std::vector<Detection> kept_dets;
    std::vector<std::vector<float>> kept_mces;

    std::unordered_map<int, std::vector<size_t>> class_map; // class_id -> indices

    // Group by class (store indices to avoid copies)
    for (size_t i = 0; i < detections.size(); ++i) {
        class_map[detections[i].class_id].push_back(i);
    }

    for (auto& [cls_id, idx_list] : class_map) {
        // Sort indices by score desc
        std::sort(idx_list.begin(), idx_list.end(),
                  [&](size_t a, size_t b) { return detections[a].score > detections[b].score; });

        std::vector<bool> suppressed(idx_list.size(), false);

        for (size_t i = 0; i < idx_list.size(); ++i) {
            if (suppressed[i]) continue;

            size_t keep_idx = idx_list[i];
            kept_dets.push_back(detections[keep_idx]);
            kept_mces.push_back(mces[keep_idx]);

            for (size_t j = i + 1; j < idx_list.size(); ++j) {
                if (suppressed[j]) continue;
                if (iou(detections[keep_idx], detections[idx_list[j]]) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return {kept_dets, kept_mces};
}

/**
 * @brief Dequantize s16 tensor with per-N (axis 0) scale/zero-point into float.
 *
 * Output order is NHWC flattened (row-major).
 *
 * @param tensor [in]  Input tensor (NHWC), dtype int16, per-N scale/zp.
 * @return std::vector<float> [out] Dequantized data in NHWC order.
 */
std::vector<float> dequantize_s16_axis0(const hbDNNTensor& tensor) {
    const hbDNNTensorShape& shape = tensor.properties.validShape;
    int N = shape.dimensionSize[0];
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3];

    const int64_t* stride = tensor.properties.stride;
    const float* scaleData = tensor.properties.scale.scaleData;
    const int* zeroPointData = tensor.properties.scale.zeroPointData;

    const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(tensor.sysMem.virAddr);

    size_t total_elems = static_cast<size_t>(N) * H * W * C;
    std::vector<float> fp32_out(total_elems);

    // Parallelize over (n,h) for cache locality
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            float scale = scaleData ? scaleData[n] : 1.0f;
            int zp = (zeroPointData && tensor.properties.scale.zeroPointLen > 0)
                     ? zeroPointData[n] : 0;

            size_t n_offset  = n * stride[0];
            size_t nh_offset = n_offset + h * stride[1];

            for (int w = 0; w < W; ++w) {
                size_t nhw_offset = nh_offset + w * stride[2];
                const int16_t* row_ptr = reinterpret_cast<const int16_t*>(base_ptr + nhw_offset);

                size_t base_idx = ((size_t)n * H * W + (size_t)h * W + w) * C;

                // Linear access along C
                for (int c = 0; c < C; ++c) {
                    fp32_out[base_idx + c] = (static_cast<int>(row_ptr[c]) - zp) * scale;
                }
            }
        }
    }

    return fp32_out;
}

/**
 * @brief Decode per-instance masks from protos and per-detection MCES vectors.
 *
 * @param detections [in]  Final detections to decode masks for.
 * @param mces       [in]  MCES vectors aligned with @p detections.
 * @param protos     [in]  Prototype mask features in NHWC order (mask_h×mask_w×C).
 * @param input_w    [in]  Model input width.
 * @param input_h    [in]  Model input height.
 * @param mask_w     [in]  Prototype width.
 * @param mask_h     [in]  Prototype height.
 * @param mask_thresh[in]  Threshold for binarization.
 * @return std::vector<cv::Mat> [out] Per-detection masks in proto scale/crop (CV_8UC1 0/1).
 */
std::vector<cv::Mat> decode_masks(
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& mces,
    const std::vector<float>& protos, // NHWC: (mask_h, mask_w, C)
    int input_w, int input_h,
    int mask_w, int mask_h,
    float mask_thresh = 0.5f
) {
    std::vector<cv::Mat> masks;
    if (detections.empty() || mces.empty()) return masks;

    int C = static_cast<int>(mces[0].size()); // channels (assumed consistent)
    float x_scale = static_cast<float>(mask_w) / input_w;
    float y_scale = static_cast<float>(mask_h) / input_h;

    masks.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        const auto& mc  = mces[i];

        // 1) Scale bbox to proto size
        int x1_corp = static_cast<int>(det.bbox[0] * x_scale);
        int y1_corp = static_cast<int>(det.bbox[1] * y_scale);
        int x2_corp = static_cast<int>(det.bbox[2] * x_scale);
        int y2_corp = static_cast<int>(det.bbox[3] * y_scale);

        // Clamp to bounds
        x1_corp = std::max(0, std::min(mask_w, x1_corp));
        x2_corp = std::max(0, std::min(mask_w, x2_corp));
        y1_corp = std::max(0, std::min(mask_h, y1_corp));
        y2_corp = std::max(0, std::min(mask_h, y2_corp));

        int crop_h = y2_corp - y1_corp;
        int crop_w = x2_corp - x1_corp;
        if (crop_h <= 0 || crop_w <= 0) {
            masks.emplace_back(); // empty mask placeholder
            continue;
        }

        // 2) Allocate cropped mask
        cv::Mat mask(crop_h, crop_w, CV_8UC1, cv::Scalar(0));

        // 3) Linear combination (proto pixel · MCES), then threshold
        for (int yy = 0; yy < crop_h; ++yy) {
            for (int xx = 0; xx < crop_w; ++xx) {
                float sum_val = 0.0f;
                int proto_y = y1_corp + yy;
                int proto_x = x1_corp + xx;
                const float* proto_pixel = &protos[(proto_y * mask_w + proto_x) * C];

                for (int c = 0; c < C; ++c) {
                    sum_val += proto_pixel[c] * mc[c];
                }

                mask.at<uint8_t>(yy, xx) = (sum_val > mask_thresh) ? 1 : 0;
            }
        }

        masks.push_back(mask);
    }

    return masks;
}

/**
 * @brief Resize each mask to its detection box and optionally apply morphology.
 *
 * @param masks         [in]  Masks cropped in proto scale.
 * @param detections    [in]  Detections aligned with @p masks.
 * @param img_w         [in]  Original image width.
 * @param img_h         [in]  Original image height.
 * @param interpolation [in]  OpenCV interpolation flag (default Lanczos4).
 * @param do_morph      [in]  Whether to apply morphological open to clean edges.
 * @return std::vector<cv::Mat> [out] Masks resized to their boxes (CV_8UC1 0/1).
 */
std::vector<cv::Mat> resize_masks_to_boxes(
    const std::vector<cv::Mat>& masks,
    const std::vector<Detection>& detections,
    int img_w, int img_h,
    int interpolation = cv::INTER_LANCZOS4,
    bool do_morph = true)
{
    std::vector<cv::Mat> resized_masks;
    resized_masks.reserve(masks.size());

    for (size_t i = 0; i < masks.size(); ++i) {
        const auto& mask = masks[i];
        const auto& bbox = detections[i].bbox;

        // Clamp box to image bounds
        int x1 = std::max(static_cast<int>(bbox[0]), 0);
        int y1 = std::max(static_cast<int>(bbox[1]), 0);
        int x2 = std::min(static_cast<int>(bbox[2]), img_w);
        int y2 = std::min(static_cast<int>(bbox[3]), img_h);

        int target_w = std::max(x2 - x1, 1);
        int target_h = std::max(y2 - y1, 1);

        cv::Mat resized;
        cv::resize(mask, resized, cv::Size(target_w, target_h), 0, 0, interpolation);

        if (do_morph) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(resized, resized, cv::MORPH_OPEN, kernel);
        }

        resized_masks.push_back(resized);
    }

    return resized_masks;
}

/**
 * @brief Construct and initialize the YOLO11_Seg model from file.
 *
 * Loads model pack, retrieves model handle, queries I/O counts and tensor
 * properties, and allocates memory for all tensors.
 *
 * @param model_path [in] Path to the *.hbm model file.
 */
YOLO11_Seg::YOLO11_Seg(std::string model_path)
{
    auto modelFileName = model_path.c_str();

    const char **model_name_list = nullptr;

    // Initialize from model file(s)
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

    // Input props
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    // Output props
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache input size
    input_h_ = input_tensors_[0].properties.validShape.dimensionSize[1];
    input_w_ = input_tensors_[0].properties.validShape.dimensionSize[2];

    // Allocate memory
    prepare_input_tensor(input_tensors_);
    prepare_output_tensor(output_tensors_);
}

/**
 * @brief Destructor: free tensor memory and release model resources.
 */
YOLO11_Seg::~YOLO11_Seg()
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
 * @brief Preprocess BGR image to the model's expected input format.
 *
 * Letterbox-resizes to (input_w_, input_h_) and converts to NV12 tensor layout.
 *
 * @param bgr_mat [in] Input image (OpenCV BGR).
 */
void YOLO11_Seg::pre_process(cv::Mat& bgr_mat)
{
    cv::Mat resized_mat;
    resized_mat.create(input_h_, input_w_, bgr_mat.type()); // allocate target size
    letterbox_resize(bgr_mat, resized_mat);                  // keep aspect ratio

    bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_); // BGR->NV12
}

/**
 * @brief Execute inference and fill output tensors.
 *
 * Creates a task, submits to UCP scheduler, blocks until completion,
 * invalidates CPU cache for outputs, and releases the task handle.
 */
void YOLO11_Seg::infer()
{
    hbUCPTaskHandle_t task_handle{nullptr}; // [out] task handle

    // Create inference task
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors_.data(), input_tensors_.data(), dnn_handle_),
                        "hbDNNInferV2 failed");

    // Submit
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                        "hbUCPSubmitTask failed");

    // Wait
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    // Ensure CPU visibility
    for (int i = 0; i < output_count_; i++) {
        hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // Release
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
}

/**
 * @brief Post-process raw outputs into final detections and instance masks.
 *
 * Steps:
 * 1) Convert probability threshold to logit threshold.
 * 2) For each head (8/16/32), decode detections and MCES vectors.
 * 3) Concatenate all head results and apply class-wise NMS (keeping MCES).
 * 4) Dequantize prototype features, decode per-instance masks, and scale boxes back.
 * 5) Resize masks to their boxes in original image coordinates.
 *
 * @param score_thres [in]  Confidence threshold (probability domain).
 * @param nms_thres   [in]  IoU threshold for NMS.
 * @param img_w       [in]  Original image width (pixels).
 * @param img_h       [in]  Original image height (pixels).
 * @return std::pair<std::vector<Detection>, std::vector<cv::Mat>> [out]
 *         Final detections and their masks resized to boxes.
 */
std::pair<std::vector<Detection>, std::vector<cv::Mat>>
YOLO11_Seg::post_process(float score_thres, float nms_thres, int img_w, int img_h)
{
    // Probability -> logit (for raw logit comparison)
    float conf_thres_raw = -std::log(1.0f / score_thres - 1.0f);

    std::vector<Detection> all_detections;
    std::vector<std::vector<float>> all_mces_data;

    // Each head: [cls, bbox, mces]
    for (size_t s = 0; s < strides.size(); ++s) {
        const hbDNNTensor& cls_tensor  = output_tensors_[3*s + 0];
        const hbDNNTensor& bbox_tensor = output_tensors_[3*s + 1];
        const hbDNNTensor& mces_tensor = output_tensors_[3*s + 2];

        std::vector<Detection> dets;
        std::vector<std::vector<float>> mces;

        filter_and_decode_detections_mces(
            cls_tensor, bbox_tensor, mces_tensor,
            conf_thres_raw,
            anchor_sizes[s], strides[s],
            weights_static,
            dets, mces
        );

        // Merge head results
        all_detections.insert(all_detections.end(),
                              std::make_move_iterator(dets.begin()),
                              std::make_move_iterator(dets.end()));
        all_mces_data.insert(all_mces_data.end(),
                             std::make_move_iterator(mces.begin()),
                             std::make_move_iterator(mces.end()));
    }

    // NMS with MCES kept in sync
    auto [final_dets, final_mces] = nms_bboxes_mces(all_detections, all_mces_data, nms_thres);

    // Dequantize prototype features (NHWC), e.g., output_tensors_[9] as protos
    auto protos = dequantize_s16_axis0(output_tensors_[9]);

    // Decode per-instance masks in proto scale
    auto masks = decode_masks(final_dets, final_mces, protos, input_w_, input_h_, 160, 160, 0.5f);

    // Map boxes back to original image (inverse letterbox)
    scale_letterbox_bboxes_back(final_dets, img_w, img_h, input_w_, input_h_);

    // Resize masks to their boxes in original image coordinates
    std::vector<cv::Mat> resized_masks = resize_masks_to_boxes(masks, final_dets, img_w, img_h);

    return {final_dets, resized_masks};
}
