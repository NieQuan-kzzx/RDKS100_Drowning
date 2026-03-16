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

#include "ultralytics_yolo11.hpp"
#include <omp.h>

/**
 * @brief Stride per detection head (from high to low resolution).
 */
static std::vector<int> strides       = {8, 16, 32};

/**
 * @brief Feature-map grid size per detection head.
 * Typical heads are 80x80 / 40x40 / 20x20.
 */
static std::vector<int> anchor_sizes  = {80, 40, 20};

/**
 * @brief Fixed bin offsets for DFL (0..15).
 */
static std::vector<int> weights_static = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

/**
 * @brief Filter class logits and decode DFL box distributions for one head.
 *
 * Workflow:
 * 1) For each (n,h,w), take argmax over C classes from cls_tensor.
 * 2) Compare the max logit with conf_thres_raw (logit domain); skip if lower.
 * 3) For each of 4 sides, read 16-bin logits from bbox_tensor, softmax, and
 *    take expectation with weights_static to get a distance.
 * 4) Convert (anchor_x,anchor_y,l,t,r,b) to (x1,y1,x2,y2) in input scale by stride.
 *
 * Parallelized with OpenMP across HxW.
 *
 * @param cls_tensor      [in]  Classification logits, shape (N,H,W,C), dtype float.
 * @param bbox_tensor     [in]  Box distributions, shape (N,H,W,64), dtype (usually) int32.
 * @param conf_thres_raw  [in]  Confidence threshold in logit space (pre-sigmoid).
 * @param grid_size       [in]  Feature map size for this head (e.g., 80/40/20).
 * @param stride          [in]  Input stride for this head (e.g., 8/16/32).
 * @param weights_static  [in]  DFL bin offsets (0..15).
 * @param detections      [out] Decoded detections appended here.
 */
void filter_and_decode_detections(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    const std::vector<int>& weights_static,
    std::vector<Detection>& detections)
{
    detections.clear();

    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int N = shape.dimensionSize[0];
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3];

    const int64_t* stride_cls  = cls_tensor.properties.stride;
    const int64_t* stride_bbox = bbox_tensor.properties.stride;

    const uint8_t* data_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* data_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);

    // Thread-local buffers to reduce contention
    std::vector<std::vector<Detection>> thread_dets(omp_get_max_threads());

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();
            auto& dets_local = thread_dets[tid];

            for (int n = 0; n < N; ++n) {
                size_t base_cls_offset  = n * stride_cls[0]  + h * stride_cls[1]  + w * stride_cls[2];
                size_t base_bbox_offset = n * stride_bbox[0] + h * stride_bbox[1] + w * stride_bbox[2];

                // Argmax over classes (float logits)
                float max_val = -1e30f;
                int max_id = 0;
                for (int c = 0; c < C; ++c) {
                    const float* ptr_cls = reinterpret_cast<const float*>(
                        data_cls + base_cls_offset + c * stride_cls[3]);
                    float val = *ptr_cls;
                    if (val > max_val) {
                        max_val = val;
                        max_id = c;
                    }
                }
                if (max_val < conf_thres_raw) continue;  // threshold in logit domain

                Detection det{};
                det.score = sigmoid(max_val);            // convert to probability
                det.class_id = max_id;

                // Decode bbox via DFL (4 sides × 16 bins)
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

                    // Softmax over bins (max trick for stability)
                    float max_bin = bins[0];
                    for (int i = 1; i < 16; ++i) if (bins[i] > max_bin) max_bin = bins[i];
                    float sum = 0.0f;
                    float probs[16];
                    for (int i = 0; i < 16; ++i) {
                        probs[i] = std::exp(bins[i] - max_bin);
                        sum += probs[i];
                    }
                    for (int i = 0; i < 16; ++i) {
                        ltrb[side] += probs[i] * weights_static[i] / sum; // expectation
                    }
                }

                // (anchor, ltrb) -> (x1,y1,x2,y2) at input scale
                det.bbox[0] = (anchor_x - ltrb[0]) * stride;
                det.bbox[1] = (anchor_y - ltrb[1]) * stride;
                det.bbox[2] = (anchor_x + ltrb[2]) * stride;
                det.bbox[3] = (anchor_y + ltrb[3]) * stride;

                dets_local.push_back(det);
            }
        }
    }

    // Merge thread-local vectors
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
    }
}

/**
 * @brief Construct a new YOLO11 object and initialize DNN resources.
 *
 * Loads the model from disk, retrieves model handle, queries input/output
 * tensor counts and properties, and allocates memory for all tensors.
 *
 * @param model_path [in] Path to the YOLOv11 *.hbm model file.
 */
YOLO11::YOLO11(std::string model_path)
{
    auto modelFileName = model_path.c_str();

    const char **model_name_list = nullptr;

    // Initialize from model file(s)
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &modelFileName, 1),
                        "hbDNNInitializeFromFiles failed");
    // Model names
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

    // Prepare tensor descriptors
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

    // Cache expected input size
    input_h_ = input_tensors_[0].properties.validShape.dimensionSize[1];
    input_w_ = input_tensors_[0].properties.validShape.dimensionSize[2];

    // Allocate memory for all tensors
    prepare_input_tensor(input_tensors_);
    prepare_output_tensor(output_tensors_);
}

/**
 * @brief Destructor: free tensor memory and release model resources.
 */
YOLO11::~YOLO11()
{
    // Free input memory
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors_[i].sysMem));
    }
    // Free output memory
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors_[i].sysMem));
    }
    // Release packed model handle
    hbDNNRelease(packed_dnn_handle_);
}

/**
 * @brief Preprocess a BGR image to match YOLOv11 input format.
 *
 * Letterbox-resizes to (input_w_, input_h_) and converts the image to
 * NV12 tensor layout expected by the runtime.
 *
 * @param bgr_mat [in] Input image in OpenCV BGR format.
 */
void YOLO11::pre_process(cv::Mat& bgr_mat)
{
    // Letterbox resize to model input size
    cv::Mat resized_mat;
    resized_mat.create(input_h_, input_w_, bgr_mat.type());
    letterbox_resize(bgr_mat, resized_mat);

    // BGR -> NV12 input tensor
    bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_);
}

/**
 * @brief Execute inference on current inputs and fill outputs.
 *
 * Creates a task, submits to UCP scheduler, waits for completion,
 * invalidates CPU cache for output tensors, and releases the task handle.
 */
void YOLO11::infer()
{
    hbUCPTaskHandle_t task_handle{nullptr};

    // Create inference task
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors_.data(), input_tensors_.data(), dnn_handle_),
                        "hbDNNInferV2 failed");

    // Submit to BPU scheduler
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                        "hbUCPSubmitTask failed");

    // Wait until finished (0 => block)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    // Ensure CPU sees fresh outputs
    for (int i = 0; i < output_count_; i++) {
        hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // Release handle
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
}

/**
 * @brief Decode & filter predictions from all heads, run NMS, and rescale boxes.
 *
 * Steps:
 * 1) Convert probability threshold to logit threshold.
 * 2) For each head (8/16/32), filter classes and decode DFL boxes.
 * 3) Concatenate detections from all heads.
 * 4) Apply NMS on concatenated detections.
 * 5) Map boxes back to original image coordinates (inverse letterbox).
 *
 * @param score_thres [in]  Confidence threshold in probability domain.
 * @param nms_thres   [in]  IoU threshold for NMS.
 * @param img_w       [in]  Original image width, in pixels.
 * @param img_h       [in]  Original image height, in pixels.
 * @return std::vector<Detection> [out] Final detections after NMS and rescaling.
 */
std::vector<Detection> YOLO11::post_process(float score_thres, float nms_thres, int img_w, int img_h)
{
    // Probability -> logit for comparison with raw logits
    float conf_thres_raw = -std::log(1.0f / score_thres - 1.0f);

    std::vector<Detection> all_detections;

    // Each head contributes [cls_head, bbox_head]
    for (size_t s = 0; s < strides.size(); ++s) {
        const hbDNNTensor& cls_tensor  = output_tensors_[2*s + 0];
        const hbDNNTensor& bbox_tensor = output_tensors_[2*s + 1];

        std::vector<Detection> dets;
        filter_and_decode_detections(cls_tensor, bbox_tensor, conf_thres_raw,
                                     anchor_sizes[s], strides[s], weights_static, dets);

        // Merge head results
        all_detections.insert(all_detections.end(),
                              std::make_move_iterator(dets.begin()),
                              std::make_move_iterator(dets.end()));
    }

    // NMS across all heads
    auto results = nms_bboxes(all_detections, nms_thres);

    // Rescale to original image (undo letterbox)
    scale_letterbox_bboxes_back(results, img_w, img_h, input_w_, input_h_);

    return results;
}
