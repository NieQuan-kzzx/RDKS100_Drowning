#include <iostream>
#include <string>
#include "gflags/gflags.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "ultralytics_yolo11_seg.hpp"
#include "common_utils.hpp"

DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11n-seg.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(test_img, "/home/sunrise/Desktop/RDKS100_Drowning/tem/pose_test.png",
              "Path to load the test image.");
DEFINE_string(label_file, "/app/res/labels/coco_classes.names",
              "Path to load COCO label mapping file.");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.7, "IoU threshold for Non-Maximum Suppression.");

/**
 * @brief Entry point: run YOLOv11 instance segmentation on a single image.
 *
 * Pipeline:
 * 1) Parse CLI flags
 * 2) Load YOLO11_Seg model
 * 3) Load & preprocess input image (letterbox + NV12 tensor)
 * 4) Run inference
 * 5) Postprocess: decode boxes/masks, NMS, rescale to original size
 * 6) Draw boxes/masks/contours and save the visualization
 *
 * @param argc [in] Number of command-line arguments.
 * @param argv [in] Array of command-line argument strings.
 * @return int  [out] Process exit code (0 on success).
 */
int main(int argc, char **argv)
{
    // Parse command-line arguments
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    // Step 1: Construct model wrapper and load the quantized *.hbm
    YOLO11_Seg yolo11_seg = YOLO11_Seg(FLAGS_model_path);

    // Step 2: Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);
    int img_w = image.cols;  // cache original width
    int img_h = image.rows;  // cache original height

    // Preprocess -> NV12 tensor (letterbox to model input size)
    yolo11_seg.pre_process(image);
    std::cout << "pre_process finished" << std::endl;

    // Step 3: Run inference on BPU
    yolo11_seg.infer();
    std::cout << "infer finished" << std::endl;

    // Step 4: Decode predictions, NMS, masks; rescale to original size
    auto [final_dets, resized_masks] =
        yolo11_seg.post_process(FLAGS_score_thres, FLAGS_nms_thres, img_w, img_h);
    std::cout << "post_process finished" << std::endl;

    // Step 5: Visualization
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    draw_boxes(image, final_dets, class_names, rdk_colors);               // draw bbox + labels
    draw_masks(image, final_dets, resized_masks, rdk_colors, 0.4f);       // alpha-blend masks
    draw_contours(image, final_dets, resized_masks, rdk_colors, 1);       // outline instance contours

    // Persist result
    cv::imwrite("result.jpg", image);
    std::cout << "[Saved] Result saved to: result.jpg" << std::endl;

    return 0;
}
