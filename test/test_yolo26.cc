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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <sys/stat.h>

// COCO 80 class names
static const char* coco_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static void draw_detections(cv::Mat& image, const std::vector<Detection>& dets)
{
    double fontScale = image.cols / 800.0;
    int thickness = std::max(1, (int)(image.cols / 600));

    for (const auto& det : dets) {
        cv::Scalar color = rdk_colors[det.class_id % rdk_colors.size()];
        cv::Rect rect((int)det.bbox[0], (int)det.bbox[1],
                       (int)(det.bbox[2] - det.bbox[0]),
                       (int)(det.bbox[3] - det.bbox[1]));

        cv::rectangle(image, rect, color, 3);

        const char* name = (det.class_id < 80) ? coco_names[det.class_id] : "unknown";
        char txt[128];
        snprintf(txt, sizeof(txt), "%s: %.2f", name, det.score);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX,
                                              fontScale, thickness, &baseline);
        cv::rectangle(image,
                      cv::Point(rect.x, rect.y - text_size.height - 10),
                      cv::Point(rect.x + text_size.width, rect.y),
                      color, -1);
        cv::putText(image, txt, cv::Point(rect.x, rect.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}

int main(int argc, char* argv[])
{
    std::string model_path = "/home/sunrise/Desktop/yolo26/yolo26n_detect_nashm_640x640_nv12.hbm";
    std::string image_path = "/home/sunrise/Desktop/RDKS100_Drowning/tem/bus.jpg";
    std::string output_dir = "/home/sunrise/Desktop/output";

    if (argc >= 2) image_path = argv[1];
    if (argc >= 3) model_path = argv[2];

    // Create output directory
    mkdir(output_dir.c_str(), 0755);

    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: cannot read image " << image_path << std::endl;
        return 1;
    }

    // Init detector
    std::cout << "Loading model..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    YOLO26Detect detector(model_path, 80, 1, 0.5f, 0.45f);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Model loaded in " << load_ms << " ms" << std::endl;

    // Run inference
    std::cout << "Processing image: " << image_path << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();

    detector.pre_process(image);
    detector.infer();
    auto results = detector.post_process(0.5f, 0.45f, image.cols, image.rows);

    auto t3 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Detected " << results.size() << " objects in " << infer_ms << " ms" << std::endl;

    // Draw and save
    draw_detections(image, results);

    std::string result_path = output_dir + "/result.jpg";
    cv::imwrite(result_path, image);
    std::cout << "Result saved to: " << result_path << std::endl;

    return 0;
}
