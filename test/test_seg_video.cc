#include <iostream>
#include <string>
#include "gflags/gflags.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "ultralytics_yolo11_seg.hpp"
#include "common_utils.hpp"

// 修改或新增相关的命令行参数
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/Water_zuoshiyan.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(video_path, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_06.mp4",
              "Path to load the test video file (or camera index like '0').");
DEFINE_string(output_path, "water_seg_result_3.mp4", 
              "Path to save the processed output video.");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_water_seg.names",
              "Path to load COCO label mapping file.");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.7, "IoU threshold for Non-Maximum Suppression.");

/**
 * @brief Entry point: run YOLOv11 instance segmentation on a video stream.
 */
int main(int argc, char **argv)
{
    // 解析命令行参数
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    // Step 1: 初始化模型（在循环外只加载一次）
    YOLO11_Seg yolo11_seg = YOLO11_Seg(FLAGS_model_path);

    // 加载类别标签
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    // Step 2: 打开输入视频文件或摄像头
    cv::VideoCapture cap(FLAGS_video_path);
    if (!cap.isOpened()) {
        std::cerr << "[Error] Could not open input video: " << FLAGS_video_path << std::endl;
        return -1;
    }

    // 获取视频的基本属性
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps   = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0; // 防止获取不到 FPS 时引发错误

    std::cout << "[Info] Video Loaded: " << frame_w << "x" << frame_h << " @ " << fps << " FPS" << std::endl;

    // Step 3: 初始化视频写入器 (使用 mp4v 编码保存为 mp4)
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); 
    writer.open(FLAGS_output_path, fourcc, fps, cv::Size(frame_w, frame_h), true);
    
    if (!writer.isOpened()) {
        std::cerr << "[Error] Could not open video writer for saving." << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;

    // Step 4: 循环读取并处理每一帧
    while (true) {
        cap >> frame; // 读取当前帧
        if (frame.empty()) {
            std::cout << "[Info] Reached the end of the video or empty frame." << std::endl;
            break; // 视频结束或读取失败则退出循环
        }

        frame_count++;
        std::cout << "--- Processing Frame: " << frame_count << " ---" << std::endl;

        // 1) 前处理（尺寸缩放、Letterbox、转换为 NV12 格式等）
        yolo11_seg.pre_process(frame);

        // 2) BPU 硬件推理
        yolo11_seg.infer();

        // 3) 后处理（解码预测结果、NMS、将 Mask 缩放回原图大小）
        auto [final_dets, resized_masks] =
            yolo11_seg.post_process(FLAGS_score_thres, FLAGS_nms_thres, frame_w, frame_h);

        // 4) 可视化绘制（直接在原帧上绘制）
        draw_boxes(frame, final_dets, class_names, rdk_colors);               // 绘制目标框和标签
        draw_masks(frame, final_dets, resized_masks, rdk_colors, 0.4f);       // 混合半透明 Mask
        draw_contours(frame, final_dets, resized_masks, rdk_colors, 1);       // 绘制边缘轮廓

        // 5) 将处理后的帧写入输出视频文件
        writer.write(frame);

        // （可选）如果是带屏幕的桌面环境，可以取消注释下方代码进行实时预览
        /*
        cv::imshow("YOLOv11 Seg Video Test", frame);
        if (cv::waitKey(1) == 'q') { // 按 'q' 键提前退出
            break;
        }
        */
    }

    // Step 5: 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "[Saved] Processed video saved to: " << FLAGS_output_path << std::endl;

    return 0;
}