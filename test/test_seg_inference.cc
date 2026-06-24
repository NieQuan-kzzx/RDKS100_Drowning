#include <iostream>
#include <string>
#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>
#include "YoloSeg.h"

DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/Water_zuoshiyan.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(video_path, "/home/sunrise/Desktop/RDKS100_Drowning/tem/test_06.mp4",
              "Path to load the test video file.");
DEFINE_string(output_path, "water_seg_result_3.mp4",
              "Path to save the output video file.");

int main(int argc, char **argv)
{
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    Inf::YoloSeg seg;
    seg.setLabels({"water"});
    if (!seg.init(FLAGS_model_path)) {
        std::cerr << "[Error] Failed to init YoloSeg" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(FLAGS_video_path);
    if (!cap.isOpened()) {
        std::cerr << "[Error] Could not open video: " << FLAGS_video_path << std::endl;
        return -1;
    }

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;

    cv::VideoWriter writer(FLAGS_output_path,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(frame_w, frame_h));
    if (!writer.isOpened()) {
        std::cerr << "[Error] Could not open video writer." << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
        frame_count++;
        std::cout << "--- Frame " << frame_count << " ---" << std::endl;

        auto results = seg.run(frame);
        seg.draw(frame, results);

        std::cout << "  detections: " << results.size() << std::endl;

        writer.write(frame);
    }

    cap.release();
    writer.release();

    std::cout << "[Saved] " << FLAGS_output_path << std::endl;
    return 0;
}
