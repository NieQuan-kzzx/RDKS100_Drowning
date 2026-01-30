#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <deque>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // 必须包含，用于 NMS

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"
#include "BYTETracker.hpp"
#include "SAHI.h" // 包含你提供的 SAHI 头文件

DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/ultralytics_YOLO.hbm", "Path to BPU model");
DEFINE_string(input_video, "/home/sunrise/Desktop/RDKS100_Drowning/tem/1.mp4", "Path to input video");
DEFINE_string(output_video, "sahi_result.mp4", "Path to save output");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_coco.names", "Path to labels");
DEFINE_double(score_thres, 0.25, "Score threshold");
DEFINE_double(nms_thres, 0.45, "NMS IoU threshold"); // SAHI 建议全局 NMS 稍微严一点

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    YOLO11 yolo11 = YOLO11(FLAGS_model_path);
    BYTETracker tracker;
    
    // 初始化 SAHI：切片 640x640，重叠率 20%
    SAHI sahi_engine(640, 640, 0.2f, 0.2f);

    cv::VideoCapture cap(FLAGS_input_video);
    if (!cap.isOpened()) return -1;

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(FLAGS_output_video, cv::VideoWriter::fourcc('m','p','4','v'), 
                          cap.get(cv::CAP_PROP_FPS), cv::Size(frame_w, frame_h));

    cv::Mat frame;
    int frame_idx = 0;
    std::deque<double> fps_history;

    while (cap.read(frame))
    {
        frame_idx++;
        if (frame.empty()) break;
        auto frame_start = std::chrono::steady_clock::now();

        // ---------------- SAHI 推理开始 ----------------
        std::vector<cv::Rect> bboxes_for_nms;
        std::vector<float> scores_for_nms;
        std::vector<int> class_ids_for_nms;

        // 1. 计算切片区域
        auto regions = sahi_engine.calculateSliceRegions(frame.rows, frame.cols);

        for (const auto& reg_pair : regions) {
            cv::Rect region = reg_pair.first;
            
            // 2. 裁剪子图 (注意：这里直接在原图上操作)
            cv::Mat slice = frame(region).clone();

            // 3. 子图推理
            yolo11.pre_process(slice);
            yolo11.infer();
            auto slice_dets = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, region.width, region.height);

            // 4. 将子图结果映射回原图坐标
            for (auto &det : slice_dets) {
                if (det.class_id != 0) { continue; }
                // det.bbox 通常是 [x1, y1, x2, y2]
                int x1 = static_cast<int>(det.bbox[0] + region.x);
                int y1 = static_cast<int>(det.bbox[1] + region.y);
                int x2 = static_cast<int>(det.bbox[2] + region.x);
                int y2 = static_cast<int>(det.bbox[3] + region.y);
                
                bboxes_for_nms.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                scores_for_nms.push_back(det.score);
                class_ids_for_nms.push_back(det.class_id);
            }
        }

        // 5. 全局 NMS (解决重叠切片导致的重复检测)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes_for_nms, scores_for_nms, FLAGS_score_thres, FLAGS_nms_thres, indices);

        // 6. 转换结果给 ByteTrack
        std::vector<Object> objects;
        for (int idx : indices) {
            Object o;
            o.rect = cv::Rect_<float>(bboxes_for_nms[idx]);
            o.label = class_ids_for_nms[idx];
            o.prob = scores_for_nms[idx];
            objects.push_back(o);
        }
        // ---------------- SAHI 推理结束 ----------------

        // 更新追踪器
        auto tracks = tracker.update(objects);

        // FPS 计算
        auto frame_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> frame_duration = frame_end - frame_start;
        fps_history.push_back(1.0 / frame_duration.count());
        if (fps_history.size() > 10) fps_history.pop_front();
        double avg_fps = 0;
        for (double f : fps_history) avg_fps += f;
        avg_fps /= fps_history.size();

        // 绘制追踪结果 (HUD)
        for (auto &t : tracks) {
            if (!t.is_activated) continue;
            cv::Rect box((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);
            cv::Scalar col = tracker.get_color(t.track_id);
            cv::rectangle(frame, box, col, 2);
            cv::putText(frame, std::to_string(t.track_id), cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
        }

        // 绘制 FPS 和 计数
        std::string hud = "FPS: " + std::to_string((int)avg_fps) + " Objects: " + std::to_string(objects.size());
        cv::putText(frame, hud, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        if (writer.isOpened()) writer.write(frame);
        if (frame_idx % 30 == 0) std::cout << "Processed " << frame_idx << " frames..." << std::endl;
    }

    std::cout << "Done! Output saved to: " << FLAGS_output_video << std::endl;
    return 0;
}