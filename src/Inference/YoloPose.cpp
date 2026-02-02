# include "YoloPose.h"
# include "PlogInitializer.h"

namespace Inf{

YoloPose::YoloPose(){
    PlogInitializer::getInstance().init(plog::verbose);
}

YoloPose::~YoloPose(){
    cleanup();
}

void YoloPose::draw(cv::Mat& frame, const std::vector<Inf::Detection>& results) {
    if (frame.empty()) return;

    // 17个关键点连接关系 (COCO标准)
    static const std::vector<std::pair<int, int>> SKELETON = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };

    // 漂亮的配色方案 (RGB)
    static const std::vector<cv::Scalar> COLORS = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170)
    };

    for (const auto& det : results) {
        // --- 1. 绘制矩形框 ---
        cv::rectangle(frame, det.rect, cv::Scalar(0, 255, 0), 2);

        // --- 2. 绘制带背景的标签 (ID + Label) ---
        std::string label = "ID:" + std::to_string(det.track_id) + " Person";
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // 标签背景矩形
        cv::Rect textBg(det.rect.x, det.rect.y - textSize.height - 10, textSize.width + 10, textSize.height + 10);
        // 防止溢出边界
        if (textBg.y < 0) textBg.y = 0;
        
        cv::rectangle(frame, textBg, cv::Scalar(0, 255, 0), -1); // 实心填充
        cv::putText(frame, label, cv::Point(textBg.x + 5, textBg.y + textSize.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // --- 3. 绘制骨骼连线 ---
        if (det.keypoints.empty()) continue;

        // 绘制线条
        for (const auto& edge : SKELETON) {
            if (edge.first >= det.keypoints.size() || edge.second >= det.keypoints.size()) continue;
            
            const auto& kp1 = det.keypoints[edge.first];
            const auto& kp2 = det.keypoints[edge.second];

            // 只有两个点置信度都够才连线
            if (kp1.score > 0.5f && kp2.score > 0.5f) {
                cv::line(frame, cv::Point(kp1.x, kp1.y), cv::Point(kp2.x, kp2.y),
                         COLORS[edge.first % COLORS.size()], 2, cv::LINE_AA);
            }
        }

        // 绘制关键点圆圈
        for (size_t i = 0; i < det.keypoints.size(); ++i) {
            const auto& kp = det.keypoints[i];
            if (kp.score > 0.5f) {
                // 外圈白色，内圈彩色，更有立体感
                cv::circle(frame, cv::Point(kp.x, kp.y), 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
                cv::circle(frame, cv::Point(kp.x, kp.y), 2, COLORS[i % COLORS.size()], -1, cv::LINE_AA);
            }
        }
    }
}

bool YoloPose::init(const std::string& model_path){
    try {
        m_yolo_pose = std::make_unique<YOLO11_Pose>(model_path);
        m_tracker = std::make_unique<BYTETracker>();
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<Detection> YoloPose::run(cv::Mat& frame) {
    std::vector<Detection> final_results;
    if (frame.empty()) return final_results;

    m_yolo_pose->pre_process(frame);
    m_yolo_pose->infer();

    auto [final_dets, resized_kpts] = m_yolo_pose->post_process(0.25f, 0.7f, 0.5f, frame.cols, frame.rows);
    
    std::vector<Object> objects;
    for (size_t i = 0; i < final_dets.size(); ++i) {
        Object o;
        auto& d = final_dets[i];
        o.rect = cv::Rect_<float>(d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]);
        o.label = d.class_id;
        o.prob = d.score;
        objects.push_back(o);
    }
    auto tracks = m_tracker->update(objects);

    for (auto &t : tracks) {
        if (!t.is_activated) continue;
        Detection d;
        d.rect = cv::Rect((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);
        d.score = t.score;
        d.track_id = t.track_id;
        d.class_id = 0;

        float max_iou = 0.0f;
        int best_idx = -1;
        for (size_t i = 0; i < objects.size(); ++i) {
            // 计算当前跟踪框(d.rect)与原始检测框(objects[i].rect)的交并比
            float inter = (d.rect & cv::Rect(objects[i].rect)).area();
            float iou = inter / (d.rect.area() + objects[i].rect.area() - inter);
            if(iou > max_iou) {
                max_iou = iou;
                best_idx = i;
            }
        }

        // 如果匹配成功（IOU > 0.5），则把对应的关键点存入
        if (best_idx != -1 && max_iou > 0.5f) {
            for (const auto& kp : resized_kpts[best_idx]) {
                Inf::Keypoint inf_kp;
                inf_kp.x = kp.x;
                inf_kp.y = kp.y;
                inf_kp.score = 1.0f / (1.0f + std::exp(-kp.score));
                d.keypoints.push_back(inf_kp);
            }
        }

        final_results.push_back(d);
    }
    return final_results;

}
void YoloPose::cleanup() {
    m_yolo_pose.reset();
    m_tracker.reset();
}

};