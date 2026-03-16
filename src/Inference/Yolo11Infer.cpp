#include "Yolo11Infer.h"
#include "common_utils.hpp"
#include "PlogInitializer.h"

namespace Inf { // 必须包裹

Yolo11Infer::Yolo11Infer() {
    PlogInitializer::getInstance().init(plog::verbose);
}

Yolo11Infer::~Yolo11Infer() {
    cleanup();
}

void Yolo11Infer::draw(cv::Mat& frame, const std::vector<Detection>& results) {
    if (frame.empty()) return;

    // --- 新增：根据画面宽度动态计算字号和粗细，防止在大图中太小 ---
    // 基础字号 0.7，画面越宽，比例越大
    double fontScale = frame.cols / 800.0; 
    int thickness = std::max(1, (int)(frame.cols / 600)); 

    for (const auto& det : results) {
        // 使用 class_id 分配颜色
        cv::Scalar color = rdk_colors[det.class_id % rdk_colors.size()];

        // 获取标签名称
        std::string label_name = (det.class_id < m_labels.size()) ? m_labels[det.class_id] : "unknown";
        std::string txt = "ID:" + std::to_string(det.track_id) + " " + label_name;

        // 1. 画矩形框 (线条加粗到 3)
        cv::rectangle(frame, det.rect, color, 3);

        // 2. 准备文字背景
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        
        // 文字背景框
        cv::rectangle(frame, 
                      cv::Point(det.rect.x, det.rect.y - text_size.height - 10),
                      cv::Point(det.rect.x + text_size.width, det.rect.y),
                      color, -1); 

        // 3. 写字 (白色文字，使用动态计算的 fontScale 和 thickness)
        cv::putText(frame, txt, cv::Point(det.rect.x, det.rect.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}

bool Yolo11Infer::init(const std::string& model_path) {
    try {
        m_yolo = std::make_unique<YOLO11>(model_path);
        m_tracker = std::make_unique<BYTETracker>();
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<Detection> Yolo11Infer::run(cv::Mat& frame) {
    std::vector<Detection> final_results;
    if (frame.empty()) return final_results;

    m_yolo->pre_process(frame);
    m_yolo->infer();
    
    // 这里调用地平线官方的 post_process，返回的是 ::Detection (带bbox)
    auto bpu_dets = m_yolo->post_process(0.25f, 0.7f, frame.cols, frame.rows);

    std::vector<Object> objects;
    for (auto &det : bpu_dets) {
        Object o;
        // 这里的 det 是 ::Detection，所以有 bbox 成员
        float x1 = det.bbox[0];
        float y1 = det.bbox[1];
        float x2 = det.bbox[2];
        float y2 = det.bbox[3];
        o.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
        o.label = det.class_id;
        o.prob = det.score;
        objects.push_back(o);
    }

    auto tracks = m_tracker->update(objects);

    for (auto &t : tracks) {
        if (!t.is_activated) continue;
        Detection d; // 这里的 d 是 Inf::Detection
        d.rect = cv::Rect((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);
        d.score = t.score;
        d.track_id = t.track_id;
        
        int best_label = 0;
        float max_iou = 0.0f;
        for (const auto& obj : objects) {
            // 计算交集 (Intersection)
            float inter_x1 = std::max((float)d.rect.x, obj.rect.x);
            float inter_y1 = std::max((float)d.rect.y, obj.rect.y);
            float inter_x2 = std::min((float)(d.rect.x + d.rect.width), obj.rect.x + obj.rect.width);
            float inter_y2 = std::min((float)(d.rect.y + d.rect.height), obj.rect.y + obj.rect.height);

            if (inter_x2 > inter_x1 && inter_y2 > inter_y1) {
                float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
                float union_area = (float)d.rect.area() + obj.rect.area() - inter_area;
                float iou = inter_area / union_area;

                if (iou > max_iou) {
                    max_iou = iou;
                    best_label = obj.label; // 匹配到最接近的原始框，取其标签
                }
            }
        }
        
        // 如果 max_iou 够大（比如 > 0.5），则认为是同一物体
        d.class_id = (max_iou > 0.5f) ? best_label : 0;
        
        final_results.push_back(d);
    }
    return final_results;
}

void Yolo11Infer::cleanup() {
    m_yolo.reset();
    m_tracker.reset();
}

} // namespace Inf