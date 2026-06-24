#include "YoloSeg.h"
#include "common_utils.hpp"
#include "PlogInitializer.h"

namespace Inf { // 必须包裹

YoloSeg::YoloSeg() {
    PlogInitializer::getInstance().init(plog::verbose);
}

YoloSeg::~YoloSeg() {
    cleanup();
}

/**
    draw_boxes(frame, final_dets, class_names, rdk_colors);               // 绘制目标框和标签
    draw_masks(frame, final_dets, resized_masks, rdk_colors, 0.4f);       // 混合半透明 Mask
    draw_contours(frame, final_dets, resized_masks, rdk_colors, 1);       // 绘制边缘轮廓
    auto [final_dets, resized_masks] =
            yolo11_seg.post_process(FLAGS_score_thres, FLAGS_nms_thres, frame_w, frame_h);
*/

void YoloSeg::draw(cv::Mat& frame, const std::vector<Detection>& results) {
    if (frame.empty()) return;

    // --- 新增：根据画面宽度动态计算字号和粗细，防止在大图中太小 ---
    // 基础字号 0.7，画面越宽，比例越大
    double fontScale = frame.cols / 800.0; 
    int thickness = std::max(1, (int)(frame.cols / 600)); 

    for (const auto& det : results) {
        cv::Scalar color = rdk_colors[det.class_id % rdk_colors.size()];

        std::string label_name = (det.class_id < m_labels.size()) ? m_labels[det.class_id] : "unknown";
        std::string txt = "ID:" + std::to_string(det.track_id) + " " + label_name;

        cv::Rect clamped_rect = det.rect & cv::Rect(0, 0, frame.cols, frame.rows);
        if (clamped_rect.empty()) continue;

        cv::rectangle(frame, clamped_rect, color, 3);

        if (!det.mask.empty()) {
            cv::Mat mask_resized;
            cv::resize(det.mask, mask_resized, clamped_rect.size());
            cv::Mat mask_color;
            cv::cvtColor(mask_resized, mask_color, cv::COLOR_GRAY2BGR);
            mask_color.setTo(color, mask_resized);

            cv::Mat roi = frame(clamped_rect);
            cv::addWeighted(roi, 0.6, mask_color, 0.4, 0, roi);
        }

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

        int text_y = std::max(clamped_rect.y, text_size.height + 10);
        cv::rectangle(frame,
                      cv::Point(clamped_rect.x, text_y - text_size.height - 10),
                      cv::Point(clamped_rect.x + text_size.width, text_y),
                      color, -1);

        cv::putText(frame, txt, cv::Point(clamped_rect.x, text_y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}

bool YoloSeg::init(const std::string& model_path) {
    try {
        m_yolo_seg = std::make_unique<YOLO11_Seg>(model_path);
        m_tracker = std::make_unique<BYTETracker>();
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<Detection> YoloSeg::run(cv::Mat& frame) {
    std::vector<Detection> final_results;
    if (frame.empty()) return final_results;

    m_yolo_seg->pre_process(frame);
    m_yolo_seg->infer();
    
    // 这里调用地平线官方的 post_process，返回的是 ::Detection (带bbox)
    auto [bpu_dets, resized_masks] = m_yolo_seg->post_process(0.25f, 0.7f, frame.cols, frame.rows);
    
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
        Detection d;
        cv::Rect raw_rect((int)t.tlwh[0], (int)t.tlwh[1], (int)t.tlwh[2], (int)t.tlwh[3]);
        d.rect = raw_rect & cv::Rect(0, 0, frame.cols, frame.rows);
        d.score = t.score;
        d.track_id = t.track_id;
        
        int best_idx = -1;
        float max_iou = 0.0f;
        for (size_t i = 0; i < objects.size(); ++i) {
            const auto& obj = objects[i];
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
                    best_idx = i;
                }
            }
        }

        if (best_idx >= 0 && best_idx < resized_masks.size()) {
            d.class_id = bpu_dets[best_idx].class_id;
            d.mask = resized_masks[best_idx].clone();

        } else {
            d.class_id = 0;
            d.mask = cv::Mat();
        }

        final_results.push_back(d);
    }
    return final_results;
}

void YoloSeg::cleanup() {
    m_yolo_seg.reset();
    m_tracker.reset();
}

} // namespace Inf