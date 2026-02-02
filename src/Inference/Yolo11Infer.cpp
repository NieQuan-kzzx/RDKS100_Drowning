#include "Yolo11Infer.h"
#include "PlogInitializer.h"

namespace Inf { // 必须包裹

// coco Name -- 80
std::vector<std::string> object_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
        "scissors", "teddy bear", "hair drier", "toothbrush"
};

Yolo11Infer::Yolo11Infer() {
    PlogInitializer::getInstance().init(plog::verbose);
}

Yolo11Infer::~Yolo11Infer() {
    cleanup();
}

void Yolo11Infer::draw(cv::Mat& frame, const std::vector<Detection>& results) {
    for (const auto& det : results) {
        cv::rectangle(frame, det.rect, cv::Scalar(255, 128, 0), 2);
        std::string txt = "ID:" + std::to_string(det.track_id) + " " + object_names[det.class_id];
        cv::putText(frame, txt, cv::Point(det.rect.x, det.rect.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 128, 0), 2);
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