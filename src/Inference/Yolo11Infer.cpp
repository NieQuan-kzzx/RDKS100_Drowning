#include "Yolo11Infer.h"
#include <plog/Log.h>

namespace Inf { // 必须包裹

Yolo11Infer::~Yolo11Infer() {
    cleanup();
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
        d.class_id = 0; 
        final_results.push_back(d);
    }
    return final_results;
}

void Yolo11Infer::cleanup() {
    m_yolo.reset();
    m_tracker.reset();
}

} // namespace Inf