#include "DrowningState.h"

void DrowningState::process(cv::Mat& frame, const std::vector<Inf::Detection>& results)
{
    std::vector<int> active_ids;
    bool any_drowning = false;

    for (auto& det : results) {
        active_ids.push_back(det.track_id);
        auto& state = m_manager[det.track_id];
        
        // 计算中心点
        cv::Point2f center(det.rect.x + det.rect.width/2.0f, det.rect.y + det.rect.height/2.0f);
        state.history_pos.push_back(center);
        if (state.history_pos.size() > 30) state.history_pos.pop_front();

        // 逻辑判定：在水下(class_id == 1)且位移小
        float dist = 100.0f;
        if (state.history_pos.size() >= 30) {
            dist = cv::norm(state.history_pos.back() - state.history_pos.front());
        }

        if (det.class_id == 1 && dist < m_moveThreshold) {
            state.underwater_count++;
        } else {
            state.underwater_count = 0;
        }

        // 绘制溺水红框
        if (state.underwater_count >= m_timeThreshold) {
            any_drowning = true;
            cv::rectangle(frame, det.rect, cv::Scalar(0, 0, 255), 4);
            cv::putText(frame, "DROWNING!", cv::Point(det.rect.x, det.rect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }

    // 全局报警横幅
    if (any_drowning) {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Rect(0,0,frame.cols,60), cv::Scalar(0,0,255), -1);
        cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);
        cv::putText(frame, "ALARM: DROWNING!", cv::Point(frame.cols/2-150, 45),
                    cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(255,255,255), 2);
    }

    // 清理消失的 ID
    for (auto it = m_manager.begin(); it != m_manager.end(); ) {
        if (std::find(active_ids.begin(), active_ids.end(), it->first) == active_ids.end()) 
            it = m_manager.erase(it);
        else ++it;
    }
}