#pragma once 
#pragma once
#include <opencv2/opencv.hpp>
#include "BaseInfer.h"
#include <map>
#include <deque>

// 业务逻辑基类
class LogicBase {
public:
    virtual ~LogicBase() = default;
    // 核心接口：执行该功能特有的判定与绘图
    virtual void process(cv::Mat& frame, const std::vector<Inf::Detection>& results) = 0;
};

// --- 功能1：溺水检测逻辑 ---
struct DrowningTrackState {
    std::deque<cv::Point2f> history_pos;
    int underwater_count = 0;
};

class DrowningLogic : public LogicBase {
private:
    std::map<int, DrowningTrackState> m_manager;
    const float m_moveThreshold = 400.0f; // 30帧内位移阈值
    const int m_timeThreshold = 10;     // 判定帧数阈值

public:
    void process(cv::Mat& frame, const std::vector<Inf::Detection>& results) override {
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
};

// --- 功能2：进水检测 (Patchcore) 逻辑 ---
class IntrusionLogic : public LogicBase {
public:
    void process(cv::Mat& frame, const std::vector<Inf::Detection>& results) override {
        // Patchcore 逻辑：如果有任何检测结果，判定为异常
        if (!results.empty()) {
            cv::putText(frame, "WATER INTRUSION!", cv::Point(50, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255), 3);
        }
    }
};