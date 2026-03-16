#pragma once
#include "LogicBase.h"
#include <deque>
#include <map>

struct DrowningTrackState {
    std::deque<cv::Point2f> history_pos;
    int underwater_count = 0;
};

class DrowningUnderSurface : public LogicBase{
public:
    void process(cv::Mat& frame, const std::vector<Inf::Detection>& results) override;

private:
    std::map<int, DrowningTrackState> m_manager;
    const float m_moveThreshold = 400.0f;
    const int m_timeThreshold = 10;
};