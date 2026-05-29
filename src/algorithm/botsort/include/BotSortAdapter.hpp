#pragma once
#include "BotSort.hpp"
#include "common_utils.hpp"

class BotSortAdapter {
private:
    BotSort tracker;

public:
    BotSortAdapter(float track_high_thresh = 0.6f, float track_low_thresh = 0.1f, 
                   float new_track_thresh = 0.7f, float match_thresh = 0.7f, 
                   int track_buffer = 30, int max_age = 30)
        : tracker(track_high_thresh, track_low_thresh, new_track_thresh, match_thresh, track_buffer, max_age) {}

    // 接口变更：需要传入当前帧用于CMC计算
    std::vector<Track> track(const std::vector<Detection>& detections, const cv::Mat& frame) {
        return tracker.track(detections, frame);
    }

    const std::vector<Track>& get_active_tracks() const {
        return tracker.get_active_tracks(); 
    }

    void reset() { tracker.reset(); }

    static cv::Scalar get_track_color(int track_id) {
        static std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
            cv::Scalar(128, 0, 255), cv::Scalar(255, 128, 0), cv::Scalar(128, 255, 0),
            cv::Scalar(0, 128, 255), cv::Scalar(255, 128, 128), cv::Scalar(128, 255, 128),
            cv::Scalar(128, 128, 255)
        };
        return colors[track_id % colors.size()];
    }

    static void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks) {
        int max_id = 0;
        for (const auto& track : tracks) {
            // 仅绘制已确认的轨迹 (过滤 Tentative 轨迹)
            if (!track.is_activated) continue; 

            cv::Scalar track_color = get_track_color(track.track_id);
            cv::Rect track_rect(cv::Point((int)track.x, (int)track.y), cv::Size((int)track.width, (int)track.height));
            cv::rectangle(frame, track_rect, track_color, 2);

            if (track.track_id > max_id) max_id = track.track_id;

            std::ostringstream label_stream;
            label_stream << "ID:" << track.track_id << " " << std::fixed << std::setprecision(2) << track.confidence;
            std::string label_text = label_stream.str();

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Rect text_bg_rect(track.x, track.y - text_size.height - 4, text_size.width + 4, text_size.height + 4);

            if (text_bg_rect.y < 0) text_bg_rect.y = track.y;
            cv::rectangle(frame, text_bg_rect, track_color, -1);
            cv::putText(frame, label_text, cv::Point(track.x + 2, text_bg_rect.y + text_size.height), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        std::ostringstream count_stream;
        count_stream << "Confirmed Tracks: " << tracks.size() << " MaxID: " << max_id;
        cv::putText(frame, count_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
};
