#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 前向声明，避免与现有的 Detection 结构体冲突
struct Detection;

struct Track {
    int track_id;               // 轨迹ID
    float x, y, width, height;  // 当前位置 (左上角坐标 + 宽高)
    float confidence;           // 置信度
    uint8_t class_id;           // 类别ID
    int age;                    // 轨迹年龄（帧数）
    int time_since_update;      // 自上次更新以来的帧数
    bool is_activated;          // 是否已激活（确认轨迹）
    int hits;                   // 连续命中次数（用于tentative机制）

    // 运动状态（用于卡尔曼滤波预测）
    float vx, vy, vw, vh;      // 速度和尺寸变化率
    float p_x, p_y, p_w, p_h;  // 位置与尺寸的不确定性（简化协方差）

    Track(int id = -1, float x = 0, float y = 0, float w = 0, float h = 0, float conf = 0, uint8_t cls = 0)
        : track_id(id), x(x), y(y), width(w), height(h), confidence(conf), class_id(cls),
          age(0), time_since_update(0), is_activated(false), hits(1),
          vx(0), vy(0), vw(0), vh(0),
          p_x(1.0f), p_y(1.0f), p_w(1.0f), p_h(1.0f) {}

    // 获取中心点坐标
    float get_center_x() const { return x + width / 2.0f; }
    float get_center_y() const { return y + height / 2.0f; }

    // 计算与另一个边界框的IoU
    float calculate_iou(const Track& other) const {
        return calculate_iou_bbox(x, y, width, height, other.x, other.y, other.width, other.height);
    }

    // 计算与Detection的IoU
    template<typename DetType>
    float calculate_iou(const DetType& det) const {
        float det_x = det.bbox[0];
        float det_y = det.bbox[1];
        float det_width = det.bbox[2] - det.bbox[0];
        float det_height = det.bbox[3] - det.bbox[1];
        return calculate_iou_bbox(x, y, width, height, det_x, det_y, det_width, det_height);
    }

private:
    static float calculate_iou_bbox(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
        float inter_x1 = std::max(x1, x2);
        float inter_y1 = std::max(y1, y2);
        float inter_x2 = std::min(x1 + w1, x2 + w2);
        float inter_y2 = std::min(y1 + h1, y2 + h2);
        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float intersection = inter_w * inter_h;
        float area1 = w1 * h1;
        float area2 = w2 * h2;
        float union_area = area1 + area2 - intersection;
        if (union_area <= 0) return 0.0f;
        return intersection / union_area;
    }
};

class BotSort {
public:
    BotSort(float track_high_thresh = 0.6f, float track_low_thresh = 0.1f, 
            float new_track_thresh = 0.7f, float match_thresh = 0.7f, 
            int track_buffer = 30, int max_age = 30);

    // 主要的跟踪函数 - 传入当前帧用于CMC
    template<typename DetectionType>
    std::vector<Track> track(const std::vector<DetectionType>& detections, const cv::Mat& frame) {
        frame_count++;

        // 0. CMC (相机运动补偿)
        cv::Mat warp_matrix = cv::Mat::eye(3, 3, CV_64F);
        compute_cmc(frame, warp_matrix);
        apply_cmc_to_tracks(warp_matrix);

        // 1. 预测阶段
        predict_tracks();

        // 2. 分离高置信度和低置信度检测
        std::vector<DetectionType> high_conf_dets;
        std::vector<DetectionType> low_conf_dets;
        for (const auto& det : detections) {
            if (det.score >= track_high_thresh) high_conf_dets.push_back(det);
            else if (det.score >= track_low_thresh) low_conf_dets.push_back(det);
        }

        // 3. 第一次关联：活跃轨迹与高置信度检测
        auto iou_matrix_1 = calculate_iou_distance_matrix<DetectionType>(active_tracks, high_conf_dets);
        auto first_associations = linear_assignment(iou_matrix_1, match_thresh);

        std::vector<bool> matched_detections(high_conf_dets.size(), false);
        std::vector<bool> matched_tracks(active_tracks.size(), false);

        for (const auto& match : first_associations) {
            int track_idx = match.first;
            int det_idx = match.second;
            update_track(active_tracks[track_idx], high_conf_dets[det_idx]);
            matched_detections[det_idx] = true;
            matched_tracks[track_idx] = true;
        }

        // 4. 第二次关联：未匹配的活跃轨迹与低置信度检测
        std::vector<Track> unmatched_tracks;
        std::vector<int> unmatched_track_indices;
        for (size_t i = 0; i < active_tracks.size(); i++) {
            if (!matched_tracks[i]) {
                unmatched_tracks.push_back(active_tracks[i]);
                unmatched_track_indices.push_back(i);
            }
        }

        if (!unmatched_tracks.empty() && !low_conf_dets.empty()) {
            auto iou_matrix_2 = calculate_iou_distance_matrix<DetectionType>(unmatched_tracks, low_conf_dets);
            auto second_associations = linear_assignment(iou_matrix_2, 0.5f);

            for (const auto& match : second_associations) {
                int u_track_idx = match.first;
                int det_idx = match.second;
                
                // 映射回原始active_tracks索引并更新
                int original_idx = unmatched_track_indices[u_track_idx];
                update_track(active_tracks[original_idx], low_conf_dets[det_idx]);
                
                // 标记为已匹配，防止进入lost_tracks
                matched_tracks[original_idx] = true; 
            }
        }

        // 5. 第三次关联：丢失轨迹与高置信度未匹配检测 (恢复机制)
        std::vector<int> still_unmatched_det_indices;
        for (size_t i = 0; i < high_conf_dets.size(); i++) {
            if (!matched_detections[i]) still_unmatched_det_indices.push_back(i);
        }

        if (!lost_tracks.empty() && !still_unmatched_det_indices.empty()) {
            std::vector<DetectionType> rem_det;
            for (int idx : still_unmatched_det_indices) rem_det.push_back(high_conf_dets[idx]);
            
            auto iou_matrix_3 = calculate_iou_distance_matrix<DetectionType>(lost_tracks, rem_det);
            auto third_associations = linear_assignment(iou_matrix_3, match_thresh);

            std::vector<bool> rem_matched(rem_det.size(), false);
            for (const auto& match : third_associations) {
                int lost_idx = match.first;
                int det_idx = match.second;
                
                // 恢复轨迹：从lost移回active
                lost_tracks[lost_idx].time_since_update = 0;
                lost_tracks[lost_idx].is_activated = true; // 重新激活
                update_track(lost_tracks[lost_idx], rem_det[det_idx]);
                active_tracks.push_back(lost_tracks[lost_idx]);
                
                rem_matched[det_idx] = true;
            }

            // 清理已恢复的lost_tracks
            lost_tracks.erase(std::remove_if(lost_tracks.begin(), lost_tracks.end(),
                [](const Track& t) { return t.time_since_update == 0; }), lost_tracks.end());

            // 更新matched_detections，防止创建重复新轨迹
            for (size_t i = 0; i < rem_matched.size(); i++) {
                if (rem_matched[i]) {
                    matched_detections[still_unmatched_det_indices[i]] = true;
                }
            }
        }

        // 6. 创建新轨迹
        for (size_t i = 0; i < high_conf_dets.size(); i++) {
            if (!matched_detections[i] && high_conf_dets[i].score >= new_track_thresh) {
                create_new_track(high_conf_dets[i]);
            }
        }

        // 7. 标记丢失与清理
        mark_lost_tracks();
        cleanup_tracks();

        // 返回确认后的轨迹
        std::vector<Track> output_tracks;
        for (const auto& track : active_tracks) {
            if (track.is_activated) {
                output_tracks.push_back(track);
            }
        }
        return output_tracks;
    }

    const std::vector<Track>& get_active_tracks() const { return active_tracks; }
    void reset();

private:
    int frame_count;
    int next_track_id;

    float track_high_thresh;
    float track_low_thresh;
    float new_track_thresh;
    float match_thresh;
    int track_buffer;
    int max_age;

    std::vector<Track> active_tracks;
    std::vector<Track> lost_tracks;

    // CMC 相关
    cv::Mat prev_gray;
    cv::Ptr<cv::Feature2D> orb;

    void predict_tracks();
    void compute_cmc(const cv::Mat& frame, cv::Mat& warp_matrix);
    void apply_cmc_to_tracks(const cv::Mat& warp_matrix);

    template<typename DetectionType>
    std::vector<std::vector<float>> calculate_iou_distance_matrix(
        const std::vector<Track>& tracks, const std::vector<DetectionType>& detections) {
        std::vector<std::vector<float>> cost_matrix(tracks.size(), std::vector<float>(detections.size()));
        for (size_t i = 0; i < tracks.size(); i++) {
            for (size_t j = 0; j < detections.size(); j++) {
                float iou = tracks[i].calculate_iou(detections[j]);
                cost_matrix[i][j] = 1.0f - iou;
            }
        }
        return cost_matrix;
    }

    std::vector<std::pair<int, int>> linear_assignment(
        const std::vector<std::vector<float>>& cost_matrix, float threshold);

    template<typename DetectionType>
    void update_track(Track& track, const DetectionType& detection) {
        // 修复Bug：先保存旧中心
        float old_cx = track.get_center_x();
        float old_cy = track.get_center_y();
        float old_w = track.width;
        float old_h = track.height;

        float det_w = detection.bbox[2] - detection.bbox[0];
        float det_h = detection.bbox[3] - detection.bbox[1];
        float det_cx = detection.bbox[0] + det_w / 2.0f;
        float det_cy = detection.bbox[1] + det_h / 2.0f;

        // 简化的卡尔曼更新: K = P / (P + R), 这里 R 为观测噪声(设为1.0)
        float k_x = track.p_x / (track.p_x + 1.0f);
        float k_y = track.p_y / (track.p_y + 1.0f);
        float k_w = track.p_w / (track.p_w + 1.0f);
        float k_h = track.p_h / (track.p_h + 1.0f);

        // 更新位置 (融合预测与观测)
        float new_cx = track.get_center_x() + k_x * (det_cx - track.get_center_x());
        float new_cy = track.get_center_y() + k_y * (det_cy - track.get_center_y());
        float new_w = track.width + k_w * (det_w - track.width);
        float new_h = track.height + k_h * (det_h - track.height);

        track.x = new_cx - new_w / 2.0f;
        track.y = new_cy - new_h / 2.0f;
        track.width = std::max(1.0f, new_w);
        track.height = std::max(1.0f, new_h);
        
        track.confidence = detection.score;
        track.class_id = detection.class_id;
        track.time_since_update = 0;
        track.hits++;

        // 确认机制
        if (track.hits >= 3) {
            track.is_activated = true;
        }

        // 计算速度 (基于旧位置)
        float alpha = 0.8f;
        track.vx = alpha * track.vx + (1 - alpha) * (new_cx - old_cx);
        track.vy = alpha * track.vy + (1 - alpha) * (new_cy - old_cy);
        track.vw = alpha * track.vw + (1 - alpha) * (new_w - old_w);
        track.vh = alpha * track.vh + (1 - alpha) * (new_h - old_h);

        // 更新协方差
        track.p_x *= (1.0f - k_x);
        track.p_y *= (1.0f - k_y);
        track.p_w *= (1.0f - k_w);
        track.p_h *= (1.0f - k_h);
    }

    template<typename DetectionType>
    void create_new_track(const DetectionType& detection) {
        Track new_track(next_track_id++, detection.bbox[0], detection.bbox[1], 
                        detection.bbox[2] - detection.bbox[0], detection.bbox[3] - detection.bbox[1], 
                        detection.score, detection.class_id);
        // 新轨迹默认 is_activated = false, hits = 1
        active_tracks.push_back(new_track);
    }

    void mark_lost_tracks();
    void cleanup_tracks();
};
