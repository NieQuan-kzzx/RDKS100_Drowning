#include "BotSort.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

BotSort::BotSort(float track_high_thresh, float track_low_thresh, float new_track_thresh, 
                 float match_thresh, int track_buffer, int max_age)
    : frame_count(0), next_track_id(0), 
      track_high_thresh(track_high_thresh), track_low_thresh(track_low_thresh), 
      new_track_thresh(new_track_thresh), match_thresh(match_thresh), 
      track_buffer(track_buffer), max_age(max_age) {
    // 初始化ORB特征提取器用于CMC
    orb = cv::ORB::create(500);
}

void BotSort::predict_tracks() {
    float q_pos = 0.1f; // 位置过程噪声
    float q_vel = 0.01f; // 速度过程噪声

    for (auto& track : active_tracks) {
        // 增加过程噪声 (协方差预测)
        track.p_x += q_pos;
        track.p_y += q_pos;
        track.p_w += q_pos;
        track.p_h += q_pos;

        // 状态预测 (恒定速度模型)
        float new_cx = track.get_center_x() + track.vx;
        float new_cy = track.get_center_y() + track.vy;
        track.width += track.vw;
        track.height += track.vh;

        track.width = std::max(1.0f, track.width);
        track.height = std::max(1.0f, track.height);

        track.x = new_cx - track.width / 2.0f;
        track.y = new_cy - track.height / 2.0f;

        track.age++;
        track.time_since_update++;
    }

    for (auto& track : lost_tracks) {
        track.age++;
        track.time_since_update++;
        // 丢失轨迹的协方差持续增大，以便在更大范围内匹配
        track.p_x += q_pos * 2.0f; 
        track.p_y += q_pos * 2.0f;
    }
}

void BotSort::compute_cmc(const cv::Mat& frame, cv::Mat& warp_matrix) {
    if (frame.empty()) return;
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame;

    if (!prev_gray.empty()) {
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        
        orb->detectAndCompute(prev_gray, cv::Mat(), kp1, desc1);
        orb->detectAndCompute(gray, cv::Mat(), kp2, desc2);

        if (kp1.size() > 10 && kp2.size() > 10) {
            cv::BFMatcher matcher(cv::NORM_HAMMING);
            std::vector<cv::DMatch> matches;
            matcher.match(desc1, desc2, matches);

            // 筛选好的匹配
            std::vector<cv::Point2f> pts1, pts2;
            for (const auto& m : matches) {
                if (m.distance < 50.0) {
                    pts1.push_back(kp1[m.queryIdx].pt);
                    pts2.push_back(kp2[m.trainIdx].pt);
                }
            }

            if (pts1.size() >= 10) {
                // 计算仿射变换 (比单应矩阵更稳定，适合2D平移旋转)
                warp_matrix = cv::estimateAffinePartial2D(pts1, pts2);
                if (warp_matrix.empty()) {
                    warp_matrix = cv::Mat::eye(3, 3, CV_64F);
                } else {
                    // 将 2x3 转换为 3x3
                    cv::Mat row = (cv::Mat_<double>(1, 3) << 0, 0, 1);
                    warp_matrix.push_back(row);
                }
            }
        }
    }
    gray.copyTo(prev_gray);
}

void BotSort::apply_cmc_to_tracks(const cv::Mat& warp_matrix) {
    if (warp_matrix.rows != 3 || warp_matrix.cols != 3) return;

    auto transform_track = [&warp_matrix](Track& track) {
        std::vector<cv::Point2f> centers = { cv::Point2f(track.get_center_x(), track.get_center_y()) };
        std::vector<cv::Point2f> transformed_centers;
        cv::perspectiveTransform(centers, transformed_centers, warp_matrix);
        
        float new_cx = transformed_centers[0].x;
        float new_cy = transformed_centers[0].y;
        
        track.x = new_cx - track.width / 2.0f;
        track.y = new_cy - track.height / 2.0f;
    };

    for (auto& track : active_tracks) transform_track(track);
    for (auto& track : lost_tracks) transform_track(track);
}

std::vector<std::pair<int, int>> BotSort::linear_assignment(
    const std::vector<std::vector<float>>& cost_matrix, float threshold) {
    
    std::vector<std::pair<int, int>> matches;
    if (cost_matrix.empty() || cost_matrix[0].empty()) return matches;

    int rows = cost_matrix.size();
    int cols = cost_matrix[0].size();

    // 使用改进的贪心匹配：先匹配代价最小的，处理竞争冲突
    struct MatchCandidate {
        int r, c;
        float cost;
        bool operator<(const MatchCandidate& other) const { return cost < other.cost; }
    };

    std::vector<MatchCandidate> candidates;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (cost_matrix[i][j] <= threshold) {
                candidates.push_back({i, j, cost_matrix[i][j]});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end());

    std::vector<bool> row_covered(rows, false);
    std::vector<bool> col_covered(cols, false);

    for (const auto& cand : candidates) {
        if (!row_covered[cand.r] && !col_covered[cand.c]) {
            matches.emplace_back(cand.r, cand.c);
            row_covered[cand.r] = true;
            col_covered[cand.c] = true;
        }
    }

    return matches;
}

void BotSort::mark_lost_tracks() {
    for (auto& track : active_tracks) {
        if (track.time_since_update > track_buffer) {
            // 如果是未确认的轨迹，直接丢弃；确认的轨迹放入lost
            if (!track.is_activated) {
                continue; // 会在下面的remove_if中删除
            }
            lost_tracks.push_back(track);
        }
    }

    // 从活跃轨迹中移除已丢失的轨迹 (修复硬编码Bug，使用 track_buffer)
    active_tracks.erase(
        std::remove_if(active_tracks.begin(), active_tracks.end(),
            [this](const Track& track) { return track.time_since_update > track_buffer; }),
        active_tracks.end());
}

void BotSort::cleanup_tracks() {
    lost_tracks.erase(
        std::remove_if(lost_tracks.begin(), lost_tracks.end(),
            [this](const Track& track) { return track.time_since_update > max_age; }),
        lost_tracks.end());
}

void BotSort::reset() {
    frame_count = 0;
    next_track_id = 0;
    active_tracks.clear();
    lost_tracks.clear();
    prev_gray.release();
}
