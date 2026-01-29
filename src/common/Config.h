#pragma once
#include <string>
#include "cereal/archives/json.hpp"
#include "cereal/cereal.hpp"

struct HikConfig {
    std::string ip;
    int port;
    std::string user;
    std::string pass;

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(ip), 
                CEREAL_NVP(port), 
                CEREAL_NVP(user), 
                CEREAL_NVP(pass));
    }
};


// struct DirConfig
// {
//     // 路径
//     std::string video_warp_map_dir;
//     std::string camera_warp_map_dir;
//     std::string camera_intrinsic_dir;
//     std::string YOLO_detect_dir;

//     template <class Archive>
//     void serialize(Archive &archive)
//     {
//         archive(CEREAL_NVP(video_warp_map_dir),
//                 CEREAL_NVP(camera_warp_map_dir),
//                 CEREAL_NVP(camera_intrinsic_dir),
//                 CEREAL_NVP(YOLO_detect_dir));
//     }
// };

// struct RuntimeConfig
// {
//     // 输入参数
//     std::string source_type;
//     int camera_num;
//     bool is_undistort;
//     bool is_sahi;

//     // 推理参数
//     float detect_conf_thresh; // 目标检测阈值
//     float track_thresh;       // 2阶段处理时的阈值
//     float high_thresh;        // 初始化一个新的轨迹时box的置信度需要大于该阈值
//     float match_thresh;       // box和轨迹匹配的IOU阈值

//     // 刷新参数(ms)
//     int calibrate_refresh_interval;
//     int track_refresh_interval;
//     int previrew_refresh_interval;

//     int video_capture_interval;

//     template <class Archive>
//     void serialize(Archive &archive)
//     {
//         archive(
//             CEREAL_NVP(source_type),
//             CEREAL_NVP(camera_num),
//             CEREAL_NVP(is_undistort),
//             CEREAL_NVP(is_sahi),
//             CEREAL_NVP(detect_conf_thresh),
//             CEREAL_NVP(track_thresh),
//             CEREAL_NVP(high_thresh),
//             CEREAL_NVP(match_thresh),
//             CEREAL_NVP(calibrate_refresh_interval),
//             CEREAL_NVP(track_refresh_interval),
//             CEREAL_NVP(previrew_refresh_interval),
//             CEREAL_NVP(video_capture_interval));
//     }
// };
