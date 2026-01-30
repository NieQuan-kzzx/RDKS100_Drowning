#include <atomic>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 包含地平线系统的核心头文件
#include "gflags/gflags.h"          // 命令行参数解析
#include "sp_codec.h"               // 编解码模块
#include "sp_vio.h"                // 视频输入输出(VPS缩放)模块
#include "sp_sys.h"                // 系统底层控制
#include "sp_display.h"            // 硬件显示模块
#include "multimedia_utils.hpp"    // 多媒体工具类
#include "common_utils.hpp"        // 通用工具

#include "ultralytics_yolo11.hpp" 

/// 全局停止标志，通过信号处理函数触发
std::atomic_bool is_stop;

// SIGINT (Ctrl+C) 信号处理函数
void signal_handler_func(int signum) {
    printf("\n收到信号:%d, 正在停止...\n", signum);
    is_stop = true;  
}

// 命令行参数
DEFINE_int32(width,  1920, "输入视频的宽度");
DEFINE_int32(height, 1080, "输入视频的高度");
DEFINE_string(input_path, "/app/res/assets/1080P_test.h264", "输入H.264文件的路径");
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/ultralytics_YOLO.hbm", "BPU量化模型文件路径");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_coco.names", "类别名称列表文件");
DEFINE_double(score_thres, 0.25, "目标检测置信度阈值");
DEFINE_double(nms_thres, 0.45, "非极大值抑制(NMS)的IoU阈值");

int main(int argc, char **argv) {
    // 解析命令行参数
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 注册 Ctrl + C 处理
    signal(SIGINT, signal_handler_func);

    // 获取视频流参数
    int stream_width  = FLAGS_width;
    int stream_height = FLAGS_height;
    char* stream_file = const_cast<char*>(FLAGS_input_path.c_str());

    int ret = 0;
    void* vio_object   = nullptr; // VPS缩放对象
    void* display_obj  = nullptr; // 显示对象

    // 1. 获取显示分辨率
    int disp_w = 0, disp_h = 0;
    sp_get_display_resolution(&disp_w, &disp_h); 
    int widths[]  = { disp_w };
    int heights[] = { disp_h };

    // 2. 加载 YOLO11 模型和标签
    YOLO11 yolo11 = YOLO11(FLAGS_model_path); 
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    // 3. 解码器初始化
    void* decoder = sp_init_decoder_module(); 
    ret = sp_start_decode(decoder, stream_file, 0, SP_ENCODER_H264, stream_width, stream_height);
    if (ret) { printf("[错误] 解码失败\n"); return -1; }

    // 4. 显示模块初始化
    display_obj = sp_init_display_module();
    ret = sp_start_display(display_obj, 11, disp_w, disp_h);

    // 5. VPS 缩放适配
    if (disp_w != stream_width || disp_h != stream_height) {
        vio_object = sp_init_vio_module(); 
        sp_open_vps(vio_object, 0, 1, SP_VPS_SCALE, stream_width, stream_height,
                         widths, heights, nullptr, nullptr, nullptr, nullptr, nullptr);
        sp_module_bind(decoder, SP_MTYPE_DECODER, vio_object, SP_MTYPE_VIO);
        sp_module_bind(vio_object, SP_MTYPE_VIO, display_obj, SP_MTYPE_DISPLAY);
    }

    // 主循环
    while (!is_stop) {
        cv::Mat yuv(stream_height * 3 / 2, stream_width, CV_8UC1);

        // 获取一帧 NV12
        ret = sp_decoder_get_image(decoder, reinterpret_cast<char*>(yuv.data));
        if (ret != 0) {
            // 循环播放逻辑
            sp_stop_decode(decoder);
            sp_start_decode(decoder, stream_file, 0, SP_ENCODER_H264, stream_width, stream_height);
            continue; 
        }

        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

        // === YOLO11 推理流程 ===
        yolo11.pre_process(bgr);  // 预处理
        yolo11.infer();            // 硬件推理
        
        // 后处理获取结果 (使用 YOLO11 的后处理函数)
        auto results = yolo11.post_process(
            static_cast<float>(FLAGS_score_thres),
            static_cast<float>(FLAGS_nms_thres),
            stream_width, stream_height
        );

        // 绘制结果到硬件显示层
        draw_detections_on_disp(display_obj, results, class_names, rdk_colors, 2);

        // 如果无需缩放，直接推背景图
        if (disp_w == stream_width && disp_h == stream_height) {
            sp_display_set_image(display_obj, reinterpret_cast<char*>(yuv.data),
                                 FRAME_BUFFER_SIZE(disp_w, disp_h), 1);
        }
    }

    // 资源释放 (省略部分 cleanup 逻辑，同原代码)
    sp_stop_display(display_obj);
    sp_stop_decode(decoder);
    return 0;
}