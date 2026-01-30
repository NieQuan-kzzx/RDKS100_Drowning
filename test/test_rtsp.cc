#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "gflags/gflags.h"
#include "sp_codec.h"
#include "sp_display.h"
#include "sp_sys.h"
#include "sp_vio.h"
#include "multimedia_utils.hpp"

#include "ultralytics_yolo11.hpp" 
#include "common_utils.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
}

std::atomic_bool  is_stop(false);

void open_rtsp(const char* rtsp_url, const char* transfer_type,
               std::string model_path, std::string label_file,
               float score_thres, float nms_thres)
{
    int ret = 0;
    char errbuf[64]{};

    // ---- FFmpeg handles ----
    AVFormatContext* ifmt_ctx   = nullptr;
    AVCodecContext* vdec_ctx   = nullptr;
    const AVCodec* vdecoder   = nullptr;
    int              video_idx  = -1;
    AVDictionary* options    = nullptr;

    // ---- SP modules ----
    void* decoder = nullptr;
    void* display = nullptr;
    void* vps     = nullptr;

    // ---- sizes ----
    int stream_w = 0, stream_h = 0;
    int disp_w   = 0, disp_h   = 0;

    // ---- 初始化 YOLO11 模型 ----
    YOLO11 yolo11 = YOLO11(model_path); 
    std::vector<std::string> class_names = load_linewise_labels(label_file);

    avformat_network_init();

    av_dict_set(&options, "stimeout", "3000000", 0);
    av_dict_set(&options, "bufsize",  "1024000", 0);
    av_dict_set(&options, "rtsp_transport", transfer_type ? transfer_type : "tcp", 0);

    if (avformat_open_input(&ifmt_ctx, rtsp_url, nullptr, &options) < 0) {
        fprintf(stderr, "Could not open input '%s'\n", rtsp_url);
        goto EXIT;
    }

    if (avformat_find_stream_info(ifmt_ctx, nullptr) < 0) {
        goto EXIT;
    }

    video_idx = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_idx < 0) {
        goto EXIT;
    }

    {
        AVStream* vst = ifmt_ctx->streams[video_idx];
        vdecoder = avcodec_find_decoder(vst->codecpar->codec_id);
        vdec_ctx = avcodec_alloc_context3(vdecoder);
        avcodec_parameters_to_context(vdec_ctx, vst->codecpar);
        avcodec_open2(vdec_ctx, vdecoder, nullptr);
    }

    stream_w = vdec_ctx->width;
    stream_h = vdec_ctx->height;

    decoder = sp_init_decoder_module();
    display = sp_init_display_module();
    vps     = sp_init_vio_module();

    sp_get_display_resolution(&disp_w, &disp_h);

    ret = sp_start_decode(decoder, const_cast<char*>(rtsp_url), 0, SP_ENCODER_H264, stream_w, stream_h);
    if (ret != 0) goto EXIT;

    ret = sp_start_display(display, 11, disp_w, disp_h);
    if (ret != 0) goto EXIT;

    if (disp_w != stream_w || disp_h != stream_h) {
        int out_w = disp_w, out_h = disp_h;
        sp_open_vps(vps, 0, 1, SP_VPS_SCALE, stream_w, stream_h, &out_w, &out_h,
                    nullptr, nullptr, nullptr, nullptr, nullptr);
        sp_module_bind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
        sp_module_bind(vps, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
    }

    // ---- Main processing loop ----
    while (!is_stop) {
        cv::Mat yuv(stream_h * 3 / 2, stream_w, CV_8UC1);

        ret = sp_decoder_get_image(decoder, reinterpret_cast<char*>(yuv.data));
        if (ret != 0) {
            // 异常断开处理
            if (disp_w != stream_w || disp_h != stream_h)
                sp_module_unbind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
            sp_stop_decode(decoder);
            sp_release_decoder_module(decoder);
            decoder = sp_init_decoder_module();
            sp_start_decode(decoder, const_cast<char*>(rtsp_url), 0, SP_ENCODER_H264, stream_w, stream_h);
            if (disp_w != stream_w || disp_h != stream_h)
                sp_module_bind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
            continue;
        }

        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

        // 调用 YOLO11 接口
        yolo11.pre_process(bgr);
        yolo11.infer();
        auto results = yolo11.post_process(
            static_cast<float>(score_thres),
            static_cast<float>(nms_thres),
            stream_w, stream_h
        );

        draw_detections_on_disp(display, results, class_names, rdk_colors, 2);

        if (disp_w == stream_w && disp_h == stream_h) {
            sp_display_set_image(display, reinterpret_cast<char*>(yuv.data),
                                 FRAME_BUFFER_SIZE(disp_w, disp_h), 1);
        }
    }

EXIT:
    // 清理资源时增加空指针检查，防止崩溃
    if (display) { sp_stop_display(display); sp_release_display_module(display); }
    if (vps)     { sp_vio_close(vps);        sp_release_vio_module(vps);       }
    if (decoder) { sp_stop_decode(decoder);  sp_release_decoder_module(decoder);}
    if (vdec_ctx) avcodec_free_context(&vdec_ctx);
    if (ifmt_ctx) avformat_close_input(&ifmt_ctx);
    avformat_network_deinit();
}

void signal_handler_func(int signum) { is_stop = true; }

DEFINE_string(rtsp_url, "rtsp://admin:waterline123456@192.168.127.15", "RTSP URL");
DEFINE_string(transfer_type, "tcp", "tcp or udp");
DEFINE_string(model_path, "/home/sunrise/Desktop/RDKS100_Drowning/models/ultralytics_YOLO.hbm", "YOLO11 model path");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_coco.names", "Label file path");
DEFINE_double(score_thres, 0.25, "Score threshold");
DEFINE_double(nms_thres, 0.45, "NMS threshold");

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    signal(SIGINT, signal_handler_func);
    open_rtsp(FLAGS_rtsp_url.c_str(), FLAGS_transfer_type.c_str(),
              FLAGS_model_path, FLAGS_label_file,
              FLAGS_score_thres, FLAGS_nms_thres);
    return 0;
}
