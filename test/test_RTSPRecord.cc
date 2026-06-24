#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <opencv2/opencv.hpp>

#include "gflags/gflags.h"
#include "sp_codec.h"
#include "sp_sys.h"
#include "multimedia_utils.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/timestamp.h>
}

std::atomic_bool is_stop(false);
void signal_handler_func(int signum) { is_stop = true; }

DEFINE_string(rtsp_url, "rtsp://admin:nuaa2026@192.168.137.15",
              "RTSP摄像头URL");
DEFINE_int32(width, 1920, "视频宽度");
DEFINE_int32(height, 1080, "视频高度");
DEFINE_int32(record_seconds, 30, "录制时长(秒)，0表示手动停止(Ctrl+C)");
DEFINE_string(output_dir, "./rtsp_records", "录像输出目录");
DEFINE_string(method, "hw", "hw=硬件解码, ffmpeg=FFMPEG转码");

// 探测RTSP流是否可达，返回真实帧率
static double probe_rtsp_stream(const std::string& url,
                                int& out_width, int& out_height)
{
    AVFormatContext* fmt = nullptr;
    AVDictionary* opts = nullptr;
    avformat_network_init();
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);

    if (avformat_open_input(&fmt, url.c_str(), nullptr, &opts) < 0) {
        fprintf(stderr, "[Probe] 无法连接RTSP: %s\n", url.c_str());
        return -1;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt);
        return -1;
    }
    int vi = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vi < 0) { avformat_close_input(&fmt); return -1; }

    AVStream* vs = fmt->streams[vi];
    out_width  = vs->codecpar->width;
    out_height = vs->codecpar->height;
    double fps = av_q2d(vs->avg_frame_rate);
    if (fps <= 0) fps = av_q2d(vs->r_frame_rate);
    if (fps <= 0) fps = 25.0;

    printf("[Probe] RTSP流信息: %dx%d, %.2f FPS, codec=%s\n",
           out_width, out_height, fps,
           avcodec_get_name(vs->codecpar->codec_id));
    avformat_close_input(&fmt);
    return fps;
}

// ============================================================
// 方式1: 硬件解码 + OpenCV 录像
// ============================================================
static bool use_hardware_decoder_record(
    const std::string& rtsp_url, int width, int height,
    const std::string& output_path, int record_seconds)
{
    // 先探测流是否可用
    int probe_w = 0, probe_h = 0;
    double probe_fps = probe_rtsp_stream(rtsp_url, probe_w, probe_h);
    if (probe_fps < 0) {
        fprintf(stderr, "[HW] RTSP流不可达，请检查摄像头连接\n");
        return false;
    }
    if (probe_w > 0 && probe_h > 0) { width = probe_w; height = probe_h; }

    void* decoder = sp_init_decoder_module();
    int ret = sp_start_decode(decoder, const_cast<char*>(rtsp_url.c_str()),
                              0, SP_ENCODER_H264, width, height);
    if (ret != 0) {
        fprintf(stderr, "[HW] 硬件解码启动失败, ret=%d\n", ret);
        return false;
    }
    printf("[HW] 硬件解码启动成功 (%dx%d)\n", width, height);

    std::vector<double> frame_intervals_ms;
    std::vector<cv::Mat> frame_buffer;
    auto prev_time = std::chrono::steady_clock::now();

    cv::VideoWriter writer;
    bool writer_opened = false;
    int frame_count = 0;
    int consecutive_fails = 0;
    auto start_time = std::chrono::steady_clock::now();

    cv::Mat yuv(height * 3 / 2, width, CV_8UC1);
    cv::Mat bgr;

    while (!is_stop) {
        ret = sp_decoder_get_image(decoder, reinterpret_cast<char*>(yuv.data));
        if (ret != 0) {
            if (++consecutive_fails == 1) {
                fprintf(stderr, "\n[HW] ⚠ 硬件解码器无法获取帧(可能已被其他进程占用)\n");
                fprintf(stderr, "[HW] 建议: 关闭主程序后重试，或用 --method=ffmpeg\n");
                fprintf(stderr, "[HW] 等待1秒后重试...\n\n");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (consecutive_fails > 20) {
                fprintf(stderr, "[HW] 放弃等待硬件解码帧\n");
                break;
            }
            continue;
        }
        consecutive_fails = 0;

        auto now = std::chrono::steady_clock::now();
        frame_count++;

        if (frame_count > 1) {
            double iv = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                now - prev_time).count();
            frame_intervals_ms.push_back(iv);
        }
        prev_time = now;

        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

        // 第一次拿到帧时初始化VideoWriter，用探测到的流帧率
        if (!writer_opened) {
            double init_fps = (probe_fps > 0) ? probe_fps : 25.0;
            writer.open(output_path,
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        init_fps, cv::Size(width, height));
            if (!writer.isOpened()) {
                fprintf(stderr, "[HW] VideoWriter 打开失败: %s\n", output_path.c_str());
                break;
            }
            writer_opened = true;
            printf("[HW] 开始录像 (%s, VideoWriter=%d FPS)\n",
                   output_path.c_str(), (int)init_fps);
        }

        writer.write(bgr);

        if (record_seconds > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            if (elapsed >= record_seconds) break;
        }
    }

    frame_buffer.clear();
    auto total_end = std::chrono::steady_clock::now();
    double total_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(
        total_end - start_time).count();

    if (writer.isOpened()) writer.release();
    if (decoder) { sp_stop_decode(decoder); sp_release_decoder_module(decoder); }

    printf("\n========== [HW] 录像诊断 ==========\n");
    printf("  录像时长:     %.1f 秒\n", total_elapsed_s);
    printf("  总帧数:       %d\n", frame_count);
    if (frame_count > 1) {
        double actual_fps = (frame_count - 1) / total_elapsed_s;
        double total_ms = 0, min_ms = 9999, max_ms = 0;
        for (auto& iv : frame_intervals_ms) {
            total_ms += iv;
            if (iv < min_ms) min_ms = iv;
            if (iv > max_ms) max_ms = iv;
        }
        double avg_ms = total_ms / frame_intervals_ms.size();
        printf("  实际平均帧率:  %.2f FPS (平均间隔 %.2f ms)\n", actual_fps, avg_ms);
        printf("  帧间隔范围:   %.2f ~ %.2f ms\n", min_ms, max_ms);
        printf("  VideoWriter:   %d FPS\n", (int)(probe_fps > 0 ? probe_fps : 25.0));

        // 判断是否会有加速/慢放
        double expected_duration = frame_count / ((probe_fps > 0) ? probe_fps : 25.0);
        double speed_ratio = total_elapsed_s / expected_duration;
        printf("  预期视频时长:  %.1f 秒 (按VideoWriter帧率计算)\n", expected_duration);
        printf("  实际录像时长:  %.1f 秒\n", total_elapsed_s);

        if (fabs(speed_ratio - 1.0) > 0.05) {
            if (speed_ratio > 1.0)
                printf("  ⚠ 诊断: 录像会慢放 %.0f%% (VideoWriter帧率偏高)\n", speed_ratio * 100);
            else
                printf("  ⚠ 诊断: 录像会加速 %.0f%% (VideoWriter帧率偏低)\n", (1.0/speed_ratio) * 100);
            printf("  ✓ 修复方案: RecordingManager 已改用实测帧率初始化VideoWriter\n");
            printf("  ✓ 重启主程序后生效\n");
        } else {
            printf("  ✓ 诊断: 录像速度正常\n");
        }
    }
    printf("==================================\n\n");
    return frame_count > 0;
}

// ============================================================
// 方式2: FFMPEG 转码 (保留原始 PTS)
// ============================================================
static bool use_ffmpeg_record(const std::string& rtsp_url,
                               const std::string& output_path,
                               int record_seconds)
{
    AVFormatContext* ifmt_ctx = nullptr;
    AVFormatContext* ofmt_ctx = nullptr;
    const AVCodec* decoder_codec = nullptr;
    AVCodecContext* decoder_ctx = nullptr;
    const AVCodec* encoder_codec = nullptr;
    AVCodecContext* encoder_ctx = nullptr;
    AVStream* out_stream = nullptr;
    AVStream* in_stream = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* sw_frame = nullptr;
    int video_idx = -1;
    int frame_count = 0;
    double input_fps = 25.0;
    std::chrono::steady_clock::time_point start_time;
    int ret = 0;

    avformat_network_init();
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "buffer_size", "1024000", 0);

    ret = avformat_open_input(&ifmt_ctx, rtsp_url.c_str(), nullptr, &opts);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "[FFMPEG] 打开RTSP失败: %s\n", errbuf);
        goto cleanup;
    }

    ret = avformat_find_stream_info(ifmt_ctx, nullptr);
    if (ret < 0) goto cleanup;

    video_idx = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_idx < 0) goto cleanup;

    in_stream = ifmt_ctx->streams[video_idx];
    printf("[FFMPEG] 输入流: %dx%d, codec=%s\n",
           in_stream->codecpar->width, in_stream->codecpar->height,
           avcodec_get_name(in_stream->codecpar->codec_id));

    decoder_codec = avcodec_find_decoder(in_stream->codecpar->codec_id);
    if (!decoder_codec) { fprintf(stderr, "找不到解码器\n"); goto cleanup; }

    decoder_ctx = avcodec_alloc_context3(decoder_codec);
    avcodec_parameters_to_context(decoder_ctx, in_stream->codecpar);
    ret = avcodec_open2(decoder_ctx, decoder_codec, nullptr);
    if (ret < 0) goto cleanup;

    avformat_alloc_output_context2(&ofmt_ctx, nullptr, "mp4", output_path.c_str());
    if (!ofmt_ctx) { fprintf(stderr, "创建输出上下文失败\n"); goto cleanup; }

    encoder_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!encoder_codec) { fprintf(stderr, "找不到H264编码器\n"); goto cleanup; }

    out_stream = avformat_new_stream(ofmt_ctx, nullptr);
    if (!out_stream) goto cleanup;

    encoder_ctx = avcodec_alloc_context3(encoder_codec);
    encoder_ctx->width = decoder_ctx->width;
    encoder_ctx->height = decoder_ctx->height;
    encoder_ctx->time_base = (AVRational){1, 1000};
    out_stream->time_base = (AVRational){1, 1000};

    input_fps = av_q2d(in_stream->avg_frame_rate);
    if (input_fps <= 0) input_fps = av_q2d(in_stream->r_frame_rate);
    if (input_fps <= 0) input_fps = 25.0;
    printf("[FFMPEG] 输入流帧率: %.2f FPS\n", input_fps);

    encoder_ctx->framerate = av_d2q(input_fps, 1000);
    encoder_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    encoder_ctx->bit_rate = 4000000;
    encoder_ctx->gop_size = 12;
    encoder_ctx->max_b_frames = 1;
    encoder_ctx->rc_min_rate = 2000000;
    encoder_ctx->rc_max_rate = 8000000;
    encoder_ctx->rc_buffer_size = 4000000;
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        encoder_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    ret = avcodec_open2(encoder_ctx, encoder_codec, nullptr);
    if (ret < 0) { fprintf(stderr, "编码器打开失败\n"); goto cleanup; }

    ret = avcodec_parameters_from_context(out_stream->codecpar, encoder_ctx);
    if (ret < 0) goto cleanup;

    if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ofmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) goto cleanup;
    }

    ret = avformat_write_header(ofmt_ctx, nullptr);
    if (ret < 0) { fprintf(stderr, "写文件头失败\n"); goto cleanup; }

    printf("[FFMPEG] 开始录像 (保留原始PTS)...\n");

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    sw_frame = av_frame_alloc();
    start_time = std::chrono::steady_clock::now();

    while (!is_stop) {
        ret = av_read_frame(ifmt_ctx, pkt);
        if (ret < 0) break;

        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt);
            continue;
        }

        ret = avcodec_send_packet(decoder_ctx, pkt);
        if (ret < 0) { av_packet_unref(pkt); continue; }

        while (ret >= 0) {
            ret = avcodec_receive_frame(decoder_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) break;

            frame_count++;

            if (frame->format != AV_PIX_FMT_YUV420P) {
                sw_frame->format = AV_PIX_FMT_YUV420P;
                sw_frame->width = frame->width;
                sw_frame->height = frame->height;
                av_frame_get_buffer(sw_frame, 0);
                av_image_copy(sw_frame->data, sw_frame->linesize,
                              (const uint8_t**)frame->data, frame->linesize,
                              (AVPixelFormat)frame->format,
                              frame->width, frame->height);
                av_frame_copy_props(sw_frame, frame);
            }

            AVFrame* enc_frame = (frame->format == AV_PIX_FMT_YUV420P) ? frame : sw_frame;

            // 用帧序号生成单调递增PTS(ms)，避免B帧导致的PTS跳跃
            enc_frame->pts = (int64_t)(frame_count * (1000.0 / input_fps));

            ret = avcodec_send_frame(encoder_ctx, enc_frame);
            if (ret < 0) break;

            while (ret >= 0) {
                ret = avcodec_receive_packet(encoder_ctx, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;

                av_packet_rescale_ts(pkt, (AVRational){1, 1000}, out_stream->time_base);
                pkt->stream_index = 0;
                av_write_frame(ofmt_ctx, pkt);
                av_packet_unref(pkt);
            }

            if (sw_frame->data[0]) av_frame_unref(sw_frame);
            av_frame_unref(frame);

            if (frame_count % 100 == 0)
                printf("[FFMPEG] 已处理 %d 帧\n", frame_count);
        }

        av_packet_unref(pkt);

        if (record_seconds > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= record_seconds) break;
        }
    }

    ret = avcodec_send_frame(encoder_ctx, nullptr);
    while (ret >= 0) {
        ret = avcodec_receive_packet(encoder_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        av_packet_rescale_ts(pkt, (AVRational){1, 1000}, out_stream->time_base);
        pkt->stream_index = 0;
        av_write_frame(ofmt_ctx, pkt);
        av_packet_unref(pkt);
    }

    av_write_trailer(ofmt_ctx);
    printf("[FFMPEG] 录像完成, 共 %d 帧\n", frame_count);

cleanup:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    if (encoder_ctx) avcodec_free_context(&encoder_ctx);
    if (decoder_ctx) avcodec_free_context(&decoder_ctx);
    if (ofmt_ctx) {
        if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&ofmt_ctx->pb);
        avformat_free_context(ofmt_ctx);
    }
    if (ifmt_ctx) avformat_close_input(&ifmt_ctx);
    avformat_network_deinit();
    return frame_count > 0;
}

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    signal(SIGINT, signal_handler_func);

    std::string out_dir = FLAGS_output_dir;
    std::string cmd = "mkdir -p " + out_dir;
    system(cmd.c_str());

    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    bool ok = false;

    if (FLAGS_method == "ffmpeg") {
        std::string out_path = out_dir + "/rtsp_ffmpeg_" + timestamp + ".mp4";
        printf("=== FFMPEG 方式 (保留PTS) ===\n");
        printf("输出: %s\n", out_path.c_str());
        ok = use_ffmpeg_record(FLAGS_rtsp_url, out_path, FLAGS_record_seconds);
    } else {
        std::string out_path = out_dir + "/rtsp_hw_" + timestamp + ".avi";
        printf("=== 硬件解码 + OpenCV 方式 ===\n");
        printf("输出: %s\n", out_path.c_str());
        ok = use_hardware_decoder_record(
            FLAGS_rtsp_url, FLAGS_width, FLAGS_height,
            out_path, FLAGS_record_seconds);
    }

    if (ok) {
        printf("\n✓ 录像完成!\n");
        if (FLAGS_record_seconds > 0)
            printf("  预期录制 %d 秒, 播放器检查时长是否匹配\n", FLAGS_record_seconds);
    } else {
        printf("\n✗ 录像失败\n");
    }
    return ok ? 0 : -1;
}
