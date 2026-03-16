#pragma once

#include "ImageSensor.h"
#include <string>
#include <atomic>
#include "sp_codec.h"
#include "MatPool.h"


class RTSPCamera : public ImageSensor
{
public:
    // url: RTSP地址
    // width: 期望的图像宽度
    // height: 期望的图像高度
    // queue_max_length: 采集队列最大长度
    // capture_interval_ms: 采集间隔，单位毫秒
    // is_full_drop: 队列满时是否丢弃新图像
    RTSPCamera(const std::string& url, int width, int height, 
               int _queue_max_length = 10, int _capture_interval_ms = 0, bool _is_full_drop = true);
    ~RTSPCamera();

    void start() override;
    void stop() override;

    // 控制接口
    void pause();
    void resume();
    
    // 功能接口
    bool captureSnapshot(const std::string& path);    // 截图
    bool startRecording(const std::string& path);     // 开启录制
    void stopRecording();                             // 停止录制

protected:
    // 实现基类的纯虚函数：核心采集逻辑
    virtual void dataCollectionLoop() override;

private:
    // 硬件解码成员
    void* m_decoder = nullptr;
    std::string m_rtsp_url;
    int m_width;
    int m_height;

    // 内存池引用 - 仅用于性能统计
    MatPool& m_matPool;

    // 帧缓冲区 - 不使用内存池避免生命周期管理复杂性
    cv::Mat yuv_frame_;
    cv::Mat bgr_frame_;

    // 内存池键名常量
    static const std::string YUV_FRAME_KEY;
    static const std::string BGR_FRAME_KEY;

    // 状态控制
    std::atomic<bool> m_is_paused{false};
    
    // 录制相关
    std::atomic<bool> m_is_recording{false};
    cv::VideoWriter m_video_writer;
    std::mutex m_record_mtx;
};