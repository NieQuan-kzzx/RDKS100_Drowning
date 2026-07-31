#pragma once

#include "ImageSensor.h"
#include <string>
#include <atomic>
#include "MatPool.h"


class RTSPCamera : public ImageSensor
{
public:
    RTSPCamera(const std::string& url, int width, int height, 
               int _queue_max_length = 10, int _capture_interval_ms = 0,
               bool _is_full_drop = true);
    ~RTSPCamera();

    void start() override;
    void stop() override;

    // 控制接口
    void pause();
    void resume();
    
    // 功能接口
    bool captureSnapshot(const std::string& path);
    bool startRecording(const std::string& path);
    void stopRecording();

protected:
    virtual void dataCollectionLoop() override;
    void softwareDecodeLoop();

private:
    std::string m_rtsp_url;
    int m_width;
    int m_height;

    MatPool& m_matPool;

    // 状态控制
    std::atomic<bool> m_is_paused{false};
    std::mutex m_start_mutex;

    // 录制相关
    std::atomic<bool> m_is_recording{false};
    cv::VideoWriter m_video_writer;
    std::mutex m_record_mtx;

    // 软解
    cv::VideoCapture m_soft_cap;
    int m_read_fail_count = 0;
};