#include "RTSPCamera.h"

RTSPCamera::RTSPCamera(const std::string& url, int width, int height,
                       int _queue_max_length, int _capture_interval_ms, bool _is_full_drop)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      m_rtsp_url(url), m_width(width), m_height(height)
{
}

RTSPCamera::~RTSPCamera()
{
    stop(); // 确保基类线程停止
}

void RTSPCamera::pause() { m_is_paused = true; }
void RTSPCamera::resume() { m_is_paused = false; }

void RTSPCamera::dataCollectionLoop()
{
    // 1. 初始化地平线硬解模块
    m_decoder = sp_init_decoder_module();
    // 传入 URL，sp_codec 会自动启动拉流线程
    int ret = sp_start_decode(m_decoder, const_cast<char*>(m_rtsp_url.c_str()), 
                             0, SP_ENCODER_H264, m_width, m_height);
    if (ret != 0) {
        PLOG_ERROR << "RTSPCamera: Failed to start hardware decoder for " << m_rtsp_url;
        return;
    }
    PLOG_INFO << "RTSPCamera: Decoding started for " << m_rtsp_url;


    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 预分配 NV12 接收缓冲区
    cv::Mat yuv_frame(m_height * 3 / 2, m_width, CV_8UC1);

    while (this->is_running)
    {
        
        if (m_is_paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 2. 从硬件拉取图像
        ret = sp_decoder_get_image(m_decoder, reinterpret_cast<char*>(yuv_frame.data));
        
        if (ret == 0) {
            PLOG_VERBOSE << "New frame decoded!"; // 打印日志
            // 3. 颜色空间转换：NV12 -> BGR
            cv::Mat bgr_frame;
            cv::cvtColor(yuv_frame, bgr_frame, cv::COLOR_YUV2BGR_NV12);

            // 4. 录制逻辑 (可选)
            if (m_is_recording) {
                std::lock_guard<std::mutex> lock(m_record_mtx);
                if (m_video_writer.isOpened()) {
                    m_video_writer.write(bgr_frame);
                }
            }

            // 5. 使用基类方法将数据入队，供 UI 或 YOLO 获取
            this->enqueueData(bgr_frame);
        }

        // 遵循基类的采集间隔配置
        if (this->capture_interval_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(this->capture_interval_ms));
        }
    }

    // 资源清理
    sp_stop_decode(m_decoder);
    sp_release_decoder_module(m_decoder);
    m_decoder = nullptr;
}

bool RTSPCamera::captureSnapshot(const std::string& path)
{
    cv::Mat frame = getLastestFrame(); // 使用基类方法
    if (frame.empty()) return false;
    return cv::imwrite(path, frame);
}

bool RTSPCamera::startRecording(const std::string& path)
{
    std::lock_guard<std::mutex> lock(m_record_mtx);
    m_video_writer.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(m_width, m_height));
    m_is_recording = m_video_writer.isOpened();
    return m_is_recording;
}

void RTSPCamera::stopRecording()
{
    std::lock_guard<std::mutex> lock(m_record_mtx);
    m_is_recording = false;
    if (m_video_writer.isOpened()) {
        m_video_writer.release();
    }
}