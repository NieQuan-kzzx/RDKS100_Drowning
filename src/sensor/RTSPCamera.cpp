#include "RTSPCamera.h"

RTSPCamera::RTSPCamera(const std::string& url, int width, int height,
                       int _queue_max_length, int _capture_interval_ms, bool _is_full_drop)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      m_rtsp_url(url), m_width(width), m_height(height) 
{
    // 构造函数逻辑
}

RTSPCamera::~RTSPCamera() {
    stop();
    // 析构时确保线程安全回收
    if (sensor_thread.joinable()) {
        sensor_thread.join();
    }
}

void RTSPCamera::start() {
    if (this->is_running.load()) {
        PLOGI << "RTSPCamera: Already running.";
        return;
    }

    // 1. 调用基类的 clear() 清空旧图像队列
    this->clear(); 

    this->is_running.store(true);

    // 2. 检查基类的 sensor_thread 是否还在运行，确保安全回收
    if (sensor_thread.joinable()) {
        sensor_thread.join();
    }

    // 3. 启动采集线程
    sensor_thread = std::thread(&RTSPCamera::dataCollectionLoop, this);
    PLOGI << "RTSPCamera: Data collection thread started.";
}

void RTSPCamera::stop() {
    // 改变标志位，让 dataCollectionLoop 退出
    this->is_running.store(false);
    PLOGI << "RTSPCamera: Stop signal sent.";
}

void RTSPCamera::pause() { m_is_paused.store(true); }
void RTSPCamera::resume() { m_is_paused.store(false); }

void RTSPCamera::dataCollectionLoop() {
    // 1. 初始化地平线硬解
    m_decoder = sp_init_decoder_module();
    int ret = sp_start_decode(m_decoder, const_cast<char*>(m_rtsp_url.c_str()), 
                             0, SP_ENCODER_H264, m_width, m_height);
    
    if (ret != 0) {
        PLOGE << "RTSPCamera: Failed to start hardware decoder, ret: " << ret;
        this->is_running.store(false);
        return;
    }

    PLOGI << "RTSPCamera: Hardware decoding started.";

    // 预分配 YUV 内存，避免循环内重复创建
    cv::Mat yuv_frame(m_height * 3 / 2, m_width, CV_8UC1);

    while (this->is_running.load()) {
        // 2. 暂停处理
        if (m_is_paused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 3. 获取硬件解码图像
        ret = sp_decoder_get_image(m_decoder, reinterpret_cast<char*>(yuv_frame.data));
        
        if (ret == 0) { 
            cv::Mat bgr_frame;
            // 地平线硬解输出通常为 NV12 格式
            cv::cvtColor(yuv_frame, bgr_frame, cv::COLOR_YUV2BGR_NV12);
            
            // 存入队列，使用 clone() 确保内存独立，防止 Segfault
            this->enqueueData(bgr_frame.clone()); 
            
            // 采集频率控制
            if (capture_interval_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(capture_interval_ms));
            }
        } else {
            // 如果获取失败，小睡一下，避免死循环空转 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // 4. 资源释放逻辑 (重要：确保在循环退出后彻底释放硬件句柄)
    if (m_decoder) {
        PLOGI << "RTSPCamera: Releasing hardware resources...";
        sp_stop_decode(m_decoder);
        sp_release_decoder_module(m_decoder);
        m_decoder = nullptr; 
        
        // 驱动释放后的“冷却”期，防止紧接着第二次启动时报错
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
        PLOGI << "RTSPCamera: Hardware released safely.";
    }
}

// 截图功能实现
bool RTSPCamera::captureSnapshot(const std::string& path) {
    cv::Mat frame = getLastestFrame(); 
    if (frame.empty()) return false;
    return cv::imwrite(path, frame);
}

// 录制功能 (此处建议保留接口，但具体的录制逻辑已移至 DetectionWorker 异步实现)
bool RTSPCamera::startRecording(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_record_mtx);
    m_video_writer.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(m_width, m_height));
    m_is_recording.store(m_video_writer.isOpened());
    return m_is_recording.load();
}

void RTSPCamera::stopRecording() {
    std::lock_guard<std::mutex> lock(m_record_mtx);
    m_is_recording.store(false);
    if (m_video_writer.isOpened()) {
        m_video_writer.release();
    }
}