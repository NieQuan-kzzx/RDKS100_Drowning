#include "RTSPCamera.h"

// 内存池键名常量定义
const std::string RTSPCamera::YUV_FRAME_KEY = "yuv_frame";
const std::string RTSPCamera::BGR_FRAME_KEY = "bgr_frame";

RTSPCamera::RTSPCamera(const std::string& url, int width, int height,
                       int _queue_max_length, int _capture_interval_ms, bool _is_full_drop)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      m_rtsp_url(url), m_width(width), m_height(height),
      m_matPool(MatPoolManager::getPool(cv::Size(width, height), CV_8UC3)) {

    // 预分配帧缓冲区 - 不使用内存池避免生命周期管理问题
    yuv_frame_ = cv::Mat(cv::Size(width, height * 3 / 2), CV_8UC1);
    bgr_frame_ = cv::Mat(cv::Size(width, height), CV_8UC3);
}

RTSPCamera::~RTSPCamera() {
    stop();
    // 析构时确保线程安全回收
    if (sensor_thread.joinable()) {
        sensor_thread.join();
    }

    // 注意：帧缓冲区不使用内存池，无需归还
}

void RTSPCamera::start() {
    std::lock_guard<std::mutex> lock(m_start_mutex);

    if (this->is_running.load()) return;

    // --- 关键修改：强制重置线程对象 ---
    if (sensor_thread.joinable()) {
        sensor_thread.join();
    }
    // 显式移动赋值一个空对象，确保旧句柄被彻底释放
    sensor_thread = std::thread(); 

    this->clear(); 
    this->is_running.store(true);

    // 启动前稍微等待一下硬件驱动彻底冷却
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    try {
        sensor_thread = std::thread(&RTSPCamera::dataCollectionLoop, this);
        PLOGI << "RTSPCamera: Data collection thread started. ID: " << sensor_thread.get_id();
    } catch (const std::exception& e) {
        PLOGE << "Failed to create thread: " << e.what();
        this->is_running.store(false);
    }
}

void RTSPCamera::stop() {
    // 避免重复停止
    if (!this->is_running.load()) return;

    // 1. 改变标志位，让 dataCollectionLoop 退出循环
    this->is_running.store(false);
    PLOGI << "RTSPCamera: Stop signal sent.";

    // 2. 等待线程结束
    // 这里非常重要：因为 dataCollectionLoop 里面有硬解资源释放，
    // 必须等待它执行完，才能返回。
    if (sensor_thread.joinable()) {
        sensor_thread.join();
        PLOGI << "RTSPCamera: Thread joined safely.";
    }
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

    while (this->is_running.load()) {
        // 2. 暂停处理
        if (m_is_paused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 3. 获取硬件解码图像
        ret = sp_decoder_get_image(m_decoder, reinterpret_cast<char*>(yuv_frame_.data));

        if (ret == 0) {
            // 使用预分配的缓冲区进行颜色转换
            cv::cvtColor(yuv_frame_, bgr_frame_, cv::COLOR_YUV2BGR_NV12);

            // 存入队列，使用 clone() 确保内存独立
            // this->enqueueData(bgr_frame_.clone());
            cv::Mat poolMat = m_matPool.getMat(); 
            if(!poolMat.empty()){
                cv::cvtColor(yuv_frame_, poolMat, cv::COLOR_YUV2BGR_NV12);
                this->enqueueData(poolMat); // ImageSensor 内部应负责处理 poolMat 的生命周期
            }

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