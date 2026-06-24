#include "RTSPCamera.h"

// 内存池键名常量定义
const std::string RTSPCamera::YUV_FRAME_KEY = "yuv_frame";
const std::string RTSPCamera::BGR_FRAME_KEY = "bgr_frame";

RTSPCamera::RTSPCamera(const std::string& url, int width, int height,
                       int _queue_max_length, int _capture_interval_ms,
                       bool _is_full_drop, DecodeMode decode_mode)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      m_rtsp_url(url), m_width(width), m_height(height),
      m_matPool(MatPoolManager::getPool(cv::Size(width, height), CV_8UC3)),
      m_decode_mode(decode_mode) {

    if (m_decode_mode == HARDWARE) {
        yuv_frame_ = cv::Mat(cv::Size(width, height * 3 / 2), CV_8UC1);
        bgr_frame_ = cv::Mat(cv::Size(width, height), CV_8UC3);
    }

    setMatPool(&m_matPool);
}

RTSPCamera::~RTSPCamera() {
    stop();
    if (sensor_thread.joinable()) {
        sensor_thread.join();
    }
}

void RTSPCamera::start() {
    std::lock_guard<std::mutex> lock(m_start_mutex);

    if (this->is_running.load()) return;

    if (sensor_thread.joinable()) {
        sensor_thread.join();
    }
    sensor_thread = std::thread();

    this->clear();
    this->is_running.store(true);

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
    if (!this->is_running.load()) return;

    this->is_running.store(false);
    PLOGI << "RTSPCamera: Stop signal sent.";

    if (sensor_thread.joinable()) {
        sensor_thread.join();
        PLOGI << "RTSPCamera: Thread joined safely.";
    }
}

void RTSPCamera::pause() { m_is_paused.store(true); }
void RTSPCamera::resume() { m_is_paused.store(false); }

void RTSPCamera::dataCollectionLoop() {
    if (m_decode_mode == HARDWARE) {
        hardwareDecodeLoop();
    } else {
        softwareDecodeLoop();
    }
}

void RTSPCamera::hardwareDecodeLoop() {
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
        ret = sp_decoder_get_image(m_decoder, reinterpret_cast<char*>(yuv_frame_.data));

        if (ret == 0) {
            if (m_is_paused.load()) {
                this->clear();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }
            cv::cvtColor(yuv_frame_, bgr_frame_, cv::COLOR_YUV2BGR_NV12);

            cv::Mat poolMat = m_matPool.getMat();
            if(!poolMat.empty()){
                cv::cvtColor(yuv_frame_, poolMat, cv::COLOR_YUV2BGR_NV12);
                this->enqueueData(poolMat);
            }

            if (capture_interval_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(capture_interval_ms));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    if (m_decoder) {
        PLOGI << "RTSPCamera: Releasing hardware resources...";
        sp_stop_decode(m_decoder);
        sp_release_decoder_module(m_decoder);
        m_decoder = nullptr;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        PLOGI << "RTSPCamera: Hardware released safely.";
    }
}

void RTSPCamera::softwareDecodeLoop() {
    PLOGI << "RTSPCamera: Opening RTSP with software decoding (cv::VideoCapture)...";

    if (!m_soft_cap.open(m_rtsp_url, cv::CAP_FFMPEG)) {
        PLOGE << "RTSPCamera: Failed to open RTSP stream via cv::VideoCapture";
        this->is_running.store(false);
        return;
    }

    m_soft_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    PLOGI << "RTSPCamera: Software decoding started.";

    cv::Mat frame;
    while (this->is_running.load()) {
        if (m_is_paused.load()) {
            this->clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        if (!m_soft_cap.read(frame)) {
            PLOGE << "RTSPCamera: Failed to read frame from software decoder";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        cv::Mat poolMat = m_matPool.getMat();
        if (!poolMat.empty()) {
            if (frame.size() != poolMat.size() || frame.type() != poolMat.type()) {
                cv::resize(frame, poolMat, poolMat.size());
            } else {
                frame.copyTo(poolMat);
            }
            this->enqueueData(poolMat);
        }

        if (capture_interval_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(capture_interval_ms));
        }
    }

    m_soft_cap.release();
    PLOGI << "RTSPCamera: Software decoder released.";
}

bool RTSPCamera::captureSnapshot(const std::string& path) {
    cv::Mat frame = getLastestFrame();
    if (frame.empty()) return false;
    return cv::imwrite(path, frame);
}

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
