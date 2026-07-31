#include "RTSPCamera.h"

RTSPCamera::RTSPCamera(const std::string& url, int width, int height,
                       int _queue_max_length, int _capture_interval_ms,
                       bool _is_full_drop)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      m_rtsp_url(url), m_width(width), m_height(height),
      m_matPool(MatPoolManager::getPool(cv::Size(width, height), CV_8UC3)) {

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
    std::lock_guard<std::mutex> lock(m_start_mutex);

    if (!this->is_running.load()) return;

    this->is_running.store(false);
    this->clear();
    cv.notify_all();
    PLOGI << "RTSPCamera: Stop signal sent.";

    if (sensor_thread.joinable()) {
        sensor_thread.join();
        PLOGI << "RTSPCamera: Thread joined safely.";
    }
}

void RTSPCamera::pause() { m_is_paused.store(true); }
void RTSPCamera::resume() { m_is_paused.store(false); }

void RTSPCamera::dataCollectionLoop() {
    softwareDecodeLoop();
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
            m_read_fail_count++;
            if (m_read_fail_count < 3) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            m_read_fail_count = 0;
            PLOGE << "RTSPCamera: Failed to read frame from software decoder, reopening stream...";
            m_soft_cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (!m_soft_cap.open(m_rtsp_url, cv::CAP_FFMPEG)) {
                PLOGE << "RTSPCamera: Failed to reopen RTSP stream";
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } else {
                m_soft_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                PLOGI << "RTSPCamera: Stream reopened successfully";
            }
            continue;
        } else {
            m_read_fail_count = 0;
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
