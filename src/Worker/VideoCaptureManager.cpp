#include "VideoCaptureManager.h"
#include <plog/Log.h>

VideoCaptureManager::VideoCaptureManager(RTSPCamera* camera, QObject* parent)
    : QObject(parent), m_camera(camera) {
}

VideoCaptureManager::~VideoCaptureManager() {
    stopCapture();
    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }
    if (m_snapshotThread.joinable()) {
        m_snapshotThread.join();
    }
}

void VideoCaptureManager::startCapture() {
    if (m_running.load()) {
        PLOGW << "VideoCaptureManager: Already running";
        return;
    }

    // 如果因为某种异常导致 stop 没处理干净，这里做最后兜底
    if (m_captureThread.joinable()) m_captureThread.join();
    if (m_snapshotThread.joinable()) m_snapshotThread.join();

    m_running.store(true);
    m_snapshotRunning.store(true);

    m_captureThread = std::thread(&VideoCaptureManager::captureLoop, this);
    m_snapshotThread = std::thread(&VideoCaptureManager::snapshotLoop, this);

    PLOGI << "VideoCaptureManager: Started";
}

void VideoCaptureManager::stopCapture() {
    if (!m_running.load() && !m_snapshotRunning.load()) {
        return;
    }

    // 1. 发出停止信号
    m_running.store(false);
    m_snapshotRunning.store(false);

    // 2. 唤醒并停止截图队列
    m_snapshotQueue.clear();
    m_snapshotQueue.enqueue(cv::Mat()); // 发送空帧唤醒 dequeue 阻塞

    // 3. 【核心修复】必须在这里 join，确保 start 之前线程已经没了
    if (m_captureThread.joinable()) {
        m_captureThread.join();
        PLOGI << "VideoCaptureManager: Capture thread joined.";
    }
    if (m_snapshotThread.joinable()) {
        m_snapshotThread.join();
        PLOGI << "VideoCaptureManager: Snapshot thread joined.";
    }
}

void VideoCaptureManager::setPaused(bool paused) {
    m_isPaused.store(paused);
    if (m_camera) {
        if (paused) {
            m_camera->pause();
        } else {
            m_camera->resume();
        }
    }
}

void VideoCaptureManager::triggerSnapshot() {
    m_needSnapshot.store(true);
}

void VideoCaptureManager::captureLoop() {
    if (!m_camera) {
        emit captureError("Camera not available");
        return;
    }

    m_camera->start();

    while (m_running.load()) {
        if (m_isPaused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        cv::Mat frame = m_camera->getData();
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 发送原始帧信号
        emit frameReady(frame.clone());

        // 处理截图请求
        if (m_needSnapshot.load()) {
            m_snapshotQueue.enqueue(frame.clone());
            m_needSnapshot.store(false);
        }
    }

    m_camera->stop();
    PLOGI << "VideoCaptureManager: Capture stopped";
}

void VideoCaptureManager::snapshotLoop() {
    while (m_snapshotRunning.load()) {
        cv::Mat snapshotFrame = m_snapshotQueue.dequeue();
        if (snapshotFrame.empty()) {
            continue; // 停止信号
        }

        // 发送截图信号
        emit snapshotReady(snapshotFrame);
    }

    PLOGI << "VideoCaptureManager: Snapshot processing stopped";
}