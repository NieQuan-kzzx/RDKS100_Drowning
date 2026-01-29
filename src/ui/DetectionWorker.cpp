#include "DetectionWorker.h"
#include <QDateTime>
#include <plog/Log.h>

DetectionWorker::DetectionWorker(RTSPCamera* cam, QObject* parent)
    : QObject(parent), m_cam(cam), m_running(false), 
      m_isPaused(false), m_needSnapshot(false), m_isRecording(false), m_is_processing(false) {}

DetectionWorker::~DetectionWorker() {
    // 1. 强制停止循环
    m_running.store(false);
    
    // 2. 停止录制并回收线程
    m_isRecording.store(false);
    if (m_recordThread.joinable()) {
        m_recordThread.join();
    }
}

// 异步录制后台线程函数
void DetectionWorker::recordLoop() {
    PLOGI << "Record Thread Started.";
    
    while (m_isRecording.load() || !m_recordQueue.empty()) {
        if (m_recordQueue.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        cv::Mat frameToRecord = m_recordQueue.dequeue(); 
        if (!frameToRecord.empty()) {
            std::lock_guard<std::mutex> lock(m_writerMtx);
            if (m_videoWriter.isOpened()) {
                m_videoWriter.write(frameToRecord);
            }
        }
    }

    // 录制线程自己负责 release，不卡主线程
    {
        std::lock_guard<std::mutex> lock(m_writerMtx);
        if (m_videoWriter.isOpened()) {
            m_videoWriter.release(); 
            PLOGI << "Record Thread: VideoWriter released successfully.";
        }
    }
}

// 核心处理循环
void DetectionWorker::processLoop() {
    // 1. 原子交换，确保同一时间只有一个 processLoop 任务在跑
    if (m_is_processing.exchange(true)) {
        PLOGW << "DetectionWorker: Process loop is already running, skipping.";
        return; 
    }

    m_running.store(true);
    PLOGI << "DetectionWorker: Process loop started.";

    // 2. 这里的 m_cam 访问要非常小心
    while (m_running.load()) {
        
        // 关键点：如果相机被 stop 了，立即安全退出
        if (!m_cam || !m_cam->isRunning()) break;

        if (m_isPaused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        cv::Mat frame = m_cam->getData(); 
        if (frame.empty()) {
            // 如果拿不到数据，可能相机正在关闭，sleep 后再次判断 loop 条件
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 截图逻辑
        if (m_needSnapshot.load()) {
            std::string timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz").toStdString();
            std::string fileName = "Snapshot_" + timeStr + ".jpg";
            if(cv::imwrite(fileName, frame)) {
                PLOGI << "Snapshot captured: " << fileName;
            }
            m_needSnapshot.store(false);
        }

        // 异步录制分发
        if (m_isRecording.load()) {
            if (m_recordQueue.size() < 100) {
                m_recordQueue.enqueue(frame.clone()); 
            }
        }

        // 发送 UI (必须 clone)
        emit frameReady(frame.clone());

        std::this_thread::yield(); 
    }

    // 3. 退出前的状态重置
    m_running.store(false);
    m_is_processing.store(false); 
    PLOGI << "DetectionWorker: Process loop stopped.";
}

void DetectionWorker::setPaused(bool p) { m_isPaused.store(p); }

void DetectionWorker::stop() { 
    m_running.store(false); 
}

void DetectionWorker::triggerSnapshot() { m_needSnapshot.store(true); }

void DetectionWorker::setRecording(bool start, const std::string& path) {
    if (start) {
        if (m_isRecording.load()) return; 

        std::lock_guard<std::mutex> lock(m_writerMtx);
        // RDK 环境下 MJPG 效率最高，1080P/25FPS
        m_videoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        
        if (m_videoWriter.isOpened()) {
            m_isRecording.store(true);
            // 确保旧线程已回收
            if (m_recordThread.joinable()) m_recordThread.join();
            m_recordThread = std::thread(&DetectionWorker::recordLoop, this);
            PLOGI << "Async Recording Started: " << path;
        }
    } else {
        PLOGI << "Requesting stop recording...";
        m_isRecording.store(false); 
        // 故意不在此时 join，避免 UI 线程等待磁盘写完
    }
}