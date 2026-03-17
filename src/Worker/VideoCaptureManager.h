#pragma once

#include <QObject>
#include <atomic>
#include <thread>
#include "RTSPCamera.h"
#include "ThreadSafeQueue.h"

/**
 * @brief 视频采集管理器 - 专门负责从摄像头采集视频数据
 */
class VideoCaptureManager : public QObject {
    Q_OBJECT

public:
    explicit VideoCaptureManager(RTSPCamera* camera, QObject* parent = nullptr);
    ~VideoCaptureManager();

    // 控制接口
    void startCapture();
    void stopCapture();
    void setPaused(bool paused);
    void triggerSnapshot();

    // 状态查询
    bool isRunning() const { return m_running.load(); }
    bool isPaused() const { return m_isPaused.load(); }

signals:
    void frameReady(cv::Mat frame);           // 原始帧就绪信号
    void snapshotReady(cv::Mat frame);        // 截图就绪信号
    void captureError(const QString& error);  // 采集错误信号

private:
    void captureLoop();  // 采集循环

private:
    RTSPCamera* m_camera;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isPaused{false};
    std::atomic<bool> m_needSnapshot{false};
    std::thread m_captureThread;

    // 截图队列（用于异步处理）
    ThreadSafeQueue<cv::Mat> m_snapshotQueue;
    std::thread m_snapshotThread;
    std::atomic<bool> m_snapshotRunning{false};
    void snapshotLoop();
};