#ifndef DETECTIONWORKER_H
#define DETECTIONWORKER_H

#include <QObject>
#include <QImage>
#include <QDateTime>
#include <atomic>
#include <mutex>
#include "RTSPCamera.h"
#include "ThreadSafeQueue.h"

class DetectionWorker : public QObject {
    Q_OBJECT
public:
    explicit DetectionWorker(RTSPCamera* cam, QObject* parent = nullptr);
    ~DetectionWorker();

    // 核心循环：丢入 ThreadPool 运行
    void processLoop();

    // 指令接口
    void setPaused(bool p);
    void stop();
    void triggerSnapshot();
    void setRecording(bool start, const std::string& path = "");

signals:
    // 发送给 UI 显示的信号
    void frameReady(cv::Mat frame);
    // 状态反馈信号（可选）
    void statusMessage(QString msg);

private:
    RTSPCamera* m_cam;
    std::atomic<bool> m_running;
    std::atomic<bool> m_isPaused;
    // 防止线程池任务重叠
    std::atomic<bool> m_is_processing;
    // 截图控制
    std::atomic<bool> m_needSnapshot;
    
    // 录制控制
    std::atomic<bool> m_isRecording;
    ThreadSafeQueue<cv::Mat> m_recordQueue; // 录制专用缓冲队列
    std::thread m_recordThread;              // 录制专用线程
    void recordLoop();                       // 录制线程函数
    cv::VideoWriter m_videoWriter;
    std::mutex m_writerMtx;
};

#endif