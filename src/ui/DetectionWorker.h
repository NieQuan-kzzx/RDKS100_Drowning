#ifndef DETECTIONWORKER_H
#define DETECTIONWORKER_H

#include <QObject>
#include <QDateTime>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include "RTSPCamera.h"
#include "ThreadSafeQueue.h"
#include "BaseInfer.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class DetectionWorker : public QObject {
    Q_OBJECT
public:
    explicit DetectionWorker(RTSPCamera* cam, QObject* parent = nullptr);
    ~DetectionWorker();

    // 双流水线函数
    void processLoop();     // 原有的：取图 -> 发送源画面信号
    void inferenceLoop();   // 新增的：独立线程做推理 -> 绘图 -> 发送带框信号 -> 录制带框视频

    void setPaused(bool p);
    void stop();
    void triggerSnapshot();
    
    // 录制控制
    void setRecording(bool start, const std::string& path = "");      // 原始画面录制
    void setInferRecording(bool start, const std::string& path = ""); // 推理画面录制
    
    void startDualRecording(const std::string& timeStr);
    void stopAllRecording();
    
    // 模型插件切换
    void switchModel(const std::string& type, const std::string& path);
    // 推理线程是否正在运行
    bool isInferRunning() const { return m_is_infer_running.load(); }

signals:
    void frameReady(cv::Mat frame);      // 源画面信号
    void inferFrameReady(cv::Mat frame); // 带推理框的画面信号

private:
    RTSPCamera* m_cam;
    std::atomic<bool> m_running;
    std::atomic<bool> m_isPaused;
    std::atomic<bool> m_is_processing;
    std::atomic<bool> m_is_infer_running; // 显式控制推理线程退出
    std::atomic<bool> m_needSnapshot;
    
    // 推理相关
    std::unique_ptr<Inf::BaseInfer> m_inferEngine; 
    std::mutex m_inferMtx;
    ThreadSafeQueue<cv::Mat> m_inferQueue; // 给推理线程喂图的队列

    // 录制相关 (原始流)
    std::atomic<bool> m_isRecording;
    ThreadSafeQueue<cv::Mat> m_recordQueue;
    std::thread m_recordThread;
    cv::VideoWriter m_videoWriter;
    std::mutex m_writerMtx;
    void recordLoop();

    // 录制相关 (推理流)
    std::atomic<bool> m_isInferRecording;
    ThreadSafeQueue<cv::Mat> m_inferRecordQueue;
    std::thread m_inferRecordThread;
    cv::VideoWriter m_inferVideoWriter;
    std::mutex m_inferWriterMtx;
    void inferRecordLoop();

    void initStorage();
};

#endif