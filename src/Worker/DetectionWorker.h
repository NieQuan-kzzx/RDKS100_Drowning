#pragma once

#include <QObject>
#include <QDateTime>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

#include "RTSPCamera.h"
#include "ThreadSafeQueue.h"
#include "BaseInfer.h"

// 前置声明，减少头文件耦合
class LogicBase; 

namespace fs = std::filesystem;

class DetectionWorker : public QObject {
    Q_OBJECT
public:
    explicit DetectionWorker(RTSPCamera* cam,  int id, QObject* parent = nullptr);
    ~DetectionWorker();

    // 核心循环线程
    void processLoop();     // 采集线程：取图 -> 原始流分发
    void inferenceLoop();   // 推理线程：AI计算 -> 业务逻辑判定 -> 结果分发

    // 状态控制
    void setPaused(bool p);
    void stop();
    void triggerSnapshot();
    
    // 录制控制
    void setOriRecording(bool start, const std::string& path = "");      // 原始路录制
    void setInferRecording(bool start, const std::string& path = ""); // 推理路录制
    
    void startDualRecording(const std::string& timeStr);
    void stopAllRecording();
    
    // 模型与逻辑插件切换
    void switchModel(const std::string& type, const std::string& path);
    
    // 状态查询
    bool isInferRunning() const { return m_is_infer_running.load(); }
    bool isPaused() const { return m_isPaused.load(); }

signals:
    void frameReady(cv::Mat frame);      // 源画面信号
    void inferFrameReady(cv::Mat frame); // 最终处理画面信号（带框、带报警）
    void snapshotReady(cv::Mat raw, cv::Mat infer, int id);

private:
    // 基础组件
    RTSPCamera* m_cam;
    std::atomic<bool> m_running;
    std::atomic<bool> m_isPaused;
    std::atomic<bool> m_is_processing;
    std::atomic<bool> m_is_infer_running;
    std::atomic<bool> m_needSnapshot;
    
    // 推理引擎与业务逻辑 (核心解耦部分)
    std::unique_ptr<Inf::BaseInfer> m_inferEngine; 
    std::unique_ptr<LogicBase>      m_currentLogic; // 动态挂载的具体业务逻辑
    std::mutex m_inferMtx;                          // 保护引擎和逻辑的切换安全
    
    ThreadSafeQueue<cv::Mat> m_inferQueue; 

    int m_id;

    // 录制相关：原始流
    std::atomic<bool> m_isRecording;
    ThreadSafeQueue<cv::Mat> m_recordQueue;
    std::thread m_recordThread;
    cv::VideoWriter m_videoWriter;
    std::mutex m_writerMtx;
    std::string m_OripendingRecordPath;
    void recordLoop();

    // 录制相关：推理/业务流
    std::atomic<bool> m_isInferRecording;
    ThreadSafeQueue<cv::Mat> m_inferRecordQueue;
    std::thread m_inferRecordThread;
    cv::VideoWriter m_inferVideoWriter;
    std::mutex m_inferWriterMtx;
    std::string m_InferpendingRecordPath;
    void inferRecordLoop();

    void initStorage();
};