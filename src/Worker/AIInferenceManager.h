#pragma once

#include <QObject>
#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include "BaseInfer.h"
#include "ThreadSafeQueue.h"
#include "LogicBase.h"

/**
 * @brief AI推理管理器 - 专门负责AI模型推理和业务逻辑处理
 */
class AIInferenceManager : public QObject {
    Q_OBJECT

public:
    explicit AIInferenceManager(QObject* parent = nullptr);
    ~AIInferenceManager();

    // 模型管理
    bool switchModel(const std::string& type, const std::string& path);

    // 推理控制
    void startInference();
    void stopInference();
    void setPaused(bool paused);
    void triggerSnapshot();

    // 数据输入
    void submitFrame(const cv::Mat& frame);

    // 状态查询
    bool isRunning() const { return m_isRunning.load(); }
    bool isPaused() const { return m_isPaused.load(); }
    std::string getCurrentModelType() const { return m_currentModelType; }

signals:
    void inferenceFrameReady(cv::Mat frame);      // 推理结果帧信号
    void snapshotReady(cv::Mat raw, cv::Mat infer); // 推理截图信号
    void inferenceError(const QString& error);    // 推理错误信号
    void modelSwitched(const QString& modelType); // 模型切换信号

private:
    void inferenceLoop();  // 推理循环
    void processResults(cv::Mat& frame, const std::vector<Inf::Detection>& results);

private:
    // 推理引擎和业务逻辑
    std::unique_ptr<Inf::BaseInfer> m_inferEngine;
    std::unique_ptr<LogicBase> m_currentLogic;
    std::mutex m_engineMutex;

    // 推理队列
    ThreadSafeQueue<cv::Mat> m_inferenceQueue;

    // 状态控制
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_isPaused{false};
    std::atomic<bool> m_needSnapshot{false};
    std::thread m_inferenceThread;

    // 当前模型信息
    std::string m_currentModelType;
    std::string m_currentModelPath;
};