#pragma once

#include <QObject>
#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include <map>
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

    void setInferenceQueueMaxSize(int size) { m_inferenceQueueMaxSize = size; }

    // 模型管理（同步版本，供特殊场景使用）
    bool switchModel(const std::string& type, const std::string& path,
                     const std::vector<std::string>& labels = {},
                     const std::map<std::string, std::string>& params = {});

    // 模型管理（异步执行，不阻塞调用线程）
    void switchModelAsync(const std::string& type, const std::string& path,
                          const std::vector<std::string>& labels = {},
                          const std::map<std::string, std::string>& params = {});

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
    void modelSwitched(const QString& modelType); // 模型切换完成信号
    void modelSwitching();                        // 模型切换开始信号（用于UI提示）

private:
    void inferenceLoop();  // 推理循环
    void processResults(cv::Mat& frame, const std::vector<Inf::Detection>& results);
    void doSwitchModel(const std::string& type, const std::string& path,
                       const std::vector<std::string>& labels,
                       const std::map<std::string, std::string>& params);

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

    // 异步模型切换线程
    std::thread m_switchThread;
    std::atomic<bool> m_isSwitching{false};

    int m_inferenceQueueMaxSize = 2;
};