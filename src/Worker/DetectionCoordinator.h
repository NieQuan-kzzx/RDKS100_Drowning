#pragma once

#include <QObject>
#include <memory>
#include <thread>
#include <atomic>
#include "VideoCaptureManager.h"
#include "AIInferenceManager.h"
#include "RecordingManager.h"

/**
 * @brief 检测协调器 - 协调各个专门的组件，提供统一的接口
 *
 * 这个类只负责组件协调
 */
class DetectionCoordinator : public QObject {
    Q_OBJECT

public:
    explicit DetectionCoordinator(RTSPCamera* camera, int workerId, QObject* parent = nullptr);
    ~DetectionCoordinator();

    // 系统控制
    void start();
    void stop();
    void setPaused(bool paused);

    // 模型管理（同步版本，供特殊场景使用）
    bool switchModel(const std::string& type, const std::string& path,
                     const std::vector<std::string>& labels = {},
                     const std::map<std::string, std::string>& params = {});

    // 模型管理（异步执行，不阻塞调用线程）
    void switchModelAsync(const std::string& type, const std::string& path,
                          const std::vector<std::string>& labels = {},
                          const std::map<std::string, std::string>& params = {});

    // 录制控制
    bool startRecording(const std::string& basePath = "");
    void stopRecording();

public slots:
    void onRecordingStarted();
    void onRecordingStopped();

    // 快照控制
    void triggerSnapshot();

    // 状态查询
    bool isRunning() const;
    bool isPaused() const;
    bool isRecording() const;
    std::string getCurrentModel() const;

    // 获取管理器引用（用于特殊配置）
    VideoCaptureManager* getCaptureManager() { return m_captureManager.get(); }
    AIInferenceManager* getInferenceManager() { return m_inferenceManager.get(); }
    RecordingManager* getRecordingManager() { return m_recordingManager.get(); }

signals:
    // 转发来自各个管理器的信号
    void frameReady(cv::Mat frame);
    void inferenceFrameReady(cv::Mat frame);
    void snapshotReady(cv::Mat raw, cv::Mat infer, int workerId);
    void recordingStarted(const QString& originalPath, const QString& inferencePath);
    void recordingStopped();
    void systemError(const QString& error);
    void modelSwitched(const QString& modelType);
    void modelSwitching();  // 模型切换开始信号

private slots:
    void onFrameReady(const cv::Mat& frame);
    void onInferenceFrameReady(const cv::Mat& frame);
    void onSnapshotReady(const cv::Mat& raw, const cv::Mat& infer);
    void onCaptureError(const QString& error);
    void onInferenceError(const QString& error);

private:
    void setupConnections();
    void initializeStorage();

    std::atomic<bool> m_recordingActive{false};

private:
    int m_workerId;

    // 专门的组件管理器
    std::unique_ptr<VideoCaptureManager> m_captureManager;
    std::unique_ptr<AIInferenceManager> m_inferenceManager;
    std::unique_ptr<RecordingManager> m_recordingManager;

    // 系统状态
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_isPaused{false};
};