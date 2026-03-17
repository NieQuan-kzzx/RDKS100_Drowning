#pragma once

#include <QObject>
#include <atomic>
#include <thread>
#include <mutex>
#include "ThreadSafeQueue.h"
#include <opencv2/opencv.hpp>

/**
 * @brief 录制管理器 - 专门负责视频录制功能
 */
class RecordingManager : public QObject {
    Q_OBJECT

public:
    explicit RecordingManager(QObject* parent = nullptr);
    ~RecordingManager();

    // 原始流录制控制
    bool startOriginalRecording(const std::string& path);
    void stopOriginalRecording();
    bool isOriginalRecording() const { return m_isOriginalRecording.load(); }

    // 推理流录制控制
    bool startInferenceRecording(const std::string& path);
    void stopInferenceRecording();
    bool isInferenceRecording() const { return m_isInferenceRecording.load(); }

    // 双录制控制
    bool startDualRecording(const std::string& basePath);
    void stopAllRecording();

    // 数据输入
    void submitOriginalFrame(const cv::Mat& frame);
    void submitInferenceFrame(const cv::Mat& frame);

    // 录制信息
    struct RecordingInfo {
        std::string originalPath;
        std::string inferencePath;
        std::chrono::system_clock::time_point startTime;
        size_t originalFrameCount;
        size_t inferenceFrameCount;
    };

    RecordingInfo getRecordingInfo() const;

signals:
    void recordingStarted(const QString& originalPath, const QString& inferencePath);
    void recordingStopped();
    void recordingError(const QString& error);
    void frameRecorded(bool isOriginal, size_t frameCount);

private:
    void originalRecordLoop();
    void inferenceRecordLoop();
    std::string generateRecordingPath(const std::string& basePath, const std::string& suffix);

private:
    // 原始流录制
    std::atomic<bool> m_isOriginalRecording{false};
    ThreadSafeQueue<cv::Mat> m_originalQueue;
    std::thread m_originalRecordThread;
    cv::VideoWriter m_originalVideoWriter;
    std::mutex m_originalWriterMutex;
    std::string m_originalRecordPath;

    // 推理流录制
    std::atomic<bool> m_isInferenceRecording{false};
    ThreadSafeQueue<cv::Mat> m_inferenceQueue;
    std::thread m_inferenceRecordThread;
    cv::VideoWriter m_inferenceVideoWriter;
    std::mutex m_inferenceWriterMutex;
    std::string m_inferenceRecordPath;

    // 录制信息
    mutable std::mutex m_infoMutex;
    RecordingInfo m_recordingInfo;
};