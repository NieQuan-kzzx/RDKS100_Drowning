#include "DetectionCoordinator.h"
#include "RTSPCamera.h"
#include <plog/Log.h>

// 条件编译：检查C++17 filesystem支持
#if __has_include(<filesystem>) && __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
// 回退到系统相关API
#include <sys/stat.h>
#include <sys/types.h>
#endif

DetectionCoordinator::DetectionCoordinator(RTSPCamera* camera, int workerId, QObject* parent)
    : QObject(parent), m_workerId(workerId) {

    // 初始化存储目录
    initializeStorage();

    // 创建专门的组件管理器
    m_captureManager = std::make_unique<VideoCaptureManager>(camera, this);
    m_inferenceManager = std::make_unique<AIInferenceManager>(this);
    m_recordingManager = std::make_unique<RecordingManager>(this);

    // 设置信号连接
    setupConnections();

    PLOGI << "DetectionCoordinator: Created for worker " << workerId;
}

DetectionCoordinator::~DetectionCoordinator() {
    stop();
    PLOGI << "DetectionCoordinator: Destroyed for worker " << m_workerId;
}

void DetectionCoordinator::start() {
    if (m_isRunning.load()) {
        PLOGW << "DetectionCoordinator: Already running";
        return;
    }

    PLOGI << "DetectionCoordinator: Starting system...";

    // 启动视频采集
    m_captureManager->startCapture();

    // 启动AI推理（如果有模型）
    if (m_inferenceManager->getCurrentModelType() == "NONE") {
        PLOGW << "DetectionCoordinator: No model loaded, inference will be disabled";
    } else {
        m_inferenceManager->startInference();
    }

    m_isRunning.store(true);
    PLOGI << "DetectionCoordinator: System started";
}

void DetectionCoordinator::stop() {
    if (!m_isRunning.load()) {
        return;
    }

    PLOGI << "DetectionCoordinator: Stopping system...";

    if (m_isPaused.load()) {
        setPaused(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    m_recordingManager->stopAllRecording();
    m_inferenceManager->stopInference();
    m_captureManager->stopCapture();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    m_isRunning.store(false);
    PLOGI << "DetectionCoordinator: System stopped";
}

void DetectionCoordinator::onRecordingStarted() {
    m_recordingActive.store(true);
    PLOGI << "DetectionCoordinator: Recording started, frame distribution optimized";
}

void DetectionCoordinator::onRecordingStopped() {
    m_recordingActive.store(false);
    PLOGI << "DetectionCoordinator: Recording stopped, frame distribution optimized";
}

void DetectionCoordinator::setPaused(bool paused) {
    if (!m_isRunning.load()) {
        PLOGW << "DetectionCoordinator: Cannot pause/resume - system not running";
        return;
    }
    m_isPaused.store(paused);

    // 暂停所有组件
    m_captureManager->setPaused(paused);
    m_inferenceManager->setPaused(paused);

    PLOGI << "DetectionCoordinator: System " << (paused ? "paused" : "resumed");
}

bool DetectionCoordinator::switchModel(const std::string& type, const std::string& path) {
    bool success = m_inferenceManager->switchModel(type, path);

    if (success && m_isRunning.load()) {
        // 如果系统正在运行，重新启动推理
        m_inferenceManager->stopInference();
        m_inferenceManager->startInference();
    }

    return success;
}

bool DetectionCoordinator::startRecording(const std::string& basePath) {
    std::string path = basePath.empty() ? "drowning_detection" : basePath;
    return m_recordingManager->startDualRecording(path);
}

void DetectionCoordinator::stopRecording() {
    m_recordingManager->stopAllRecording();
}

void DetectionCoordinator::triggerSnapshot() {
    // 触发截图（采集和推理都会处理）
    m_captureManager->triggerSnapshot();
    m_inferenceManager->triggerSnapshot();
}

bool DetectionCoordinator::isRunning() const {
    return m_isRunning.load();
}

bool DetectionCoordinator::isPaused() const {
    return m_isPaused.load();
}

bool DetectionCoordinator::isRecording() const {
    return m_recordingManager->isOriginalRecording() ||
           m_recordingManager->isInferenceRecording();
}

std::string DetectionCoordinator::getCurrentModel() const {
    return m_inferenceManager->getCurrentModelType();
}

void DetectionCoordinator::setupConnections() {
    // 连接视频采集信号
    connect(m_captureManager.get(), &VideoCaptureManager::frameReady,
            this, &DetectionCoordinator::onFrameReady);
    connect(m_captureManager.get(), &VideoCaptureManager::snapshotReady,
            this, [this](const cv::Mat& frame) {
                emit snapshotReady(frame, cv::Mat(), m_workerId);
            });
    connect(m_captureManager.get(), &VideoCaptureManager::captureError,
            this, &DetectionCoordinator::onCaptureError);

    // 连接AI推理信号
    connect(m_inferenceManager.get(), &AIInferenceManager::inferenceFrameReady,
            this, &DetectionCoordinator::onInferenceFrameReady);
    connect(m_inferenceManager.get(), &AIInferenceManager::snapshotReady,
            this, &DetectionCoordinator::onSnapshotReady);
    connect(m_inferenceManager.get(), &AIInferenceManager::inferenceError,
            this, &DetectionCoordinator::onInferenceError);
    connect(m_inferenceManager.get(), &AIInferenceManager::modelSwitched,
            this, &DetectionCoordinator::modelSwitched);

    // 连接录制信号
    connect(m_recordingManager.get(), &RecordingManager::recordingStarted,
            this, &DetectionCoordinator::recordingStarted);
    connect(m_recordingManager.get(), &RecordingManager::recordingStarted,
            this, &DetectionCoordinator::onRecordingStarted);
    connect(m_recordingManager.get(), &RecordingManager::recordingStopped,
            this, &DetectionCoordinator::recordingStopped);
    connect(m_recordingManager.get(), &RecordingManager::recordingStopped,
            this, &DetectionCoordinator::onRecordingStopped);
    connect(m_recordingManager.get(), &RecordingManager::recordingError,
            this, &DetectionCoordinator::systemError);
}

void DetectionCoordinator::initializeStorage() {
    // 创建必要的目录
    std::vector<std::string> directories = {"snapshots", "records"};

    for (const auto& dir : directories) {
#if __has_include(<filesystem>) && __cplusplus >= 201703L
        // 使用C++17 filesystem
        namespace fs = std::filesystem;
        try {
            if (!fs::exists(dir)) {
                fs::create_directory(dir);
                PLOGI << "DetectionCoordinator: Created directory: " << dir;
            }
        } catch (const std::exception& e) {
            PLOGE << "DetectionCoordinator: Failed to create directory " << dir << ": " << e.what();
        }
#else
        // 使用系统调用
        struct stat info;
        if (stat(dir.c_str(), &info) != 0) {
            // 目录不存在，创建它
            if (mkdir(dir.c_str(), 0777) == 0) {
                PLOGI << "DetectionCoordinator: Created directory: " << dir;
            } else {
                PLOGE << "DetectionCoordinator: Failed to create directory " << dir;
            }
        }
#endif
    }
}

void DetectionCoordinator::onFrameReady(const cv::Mat& frame) {
    emit frameReady(frame);
    if (m_recordingActive.load()) {
        m_recordingManager->submitOriginalFrame(frame);
    }
    if (m_inferenceManager->isRunning()) {
        m_inferenceManager->submitFrame(frame);
    }
}

void DetectionCoordinator::onInferenceFrameReady(const cv::Mat& frame) {
    // 录制推理帧
    m_recordingManager->submitInferenceFrame(frame);

    // 发送信号给UI
    emit inferenceFrameReady(frame);
}

void DetectionCoordinator::onSnapshotReady(const cv::Mat& raw, const cv::Mat& infer) {
    emit snapshotReady(raw, infer, m_workerId);
}

void DetectionCoordinator::onCaptureError(const QString& error) {
    PLOGE << "DetectionCoordinator: Capture error: " << error.toStdString();
    emit systemError("Capture Error: " + error);
}

void DetectionCoordinator::onInferenceError(const QString& error) {
    PLOGE << "DetectionCoordinator: Inference error: " << error.toStdString();
    emit systemError("Inference Error: " + error);
}