#include "RecordingManager.h"
#include <plog/Log.h>
#include <chrono>
#include <iomanip>
#include <sstream>

// 条件编译：检查C++17 filesystem支持
#if __has_include(<filesystem>) && __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
// 回退到experimental/filesystem或系统相关API
#include <sys/stat.h>
#include <sys/types.h>
#endif

RecordingManager::RecordingManager(QObject* parent)
    : QObject(parent) {
}

void RecordingManager::setRecordingConfig(const RecordingConfig& config) {
    m_recordingConfig = config;
    PLOGI << "RecordingManager: Config updated - FPS: " << config.fps 
          << ", Codec: " << config.codec 
          << ", Queue size: " << config.queue_size;
}

int RecordingManager::getFourCC() const {
    if (m_recordingConfig.codec == "XVID") {
        return cv::VideoWriter::fourcc('X','V','I','D');
    } else if (m_recordingConfig.codec == "H264") {
        return cv::VideoWriter::fourcc('H','2','6','4');
    } else if (m_recordingConfig.codec == "MP4V") {
        return cv::VideoWriter::fourcc('M','P','4','V');
    } else if (m_recordingConfig.codec == "IYUV") {
        return cv::VideoWriter::fourcc('I','Y','U','V');
    }
    return cv::VideoWriter::fourcc('M','J','P','G');
}

std::string RecordingManager::getFileExtension() const {
    if (m_recordingConfig.codec == "MP4V") return ".mp4";
    if (m_recordingConfig.codec == "H264") return ".h264";
    return ".avi";
}

RecordingManager::~RecordingManager() {
    stopAllRecording();
    if (m_originalRecordThread.joinable()) {
        m_originalRecordThread.join();
    }
    if (m_inferenceRecordThread.joinable()) {
        m_inferenceRecordThread.join();
    }
}

bool RecordingManager::startOriginalRecording(const std::string& path) {
    // 如果正在录制，先停止旧线程
    if (m_isOriginalRecording.load()) {
        stopOriginalRecording();
    }

    // 异步等待旧线程退出，避免阻塞调用线程（如UI线程）
    if (m_originalRecordThread.joinable()) {
        auto oldThread = std::move(m_originalRecordThread);
        std::thread([thread = std::move(oldThread)]() mutable {
            thread.join();
        }).detach();
    }

    m_originalRecordPath = path;
    m_isOriginalRecording.store(true);
    m_originalRecordThread = std::thread(&RecordingManager::originalRecordLoop, this);

    PLOGI << "RecordingManager: Original recording started: " << path;
    return true;
}

void RecordingManager::stopOriginalRecording() {
    m_isOriginalRecording.store(false);
    // 入队空帧（毒丸）唤醒阻塞在 dequeue_timeout 的线程
    // 不 join()，不阻塞调用线程；线程会在消费毒丸后自行退出
    // join() 由 startOriginalRecording() 和析构函数负责
    m_originalQueue.enqueue(cv::Mat());
}

bool RecordingManager::startInferenceRecording(const std::string& path) {
    // 如果正在录制，先停止旧线程
    if (m_isInferenceRecording.load()) {
        stopInferenceRecording();
    }

    // 异步等待旧线程退出，避免阻塞调用线程（如UI线程）
    if (m_inferenceRecordThread.joinable()) {
        auto oldThread = std::move(m_inferenceRecordThread);
        std::thread([thread = std::move(oldThread)]() mutable {
            thread.join();
        }).detach();
    }

    m_inferenceRecordPath = path;
    m_isInferenceRecording.store(true);
    m_inferenceRecordThread = std::thread(&RecordingManager::inferenceRecordLoop, this);

    PLOGI << "RecordingManager: Inference recording started: " << path;
    return true;
}

void RecordingManager::stopInferenceRecording() {
    m_isInferenceRecording.store(false);
    // 入队毒丸唤醒线程，不 join()，不阻塞调用线程
    m_inferenceQueue.enqueue(cv::Mat());
}

bool RecordingManager::startDualRecording(const std::string& basePath) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");

    std::string ext = getFileExtension();
    std::string originalPath = generateRecordingPath(basePath, "_original_" + ss.str() + ext);
    std::string inferencePath = generateRecordingPath(basePath, "_inference_" + ss.str() + ext);

    bool originalSuccess = startOriginalRecording(originalPath);
    bool inferenceSuccess = startInferenceRecording(inferencePath);

    if (originalSuccess && inferenceSuccess) {
        std::lock_guard<std::mutex> lock(m_infoMutex);
        m_recordingInfo.originalPath = originalPath;
        m_recordingInfo.inferencePath = inferencePath;
        m_recordingInfo.startTime = now;
        m_recordingInfo.originalFrameCount = 0;
        m_recordingInfo.inferenceFrameCount = 0;

        emit recordingStarted(QString::fromStdString(originalPath),
                             QString::fromStdString(inferencePath));
        return true;
    }

    // 如果失败，停止已开始的录制
    if (originalSuccess) stopOriginalRecording();
    if (inferenceSuccess) stopInferenceRecording();

    return false;
}

void RecordingManager::stopAllRecording() {
    stopOriginalRecording();
    stopInferenceRecording();

    emit recordingStopped();

    // 重置录制信息
    std::lock_guard<std::mutex> lock(m_infoMutex);
    m_recordingInfo = RecordingInfo();
}

void RecordingManager::submitOriginalFrame(const cv::Mat& frame) {
    if (!m_isOriginalRecording.load()) return;
    m_originalQueue.enqueue(frame.clone());
}

void RecordingManager::submitInferenceFrame(const cv::Mat& frame) {
    if (!m_isInferenceRecording.load()) return;
    m_inferenceQueue.enqueue(frame.clone());
}

void RecordingManager::setRecordingPerformanceMode(bool highPerformance) {
    m_highPerformanceMode.store(highPerformance);
    PLOGI << "RecordingManager: Performance mode set to " << (highPerformance ? "High Performance" : "+High Quality");
}

RecordingManager::RecordingInfo RecordingManager::getRecordingInfo() const {
    std::lock_guard<std::mutex> lock(m_infoMutex);
    return m_recordingInfo;
}



void RecordingManager::originalRecordLoop() {
    PLOGI << "RecordingManager: Original record thread started";

    while (m_isOriginalRecording.load() || !m_originalQueue.empty()) {

        // ---- 初始化阶段：等待第一帧后打开VideoWriter ----
        if (!m_originalVideoWriter.isOpened()) {
            cv::Mat firstFrame = m_originalQueue.dequeue_timeout(500);
            if (firstFrame.empty()) {
                if (!m_isOriginalRecording.load()) break;
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(m_originalWriterMutex);
                int fourcc = getFourCC();

                m_originalVideoWriter.open(m_originalRecordPath,
                                         fourcc,
                                         m_recordingConfig.fps, firstFrame.size());

                if (!m_originalVideoWriter.isOpened()) {
                    PLOGE << "RecordingManager: Failed to open original video writer";
                    emit recordingError("Failed to start original recording");
                    m_isOriginalRecording.store(false);
                    continue;
                }
                PLOGI << "RecordingManager: Original VideoWriter opened: "
                      << firstFrame.cols << "x" << firstFrame.rows
                      << " @ " << m_recordingConfig.fps << " FPS";

                m_originalVideoWriter.write(firstFrame);
                {
                    std::lock_guard<std::mutex> infoLock(m_infoMutex);
                    m_recordingInfo.originalFrameCount++;
                }
            }
            continue;
        }

        // ---- 正常写入循环 ----
        cv::Mat frame = m_originalQueue.dequeue_timeout(500);
        if (frame.empty()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(m_originalWriterMutex);
            m_originalVideoWriter.write(frame);
        }

        {
            std::lock_guard<std::mutex> lock(m_infoMutex);
            m_recordingInfo.originalFrameCount++;
        }

        emit frameRecorded(true, m_recordingInfo.originalFrameCount);
    }

    // 安全关闭
    std::lock_guard<std::mutex> lock(m_originalWriterMutex);
    if (m_originalVideoWriter.isOpened()) {
        m_originalVideoWriter.release();
        PLOGI << "RecordingManager: Original record thread stopped safely";
    }
}

void RecordingManager::inferenceRecordLoop() {
    PLOGI << "RecordingManager: Inference record thread started";

    while (m_isInferenceRecording.load() || !m_inferenceQueue.empty()) {

        // ---- 初始化阶段：等待第一帧后打开VideoWriter ----
        if (!m_inferenceVideoWriter.isOpened()) {
            cv::Mat firstFrame = m_inferenceQueue.dequeue_timeout(500);
            if (firstFrame.empty()) {
                if (!m_isInferenceRecording.load()) break;
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(m_inferenceWriterMutex);
                int fourcc = getFourCC();

                m_inferenceVideoWriter.open(m_inferenceRecordPath,
                                          fourcc,
                                          m_recordingConfig.fps, firstFrame.size());

                if (!m_inferenceVideoWriter.isOpened()) {
                    PLOGE << "RecordingManager: Failed to open inference video writer";
                    emit recordingError("Failed to start inference recording");
                    m_isInferenceRecording.store(false);
                    continue;
                }
                PLOGI << "RecordingManager: Inference VideoWriter opened: "
                      << firstFrame.cols << "x" << firstFrame.rows
                      << " @ " << m_recordingConfig.fps << " FPS";

                m_inferenceVideoWriter.write(firstFrame);
                {
                    std::lock_guard<std::mutex> infoLock(m_infoMutex);
                    m_recordingInfo.inferenceFrameCount++;
                }
            }
            continue;
        }

        // ---- 正常写入循环 ----
        cv::Mat frame = m_inferenceQueue.dequeue_timeout(500);
        if (frame.empty()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(m_inferenceWriterMutex);
            m_inferenceVideoWriter.write(frame);
        }

        {
            std::lock_guard<std::mutex> lock(m_infoMutex);
            m_recordingInfo.inferenceFrameCount++;
        }

        emit frameRecorded(false, m_recordingInfo.inferenceFrameCount);
    }

    // 安全关闭
    std::lock_guard<std::mutex> lock(m_inferenceWriterMutex);
    if (m_inferenceVideoWriter.isOpened()) {
        m_inferenceVideoWriter.release();
        PLOGI << "RecordingManager: Inference record thread stopped safely";
    }
}

// std::string RecordingManager::generateRecordingPath(const std::string& basePath, const std::string& suffix) {
//     // 确保records目录存在
//     std::string recordsDir = "records";

// #if __has_include(<filesystem>) && __cplusplus >= 201703L
//     // 使用C++17 filesystem
//     namespace fs = std::filesystem;
//     try {
//         if (!fs::exists(recordsDir)) {
//             fs::create_directory(recordsDir);
//         }
//     } catch (const std::exception& e) {
//         PLOGE << "RecordingManager: Failed to create records directory: " << e.what();
//     }
// #else
//     // 使用系统调用
//     struct stat info;
//     if (stat(recordsDir.c_str(), &info) != 0) {
//         // 目录不存在，创建它
//         if (mkdir(recordsDir.c_str(), 0777) != 0) {
//             PLOGE << "RecordingManager: Failed to create records directory";
//         }
//     }
// #endif

//     // 生成完整路径
//     return recordsDir + "/" + basePath + suffix;
// }


std::string RecordingManager::generateRecordingPath(const std::string& basePath, const std::string& suffix) {
    // === 将录像目录指向 U 盘 ===
    std::string recordsDir = "/media/76E8-CACF/records"; 
    // ===========================

#if __has_include(<filesystem>) && __cplusplus >= 201703L
    // 使用C++17 filesystem
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(recordsDir)) {
            fs::create_directory(recordsDir);
        }
    } catch (const std::exception& e) {
        PLOGE << "RecordingManager: Failed to create records directory: " << e.what();
    }
#else
    // 使用系统调用
    struct stat info;
    if (stat(recordsDir.c_str(), &info) != 0) {
        // 目录不存在，创建它
        if (mkdir(recordsDir.c_str(), 0777) != 0) {
            PLOGE << "RecordingManager: Failed to create records directory";
        }
    }
#endif

    // 生成完整路径
    return recordsDir + "/" + basePath + suffix;
}
