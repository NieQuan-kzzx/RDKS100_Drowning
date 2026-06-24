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
    std::lock_guard<std::mutex> lock(m_originalWriterMutex);

    if (m_isOriginalRecording.load()) {
        PLOGW << "RecordingManager: Original recording already in progress";
        return false;
    }

    m_originalRecordPath = path;
    m_isOriginalRecording.store(true);

    if (m_originalRecordThread.joinable()) {
        m_originalRecordThread.detach();
    }
    m_originalRecordThread = std::thread(&RecordingManager::originalRecordLoop, this);

    PLOGI << "RecordingManager: Original recording started: " << path;
    return true;
}

void RecordingManager::stopOriginalRecording() {
    m_isOriginalRecording.store(false);
    m_originalQueue.clear();
    m_originalQueue.enqueue(cv::Mat()); // 停止信号
}

bool RecordingManager::startInferenceRecording(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_inferenceWriterMutex);

    if (m_isInferenceRecording.load()) {
        PLOGW << "RecordingManager: Inference recording already in progress";
        return false;
    }

    m_inferenceRecordPath = path;
    m_isInferenceRecording.store(true);

    if (m_inferenceRecordThread.joinable()) {
        m_inferenceRecordThread.detach();
    }
    m_inferenceRecordThread = std::thread(&RecordingManager::inferenceRecordLoop, this);

    PLOGI << "RecordingManager: Inference recording started: " << path;
    return true;
}

void RecordingManager::stopInferenceRecording() {
    m_isInferenceRecording.store(false);
    m_inferenceQueue.clear();
    m_inferenceQueue.enqueue(cv::Mat()); // 停止信号
}

bool RecordingManager::startDualRecording(const std::string& basePath) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");

    std::string originalPath = generateRecordingPath(basePath, "_original_" + ss.str() + ".avi");
    std::string inferencePath = generateRecordingPath(basePath, "_inference_" + ss.str() + ".avi");

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

// 采集前N帧测出真实帧率，再用实际帧率初始化VideoWriter
static double measure_actual_fps(
    ThreadSafeQueue<cv::Mat>& queue,
    const std::atomic<bool>& is_recording,
    std::vector<cv::Mat>& buffer,
    int skip_count,
    int min_samples,
    int max_wait_ms)
{
    auto start = std::chrono::steady_clock::now();
    auto last = start;

    int skipped = 0;
    while ((int)buffer.size() < min_samples) {
        if (!is_recording.load() && queue.empty()) break;
        cv::Mat f = queue.dequeue();
        if (f.empty()) continue;
        auto now = std::chrono::steady_clock::now();
        if (skipped < skip_count) {
            ++skipped;
            continue;
        }
        if (buffer.empty()) {
            start = now;
            last = now;
        }
        buffer.push_back(f);
        last = now;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() >= max_wait_ms)
            break;
    }

    if (buffer.size() < 2) return 15.0;

    double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - start).count();
    double fps = (buffer.size() - 1) / elapsed_s;

    if (fps < 5.0)  fps = 5.0;
    if (fps > 30.0) fps = 30.0;

    PLOGI << "RecordingManager: Measured actual FPS = " << fps
          << " (samples=" << buffer.size() << ", elapsed=" << elapsed_s << "s)";
    return fps;
}

void RecordingManager::originalRecordLoop() {
    PLOGI << "RecordingManager: Original record thread started";

    std::vector<cv::Mat> frameBuffer;
    double actualFps = 15.0;

    while (m_isOriginalRecording.load() || !m_originalQueue.empty()) {

        // ---- 初始化阶段：先采集帧测FPS，再打开VideoWriter ----
        if (!m_originalVideoWriter.isOpened()) {
            if (frameBuffer.empty()) {
                PLOGI << "RecordingManager: Measuring actual frame rate for original recording...";
                actualFps = measure_actual_fps(m_originalQueue, m_isOriginalRecording,
                                               frameBuffer, 10, 30, 3500);
            }

            // 不再等待更多帧，直接用测到的FPS打开Writer
            std::lock_guard<std::mutex> lock(m_originalWriterMutex);
            double fps = m_highPerformanceMode.load()
                ? std::min(actualFps, 20.0)
                : actualFps;
            m_originalVideoWriter.open(m_originalRecordPath,
                                     cv::VideoWriter::fourcc('M','J','P','G'),
                                     fps, frameBuffer.front().size());

            if (!m_originalVideoWriter.isOpened()) {
                PLOGE << "RecordingManager: Failed to open original video writer";
                emit recordingError("Failed to start original recording");
                m_isOriginalRecording.store(false);
                continue;
            }
            PLOGI << "RecordingManager: Original VideoWriter opened: "
                  << frameBuffer.front().cols << "x" << frameBuffer.front().rows
                  << " @ " << fps << " FPS (actual)";

            // 写入缓存的帧
            for (auto& f : frameBuffer) {
                m_originalVideoWriter.write(f);
                {
                    std::lock_guard<std::mutex> lock(m_infoMutex);
                    m_recordingInfo.originalFrameCount++;
                }
                f.release();
            }
            frameBuffer.clear();
            continue;
        }

        // ---- 正常写入循环 ----
        cv::Mat frame = m_originalQueue.dequeue();
        if (frame.empty()) continue;

        m_originalVideoWriter.write(frame);

        {
            std::lock_guard<std::mutex> lock(m_infoMutex);
            m_recordingInfo.originalFrameCount++;
        }

        emit frameRecorded(true, m_recordingInfo.originalFrameCount);
        frame.release();
    }

    frameBuffer.clear();

    // 安全关闭
    std::lock_guard<std::mutex> lock(m_originalWriterMutex);
    if (m_originalVideoWriter.isOpened()) {
        m_originalVideoWriter.release();
        PLOGI << "RecordingManager: Original record thread stopped safely";
    }
}

void RecordingManager::inferenceRecordLoop() {
    PLOGI << "RecordingManager: Inference record thread started";

    std::vector<cv::Mat> frameBuffer;
    double actualFps = 15.0;

    while (m_isInferenceRecording.load() || !m_inferenceQueue.empty()) {

        // ---- 初始化阶段：先采集帧测FPS，再打开VideoWriter ----
        if (!m_inferenceVideoWriter.isOpened()) {
            if (frameBuffer.empty()) {
                PLOGI << "RecordingManager: Measuring actual frame rate for inference recording...";
                actualFps = measure_actual_fps(m_inferenceQueue, m_isInferenceRecording,
                                               frameBuffer, 10, 30, 3500);
            }

            std::lock_guard<std::mutex> lock(m_inferenceWriterMutex);
            double fps = m_highPerformanceMode.load()
                ? std::min(actualFps, 20.0)
                : actualFps;
            m_inferenceVideoWriter.open(m_inferenceRecordPath,
                                      cv::VideoWriter::fourcc('M','J','P','G'),
                                      fps, frameBuffer.front().size());

            if (!m_inferenceVideoWriter.isOpened()) {
                PLOGE << "RecordingManager: Failed to open inference video writer";
                emit recordingError("Failed to start inference recording");
                m_isInferenceRecording.store(false);
                continue;
            }
            PLOGI << "RecordingManager: Inference VideoWriter opened: "
                  << frameBuffer.front().cols << "x" << frameBuffer.front().rows
                  << " @ " << fps << " FPS (actual)";

            // 写入缓存的帧
            for (auto& f : frameBuffer) {
                m_inferenceVideoWriter.write(f);
                {
                    std::lock_guard<std::mutex> lock(m_infoMutex);
                    m_recordingInfo.inferenceFrameCount++;
                }
                f.release();
            }
            frameBuffer.clear();
            continue;
        }

        // ---- 正常写入循环 ----
        cv::Mat frame = m_inferenceQueue.dequeue();
        if (frame.empty()) continue;

        m_inferenceVideoWriter.write(frame);

        {
            std::lock_guard<std::mutex> lock(m_infoMutex);
            m_recordingInfo.inferenceFrameCount++;
        }

        emit frameRecorded(false, m_recordingInfo.inferenceFrameCount);
        frame.release();
    }

    frameBuffer.clear();

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
    // === 修改部分：将录像目录指向 U 盘 ===
    // 注意：路径包含空格，必须作为完整字符串赋值
    std::string recordsDir = "/media/UBUNTU 18_0/records"; 
    // ======================================

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
