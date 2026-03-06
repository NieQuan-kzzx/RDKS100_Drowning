#include "DetectionWorker.h"
#include "Yolo11Infer.h"
#include "YoloPose.h"
#include "Patchcore.h"
#include <plog/Log.h>
#include <algorithm> 

namespace fs = std::filesystem;

DetectionWorker::DetectionWorker(RTSPCamera* cam, QObject* parent)
    : QObject(parent), m_cam(cam), m_running(false), m_is_infer_running(false),
      m_isPaused(false), m_needSnapshot(false), m_isRecording(false), 
      m_is_processing(false), m_isInferRecording(false) {
        initStorage();
      }

DetectionWorker::~DetectionWorker() {
    stop();
    if (m_recordThread.joinable()) m_recordThread.join();
}

void DetectionWorker::initStorage() {
    if (!fs::exists("snapshots")){
        fs::create_directory("snapshots");
        PLOGI << "Create directory: snapshots";
    }
    if (!fs::exists("records")){
        fs::create_directory("records");
        PLOGI << "Create directory: records";
    }
}

void DetectionWorker::stop() { 
    m_running.store(false); 
    m_is_infer_running.store(false);
    m_isRecording.store(false);
    m_isInferRecording.store(false);
    m_inferQueue.clear();
    m_inferQueue.enqueue(cv::Mat());
    m_recordQueue.enqueue(cv::Mat());
    m_inferRecordQueue.enqueue(cv::Mat());
}

void DetectionWorker::switchModel(const std::string& type, const std::string& path) {
    std::lock_guard<std::mutex> lock(m_inferMtx);
    PLOGI << "Switching model to: " << type;
    
    std::unique_ptr<Inf::BaseInfer> nextEngine;

    if (type == "YOLO") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        // 设置标签顺序：0-水面, 1-水下 (对应 InferenceLogic 中的判定)
        yolo->setLabels({"person at surface", "person underwater"});
        nextEngine = std::move(yolo);
    } 
    else if (type == "SWIMMER"){
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels({"swimmer"});
        nextEngine = std::move(yolo);
    }
    else if (type == "Patchcore"){
        nextEngine = std::make_unique<Inf::Patchcore>();
    }

    if (nextEngine) {
        if (nextEngine->init(path)) {
            m_inferEngine = std::move(nextEngine);
            PLOGI << "Model switched successfully: " << type;
        } else {
            PLOGE << "Failed to initialize model: " << path;
        }
    }
}

void DetectionWorker::processLoop() {
    if (m_is_processing.exchange(true)) return;
    m_running.store(true);

    while (m_running.load()) {
        if (m_isPaused.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        if (!m_cam || !m_cam->isRunning()) break;
        cv::Mat frame = m_cam->getData(); 
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        emit frameReady(frame.clone());

        if (!m_is_infer_running.load() && m_needSnapshot.load()) {
            std::string timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss").toStdString();
            cv::imwrite("snapshots/Raw_" + timeStr + ".jpg", frame);
            m_needSnapshot.store(false); 
        }

        if (m_is_infer_running.load() && m_inferQueue.size() < 2) {
            m_inferQueue.enqueue(frame.clone());
        }

        if (m_isRecording.load()) {
            if (m_recordQueue.size() < 50) m_recordQueue.enqueue(frame.clone());
        }
    }
    m_is_processing.store(false);
}

void DetectionWorker::inferenceLoop() {
    if (m_is_infer_running.exchange(true)) return;
    PLOGI << "Inference Loop Started.";

    const float move_threshold = 400.0f; // 静止判定位移阈值
    const int time_threshold = 10;    // 溺水时间判定帧数 (约4秒 @25fps)

    while (m_is_infer_running.load()) {
        if (m_isPaused.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        cv::Mat frame = m_inferQueue.dequeue();
        if (frame.empty()) continue;

        cv::Mat oriFrame = frame.clone();
        std::vector<Inf::Detection> results;
        std::vector<int> active_ids;
        bool global_drowning_alert = false;

        {
            std::lock_guard<std::mutex> lock(m_inferMtx);
            if (m_inferEngine) {
                // 1. 执行推理与跟踪
                results = m_inferEngine->run(frame);

                // 2. 溺水逻辑处理
                for (auto& det : results) {
                    active_ids.push_back(det.track_id);
                    
                    // 适配：使用 det.rect 计算中心点
                    cv::Point2f center(det.rect.x + det.rect.width / 2.0f, 
                                     det.rect.y + det.rect.height / 2.0f);
                    
                    auto& state = m_drowningManager[det.track_id];
                    state.history_pos.push_back(center);
                    if (state.history_pos.size() > 50) state.history_pos.pop_front();

                    // 计算近30帧的累积位移
                    float displacement = 100.0f;
                    if (state.history_pos.size() >= 30) {
                        displacement = cv::norm(state.history_pos.back() - state.history_pos.front());
                    }

                    // 判定水下状态：根据 switchModel 中的设置，class_id 1 为 underwater
                    bool is_underwater = (det.class_id == 1); 

                    // 核心逻辑：在水下且位移小于阈值
                    if (is_underwater && displacement < move_threshold) {
                        state.underwater_count++;
                    } else {
                        state.underwater_count = 0;
                        state.is_drowned = false;
                    }

                    if (state.underwater_count >= time_threshold) {
                        state.is_drowned = true;
                        global_drowning_alert = true;
                    }

                    // 3. 调用引擎原有的绘制函数 (画常规框和标签)
                    m_inferEngine->draw(frame, results);

                    // 如果检测到溺水，在 draw 之前先画出红色醒目提示
                    if (state.is_drowned) {
                        cv::rectangle(frame, det.rect, cv::Scalar(0, 0, 255), 4);
                        cv::putText(frame, "DROWNING!", cv::Point(det.rect.x + 5, det.rect.y + 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                    }
                }          
            }
        }

        // 全局报警 UI 渲染
        if (global_drowning_alert) {
            cv::Mat overlay = frame.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(frame.cols, 80), cv::Scalar(0, 0, 255), -1);
            cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);
            cv::putText(frame, "WARNING: DROWNING DETECTED!", cv::Point(frame.cols/2 - 300, 55), 
                        cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
        }

        // 清理缓存：移除已经不再出现的 ID
        for (auto it = m_drowningManager.begin(); it != m_drowningManager.end(); ) {
            if (std::find(active_ids.begin(), active_ids.end(), it->first) == active_ids.end()) {
                it = m_drowningManager.erase(it);
            } else {
                ++it;
            }
        }

        // 处理截图
        if (m_needSnapshot.load()) {
            std::string timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz").toStdString();
            cv::imwrite("snapshots/Raw_" + timeStr + ".jpg", oriFrame); 
            cv::imwrite("snapshots/Infer_" + timeStr + ".jpg", frame);  
            m_needSnapshot.store(false);
        }

        // 处理推理流录制
        if (m_isInferRecording.load()) {
           if (m_inferRecordQueue.size() < 30) {
            m_inferRecordQueue.enqueue(frame.clone());
            }
        }

        emit inferFrameReady(frame.clone());
    }
    m_is_infer_running.store(false);
}

// --- 录制相关控制 (保持原有逻辑) ---

void DetectionWorker::setInferRecording(bool start, const std::string& path) {
    std::lock_guard<std::mutex> lock(m_inferWriterMtx);
    if (start) {
        if (m_inferVideoWriter.isOpened()) m_inferVideoWriter.release();
        m_inferVideoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        if (m_inferVideoWriter.isOpened()) {
            m_isInferRecording.store(true);
            if (m_inferRecordThread.joinable()) m_inferRecordThread.detach(); 
            m_inferRecordThread = std::thread(&DetectionWorker::inferRecordLoop, this);
        }
    } else {
        m_isInferRecording.store(false);
    }
}

void DetectionWorker::setRecording(bool start, const std::string& path) {
    std::lock_guard<std::mutex> lock(m_writerMtx);
    if (start) {
        if (m_videoWriter.isOpened()) m_videoWriter.release();
        m_videoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        if (m_videoWriter.isOpened()) {
            m_isRecording.store(true);
            if (!m_recordThread.joinable()) {
                m_recordThread = std::thread(&DetectionWorker::recordLoop, this);
            }
        }
    } else {
        m_isRecording.store(false); 
    }
}

void DetectionWorker::recordLoop() {
    while (m_isRecording.load() || !m_recordQueue.empty()) {
        cv::Mat f = m_recordQueue.dequeue();
        if (f.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
        std::lock_guard<std::mutex> lock(m_writerMtx);
        if (m_videoWriter.isOpened()) m_videoWriter.write(f);
    }
    std::lock_guard<std::mutex> lock(m_writerMtx);
    if (m_videoWriter.isOpened()) m_videoWriter.release();
}

void DetectionWorker::inferRecordLoop() {
    while (m_isInferRecording.load() || !m_inferRecordQueue.empty()) {
        cv::Mat f = m_inferRecordQueue.dequeue();
        if (f.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
        std::lock_guard<std::mutex> lock(m_inferWriterMtx);
        if (m_inferVideoWriter.isOpened()) m_inferVideoWriter.write(f);
    }
    std::lock_guard<std::mutex> lock(m_inferWriterMtx);
    if (m_inferVideoWriter.isOpened()) m_inferVideoWriter.release();
}

void DetectionWorker::setPaused(bool p) { m_isPaused.store(p); }
void DetectionWorker::triggerSnapshot() { m_needSnapshot.store(true); }