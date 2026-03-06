#include "DetectionWorker.h"
#include "LogicBase.h"
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
    std::unique_ptr<LogicBase> nextLogic;

    if (type == "YOLO") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        // 设置标签顺序：0-水面, 1-水下 (对应 InferenceLogic 中的判定)
        yolo->setLabels({"person at surface", "person underwater"});
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningLogic>();
    } 
    else if (type == "SWIMMER"){
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels({"swimmer"});
        nextEngine = std::move(yolo);
    }
    else if (type == "Patchcore"){
        nextEngine = std::make_unique<Inf::Patchcore>();
    }

    if (nextEngine && nextEngine->init(path)) {
            m_inferEngine = std::move(nextEngine);
            m_currentLogic = std::move(nextLogic);
            PLOGI << "Model and Logic switched successfully: " << type;
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

    while (m_is_infer_running.load()) {
        if (m_isPaused.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        cv::Mat frame = m_inferQueue.dequeue();
        if (frame.empty()) continue;

        cv::Mat oriFrame = frame.clone();
        std::vector<Inf::Detection> results;

        {
            std::lock_guard<std::mutex> lock(m_inferMtx);
            if (m_inferEngine) {
                // 1. 执行 AI 推理
                results = m_inferEngine->run(frame);

                // 2. 调用引擎的基础绘图 (画常规框)
                m_inferEngine->draw(frame, results);

                // 3. 调用逻辑插件处理高级业务 (溺水判定、红框、全局报警)
                if (m_currentLogic) {
                    m_currentLogic->process(frame, results);
                }
            }
        }

        // 处理截图和信号发送
        if (m_needSnapshot.load()) {
            std::string timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz").toStdString();
            cv::imwrite("snapshots/Raw_" + timeStr + ".jpg", oriFrame); 
            cv::imwrite("snapshots/Infer_" + timeStr + ".jpg", frame);  
            m_needSnapshot.store(false);
        }

        if (m_isInferRecording.load()) {
            if (m_inferRecordQueue.size() < 30) m_inferRecordQueue.enqueue(frame.clone());
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