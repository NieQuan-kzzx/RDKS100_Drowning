#include "DetectionWorker.h"

#include "Yolo11Infer.h"
#include "YoloPose.h"
#include "Patchcore.h"
#include <plog/Log.h>
#include <algorithm> 

#include "DrowningUnderSurface.h"
#include "DrowningState.h"

namespace fs = std::filesystem;

// 内存池键名常量定义
const std::string DetectionWorker::FRAME_CLONE_KEY = "frame_clone";
const std::string DetectionWorker::RESIZE_1280x720_KEY = "resize_1280x720";
const std::string DetectionWorker::SHARED_FRAME_KEY = "shared_frame";

DetectionWorker::DetectionWorker(RTSPCamera* cam,  int id, QObject* parent)
    : QObject(parent), m_cam(cam), m_id(id), m_running(false), m_is_infer_running(false),
      m_isPaused(false), m_needSnapshot(false), m_isRecording(false),
      m_is_processing(false), m_isInferRecording(false),
      m_matPool(MatPoolManager::getInstance()) {
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
        nextLogic = std::make_unique<DrowningUnderSurface>();
    } 
    else if (type == "SWIMMER"){
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels({"drowning", "swimming"});
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningState>();
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

        auto shareFrame = std::make_shared<cv::Mat>(frame);
        emit frameReady(*shareFrame);

        if (!m_is_infer_running.load() && m_needSnapshot.load()) {
            emit snapshotReady(frame.clone(), cv::Mat(), m_id);
            m_needSnapshot.store(false);
        }

        if (m_is_infer_running.load() && m_inferQueue.size() < 2) {
            m_inferQueue.enqueue(frame.clone());
        }

        if (m_isRecording.load()) {
            if (m_recordQueue.size() < 30) {
                cv::Mat smallFrame;
                cv::resize(frame, smallFrame, cv::Size(1280, 720));
                m_recordQueue.enqueue(smallFrame);
            }
        }

        // emit frameReady(frame.clone());

        // if (!m_is_infer_running.load() && m_needSnapshot.load()) {
        //     emit snapshotReady(frame.clone(), cv::Mat(), m_id);
        //     m_needSnapshot.store(false); 
        // }

        // if (m_is_infer_running.load() && m_inferQueue.size() < 2) {
        //     m_inferQueue.enqueue(frame.clone());
        // }

        // if (m_isRecording.load()) {
        //     if (m_recordQueue.size() < 30) m_recordQueue.enqueue(frame.clone());
        // }
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
            emit snapshotReady(oriFrame, frame.clone(), m_id);
            m_needSnapshot.store(false);
        }

        if (m_isInferRecording.load()) {
            if (m_inferRecordQueue.size() < 30) {
                cv::Mat smallInferFrame;
                cv::resize(frame, smallInferFrame, cv::Size(1280, 720));
                m_inferRecordQueue.enqueue(smallInferFrame);
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
        //if (m_inferVideoWriter.isOpened()) m_inferVideoWriter.release();
        m_InferpendingRecordPath = path;
        // m_inferVideoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        // if (m_inferVideoWriter.isOpened()) {
        m_isInferRecording.store(true);
        if (m_inferRecordThread.joinable()) m_inferRecordThread.detach(); 
        m_inferRecordThread = std::thread(&DetectionWorker::inferRecordLoop, this);
    } else {
        m_isInferRecording.store(false);
    }
}

void DetectionWorker::setOriRecording(bool start, const std::string& path) {
    std::lock_guard<std::mutex> lock(m_writerMtx);
    if (start) {
        // if (m_videoWriter.isOpened()) m_videoWriter.release();
        // m_videoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        m_OripendingRecordPath = path;
        // if (m_videoWriter.isOpened()) {
        m_isRecording.store(true);
        if (m_recordThread.joinable()) m_recordThread.detach();
        m_recordThread = std::thread(&DetectionWorker::recordLoop, this);
        // }
    } else {
        m_isRecording.store(false); 
    }
}

void DetectionWorker::recordLoop() {
    PLOGI << "Raw Record Thread Started.";
    while (m_isRecording.load() || !m_recordQueue.empty()) {
        cv::Mat f = m_recordQueue.dequeue();
        
        // 1. 检查空帧（dequeue 超时或停止信号）
        if (f.empty()) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); 
            continue; 
        }

        // 2. 检查并开启 VideoWriter（仅在第一次进入或未打开时加锁初始化）
        if (!m_videoWriter.isOpened()) {
            std::lock_guard<std::mutex> lock(m_writerMtx);
            // 建议：将 MJPG 换成 XVID (mp4v) 通常对磁盘压力更小
            // 帧率建议设为 20，如果你的摄像头是 25fps 但处理有延迟，设为 20 会更流畅
            m_videoWriter.open(m_OripendingRecordPath, 
                             cv::VideoWriter::fourcc('M','J','P','G'), 
                             20, 
                             f.size()); 
            
            if (!m_videoWriter.isOpened()) {
                PLOGE << "Failed to open Raw VideoWriter!";
                m_isRecording.store(false); // 开启失败则停止录制标志
                continue;
            }
            PLOGI << "Raw VideoWriter opened: " << f.cols << "x" << f.rows;
        }

        // 3. 写入操作（注意：write 本身是耗时操作，不应放在全局互斥锁里阻塞其他功能）
        // 只要保证 m_videoWriter 对象不被 release 即可
        m_videoWriter.write(f);

        // 4. 显式释放内存，防止队列积压导致内存溢出
        // Check if this Mat came from our pool and return it
        // Note: This is a simplified approach - in practice, we'd need a way to track pool Mats
        f.release();
    }

    // 录制结束，释放资源
    std::lock_guard<std::mutex> lock(m_writerMtx);
    if (m_videoWriter.isOpened()) {
        m_videoWriter.release();
        PLOGI << "Raw Record Thread Stopped Safely.";
    }
}

void DetectionWorker::inferRecordLoop() {
    PLOGI << "Infer Record Thread Started.";
    while (m_isInferRecording.load() || !m_inferRecordQueue.empty()) {
        cv::Mat f = m_inferRecordQueue.dequeue();
        
        if (f.empty()) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); 
            continue; 
        }

        // 1. 初始化推理路写入器
        if (!m_inferVideoWriter.isOpened()) {
            std::lock_guard<std::mutex> lock(m_inferWriterMtx);
            m_inferVideoWriter.open(m_InferpendingRecordPath, 
                                    cv::VideoWriter::fourcc('M','J','P','G'), 
                                    20, // 推理路帧率通常不稳定，建议设低一点（如20）防止视频变快进
                                    f.size());
            
            if (!m_inferVideoWriter.isOpened()) {
                PLOGE << "Failed to open Infer VideoWriter!";
                m_isInferRecording.store(false);
                continue;
            }
            PLOGI << "Infer VideoWriter opened: " << f.cols << "x" << f.rows;
        }

        // 2. 写入包含 AI 结果的帧
        m_inferVideoWriter.write(f);

        // 3. 释放资源
        f.release();
    }

    // 安全关闭
    std::lock_guard<std::mutex> lock(m_inferWriterMtx);
    if (m_inferVideoWriter.isOpened()) {
        m_inferVideoWriter.release();
        PLOGI << "Infer Record Thread Stopped Safely.";
    }
}

void DetectionWorker::setPaused(bool p) { m_isPaused.store(p); }
void DetectionWorker::triggerSnapshot() { m_needSnapshot.store(true); }

void DetectionWorker::printPerformanceReport() {
    m_matPool.printStats();
}