#include "DetectionWorker.h"
#include "Yolo11Infer.h"
#include "YoloPose.h"
#include "Patchcore.h"
#include <plog/Log.h>

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
    // 创建截图目录
    if (!fs::exists("snapshots")){
        fs::create_directory("snapshots");
        PLOGI << "Create directory: snapshots";
    }
    // 创建录像文件
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

    // 1. 根据类型创建实例
    if (type == "YOLO") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
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
        // 如果 Patchcore 也需要 labels，在这里 set
    }

    // 2. 统一初始化
    if (nextEngine) {
        if (nextEngine->init(path)) {
            // 只有初始化成功，才替换正在运行的引擎
            m_inferEngine = std::move(nextEngine);
            PLOGI << "Model switched successfully: " << type;
        } else {
            PLOGE << "Failed to initialize model: " << path;
        }
    }
}

// --- 线程1：抓取与源画面展示 ---
void DetectionWorker::processLoop() {
    if (m_is_processing.exchange(true)) return;
    m_running.store(true);

    while (m_running.load()) {
        if (!m_running.load()) break;

        if (m_isPaused.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!m_running.load()) break;
            continue;
        }
        if (!m_cam || !m_cam->isRunning()) break;
        cv::Mat frame = m_cam->getData(); 
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 1. 发送源画面
        emit frameReady(frame.clone());

        // 2. 兜底截图逻辑：如果推理线程没开，但用户点了截图
        // 这样可以保证即使没开启 AI，左边的截图功能依然有效
        if (!m_is_infer_running.load() && m_needSnapshot.load()) {
            if (!frame.empty()){
                std::string timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss").toStdString();
                cv::imwrite("snapshots/Raw_" + timeStr + ".jpg", frame);
                m_needSnapshot.store(false); 
                PLOGI << "Raw snapshot saved (Inference not active).";
            }
        }

        // 3. 投喂推理线程
        if (m_is_infer_running.load() && m_inferQueue.size() < 2) {
            m_inferQueue.enqueue(frame.clone());
        }

        // 4. 原始流录制 (不带框)
        if (m_isRecording.load()) {
            // 注意：这里建议直接写 VideoWriter，或者确保 recordQueue 有消费者
            if (m_recordQueue.size() < 50) m_recordQueue.enqueue(frame.clone());
        }
    }
    m_is_processing.store(false);
}

// --- 线程2：独立推理与结果展示 ---
void DetectionWorker::inferenceLoop() {
    if (m_is_infer_running.exchange(true)) return;
    PLOGI << "Inference Loop Started.";

    while (m_is_infer_running.load()) {
        if (m_isPaused.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!m_is_infer_running.load()) break;
            continue;
        }

        cv::Mat frame = m_inferQueue.dequeue();
        if (frame.empty()) {
            if (!m_is_infer_running.load()) break;
            continue;
        }

        // --- 1. 立即备份原始帧 (用于双路截图和双路录制) ---
        cv::Mat oriFrame = frame.clone();

        // --- 2. 执行 AI 推理 ---
        std::vector<Inf::Detection> results;
        {
            std::lock_guard<std::mutex> lock(m_inferMtx);
            if (m_inferEngine) {
                results = m_inferEngine->run(frame);
                m_inferEngine->draw(frame, results);
            }
        }

        // --- 3. 双路截图处理 ---
        if (m_needSnapshot.load()) {
            std::string timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz").toStdString();
            cv::imwrite("snapshots/Raw_" + timeStr + ".jpg", oriFrame); // 保存原图
            cv::imwrite("snapshots/Infer_" + timeStr + ".jpg", frame);  // 保存带框图
            m_needSnapshot.store(false);
            PLOGI << "Dual Snapshots saved.";
        }
        // --- 4. 双路录制处理 ---
        if (m_isInferRecording.load()) {
           if (m_inferRecordQueue.size() < 30) {
            m_inferRecordQueue.enqueue(frame.clone());
            }
        }
        // 发送 UI 显示
        emit inferFrameReady(frame.clone());
    }
    m_is_infer_running.store(false);
}

void DetectionWorker::setInferRecording(bool start, const std::string& path) {
    std::lock_guard<std::mutex> lock(m_inferWriterMtx);
    if (start) {
        // 如果之前有没关掉的，先释放
        if (m_inferVideoWriter.isOpened()) m_inferVideoWriter.release();

        m_inferVideoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        
        if (m_inferVideoWriter.isOpened()) {
            m_isInferRecording.store(true);
            // 启动专用的推理录制线程（逻辑同原始流录制）
            if (m_inferRecordThread.joinable()) m_inferRecordThread.detach(); 
            m_inferRecordThread = std::thread(&DetectionWorker::inferRecordLoop, this);
            PLOGI << "Infer recording thread started.";
        }
    } else {
        // 仅仅修改标志位，不在这里 release，让 inferRecordLoop 写完剩余队列后自行 release
        m_isInferRecording.store(false);
    }
}

void DetectionWorker::setRecording(bool start, const std::string& path) {
    std::lock_guard<std::mutex> lock(m_writerMtx);
    if (start) {
        if (m_videoWriter.isOpened()) m_videoWriter.release(); // 确保先关闭旧的

        m_videoWriter.open(path, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(1920, 1080));
        if (m_videoWriter.isOpened()) {
            m_isRecording.store(true);
            // 只有当线程没在跑，或者已经跑完了，才创建新线程
            if (!m_recordThread.joinable()) {
                m_recordThread = std::thread(&DetectionWorker::recordLoop, this);
            }
        }
    } else {
        m_isRecording.store(false); 
        // 线程会在 recordLoop 里发现 isRecording 为 false 且队列空了后自动退出并释放 writer
    }
}

void DetectionWorker::startDualRecording(const std::string& timeStr) {
    // 1. 开启原始路 (左路)
    std::string rawPath = "records/Raw_" + timeStr + ".avi";
    this->setRecording(true, rawPath);

    // 2. 如果推理开着，开启推理路 (右路)
    if (m_is_infer_running.load()) {
        std::string inferPath = "records/Infer_" + timeStr + ".avi";
        this->setInferRecording(true, inferPath);
        PLOGI << "Dual recording initiated.";
    }
}

void DetectionWorker::stopAllRecording() {
    this->setRecording(false);
    this->setInferRecording(false);
    PLOGI << "All recordings stopped.";
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
    PLOGI << "Infer Record Loop Enter.";
    while (m_isInferRecording.load() || !m_inferRecordQueue.empty()) {
        cv::Mat f = m_inferRecordQueue.dequeue();
        if (f.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::lock_guard<std::mutex> lock(m_inferWriterMtx);
        if (m_inferVideoWriter.isOpened()) {
            m_inferVideoWriter.write(f);
        }
    }
    
    // 队列写完且开关关闭后，安全释放资源
    std::lock_guard<std::mutex> lock(m_inferWriterMtx);
    if (m_inferVideoWriter.isOpened()) {
        m_inferVideoWriter.release();
    }
    PLOGI << "Infer Record Loop Exit and file saved.";
}

void DetectionWorker::setPaused(bool p) { m_isPaused.store(p); }
void DetectionWorker::triggerSnapshot() {
    m_needSnapshot.store(true);
    PLOGI << "Dual snapshot requested";
}