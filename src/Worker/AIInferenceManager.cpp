#include "AIInferenceManager.h"
#include "Yolo11Infer.h"
#include "Patchcore.h"
#include "YoloSeg.h"
#include "DrowningUnderSurface.h"
#include "DrowningState.h"
#include "WaterIngress.h"
#include <plog/Log.h>

AIInferenceManager::AIInferenceManager(QObject* parent)
    : QObject(parent)
    , m_currentModelType("NONE")
    , m_currentModelPath("") {
}

AIInferenceManager::~AIInferenceManager() {
    // 1. 先断开所有连接，防止线程在退出过程中通过 emit 触发主线程已销毁的对象
    this->disconnect(); 
    
    // 2. 停止逻辑
    m_isRunning.store(false);
    m_inferenceQueue.clear();
    m_inferenceQueue.enqueue(cv::Mat()); // 唤醒
    
    // 3. 等待线程结束
    if (m_inferenceThread.joinable()) {
        m_inferenceThread.join();
    }
    PLOGI << "AIInferenceManager: Destroyed safely.";
}

bool AIInferenceManager::switchModel(const std::string& type, const std::string& path,
                                     const std::vector<std::string>& labels,
                                     const std::map<std::string, std::string>& params) {
    PLOGI << "AIInferenceManager: Switching model to: " << type;

    // 0. Release old engine FIRST to free BPU resources before loading a new one
    {
        std::lock_guard<std::mutex> lock(m_engineMutex);
        m_inferEngine.reset();
        m_currentLogic.reset();
    }

    // 1. Create new engine and logic (outside mutex, may block on BPU init)
    std::unique_ptr<Inf::BaseInfer> nextEngine;
    std::unique_ptr<LogicBase> nextLogic;

    if (type == "YOLO") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels(labels.empty() ? std::vector<std::string>{"person"} : labels);
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningUnderSurface>();
    }
    else if (type == "DROWNING") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels(labels.empty() ? std::vector<std::string>{"person at surface", "person underwater"} : labels);
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningState>();
    }
    else if (type == "SWIMMER") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels(labels.empty() ? std::vector<std::string>{"drowning", "swimming"} : labels);
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningState>();
    }
    else if (type == "Patchcore") {
        nextEngine = std::make_unique<Inf::Patchcore>();
    }
    else if (type == "YOLOSEG") {
        auto yolo = std::make_unique<Inf::YoloSeg>();
        yolo->setLabels(labels.empty() ? std::vector<std::string>{"water"} : labels);
        nextEngine = std::move(yolo);
    }
    else if (type == "WATER_INGRESS") {
        auto patchcore = std::make_unique<Inf::Patchcore>();
        patchcore->setLabels({"anomaly"});
        nextEngine = std::move(patchcore);

        auto water_logic = std::make_unique<WaterIngress>();
        water_logic->setSegLabels(labels.empty() ? std::vector<std::string>{"water"} : labels);

        auto it_id = params.find("water_class_id");
        water_logic->setWaterClassId(it_id != params.end() ? std::stoi(it_id->second) : 0);

        auto it_thr = params.find("patchcore_threshold");
        water_logic->setPatchcoreThreshold(it_thr != params.end() ? std::stof(it_thr->second) : 50.0f);

        auto it_seg = params.find("seg_model_path");
        if (it_seg != params.end()) {
            if (!water_logic->initSeg(it_seg->second)) {
                PLOGE << "AIInferenceManager: Failed to init seg model for WATER_INGRESS";
            }
        } else {
            PLOGW << "AIInferenceManager: No seg_model_path in params, water detection disabled";
        }

        nextLogic = std::move(water_logic);
    }
    else {
        PLOGE << "AIInferenceManager: Unknown model type: " << type;
        return false;
    }

    if (!nextEngine || !nextEngine->init(path)) {
        PLOGE << "AIInferenceManager: Failed to init model for " << type;
        emit inferenceError("Failed to switch model: " + QString::fromStdString(type));
        return false;
    }

    // 2. Atomically swap engines under mutex — inference loop picks it up next frame
    {
        std::lock_guard<std::mutex> lock(m_engineMutex);
        m_inferEngine = std::move(nextEngine);
        m_currentLogic = std::move(nextLogic);
        m_currentModelType = type;
        m_currentModelPath = path;
    }

    PLOGI << "AIInferenceManager: Model switched successfully to " << type;
    emit modelSwitched(QString::fromStdString(type));
    return true;
}

void AIInferenceManager::startInference() {
    std::lock_guard<std::mutex> lock(m_engineMutex);
    
    if (m_isRunning.load()) {
        PLOGW << "AIInferenceManager: Already running, ignoring start request";
        return;
    }
    if (!m_inferEngine) {
        PLOGE << "AIInferenceManager: Cannot start - No inference engine loaded!";
        return;
    }

    // 清除残留的空帧，防止 stopInference 入队的退出信号被新线程误消费
    m_inferenceQueue.clear();

    m_isRunning.store(true);
    m_inferenceThread = std::thread(&AIInferenceManager::inferenceLoop, this);
    PLOGI << "AIInferenceManager: Inference thread launched.";
}

void AIInferenceManager::stopInference() {
    // 1. 原子操作停止标志
    if (!m_isRunning.exchange(false)) {
        PLOGW << "AIInferenceManager: stopInference called but wasn't running";
        return; 
    }

    PLOGI << "AIInferenceManager: Stopping inference...";

    // 2. 【重要】不要在这里调用 this->disconnect() !! 
    // 否则 switchModel 之后信号就发不出去了

    // 3. 唤醒阻塞在 dequeue 的线程
    m_inferenceQueue.clear();
    m_inferenceQueue.enqueue(cv::Mat()); 

    if (m_inferenceThread.joinable()) {
        m_inferenceThread.join();
    }

    // 4. 线程已退出，清除可能残留的空帧（线程若不在 dequeue 而在执行推理，
    //    入队的空帧不会被消费，会残留到下一次 startInference）
    m_inferenceQueue.clear();

    // 5. 重置模型类型为 NONE，下次 DetectionCoordinator::start() 不会自动启动推理。
    //    引擎保留不释放，后续 switchModel() 确认时会替换为新引擎。
    m_currentModelType = "NONE";
    PLOGI << "AIInferenceManager: Stopped, model type reset to NONE.";
}

void AIInferenceManager::setPaused(bool paused) {
    m_isPaused.store(paused);
}

void AIInferenceManager::triggerSnapshot() {
    m_needSnapshot.store(true);
}

void AIInferenceManager::submitFrame(const cv::Mat& frame) {
    if (!m_isRunning.load() || m_isPaused.load() || frame.empty()) {
        return;
    }
    
    // 限制队列长度。如果推理太慢，直接丢弃老帧，确保实时性
    if (m_inferenceQueue.size() > m_inferenceQueueMaxSize) {
        m_inferenceQueue.clear();
    }
    m_inferenceQueue.enqueue(frame);
}

void AIInferenceManager::inferenceLoop() {
    PLOGI << "AIInferenceManager: Inference loop started";

    // Engine is validated under m_engineMutex in startInference()
    // before the thread is created, and swapModel() provides the new
    // engine atomically under the same mutex. The in-loop check below
    // handles the nullptr case safely under the lock.

    while (m_isRunning.load()) {
        // 1. 从队列获取帧
        cv::Mat frame = m_inferenceQueue.dequeue();
        
        // 2. 检查退出信号：如果 frame 为空或者isRunning变为false，立即跳出
        if (!m_isRunning.load() || frame.empty()) {
            break;
        }

        if (m_isPaused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        cv::Mat oriFrame = frame.clone();
        std::vector<Inf::Detection> results;

        {
            // 3. 获取引擎锁
            std::lock_guard<std::mutex> lock(m_engineMutex);
            
            // 再次检查状态，防止在等待锁的过程中程序已要求停止
            if (!m_isRunning.load()) break;

            if (m_inferEngine) {
                try {
                    // AI 推理
                    results = m_inferEngine->run(frame);

                    if (m_currentLogic && m_isRunning.load()) {
                        m_currentLogic->process(frame, results);
                    }

                    m_inferEngine->draw(frame, results);

                    processResults(frame, results);

                } catch (const std::exception& e) {
                    PLOGE << "AIInferenceManager: Inference error: " << e.what();
                    // 只有在运行状态下才 emit 错误
                    if (m_isRunning.load()) {
                        emit inferenceError(QString::fromStdString(e.what()));
                    }
                }
            }
        }

        // 4. 发送结果前最后的“生命值”检查
        // 如果此时 stopInference() 已被调用，这里的 emit 将不会触发任何效果（因为已 disconnect）
        if (!m_isRunning.load()) break;

        // 处理截图
        if (m_needSnapshot.load()) {
            emit snapshotReady(oriFrame, frame);
            m_needSnapshot.store(false);
        }

        // 发送推理结果帧
        emit inferenceFrameReady(frame);
    }

    PLOGI << "AIInferenceManager: Inference loop stopped";
}

void AIInferenceManager::processResults(cv::Mat& frame, const std::vector<Inf::Detection>& results) {
    // 这里可以添加通用的结果处理逻辑
    // 例如：统计检测数量、计算置信度等

    // 当前主要依赖具体的业务逻辑类处理
    // 可以在这里添加额外的后处理逻辑
}