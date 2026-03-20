#include "AIInferenceManager.h"
#include "Yolo11Infer.h"
#include "Patchcore.h"
#include "DrowningUnderSurface.h"
#include "DrowningState.h"
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

bool AIInferenceManager::switchModel(const std::string& type, const std::string& path) {
    if (m_isRunning.load()) {
        stopInference();
    }

    std::lock_guard<std::mutex> lock(m_engineMutex);
    PLOGI << "AIInferenceManager: Switching model to: " << type;

    m_inferEngine.reset();
    m_currentLogic.reset();

    std::unique_ptr<Inf::BaseInfer> nextEngine;
    std::unique_ptr<LogicBase> nextLogic;

    if (type == "YOLO") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels({"person at surface", "person underwater"});
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningUnderSurface>();
    }
    else if (type == "SWIMMER") {
        auto yolo = std::make_unique<Inf::Yolo11Infer>();
        yolo->setLabels({"drowning", "swimming"});
        nextEngine = std::move(yolo);
        nextLogic = std::make_unique<DrowningState>();
    }
    else if (type == "Patchcore") {
        nextEngine = std::make_unique<Inf::Patchcore>();
    }
    else {
        PLOGE << "AIInferenceManager: Unknown model type: " << type;
        return false;
    }

    if (nextEngine && nextEngine->init(path)) {
        m_inferEngine = std::move(nextEngine);
        m_currentLogic = std::move(nextLogic);
        m_currentModelType = type;
        m_currentModelPath = path;

        PLOGI << "AIInferenceManager: Model switched successfully to " << type;
        emit modelSwitched(QString::fromStdString(type));
        return true;
    }

    PLOGE << "AIInferenceManager: Failed to switch model to " << type;
    emit inferenceError("Failed to switch model: " + QString::fromStdString(type));
    return false;
}

void AIInferenceManager::startInference() {
    std::lock_guard<std::mutex> lock(m_engineMutex); // 启动时也需要加锁检查引擎
    
    if (m_isRunning.load()) {
        PLOGW << "AIInferenceManager: Already running, ignoring start request";
        return;
    }
    // 这里会出现一个INFO的错误误导日志，但实际上是正常的：因为如果没有加载模型，线程会立即退出，这时我们不应该认为是“失败”，而是正常的“无模型可运行”状态。
    if (!m_inferEngine) {
        PLOGE << "AIInferenceManager: Cannot start - No inference engine loaded!";
        return;
    }

    m_isRunning.store(true);
    m_inferenceThread = std::thread(&AIInferenceManager::inferenceLoop, this);
    PLOGI << "AIInferenceManager: Inference thread launched.";
}

void AIInferenceManager::stopInference() {
    // 1. 原子操作停止标志
    if (!m_isRunning.exchange(false)) {
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

    // 4. 清理引擎缓存，释放 MatPool 引用
    {
        std::lock_guard<std::mutex> lock(m_engineMutex);
        m_inferEngine.reset(); 
        m_currentLogic.reset();
    }
    PLOGI << "AIInferenceManager: Stopped and resources cleared.";
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
    if (m_inferenceQueue.size() > 2) {
        m_inferenceQueue.clear();
    }
    m_inferenceQueue.enqueue(frame);
}

void AIInferenceManager::inferenceLoop() {
    PLOGI << "AIInferenceManager: Inference loop started";

    if (m_inferEngine == nullptr) {
        PLOGE << "No model loaded, thread exiting!";
        return; 
    }

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
                    // AI 推理及业务处理
                    results = m_inferEngine->run(frame);
                    m_inferEngine->draw(frame, results);

                    if (m_currentLogic && m_isRunning.load()) {
                        m_currentLogic->process(frame, results);
                    }

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