#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QTimer>
#include <QDateTime>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_config(AppConfig::loadFromFile("/home/sunrise/Desktop/RDKS100_Drowning/config/config.json"))
{
    ui->setupUi(this);
    qRegisterMetaType<cv::Mat>("cv::Mat");

    initSystems();

    // 状态监控定时器
    QTimer *statusTimer = new QTimer(this);
    connect(statusTimer, &QTimer::timeout, this, &MainWindow::updateButtonStates);
    statusTimer->start(500);

    // 绑定录制按钮
    connect(ui->radioStartRecord, &QRadioButton::clicked, this, [this](){ handleRecording(true); });
    connect(ui->radioStopRecord, &QRadioButton::clicked, this, [this](){ handleRecording(false); });

    // 截图回调
    connect(m_coordinator_1, &DetectionCoordinator::snapshotReady, this, &MainWindow::handleSnapshot);
    connect(m_coordinator_2, &DetectionCoordinator::snapshotReady, this, &MainWindow::handleSnapshot);

    // 模型切换状态回调
    connect(m_coordinator_1, &DetectionCoordinator::modelSwitching, this, [this](){
        ui->statusbar->showMessage("正在加载模型...", 10000);
    });
    connect(m_coordinator_2, &DetectionCoordinator::modelSwitching, this, [this](){
        ui->statusbar->showMessage("正在加载模型...", 10000);
    });
    connect(m_coordinator_1, &DetectionCoordinator::modelSwitched, this, [this](const QString& modelType){
        ui->btnConfirm->setEnabled(true);
        ui->statusbar->showMessage("模型 " + modelType + " 加载完成", 3000);
        // 如果正在录制，启动推理流录制
        if (ui->radioStartRecord->isChecked()) {
            QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
            m_coordinator_1->startRecording("Cam1_Infer_" + timeStr.toStdString());
            m_coordinator_2->startRecording("Cam2_Infer_" + timeStr.toStdString());
            auto recordingManager1 = m_coordinator_1->getRecordingManager();
            auto recordingManager2 = m_coordinator_2->getRecordingManager();
            if (recordingManager1) recordingManager1->setRecordingPerformanceMode(true);
            if (recordingManager2) recordingManager2->setRecordingPerformanceMode(true);
        }
    });
    connect(m_coordinator_2, &DetectionCoordinator::modelSwitched, this, [this](const QString& modelType){
        ui->btnConfirm->setEnabled(true);
        ui->statusbar->showMessage("模型 " + modelType + " 加载完成", 3000);
    });
    connect(m_coordinator_1, &DetectionCoordinator::systemError, this, [this](const QString& error){
        ui->btnConfirm->setEnabled(true);
        ui->statusbar->showMessage("模型加载失败: " + error, 5000);
    });
    connect(m_coordinator_2, &DetectionCoordinator::systemError, this, [this](const QString& error){
        ui->btnConfirm->setEnabled(true);
        ui->statusbar->showMessage("模型加载失败: " + error, 5000);
    });
}

MainWindow::~MainWindow()
{
    // 安全的析构顺序
    // 步骤一：断开信号连接
    if (m_coordinator_1) m_coordinator_1->disconnect(this);
    if (m_coordinator_2) m_coordinator_2->disconnect(this);

    // 步骤二：停止协调器（先停协调器，再停摄像头）
    if (m_coordinator_1) {
        m_coordinator_1->setPaused(false);
        m_coordinator_1->stop();
    }
    if (m_coordinator_2) {
        m_coordinator_2->setPaused(false);
        m_coordinator_2->stop();
    }

    // 步骤三：停止并释放摄像头
    if (m_cam_1) { m_cam_1->stop(); delete m_cam_1; m_cam_1 = nullptr; }
    if (m_cam_2) { m_cam_2->stop(); delete m_cam_2; m_cam_2 = nullptr; }

    // 步骤四：释放线程池（必须 join 所有工作线程，否则进程无法退出）
    if (m_pool) { delete m_pool; m_pool = nullptr; }

    // 步骤五：清理UI
    delete ui;
}

void MainWindow::initSystems()
{
    auto makeCam = [&](const CameraConfig& c) -> RTSPCamera* {
        RTSPCamera::DecodeMode mode = (c.decode_mode == "SOFTWARE")
            ? RTSPCamera::SOFTWARE : RTSPCamera::HARDWARE;
        return new RTSPCamera(c.url, c.width, c.height,
                              c.queue_max_length, c.capture_interval_ms,
                              c.is_full_drop, mode);
    };
    if (m_config.cameras.size() >= 1)
        m_cam_1 = makeCam(m_config.cameras[0]);
    if (m_config.cameras.size() >= 2)
        m_cam_2 = makeCam(m_config.cameras[1]);
    m_pool = new ThreadPool(4);
    m_coordinator_1 = new DetectionCoordinator(m_cam_1, 1, this);
    m_coordinator_2 = new DetectionCoordinator(m_cam_2, 2, this);

    m_coordinator_1->getInferenceManager()->setInferenceQueueMaxSize(m_config.inference_queue_max_size);
    m_coordinator_2->getInferenceManager()->setInferenceQueueMaxSize(m_config.inference_queue_max_size);

    updateButtonStates();
}

// ---------------- 录制逻辑 (修复重定义) ----------------

void MainWindow::handleRecording(bool start)
{
    if ((!m_cam_1 || !m_cam_1->isRunning()) && (!m_cam_2 || !m_cam_2->isRunning())) return;

    if (start) {
        QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        
        m_coordinator_1->startRecording("Cam1_" + timeStr.toStdString());
        m_coordinator_2->startRecording("Cam2_" + timeStr.toStdString());

        if (m_coordinator_1->isRunning() || m_coordinator_2->isRunning()) {
            ui->statusbar->showMessage("已开启双路录制", 3000);
        } else {
            ui->statusbar->showMessage("已开启原始路录制 (AI未就绪)", 3000);
        }
    } else {
        m_coordinator_1->stopRecording();
        m_coordinator_2->stopRecording();
        ui->statusbar->showMessage("已停止所有录制", 3000);
    }
}

// ---------------- 截图逻辑 (修复未声明变量) ----------------

void MainWindow::on_btnCapture_clicked()
{
    if (!m_cam_1 || !m_cam_1->isRunning()) return;

    // 修正：改用 m_coordinator 接口
    if(m_coordinator_1) m_coordinator_1->triggerSnapshot();
    if(m_coordinator_2) m_coordinator_2->triggerSnapshot();
    
    ui->statusbar->showMessage("截图指令已发送", 2000);
}

// ---------------- 模型切换 (修复未声明与捕获) ----------------

void MainWindow::on_btnConfirm_clicked()
{
    if (!m_cam_1 || !m_cam_1->isRunning()) return;

    QString selectedMode_1 = ui->comboBoxModels_1->currentText();
    QString selectedMode_2 = ui->comboBoxModels_2->currentText();
    
    // 从配置中查找对应模式的模型参数
    auto findModel = [&](const QString& mode) -> ModelEntry {
        std::string key = mode.toStdString();
        for (const auto& m : m_config.models) {
            if (m.key == key) return m;
        }
        // 回退：找"行人检测"或默认值
        for (const auto& m : m_config.models) {
            if (m.key == "行人检测") return m;
        }
        ModelEntry fb;
        fb.type = "YOLO";
        fb.path = "/home/sunrise/Desktop/RDKS100_Drowning/models/Zuoshiyan.hbm";
        fb.labels = {"person"};
        return fb;
    };

    ModelEntry model_1 = findModel(selectedMode_1);
    ModelEntry model_2 = findModel(selectedMode_2);

    // 使用异步版本，不阻塞主线程
    m_coordinator_1->switchModelAsync(model_1.type, model_1.path, model_1.labels, model_1.params);
    m_coordinator_2->switchModelAsync(model_2.type, model_2.path, model_2.labels, model_2.params);

    // 禁用按钮，防止重复点击
    ui->btnConfirm->setEnabled(false);
    ui->statusbar->showMessage("正在加载模型，请稍候...", 10000);
}

void MainWindow::updateButtonStates()
{
    bool isCamRunning = m_cam_1 && m_cam_1->isRunning();
    
    // UI 可用性控制
    ui->btnCapture->setEnabled(isCamRunning); 
    ui->btnPause->setEnabled(isCamRunning);
    ui->radioStartRecord->setEnabled(isCamRunning);
    ui->radioStopRecord->setEnabled(isCamRunning);
    ui->btnConfirm->setEnabled(isCamRunning);

    // 逻辑保护：如果摄像头意外断开，强制重置录制单选框
    if (!isCamRunning && ui->radioStartRecord->isChecked()) {
        ui->radioStopRecord->setChecked(true);
        handleRecording(false);
    }
}

// ---------------- 图像处理与显示 ----------------

void MainWindow::updateUI(cv::Mat frame)
{
    if (frame.empty()) return;

    cv::Mat showFrame;
    cv::resize(frame, showFrame, cv::Size(m_config.display.resize_width, m_config.display.resize_height));

    QImage img = matToQImage(showFrame);
    ui->labelOriginal_1->setPixmap(QPixmap::fromImage(img).scaled(
        ui->labelOriginal_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui->labelOriginal_2->setPixmap(QPixmap::fromImage(img).scaled(
        ui->labelOriginal_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.type() == CV_8UC3) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888);
    } else if (mat.type() == CV_8UC1) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    }
    return QImage();
}

// ---------------- 按钮逻辑实现 ----------------

void MainWindow::on_btnOpen_clicked() {
    // --- 摄像头 1 的连接 ---
    connect(m_coordinator_1, &DetectionCoordinator::frameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(m_config.display.resize_width, m_config.display.resize_height));
        QImage img = matToQImage(showFrame);
        ui->labelOriginal_1->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelOriginal_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    connect(m_coordinator_1, &DetectionCoordinator::inferenceFrameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(m_config.display.resize_width, m_config.display.resize_height));
        QImage img = matToQImage(showFrame);
        ui->labelProcessed_1->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelProcessed_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    // --- 摄像头 2 的连接 ---
    connect(m_coordinator_2, &DetectionCoordinator::frameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(m_config.display.resize_width, m_config.display.resize_height));
        QImage img = matToQImage(showFrame);
        ui->labelOriginal_2->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelOriginal_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    connect(m_coordinator_2, &DetectionCoordinator::inferenceFrameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(m_config.display.resize_width, m_config.display.resize_height));
        QImage img = matToQImage(showFrame);
        ui->labelProcessed_2->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelProcessed_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    // 启动系统（coordinator会管理camera的启动）
    if (m_coordinator_1) m_coordinator_1->start();
    if (m_coordinator_2) m_coordinator_2->start();
    
    ui->statusbar->showMessage("系统已启动", 3000);
}

void MainWindow::on_btnClose_clicked() {
    if (m_coordinator_1) m_coordinator_1->disconnect(this);
    if (m_coordinator_2) m_coordinator_2->disconnect(this);

    handleRecording(false);

    // Coordinator -> Camera
    if (m_coordinator_1) {
        m_coordinator_1->setPaused(false);
        m_coordinator_1->stop();
    }
    if (m_coordinator_2) {
        m_coordinator_2->setPaused(false);
        m_coordinator_2->stop();
    }
    if (m_cam_1) m_cam_1->stop();
    if (m_cam_2) m_cam_2->stop();

    ui->btnPause->setText("暂停拍摄"); 

    // 断开所有连接防止内存池在关闭后仍被异步调用

    ui->labelOriginal_1->clear();
    ui->labelProcessed_1->clear();
    ui->labelOriginal_1->setText("摄像头已关闭");
    
    ui->labelOriginal_2->clear();
    ui->labelProcessed_2->clear();
    ui->labelOriginal_2->setText("摄像头已关闭");

    updateButtonStates();
}

void MainWindow::on_btnPause_clicked() {
    if (!m_coordinator_1) return;

    bool currentState = m_coordinator_1->isPaused();
    bool newState = !currentState;

    m_coordinator_1->setPaused(newState);
    if (m_coordinator_2) m_coordinator_2->setPaused(newState);
    
    ui->btnPause->setText(newState ? "恢复拍摄" : "暂停拍摄");
}

void MainWindow::handleSnapshot(cv::Mat raw, cv::Mat infer, int id) {
    QString dirPath = "/media/UBUNTU 18_01/records/";
    QDir dir;
    if (!dir.exists(dirPath)) dir.mkpath(dirPath); 

    QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz");

    if (!raw.empty()) {
        std::string rawName = QString("%1Cam%2_Raw_%3.jpg").arg(dirPath).arg(id).arg(timeStr).toStdString();
        cv::imwrite(rawName, raw);
    }
    if (!infer.empty()) {
        std::string inferName = QString("%1Cam%2_Infer_%3.jpg").arg(dirPath).arg(id).arg(timeStr).toStdString();
        cv::imwrite(inferName, infer);
    }
}
