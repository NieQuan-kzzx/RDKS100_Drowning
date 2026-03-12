#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QTimer>
#include <QDateTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    qRegisterMetaType<cv::Mat>("cv::Mat");

    initSystems();

    // 启动状态监控定时器，每 500ms 更新一次 UI 按钮可用性
    QTimer *statusTimer = new QTimer(this);
    connect(statusTimer, &QTimer::timeout, this, &MainWindow::updateButtonStates);
    statusTimer->start(500);

    // 绑定录制按钮点击事件
    connect(ui->radioStartRecord, &QRadioButton::clicked, this, [this](){ handleRecording(true); });
    connect(ui->radioStopRecord, &QRadioButton::clicked, this, [this](){ handleRecording(false); });
}

MainWindow::~MainWindow()
{
    if (m_worker_1) m_worker_1->stop();
    if (m_worker_2) m_worker_2->stop();

    if (m_cam_1) m_cam_1->stop();
    if (m_cam_2) m_cam_2->stop();
    delete ui;
}

void MainWindow::initSystems()
{
    // rtsp://admin:nuaa2026@192.168.127.15
    // rtsp://127.0.0.1/assets/swim_fixed.h264
    m_cam_1 = new RTSPCamera("rtsp://admin:nuaa2026@192.168.127.15", 1920, 1080, 10, 0, false);
    m_cam_2 = new RTSPCamera("rtsp://127.0.0.1/assets/swim_fixed.h264", 1920, 1080, 10, 0, false);
    m_pool = new ThreadPool(4);
    m_worker_1 = new DetectionWorker(m_cam_1, this);
    m_worker_2 = new DetectionWorker(m_cam_2, this);

    // 初始按钮状态刷新
    updateButtonStates();
}

// ---------------- 录制统一调度逻辑 ----------------

void MainWindow::handleRecording(bool start)
{
    if ((!m_cam_1 || !m_cam_1->isRunning()) && (!m_cam_2 || !m_cam_2->isRunning())) return;

    if (start) {
        QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        
        // 1. 启动原始路录制
        QString rawPath_1 = "records/Raw_1_" + timeStr + ".avi";
        QString inferPath_1 = "records/Infer_1_" + timeStr + ".avi";
        QString rawPath_2 = "records/Raw_2_" + timeStr + ".avi";
        QString inferPath_2 = "records/Infer_2_" + timeStr + ".avi";
        m_worker_1->setRecording(true, rawPath_1.toStdString());
        m_worker_2->setRecording(true, rawPath_2.toStdString());

        // 2. 如果推理正在运行，同时启动推理路录制
        if (m_worker_1->isInferRunning()) {    
            m_worker_1->setInferRecording(true, inferPath_1.toStdString());
            ui->statusbar->showMessage("已开启双路录制", 3000);
        } else {
            ui->statusbar->showMessage("已开启原始路录制 (AI未就绪)", 3000);
        }
        if (m_worker_2->isInferRunning()) {    
            m_worker_2->setInferRecording(true, inferPath_2.toStdString());
            ui->statusbar->showMessage("已开启双路录制", 3000);
        } else {
            ui->statusbar->showMessage("已开启原始路录制 (AI未就绪)", 3000);
        }

        // auto startWorkerRecording = [&](DetectionWorker* worker, QString prefix) {
        //     if (!worker->getCamera() || !worker->getCamera()->isRunning()) return;

        //     // 1. 原始路录制
        //     QString rawPath = QString("records/%1_Raw_%2.avi").arg(prefix).arg(timeStr);
        //     worker->setRecording(true, rawPath.toStdString());

        //     // 2. 推理路录制
        //     if (worker->isInferRunning()) {
        //         QString inferPath = QString("records/%1_Infer_%2.avi").arg(prefix).arg(timeStr);
        //         worker->setInferRecording(true, inferPath.toStdString());
        //     }
        // };
        
        // // 分别启动两路
        // startWorkerRecording(m_worker_1, "Cam1");
        // startWorkerRecording(m_worker_2, "Cam2");

        // ui->statusbar->showMessage("已开始录制", 3000);
    } else {
        // 关闭所有录制
        m_worker_1->setRecording(false);
        m_worker_1->setInferRecording(false);
        m_worker_2->setRecording(false);
        m_worker_2->setInferRecording(false);
        ui->statusbar->showMessage("已停止所有录制", 3000);
    }
}

// ---------------- 按钮逻辑实现 ----------------

void MainWindow::on_btnOpen_clicked() {
    // --- 摄像头 1 的连接 ---
    connect(m_worker_1, &DetectionWorker::frameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(640, 360));
        QImage img = matToQImage(showFrame);
        ui->labelOriginal_1->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelOriginal_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    connect(m_worker_1, &DetectionWorker::inferFrameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(640, 360));
        QImage img = matToQImage(showFrame);
        ui->labelProcessed_1->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelProcessed_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    // --- 摄像头 2 的连接 ---
    connect(m_worker_2, &DetectionWorker::frameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(640, 360));
        QImage img = matToQImage(showFrame);
        ui->labelOriginal_2->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelOriginal_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    connect(m_worker_2, &DetectionWorker::inferFrameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(640, 360));
        QImage img = matToQImage(showFrame);
        ui->labelProcessed_2->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelProcessed_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    // 启动线程...
    m_cam_1->start();
    m_cam_2->start();
    m_pool->enqueue([this](){ m_worker_1->processLoop(); });
    m_pool->enqueue([this](){ m_worker_2->processLoop(); });
}

void MainWindow::on_btnClose_clicked() {
    // 1. 立即停止所有工作位
    m_worker_1->stop(); 
    m_worker_2->stop();
    m_worker_1->setPaused(false); // 强制取消暂停状态，防止线程卡在暂停循环里
    m_worker_2->setPaused(false);

    m_cam_1->stop();
    m_cam_2->stop();
    handleRecording(false); 

    // 2. 重置按钮文字（解决点击两次的问题）
    ui->btnPause->setText("暂停拍摄"); 

    // 3. UI 清理
    disconnect(m_worker_1, &DetectionWorker::frameReady, this, &MainWindow::updateUI);
    disconnect(m_worker_1, &DetectionWorker::inferFrameReady, nullptr, nullptr);
    disconnect(m_worker_2, &DetectionWorker::frameReady, this, &MainWindow::updateUI);
    disconnect(m_worker_2, &DetectionWorker::inferFrameReady, nullptr, nullptr);

    ui->labelOriginal_1->clear();
    ui->labelProcessed_1->clear();
    ui->labelOriginal_1->setText("摄像头已关闭");
    ui->labelProcessed_1->setText("推理已停止");
    
    ui->labelOriginal_2->clear();
    ui->labelProcessed_2->clear();
    ui->labelOriginal_2->setText("摄像头已关闭");
    ui->labelProcessed_2->setText("推理已停止");

    updateButtonStates();
}

void MainWindow::on_btnPause_clicked()
{
    if (!m_cam_1 || !m_cam_1->isRunning()) return;

    bool currentState = m_worker_1->isPaused();
    bool newState = !currentState;

    m_worker_1->setPaused(newState);
    m_worker_2->setPaused(newState);
    ui->btnPause->setText(newState ? "恢复拍摄" : "暂停拍摄");
}

void MainWindow::on_btnCapture_clicked()
{
    if (!m_cam_1 || !m_cam_1->isRunning()) return;

    // Worker 内部会自动判断单路或双路截图
    m_worker_1->triggerSnapshot(); 
    m_worker_2->triggerSnapshot(); 
    ui->statusbar->showMessage("截图指令已发送", 2000);
}

void MainWindow::on_btnConfirm_clicked()
{
    if (!m_cam_1 || !m_cam_1->isRunning()) return;

    QString selectedMode_1 = ui->comboBoxModels_1->currentText();
    QString selectedMode_2 = ui->comboBoxModels_2->currentText();
    std::string modelType_1 = "YOLO"; 
    std::string modelPath_1 = "/home/sunrise/Desktop/RDKS100_Drowning/models/Under_Surface_v1.hbm";
    std::string modelType_2 = "YOLO"; 
    std::string modelPath_2 = "/home/sunrise/Desktop/RDKS100_Drowning/models/Under_Surface_v1.hbm";

    if (selectedMode_1.contains("溺水检测")) {
        modelType_1 = "YOLO";
        modelPath_1 = "/home/sunrise/Desktop/RDKS100_Drowning/models/Under_Surface_v1.hbm";
    } 
    else if (selectedMode_1.contains("游泳检测")) {
        modelType_1 = "SWIMMER";
        modelPath_1 = "/home/sunrise/Desktop/RDKS100_Drowning/models/drowning_TwoSelect.hbm";
    }
    else if (selectedMode_1.contains("进水检测")) {
        modelType_1 = "Patchcore";
        modelPath_1 = "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm";
    }

    if (selectedMode_2.contains("溺水检测")) {
        modelType_2 = "YOLO";
        modelPath_2 = "/home/sunrise/Desktop/RDKS100_Drowning/models/Under_Surface_v1.hbm";
    } 
    else if (selectedMode_2.contains("游泳检测")) {
        modelType_2 = "SWIMMER";
        modelPath_2 = "/home/sunrise/Desktop/RDKS100_Drowning/models/drowning_TwoSelect.hbm";
    }
    else if (selectedMode_2.contains("进水检测")) {
        modelType_2 = "Patchcore";
        modelPath_2 = "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm";
    }

    m_worker_1->switchModel(modelType_1, modelPath_1);
    m_worker_2->switchModel(modelType_2, modelPath_2);

    // 启动推理循环
    m_pool->enqueue([this](){ m_worker_1->inferenceLoop(); });
    m_pool->enqueue([this](){ m_worker_2->inferenceLoop(); });

    // --- 联动点：如果正在录制中，开启模型的一瞬间自动补齐推理流录制 ---
    if (ui->radioStartRecord->isChecked()) {
        QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        
        QString inferPath_1 = "records/Infer_1_" + timeStr + ".avi";
        QString inferPath_2 = "records/Infer_2_" + timeStr + ".avi";
        m_worker_1->setInferRecording(true, inferPath_1.toStdString());
        m_worker_2->setInferRecording(true, inferPath_2.toStdString());
    }
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
    cv::resize(frame, showFrame, cv::Size(640, 360));

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