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
    if (m_worker) m_worker->stop();
    if (m_cam) m_cam->stop();
    delete ui;
}

void MainWindow::initSystems()
{
    m_cam = new RTSPCamera("rtsp://admin:waterline123456@192.168.127.15", 1920, 1080, 10, 0, false);
    m_pool = new ThreadPool(4);
    m_worker = new DetectionWorker(m_cam, this);

    // 初始按钮状态刷新
    updateButtonStates();
}

// ---------------- 录制统一调度逻辑 ----------------

void MainWindow::handleRecording(bool start)
{
    if (!m_cam || !m_cam->isRunning()) return;

    if (start) {
        QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        
        // 1. 启动原始路录制
        QString rawPath = "records/Raw_" + timeStr + ".avi";
        m_worker->setRecording(true, rawPath.toStdString());

        // 2. 如果推理正在运行，同时启动推理路录制
        if (m_worker->isInferRunning()) {
            QString inferPath = "records/Infer_" + timeStr + ".avi";
            m_worker->setInferRecording(true, inferPath.toStdString());
            ui->statusbar->showMessage("已开启双路录制", 3000);
        } else {
            ui->statusbar->showMessage("已开启原始路录制 (AI未就绪)", 3000);
        }
    } else {
        // 关闭所有录制
        m_worker->setRecording(false);
        m_worker->setInferRecording(false);
        ui->statusbar->showMessage("已停止所有录制", 3000);
    }
}

// ---------------- 按钮逻辑实现 ----------------

void MainWindow::on_btnOpen_clicked() {
    // 连接信号
    connect(m_worker, &DetectionWorker::frameReady, this, &MainWindow::updateUI, Qt::UniqueConnection);
    connect(m_worker, &DetectionWorker::inferFrameReady, this, [this](cv::Mat frame){
        if (frame.empty()) return;
        cv::Mat showFrame;
        cv::resize(frame, showFrame, cv::Size(640, 360));
        QImage img = matToQImage(showFrame);
        ui->labelProcessed->setPixmap(QPixmap::fromImage(img).scaled(
            ui->labelProcessed->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }, Qt::UniqueConnection);

    m_cam->start(); 
    m_pool->enqueue([this](){ m_worker->processLoop(); });
}

void MainWindow::on_btnClose_clicked() {
    // 停止录制和工作流
    handleRecording(false); // 确保录制安全关闭
    m_worker->stop();
    m_cam->stop();

    updateButtonStates();

    // 断开显示信号，清空 Label
    disconnect(m_worker, &DetectionWorker::frameReady, this, &MainWindow::updateUI);
    disconnect(m_worker, &DetectionWorker::inferFrameReady, nullptr, nullptr);

    ui->labelOriginal->clear();
    ui->labelProcessed->clear();
    ui->labelOriginal->setText("摄像头已关闭");
    ui->labelProcessed->setText("推理已停止");
}

void MainWindow::on_btnPause_clicked()
{
    if (!m_cam || !m_cam->isRunning()) return;

    static bool isPaused = false;
    isPaused = !isPaused;
    m_worker->setPaused(isPaused);
    ui->btnPause->setText(isPaused ? "恢复拍摄" : "暂停拍摄");
}

void MainWindow::on_btnCapture_clicked()
{
    if (!m_cam || !m_cam->isRunning()) return;

    // Worker 内部会自动判断单路或双路截图
    m_worker->triggerSnapshot(); 
    ui->statusbar->showMessage("截图指令已发送", 2000);
}

void MainWindow::on_btnConfirm_clicked()
{
    if (!m_cam || !m_cam->isRunning()) return;

    QString selectedMode = ui->comboBoxModels->currentText();
    std::string modelType = "YOLO11"; 
    std::string modelPath = "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11s.hbm";

    if (selectedMode.contains("目标检测")) {
        modelType = "YOLO";
        modelPath = "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11s.hbm";
    } 
    else if (selectedMode.contains("特征点检测")) {
        modelType = "YOLOPose";
        modelPath = "/home/sunrise/Desktop/RDKS100_Drowning/models/YOLO11n-pose.hbm";
    }
    else if (selectedMode.contains("进水检测")) {
        modelType = "Patchcore";
        modelPath = "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm";
    }

    m_worker->switchModel(modelType, modelPath);
    
    // 启动推理循环
    m_pool->enqueue([this](){ m_worker->inferenceLoop(); });

    // --- 联动点：如果正在录制中，开启模型的一瞬间自动补齐推理流录制 ---
    if (ui->radioStartRecord->isChecked()) {
        QString timeStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QString inferPath = "records/Infer_" + timeStr + ".avi";
        m_worker->setInferRecording(true, inferPath.toStdString());
    }
}

void MainWindow::updateButtonStates()
{
    bool isCamRunning = m_cam && m_cam->isRunning();
    
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
    ui->labelOriginal->setPixmap(QPixmap::fromImage(img).scaled(
        ui->labelOriginal->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.type() == CV_8UC3) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888).rgbSwapped();
    } else if (mat.type() == CV_8UC1) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    }
    return QImage();
}