#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 1. 必须注册 cv::Mat 类型，否则跨线程信号槽无法传递图像
    qRegisterMetaType<cv::Mat>("cv::Mat");

    // 2. 初始化系统
    initSystems();
}

MainWindow::~MainWindow()
{
    // 停止所有后台逻辑
    if (m_worker) m_worker->stop();
    if (m_cam) m_cam->stop();
    
    delete ui;
    // ThreadPool 和 RTSPCamera 的析构函数会自动处理线程回收
}

void MainWindow::initSystems()
{
    // 初始化摄像头 (RTSP地址, 宽, 高, 队列长度, 间隔, 是否丢弃新帧)
    m_cam = new RTSPCamera("rtsp://admin:waterline123456@192.168.127.15", 1920, 1080, 10, 0, false);

    // 初始化线程池 (建议线程数根据 CPU 核心数设置)
    m_pool = new ThreadPool(4);

    // 初始化工作者
    m_worker = new DetectionWorker(m_cam, this);

    // 连接信号：当 Worker 处理完图像时，通知 UI 更新
    connect(m_worker, &DetectionWorker::frameReady, this, &MainWindow::updateUI);
    connect(ui->radioStartRecord, &QRadioButton::clicked, this, [this]() {
        QString fileName = "Record_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".avi";
        m_worker->setRecording(true, fileName.toStdString());
        PLOGI << "Start Recording: " << fileName.toStdString();
    });
    connect(ui->radioStopRecord, &QRadioButton::clicked, this, [this]() {
        m_worker->setRecording(false);
        PLOGI << "Stop Recording";
    });
}

// ---------------- 按钮逻辑实现 ----------------
void MainWindow::on_btnOpen_clicked() {
    // 1. 信号安全连接：防止多次 connect 导致处理函数被调用多次
    disconnect(m_worker, &DetectionWorker::frameReady, this, &MainWindow::updateUI);
    connect(m_worker, &DetectionWorker::frameReady, this, &MainWindow::updateUI, Qt::UniqueConnection);

    // 2. 启动硬件
    m_cam->start(); 

    // 3. 启动处理逻辑
    m_pool->enqueue([this](){ m_worker->processLoop(); });
    
    ui->btnOpen->setEnabled(false);
    ui->btnClose->setEnabled(true);
}

void MainWindow::on_btnClose_clicked() {
    // 1. 掐断信号流，防止 UI 在销毁期间接收数据导致崩溃
    disconnect(m_worker, &DetectionWorker::frameReady, this, &MainWindow::updateUI);

    // 2. 停止逻辑和硬件
    m_worker->stop();
    m_cam->stop();

    // 3. 强制给各个线程一点点退出循环的时间
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    ui->labelOriginal->clear();
    ui->labelProcessed->clear();
    
    ui->btnOpen->setEnabled(true);
    ui->btnClose->setEnabled(false);
    PLOGI << "System closed.";
}

void MainWindow::on_btnPause_clicked()
{
    static bool isPaused = false;
    isPaused = !isPaused;
    
    m_worker->setPaused(isPaused);
    ui->btnPause->setText(isPaused ? "恢复拍摄" : "暂停拍摄");
}

void MainWindow::on_btnCapture_clicked()
{
    // 触发 Worker 内部的截图逻辑
    m_worker->triggerSnapshot();
}

void MainWindow::on_btnConfirm_clicked()
{
    // 这里可以获取下拉框选中的模型，传递给 Worker 加载模型
    QString selectedModel = ui->comboBoxModels->currentText();
    // m_worker->loadModel(selectedModel); // 预留给 YOLO 推理接口
}

// ---------------- 图像处理与显示 ----------------

void MainWindow::updateUI(cv::Mat frame)
{
    if (frame.empty()) return;

    // 将 1080P 缩放为适合显示的尺寸，减少 UI 渲染压力
    cv::Mat showFrame;
    cv::resize(frame, showFrame, cv::Size(640, 360));

    // 转换为 QImage 并显示到左侧 Label
    QImage img = matToQImage(showFrame);
    ui->labelOriginal->setPixmap(QPixmap::fromImage(img).scaled(
        ui->labelOriginal->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.type() == CV_8UC3) {
        // BGR -> RGB
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888).rgbSwapped();
    } else if (mat.type() == CV_8UC1) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    }
    return QImage();
}