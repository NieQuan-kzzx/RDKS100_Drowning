#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDateTime>
#include <QMetaType>
#include <QPixmap>
#include <opencv2/opencv.hpp>

// 包含你的自定义类
#include "RTSPCamera.h"
#include "ThreadPool.h"
#include "DetectionWorker.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 按钮槽函数
    void on_btnOpen_clicked();      // 开启摄像头
    void on_btnClose_clicked();     // 关闭摄像头
    void on_btnPause_clicked();     // 暂停/恢复
    void on_btnCapture_clicked();   // 截图
    void on_btnConfirm_clicked();   // 确认模型
    
    // UI 状态维护
    void updateButtonStates();

private:
    // 内部逻辑：处理录制开启/关闭
    void handleRecording(bool start); 
    
    // 接收 Worker 传回的图像进行显示
    void updateUI(cv::Mat frame);

private:
    Ui::MainWindow *ui;

    // 核心组件
    RTSPCamera      *m_cam    = nullptr;
    ThreadPool      *m_pool   = nullptr;
    DetectionWorker *m_worker = nullptr;

    // 工具函数
    QImage matToQImage(const cv::Mat& mat);
    void initSystems();
};

#endif // MAINWINDOW_H