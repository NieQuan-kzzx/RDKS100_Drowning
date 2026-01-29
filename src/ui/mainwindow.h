#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QTimer>
#include "RTSPCamera.h" // 摄像头类，实现开发板解码等功能，开启/关闭/暂停/截取等功能
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // // 对于摄像头的操作，需要封装一个单独的类来实现摄像头的操作，实现开启/关闭/暂停/截取等功能

    // // 定义所用到的函数
    // // PushButton类
    // void OpenCamera(); // 开始摄像头
    // void CloseCamera(); // 关闭摄像头
    // void PauseCamera(); // 暂停拍摄
    // void CaptureCamera(); // 截取当前帧
    // void ModelRun(); // 模型运行--主要供确认按钮MakeSure调用
    // void StartRecording(); // 开始录制
    // void StopRecording(); // 停止录制

    // // ComboBox类
    // void ModelSelect(); // 模型选择

    // // QLabel类
    // void ShowOriginal(QImage img); // 显示原始图像
    // void ShowProcessed(QImage img); // 显示处理后的图像

private slots:

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
