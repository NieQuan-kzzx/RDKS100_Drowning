#include <iostream>
#include <csignal>
#include "HikCamera.h"
#include "Config.h"
#include "PlogInitializer.h"

// 在这里加上地平线的封装的解码库
#include "sp_codec.h"

// 全局变量，用于优雅退出
bool g_keep_running = true;
void signalHandler(int signum) {
    g_keep_running = false;
}

int main() {
    // 1. 初始化日志
    PlogInitializer::getInstance().init(plog::info);
    PLOGI << "===== HikCamera Stream Callback Test =====";

    // 注册信号处理 (Ctrl+C)
    signal(SIGINT, signalHandler);

    // 初始化解码器
    void* decoder = nullptr;
    decoder = sp_init_decoder_module();

    // 2. 准备配置 (实际项目中建议从 Cereal 加载)
    HikConfig config;
    config.ip = "192.168.127.15";
    config.port = 8000;
    config.user = "admin";
    config.pass = "waterline123456";
    
    // 3. 实例化 HikCamera
    // 参数说明：配置, 队列长度(10), 是否丢弃旧帧(true), 采集间隔(33ms -> 约30fps)
    HikCamera camera(config, 10, true, 33);

    // 4. 启动采集线程
    PLOGI << "Starting camera data collection...";
    camera.start(); 

    // 5. 模拟算法处理逻辑 (消费者)
    PLOGI << "Consumer loop started. Press Ctrl+C to stop.";
    while (g_keep_running) {
        // 尝试从 ImageSensor 的队列中获取一帧
        cv::Mat frame = camera.getData();
        if (!frame.empty()) {
            // 如果你还没写解码器，这里拿到的可能是空 Mat 或 占位符
            // 一旦解码器接入，这里就是 YOLO 处理的地方
            PLOGD << "Consumer: Got a frame from queue. Size: " << frame.cols << "x" << frame.rows;
            
            // 如果有 X11 转发，可以尝试显示
            // cv::imshow("Real-time Stream", frame);
            // cv::waitKey(1);
        } else {
            // 队列为空，稍等一下
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // 6. 安全停止
    PLOGI << "Stopping camera...";
    camera.stop(); // 这会触发 is_running = false 并释放 SDK 资源

    PLOGI << "Test finished.";
    return 0;
}