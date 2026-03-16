#pragma once

#include <iostream>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#include "opencv2/opencv.hpp"

#include "plog/Log.h"
#include "plog/Init.h"
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"

// 定义了图像采集器的基类，封装了采集线程、数据队列、帧获取等通用功能，便于不同采集设别的统一管理。

class ImageSensor
{
public:
    ImageSensor(int _queue_max_length, int _capture_interval_ms, bool _is_full_drop);
    ~ImageSensor();
    virtual void start();
    virtual void stop();
    bool isRunning() const { return is_running.load(); }
    void clear();
    void enqueueData(const cv::Mat& img);
    virtual cv::Mat getData();
    virtual cv::Mat getDataNoBlock();
    cv::Mat getLastestFrame();
protected:
    virtual void dataCollectionLoop() = 0;

    int sensor_id;
    int queue_max_length;
    bool is_full_drop;
    std::atomic<bool> is_running;
    std::deque<cv::Mat> images;
    cv::Mat latest_frame;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread sensor_thread;
    int capture_interval_ms;
};
