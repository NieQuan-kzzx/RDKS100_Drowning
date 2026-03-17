#include "ImageSensor.h"
#include "PlogInitializer.h"

ImageSensor::ImageSensor(int _queue_max_length, int _capture_interval_ms, bool _is_full_drop)
    : queue_max_length(_queue_max_length),
      is_full_drop(_is_full_drop),
      is_running(false),
      capture_interval_ms(_capture_interval_ms)
{
    // 确保 plog 已初始化（库自动初始化，不与主程序冲突）
    ENSURE_PLOG_INITIALIZED();
}

ImageSensor::~ImageSensor()
{
    this->stop();
}

void ImageSensor::start()
{
    // 基类逻辑改为：仅设置状态，不直接创建线程
    // 或者完全交由派生类实现
    if (this->is_running.load() == false)
    {
        this->is_running.store(true);
        PLOGI << "ImageSensor State: Running";
    }
}

void ImageSensor::stop()
{
    if (this->is_running.load() == false)
        return;

    this->is_running.store(false);
    
    // 这里的 join 逻辑要小心，确保线程确实是由这里管理的
    if (this->sensor_thread.joinable()) {
        this->sensor_thread.join();
    }
    PLOGI << "ImageSensor Stop!";
}

void ImageSensor::clear()
{
    std::unique_lock<std::mutex> lock(mutex);
    while (!images.empty())
    {
        images.pop_front();
    }
}

void ImageSensor::enqueueData(const cv::Mat &img)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (images.size() >= queue_max_length)
    {
        if (this->is_full_drop == true)
        {
            PLOGV << "Drop latest image because queue is full!";
            return;
        }
        else
        {
            PLOGV << "Drop oldest image because queue is full!";
            images.pop_front();
        }
    }
    images.push_back(img.clone());
    latest_frame = img.clone();
    cv.notify_one();
    // PLOGV << "enqueueData! queue size: " << this->images.size();
}

cv::Mat ImageSensor::getDataNoBlock()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (this->images.empty())
    {
        // PLOGV << "No data available, returning empty Mat.";
        return cv::Mat();
    }
    cv::Mat img = this->images.front();
    this->images.pop_front();
    return img;
}

cv::Mat ImageSensor::getData()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (this->images.empty())
    {
        PLOGV << "Blocking wait for new video frame...";
        cv.wait(lock);
    }
    cv::Mat frame = this->images.front();
    this->images.pop_front();
    
    PLOGV << "getData! queue size: " << this->images.size();
    return frame;
}


cv::Mat ImageSensor::getLastestFrame()
{
    std::unique_lock<std::mutex> lock(mutex);
    return this->latest_frame.clone();
}