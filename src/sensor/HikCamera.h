#pragma once
#include <iostream>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstdint> // 必须包含，确保 int32_t 等类型可用
#include "ImageSensor.h"
#include "Config.h"
#include "HCNetSDK.h"

class HikCamera : public ImageSensor
{
public:
    HikCamera(const HikConfig& _config, int _queue_max_length, bool _is_full_drop,
              int _capture_interval_ms);
    virtual ~HikCamera();

    // 供外部（如 main 函数或解码逻辑）获取 H.264 原始包
    std::vector<unsigned char> getRawPacket();

private:
    // 覆盖基类的循环函数
    virtual void dataCollectionLoop() override;
    
    // SDK 核心操作
    bool initSDK();
    bool login();
    bool startRealPlay();
    void stopRealPlay();

    // 核心回调：由海康 SDK 内部线程触发
    static void CALLBACK RealDataCallBack(int lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void* pUser);

    HikConfig config;
    long userID = -1;
    long realHandle = -1;

    // --- 原始流队列（生产者-消费者模型） ---
    std::queue<std::vector<unsigned char>> raw_packet_queue;
    std::mutex queue_mtx;
    std::condition_variable queue_cv;
    const size_t max_raw_q_size = 25; // 限制缓存长度，防止内存堆积
};