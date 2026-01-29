#include "HikCamera.h"
#include <cstring>
#include <plog/Log.h>

HikCamera::HikCamera(const HikConfig& _config, int _queue_max_length, bool _is_full_drop,
                    int _capture_interval_ms)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      config(_config)
{
}

HikCamera::~HikCamera()
{
    this->is_running = false;
    stopRealPlay();
}


// 生产者：将 SDK 吐出的数据压入队列
void CALLBACK HikCamera::RealDataCallBack(int lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void* pUser)
{
    HikCamera* pThis = static_cast<HikCamera*>(pUser);
    
    // NET_DVR_STREAMDATA 表示这是 H.264/H.265 码流数据
    if (dwDataType == NET_DVR_STREAMDATA && pBuffer != nullptr && dwBufSize > 0)
    {
        std::lock_guard<std::mutex> lock(pThis->queue_mtx);
        
        // 如果消费太慢，丢弃最老的包以保持实时性
        if (pThis->raw_packet_queue.size() > pThis->max_raw_q_size) {
            pThis->raw_packet_queue.pop();
        }

        // 构造 packet 并存入队列
        std::vector<unsigned char> packet(pBuffer, pBuffer + dwBufSize);
        pThis->raw_packet_queue.push(std::move(packet));
        
        // 通知等待的消费者
        pThis->queue_cv.notify_one();
    }
}

// 消费者：供主线程提取码流喂给地平线解码器
std::vector<unsigned char> HikCamera::getRawPacket()
{
    std::unique_lock<std::mutex> lock(queue_mtx);
    
    // 如果队列为空，最多等待 10ms，避免主循环空转 CPU 占用过高
    if (queue_cv.wait_for(lock, std::chrono::milliseconds(10), [this] { return !raw_packet_queue.empty(); })) {
        std::vector<unsigned char> packet = std::move(raw_packet_queue.front());
        raw_packet_queue.pop();
        return packet;
    }
    
    return {}; // 返回空包
}

void HikCamera::dataCollectionLoop()
{
    if (!initSDK()) {
        PLOG_ERROR << "Init Hik SDK failed";
        return;
    }
    if (!login()) {
        PLOG_ERROR << "Hik login failed, IP: " << config.ip;
        this->is_running = false;
        return;
    }
    if (!startRealPlay()) {
        PLOG_ERROR << "Start RealPlay failed";
        this->is_running = false;
        return;
    }

    PLOG_INFO << "HikCamera [" << config.ip << "] collection loop is active.";

    // 回调在独立线程工作，此处只需守护状态
    while (this->is_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    stopRealPlay();
}

bool HikCamera::initSDK()
{
    bool ret = NET_DVR_Init();
    NET_DVR_SetConnectTime(2000, 1);
    NET_DVR_SetReconnect(10000, true);
    return ret;
}

bool HikCamera::login()
{
    NET_DVR_USER_LOGIN_INFO loginInfo = {0};
    strncpy(loginInfo.sDeviceAddress, config.ip.c_str(), sizeof(loginInfo.sDeviceAddress));
    loginInfo.wPort = config.port;
    strncpy(loginInfo.sUserName, config.user.c_str(), sizeof(loginInfo.sUserName));
    strncpy(loginInfo.sPassword, config.pass.c_str(), sizeof(loginInfo.sPassword));

    NET_DVR_DEVICEINFO_V40 deviceInfo = {0};
    this->userID = NET_DVR_Login_V40(&loginInfo, &deviceInfo);
    return this->userID >= 0;
}

bool HikCamera::startRealPlay()
{
    NET_DVR_PREVIEWINFO playInfo = {0};
    playInfo.lChannel = 1;
    playInfo.dwStreamType = 0; // 主码流
    playInfo.dwLinkMode = 0;   // TCP
    playInfo.bBlocked = 1;
    playInfo.hPlayWnd = 0;     // 无播放窗口，只取码流

    this->realHandle = NET_DVR_RealPlay_V40(this->userID, &playInfo, RealDataCallBack, this);
    return this->realHandle >= 0;
}

void HikCamera::stopRealPlay()
{
    if (this->realHandle >= 0) NET_DVR_StopRealPlay(this->realHandle);
    if (this->userID >= 0) NET_DVR_Logout(this->userID);
    NET_DVR_Cleanup();
}