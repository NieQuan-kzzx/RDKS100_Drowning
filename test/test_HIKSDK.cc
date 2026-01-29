#include <iostream>
#include <cstdio>
#include <unistd.h>
#include "HCNetSDK.h"
#include "PlogInitializer.h"

int main() {
    PlogInitializer::getInstance().init(plog::verbose);
    
    if (!NET_DVR_Init()) {
        PLOGF << "SDK 初始化失败，错误码: " << NET_DVR_GetLastError();
        return -1;
    }

    // --- 登录部分保持不变 ---
    NET_DVR_USER_LOGIN_INFO loginInfo = {0};
    NET_DVR_DEVICEINFO_V40 deviceInfo = {0};
    sprintf(loginInfo.sDeviceAddress, "192.168.127.15");
    loginInfo.wPort = 8000;
    sprintf(loginInfo.sUserName, "admin");
    sprintf(loginInfo.sPassword, "waterline123456");

    long userId = NET_DVR_Login_V40(&loginInfo, &deviceInfo);
    if (userId < 0) {
        PLOGE << "登录失败，错误码: " << NET_DVR_GetLastError();
        NET_DVR_Cleanup();
        return -1;
    }
    PLOGI << "设备登录成功！UserID: " << userId;

    // --- 核心测试：启动预览并保存原始流 ---
    NET_DVR_PREVIEWINFO previewInfo = {0};
    previewInfo.lChannel = 1;
    previewInfo.dwStreamType = 0; 
    previewInfo.dwLinkMode = 0;
    previewInfo.bBlocked = 1;

    long realHandle = NET_DVR_RealPlay_V40(userId, &previewInfo, NULL, NULL);

    if (realHandle < 0) {
        PLOGE << "启动预览失败，错误码: " << NET_DVR_GetLastError();
    } else {
        PLOGI << "预览流通道已建立！Handle: " << realHandle;

        // 【关键点】尝试将流直接保存到本地文件
        // 如果这个函数返回 true 且文件大小在增长，说明 SDK 确实收到了数据
        if (NET_DVR_SaveRealData(realHandle, (char*)"./test_stream.mp4")) {
            PLOGI << "正在将原始流保存到 ./test_stream.mp4，持续 5 秒...";
            sleep(5); 
            NET_DVR_StopSaveRealData(realHandle);
            PLOGI << "保存结束，请检查文件大小。";
        } else {
            PLOGE << "流保存失败，错误码: " << NET_DVR_GetLastError();
        }

        NET_DVR_StopRealPlay(realHandle);
    }

    NET_DVR_Logout(userId);
    NET_DVR_Cleanup();
    PLOGI << "测试结束。";
    return 0;
}

// #include <iostream>
// #include <cstdio>
// #include <unistd.h>
// #include "HCNetSDK.h"
// #include "PlogInitializer.h"

// int main() {
//     PlogInitializer::getInstance().init(plog::verbose);
//     // 1. 初始化 SDK
//     if (!NET_DVR_Init()) {
//         std::cerr << "SDK 初始化失败，错误码: " << NET_DVR_GetLastError() << std::endl;
//         return -1;
//     }
//     PLOGD << "This is a DEBUG message (Level 2)";
//     // 2. 登录设备
//     NET_DVR_USER_LOGIN_INFO loginInfo = {0};
//     NET_DVR_DEVICEINFO_V40 deviceInfo = {0};

//     // --- 请根据你的实际情况修改以下参数 ---
//     sprintf(loginInfo.sDeviceAddress, "192.168.127.15"); // 摄像头 IP
//     loginInfo.wPort = 8000;                           // SDK 默认端口
//     sprintf(loginInfo.sUserName, "admin");            // 用户名
//     sprintf(loginInfo.sPassword, "waterline123456");      // 密码

//     long userId = NET_DVR_Login_V40(&loginInfo, &deviceInfo);

//     if (userId < 0) {
//         std::cerr << "登录失败，错误码: " << NET_DVR_GetLastError() << std::endl;
//         NET_DVR_Cleanup();
//         return -1;
//     }
//     std::cout << "设备登录成功！UserID: " << userId << std::endl;

//     // 3. 启动预览 (这里仅做启动测试，不显示窗口)
//     NET_DVR_PREVIEWINFO previewInfo = {0};
//     previewInfo.lChannel = 1;        // 通道号，通常从 1 开始
//     previewInfo.dwStreamType = 0;    // 0-主码流，1-子码流
//     previewInfo.dwLinkMode = 0;      // TCP 方式
//     previewInfo.bBlocked = 1;        // 阻塞模式

//     long realHandle = NET_DVR_RealPlay_V40(userId, &previewInfo, NULL, NULL);

//     if (realHandle < 0) {
//         std::cerr << "启动预览失败，错误码: " << NET_DVR_GetLastError() << std::endl;
//     } else {
//         std::cout << "预览流启动成功！Handle: " << realHandle << std::endl;
//         std::cout << "测试持续 5 秒后退出..." << std::endl;
//         sleep(5); // 维持预览 5 秒
//         NET_DVR_StopRealPlay(realHandle);
//     }

//     // 4. 注销与释放
//     NET_DVR_Logout(userId);
//     NET_DVR_Cleanup();
//     std::cout << "测试结束。" << std::endl;

//     return 0;
// }