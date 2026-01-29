#include <iostream>
#include <fstream>
#include <cstdio>
#include <unistd.h>

#include "HCNetSDK.h"
#include "PlogInitializer.h"
#include "Config.h"

int main() {
    // --- 步骤 1: 初始化 Plog 日志 ---
    PlogInitializer::getInstance().init(plog::verbose);
    PLOGI << "Integration Test Started...";

    // --- 步骤 2: 使用 Cereal 从 JSON 加载相机参数 ---
    HikConfig cfg;
    try {
        std::ifstream is("/home/sunrise/Desktop/RDKS100_Drowning/configs/test_configs.json");
        if (!is.is_open()) {
            PLOGF << "Cannot find config.json! Please create it first.";
            return -1;
        }
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp("HikCamera", cfg));
        PLOGI << "Config loaded: IP=" << cfg.ip << ", User=" << cfg.user;
    } catch (const std::exception& e) {
        PLOGE << "Cereal load error: " << e.what();
        return -1;
    }

    // --- 步骤 3: 初始化 SDK ---
    if (!NET_DVR_Init()) {
        PLOGE << "SDK Init Failed, Error Code: " << NET_DVR_GetLastError();
        return -1;
    }
    PLOGD << "SDK Initialized successfully.";

    // --- 步骤 4: 登录设备 (使用从配置读取的变量) ---
    NET_DVR_USER_LOGIN_INFO loginInfo = {0};
    NET_DVR_DEVICEINFO_V40 deviceInfo = {0};

    // 使用刚才加载的 cfg 对象
    sprintf(loginInfo.sDeviceAddress, "%s", cfg.ip.c_str());
    loginInfo.wPort = cfg.port;
    sprintf(loginInfo.sUserName, "%s", cfg.user.c_str());
    sprintf(loginInfo.sPassword, "%s", cfg.pass.c_str());

    long userId = NET_DVR_Login_V40(&loginInfo, &deviceInfo);

    if (userId < 0) {
        PLOGE << "Login Failed, Error Code: " << NET_DVR_GetLastError();
        NET_DVR_Cleanup();
        return -1;
    }
    PLOGI << "Device Login Success! UserID: " << userId;

    // --- 步骤 5: 启动预览 ---
    NET_DVR_PREVIEWINFO previewInfo = {0};
    previewInfo.lChannel = 1;
    previewInfo.dwStreamType = 0;
    previewInfo.dwLinkMode = 0;
    previewInfo.bBlocked = 1;

    long realHandle = NET_DVR_RealPlay_V40(userId, &previewInfo, NULL, NULL);

    if (realHandle < 0) {
        PLOGE << "RealPlay Failed, Error Code: " << NET_DVR_GetLastError();
    } else {
        PLOGI << "Stream Started! Handle: " << realHandle;
        PLOGI << "Testing stream for 5 seconds...";
        sleep(5); 
        NET_DVR_StopRealPlay(realHandle);
        PLOGD << "Stream stopped.";
    }

    // --- 步骤 6: 注销与释放 ---
    NET_DVR_Logout(userId);
    NET_DVR_Cleanup();
    PLOGI << "Test Finished Successfully.";

    return 0;
}