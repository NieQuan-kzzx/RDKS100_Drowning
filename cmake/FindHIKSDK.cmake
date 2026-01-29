# 设置 SDK 的根目录
set(HIK_NET_SDK_ROOT "/home/sunrise/Desktop/HCNetSDKV6.1.9.45_build20220902_ArmLinux64_ZH")

# 头文件在 incCn
set(HIKSDK_INCLUDE_DIRS "${HIK_NET_SDK_ROOT}/incCn")

# 库文件在 MakeAll
set(HIKSDK_LIB_DIR "${HIK_NET_SDK_ROOT}/MakeAll")

# 需要链接的核心库
list(APPEND HIKSDK_LIBRARIES 
    "${HIKSDK_LIB_DIR}/libhcnetsdk.so"
    "${HIKSDK_LIB_DIR}/libHCCore.so"
    "${HIKSDK_LIB_DIR}/libhpr.so"
    "${HIKSDK_LIB_DIR}/libPlayCtrl.so"
)

# set (HIKSDK_DIR "/opt/MVS")

# set (HIKSDK_INCLUDE_DIRS "${HIKSDK_DIR}/include/")

# set (HIKSDK_LIB_DIR "${HIKSDK_DIR}/lib/aarch64")

# list (APPEND HIKSDK_LIBRARIES 
#     "${HIKSDK_LIB_DIR}/libMvCameraControl.so"
# )