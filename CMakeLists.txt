cmake_minimum_required(VERSION 3.10)
project(RDKS100_Drowning)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 1. 查找依赖
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(HIKSDK REQUIRED)

# 2. 定义路径变量
set(HOBOT_INCLUDE_DIRS
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    /app/cdev_demo/bpu/utils/inc
    /usr/include 
)

set(HOBOT_LINK_LIBS dnn hbucp gflags fmt spcdev)
set(HOBOT_LIB_DIRS /usr/hobot/lib /usr/lib)

list(APPEND 3DPARTY_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${HIKSDK_INCLUDE_DIRS}
        /home/sunrise/Desktop/3rd-party/cereal/include
        /home/sunrise/Desktop/3rd-party/plog/include
)

# 告知编译器和链接器全局路径
include_directories(${HOBOT_INCLUDE_DIRS})
link_directories(${HOBOT_LIB_DIRS})

add_library(hobot_utils SHARED
    /app/cdev_demo/bpu/utils/src/common_utils.cc
    /app/cdev_demo/bpu/utils/src/postprocess_utils.cc
    /app/cdev_demo/bpu/utils/src/preprocess_utils.cc
    /app/cdev_demo/bpu/utils/src/multimedia_utils.cc
)

target_link_directories(hobot_utils PRIVATE ${HOBOT_LIB_DIRS})

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)

target_link_libraries(hobot_utils 
    PUBLIC 
    ${OpenCV_LIBS}
    ${HIKSDK_LIBS}
    ${HOBOT_LINK_LIBS}
    PkgConfig::FFMPEG
)

# 3. 进入子目录
add_subdirectory(src)
add_subdirectory(test)