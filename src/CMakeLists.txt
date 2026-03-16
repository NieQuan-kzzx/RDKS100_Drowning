# 1. 添加子目录以加载 base 和 algorithm 目标
add_subdirectory(base)
add_subdirectory(algorithm)
add_subdirectory(ui)
add_subdirectory(common)
add_subdirectory(sensor)
add_subdirectory(Inference)
add_subdirectory(Logic)
add_subdirectory(Worker)

# 2. 定义可执行程序
add_executable(RDKS100_Drowning 
    main.cc
)

# 3. 配置头文件搜索路径
target_include_directories(RDKS100_Drowning 
PRIVATE
    ${HOBOT_INCLUDE_DIRS}
)

# 4. 链接自定义模块、公共工具库和地平线库
target_link_libraries(RDKS100_Drowning 
PRIVATE          
    lib_algorithm
    lib_base
    lib_common
    lib_sensor
    lib_ui
    lib_inference
    lib_logic
    lib_worker
    hobot_utils
    ${HOBOT_LINK_LIBS}
    ${OpenCV_LIBS}
    ${HIKSDK_LIBS}
)