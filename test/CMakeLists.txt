# 1. 自动搜索 test 目录下所有的 .cc 文件
file(GLOB TEST_FILES "*.cc")

foreach(test_src ${TEST_FILES})
    # 获取文件名（如 test_Pose）
    get_filename_component(test_name ${test_src} NAME_WE)

    # 2. 定义测试可执行文件
    add_executable(${test_name} ${test_src})

    # 3. 配置路径（只需要配置本测试目录特有的，其他的会自动从库里带过来）
    target_link_directories(${test_name} PRIVATE ${HOBOT_LIB_DIRS})

    # 4. 链接依赖
    # 注意：因为 lib_algorithm 现在的 include 是 PUBLIC 的，
    # 链接它之后，test 就不需要再手动写一长串 target_include_directories 了！
    target_link_libraries(${test_name}
        PRIVATE
        lib_algorithm
        lib_base  
        hobot_utils      
        lib_common
        lib_sensor
        ${HOBOT_LINK_LIBS}
        ${OpenCV_LIBS}
        ${HIKSDK_LIBRARIES}
    )

    message(STATUS "Adding test target: ${test_name}")
endforeach()
