# 1. 搜集所有源文件 (保持不变)
file(GLOB_RECURSE ALGO_SRCS 
    "${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo_pose/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo_seg/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/SAHI/src/*.cc"
)

# 2. 生成算法动态库
add_library(lib_algorithm SHARED ${ALGO_SRCS})

# 3. 配置头文件搜索路径
# 建议将路径改为 PUBLIC，这样链接 lib_algorithm 的 target (如 main 或 test) 
# 会自动获得这些 include 路径，不需要在其他地方重复写了。
target_include_directories(lib_algorithm
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo_pose/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo_seg/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/SAHI/include"
    ${Eigen3_INCLUDE_DIRS}
    ${HOBOT_INCLUDE_DIRS}
)

# 4. 链接依赖库
target_link_libraries(lib_algorithm 
PUBLIC
    ${OpenCV_LIBS} 
    Eigen3::Eigen
    hobot_utils
PRIVATE
    ${HOBOT_LINK_LIBS} 
    OpenMP::OpenMP_CXX
)