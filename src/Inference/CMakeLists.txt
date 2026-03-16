add_library(lib_inference
SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/Yolo11Infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/YoloPose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Patchcore.cpp
)

target_include_directories(lib_inference
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${3DPARTY_INCLUDE_DIRS}
)

target_link_libraries(lib_inference
PRIVATE 
    ${OpenCV_LIBRARIES}
    lib_algorithm
    lib_common
    lib_sensor
    lib_base
)