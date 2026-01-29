add_library(lib_sensor 
SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/ImageSensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HikCamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RTSPCamera.cpp
)

target_include_directories(lib_sensor 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${3DPARTY_INCLUDE_DIRS}
)

target_link_libraries(lib_sensor 
PRIVATE
    ${OpenCV_LIBRARIES}
    ${HIKSDK_LIBRARIES}
    lib_base
    lib_common
)