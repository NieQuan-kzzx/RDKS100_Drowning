set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

add_library(lib_worker
SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/DetectionCoordinator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VideoCaptureManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AIInferenceManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RecordingManager.cpp
)

target_include_directories(lib_worker
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${3DPARTY_INCLUDE_DIRS}
)

target_link_libraries(lib_worker
PRIVATE 
    ${OpenCV_LIBRARIES}
    lib_common
    lib_inference
    lib_sensor
    lib_logic
    lib_base
    lib_algorithm
PUBLIC
    Qt::Widgets
)