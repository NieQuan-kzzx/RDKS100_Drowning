set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

add_library(lib_ui 
SHARED 
    mainwindow.cpp
    mainwindow.ui
    DetectionWorker.cpp
)

target_include_directories(lib_ui 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lib_ui
PRIVATE
    ${OpenCV_LIBRARIES}
    lib_base
    lib_algorithm
    lib_common
    lib_sensor
    lib_inference
    lib_logic
PUBLIC
    Qt::Widgets
)
