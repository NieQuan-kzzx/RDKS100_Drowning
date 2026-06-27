add_library(lib_logic
SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/DrowningUnderSurface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DrowningState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WaterIngress.cpp
)

target_include_directories(lib_logic
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${3DPARTY_INCLUDE_DIRS}
)


target_link_libraries(lib_logic
PRIVATE 
    ${OpenCV_LIBRARIES}
    lib_common
    lib_inference
    lib_algorithm
)