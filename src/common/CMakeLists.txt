add_library(lib_common 
SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/ImageSaver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/MatPool.cpp
)

target_include_directories(lib_common 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${3DPARTY_INCLUDE_DIRS}
)

target_link_libraries(lib_common
PRIVATE 
    ${OpenCV_LIBRARIES}
)