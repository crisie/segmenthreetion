cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(segmenthreetion)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)

add_executable (segmenthreetion main.cpp)
target_link_libraries (segmenthreetion ${OpenCV_LIBS} ${PCL_LIBRARIES})
