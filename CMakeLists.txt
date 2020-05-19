cmake_minimum_required(VERSION 2.8)

project(ComputerVision)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(
    Lab6
    src/lab6.cpp
)

target_link_libraries(Lab1_rgb ${OpenCV_LIBS})
target_link_libraries(Lab1_hsv ${OpenCV_LIBS})
target_link_libraries(Lab2 ${OpenCV_LIBS})
target_link_libraries(Lab3 ${OpenCV_LIBS})
target_link_libraries(Lab3 Filter)
target_link_libraries(Lab4 ${OpenCV_LIBS})
target_link_libraries(Lab5 ${OpenCV_LIBS})
target_link_libraries(Lab6 ${OpenCV_LIBS})
