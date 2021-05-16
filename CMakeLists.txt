cmake_minimum_required (VERSION 3.0.2)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(Eigen3 REQUIRED)

set(target_name "basic")
add_library(${target_name} basic.cpp common.cpp layer.cpp)

target_link_libraries(${target_name}
        gflags
        glog::glog
        ${OpenCV_LIBS}
        )

set(target_name "number")
add_executable(${target_name}
   number.cpp )

target_link_libraries(${target_name}
        basic
)