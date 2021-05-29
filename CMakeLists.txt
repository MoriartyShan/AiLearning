cmake_minimum_required (VERSION 3.0.2)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED core cudaarithm)
#find_package(Eigen3 REQUIRED)

set(target_name "basic")
add_library(${target_name}
        basic.cpp
        optimizer.cpp
        common.cpp
        neuron.cpp)

target_link_libraries(${target_name}
        gflags
        glog::glog
        opencv_core
        )

set(target_name "number")
add_executable(${target_name}
   number.cpp )

target_link_libraries(${target_name}
        basic
)


set(target_name "analyse")
add_executable(${target_name}
        analyse.cpp )

target_link_libraries(${target_name}
        basic
        )

find_package(benchmark)
if(benchmark_FOUND)
    set(target_name "benchmark_test")
    add_executable(${target_name}
            benchmark_test.cpp)

    target_link_libraries(${target_name}
            benchmark::benchmark
            opencv_cudaarithm
            )
else()
    message(STATUS "benchmark not found")
endif()


