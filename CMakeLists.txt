cmake_minimum_required (VERSION 3.0.2)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ViennaCL REQUIRED)
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
        opencv_cudaarithm
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

#    find_package(CUDA)

    add_executable(${target_name}
            benchmark_test.cpp)

#    add_compile_options(fopenmp)
#    target_compile_options(${target_name} PUBLIC -fopenmp)

#    MESSAGE(STATUS "ViennaCL dir ${VIENNACL_INCLUDE_DIRS}")
#    MESSAGE(STATUS "CUDA dir ${CUDA_INCLUDE_DIRS}")

#    add_compile_definitions(VIENNACL_WITH_CUDA ON)

    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")
    add_compile_definitions(VIENNACL_WITH_OPENMP ON)
    add_compile_definitions(EIGEN_USE_MKL_ALL ON)

    target_include_directories(${target_name} PUBLIC
            ${VIENNACL_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            "/opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/include"
            )

    file(GLOB mkl_libs
            /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64/*.so
            )

#
    target_link_libraries(${target_name}
            benchmark::benchmark
            opencv_cudaarithm
            ${mkl_libs}
            )
else()
    message(STATUS "benchmark not found")
endif()

set(target_name "gpu_cpu_compare")
add_executable(${target_name}
        gpu_cpu_compare.cpp)

target_link_libraries(${target_name}
        opencv_cudaarithm
        glog::glog
        )

