cmake_minimum_required (VERSION 3.0.2)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(glog REQUIRED)
find_package(ViennaCL REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED core)

set(LINK_LIBS "")

OPTION(EIGEN_MODE "matrix operation use eigen" OFF)
OPTION(OPENCV_CUDA_MODE "matrix operation use cuda" OFF)
OPTION(OPENCV_CPU_MODE "matrix operation use opencv" OFF)

if(EIGEN_MODE)
    message(STATUS "matrix mode EIGEN_MODE")

    find_package(Eigen3 REQUIRED)
#    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")
    add_compile_definitions(EIGEN_USE_MKL_ALL ON)
    add_compile_definitions(EIGEN_MODE ON)
    add_compile_options(-m64)
    #mkl support
    include_directories(${target_name} PUBLIC
            "/opt/intel/mkl/include"
            )

    set(LINK_LIBS
            /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
            /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.so
            /opt/intel/mkl/lib/intel64/libmkl_core.so
            /usr/lib/x86_64-linux-gnu/libgomp.so.1
            /opt/intel/mkl/lib/intel64/libmkl_scalapack_lp64.so
            /opt/intel/mkl/lib/intel64/libmkl_cdft_core.so
            /opt/intel/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.so
            )
elseif(OPENCV_CUDA_MODE)
    message(STATUS "matrix mode OPENCV_CUDA_MODE")
    enable_language(CUDA)
    set(target_name "cuda_support")

    find_package(OpenCV REQUIRED cudaarithm)
    add_compile_definitions(OPENCV_CUDA_MODE ON)
    add_library(${target_name}
            cuda_support.cu
            )

    target_link_libraries(${target_name}
            gflags
            glog::glog
            opencv_core
            opencv_cudaarithm
            )
    set(LINK_LIBS  cuda_support)
elseif(OPENCV_CPU_MODE)
    message(STATUS "matrix mode OPENCV_CPU_MODE")
    add_compile_definitions(OPENCV_CPU_MODE ON)
else()
    message(FATAL_ERROR "should appoint a option:EIGEN_MODE/OPENCV_CUDA_MODE/OPENCV_CPU_MODE")
endif()


set(target_name "basic")

add_library(${target_name}
        activer.cc
        basic.cc
        common.cc
        constructor.cc
        matrix_utils.cc
        neuron.cc
        optimizer.cc
        )

message(STATUS "Linking libs ${LINK_LIBS}")
target_link_libraries(${target_name}
        gflags
        glog::glog
        opencv_core
        ${LINK_LIBS}
        )

set(target_name "number")
add_executable(${target_name}
        number.cc)

target_link_libraries(${target_name}
        basic
)


set(target_name "analyse")
add_executable(${target_name}
        analyse.cc)

target_link_libraries(${target_name}
        basic
        )

find_package(benchmark)
if(false AND benchmark_FOUND)
    set(target_name "benchmark_test")

    add_executable(${target_name}
            benchmark_test.cc)

    add_compile_definitions(VIENNACL_WITH_OPENMP ON)

    message(STATUS "complier ${CMAKE_CXX_COMPILER_ID}")
    target_include_directories(${target_name} PUBLIC
            ${VIENNACL_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            "/opt/intel/mkl/include"
            )

    target_link_libraries(${target_name}
            benchmark::benchmark
            basic
            )
else()
    message(STATUS "benchmark not found")
endif()

set(target_name "gpu_cpu_compare")
add_executable(${target_name}
        gpu_cpu_compare.cc)
#add_compile_definitions(EIGEN_USE_MKL_ALL ON)
target_link_libraries(${target_name}
        opencv_cudaarithm
        glog::glog
        /opt/intel/mkl/lib/intel64/libmkl_rt.so
        cuda_support
        basic
#        libmkl_sequential.so
        )

