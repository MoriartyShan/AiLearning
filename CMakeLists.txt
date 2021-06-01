cmake_minimum_required (VERSION 3.0.2)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ViennaCL REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED core cudaarithm)

set(target_name "basic")
add_library(${target_name}
        activer.cc
        constructor.cc
        basic.cc
        optimizer.cc
        common.cc
        neuron.cc
        )

target_link_libraries(${target_name}
        gflags
        glog::glog
        opencv_core
        opencv_cudaarithm
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

#    find_package(CUDA)

    add_executable(${target_name}
            benchmark_test.cc)

#    add_compile_options(fopenmp)
#    target_compile_options(${target_name} PUBLIC -fopenmp)

#    MESSAGE(STATUS "ViennaCL dir ${VIENNACL_INCLUDE_DIRS}")
#    MESSAGE(STATUS "CUDA dir ${CUDA_INCLUDE_DIRS}")

#    add_compile_definitions(VIENNACL_WITH_CUDA ON)

#    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")
    add_compile_definitions(VIENNACL_WITH_OPENMP ON)
    add_compile_definitions(EIGEN_USE_MKL_ALL ON)
#    add_compile_definitions(EIGEN_USE_MKL ON)

#    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#        find_package(BLAS)
#        include_directories(${BLAS_INCLUDE_DIRS})
#        set(PROJECT_LIBS ${BLAS_LIBRARIES})
#        add_definitions(-DEIGEN_USE_MKL_ALL)
#        message(STATUS "use mkl")
#    endif()
    message(STATUS "complier ${CMAKE_CXX_COMPILER_ID}")
    target_include_directories(${target_name} PUBLIC
            ${VIENNACL_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            "/opt/intel/mkl/include"
            )

    file(GLOB mkl_libs
            /opt/intel/mkl/lib/intel64/libmkl_rt.so
            )

#
    target_link_libraries(${target_name}
            benchmark::benchmark
            opencv_cudaarithm
            ${mkl_libs}
            glog::glog
            )
else()
    message(STATUS "benchmark not found")
endif()

set(target_name "gpu_cpu_compare")
add_executable(${target_name}
        gpu_cpu_compare.cc)

target_link_libraries(${target_name}
        opencv_cudaarithm
        glog::glog
        )

