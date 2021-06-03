cmake_minimum_required (VERSION 3.0.2)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ViennaCL REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED core cudaarithm)


enable_language(CUDA)

set(target_name "cuda_support")

#target_include_directories(${target_name} PUBLIC
#        "/usr/local/cuda/include"
#        )

add_library(${target_name}
        cuda_support.cu
        )

target_link_libraries(${target_name}
        gflags
        glog::glog
        opencv_core
        opencv_cudaarithm
        )



SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")
add_compile_definitions(EIGEN_USE_MKL_ALL ON)

set(target_name "basic")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

#add_compile_definitions(EIGEN_USE_MKL ON)
#add_compile_definitions(MKL_DIRECT_CALL ON)
add_compile_options(-m64)
add_library(${target_name}
        activer.cc
        basic.cc
        common.cc
        constructor.cc
        matrix_utils.cc
        neuron.cc
        optimizer.cc
        )

include_directories(${target_name} PUBLIC
        "/opt/intel/mkl/include"
        )


set(mkl_libs
        #1

        #2
#                    /opt/intel/mkl/lib/intel64/libmkl_rt.so

        #3
#                    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
#                    /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.so
#                    /opt/intel/mkl/lib/intel64/libmkl_core.so
#                    /usr/lib/x86_64-linux-gnu/libgomp.so.1
#                /opt/intel/mkl/lib/intel64/libmkl_scalapack_lp64.so
#                /opt/intel/mkl/lib/intel64/libmkl_cdft_core.so
#                /opt/intel/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.so


        #4
                    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
                    /opt/intel/mkl/lib/intel64/libmkl_sequential.so
                    /opt/intel/mkl/lib/intel64/libmkl_core.so
        /opt/intel/mkl/lib/intel64/libmkl_scalapack_lp64.so
        /opt/intel/mkl/lib/intel64/libmkl_cdft_core.so
        /opt/intel/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.so

        #5
#                    /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
#                    /opt/intel/mkl/lib/intel64/libmkl_intel_thread.so
#                    /opt/intel/mkl/lib/intel64/libmkl_core.so
#                    /opt/intel/lib/intel64_lin/libiomp5.so

        #6
#        /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
#        /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.so
#        /opt/intel/mkl/lib/intel64/libmkl_core.so
#        /opt/intel/mkl/lib/intel64/libmkl_scalapack_lp64.so
#        /opt/intel/mkl/lib/intel64/libmkl_cdft_core.so
#        /opt/intel/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.so
#        /usr/lib/x86_64-linux-gnu/libtbb.so
        )

message(STATUS "mkl_libs ${mkl_libs}")

target_link_libraries(${target_name}
        gflags
        glog::glog
        opencv_core
        opencv_cudaarithm
        ${mkl_libs}
        cuda_support
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
if(true AND benchmark_FOUND)
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

