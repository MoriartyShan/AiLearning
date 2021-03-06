//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
#include <opencv2/opencv.hpp>

//#define OPENCV_CUDA_MODE
//#define OPENCV_CPU_MODE
//#define EIGEN_MODE

#ifdef OPENCV_CUDA_MODE
#include <opencv2/cudaarithm.hpp>

inline std::ostream& operator <<(std::ostream& os, const cv::cuda::GpuMat& m) {
  os << cv::Mat(m);
  return os;
}

#elif defined(OPENCV_CPU_MODE)

#elif defined(EIGEN_MODE)
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#else
#error "You must specify one mode"
#endif

#define ACHECK while(false)CHECK

namespace AiLearning {

#if 1
using scalar = float;
#define CV_TYPE CV_32FC1
#else
using scalar = double;
#define CV_TYPE CV_64FC1
#endif


#ifdef OPENCV_CPU_MODE
using Matrix = cv::Mat;
#elif defined(OPENCV_CUDA_MODE)
extern cv::cuda::Stream cu_stream;
using Matrix = cv::cuda::GpuMat;

#elif defined(EIGEN_MODE)
using Matrix = Eigen::Matrix<
  scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
#error "You must specify one mode"
#endif

using InputMatrix = const Matrix&;
using OutputMatrix = Matrix&;
using InputOutputMatrix = Matrix&;

void Sigmoid(cv::Mat &matrix);
void Random(cv::Mat &matrix);

/*
 * convert mat type to CV_TYPE if it is not
 * */
void ConvertType(cv::Mat& mat);


}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
