//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
//#define OPENCV_CUDA_MODE


#ifdef OPENCV_CUDA_MODE
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#elif defined(OPENCV_CPU_MODE)
#include <opencv2/opencv.hpp>
#elif defined(EIGEN_MODE)
#include <Eigen/Core>
#else
#error "You must specify one mode"
#endif
#include <Eigen/Dense>

namespace AiLearning {

#if 0
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
#else

#endif

using InputMatrix = const Matrix&;
using OutputMatrix = Matrix&;

void Sigmoid(cv::Mat &matrix);

void ELU(Matrix &matrix);
void derivativesELU(Matrix &matrix);

void Softmax(Matrix &matrix);
void derivativesSoftmax(Matrix &matrix);

void RELU(Matrix &matrix);
void derivativesRELU(Matrix &matrix);

void Tanh(Matrix &matrix);
void derivateTanh(Matrix &matrix);
#ifdef OPENCV_CUDA_MODE
void Random(cv::cuda::GpuMat &matrix);
#endif
void Random(cv::Mat &matrix);
bool check(const Matrix &matrix);

}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
