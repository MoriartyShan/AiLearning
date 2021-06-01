//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
#include <opencv2/opencv.hpp>
#define CPU_MODE


#ifdef GPU_MODE
#include <opencv2/cudaarithm.hpp>
#endif
#include <Eigen/Dense>

namespace AiLearning {

template<class T, class U>
struct IsSameType
{
  enum {result=false};
};

template<class T>
struct IsSameType<T, T>
{
  enum {result=true};
};


#if 0
using scalar = float;
#define CV_TYPE CV_32FC1
#else
using scalar = double;
#define CV_TYPE CV_64FC1
#endif

#ifdef CPU_MODE
using Matrix = cv::Mat;
#elif defined(GPU_MODE)
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
#ifdef GPU_MODE
void Random(cv::cuda::GpuMat &matrix);
#endif
void Random(cv::Mat &matrix);
bool check(const Matrix &matrix);

}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
