//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

namespace AiLearning {
#define GPU_MODE

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
#define cu_multiply(src1, src2, dst) cv::cuda::multiply(src1, src2, dst, 1, -1, cu_stream)
#else
#endif

void Sigmoid(Matrix &matrix);
void Sigmoid(cv::Mat &matrix);
void derivativesSigmoid(Matrix &matrix);

void ELU(Matrix &matrix);
void derivativesELU(Matrix &matrix);

void Softmax(Matrix &matrix);
void derivativesSoftmax(Matrix &matrix);

void RELU(Matrix &matrix);
void derivativesRELU(Matrix &matrix);

void Tanh(Matrix &matrix);
void derivateTanh(Matrix &matrix);

void Random(cv::cuda::GpuMat &matrix);
void Random(cv::Mat &matrix);
bool check(const Matrix &matrix);

}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
