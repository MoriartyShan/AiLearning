//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
#include <opencv2/opencv.hpp>

namespace AiLearning {
#if 0
using scalar = float;
#define CV_TYPE CV_32FC1
#else
using scalar = double;
#define CV_TYPE CV_64FC1
#endif
void Sigmoid(cv::Mat &matrix);
void derivativesSigmoid(cv::Mat &matrix);

void ELU(cv::Mat &matrix);
void derivativesELU(cv::Mat &matrix);

void Softmax(cv::Mat &matrix);
void derivativesSoftmax(cv::Mat &matrix);


void Random(cv::Mat &matrix);

}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
