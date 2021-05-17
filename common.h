//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
#include <opencv2/opencv.hpp>

namespace AiLearning {


void Sigmoid(cv::Mat &matrix);



void ELU(cv::Mat &matrix);
void derivativesELU(cv::Mat &matrix);

void Random(cv::Mat &matrix);
}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
