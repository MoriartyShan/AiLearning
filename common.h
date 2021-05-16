//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_COMMON_H
#define NEURALNETWORK_COMMON_H
#include <opencv2/opencv.hpp>

namespace AiLearning {
template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1 / (std::exp(-data[i]) + 1.0);
  }
}

void Sigmoid(cv::Mat &matrix);

void Random(cv::Mat &matrix);
}//namespace AiLearning
#endif //NEURALNETWORK_COMMON_H
