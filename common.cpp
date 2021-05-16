//
// Created by moriarty on 2021/5/16.
//
#include "common.h"
#include <glog/logging.h>

#define RAND8BIT() (((uint32_t)rand()) & 0xff)
#define RAND32BIT() (RAND8BIT() | (RAND8BIT() << 8) | (RAND8BIT() << 16) | (RAND8BIT() << 24) )
#define RANDOM(a) (((RAND32BIT() / (double)0xffffffff) - 0.5) * 2 * (a))


namespace AiLearning {
void Random(cv::Mat &matrix) {
  CHECK(matrix.type() == CV_32FC1);
  float max = -1;
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      matrix.at<float>(i, j) = RANDOM(1);
      if (matrix.at<float>(i, j) > max) {
        max = matrix.at<float>(i, j);
      }
    }
  }
}

void Sigmoid(cv::Mat &matrix) {
  if (matrix.type() == CV_32FC1) {
    Sigmoid<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    Sigmoid<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}
}//namespace AiLearning
