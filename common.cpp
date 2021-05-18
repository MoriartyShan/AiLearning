//
// Created by moriarty on 2021/5/16.
//
#include "common.h"
#include <glog/logging.h>

#define RAND8BIT() (((uint32_t)rand()) & 0xff)
#define RAND32BIT() (RAND8BIT() | (RAND8BIT() << 8) | (RAND8BIT() << 16) | (RAND8BIT() << 24) )
#define RANDOM(from, to) (((RAND32BIT() / (double)0xffffffff)) * (to - from) + (from))


namespace AiLearning {
void Random(cv::Mat &matrix) {
  CHECK(matrix.type() == CV_TYPE);
  float max = -1;
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      matrix.at<scalar>(i, j) = RANDOM(-0.999, 0.999);
//      LOG(ERROR) << matrix.at<float>(i, j);
      if (matrix.at<scalar>(i, j) > max) {
        max = matrix.at<scalar>(i, j);
      }
    }
  }
}

template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1 / (std::exp(-data[i]) + 1.0);
  }
}

void Sigmoid(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    Sigmoid<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    Sigmoid<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}

template<typename T>
void ELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] <= 0) {
      T d = data[i];
      data[i] = std::exp(data[i]) - 1;
      //LOG(ERROR) << "d[" << i << "]" << d << "," << data[i];
    }
  }
}

void ELU(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    ELU<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    ELU<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}


template<typename T>
void derivativesELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] > 0) {
      data[i] = 1;
    } else {
      data[i] = data[i] + 1;
//      CHECK(data[i] > 0) << data[i] << "," << i;
    }
  }
}

void derivativesELU(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    derivativesELU<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    derivativesELU<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}


void derivativesSoftmax(cv::Mat &matrix) {
  matrix = matrix - matrix.mul(matrix);
  return;
}

scalar Softmax(cv::Mat &matrix) {
  cv::Mat exp;
  cv::exp(matrix, exp);
  CHECK(exp.channels() == 1) << exp.channels();
  scalar sum = cv::sum(exp)(0);
  matrix = exp / sum;
  return sum;
}

}//namespace AiLearning
