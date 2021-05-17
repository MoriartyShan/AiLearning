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

template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1 / (std::exp(-data[i]) + 1.0);
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

template<typename T>
void ELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] <= 0) {
      data[i] = std::exp(data[i]) - 1;
    }
  }
}

void ELU(cv::Mat &matrix) {
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
    }
  }
}

void derivativesELU(cv::Mat &matrix) {
  if (matrix.type() == CV_32FC1) {
    derivativesELU<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    derivativesELU<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}
}//namespace AiLearning
