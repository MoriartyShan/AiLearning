//
// Created by moriarty on 2021/5/16.
//
#include "common.h"
#include <glog/logging.h>

#define RAND8BIT() (((uint32_t)rand()) & 0xff)
#define RAND32BIT() (RAND8BIT() | (RAND8BIT() << 8) | (RAND8BIT() << 16) | (RAND8BIT() << 24) )
#define RANDOM(from, to) (((RAND32BIT() / (double)0xffffffff)) * (to - from) + (from))


namespace cv {
template<typename T>
double min(T *data, const int size) {
  T min = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] < min) {
      min = data[i];
    }
  }
  return min;
}

double min(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    return min<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    return min<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
  return -1;
}

template<typename T>
double max(T *data, const int size) {
  T max = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] > max) {
      max = data[i];
    }
  }
  return max;
}

double max(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    return max<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    return max<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
  return -1;
}

}

namespace AiLearning {
void Random(cv::Mat &matrix) {
  CHECK(matrix.type() == CV_TYPE);
#if 1
  cv::randu(matrix, 0, 0.9999);
//  LOG(ERROR) << matrix.at<scalar>(0, 0);
//  for (int i = 0; i < matrix.rows; i++) {
//    for (int j = 0; j < matrix.cols; j++) {
//      CHECK(matrix.at<scalar>(i, j) > -0.9999 && matrix.at<scalar>(i, j) < 0.9999) << matrix.at<scalar>(i, j);
//    }
//  }
#else
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      matrix.at<scalar>(i, j) = RANDOM(-0.999, 0.999);
    }
  }
#endif
}

template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1 / (std::exp(-data[i]) + 1.0);
  }
}

void Sigmoid(cv::Mat &matrix) {
  cv::Mat tmp = -matrix;
  cv::exp(tmp, matrix);
  matrix = 1 / (matrix + 1);
}

void derivativesSigmoid(cv::Mat &matrix) {
  matrix = matrix.mul(1 - matrix);
}

#define ELU_COEF 1.0
template<typename T>
void ELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] <= 0) {
      data[i] = ELU_COEF * (std::exp(data[i]) - 1);
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
      data[i] = data[i] + ELU_COEF;
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

template<typename T>
void derivativesRELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] >= 0) {
      data[i] = 1;
    } else {
      data[i] = 0;
    }
  }
}


void derivativesRELU(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    derivativesRELU<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    derivativesRELU<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}

template<typename T>
void RELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] < 0) {
      data[i] = 0;
    }
  }
}

void RELU(cv::Mat &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    RELU<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    RELU<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}

void derivativesSoftmax(cv::Mat &matrix) {
  matrix = matrix - matrix.mul(matrix);
  return;
}

void Softmax(cv::Mat &matrix) {
  scalar max = cv::max(matrix);
  scalar diff = (max > 500) ? (max - 500) : 0;
  cv::Mat exp;
  matrix = matrix - diff;
  cv::exp(matrix, exp);
  CHECK(exp.channels() == 1) << exp.channels();
  scalar sum = cv::sum(exp)(0);
  matrix = exp / sum;
  return;
}

}//namespace AiLearning
