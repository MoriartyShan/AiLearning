//
// Created by moriarty on 2021/5/16.
//
#include "common.h"
#include "matrix_utils.h"
#include "timer.h"
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

double min(AiLearning::Matrix &matrix) {
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

double max(AiLearning::Matrix &matrix) {
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
#ifdef GPU_MODE
cv::cuda::Stream cu_stream;
#endif

template<typename T>
void Random(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = RANDOM(-0.999, 0.999);
  }
}

void Random(Matrix &matrix) {
  CHECK(matrix.type() == CV_TYPE);
#ifdef CPU_MODE
  cv::randu(matrix, -0.9999, 0.9999);
#else
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    return Random<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    return Random<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
#endif
}

void Random(cv::Mat &matrix) {
  CHECK(matrix.type() == CV_TYPE);
#ifdef CPU_MODE
  cv::randu(matrix, -0.9999, 0.9999);
#else
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    return Random<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    return Random<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
#endif
}

template<typename T>
bool check(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      LOG(ERROR) << "data[" << i << "] is invalid " << data[i];
      return false;
    }
  }
  return true;
}

bool check(const Matrix &matrix) {
  return  true;
#if 1
  return cv::checkRange(cv::Mat(matrix));
#else
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    return check<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    return check<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
  return false;
#endif
}

template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] > 0) {
      data[i] = 1 / (std::exp(-data[i]) + 1.0);
    } else {
      T exp = std::exp(data[i]);
      data[i] = exp / (1 + exp);
    }
  }
}

void Sigmoid(cv::Mat &matrix) {
#if 0
  Matrix tmp = -matrix;
  cv::exp(tmp, matrix);
  matrix = 1 / (matrix + 1);
#else
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    Sigmoid<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    Sigmoid<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
#endif
}

#define ELU_COEF 1.0
template<typename T>
void ELU(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] <= 0) {
      data[i] = ELU_COEF * (std::exp(data[i]) - 1);
    }
  }
}

void ELU(Matrix &matrix) {
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

void derivativesELU(Matrix &matrix) {
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


void derivativesRELU(Matrix &matrix) {
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

void RELU(Matrix &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    RELU<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    RELU<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}

void derivativesSoftmax(Matrix &matrix) {
#ifdef CPU_MODE
  matrix = matrix.mul(1 - matrix);
#elif defined(GPU_MODE)
  static Matrix tmp1, tmp2;
  MatrixUtils::subtract(1, matrix, tmp1);
  MatrixUtils::multiply(matrix, tmp1, tmp2);
  tmp2.copyTo(matrix);
#endif
}

void Softmax(Matrix &matrix) {
#ifdef CPU_MODE
  scalar max = cv::max(matrix);
  Matrix exp;
  matrix = matrix - max;
  cv::exp(matrix, exp);
  CHECK(exp.channels() == 1) << exp.channels();
  scalar sum = cv::sum(exp)(0);
  matrix = exp / sum;
#else
  scalar max;
  minMax(matrix, nullptr, &max);
  static Matrix exp, tmp1;

  MatrixUtils::subtract(matrix, max, tmp1);
  MatrixUtils::exp(tmp1, exp);
  scalar sum = MatrixUtils::sum(exp);
  MatrixUtils::divide(exp, sum, matrix);
#endif
  return;
}

void derivateTanh(Matrix &matrix) {
#ifdef CPU_MODE
  matrix = matrix.mul(1 - matrix);
#elif defined(GPU_MODE)
  static Matrix tmp1, tmp2;
  MatrixUtils::subtract(1, matrix, tmp1);
  MatrixUtils::multiply(matrix, tmp1, tmp2);
  tmp2.copyTo(matrix);
#endif
}

template<typename T>
void Tanh(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    if (data[i] > 0) {
      T exp = std::exp(-2 * data[i]);
      data[i] = (1 - exp) / (1 + exp);
    } else {
      T exp = std::exp(2 * data[i]);
      data[i] = (exp - 1) / (1 + exp);
    }
  }
}

void Tanh(Matrix &matrix) {
  CHECK(matrix.isContinuous());
  if (matrix.type() == CV_32FC1) {
    Tanh<float>((float *) matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    Tanh<double>((double *) matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}

}//namespace AiLearning
