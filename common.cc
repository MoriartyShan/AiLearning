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

namespace AiLearning {
#ifdef OPENCV_CUDA_MODE
cv::cuda::Stream cu_stream;
#endif

template<typename T>
void Random(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = RANDOM(-0.999, 0.999);
  }
}


void Random(cv::Mat &matrix) {
  CHECK(matrix.type() == CV_TYPE);
#if 1
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

void SigmoidSingle(double &p) {
  if (p > 0) {
    p = 1 / (std::exp(-p) + 1.0);
  } else {
    double exp = std::exp(p);
    p = exp / (1 + exp);
  }
  return;
}

template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    SigmoidSingle(data[i]);
  }
}

void Sigmoid(cv::Mat &matrix) {
#if 1
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p > 0) {
      p = 1 / (std::exp(-p) + 1.0);
    } else {
      scalar exp = std::exp(p);
      p = exp / (1 + exp);
    }
  });
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

void derivativesSoftmax(Matrix &matrix) {
#ifdef OPENCV_CPU_MODE
  matrix = matrix.mul(1 - matrix);
#elif defined(GPU_MODE)
  static Matrix tmp1, tmp2;
  MatrixUtils::subtract(1, matrix, tmp1);
  MatrixUtils::multiply(matrix, tmp1, tmp2);
  tmp2.copyTo(matrix);
#endif
}

void Softmax(Matrix &matrix) {
#ifdef OPENCV_CPU_MODE
  LOG(FATAL) << "implement";
  scalar max = 1;// cv::max(matrix);
  Matrix exp;
  matrix = matrix - max;
  cv::exp(matrix, exp);
  CHECK(exp.channels() == 1) << exp.channels();
  scalar sum = cv::sum(exp)(0);
  matrix = exp / sum;
#elif defined(OPENCV_CUDA_MODE)
  double max;
  cv::cuda::minMax(matrix, nullptr, &max);
  static Matrix exp, tmp1;

  MatrixUtils::subtract(matrix, max, tmp1);
  MatrixUtils::exp(tmp1, exp);
  scalar sum = MatrixUtils::sum(exp);
  MatrixUtils::divide(exp, sum, matrix);
#elif defined(EIGEN_MODE)
  LOG(ERROR) << "remember to implement this function:" << __func__;
#else
#error "dd"
#endif
  return;
}

void derivateTanh(Matrix &matrix) {
#ifdef OPENCV_CPU_MODE
  matrix = matrix.mul(1 - matrix);
#elif defined(OPENCV_CUDA_MODE)
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

void Tanh(cv::Mat &matrix) {
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
