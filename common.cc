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


void Sigmoid(cv::Mat &matrix) {
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p > 0) {
      p = 1 / (std::exp(-p) + 1.0);
    } else {
      scalar exp = std::exp(p);
      p = exp / (1 + exp);
    }
  });
}

/*
 * convert mat type to CV_TYPE if it is not
 * */
void ConvertType(cv::Mat& mat) {
  if (mat.type() != CV_TYPE) {
    mat.convertTo(mat, CV_TYPE);
  }
}



}//namespace AiLearning
