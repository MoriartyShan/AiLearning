//
// Created by moriarty on 6/1/21.
//

#ifndef NEURALNETWORK_MATRIX_UTILS_H
#define NEURALNETWORK_MATRIX_UTILS_H
#include "common.h"
#include <glog/logging.h>
#include <typeinfo>

namespace AiLearning {
namespace MatrixUtils {
//from opencv
enum GemmFlags {
  GEMM_1_T = 1, //!< transposes src1
  GEMM_2_T = 2, //!< transposes src2
  GEMM_3_T = 4 //!< transposes src3
};
#ifdef OPENCV_CUDA_MODE
/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
inline void gemm(InputMatrix src1, InputMatrix src2, double alpha,
          InputMatrix src3, double beta, Matrix &dst, int flags) {
//  LOG(ERROR) << "using matrix";
  cv::cuda::gemm(src1, src2, alpha, src3, beta,
                 dst, flags, cu_stream);
  return;
}


/*
 * @dst = @src1 + @src2;
 * _Type1 and _Type2 is matrix or scalar
 * */
template<typename _Type1, typename _Type2>
inline void add(_Type1 src1, _Type2 src2, Matrix &dst) {
  cv::cuda::add(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
inline void addWeighted(
  InputMatrix src1, double alpha, InputMatrix src2,
  double beta, double gamma, Matrix &dst) {
  cv::cuda::addWeighted(src1, alpha, src2, beta, gamma, dst, -1, cu_stream);
  return;
}

/*
 * @dst = @src1 - @src2;
 * */
template<typename _Type1, typename _Type2>
inline void subtract(_Type1 src1, _Type2 src2, Matrix &dst) {
  cv::cuda::subtract(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}

/* element wise
 * @dst = @src1.mul(@src2);
 * */
template<typename _Type1, typename _Type2>
inline void multiply(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  cv::cuda::multiply(src1, src2, dst, 1, -1, cu_stream);
  return;
}

/*
 * @dst = @src1 / @src2;
 * */
template<typename _Type1, typename _Type2>
inline void divide(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  cv::cuda::divide(src1, src2, dst, 1, -1, cu_stream);
  return;
}



inline void sqrt(InputMatrix src, OutputMatrix dst) {
  cv::cuda::sqrt(src, dst, cu_stream);
  return;
}

inline void exp(InputMatrix src, OutputMatrix dst) {
  cv::cuda::exp(src, dst, cu_stream);
  return;
}

inline double sum(InputMatrix src) {
  return cv::cuda::sum(src)(0);
}

inline double norml2(InputMatrix src1) {
  return cv::cuda::norm(src1, cv::NORM_L2);
}
#elif defined(OPENCV_CPU_MODE)
/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
inline void gemm(InputMatrix src1, InputMatrix src2, double alpha,
                 InputMatrix src3, double beta, Matrix &dst, int flags) {
//  LOG(ERROR) << "using matrix";
#if 0
  const Matrix& _src1 = ((flags & GemmFlags::GEMM_1_T) == 0) ? src1 : src1.t();
  const Matrix& _src2 = ((flags & GemmFlags::GEMM_2_T) == 0) ? src2 : src2.t();
  const Matrix& _src3 = ((flags & GemmFlags::GEMM_3_T) == 0) ? src3 : src3.t();
//  LOG(ERROR) << _src1.size() << "," << _src2.size() << "," << _src3.size();
//  LOG(ERROR) << src1.size() << "," << src2.size() << "," << src3.size();

  if (src3.empty() || std::abs(beta) < std::numeric_limits<scalar>::epsilon()) {
    dst = alpha * _src1 * _src2;
  } else {
    dst = alpha * _src1 * _src2 + beta * src3;
  }

//  LOG(ERROR) << src1.size() << "," << src2.size() << "," << src3.size();

#else
  cv::gemm(src1, src2, alpha, src3, beta, dst, flags);
#endif
  return;
}


/*
 * @dst = @src1 + @src2;
 * _Type1 and _Type2 is matrix or scalar
 * */
template<typename _Type1, typename _Type2>
inline void add(_Type1 src1, _Type2 src2, Matrix &dst) {
  dst = src1 + src2;
  return;
}

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
inline void addWeighted(
    InputMatrix src1, double alpha, InputMatrix src2,
    double beta, double gamma, OutputMatrix dst) {
  dst = src1 * alpha + src2 * beta + gamma;
  return;
}

/*
 * @dst = @src1 - @src2;
 * */
template<typename _Type1, typename _Type2>
inline void subtract(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  dst = src1 - src2;
  return;
}

/* element wise
 * @dst = @src1.mul(@src2);
 * */
template<typename _Type1, typename _Type2>
void multiply(const _Type1 src1, const _Type2 src2, OutputMatrix dst) {
  cv::multiply(src1, src2, dst);
  return;
}



/*
 * @dst = @src1 / @src2;
 * */
template<typename _Type1, typename _Type2>
void divide(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  dst = src1 / src2;
  return;
}



inline void sqrt(InputMatrix src, OutputMatrix dst) {
  cv::sqrt(src, dst);
  return;
}

inline void exp(InputMatrix src, OutputMatrix dst) {
  cv::exp(src, dst);
  return;
}

inline double sum(InputMatrix src) {
  return cv::sum(src)(0);
}

inline double norml2(InputMatrix src1) {
  return cv::norm(src1, cv::NORM_L2);
}
#elif defined(EIGEN_MODE)
/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
inline void gemm(InputMatrix src1, InputMatrix src2, double alpha,
                 InputMatrix src3, double beta, OutputMatrix dst, int flags) {
//  LOG(ERROR) << "using matrix";
  InputMatrix _src1 = ((flags & GemmFlags::GEMM_1_T) == 0) ? src1 : src1.transpose();
  InputMatrix _src2 = ((flags & GemmFlags::GEMM_2_T) == 0) ? src2 : src2.transpose();
  InputMatrix _src3 = ((flags & GemmFlags::GEMM_3_T) == 0) ? src3 : src3.transpose();
//  LOG(ERROR) << _src1.size() << "," << _src2.size() << "," << _src3.size();
//  LOG(ERROR) << src1.size() << "," << src2.size() << "," << src3.size();

  if ((_src3.size() == 0) || std::abs(beta) < std::numeric_limits<scalar>::epsilon()) {
    dst = alpha * _src1 * _src2;
  } else {
    dst = alpha * _src1 * _src2 + beta * src3;
  }
  return;
}


/*
 * @dst = @src1 + @src2;
 * _Type1 and _Type2 is matrix or scalar
 * */
template<typename _Type1, typename _Type2>
inline void add(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  dst = src1 + src2;
  return;
}

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
inline void addWeighted(
    InputMatrix src1, double alpha, InputMatrix src2,
    double beta, double gamma, OutputMatrix dst) {
  dst = (src1 * alpha + src2 * beta).array() + gamma;
  return;
}

/*
 * @dst = @src1 - @src2;
 * */
template<typename _Type1, typename _Type2>
inline void subtract(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  dst = src1 - src2;
  return;
}

/* element wise
 * @dst = @src1.mul(@src2);
 * */
template<typename _Type1, typename _Type2>
inline void multiply(_Type1 src1, _Type2 src2, OutputMatrix dst) {
  dst = src1 * src2;
  return;
}

template<>
inline void multiply<InputMatrix, InputMatrix>(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1.array() * src2.array();
  return;
}

/*
 * @dst = @src1 / @src2;
 * */
inline void divide(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1.array() / src2;
  return;
}

inline void divide(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 / src2.array();
  return;
}

inline void divide(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1.array() / src2.array();
  return;
}




inline void sqrt(InputMatrix src, OutputMatrix dst) {
  dst = src.array().sqrt();
  return;
}

inline void exp(InputMatrix src, OutputMatrix dst) {
  dst = src.array().exp();
  return;
}

inline double sum(InputMatrix src) {
  return src.sum();
}

inline double norml2(InputMatrix src1) {
  return src1.norm();
}
#else
#error "You must specify one mode"
#endif
}//namespace MatrixUtils
}//namespace AiLearning

#endif //NEURALNETWORK_MATRIX_UTILS_H
