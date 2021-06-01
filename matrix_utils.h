//
// Created by moriarty on 6/1/21.
//

#ifndef NEURALNETWORK_MATRIX_UTILS_H
#define NEURALNETWORK_MATRIX_UTILS_H
#include "common.h"

namespace AiLearning {
namespace MatrixUtils {
//from opencv
enum GemmFlags {
  GEMM_1_T = 1, //!< transposes src1
  GEMM_2_T = 2, //!< transposes src2
  GEMM_3_T = 4 //!< transposes src3
};

/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
inline void gemm(const Matrix &src1, const Matrix &src2, double alpha,
          const Matrix &src3, double beta, Matrix &dst, int flags) {
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

}//namespace MatrixUtils
}//namespace AiLearning

#endif //NEURALNETWORK_MATRIX_UTILS_H
