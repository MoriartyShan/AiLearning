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

Matrix createMatrix(int rows, int cols, int type);
bool isEmpty(InputMatrix mat);
void Random(Matrix &matrix);
void CopyTo(InputMatrix src, OutputMatrix dst);
void CopyTo(InputMatrix src, cv::Mat& dst);
void CopyTo(const cv::Mat& src, OutputMatrix dst);
void setZeros(InputOutputMatrix mat);
void setTo(InputOutputMatrix mat, double v);
bool check(const Matrix &matrix);

cv::Size MatrixSize(InputMatrix mat);

/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
void gemm(InputMatrix src1, InputMatrix src2, double alpha,
                 InputMatrix src3, double beta, Matrix &dst, int flags);


/*
 * @dst = @src1 + @src2;
 * _Type1 and _Type2 is matrix or scalar
 * */
void add(InputMatrix src1, scalar src2, OutputMatrix dst);
void add(scalar src1, InputMatrix src2, OutputMatrix dst);
void add(InputMatrix src1, InputMatrix src2, OutputMatrix dst);

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
void addWeighted(
  InputMatrix src1, double alpha, InputMatrix src2,
  double beta, double gamma, Matrix &dst);

/*
 * @dst = @src1 - @src2;
 * */
void subtract(InputMatrix src1, scalar src2, OutputMatrix dst);
void subtract(scalar src1, InputMatrix src2, OutputMatrix dst);
void subtract(InputMatrix src1, InputMatrix src2, OutputMatrix dst);

/* element wise
 * @dst = @src1.mul(@src2);
 * */
void multiply(InputMatrix src1, scalar src2, OutputMatrix dst);
void multiply(scalar src1, InputMatrix src2, OutputMatrix dst);
void multiply(InputMatrix src1, InputMatrix src2, OutputMatrix dst);

/*
 * @dst = @src1 / @src2;
 * */
void divide(InputMatrix src1, scalar src2, OutputMatrix dst);
void divide(scalar src1, InputMatrix src2, OutputMatrix dst);
void divide(InputMatrix src1, InputMatrix src2, OutputMatrix dst);


void sqrt(InputMatrix src, OutputMatrix dst);
void exp(InputMatrix src, OutputMatrix dst);
double sum(InputMatrix src);
double norml2(InputMatrix src1);

void derivativeRELU(InputOutputMatrix matrix);
void RELU(InputOutputMatrix mat);
void ELU(InputOutputMatrix mat);
void derivativeELU(InputOutputMatrix matrix);
void Tanh(InputOutputMatrix mat);
void Sigmoid(InputOutputMatrix matrix);
void Softmax(InputOutputMatrix matrix);


}//namespace MatrixUtils
}//namespace AiLearning

#endif //NEURALNETWORK_MATRIX_UTILS_H
