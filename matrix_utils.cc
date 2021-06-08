//
// Created by moriarty on 6/2/21.
//
#include "matrix_utils.h"
#include "timer.h"


namespace AiLearning {
namespace MatrixUtils {
#define ELU_COEF 1.0

#ifdef OPENCV_CUDA_MODE
void combineMatrix(const std::vector<Matrix>& src, OutputMatrix dst) {
  const int cols = src.size();
  resizeMatrix(dst, src.front().rows, cols);
  for (int i = 0; i < cols; i++) {
    ACHECK(src[i].cols == 1) << src[i].size() << ", size should b nx1";
    src[i].copyTo(dst.col(i));

  }
  return;
}

void resizeMatrix(OutputMatrix matrix, int rows, int cols) {
  if (matrix.rows != rows || matrix.cols != cols) {
    matrix.create(rows, cols, CV_TYPE);
  }
}

Matrix createMatrix(int rows, int cols, int type) {
  return Matrix(rows, cols, type);
}

bool isEmpty(InputMatrix mat) {
  return mat.empty();
}

void Random(Matrix &matrix) {
  CHECK(matrix.type() == CV_TYPE);
  cv::Mat r(matrix);
  AiLearning::Random(r);
  matrix.upload(r);
}

void CopyTo(InputMatrix src, OutputMatrix dst) {
  src.copyTo(dst);
}

void CopyTo(InputMatrix src, cv::Mat& dst) {
  src.download(dst);
}

void CopyTo(const cv::Mat& src, OutputMatrix dst) {
  dst.upload(src);
}

void setZeros(InputOutputMatrix mat) {
  mat.setTo(0);
}

void setTo(InputOutputMatrix mat, double v) {
  mat.setTo(v);
}

bool check(const Matrix &matrix) {
  return cv::checkRange(cv::Mat(matrix));
}

cv::Size MatrixSize(InputMatrix mat) {
  return mat.size();
}

/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
void gemm(InputMatrix src1, InputMatrix src2, double alpha,
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
void add(InputMatrix src1, scalar src2, OutputMatrix dst) {
  cv::cuda::add(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}
void add(scalar src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::add(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}
void add(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::add(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
void addWeighted(
  InputMatrix src1, double alpha, InputMatrix src2,
  double beta, double gamma, Matrix &dst) {
  cv::cuda::addWeighted(src1, alpha, src2, beta, gamma, dst, -1, cu_stream);
  return;
}

/*
 * @dst = @src1 - @src2;
 * */
void subtract(InputMatrix src1, scalar src2, OutputMatrix dst) {
  cv::cuda::subtract(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}
void subtract(scalar src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::subtract(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}
void subtract(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::subtract(src1, src2, dst, cv::noArray(), -1, cu_stream);
  return;
}

/* element wise
 * @dst = @src1.mul(@src2);
 * */
void multiply(InputMatrix src1, scalar src2, OutputMatrix dst) {
  cv::cuda::multiply(src1, src2, dst, 1, -1, cu_stream);
  return;
}
void multiply(scalar src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::multiply(src1, src2, dst, 1, -1, cu_stream);
  return;
}
void multiply(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::multiply(src1, src2, dst, 1, -1, cu_stream);
  return;
}


/*
 * @dst = @src1 / @src2;
 * */
void divide(InputMatrix src1, scalar src2, OutputMatrix dst) {
  cv::cuda::divide(src1, src2, dst, 1, -1, cu_stream);
  return;
}
void divide(scalar src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::divide(src1, src2, dst, 1, -1, cu_stream);
  return;
}
void divide(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  cv::cuda::divide(src1, src2, dst, 1, -1, cu_stream);
  return;
}



void sqrt(InputMatrix src, OutputMatrix dst) {
  cv::cuda::sqrt(src, dst, cu_stream);
  return;
}

void exp(InputMatrix src, OutputMatrix dst) {
  cv::cuda::exp(src, dst, cu_stream);
  return;
}

double sum(InputMatrix src) {
  return cv::cuda::sum(src)(0);
}

double norml2(InputMatrix src1) {
  return cv::cuda::norm(src1, cv::NORM_L2);
}

void derivativeTanh(InputOutputMatrix &matrix) {
  cv::cuda::pow(matrix, 2, matrix, cu_stream);
  MatrixUtils::subtract(1, matrix, matrix);
}

void Softmax(InputOutputMatrix matrix) {
  double max;
  cv::cuda::minMax(matrix, nullptr, &max);
  MatrixUtils::subtract(matrix, max, matrix);
  cv::cuda::exp(matrix, matrix, cu_stream);
  scalar sum = cv::cuda::sum(matrix)(0);
  MatrixUtils::divide(matrix, sum, matrix);
}


///implemented in cuda_support
//void Sigmoid(InputOutputMatrix &matrix) {}
//void Tanh(InputOutputMatrix mat){};
//void derivativeRELU(InputOutputMatrix matrix);
//void RELU(InputOutputMatrix mat);
//void ELU(InputOutputMatrix mat);
//void derivativeELU(InputOutputMatrix matrix);


#elif defined(OPENCV_CPU_MODE)
void combineMatrix(const std::vector<Matrix>& src, OutputMatrix dst) {
  const int cols = src.size();
  resizeMatrix(dst, src.front().rows, cols);
  for (int i = 0; i < cols; i++) {
    ACHECK(src[i].cols == 1) << src[i].size() << ", size should b nx1";
    src[i].copyTo(dst.col(i));
  }
  return;
}

void resizeMatrix(OutputMatrix matrix, int rows, int cols) {
  if (matrix.rows != rows || matrix.cols != cols) {
    matrix.create(rows, cols, CV_TYPE);
  }
}

Matrix createMatrix(int rows, int cols, int type) {
  return Matrix(rows, cols, type);
}

bool isEmpty(InputMatrix mat) {
  return mat.empty();
}

void CopyTo(InputMatrix src, OutputMatrix dst) {
  src.copyTo(dst);
}

void setZeros(InputOutputMatrix mat) {
  mat.setTo(0);
}

void setTo(InputOutputMatrix mat, double v) {
  mat.setTo(v);
}

void Random(Matrix &matrix) {
  AiLearning::Random(matrix);
}

scalar* get(InputOutputMatrix mat, int i, int j) {
  CHECK(mat.type() == CV_TYPE) << mat.type() << "," << CV_TYPE;
  return mat.ptr<scalar>(i, j);
}

cv::Size MatrixSize(InputMatrix mat) {
  return mat.size();
}

bool check(const Matrix &matrix) {
  return cv::checkRange(matrix);
}

/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
void gemm(InputMatrix src1, InputMatrix src2, double alpha,
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
void add(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1 + src2;
}
void add(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 + src2;
}
void add(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 + src2;
}

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
void addWeighted(
    InputMatrix src1, double alpha, InputMatrix src2,
    double beta, double gamma, OutputMatrix dst) {
  dst = src1 * alpha + src2 * beta + gamma;
  return;
}

/*
 * @dst = @src1 - @src2;
 * */
void subtract(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1 - src2;
  return;
}
void subtract(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 - src2;
  return;
}
void subtract(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 - src2;
  return;
}

/* element wise
 * @dst = @src1.mul(@src2);
 * */
void multiply(InputMatrix src1, scalar src2, OutputMatrix dst) {
  cv::multiply(src1, src2, dst);
  return;
}
void multiply(scalar src1, InputMatrix src2, OutputMatrix dst) {
  cv::multiply(src1, src2, dst);
  return;
}
void multiply(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  cv::multiply(src1, src2, dst);
  return;
}

/*
 * @dst = @src1 / @src2;
 * */
void divide(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1 / src2;
  return;
}
void divide(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 / src2;
  return;
}
void divide(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 / src2;
  return;
}



void sqrt(InputMatrix src, OutputMatrix dst) {
  cv::sqrt(src, dst);
  return;
}

void exp(InputMatrix src, OutputMatrix dst) {
  cv::exp(src, dst);
  return;
}

double sum(InputMatrix src) {
  return cv::sum(src)(0);
}

double norml2(InputMatrix src1) {
  return cv::norm(src1, cv::NORM_L2);
}

void Sigmoid(InputOutputMatrix &matrix) {
  AiLearning::Sigmoid(matrix);
}



void derivativeRELU(InputOutputMatrix matrix) {
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p > 0) {
      p = 1;
    } else {
      p = 0;
    }
  });
}

void RELU(InputOutputMatrix matrix) {
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p < 0) {
      p = 0;
    }
  });
}

void derivativeELU(InputOutputMatrix matrix) {
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p > 0) {
      p = 1;
    } else {
      p = p + ELU_COEF;
    }
  });
}

void ELU(InputOutputMatrix matrix) {
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p <= 0) {
      p = ELU_COEF * (std::exp(p) - 1);
    }
  });
}

void Tanh(InputOutputMatrix matrix) {
  matrix.forEach<scalar>([](scalar &p, const int * position) {
    if (p > 0) {
      scalar exp = std::exp(-2 * p);
      p = (1 - exp) / (1 + exp);
    } else {
      scalar exp = std::exp(2 * p);
      p = (exp - 1) / (1 + exp);
    }
  });
}

void derivativeTanh(InputOutputMatrix &matrix) {
  matrix = 1 - matrix.mul(matrix);
}

void Softmax(InputOutputMatrix matrix) {
  double max;
  cv::minMaxIdx(matrix, nullptr, &max);
  matrix -= max;
  cv::exp(matrix, matrix);
  scalar sum = cv::sum(matrix)(0);
  matrix /= sum;
}

#elif defined(EIGEN_MODE)
///basic fuctions

void combineMatrix(const std::vector<Matrix>& src, OutputMatrix dst) {
  const int cols = src.size();
  resizeMatrix(dst, src.front().rows(), cols);
  for (int i = 0; i < cols; i++) {
      ACHECK(src[i].cols() == 1) << src[i].size() << ", size should b nx1";
    dst.col(i) = src[i];
  }
  return;
}

void resizeMatrix(OutputMatrix matrix, int rows, int cols) {
  if (matrix.rows() != rows || matrix.cols() != cols) {
    matrix.resize(rows, cols);
  }
}

Matrix createMatrix(int rows, int cols, int type) {
  return Matrix::Zero(rows, cols);
}

void CopyTo(InputMatrix src, OutputMatrix dst) {
  dst = src;
}

void CopyTo(InputMatrix src, cv::Mat& dst) {
  cv::eigen2cv(src, dst);
}

void CopyTo(const cv::Mat& src, OutputMatrix dst) {
  dst.resize(src.rows, src.cols);
  cv::cv2eigen(src, dst);
}

bool isEmpty(InputMatrix mat) {
  return (mat.size() == 0);
}

void setZeros(InputOutputMatrix mat) {
  mat.setZero();
}

void setTo(InputOutputMatrix mat, double v) {
  mat.setConstant(v);
}

scalar* get(InputOutputMatrix mat, int i, int j) {
  return &mat(i, j);
}

cv::Size MatrixSize(InputMatrix mat) {
  return cv::Size(mat.cols(), mat.rows());
}

bool check(const Matrix &matrix) {
  return matrix.allFinite();
}

void Random(Matrix &matrix) {
  matrix.setRandom();
}

/*
 * @dst = @alpha * @src1 * @src2 + @beta * @src3;
 * @flags:GemmFlags
 * */
void gemm(InputMatrix src1, InputMatrix src2, double alpha,
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
void add(InputMatrix src1, scalar src2, OutputMatrix dst) {
//  MicrosecondTimer timer(__func__ );
//  timer.begin();
  dst = src1.array() + src2;
//  timer.end();
  return;
}

void add(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
//  MicrosecondTimer timer(__func__ );
//  timer.begin();
  dst = src1 + src2;
//  timer.end();
  return;
}

/*
 * @dst = @alpha * @src1 + @beta * @src2 + @gamma;
 * */
void addWeighted(
    InputMatrix src1, double alpha, InputMatrix src2,
    double beta, double gamma, OutputMatrix dst) {
  dst = (src1 * alpha + src2 * beta).array() + gamma;
  return;
}

/*
 * @dst = @src1 - @src2;
 * */
void subtract(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 - src2;
  return;
}
void subtract(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 - src2.array();
  return;
}
void subtract(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1.array() - src2;
  return;
}

/* element wise
 * @dst = @src1.mul(@src2);
 * */
void multiply(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 * src2;
  return;
}

void multiply(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1 * src2;
  return;
}

void multiply(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1.array() * src2.array();
  return;
}

/*
 * @dst = @src1 / @src2;
 * */
void divide(InputMatrix src1, scalar src2, OutputMatrix dst) {
  dst = src1.array() / src2;
  return;
}

void divide(scalar src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1 / src2.array();
  return;
}

void divide(InputMatrix src1, InputMatrix src2, OutputMatrix dst) {
  dst = src1.array() / src2.array();
  return;
}

void sqrt(InputMatrix src, OutputMatrix dst) {
  if (isEmpty(dst)) {
    dst.resize(src.rows(), src.cols());
  }
  dst = src.array().sqrt();
  return;
}

void exp(InputMatrix src, OutputMatrix dst) {
  if (isEmpty(dst)) {
    dst.resize(src.rows(), src.cols());
  }
  dst = src.array().exp();
  return;
}

double sum(InputMatrix src) {
  return src.sum();
}

double norml2(InputMatrix src1) {
  return src1.norm();
}

void derivativeRELU(InputOutputMatrix matrix) {
  matrix = matrix.unaryExpr([](scalar p) {
    if (p > 0) {
      return scalar(1);
    }
    return scalar(0);
  });
}

void RELU(InputOutputMatrix matrix) {
  matrix = matrix.unaryExpr([](scalar p) {
    if (p >= 0) {
      return p;
    }
    return scalar(0);
  });
}

void derivativeELU(InputOutputMatrix matrix) {
  matrix = matrix.unaryExpr([](scalar p) {
    scalar res = p;
    if (res > 0) {
      res = 1;
    } else {
      res = res + ELU_COEF;
    }
    return res;
  });
}

void ELU(InputOutputMatrix matrix) {
  matrix = matrix.unaryExpr([](scalar p) {
    scalar res = p;
    if (res < 0) {
      res = ELU_COEF * (std::exp(res) - 1);
    }
    return res;
  });
}

void Tanh(InputOutputMatrix matrix) {
  matrix = matrix.unaryExpr([](scalar p) {
    scalar res;
    if (p > 0) {
      scalar exp = std::exp(-2 * p);
      res = (1 - exp) / (1 + exp);
    } else {
      scalar exp = std::exp(2 * p);
      res = (exp - 1) / (1 + exp);
    }
    return res;
  });
}

void derivativeTanh(InputOutputMatrix &matrix) {
  matrix = 1 - matrix.array() * matrix.array();
}

void Sigmoid(InputOutputMatrix matrix) {
  matrix = matrix.array().unaryExpr([](scalar p) {
    scalar res;
    if (p > 0) {
      res = 1 / (std::exp(-p) + 1.0);
    } else {
      double exp = std::exp(p);
      res = exp / (1 + exp);
    }
    return res;
  });
}

void Softmax(InputOutputMatrix matrix) {
  scalar max = matrix.maxCoeff();
  matrix.array() -= max;
  matrix = matrix.array().exp();
  scalar sum = matrix.sum();
  matrix /= sum;
}

#else
#error "You must specify one mode"
#endif

}//namespace MatrixUtils
}//namespace AiLearning