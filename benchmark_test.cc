//
// Created by moriarty on 2021/5/28.
//
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>

#include <viennacl/matrix.hpp>
#include "viennacl/linalg/prod.hpp"

//#include "viennacl/linalg/cuda/matrix_operations.hpp"
#include "viennacl/tools/random.hpp"

#include "matrix_utils.h"
#if 1
const int rows = 784, cols = 100, rows2 = cols, cols2 = 1;
#else
const int rows = 7840, cols = 1000, rows2 = cols, cols2 = 1;
#endif
const double alpha = 11.3, beta = 1.5, mgamma = 100.0;
#ifndef CV_TYPE
#if 0
using scalar = float;
#define CV_TYPE CV_32FC1
#else
using scalar = double;
#define CV_TYPE CV_64FC1
#endif
#else
using scalar = AiLearning::scalar;
#endif

void test_viennacl_linalg_cuda(benchmark::State& state) {
  viennacl::matrix<scalar> m1(rows, cols), m2(rows2, cols2), m3(rows, cols2), m4(rows, cols2);
  viennacl::tools::uniform_random_numbers<scalar> randomNumber;
#define RandomViennacl(matrix) do { \
  for (unsigned int i = 0; i < matrix.size1(); ++i)\
    for (unsigned int j = 0; j < matrix.size2(); ++j)\
      matrix(i,j) = randomNumber();\
} while(0)


  RandomViennacl(m1);
  RandomViennacl(m2);
  RandomViennacl(m3);
#undef RandomViennacl

  for (auto _ : state) {
//    viennacl::linalg::cuda::detail::prod(m1, false, m2, false, m4, 1, 1);
//    viennacl::linalg::cuda::detail::add
//    m4 *= alpha;
//    m4 =  m3;
  }
}

static void cpugpumulnostream(benchmark::State& state) {
  cv::Mat m1(rows, cols, CV_TYPE), m2(rows, cols, CV_TYPE), m3(rows, cols, CV_TYPE);
  cv::randu(m1, -100, 100);
  cv::randu(m2, -100, 100);
  cv::randu(m3, -100, 100);
  for (auto _ : state) {
    // This code gets timed
    cv::cuda::multiply(m1, m2, m3);
  }
}

static void cpugpumul(benchmark::State& state) {
  cv::cuda::Stream stream;
  cv::Mat m1(rows, cols, CV_TYPE), m2(rows, cols, CV_TYPE), m3(rows, cols, CV_TYPE);
  cv::randu(m1, -100, 100);
  cv::randu(m2, -100, 100);
  cv::randu(m3, -100, 100);
  for (auto _ : state) {
    // This code gets timed
    cv::cuda::multiply(m1, m2, m3, 1, -1, stream);
  }
}

static void gpumulnostream(benchmark::State& state) {
  cv::Mat tmp(rows, cols, CV_TYPE);
  cv::cuda::GpuMat m1, m2, m3;

  cv::randu(tmp, -100, 100);
  m1.upload(tmp);

  cv::randu(tmp, -100, 100);
  m2.upload(tmp);

  cv::randu(tmp, -100, 100);
  m3.upload(tmp);
  for (auto _ : state) {
    cv::cuda::multiply(m1, m2, m3);
  }
}

static void gpumul(benchmark::State& state) {
  cv::cuda::Stream stream;
  cv::Mat tmp(rows, cols, CV_TYPE);
  cv::cuda::GpuMat m1, m2, m3;

  cv::randu(tmp, -100, 100);
  m1.upload(tmp);

  cv::randu(tmp, -100, 100);
  m2.upload(tmp);

  cv::randu(tmp, -100, 100);
  m3.upload(tmp);

  for (auto _ : state) {
    cv::cuda::multiply(m1, m2, m3, 1, -1, stream);
  }
}

static void cpumul(benchmark::State& state) {
  cv::Mat m1(rows, cols, CV_TYPE), m2(rows, cols, CV_TYPE), m3(rows, cols, CV_TYPE);
  cv::randu(m1, -100, 100);
  cv::randu(m2, -100, 100);
  cv::randu(m3, -100, 100);
  for (auto _ : state) {
    m3 = m1.mul(m2);
  }
}

static void cpumulmatrix(benchmark::State& state) {
  cv::Mat m1(rows, cols, CV_TYPE), m2(rows2, cols2, CV_TYPE);
  cv::Mat m3(rows, cols2, CV_TYPE), m4(rows, cols2, CV_TYPE);
  cv::randu(m1, -100, 100);
  cv::randu(m2, -100, 100);
  cv::randu(m3, -100, 100);
  cv::randu(m4, -100, 100);
  for (auto _ : state) {
    m4 = alpha * m1 * m2 + beta * m3;
  }
}

static void gpugemm(benchmark::State& state) {
  cv::cuda::Stream stream;
  cv::Mat tmp(rows, cols, CV_TYPE);
  cv::cuda::GpuMat m1, m2, m3, m4;

  cv::randu(tmp, -100, 100);
  m1.upload(tmp);

  tmp.create(rows2, cols2,CV_TYPE);
  cv::randu(tmp, -100, 100);
  m2.upload(tmp);

  tmp.create(rows, cols2,CV_TYPE);
  cv::randu(tmp, -100, 100);
  m3.upload(tmp);

  tmp.create(rows, cols2,CV_TYPE);
  cv::randu(tmp, -100, 100);
  m4.upload(tmp);

  for (auto _ : state) {
    cv::cuda::gemm(m1, m2, alpha, m3, beta, m4, 0, stream);
    m4.release();
  }
}


static void gpugemm2(benchmark::State& state) {
  cv::cuda::Stream stream;
  cv::Mat tmp(rows, cols, CV_TYPE), cm1;
  cv::cuda::GpuMat m1, m2, m3, m4;

  cv::randu(tmp, -100, 100);
  m1.upload(tmp);
  cm1 = tmp.clone();

  tmp.create(cols, rows,CV_TYPE);
  cv::randu(tmp, -100, 100);
  m2.upload(tmp);

  tmp.create(rows, rows,CV_TYPE);
  cv::randu(tmp, -100, 100);
  m3.upload(tmp);

  cv::randu(tmp, -100, 100);
  m4.upload(tmp);

  for (auto _ : state) {
    m1.upload(cm1);
    cv::cuda::gemm(m1, m2, alpha, m3, beta, m1, 0, stream);
  }
}


void test_eigen_parallel(benchmark::State& state) {
  int c = 0;
  LOG(ERROR) << "eigen thread = " << Eigen::nbThreads() << "," << state.thread_index << "," << state.threads;
  Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> m1, m2, m3, m4;

  m1.resize(rows, cols);
  m1.setRandom();

  m2.resize(rows2, cols2);
  m2.setRandom();

  m3.resize(rows, cols2);
  m3.setRandom();

  m4.resize(rows, cols2);
  m4.setRandom();

  for (auto _ : state) {
    c++;
    m4 = alpha * m1 * m2 + beta * m3;
  }
  LOG(ERROR) << "eigen end = " << Eigen::nbThreads() << "," << state.thread_index << "," << state.threads << "," << c;
}

void test_viennacl_linalg(benchmark::State& state) {
  viennacl::matrix<scalar> m1(rows, cols), m2(rows2, cols2), m3(rows, cols2), m4(rows, cols2);
  viennacl::tools::uniform_random_numbers<scalar> randomNumber;
#define RandomViennacl(matrix) do { \
  for (unsigned int i = 0; i < matrix.size1(); ++i)\
    for (unsigned int j = 0; j < matrix.size2(); ++j)\
      matrix(i,j) = randomNumber();\
} while(0)


  RandomViennacl(m1);
  RandomViennacl(m2);
  RandomViennacl(m3);

#undef RandomViennacl
  for (auto _ : state) {
    m4 = alpha * viennacl::linalg::prod(m1, m2) + beta * m3;
  }
}

void sigmoid_cv(benchmark::State& state) {
  cv::Mat mat(rows, cols, CV_64FC1);
  cv::randu(mat, -0.9999, 0.9999);
  cv::Mat o = mat.clone();
  for (auto _ : state) {
//    o.copyTo(mat);
    mat.forEach<double>([](double &p, const int * position) {
      if (p > 0) {
        p = 1 / (std::exp(-p) + 1.0);
      } else {
        double exp = std::exp(p);
        p = exp / (1 + exp);
      }
    });
  }
}


void sigmoid_for(benchmark::State& state) {
  cv::Mat mat(rows, cols, CV_64FC1);
  cv::randu(mat, -0.9999, 0.9999);
  cv::Mat o = mat.clone();
  const int size = rows * cols;
  for (auto _ : state) {
    double* data = (double*)mat.data;
    double *to = (double*)o.data;
    for (int i = 0; i < size; i++) {
      if (data[i] > 0) {
        to[i] = 1 / (std::exp(-data[i]) + 1.0);
      } else {
        double exp = std::exp(data[i]);
        to[i] = exp / (1 + exp);
      }
    }
  }
}

void sigmoid_eigen(benchmark::State& state) {
  AiLearning::Matrix matrix = Eigen::MatrixXd::Random(rows, cols);
  AiLearning::Matrix pp = matrix;
  for (auto _ : state) {
    AiLearning::MatrixUtils::Sigmoid(matrix);
  }
}


void Sigmoid(cv::cuda::GpuMat &matrix) {};
void thrust_sigmoid_cuda(benchmark::State& state) {
  cv::Mat mat(rows, cols, CV_64FC1);
  cv::randu(mat, -0.9999, 0.9999);
  cv::cuda::GpuMat gpumat(mat);
  for (auto _ : state) {
    Sigmoid(gpumat);
  }
}

using namespace AiLearning;

void add_ele_row(benchmark::State& state) {
  Eigen::Matrix<
    scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat1(rows, cols), mat2(rows, cols), mat3(rows, cols);
  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();
  for (auto _ : state) {
    AiLearning::MatrixUtils::add(mat1, mat2, mat3);
  }
}

void add_ele_col(benchmark::State& state) {
  Eigen::Matrix<
    scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mat1(rows, cols), mat2(rows, cols), mat3(rows, cols);
  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();
  for (auto _ : state) {
    mat3 = mat1 + mat2;
  }
}

void add_ele_Xf(benchmark::State& state) {
  Eigen::MatrixXf mat1(rows, cols), mat2(rows, cols), mat3(rows, cols);
  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();
  for (auto _ : state) {
    mat3 = mat1 + mat2;
  }
}

void gemm(benchmark::State& state) {
  Eigen::Matrix<
    scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    mat1(rows, cols), mat2(rows, cols), mat3(rows, cols), mat4(rows, cols);
  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();
  mat4.setRandom();
  for (auto _ : state) {
    MatrixUtils::gemm(mat1, mat2, alpha, mat3, beta, mat4, 0);
  }
}

void addWeighted(benchmark::State& state) {
  Eigen::Matrix<
    scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    mat1(rows, cols), mat2(rows, cols), mat3(rows, cols);
  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();
  for (auto _ : state) {
    MatrixUtils::addWeighted(mat1, alpha, mat2, beta, mgamma, mat3);
  }
}

#define TestFunctionABC(name) void name(benchmark::State& state) {\
    Eigen::Matrix<\
      scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>\
      mat1(rows, cols), mat2(rows, cols), mat3(rows, cols);\
    mat1.setRandom();\
    mat2.setRandom();\
    mat3.setRandom();\
    for (auto _ : state) {\
      MatrixUtils::name(mat1, mat2, mat3);\
    }\
  }\
  void name(benchmark::State& state)

#define TestFunctionIO(name) \
void name(benchmark::State& state) {\
  Eigen::Matrix<\
    scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>\
    mat1(rows, cols), mat2(rows, cols);\
  mat1.setRandom();          \
  mat1.array() += 4;                           \
  mat2.setRandom();\
  for (auto _ : state) {\
    MatrixUtils::name(mat1, mat2);\
  }\
}\
void name(benchmark::State& state)



TestFunctionIO(sqrt);

TestFunctionABC(multiply);
TestFunctionABC(subtract);
TestFunctionABC(divide);
TestFunctionABC(add);
TestFunctionIO(exp);


namespace AiLearning{
namespace MatrixUtils{
void raw_sqrt_ptr(const scalar *src, scalar *dst, const int size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = std::sqrt(src[i]);
  }
  return;
}

void new_sqrt(InputMatrix src, OutputMatrix dst) {
  if (MatrixUtils::isEmpty(dst)) {
    dst.resize(src.rows(), src.cols());
  }

  const int split = 6;
  const size_t n = src.rows() * src.cols() / split;
  const scalar *_src = src.data();
  scalar *_dst = dst.data();

#pragma omp parallel for
  for (int i = 0; i < split; i++) {
    if (i != (split - 1)) {
      raw_sqrt_ptr(_src + i * n, _dst + i * n, n);
    } else {
      raw_sqrt_ptr(_src + i * n, _dst + i * n, src.rows() * src.cols() - n * i);
    }
  }

  return;
}


void raw_sqrt(InputMatrix src, OutputMatrix dst) {
  if (MatrixUtils::isEmpty(dst)) {
    dst.resize(src.rows(), src.cols());
  }

  const int split = 1;
  const size_t n = src.rows() * src.cols() / split;
  const scalar *_src = src.data();
  scalar *_dst = dst.data();
  raw_sqrt_ptr(_src, _dst, n);

  return;
}
}
}


TestFunctionIO(raw_sqrt);
TestFunctionIO(new_sqrt);
void multiply2(benchmark::State& state) {
    Eigen::Matrix<\
      scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>\
      mat1(rows, cols), mat2(rows, cols), mat3(rows, cols);\
    for (auto _ : state) {\
      MatrixUtils::multiply(mat1, mat2, mat3);\
      mat2 = mat3;\
    }\
  }\

//BENCHMARK(cpumul);
//BENCHMARK(cpugpumul);
//BENCHMARK(gpumul);
//BENCHMARK(gpumulnostream);
//BENCHMARK(cpugpumulnostream);
//
//BENCHMARK(test_eigen_parallel);
//BENCHMARK(cpumulmatrix);
//BENCHMARK(gpugemm2);
//BENCHMARK(gpugemm);
//BENCHMARK(test_viennacl_linalg);

BENCHMARK(sigmoid_for);
//BENCHMARK(sigmoid_cv);
BENCHMARK(sigmoid_eigen);
//BENCHMARK(thrust_sigmoid_cuda);

//BENCHMARK(add_ele_Xf);
//BENCHMARK(add_ele_col);
//BENCHMARK(add);
//BENCHMARK(gemm);
//BENCHMARK(addWeighted);
//BENCHMARK(multiply);
//BENCHMARK(multiply2);
//BENCHMARK(subtract);
//BENCHMARK(divide);
//BENCHMARK(exp);
//BENCHMARK(sqrt);
//BENCHMARK(raw_sqrt);
//BENCHMARK(new_sqrt);

//BENCHMARK_MAIN();
int main(int argc, char** argv) {
  Eigen::initParallel();
  Eigen::setNbThreads(6);
  LOG(ERROR) << "set eigen parallel thread number " << Eigen::nbThreads();

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}