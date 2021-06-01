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

const int rows = 784, cols = 100, rows2 = cols, cols2 = 1;
const double alpha = 11.3, belta = 1.5;
using Scalar_ = float;
#define CV_TYPE CV_32FC1


void test_viennacl_linalg_cuda(benchmark::State& state) {
  viennacl::matrix<Scalar_> m1(rows, cols), m2(rows2, cols2), m3(rows, cols2), m4(rows, cols2);
  viennacl::tools::uniform_random_numbers<Scalar_> randomNumber;
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
    m4 = alpha * m1 * m2 + belta * m3;
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
    cv::cuda::gemm(m1, m2, alpha, m3, belta, m4, 0, stream);
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
    cv::cuda::gemm(m1, m2, alpha, m3, belta, m1, 0, stream);
  }
}


void test_eigen_parallel(benchmark::State& state) {
  int c = 0;
  LOG(ERROR) << "eigen thread = " << Eigen::nbThreads() << "," << state.thread_index << "," << state.threads;
  Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic> m1, m2, m3, m4;

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
    m4 = alpha * m1 * m2 + belta * m3;
  }
  LOG(ERROR) << "eigen end = " << Eigen::nbThreads() << "," << state.thread_index << "," << state.threads << "," << c;
}

void test_viennacl_linalg(benchmark::State& state) {
  viennacl::matrix<Scalar_> m1(rows, cols), m2(rows2, cols2), m3(rows, cols2), m4(rows, cols2);
  viennacl::tools::uniform_random_numbers<Scalar_> randomNumber;
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
    m4 = alpha * viennacl::linalg::prod(m1, m2) + belta * m3;
  }
}

//BENCHMARK(cpumul);
//BENCHMARK(cpugpumul);
//BENCHMARK(gpumul);
//BENCHMARK(gpumulnostream);
//BENCHMARK(cpugpumulnostream);
//
BENCHMARK(test_eigen_parallel);
BENCHMARK(gpugemm2);
BENCHMARK(cpumulmatrix);
BENCHMARK(gpugemm);
BENCHMARK(test_viennacl_linalg);

//BENCHMARK_MAIN();
int main(int argc, char** argv) {
  Eigen::initParallel();
  Eigen::setNbThreads(1);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}