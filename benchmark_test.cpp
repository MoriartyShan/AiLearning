//
// Created by moriarty on 2021/5/28.
//
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

const int rows = 8192, cols = 1103;
const double alpha = 11.3, belta = 1.5;

#define LOG std::cout << __FILE__ << ":" << __LINE__ << ": " << time(0) << ","

static void cpugpumulnostream(benchmark::State& state) {
  cv::Mat m1(rows, cols, CV_64FC1), m2(rows, cols, CV_64FC1), m3(rows, cols, CV_64FC1);
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
  cv::Mat m1(rows, cols, CV_64FC1), m2(rows, cols, CV_64FC1), m3(rows, cols, CV_64FC1);
  cv::randu(m1, -100, 100);
  cv::randu(m2, -100, 100);
  cv::randu(m3, -100, 100);
  for (auto _ : state) {
    // This code gets timed
    cv::cuda::multiply(m1, m2, m3, 1, -1, stream);
  }
}

static void gpumulnostream(benchmark::State& state) {
  cv::Mat tmp(rows, cols, CV_64FC1);
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
  cv::Mat tmp(rows, cols, CV_64FC1);
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
  cv::Mat m1(rows, cols, CV_64FC1), m2(rows, cols, CV_64FC1), m3(rows, cols, CV_64FC1);
  cv::randu(m1, -100, 100);
  cv::randu(m2, -100, 100);
  cv::randu(m3, -100, 100);
  for (auto _ : state) {
    m3 = m1.mul(m2);
  }
}

static void cpumulmatrix(benchmark::State& state) {
  cv::Mat m1(rows, cols, CV_64FC1), m2(cols, rows, CV_64FC1);
  cv::Mat m3(rows, rows, CV_64FC1), m4(rows, rows, CV_64FC1);
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
  cv::Mat tmp(rows, cols, CV_64FC1);
  cv::cuda::GpuMat m1, m2, m3, m4;

  cv::randu(tmp, -100, 100);
  m1.upload(tmp);

  tmp.create(cols, rows,CV_64FC1);
  cv::randu(tmp, -100, 100);
  m2.upload(tmp);

  tmp.create(rows, rows,CV_64FC1);
  cv::randu(tmp, -100, 100);
  m3.upload(tmp);

//  cv::randu(tmp, -100, 100);
//  m4.upload(tmp);

  for (auto _ : state) {
    cv::cuda::gemm(m1, m2, alpha, m3, belta, m4, 0, stream);
    m4.release();
  }
}


static void gpugemm2(benchmark::State& state) {
  cv::cuda::Stream stream;
  cv::Mat tmp(rows, cols, CV_64FC1), cm1;
  cv::cuda::GpuMat m1, m2, m3, m4;

  cv::randu(tmp, -100, 100);
  m1.upload(tmp);
  cm1 = tmp.clone();

  tmp.create(cols, rows,CV_64FC1);
  cv::randu(tmp, -100, 100);
  m2.upload(tmp);

  tmp.create(rows, rows,CV_64FC1);
  cv::randu(tmp, -100, 100);
  m3.upload(tmp);

  cv::randu(tmp, -100, 100);
  m4.upload(tmp);

  for (auto _ : state) {
    m1.upload(cm1);
    cv::cuda::gemm(m1, m2, alpha, m3, belta, m1, 0, stream);
  }
}

BENCHMARK(cpumul);
BENCHMARK(cpugpumul);
BENCHMARK(gpumul);
BENCHMARK(gpumulnostream);
BENCHMARK(cpugpumulnostream);

BENCHMARK(cpumulmatrix);
BENCHMARK(gpugemm);
BENCHMARK(gpugemm2);

//BENCHMARK_MAIN();
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}