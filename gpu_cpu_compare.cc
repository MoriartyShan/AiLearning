//
// Created by moriarty on 2021/5/29.
//

#include <glog/logging.h>
#include <opencv2/cudaarithm.hpp>
#include <Eigen/Core>

void test_addWeighted() {
  const double alpha = 1.2, beta = 4.56, gamma = -4.1;
  const int rows = 200, cols = 300;
  cv::cuda::Stream stream;
  cv::Mat cp1(rows, cols, CV_64FC1), cp2(rows, cols, CV_64FC1), cp3(rows, cols, CV_64FC1);
  cv::randu(cp1, -100, 100);
  cv::randu(cp2, -100, 100);
  cv::randu(cp3, -100, 100);

  cv::cuda::GpuMat m1(cp1), m2(cp2), m3(cp3);

  cv::cuda::addWeighted(m1, alpha, m2, beta, gamma, m1, -1, stream);
  cp1 = cp1 * alpha + cp2 * beta + gamma;
  m1.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);

  cv::cuda::addWeighted(m1, alpha, m2, beta, gamma, m2, -1, stream);
  cp1 = cp1 * alpha + cp2 * beta + gamma;
  m2.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);

}


void test_multiply() {
  const double alpha = 1.2, beta = 4.56, gamma = -4.1;
  const int rows = 200, cols = 300;
  cv::cuda::Stream stream;
  cv::Mat cp1(rows, cols, CV_64FC1), cp2(rows, cols, CV_64FC1), cp3(rows, cols, CV_64FC1);
  cv::randu(cp1, -100, 100);
  cv::randu(cp2, -100, 100);
  cv::randu(cp3, -100, 100);

  cv::cuda::GpuMat m1(cp1), m2(cp2), m3(cp3);

  cv::cuda::multiply(m1, m2, m1, 1, -1, stream);
  cp1 = cp1.mul(cp2);
  m1.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);

  cv::cuda::multiply(m1, m2, m2, 1, -1, stream);
  cp1 = cp1.mul(cp2);
  m2.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);
}


void test_add() {
  const double alpha = 1.2, beta = 4.56, gamma = -4.1;
  const int rows = 200, cols = 300;
  cv::cuda::Stream stream;
  cv::Mat cp1(rows, cols, CV_64FC1), cp2(rows, cols, CV_64FC1), cp3(rows, cols, CV_64FC1);
  cv::randu(cp1, -100, 100);
  cv::randu(cp2, -100, 100);
  cv::randu(cp3, -100, 100);

  cv::cuda::GpuMat m1(cp1), m2(cp2), m3(cp3);

  cv::cuda::add(m1, m2, m1, cv::noArray(), -1, stream);
  cp1 = cp1 + cp2;
  m1.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);

  cv::cuda::add(m1, m2, m2, cv::noArray(), -1, stream);
  cp1 = cp1 + cp2;
  m2.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);
}


void test_divide() {
  const double alpha = 1.2, beta = 4.56, gamma = -4.1;
  const int rows = 200, cols = 300;
  cv::cuda::Stream stream;
  cv::Mat cp1(rows, cols, CV_64FC1), cp2(rows, cols, CV_64FC1), cp3(rows, cols, CV_64FC1);
  cv::randu(cp1, -100, 100);
  cv::randu(cp2, -100, 100);
  cv::randu(cp3, -100, 100);

  cv::cuda::GpuMat m1(cp1), m2(cp2), m3(cp3);

  cv::cuda::divide(m1, m2, m1, 1, -1, stream);
  cp1 = cp1 / cp2;
  m1.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);

  cv::cuda::divide(m1, m2, m2, 1, -1, stream);
  cp1 = cp1 / cp2;
  m2.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);

}

void test_subtract() {
  const double alpha = 1.2, beta = 4.56, gamma = -4.1;
  const int rows = 200, cols = 300;
  cv::cuda::Stream stream;
  cv::Mat cp1(rows, cols, CV_64FC1), cp2(rows, cols, CV_64FC1), cp3(rows, cols, CV_64FC1);
  cv::randu(cp1, -100, 100);
  cv::randu(cp2, -100, 100);
  cv::randu(cp3, -100, 100);

  cv::cuda::GpuMat m1(cp1), m2(cp2), m3(cp3);

  cv::cuda::subtract(m1, m2, m1, cv::noArray(), -1, stream);
  cp1 = cp1 - cp2;
  m1.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);


  cv::cuda::subtract(m1, m2, m2, cv::noArray(), -1, stream);
  cp1 = cp1 - cp2;
  m2.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);
}

void test_gemm() {
  const double alpha = 1.2, beta = 4.56, gamma = -4.1;
  const int rows = 200, cols = 300;
  cv::cuda::Stream stream;
  cv::Mat cp1(rows, cols, CV_64FC1), cp2(cols, rows, CV_64FC1), cp3(rows, rows, CV_64FC1);
  cv::randu(cp1, -100, 100);
  cv::randu(cp2, -100, 100);
  cv::randu(cp3, -100, 100);

  cv::cuda::GpuMat m1(cp1), m2(cp2), m3(cp3);

  cv::cuda::gemm(m1, m2, alpha, m3, beta, m1, 0, stream);
  cp1 = alpha * cp1 * cp2 + beta * cp3;
  m1.download(cp3);
  LOG(ERROR) << __func__  << ":dirrerent = " << cv::norm(cp3 - cp1);
}
using EigenMatrix = Eigen::Matrix<
  double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
int main(int argc, char **argv) {
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::InitGoogleLogging(argv[0]); // option --[also]logtostderr
//  test_addWeighted();
//  test_multiply();
//  test_add();
//  test_divide();
//  test_subtract();
//  test_gemm();

  EigenMatrix m = EigenMatrix::Random(2, 3), n = m;
//  n.setRandom();
  m = m.array() + 1;
  n = m.cwiseSqrt();
  auto p = m;
  p = m.array().sqrt();
  LOG(ERROR) << "m:\n" << m;
  LOG(ERROR) << "n:\n" << m.cwiseSqrt();
  LOG(ERROR) << "p:\n" << p;


//  m.setConstant(3);
//  EigenMatrix p = m.array().sqrt();

//  m.addTo(n);
//  LOG(ERROR) << "p:\n" << m;
//  LOG(ERROR) << "n:\n" << n;
  return 0;
}