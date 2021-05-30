//
// Created by moriarty on 5/26/21.
//
#include "optimizer.h"
#include <glog/logging.h>

namespace AiLearning {


class GDOptimizer : public Optimizer{
private:
  double _learning_rate = 0.1;
  Matrix _output, _tmp;
public:
  GDOptimizer(){}
  const Matrix& UpdateParameter(const Matrix &error, const Matrix &Ok, const Matrix &Ik) override {
    if (Ok.empty()) {
      cv::cuda::gemm(
          error, Ik, _learning_rate, Matrix(), 0, _output, cv::GEMM_2_T, cu_stream);
    } else {
      cu_multiply(error, Ok, _tmp);
      cv::cuda::gemm(
          _tmp, Ik, _learning_rate, Matrix(), 0, _output, cv::GEMM_2_T, cu_stream);
    }

    return _output;

  }
};

class AdamOptimizer : public Optimizer {
private:
  Matrix _mt[2], _vt[2], _gt, _gt2, _sqrt_vt;
  int _t = 0;
  const double _belt1 = 0.9, _belt2 = 0.999, _alpha = 0.001, _e = 1e-7;
  double _belt1t = _belt1, _belt2t = _belt2;
  Matrix _output, _tmp;

  const Matrix& UpdateParameter(const Matrix &gt) {
    if (_mt[0].empty() || _vt[0].empty()) {
      gt.copyTo(_mt[0]);
      gt.copyTo(_vt[0]);
      _mt[0].setTo(0);
      _vt[0].setTo(0);
    }
    const int8_t cur = _t % 2, update = (_t + 1) % 2;

    cv::cuda::addWeighted(_mt[cur], _belt1, gt, (1 - _belt1), 0, _mt[update], -1, cu_stream);

    cu_multiply(gt, gt, _gt2);
    cv::cuda::addWeighted(_vt[cur], _belt2, _gt2, (1 - _belt2), 0, _vt[update], -1, cu_stream);
    double alphat = _alpha * std::sqrt(1 - _belt2t) / (1 - _belt1t);
    cv::cuda::sqrt(_vt[update], _sqrt_vt, cu_stream);
    _t++;
    _belt1t *= _belt1;
    _belt2t *= _belt2;

    cv::cuda::add(_sqrt_vt, _e, _sqrt_vt);
    cu_multiply(alphat, _mt[update], _mt[update]);
    cv::cuda::divide(_mt[update], _sqrt_vt, _output, 1, -1, cu_stream);
    return _output;
  }

public:
  AdamOptimizer(){}

  const Matrix& UpdateParameter(const Matrix &error, const Matrix &Ok, const Matrix &Ik) override {
    if (Ok.empty()) {
      cv::cuda::gemm(
          error, Ik, 1, Matrix(), 0, _gt, cv::GEMM_2_T, cu_stream);

    } else {
      cu_multiply(error, Ok, _tmp);
      cv::cuda::gemm(
          _tmp, Ik, 1, Matrix(), 0, _gt, cv::GEMM_2_T, cu_stream);
    }
    return UpdateParameter(_gt);
  }

};

OptimizerPtr Optimizer::create(const std::string &name) {
  if (name == "GD") {
    return static_cast<OptimizerPtr>(std::make_shared<GDOptimizer>());
  } else if (name == "Adam") {
    return static_cast<OptimizerPtr>(std::make_shared<AdamOptimizer>());
  }
  LOG(ERROR) << "invalid input Optimizer name [" << name << "]";
  return nullptr;
}

}//namespace AiLearning