//
// Created by moriarty on 5/26/21.
//
#include "optimizer.h"
#include "matrix_utils.h"
#include "timer.h"
#include <glog/logging.h>

namespace AiLearning {


class GDOptimizer : public Optimizer{
private:
  double _learning_rate = 0.1;
  Matrix _output, _tmp;
public:
  GDOptimizer(){}
  const Matrix& UpdateParameter(const Matrix &error, const Matrix &Ok, const Matrix &Ik) override {
    if (MatrixUtils::isEmpty(Ok)) {
      MatrixUtils::gemm(
          error, Ik, _learning_rate, Matrix(), 0, _output, MatrixUtils::GEMM_2_T);
    } else {
      MatrixUtils::multiply(error, Ok, _tmp);
      MatrixUtils::gemm(
          _tmp, Ik, _learning_rate, Matrix(), 0, _output, MatrixUtils::GEMM_2_T);
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
    MicrosecondTimer timer(__func__ );
    timer.begin();
    if (MatrixUtils::isEmpty(_mt[0]) || MatrixUtils::isEmpty(_vt[0])) {
      MatrixUtils::CopyTo(gt, _mt[0]);
      MatrixUtils::CopyTo(gt, _vt[0]);

      MatrixUtils::setZeros(_mt[0]);
      MatrixUtils::setZeros(_vt[0]);
    }
    const int8_t cur = _t % 2, update = (_t + 1) % 2;

    MatrixUtils::addWeighted(_mt[cur], _belt1, gt, (1 - _belt1), 0, _mt[update]);

    MatrixUtils::multiply(gt, gt, _gt2);
    MatrixUtils::addWeighted(_vt[cur], _belt2, _gt2, (1 - _belt2), 0, _vt[update]);
    double alphat = _alpha * std::sqrt(1 - _belt2t) / (1 - _belt1t);
    MatrixUtils::sqrt(_vt[update], _sqrt_vt);
    _t++;
    _belt1t *= _belt1;
    _belt2t *= _belt2;

    MatrixUtils::add(_sqrt_vt, _e, _sqrt_vt);
    MatrixUtils::divide(_mt[update], _sqrt_vt, _output);
    MatrixUtils::multiply(alphat, _output, _output);
    timer.end();
    return _output;
  }

public:
  AdamOptimizer(){}

  const Matrix& UpdateParameter(const Matrix &error, const Matrix &Ok, const Matrix &Ik) override {
    if (MatrixUtils::isEmpty(Ok)) {
      MatrixUtils::gemm(
          error, Ik, 1, Matrix(), 0, _gt, MatrixUtils::GEMM_2_T);

    } else {
      MatrixUtils::multiply(error, Ok, _tmp);
      MatrixUtils::gemm(
          _tmp, Ik, 1, Matrix(), 0, _gt, MatrixUtils::GEMM_2_T);
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