//
// Created by moriarty on 5/26/21.
//

#ifndef NEURALNETWORK_OPTIMIZER_H
#define NEURALNETWORK_OPTIMIZER_H
#include <opencv2/opencv.hpp>


namespace AiLearning{
class Optimizer {
private:
  cv::Mat _mt, _vt;
  const double _belt1 = 0.9, _belt2 = 0.999, _alpha = 0.001, _e = 1e-7;
  double _belt1t = 1, _belt2t = 1;
public:
  Optimizer(){}
  cv::Mat UpdateParameter(const cv::Mat &gt) {
    if (_mt.empty() || _vt.empty()) {
      gt.copyTo(_mt);
      gt.copyTo(_vt);
      _mt = 0;
      _vt = 0;
    }

    _belt1t *= _belt1;
    _belt2t *= _belt2;

    _mt = _belt1 * _mt + (1 - _belt1) * gt;
    _vt = _belt2 * _vt + (1 - _belt2) * gt.mul(gt);
#if 1
    double alphat = _alpha * std::sqrt(1 - _belt2t) / (1 - _belt1t);
    cv::Mat sqrt_vt;
    cv::sqrt(_vt, sqrt_vt);
    return alphat * _mt / (sqrt_vt + _e);

#else
    cv::Mat dmt = _mt / (1 - _belt1t);
    cv::Mat dvt = _vt / (1 - _belt2t), sqrt_dvt;
    cv::sqrt(dvt, sqrt_dvt);
    return _alpha * dmt / (sqrt_dvt + _e);
#endif
  }
};


}//namespace AiLearning



#endif //NEURALNETWORK_OPTIMIZER_H
