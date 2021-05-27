//
// Created by moriarty on 5/26/21.
//

#ifndef NEURALNETWORK_OPTIMIZER_H
#define NEURALNETWORK_OPTIMIZER_H
#include <opencv2/opencv.hpp>
#include <memory>

namespace AiLearning{
class Optimizer;
using OptimizerPtr = std::shared_ptr<Optimizer>;

class Optimizer {
public:
  virtual cv::Mat UpdateParameter(const cv::Mat &error, const cv::Mat &Ok, const cv::Mat &Ik) = 0;
  static OptimizerPtr create(const std::string &name);
};




}//namespace AiLearning



#endif //NEURALNETWORK_OPTIMIZER_H
