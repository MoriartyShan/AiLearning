//
// Created by moriarty on 5/26/21.
//

#ifndef NEURALNETWORK_OPTIMIZER_H
#define NEURALNETWORK_OPTIMIZER_H
#include "common.h"

#include <memory>

namespace AiLearning{
class Optimizer;
using OptimizerPtr = std::shared_ptr<Optimizer>;

class Optimizer {
public:
  virtual const Matrix& UpdateParameter(
    const std::vector<Matrix> &error, const std::vector<Matrix>& Ok, const std::vector<Matrix>& Ik) = 0;
  static OptimizerPtr create(const std::string &name);
};




}//namespace AiLearning



#endif //NEURALNETWORK_OPTIMIZER_H
