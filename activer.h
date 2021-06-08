//
// Created by moriarty on 6/1/21.
//

#ifndef NEURALNETWORK_ACTIVER_H
#define NEURALNETWORK_ACTIVER_H
#include "common.h"

namespace AiLearning {
class Activer;
using ActiverPtr = std::shared_ptr<Activer>;

class Activer {
private:
  const std::string _name;
public:
  Activer(const std::string &name) : _name(name) {}
  virtual void active(std::vector<Matrix> &mat) = 0;
  virtual void derivatives(std::vector<Matrix> &mat) = 0;
  const std::string &name() const {return _name;}
  static ActiverPtr create(const std::string &name);
};





}




#endif //NEURALNETWORK_ACTIVER_H
