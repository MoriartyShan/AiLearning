//
// Created by moriarty on 6/1/21.
//

#include "activer.h"
#include "timer.h"
#include "matrix_utils.h"

namespace AiLearning {

class SigmoidActiver : public Activer {
private:
  Matrix _tmp;
public:
  SigmoidActiver(const std::string &name) : Activer(name) {}

  void active(Matrix &mat) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::Sigmoid(mat);
    timer.end();
  }

  void derivatives(Matrix &mat) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    MatrixUtils::subtract(1, mat, _tmp);
    MatrixUtils::multiply(mat, _tmp, mat);
    timer.end();
  }
};


ActiverPtr Activer::create(const std::string &name) {
  ActiverPtr ptr;
  if (name == "Sigmoid") {
    auto _ptr = std::make_shared<SigmoidActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else {
    LOG(FATAL) << "not implement this type of activer:" << name;
  }
  return ptr;
}

}//namespace AiLearning