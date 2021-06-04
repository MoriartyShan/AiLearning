//
// Created by moriarty on 6/1/21.
//

#include "activer.h"
#include "timer.h"
#include "matrix_utils.h"

namespace AiLearning {


//active function whose derivatives is (1 - s) * s
#define SigTypeActiverDefine(Name) class Name##Activer : public Activer {\
private:\
  Matrix _tmp;\
public:\
  Name##Activer(const std::string &name) : Activer(name) {}\
  void active(Matrix &mat) override {\
    MicrosecondTimer timer(__func__);\
    timer.begin();\
    AiLearning::MatrixUtils::Name(mat);\
    timer.end();\
  }\
  void derivatives(Matrix &mat) override {\
    MicrosecondTimer timer(__func__);\
    timer.begin();\
    MatrixUtils::subtract(1, mat, _tmp);\
    MatrixUtils::multiply(mat, _tmp, mat);\
    timer.end();\
  }\
}

SigTypeActiverDefine(Sigmoid);
SigTypeActiverDefine(Softmax);


#define ActiverDefine(Name) class Name##Activer : public Activer {\
public:\
  Name##Activer(const std::string &name) : Activer(name) {}\
  void active(Matrix &mat) override {\
    MicrosecondTimer timer(__func__);\
    timer.begin();\
    AiLearning::MatrixUtils::Name(mat);\
    timer.end();\
  }\
  void derivatives(Matrix &mat) override {\
    MicrosecondTimer timer(__func__);\
    timer.begin();\
    AiLearning::MatrixUtils::derivative##Name(mat);\
    timer.end();\
  }\
}

ActiverDefine(ELU);
ActiverDefine(RELU);
ActiverDefine(Tanh);

ActiverPtr Activer::create(const std::string &name) {
  ActiverPtr ptr;
  if (name == "Sigmoid") {
    auto _ptr = std::make_shared<SigmoidActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else if (name == "Softmax") {
    auto _ptr = std::make_shared<SoftmaxActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else if (name == "ELU") {
    auto _ptr = std::make_shared<ELUActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else if (name == "RELU") {
    auto _ptr = std::make_shared<RELUActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else if (name == "Tanh") {
    auto _ptr = std::make_shared<TanhActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else {
    LOG(FATAL) << "not implement this type of activer:" << name;
  }
  return ptr;
}

}//namespace AiLearning