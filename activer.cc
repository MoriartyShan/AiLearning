//
// Created by moriarty on 6/1/21.
//

#include "activer.h"
#include "timer.h"
#include "matrix_utils.h"

namespace AiLearning {



#define SigTypeActiverDefine(name) class name##Activer : public Activer {\
private:\
  Matrix _tmp;\
public:\
  name##Activer(const std::string &name) : Activer(name) {}\
  void active(Matrix &mat) override {\
    MicrosecondTimer timer(__func__);\
    timer.begin();\
    AiLearning::MatrixUtils::name(mat);\
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

class TanhActiver : public Activer {
private:
  Matrix _tmp;
public:
  TanhActiver(const std::string &name) : Activer(name) {}

  void active(Matrix &mat) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::Tanh(mat);
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


class ELUActiver : public Activer {
public:
  ELUActiver(const std::string &name) : Activer(name) {}

  void active(Matrix &mat) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::ELU(mat);
    timer.end();
  }

  void derivatives(Matrix &matrix) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::derivativeELU(matrix);
    timer.end();
  }
};


class RELUActiver : public Activer {
public:
  RELUActiver(const std::string &name) : Activer(name) {}

  void active(Matrix &mat) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::RELU(mat);
    timer.end();
  }

  void derivatives(Matrix &matrix) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::derivativeRELU(matrix);
    timer.end();
  }
};

class SoftmaxActiver : public Activer {
private:
  Matrix _tmp;
public:
  SoftmaxActiver(const std::string &name) : Activer(name) {}

  void active(Matrix &mat) override {
    MicrosecondTimer timer(__func__);
    timer.begin();
    AiLearning::MatrixUtils::Softmax(mat);
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
  } else if (name == "Softmax") {
    auto _ptr = std::make_shared<SoftmaxActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  }  else if (name == "ELU") {
    auto _ptr = std::make_shared<ELUActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  }  else if (name == "RELU") {
    auto _ptr = std::make_shared<RELUActiver>(name);
    ptr = static_cast<ActiverPtr>(_ptr);
  } else {
    LOG(FATAL) << "not implement this type of activer:" << name;
  }
  return ptr;
}

}//namespace AiLearning