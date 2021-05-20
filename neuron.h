//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H
#include "common.h"
#include "glog/logging.h"
#include <memory>

namespace AiLearning{
using ActiveFuction = void (*) (cv::Mat &);
using DerivativesFuction = void (*) (cv::Mat &);

class Neuron{
protected:
  const std::string _active;
  ActiveFuction _active_func;
  DerivativesFuction _derivatives_func;
  cv::Mat _Who;
  cv::Mat _processing;
  const cv::Mat *_in;
  const int _id;

  static int global_index() {
    static int idx = 0;
    return idx++;
  }

  void set_active() {
    if (_active == "Sigmoid") {
      _active_func = Sigmoid;
      _derivatives_func = derivativesSigmoid;
    } else if (_active == "ELU") {
      _active_func = ELU;
      _derivatives_func = derivativesELU;
    } else if (_active == "Softmax") {
      _active_func = Softmax;
      _derivatives_func = derivativesSoftmax;
    } else {
      LOG(FATAL) << "not implemented:" << _active;
    }
  }

public:
  Neuron(const cv::Mat &Who, const std::string &active):
      _Who(Who.clone()), _id(global_index()), _active(active) {
    set_active();
  }
  Neuron(const int in, const int out, const std::string &active);

  const cv::Mat& Who() const {return _Who;}
  const cv::Mat& processing() const {return _processing;}
  const int input_size() const {return _Who.cols;}
  const int output_size() const {return _Who.rows;}
  int id() const {return _id;}

  virtual const cv::Mat& query(const cv::Mat &in);
  virtual const cv::Mat back_propogate(
      const float learning_rate, const cv::Mat &error);
  virtual const std::string& type() const {
    return _active;
  };

};

class MulNetWork {
private:
  std::vector<std::shared_ptr<Neuron>> _layers;

  cv::Mat _softmax, _exp;
  scalar _sum;
public:
  MulNetWork() {}
  MulNetWork(const std::vector<int> &nodes);
  size_t layer_nb() const {return _layers.size();}
  int input_size() const {return _layers.front()->input_size();}
  int output_size() const {return _layers.back()->output_size();}

  scalar train(
      const cv::Mat &in, const cv::Mat &target, const float learning_rate = 0.1);

  const cv::Mat& query(const cv::Mat &in);

  void write(const std::string &path) const;

  void read(const std::string &path);

  const std::shared_ptr<Neuron> layer(const int i) const {return _layers[i];}

};

}


#endif //NEURALNETWORK_NEURON_H
