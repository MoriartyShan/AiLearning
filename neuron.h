//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H
#include "common.h"
#include "glog/logging.h"
#include "optimizer.h"
#include <memory>

namespace AiLearning{
using ActiveFuction = void (*) (cv::Mat &);
using DerivativesFuction = void (*) (cv::Mat &);

class Neuron;
class MulNetWork;

using NeuronPtr = std::shared_ptr<Neuron>;
using MulNetWorkPtr = std::shared_ptr<MulNetWork>;

struct NeuronConstructor{
  const MulNetWork *_netWork_ptr;

  std::vector<int> _prev_neurons_idx;
  std::vector<cv::Mat> _Whos;

  std::vector<int> _next_neurons_idx;

  int _input_data_size;
  int _output_data_size;

  std::string _active_func;
  bool check_data() const;

  void write(
      cv::FileStorage &fs, const int id,
      const bool output_matrix = true) const;

  bool read(cv::FileStorage &fs, const int id);

};

class Neuron{
private:
  const std::string _active;
  ActiveFuction _active_func;
  DerivativesFuction _derivatives_func;
  const MulNetWork *_netWork_ptr;

  const cv::Mat *_in;//only useful for the first neuron

  std::vector<cv::Mat> _Whos;
  std::vector<Optimizer> _optimizers;
  cv::Mat _processing;
//  const size_t _input_data_size;//input data size
  const int _output_data_size;//output data size
  const std::vector<int> _prev_neurons_idx;
  std::map<int, cv::Mat> _prev_neurons_error; //neuron index, error

  const std::vector<int> _next_neurons_idx;

  const int _id;

  double _learning_rate = 0.1;
  double _loss, _prev_loss = -1;

  static int global_index() {
    static int idx = 0;
    return idx++;
  }

  void set_active() {
    const auto &ac = _active;
    if (ac == "Sigmoid") {
      _active_func = Sigmoid;
      _derivatives_func = derivativesSigmoid;
    } else if (ac == "ELU") {
      _active_func = ELU;
      _derivatives_func = derivativesELU;
    } else if (ac == "Softmax") {
      _active_func = Softmax;
      _derivatives_func = derivativesSoftmax;
    } else if ("RELU" == ac) {
      _active_func = RELU;
      _derivatives_func = derivativesRELU;
    } else if ("Tanh" == ac) {
      _active_func = Tanh;
      _derivatives_func = derivateTanh;
    } else {
      LOG(FATAL) << "not implemented:" << _active;
    }
  }

public:
  Neuron(const NeuronConstructor& constructor);
  void constructor(NeuronConstructor &c) const;

  const std::vector<cv::Mat> &Whos() const {return _Whos;}
  const cv::Mat& Who(const int i) const {return _Whos[i];}
  size_t num_prev() const {return _prev_neurons_idx.size();}
  const cv::Mat& processing() const {return _processing;}

  int id() const {return _id;}
  size_t output_data_size() const {return _output_data_size;}
  void query(const cv::Mat &in);
  void query();
  void back_propogate(
      const float learning_rate, const cv::Mat &error);
  void back_propogate(const float learning_rate);
  const std::string& type() const {
    return _active;
  };

  const cv::Mat &prev_error(const int i) const {
    return _prev_neurons_error.at(i);
  }

  void update_loss(const double loss) {
    _loss += loss;
  }

  void reset_loss() {
    _loss = 0;
  }

  void update_learning_rate() {
    if (_prev_loss > 0 && _loss > _prev_loss) {
      _learning_rate *= 0.5;
      if (_learning_rate < 1e-6) {
        _learning_rate = 1e-6;
      }
//      LOG(ERROR) << "change neuron_" << id() << " learning rate to " << _learning_rate << ", " << _loss << ">" << _prev_loss;
    }
    _prev_loss = _loss;
    reset_loss();
  }

};

class MulNetWork {
private:
  std::vector<NeuronPtr> _neurons;
public:
  MulNetWork() {}
  MulNetWork(const std::vector<NeuronConstructor> &constructors);
  size_t neurons_num() const {return _neurons.size();}

  scalar train(
      const cv::Mat &in, const cv::Mat &target, const float learning_rate = 0.1);

  const cv::Mat& query(const cv::Mat &in);

  void write(const std::string &path, const bool output_matrix) const;

  void read(const std::string &path);

  const std::vector<NeuronPtr>& neurons() const {return _neurons;}
  const NeuronPtr& neuron(const int i) const {return _neurons[i];}

  void reset_loss() {
    for (auto &n : _neurons) {
      n->reset_loss();
    }
  }

  void update_learning_rate() {
    for (auto &n : _neurons) {
      n->update_learning_rate();
    }
  }

};

}


#endif //NEURALNETWORK_NEURON_H
