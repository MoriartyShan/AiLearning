//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H
#include "common.h"
#include "constructor.h"
#include "optimizer.h"

#include <glog/logging.h>
#include <memory>

namespace AiLearning{
using ActiveFuction = void (*) (Matrix &);
using DerivativesFuction = void (*) (Matrix &);

class Neuron;
class MulNetWork;

using NeuronPtr = std::shared_ptr<Neuron>;
using MulNetWorkPtr = std::shared_ptr<MulNetWork>;

class Neuron{
private:
  const std::string _active;
  ActiveFuction _active_func;
  DerivativesFuction _derivatives_func;
  const MulNetWork *_netWork_ptr;

  const Matrix *_in;//only useful for the first neuron

  std::vector<Matrix> _Whos;
  std::vector<OptimizerPtr> _optimizers;
  Matrix _processing, _tmp, _error;
//  const size_t _input_data_size;//input data size
  const int _output_data_size;//output data size
  const std::vector<int> _prev_neurons_idx;
  std::map<int, Matrix> _prev_neurons_error; //neuron index, error

  std::vector<int> _next_neurons_idx;

  const int _id;

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

  const std::vector<Matrix> &Whos() const {return _Whos;}
  const Matrix& Who(const int i) const {return _Whos[i];}
  size_t num_prev() const {return _prev_neurons_idx.size();}
  const Matrix& processing() const {return _processing;}

  int id() const {return _id;}
  size_t output_data_size() const {return _output_data_size;}
  void query(const Matrix &in);
  void query();
  void back_propogate(
      const float learning_rate, const Matrix &error);
  void back_propogate(const float learning_rate);
  const std::string& type() const {
    return _active;
  };

  void regist_next_neuron(const int i);

  bool is_next_neuron(const int i) const;

  const Matrix &prev_error(const int i) const {
    return _prev_neurons_error.at(i);
  }

  bool is_prev_neuron(const int i) const;
  bool check_consistency() const;
};

class MulNetWork {
private:
  std::vector<NeuronPtr> _neurons;
  Matrix _last_error;
public:
  MulNetWork() {}
  MulNetWork(const std::vector<NeuronConstructor> &constructors);
  size_t neurons_num() const {return _neurons.size();}

  scalar train(
      const Matrix &in, const Matrix &target, const float learning_rate = 0.1);

  const Matrix& query(const Matrix &in);

  void write(const std::string &path, const bool output_matrix) const;

  void read(const std::string &path);

  const std::vector<NeuronPtr>& neurons() const {return _neurons;}
  const NeuronPtr& neuron(const int i) const {return _neurons[i];}
  NeuronPtr& neuron(const int i) {return _neurons[i];}
  bool check_consistency() const;
};

}


#endif //NEURALNETWORK_NEURON_H
