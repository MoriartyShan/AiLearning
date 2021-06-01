//
// Created by moriarty on 6/1/21.
//

#include "constructor.h"
#include "neuron.h"
#include <glog/logging.h>

namespace AiLearning {

void NeuronConstructor::write(cv::FileStorage &fs, const int id, const bool output_matrix) const {
  fs << ("neuron_" + std::to_string(id)) << "{"
     << "_prev_neurons_idx" << _prev_neurons_idx
     << "_next_neurons_idx" << _next_neurons_idx
     << "_input_data_size" << _input_data_size
     << "_output_data_size" << _output_data_size
     << "_active_func" << _active_func;
  if (output_matrix) {
    fs << "_Whos" << _Whos;
  } else {
    fs << "_Whos" << std::vector<cv::Mat>();
  }
  fs << "}";
}

bool NeuronConstructor::read(cv::FileStorage &fs, const int id) {
  auto node = fs[("neuron_" + std::to_string(id))];
  if (node.isNone()) {
    return false;
  }
  node["_prev_neurons_idx"] >> _prev_neurons_idx;
  node["_Whos"] >> _Whos;
  node["_next_neurons_idx"] >> _next_neurons_idx;
  node["_input_data_size"] >> _input_data_size;
  node["_output_data_size"] >> _output_data_size;
  node["_active_func"] >> _active_func;

  std::sort(_prev_neurons_idx.begin(), _prev_neurons_idx.end());
  std::sort(_next_neurons_idx.begin(), _next_neurons_idx.end());
//  LOG(ERROR) << "read Who " << _Whos.size() << "," << _Whos[0].size();

  return true;
}

bool NeuronConstructor::check_data() const {
  if (_active_func.empty()) {
    LOG(ERROR) << "_active_func is empty";
    return false;
  }
  if (_output_data_size <= 0) {
    LOG(ERROR) << "_node_num is wrong," << _output_data_size;
    return false;
  }

  const size_t num_all_neurons = _netWork_ptr->neurons_num();
  const size_t prev_neurons_num = _prev_neurons_idx.size();

  for (auto p : _prev_neurons_idx) {
    if (p >= num_all_neurons) {
      LOG(ERROR) << "current has neuron number = "
                 << num_all_neurons << ",but index to " << p;
      return false;
    }
  }

  if (!_Whos.empty()) {
    if (prev_neurons_num != _Whos.size()) {
      LOG(ERROR) << "previous neurons number should be the same as _Whos.size() "
                 << prev_neurons_num << "," << _Whos.size();
      return false;
    }
    for (size_t i = 0; i < prev_neurons_num; i++) {
      if (_Whos[i].rows != _output_data_size) {
        LOG(ERROR) << "Who[" << i << "] row number should be equal to _node_num "
                   << _Whos[i].size() << "," << _output_data_size;
        return false;
      }

      if (_Whos[i].cols != _netWork_ptr->neuron(_prev_neurons_idx[i])->output_data_size()) {
        LOG(ERROR) << "Who[" << i << "] row number should be equal to previous neuron's _node_num "
                   << _Whos[i].size() << "," << _prev_neurons_idx[i] << ","
                   << _netWork_ptr->neuron(_prev_neurons_idx[i])->output_data_size();
        return false;
      }
    }
  }
}

}//namespace AiLearning
