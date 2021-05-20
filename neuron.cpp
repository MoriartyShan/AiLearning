//
// Created by moriarty on 2021/5/16.
//
#include "neuron.h"


namespace AiLearning{


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
//    for (size_t i = 0; i < prev_neurons_num; i++) {
//      if (_Whos[i].rows != _node_num) {
//        LOG(ERROR) << "Who[" << i << "] row number should be equal to _node_num "
//                   << _Whos[i].size() << "," << _node_num;
//        return false;
//      }
//
//      if (_Whos[i].cols != _netWork_ptr->neuron(_prev_neurons_idx[i])->node_num()) {
//        LOG(ERROR) << "Who[" << i << "] row number should be equal to previous neuron's _node_num "
//                   << _Whos[i].size() << "," << _prev_neurons_idx[i] << ","
//                   << _netWork_ptr->neuron(_prev_neurons_idx[i])->node_num();
//        return false;
//      }
//    }
  }
}

Neuron::Neuron(const NeuronConstructor& constructor) :
    _active(constructor._active_func),
    _id(global_index()),
    _prev_neurons_idx(constructor._prev_neurons_idx),
    _netWork_ptr(constructor._netWork_ptr),
    _output_data_size(constructor._output_data_size),
    _next_neurons_idx(constructor._next_neurons_idx){

  set_active();

  _processing.create(_output_data_size, 1, CV_TYPE);

  if (constructor._Whos.empty()) {
    if (_prev_neurons_idx.empty()) {
      ///no prev neuron, the first one
      cv::Mat mat(_output_data_size,
                  constructor._input_data_size, CV_TYPE);
      Random(mat);
      _Whos.emplace_back(mat);
    } else {
      _Whos.reserve(constructor._prev_neurons_idx.size());
      for (auto prev : _prev_neurons_idx) {
        cv::Mat mat(_output_data_size,
                    _netWork_ptr->neuron(prev)->output_data_size(), CV_TYPE);
        Random(mat);
        _Whos.emplace_back(mat);
        _prev_neurons_error.emplace(prev, cv::Mat());
        LOG(ERROR) << "set _prev_neurons_error of " << id() << "," << prev;
      }
    }

  } else {
    _Whos.reserve(constructor._Whos.size());
    for (auto &Who : constructor._Whos) {
      CHECK(Who.rows == output_data_size()) << "Who size = "
        << Who.rows << "," << Who.cols << "; node_num =" << output_data_size();
      _Whos.emplace_back(Who.clone());
    }
  }
}

void Neuron::query(const cv::Mat &in){
  _in = &in;
  _processing = Who(0) * in;
  _active_func(_processing);
  return;
}

void Neuron::query() {
  const size_t prev_num = num_prev();
  _in = nullptr;
  _processing.setTo(0);
  for (size_t i = 0; i < prev_num; i++) {
    _processing += Who(i) * _netWork_ptr->neuron(_prev_neurons_idx[i])->processing();
  }
  _active_func(_processing);

  return;
}

void Neuron::back_propogate(
  const float learning_rate, const cv::Mat &error) {
  const size_t prev_num = num_prev();
//  LOG(FATAL) << "error " << error.t();
  _derivatives_func(_processing);
  for (size_t i = 0; i < prev_num; i++) {
    _prev_neurons_error.at(_prev_neurons_idx[i]) = Who(i).t() * error;
    _Whos[i] += learning_rate *
            (error.mul(_processing)) * (_netWork_ptr->neuron(i)->processing()).t();
  }
  if (_in != nullptr) {
    CHECK(prev_num == 0) << prev_num << "," << id();
    _Whos[0] += learning_rate *
                (error.mul(_processing)) * _in->t();
  }

  return;
}

void Neuron::back_propogate(const float learning_rate) {
  cv::Mat error(output_data_size(), 1, CV_TYPE);
  error = 0;
  for (auto next : _next_neurons_idx) {
    error += _netWork_ptr->neuron(next)->prev_error(id());
  }
  back_propogate(learning_rate, error);
  return;
}


MulNetWork::MulNetWork(const std::vector<int> &nodes) {
  const size_t size = nodes.size();
  _neurons.reserve(size - 1);

  NeuronConstructor constructor;
  constructor._netWork_ptr = this;
  constructor._active_func = "Sigmoid";
  constructor._Whos.clear();

  constructor._input_data_size = nodes[0];
  constructor._output_data_size = nodes[1];
  constructor._prev_neurons_idx.clear();
  constructor._next_neurons_idx.emplace_back(1);

  std::shared_ptr<Neuron> ptr =
    std::make_shared<Neuron>(constructor);
  _neurons.emplace_back(static_cast<std::shared_ptr<Neuron>>(ptr));

  CHECK(ptr->id() == 0) << ptr->id();

  for (size_t i = 1; i < size - 1; i++) {
    constructor._input_data_size = nodes[i - 1];
    constructor._output_data_size = nodes[i];
    constructor._prev_neurons_idx.resize(1);
    constructor._prev_neurons_idx[0] = i - 1;
    constructor._next_neurons_idx.resize(1);
    constructor._next_neurons_idx[0] = i + 1;

    ptr = std::make_shared<Neuron>(constructor);
    _neurons.emplace_back(ptr);
    CHECK(ptr->id() == i) << ptr->id() << "," << i;
  }

  constructor._input_data_size = nodes[size - 2];
  constructor._output_data_size = nodes[size - 1];
  constructor._prev_neurons_idx.resize(1);
  constructor._prev_neurons_idx[0] = size - 2;
  constructor._next_neurons_idx.clear();

  ptr = std::make_shared<Neuron>(constructor);
  _neurons.emplace_back(ptr);
  CHECK(ptr->id() == size - 1) << ptr->id() << "," << size - 1;
}

const cv::Mat& MulNetWork::query(const cv::Mat &in) {
  neuron(0)->query(in);
  for (size_t i = 1; i < neurons_num(); i++) {
    neuron(i)->query();
  }
  return neuron(neurons_num() - 1)->processing();
}

scalar MulNetWork::train(const cv::Mat &in, const cv::Mat &target, const float learning_rate) {
  cv::Mat cur_error = (target - query(in));
  double loss = cv::norm(cur_error);

//  LOG(ERROR) << "loss = " << cv::norm(cur_error) << "," << cur_error.t();
  auto neuron = neurons().rbegin();
  (*neuron)->back_propogate(learning_rate, cur_error);
  for (neuron++; neuron != neurons().rend(); neuron++) {
    (*neuron)->back_propogate(learning_rate);
  }

  return loss;
}


void MulNetWork::write(const std::string &path) const {
#if 0
  std::vector<cv::Mat> layers(_layers.size());

  for (int i = 0; i < layer_nb(); i++) {
    layers[i] = _layers[i]->Who();
  }

  cv::FileStorage file(path, cv::FileStorage::WRITE);
  CHECK(file.isOpened()) << "path:" << path << " open fail";
  cv::write(file, "layer_nb", (int)layer_nb());
  cv::write(file, "weights", layers);
  file.release();
#endif
}

void MulNetWork::read(const std::string &path) {
#if 0
  std::vector<cv::Mat> layers;
  cv::FileStorage file(path, cv::FileStorage::READ);
  CHECK(file.isOpened()) << "path:" << path << " open fail";
  int layer_nb;
  cv::read(file["layer_nb"], layer_nb, -1);
  CHECK(layer_nb > 0) << layer_nb;
  layers.reserve(layer_nb);
  cv::read(file["weights"], layers);
  file.release();

  _layers.clear();
  for (auto &l : layers) {
    auto ptr = std::make_shared<Neuron>(l, "");
    _layers.emplace_back(static_cast<std::shared_ptr<Neuron>>(ptr));
  }
#endif
  return;
}

}//namespace AiLearning