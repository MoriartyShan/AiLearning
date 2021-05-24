//
// Created by moriarty on 2021/5/16.
//
#include "neuron.h"


namespace AiLearning{

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
    LOG(ERROR) << "constructor input who empty, " << id()
               << "," << _prev_neurons_idx.size();
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
      }
    }

  } else {
    _Whos.reserve(constructor._Whos.size());
    LOG(ERROR) << "constructor input who has, " << id() << "," << constructor._Whos.size();
    for (auto &Who : constructor._Whos) {
      CHECK(Who.rows == output_data_size()) << "Who size = "
        << Who.rows << "," << Who.cols << "; node_num =" << output_data_size();
      _Whos.emplace_back(Who.clone());
    }
  }

  if (!_prev_neurons_idx.empty()) {
    for (auto prev : _prev_neurons_idx) {
      _prev_neurons_error.emplace(prev, cv::Mat());
    }
  }

}

void Neuron::constructor(NeuronConstructor &c) const {
  c._next_neurons_idx = _next_neurons_idx;
  c._output_data_size = _output_data_size;
  if (_Whos.size() == 1) {
    c._input_data_size = _Whos.front().cols;
  } else {
    c._input_data_size = 0;
  }

  c._active_func = _active;
  c._prev_neurons_idx = _prev_neurons_idx;
  c._netWork_ptr = _netWork_ptr;
  c._Whos = _Whos;
  return;
}

void Neuron::query(const cv::Mat &in){
  _in = &in;
  _processing = Who(0) * in;
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << processing().t();
  _active_func(_processing);
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << processing().t();
//  LOG(ERROR) << "neuron_" << id() << " output " << _processing.size();
  return;
}

void Neuron::query() {
  const size_t prev_num = num_prev();
  _in = nullptr;
  _processing.setTo(0);
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << processing().t();
  for (size_t i = 0; i < prev_num; i++) {
//    LOG(INFO) << "who:\n" << Who(i);
//    std::ofstream file("./who" + std::to_string(i) + ".csv");
//    file << cv::Formatter::get(cv::Formatter::FMT_CSV)->format(Who(i)) << std::endl;
//    file.close();
//    file.open("./process" + std::to_string(i) + ".csv");
//    file << cv::Formatter::get(cv::Formatter::FMT_CSV)->format(
//        _netWork_ptr->neuron(_prev_neurons_idx[i])->processing()) << std::endl;
//    file.close();
    _processing += Who(i) * _netWork_ptr->neuron(_prev_neurons_idx[i])->processing();
//    LOG(ERROR) << "neuron_" << id() << " read from neuron_" << _prev_neurons_idx[i] << " size "
//               << ",who size " << Who(i).size()
//               << _netWork_ptr->neuron(_prev_neurons_idx[i])->processing().size();
  }
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << processing().t();
  _active_func(_processing);
//  LOG(ERROR) << "neuron_" << id() << " output " << _processing.size();
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << processing().t();
  return;
}

void Neuron::back_propogate(
  const float learning_rate, const cv::Mat &error) {
  const size_t prev_num = num_prev();
  bool cross = false;
  const bool use_cross_entropy = false;
  if (use_cross_entropy &&
      _next_neurons_idx.empty()
      && (_active == "Softmax" || _active == "Sigmoid")) {
    ///(Cross Entropy) & Softmax
    ///derivate (Tk - Ok) * Oj, Oj is the input from jth neuron of last level
    cross = true;
  } else {
    _derivatives_func(_processing);
  }

  CHECK(check(processing())) << "neuron_" << id() << ",_process " << _processing.t() << "," << id();
  for (size_t i = 0; i < prev_num; i++) {
    _prev_neurons_error.at(_prev_neurons_idx[i]) = Who(i).t() * error;
    if (cross) {
      _Whos[i] += learning_rate *
                  error * (_netWork_ptr->neuron(_prev_neurons_idx[i])->processing()).t();
//      LOG(ERROR) << "cross";
    } else {
      _Whos[i] += learning_rate *
                  (error.mul(_processing)) * (_netWork_ptr->neuron(_prev_neurons_idx[i])->processing()).t();
    }


    CHECK(check(Who(i))) << "neuron_" << id() << ",Who(" << i << ")" << _processing.t();
  }
  if (_in != nullptr) {
    CHECK(prev_num == 0) << prev_num << "," << id();
    _Whos[0] += learning_rate *
                (error.mul(_processing)) * _in->t();
    CHECK(check(Who(0))) << "neuron_" << id() << ",Who 0 " << _processing.t();
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


MulNetWork::MulNetWork(const std::vector<NeuronConstructor> &constructors) {
  const size_t size = constructors.size();
  for (size_t i = 0; i < size; i++) {
    _neurons.emplace_back(std::make_shared<Neuron>(constructors[i]));
    CHECK(_neurons.back()->id() == i) << _neurons.back()->id() << "," << i;
  }
}

const cv::Mat& MulNetWork::query(const cv::Mat &in) {
  neuron(0)->query(in);
  for (size_t i = 1; i < neurons_num(); i++) {
    neuron(i)->query();
  }
  return neurons().back()->processing();
}

scalar MulNetWork::train(const cv::Mat &in, const cv::Mat &target, const float learning_rate) {
  cv::Mat cur_error = (target - query(in));
  double loss = cv::norm(cur_error);
//  LOG(INFO) << "loss = " << loss << "," << cur_error.t();
  auto neuron = neurons().rbegin();
  (*neuron)->back_propogate(learning_rate, cur_error);
  for (neuron++; neuron != neurons().rend(); neuron++) {
    (*neuron)->back_propogate(learning_rate);
  }

  return loss;
}


void MulNetWork::write(const std::string &path, const bool output_matrix) const {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  CHECK(fs.isOpened()) << path << " open fail";
  NeuronConstructor constructor;

  fs << "neurons_num" << (int)neurons_num();
  for (int i = 0; i < neurons_num(); i++) {
    neuron(i)->constructor(constructor);
    constructor.write(fs, i, output_matrix);
  }
  fs.release();
}

void MulNetWork::read(const std::string &path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  CHECK(fs.isOpened()) << "path:" << path << " open fail";
  int layer_nb;
  NeuronConstructor constructor;
  constructor._netWork_ptr = this;

  cv::read(fs["neurons_num"], layer_nb, -1);
  CHECK(layer_nb > 0) << layer_nb;
  _neurons.reserve(layer_nb);

  for (int i = 0; i < layer_nb; i++) {
    if (!constructor.read(fs,  i)) {
      LOG(FATAL) << "layer number is " << layer_nb << ", can not find neuron_" << i;
    }
    _neurons.emplace_back(std::make_shared<Neuron>(constructor));
  }
  return;
}

}//namespace AiLearning