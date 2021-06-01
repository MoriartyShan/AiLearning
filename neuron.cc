//
// Created by moriarty on 2021/5/16.
//
#include "neuron.h"
#include "common.h"
#include "timer.h"

namespace AiLearning{

Neuron::Neuron(const NeuronConstructor& constructor) :
    _active(constructor._active_func),
    _id(global_index()),
    _prev_neurons_idx(constructor._prev_neurons_idx),
    _netWork_ptr(constructor._netWork_ptr),
    _output_data_size(constructor._output_data_size),
    _next_neurons_idx(constructor._next_neurons_idx){

  std::sort(_next_neurons_idx.begin(), _next_neurons_idx.end());

  set_active();

  _processing.create(_output_data_size, 1, CV_TYPE);
  const std::string opt("Adam");
  if (constructor._Whos.empty()) {
    LOG(ERROR) << "constructor input who empty, " << id()
               << "," << _prev_neurons_idx.size();
    if (_prev_neurons_idx.empty()) {
      ///no prev neuron, the first one
      cv::Mat mat(_output_data_size,
                  constructor._input_data_size, CV_TYPE);
      Random(mat);
      _Whos.emplace_back(mat);
      _optimizers.emplace_back(Optimizer::create(opt));
    } else {
      _Whos.reserve(constructor._prev_neurons_idx.size());
      _optimizers.reserve(constructor._prev_neurons_idx.size());
      for (auto prev : _prev_neurons_idx) {
        cv::Mat mat(_output_data_size,
                    _netWork_ptr->neuron(prev)->output_data_size(), CV_TYPE);
        Random(mat);
        _Whos.emplace_back(mat);
        _optimizers.emplace_back(Optimizer::create(opt));
      }
    }

  } else {
    _Whos.reserve(constructor._Whos.size());
    _optimizers.reserve(constructor._Whos.size());

    LOG(ERROR) << "constructor input who has, " << id() << "," << constructor._Whos.size();
    for (auto &Who : constructor._Whos) {
      CHECK(Who.rows == output_data_size()) << "Who size = "
        << Who.rows << "," << Who.cols << "; node_num =" << output_data_size();
      _Whos.emplace_back(Who);
      _optimizers.emplace_back(Optimizer::create(opt));
    }
  }

  if (!_prev_neurons_idx.empty()) {
    for (auto prev : _prev_neurons_idx) {
      _prev_neurons_error.emplace(prev, Matrix());
      _netWork_ptr->neuron(prev)->regist_next_neuron(id());
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

  c._Whos.resize(_Whos.size());
  for (int i = 0; i < _Whos.size(); i++) {
    _Whos[i].download(c._Whos[i]);
  }
  return;
}

void Neuron::query(const Matrix &in){
  MicrosecondTimer timer(__func__);
  timer.begin();
  _in = &in;
  cv::cuda::gemm(Who(0), in, 1, Matrix(), 0, _processing, 0, cu_stream);

  CHECK(check(processing())) << "neuron_" << id() << ",_process " << cv::Mat(processing()).t();
  _active_func(_processing);
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << cv::Mat(processing()).t();
  timer.end();
//  LOG(ERROR) << "neuron_" << id() << " output " << _processing.size();
  return;
}

void Neuron::query() {
  MicrosecondTimer timer(__func__), timer1;
  timer.begin();
  const size_t prev_num = num_prev();
  _in = nullptr;
  timer1.begin("setto 0");
  _processing.setTo(0);
  timer1.end();
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << cv::Mat(processing()).t();
  for (size_t i = 0; i < prev_num; i++) {
    timer1.begin("gemm");
    cv::cuda::gemm(Who(i), _netWork_ptr->neuron(_prev_neurons_idx[i])->processing(), 1, _processing, 1, _tmp, 0, cu_stream);
    timer1.end();
    timer1.begin("copyTo");
    _tmp.copyTo(_processing);
    timer1.end();
  }
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << cv::Mat(processing()).t();
  _active_func(_processing);
  CHECK(check(processing())) << "neuron_" << id() << ",_process " << cv::Mat(processing()).t();
  timer.end();
  return;
}

void Neuron::back_propogate(
  const float learning_rate, const Matrix &error) {
  MicrosecondTimer timer(__func__), timer1;
  timer.begin();
  const size_t prev_num = num_prev();
  bool cross = false;
  const bool use_cross_entropy = true;

  if (use_cross_entropy &&
      _next_neurons_idx.empty()
      && (_active == "Softmax" || _active == "Sigmoid")) {
    ///(Cross Entropy) & Softmax
    ///derivate (Tk - Ok) * Oj, Oj is the input from jth neuron of last level
    cross = true;
  } else {
    _derivatives_func(_processing);
  }
#define TEST_OPTIMIZER true
  CHECK(check(processing())) << "neuron_" << id() << ",_process "
      << cv::Mat(processing()).t() << "," << id();
  for (size_t i = 0; i < prev_num; i++) {
    timer1.begin("gemm back");
    cv::cuda::gemm(
        Who(i), error, 1.0, cv::noArray(), 0,
        _prev_neurons_error.at(_prev_neurons_idx[i]),
        cv::GEMM_1_T, cu_stream);
    timer1.end();
    timer1.begin("update who");
    if (cross) {
#if TEST_OPTIMIZER
      cv::cuda::add(
          Who(i),
          _optimizers[i]->UpdateParameter(
              error, Matrix(), _netWork_ptr->neuron(_prev_neurons_idx[i])->processing()),
           _Whos[i], cv::noArray(), -1, cu_stream);
#else
      _Whos[i] += learning_rate *
                  error * (_netWork_ptr->neuron(_prev_neurons_idx[i])->processing()).t();
#endif
//      LOG(ERROR) << "cross";
    } else {
#if TEST_OPTIMIZER
      cv::cuda::add(
          Who(i),
          _optimizers[i]->UpdateParameter(
              error,processing(), _netWork_ptr->neuron(_prev_neurons_idx[i])->processing()),
          _Whos[i], cv::noArray(), -1, cu_stream);
#else
      _Whos[i] += learning_rate *
                  (error.mul(_processing)) * (_netWork_ptr->neuron(_prev_neurons_idx[i])->processing()).t();
#endif
    }
    timer1.end();
    CHECK(check(Who(i))) << "neuron_" << id() << ",Who(" << i << ")" << cv::Mat(processing()).t();
  }
  if (_in != nullptr) {
    timer1.begin("update who0");
    CHECK(prev_num == 0) << prev_num << "," << id();
#if TEST_OPTIMIZER
    cv::cuda::add(
        Who(0),
        _optimizers[0]->UpdateParameter(
            error, processing(), *_in),
        _Whos[0], cv::noArray(), -1, cu_stream);
#else
    _Whos[0] += learning_rate *
                (error.mul(_processing)) * _in->t();
#endif
//    LOG(ERROR) << "update who of the first one";
    CHECK(check(Who(0))) << "neuron_" << id() << ",Who 0 " << cv::Mat(processing()).t();
    timer1.end();
  }
  timer.end();
  return;
}

void Neuron::back_propogate(const float learning_rate) {
  MicrosecondTimer timer(__func__);
  timer.begin();
  if (_error.empty()) {
    _error.create(output_data_size(), 1, CV_TYPE);
  }
  _error.setTo(0);
  for (auto next : _next_neurons_idx) {
    cv::cuda::add(
        _error, _netWork_ptr->neuron(next)->prev_error(id()),
        _error, cv::noArray(), -1, cu_stream);
  }
  back_propogate(learning_rate, _error);
  timer.end();
  return;
}

void Neuron::regist_next_neuron(const int i) {
  if (!is_next_neuron(i)) {
    _next_neurons_idx.emplace_back(i);
    std::sort(_next_neurons_idx.begin(), _next_neurons_idx.end());
  }
};

bool Neuron::is_next_neuron(const int i) const {
  for (auto &id : _next_neurons_idx) {
    if (i == id) {
      return true;
    }
  }
  return false;
};

bool Neuron::is_prev_neuron(const int i) const {
  for (auto &id : _prev_neurons_idx) {
    if (i == id) {
      return true;
    }
  }
  return false;
}

bool Neuron::check_consistency() const {
  if (_Whos.size() != _optimizers.size()) {
    LOG(ERROR) << "Neuron_" << id() << " who and optimzer size dismatch ("
               << _Whos.size() << "," << _optimizers.size() << ")";
    return false;
  }

  if (_Whos.size() != _prev_neurons_idx.size()) {
    LOG(ERROR) << "Neuron_" << id() << " who and _prev_neurons_idx size dismatch ("
               << _Whos.size() << "," << _prev_neurons_idx.size() << ")";
    return false;
  }

  if (_Whos.size() != _prev_neurons_error.size()) {
    LOG(ERROR) << "Neuron_" << id() << " who and _prev_neurons_error size dismatch ("
               << _Whos.size() << "," << _prev_neurons_error.size() << ")";
    return false;
  }

  const size_t prev_size = _Whos.size();
  for (size_t i = 0; i < prev_size; i++) {
    const cv::Size who_size = Who(i).size();
    auto &prev_neuron = _netWork_ptr->neuron(_prev_neurons_idx[i]);
    if (who_size.width != prev_neuron->output_data_size()) {
      LOG(ERROR) << "Neuron_" << id() << " who[" << i
                 << "] size dismatch with Neuron_" << _prev_neurons_idx[i]
                 << " output size(" << who_size.width << ","
                 << prev_neuron->output_data_size();
      return false;
    }

    if (who_size.height != output_data_size()) {
      LOG(ERROR) << "Neuron_" << id() << " who[" << i
                 << "] size dismatch with output " << who_size.height << ","
                 << output_data_size();
      return false;
    }

    if (!prev_neuron->is_next_neuron(id())) {
      LOG(ERROR) << "Neuron_" << id() << " is not the next neuron of Neuron_" << prev_neuron->id();
      return false;
    }
  }

  const size_t next_size = _next_neurons_idx.size();
  for (size_t i = 0; i < next_size; i++) {
    if (!(_netWork_ptr->neuron(_next_neurons_idx[i])->is_prev_neuron(id()))) {
      LOG(ERROR) << "Neuron_" << id() << " is not in the list of Neuron_"
                 << _next_neurons_idx[i] << "'s prev neurons";
      return false;
    }
  }

  return true;
}

MulNetWork::MulNetWork(const std::vector<NeuronConstructor> &constructors) {
  const size_t size = constructors.size();
  for (size_t i = 0; i < size; i++) {
    _neurons.emplace_back(std::make_shared<Neuron>(constructors[i]));
    CHECK(_neurons.back()->id() == i) << _neurons.back()->id() << "," << i;
  }
  CHECK(check_consistency());
}

const Matrix& MulNetWork::query(const Matrix &in) {
  MicrosecondTimer timer(__func__);
  timer.begin();
  neuron(0)->query(in);
  for (size_t i = 1; i < neurons_num(); i++) {
    neuron(i)->query();
  }
  timer.end();
  return neurons().back()->processing();
}

scalar MulNetWork::train(const Matrix &in, const Matrix &target, const float learning_rate) {
  MicrosecondTimer timer(__func__);
  timer.begin();
  auto &query_result = query(in);
  cv::cuda::subtract(target, query_result, _last_error, cv::noArray(), -1, cu_stream);
  double loss = cv::cuda::norm(_last_error, cv::NORM_L2);
//  LOG(INFO) << "loss = " << loss << "," << cur_error.t();
  auto neuron = neurons().rbegin();
  (*neuron)->back_propogate(learning_rate, _last_error);
  for (neuron++; neuron != neurons().rend(); neuron++) {
    (*neuron)->back_propogate(learning_rate);
  }
  timer.end();
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
bool MulNetWork::check_consistency() const {
  const size_t size = neurons_num();
  for (size_t i = 0; i < size; i++) {
    auto n = neuron(i);
    if (n->id() != i) {
      LOG(ERROR) << "neuron_" << n->id() << " at wrong position " << i << "," << size;
      return false;
    }
    if (!n->check_consistency()) {
      LOG(ERROR) << "Neuron_" << n->id() << " check fail";
      return false;
    }
  }
  for (auto &n : neurons()) {

  }
}
}//namespace AiLearning