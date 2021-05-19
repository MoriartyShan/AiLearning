//
// Created by moriarty on 2021/5/16.
//
#include "layer.h"


namespace AiLearning{
Neuron::Neuron(const int in, const int out, const std::string &active) :
    _id(global_index()), _active(active) {
  set_active();
  _Who.create(out, in, CV_TYPE);
  Random(_Who);
}

const cv::Mat& Neuron::query(const cv::Mat &in){
#if 0
  _in = &in;
  _processing = Who() * in;
  _active_func(_processing);
  return processing();
#else
  _in = &in;
  _processing = Who() * in;
  Sigmoid(_processing);
  return processing();
#endif
}

const cv::Mat Neuron::back_propogate(
  const float learning_rate, const cv::Mat &error) {
#if 0
  cv::Mat prev_error = _Who.t() * error;

  _derivatives_func(_processing);

  _Who += learning_rate *
          (error.mul(_processing)) * (*_in).t();
  return prev_error;
#else
  cv::Mat prev_error = _Who.t() * error;
  _Who += learning_rate *
          ((error.mul(_processing)).mul(1 - _processing)) * (*_in).t();
  return prev_error;
#endif
}

MulNetWork::MulNetWork(const std::vector<int> &nodes) {
  const size_t size = nodes.size();
  _layers.reserve(size - 1);
  const std::string active = "Sigmoid";
  for (size_t i = 1; i < size; i++) {
    std::shared_ptr<Neuron> ptr =
        std::make_shared<Neuron>(nodes[i - 1], nodes[i], active);
    _layers.emplace_back(static_cast<std::shared_ptr<Neuron>>(ptr));
  }
}

void MulNetWork::train(const cv::Mat &in, const cv::Mat &target, const float learning_rate) {
  cv::Mat cur_error = (target - query(in));
//  LOG(ERROR) << "loss = " << cv::norm(cur_error) << "," << cur_error.t();
#if 0
  derivativesSoftmax(_softmax);
  cur_error = cur_error.mul(_softmax);
#endif
  for (auto layer = _layers.rbegin(); layer != _layers.rend(); layer++) {
    cur_error = (*layer)->back_propogate(learning_rate, cur_error);
  }

  return;
}

const cv::Mat& MulNetWork::query(const cv::Mat &in) {
  const cv::Mat *in_ptr = &in;
  for (size_t i = 0; i < layer_nb(); i++) {
    in_ptr = &(_layers[i]->query(*in_ptr));
  }

  _softmax = *in_ptr;
#if 0
  _sum = Softmax(_softmax);
#endif
  return _softmax;
}

void MulNetWork::write(const std::string &path) const {
  std::vector<cv::Mat> layers(_layers.size());

  for (int i = 0; i < layer_nb(); i++) {
    layers[i] = _layers[i]->Who();
  }

  cv::FileStorage file(path, cv::FileStorage::WRITE);
  CHECK(file.isOpened()) << "path:" << path << " open fail";
  cv::write(file, "layer_nb", (int)layer_nb());
  cv::write(file, "weights", layers);
  file.release();
}

void MulNetWork::read(const std::string &path) {
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
  return;
}

}//namespace AiLearning