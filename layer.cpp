//
// Created by moriarty on 2021/5/16.
//
#include "layer.h"


namespace AiLearning{
Layer::Layer(const int in, const int out) :
    _id(global_index()) {
  _Who.create(out, in, CV_32FC1);
  Random(_Who);
}


const cv::Mat& Layer::query(const cv::Mat &in) {
  _in = &in;
  _processing = Who() * in;
  ELU(_processing);
  return processing();
}

const cv::Mat Layer::back_propogate(
    const float learning_rate, const cv::Mat &error) {
  cv::Mat prev_error = _Who.t() * error;

  cv::Mat derivatives = _processing.clone();
  derivativesELU(derivatives);

  _Who += learning_rate *
          ((error.mul(derivatives))) * (*_in).t();
  return prev_error;
}

MulNetWork::MulNetWork(const std::vector<int> &nodes) {
  const size_t size = nodes.size();
  _layers.reserve(size - 1);

  for (size_t i = 1; i < size; i++) {
    _layers.emplace_back(std::make_shared<Layer>(nodes[i - 1], nodes[i]));
  }
}

void MulNetWork::train(const cv::Mat &in, const cv::Mat &target, const float learning_rate) {
  cv::Mat cur_error = target - query(in);
  for (auto layer = _layers.rbegin(); layer != _layers.rend(); layer++) {
    LOG(INFO) << "current error = " << cur_error.t();
    cur_error = (*layer)->back_propogate(learning_rate, cur_error);
  }

  return;
}

const cv::Mat& MulNetWork::query(const cv::Mat &in) {
  const cv::Mat *in_ptr = &in;
  for (size_t i = 0; i < layer_nb(); i++) {
    in_ptr = &(_layers[i]->query(*in_ptr));
  }
  return *in_ptr;
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
    _layers.emplace_back(std::make_shared<Layer>(l));
  }
  return;
}

}//namespace AiLearning