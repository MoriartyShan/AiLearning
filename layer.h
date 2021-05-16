//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H
#include "common.h"
#include "glog/logging.h"
#include <memory>

namespace AiLearning{
class Layer{
private:
  cv::Mat _Who;
  cv::Mat _processing;
  const cv::Mat *_in;
  const int _id;

  static int global_index() {
    static int idx = 0;
    return idx++;
  }
public:
  Layer(const cv::Mat &Who): _Who(Who.clone()), _id(global_index()) {}
  Layer(const int in, const int out) :
      _id(global_index()) {
    _Who.create(out, in, CV_32FC1);
    Random(_Who);
  }

  const cv::Mat& Who() const {return _Who;}
  const cv::Mat& processing() const {return _processing;}
  const int input_size() const {return _Who.cols;}
  const int output_size() const {return _Who.rows;}
  int id() const {return _id;}

  const cv::Mat& query(const cv::Mat &in) {
    _in = &in;
    _processing = Who() * in;
    Sigmoid(_processing);
    return processing();
  }

  const cv::Mat back_propogate(
      const float learning_rate, const cv::Mat &error) {
    cv::Mat prev_error = _Who.t() * error;
    _Who += learning_rate *
        ((error.mul(_processing)).mul(1 - _processing)) * (*_in).t();
    return prev_error;
  }
};

class MulNetWork {
private:
  std::vector<std::shared_ptr<Layer>> _layers;
public:
  MulNetWork() {}
  MulNetWork(const std::vector<int> &nodes) {
    const size_t size = nodes.size();
    _layers.reserve(size - 1);

    for (size_t i = 1; i < size; i++) {
      _layers.emplace_back(std::make_shared<Layer>(nodes[i - 1], nodes[i]));
    }
  }
  size_t layer_nb() const {return _layers.size();}
  int input_size() const {return _layers.front()->input_size();}
  int output_size() const {return _layers.back()->output_size();}

  void train(const cv::Mat &in, const cv::Mat &target, const float learning_rate = 0.1) {
    cv::Mat cur_error = target - query(in);
    for (auto layer = _layers.rbegin(); layer != _layers.rend(); layer++) {
      cur_error = (*layer)->back_propogate(learning_rate, cur_error);
    }

    return;
  }

  const cv::Mat& query(const cv::Mat &in) {
    const cv::Mat *in_ptr = &in;
    for (size_t i = 0; i < layer_nb(); i++) {
      in_ptr = &(_layers[i]->query(*in_ptr));
    }
    return *in_ptr;
  }

  void write(const std::string &path) const {
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

  void read(const std::string &path) {
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

};

}


#endif //NEURALNETWORK_LAYER_H
