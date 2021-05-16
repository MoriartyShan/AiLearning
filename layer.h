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
  const size_t _layer_nb;
public:
  MulNetWork(const std::vector<int> &nodes) : _layer_nb(nodes.size() - 1){
    const size_t size = nodes.size();
    _layers.reserve(size - 1);

    for (size_t i = 1; i < size; i++) {
      _layers.emplace_back(std::make_shared<Layer>(nodes[i - 1], nodes[i]));
    }
  }
  size_t layer_nb() const {return _layer_nb;}


  void train(const cv::Mat &in, const cv::Mat &target, const float learning_rate = 0.1) {
    cv::Mat cur_error = target - query(in);
    for (auto layer = _layers.rbegin(); layer != _layers.rend(); layer++) {
      cur_error = (*layer)->back_propogate(learning_rate, cur_error);
    }

    return;
  }

  const cv::Mat& query(const cv::Mat &in) {
    const cv::Mat *in_ptr = &in;
    for (size_t i = 0; i < _layer_nb; i++) {
      in_ptr = &(_layers[i]->query(*in_ptr));
    }
    return *in_ptr;
  }

};

}


#endif //NEURALNETWORK_LAYER_H
