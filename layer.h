//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H
#include "common.h"

namespace AiLearning{
class Layer{
private:
  Layer *_prev, *_next;
  cv::Mat _Who;
  cv::Mat _processing;
  const cv::Mat *_in;
  const int _node;
  const int _id;

  static int global_index() {
    static int idx = 0;
    return idx++;
  }
public:
  Layer(const int node): _node(node), _id(global_index()){}

  const cv::Mat& Who() const {return _Who;}
  const cv::Mat& processing() const {return _processing;}
  const int node() const {return _node;}
  int id() const {return _id;}
  void init(Layer *prev, Layer *next, const int next_node = 0) {
    _prev = prev;
    if (next != nullptr) {
      _next = next;
      _Who.create(_next->node(), node(), CV_32FC1);
    } else {
      _Who.create(next_node, node(), CV_32FC1);
    }

    Random(_Who);
  }

  Layer *query(const cv::Mat &in) {
    _in = &in;
    _processing = Who() * in;
    Sigmoid(_processing);
    if (_next != nullptr) {
      return _next->query(processing());
    }
    return this;
  }

  const cv::Mat back_process(
      const float learning_rate, const cv::Mat &error) {
    cv::Mat prev_error = _Who.t() * error;
    _Who += learning_rate *
        ((error.mul(_processing)).mul(1 - _processing)) * (*_in).t();
    return prev_error;
  }

  void train(const cv::Mat &in, const cv::Mat &target) {
    const float learning_rate = 0.1;

    Layer *last = query(in);
    cv::Mat cur_error = target - last->processing();
    while (last != nullptr) {
      cur_error = last->back_process(learning_rate, cur_error);
      last = last->_prev;
    }
    return;
  }

};

}


#endif //NEURALNETWORK_LAYER_H
