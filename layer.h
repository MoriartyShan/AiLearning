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
  Layer(const int in, const int out);

  const cv::Mat& Who() const {return _Who;}
  const cv::Mat& processing() const {return _processing;}
  const int input_size() const {return _Who.cols;}
  const int output_size() const {return _Who.rows;}
  int id() const {return _id;}

  const cv::Mat& query(const cv::Mat &in);
  const cv::Mat back_propogate(
      const float learning_rate, const cv::Mat &error);

};

class MulNetWork {
private:
  std::vector<std::shared_ptr<Layer>> _layers;
public:
  MulNetWork() {}
  MulNetWork(const std::vector<int> &nodes);
  size_t layer_nb() const {return _layers.size();}
  int input_size() const {return _layers.front()->input_size();}
  int output_size() const {return _layers.back()->output_size();}

  void train(
      const cv::Mat &in, const cv::Mat &target, const float learning_rate = 0.1);

  const cv::Mat& query(const cv::Mat &in);

  void write(const std::string &path) const;

  void read(const std::string &path);

};

}


#endif //NEURALNETWORK_LAYER_H
