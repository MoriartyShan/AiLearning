//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H
#include "common.h"
#include "glog/logging.h"
#include <memory>

namespace AiLearning{
class Neuron{
protected:
  cv::Mat _Who;
  cv::Mat _processing;
  const cv::Mat *_in;
  const int _id;

  static int global_index() {
    static int idx = 0;
    return idx++;
  }
public:
  Neuron(const cv::Mat &Who): _Who(Who.clone()), _id(global_index()) {}
  Neuron(const int in, const int out);

  const cv::Mat& Who() const {return _Who;}
  const cv::Mat& processing() const {return _processing;}
  const int input_size() const {return _Who.cols;}
  const int output_size() const {return _Who.rows;}
  int id() const {return _id;}

  virtual const cv::Mat& query(const cv::Mat &in) = 0;
  virtual const cv::Mat back_propogate(
      const float learning_rate, const cv::Mat &error) = 0;
  virtual const std::string& type() const = 0;

};

class SigmoidNeuron : public Neuron{
private:
  static const std::string _type;
public:
  SigmoidNeuron(const cv::Mat &Who): Neuron(Who) {}
  SigmoidNeuron(const int in, const int out) : Neuron(in, out) {};

  const cv::Mat& query(const cv::Mat &in) override;
  const cv::Mat back_propogate(
    const float learning_rate, const cv::Mat &error) override;
  const std::string& type() const override {return _type;}

};

class ELUNeuron : public Neuron{
private:
  const static std::string _type;
public:
  ELUNeuron(const cv::Mat &Who): Neuron(Who) {}
  ELUNeuron(const int in, const int out) : Neuron(in, out) {};

  const cv::Mat& query(const cv::Mat &in) override;
  const cv::Mat back_propogate(
    const float learning_rate, const cv::Mat &error) override;
  const std::string& type() const override {return _type;}
};

class MulNetWork {
private:
  std::vector<std::shared_ptr<Neuron>> _layers;
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

  const std::shared_ptr<Neuron> layer(const int i) const {return _layers[i];}

};

}


#endif //NEURALNETWORK_LAYER_H
