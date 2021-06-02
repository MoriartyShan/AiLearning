//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_BASIC_H
#define NEURALNETWORK_BASIC_H
#include "common.h"
#include "neuron.h"
#include <opencv2/opencv.hpp>


namespace AiLearning {
class NetWorks {
private:
  double learning_rate = 0.1;
  cv::Mat _Wih; //input_hidden
  cv::Mat _Who; //hidden_output
  int _inode;
  int _hnode;
  int _onode;
public:
  NetWorks(std::string &path) {
    read_work(path);
  }
  NetWorks(const MulNetWork &mul) {
#if defined(OPENCV_CUDA_MODE) || defined(OPENCV_CPU_MODE)
    _Wih = cv::Mat(mul.neuron(0)->Who(0).clone());
    _Who = cv::Mat(mul.neuron(1)->Who(0).clone());
#elif defined(EIGEN_MODE)
    cv::eigen2cv(mul.neuron(0)->Who(0), _Wih);
    cv::eigen2cv(mul.neuron(1)->Who(0), _Who);
#else
#error "You must specify one mode"
#endif
  }

  NetWorks(const int inode, const int hnode, const int onode);

  cv::Mat query(const cv::Mat &in) const;

  void train(const cv::Mat &in, const cv::Mat &target);

  void write_work(const std::string &path);

  void read_work(const std::string &path);

};

}//namespace AiLearning
#endif //NEURALNETWORK_BASIC_H
