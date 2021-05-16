//
// Created by moriarty on 2021/5/16.
//

#ifndef NEURALNETWORK_BASIC_H
#define NEURALNETWORK_BASIC_H
#include "common.h"
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

  NetWorks(const int inode, const int hnode, const int onode);

  cv::Mat query(const cv::Mat &in) const;

  void train(const cv::Mat &in, const cv::Mat &target);

  void write_work(const std::string &path);

  void read_work(const std::string &path);

};

}//namespace AiLearning
#endif //NEURALNETWORK_BASIC_H