//
// Created by moriarty on 5/17/21.
//
#include "layer.h"
int main() {
  AiLearning::MulNetWork netWork;
  std::vector<std::array<cv::Mat, 3>> weights;
  weights.reserve(266);

  const std::string path = "/home/moriarty/Files/Downloads/weight/";

#define READ(a) do { \
  netWork.read(path + (a));\
  weights.emplace_back();\
  auto &back = weights.back();\
  back[0] = netWork.layer(0)->Who().clone();\
  back[1] = netWork.layer(1)->Who().clone();\
  back[2] = netWork.layer(2)->Who().clone();\
} while(0);

  READ("init.yaml");

  for (int i = 0; i < 266; i++) {
    READ("weight_" + std::to_string(i));
  }

  std::vector<double> lines = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  std::ofstream file;

  for (int i = 0; i < 3; i++) {
    file.open("layer_" + std::to_string(i) + ".csv");

    for (int k = 0; k < lines.size(); k++) {
      int lid =
      for (int j = 0; j < weights.size(); j++) {
        weights[j][i].at<float>()
      }
    }



  }



};