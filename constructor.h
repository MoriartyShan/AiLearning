//
// Created by moriarty on 6/1/21.
//

#ifndef NEURALNETWORK_CONSTRUCTOR_H
#define NEURALNETWORK_CONSTRUCTOR_H
#include <opencv2/opencv.hpp>


namespace AiLearning {
class MulNetWork;

struct NeuronConstructor {

  const MulNetWork *_netWork_ptr;

  std::vector<int> _prev_neurons_idx;
  std::vector<cv::Mat> _Whos;

  std::vector<int> _next_neurons_idx;

  int _input_data_size;
  int _output_data_size;

  std::string _active_func;

  bool check_data() const;

  void write(
    cv::FileStorage &fs, const int id,
    const bool output_matrix = true) const;

  bool read(cv::FileStorage &fs, const int id);

};
}//namespace AiLearning

#endif //NEURALNETWORK_CONSTRUCTOR_H
