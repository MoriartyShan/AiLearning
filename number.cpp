//
// Created by moriarty on 2021/5/16.
//
#include "basic.h"
#include "layer.h"
#include <glog/logging.h>
#include <gflags/gflags.h>


bool create_input(const std::string& line, cv::Mat& res, cv::Mat &target) {
  std::stringstream ss(line);
  res = cv::Mat(784, 1, CV_32FC1);
  target = cv::Mat::zeros(10, 1, CV_32FC1);
  int cur;
  char comma;
#define INSTREAM ss >> cur; ss >> comma;
#define GIVE_VALUE(a) res.at<float>(a, 0) = (cur / 255.0) * 0.99 + 0.01;
  INSTREAM;
  for (int i = 0; i < 10; i++) {
    target.at<float>(i, 0) = 0.0001;
  }
  target.at<float>(cur, 0) = 0.9999;

  for (int i = 0; i < 783; i++) {
    INSTREAM;
    GIVE_VALUE(i);
  }
  ss >> cur;
  GIVE_VALUE(783);
  //LOG(ERROR) << target.t() << "\n" << res;
  //getchar();
  return true;
}

AiLearning::NetWorks train(const int epoch = 5) {
  const std::string root = "/home/moriarty/Datasets/python_learn_network/";
  const std::string data = "mnist_train.csv";
  AiLearning::NetWorks work(784, 100, 10);

  for (int e = 0; e < epoch; e++) {
    std::ifstream file(root + "/" + data);
    CHECK(file.is_open()) << root + "/" + data << " open fail";
    std::string line;
    cv::Mat input, target;
    while (!file.eof()) {
      std::getline(file, line);
      if (!line.empty()) {
        create_input(line, input, target);
        work.train(input, target);
      }
    }
    file.close();
  }

  work.write_work(root + "/weight.yaml");
  return work;
}

std::pair<int, float> get_res(const cv::Mat& res) {
  const int row = 10;
  CHECK(res.cols == 1 && res.rows == row) << res;
  CHECK(res.type() == CV_32FC1);
  double all = 0;
  double max = res.at<float>(0, 0);
  int max_index = 0;

  for (int i = 0; i < row; i++) {
    const float cur = res.at<float>(i, 0);
    all += cur;
    if (cur > max) {
      max = cur;
      max_index = i;
    }
  }
  return std::pair<int, float>(max_index, max/all);
}

void query(const AiLearning::NetWorks &work) {
  const std::string root = "/home/moriarty/Datasets/python_learn_network/";
  const std::string data = "mnist_test.csv";

  std::ifstream file(root + "/" + data);
  CHECK(file.is_open()) << root + "/" + data << " open fail";
  std::string line;
  cv::Mat input, target;

  int right = 0, wrong = 0;

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      create_input(line, input, target);
      cv::Mat res = work.query(input);
      auto real = get_res(target);
      auto detect = get_res(res);

      LOG(ERROR) << "[real, possiblity, detect, possiblity], [" << real.first
                 << "," << real.second << "," << detect.first << ","
                 << detect.second << "]";

      if (real.first == detect.first) {
        right++;
      } else {
        wrong++;
      }

      LOG(INFO) << "tar = " << target.t();
      LOG(INFO) << "res = " << res.t();
    }
  }
  file.close();

  LOG(ERROR) << "[right, wrong], [" << right << "," << wrong << "]="
             << right / (float)(right + wrong);
}


float query(AiLearning::MulNetWork &netWork) {
  const std::string root = "/home/moriarty/Datasets/python_learn_network/";
  const std::string data = "mnist_test.csv";

  std::ifstream file(root + "/" + data);
  CHECK(file.is_open()) << root + "/" + data << " open fail";
  std::string line;
  cv::Mat input, target;

  int right = 0, wrong = 0;

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      create_input(line, input, target);
      const cv::Mat res = netWork.query(input);
      auto real = get_res(target);
      auto detect = get_res(res);

      LOG(INFO) << "[real, possiblity, detect, possiblity], [" << real.first
                 << "," << real.second << "," << detect.first << ","
                 << detect.second << "],"
                 << (detect.first == real.first ? "right" : "wrong");

      if (real.first == detect.first) {
        right++;
      } else {
        wrong++;
      }

      LOG(INFO) << "tar = " << target.t();
      LOG(INFO) << "res = " << res.t();
    }
  }
  file.close();

  LOG(INFO) << "[right, wrong], [" << right << "," << wrong << "]="
             << right / (float)(right + wrong);
  return right / (float)(right + wrong);
}


int main(int argc, char **argv) {
  const std::string execute = argv[0];
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]); // option --[also]logtostderr
  //        --stderrthreshold=0


  srand(time(0));
  if (false) {
    AiLearning::NetWorks work = train();
    query(work);
  } else {
    std::vector<int> nodes = {784, 80, 186, 10};
    AiLearning::MulNetWork netWork(nodes);
//    netWork.read("/home/moriarty/Documents/4.yaml");

    const std::string root = "/home/moriarty/Datasets/python_learn_network/";
    const std::string data = "mnist_train.csv";
    const int epoch = 5;
    float last_rate = 0;
    float learning_rate = 0.1;
    for (int e = 0; e < epoch; e++) {

      std::ifstream file(root + "/" + data);
      CHECK(file.is_open()) << root + "/" + data << " open fail";
      std::string line;
      cv::Mat input, target;
      while (!file.eof()) {
        std::getline(file, line);
        if (!line.empty()) {
          create_input(line, input, target);
          netWork.train(input, target);
        }
      }
      file.close();
      float rate = query(netWork);
#if 0
      if (rate - last_rate < 0) {
        learning_rate = 0.1;
      } else if (rate - last_rate < 0.01) {
        learning_rate = 10.0 * (rate - last_rate);
      } else if (rate - last_rate < 0.1) {
        learning_rate = rate - last_rate;
      } else {
        learning_rate = 0.1;
      }
#endif
      LOG(ERROR) << "epoch[" << e << "] = " << rate << ", learning rate change to " << learning_rate;
      last_rate = rate;
      netWork.write("/home/moriarty/Documents/weight_" + std::to_string(e) + ".yaml");
    }

  }

  return 0;
}