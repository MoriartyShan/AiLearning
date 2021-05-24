//
// Created by moriarty on 2021/5/16.
//
#include "basic.h"
#include "neuron.h"
#include "common.h"
#include <glog/logging.h>
#include <gflags/gflags.h>
DEFINE_string(data, "", "path to tran and test data");
DEFINE_string(train, "", "train data name");
DEFINE_string(test, "", "test data name");
DEFINE_string(weight, "./", "write weight files to");
DEFINE_string(model, "", "file to model.yaml");
DEFINE_double(learning_rate, 0.0001, "learning rate");

using scalar = AiLearning::scalar;

bool create_input(const std::string& line, cv::Mat& res, cv::Mat &target) {
  std::stringstream ss(line);
  res = cv::Mat(784, 1, CV_TYPE);
  target = cv::Mat::zeros(10, 1, CV_TYPE);
  int cur;
  char comma;
#define INSTREAM ss >> cur; ss >> comma;
#define GIVE_VALUE(a) res.at<scalar>(a, 0) = (cur / 255.0) * 0.99 + 0.01;
  INSTREAM;
  for (int i = 0; i < 10; i++) {
    target.at<scalar>(i, 0) = 0.0001;
  }
  target.at<scalar>(cur, 0) = 0.9999;

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

AiLearning::NetWorks train(const int epoch = 5, AiLearning::NetWorks *input_work = nullptr) {
  const std::string root = FLAGS_data + "/";
  const std::string data = FLAGS_train;
  AiLearning::NetWorks work(784, 100, 10);

  for (int e = 0; e < epoch; e++) {
    std::ifstream file(root + data);
    CHECK(file.is_open()) << root + data << " open fail";
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

std::pair<int, scalar> get_res(const cv::Mat& res) {
  const int row = 10;
  CHECK(res.cols == 1 && res.rows == row) << res;
  CHECK(res.type() == CV_TYPE);
  double all = 0;
  double max = res.at<scalar>(0, 0);
  int max_index = 0;

  for (int i = 0; i < row; i++) {
    const scalar cur = res.at<scalar>(i, 0);
    all += cur;
    if (cur > max) {
      max = cur;
      max_index = i;
    }
  }
  return std::pair<int, scalar>(max_index, max/all);
}

scalar query(const AiLearning::NetWorks &work) {
  const std::string root = FLAGS_data + "/";
  const std::string data = FLAGS_test;

  std::ifstream file(root + data);
  CHECK(file.is_open()) << root + data << " open fail";
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

      LOG(INFO) << "[real, possiblity, detect, possiblity], [" << real.first
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
             << right / (scalar)(right + wrong);
  return right / (scalar)(right + wrong);
}


std::pair<scalar, scalar> query(AiLearning::MulNetWork &netWork) {
  const std::string root = FLAGS_data + "/";
  const std::string data = FLAGS_test;

  std::ifstream file(root + data);
  CHECK(file.is_open()) << root + data << " open fail";
  std::string line;
  cv::Mat input, target;

  int right = 0, wrong = 0;
  scalar loss = 0;

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      create_input(line, input, target);
      const cv::Mat res = netWork.query(input);
      auto real = get_res(target);
      auto detect = get_res(res);
      loss += cv::norm(target - res);

      LOG(INFO) << "[real, possiblity, detect, possiblity], [" << real.first
                 << "," << real.second << "," << detect.first << ","
                 << detect.second << "],"
                 << (detect.first == real.first ? "right" : "wrong") << "," << cv::norm(res - target);

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
             << right / (scalar)(right + wrong);
  return std::pair<scalar, scalar>(
      right / (scalar)(right + wrong), loss / (wrong + right));
}


int main(int argc, char **argv) {
  const std::string execute = argv[0];
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]); // option --[also]logtostderr
  //        --stderrthreshold=0

  int seed = time(0);
  LOG(ERROR) << "rand seed = " << seed;
  std::srand(seed);
  cv::setRNGSeed(seed);

  if (false) {
    AiLearning::NetWorks work = train(1);
    query(work);
  } else {
    scalar last_loss = -1;
    AiLearning::MulNetWork netWork;
    netWork.read(FLAGS_model);
    netWork.write("/home/moriarty/init_weight.yaml", true);
    const std::string root = FLAGS_data + "/";
    const std::string data = FLAGS_train;
    const int epoch = 100;
    scalar learning_rate = FLAGS_learning_rate;
    int best_epoch = 0;
    scalar best_loss = -1;
    for (int e = 0; e < epoch; e++) {
      scalar loss = 0;
      int train_size = 0;
      std::ifstream file(root + data);
      CHECK(file.is_open()) << root + data << " open fail";
      std::string line;
      cv::Mat input, target;
      while (!file.eof()) {
        std::getline(file, line);
        if (!line.empty()) {
          create_input(line, input, target);
          loss += netWork.train(input, target, learning_rate);
          train_size++;
        }
      }
      file.close();
      std::pair<scalar, scalar> query_result = query(netWork);
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
      if (best_loss < 0 || loss < best_loss) {
        best_loss = loss;
        best_epoch = e;
      }
      LOG(ERROR) << "epoch[" << e << "]:" << std::setprecision(8)
                 << "accuracy," << query_result.first
                 << ",query loss," << query_result.second
                 << ",learning rate," << learning_rate
                 << ",best epoch, " << best_epoch
                 << ",best loss, " << best_loss
                 << ",total loss," << loss
                 << ",dataset size," << train_size
                 << ",train loss," << loss/train_size;
      if (last_loss > 0 && last_loss < loss) {
        learning_rate *= 0.5;
      }
      if (learning_rate < 0.000001) {
        learning_rate = 0.000001;
      }
      last_loss = loss;
      netWork.write(FLAGS_weight + "/weight_" + std::to_string(e) + ".yaml", true);
    }

  }

  return 0;
}