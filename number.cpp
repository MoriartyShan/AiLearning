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

void create_std(std::vector<cv::Mat> &res) {
  const int num = 10;
  res.reserve(num);
  for (int i = 0; i < num; i++) {
    cv::Mat target(10, 1, CV_TYPE);
    target = 0.0001;
    target.at<scalar>(i, 0) = 0.9999;
    res.emplace_back(target);
  }
}

int create_input(const std::string& line, cv::Mat& res) {
  std::stringstream ss(line);
  res = cv::Mat(784, 1, CV_TYPE);

  int cur;
  char comma;
#define INSTREAM ss >> cur; ss >> comma;
#define GIVE_VALUE(a) res.at<scalar>(a, 0) = (cur / 255.0) * 0.99 + 0.01;
  INSTREAM;

  int std = cur;
  for (int i = 0; i < 783; i++) {
    INSTREAM;
    GIVE_VALUE(i);
  }
  ss >> cur;
  GIVE_VALUE(783);
  //LOG(ERROR) << target.t() << "\n" << res;
  //getchar();
  return std;
}

AiLearning::NetWorks train(const int epoch = 5, AiLearning::NetWorks *input_work = nullptr) {
  const std::string root = FLAGS_data + "/";
  const std::string data = FLAGS_train;
  AiLearning::NetWorks work(784, 100, 10);
  std::vector<cv::Mat> std_res;
  create_std(std_res);

  for (int e = 0; e < epoch; e++) {
    std::ifstream file(root + data);
    CHECK(file.is_open()) << root + data << " open fail";
    std::string line;
    cv::Mat input;
    while (!file.eof()) {
      std::getline(file, line);
      if (!line.empty()) {
        int number = create_input(line, input);
        work.train(input, std_res[number]);
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
  cv::Mat input;
  std::vector<cv::Mat> std_res;
  int right = 0, wrong = 0;

  create_std(std_res);

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      int number = create_input(line, input);
      cv::Mat res = work.query(input);
      auto real = get_res(std_res[number]);
      auto detect = get_res(res);

      LOG(INFO) << "[real, possiblity, detect, possiblity], [" << real.first
                 << "," << real.second << "," << detect.first << ","
                 << detect.second << "]";

      if (real.first == detect.first) {
        right++;
      } else {
        wrong++;
      }

      LOG(INFO) << "tar = " << std_res[number].t();
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
  cv::Mat input;

  int right = 0, wrong = 0;
  scalar loss = 0;
  std::vector<cv::Mat> std_res;
  create_std(std_res);

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      int number = create_input(line, input);
      const cv::Mat res = netWork.query(input);
      std::pair<int, scalar> real = std::pair<int, scalar>(number, 0.999);
      auto detect = get_res(res);
      loss += cv::norm(std_res[number] - res);

      LOG(INFO) << "[real, possiblity, detect, possiblity], [" << real.first
                 << "," << real.second << "," << detect.first << ","
                 << detect.second << "],"
                 << (detect.first == real.first ? "right" : "wrong") << ","
                 << cv::norm(res - std_res[number]);

      if (real.first == detect.first) {
        right++;
      } else {
        wrong++;
      }

      LOG(INFO) << "tar = " << std_res[number].t();
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

    std::vector<cv::Mat> std_res;
    create_std(std_res);
    for (int e = 0; e < epoch; e++) {
      scalar loss = 0;
      int train_size = 0;
      std::ifstream file(root + data);
      CHECK(file.is_open()) << root + data << " open fail";
      std::string line;
      cv::Mat input;

      while (!file.eof()) {
        std::getline(file, line);
        if (!line.empty()) {
          int number = create_input(line, input);
          loss += netWork.train(input, std_res[number], learning_rate);
          train_size++;
        }
      }
      file.close();

      std::pair<scalar, scalar> query_result = query(netWork);
      bool bad_condition = false;
      if (best_loss < 0 || loss < best_loss) {
        best_loss = loss;
        best_epoch = e;
      }

      if (last_loss > 0 && last_loss < loss) {
        bad_condition = true;
      }

      LOG(ERROR) << "epoch[" << e << "]:" << std::setprecision(8)
                 << "accuracy," << query_result.first
                 << ",query loss," << query_result.second
                 << ",best epoch, " << best_epoch
                 << ",best loss, " << best_loss
                 << ",total loss," << loss
                 << ",dataset size," << train_size
                 << ",train loss," << loss/train_size
                 << "," << (bad_condition ? "bad" : "good");

      last_loss = loss;
      netWork.write(FLAGS_weight + "/weight_" + std::to_string(e) + ".yaml", true);
    }

  }

  return 0;
}