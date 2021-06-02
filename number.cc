//
// Created by moriarty on 2021/5/16.
//
#include "basic.h"
#include "neuron.h"
#include "common.h"
#include "matrix_utils.h"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <fstream>

DEFINE_string(data, "", "path to tran and test data");
DEFINE_string(train, "", "train data name");
DEFINE_string(test, "", "test data name");
DEFINE_string(weight, "./", "write weight files to");
DEFINE_string(model, "", "file to model.yaml");
DEFINE_double(learning_rate, 0.0001, "learning rate");

using scalar = AiLearning::scalar;

void create_std(std::vector<AiLearning::Matrix> &res) {
  const int num = 10;
  res.reserve(num);
  for (int i = 0; i < num; i++) {
    cv::Mat target(10, 1, CV_TYPE);
    target.setTo(0.0001);
    target.at<scalar>(i, 0) = 0.9999;
    res.emplace_back();
    AiLearning::MatrixUtils::CopyTo(target, res.back());
//    LOG(ERROR) << "target [" << i << "] = " << target;
  }
}

int create_input(const std::string& line, AiLearning::Matrix& _res) {
  std::stringstream ss(line);
  cv::Mat res(784, 1, CV_TYPE);
  scalar *data = (scalar *)res.data;
  int cur;
  char comma;
#define INSTREAM ss >> cur; ss >> comma;
#define GIVE_VALUE(a) data[a] = (cur / 255.0) * 0.99 + 0.01;
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
  AiLearning::MatrixUtils::CopyTo(res, _res);
  return std;
}

AiLearning::NetWorks train(const int epoch = 5, AiLearning::NetWorks *input_work = nullptr) {
  const std::string root = FLAGS_data + "/";
  const std::string data = FLAGS_train;
  AiLearning::NetWorks work(784, 100, 10);
  std::vector<cv::Mat> std_res;
  {
    std::vector<AiLearning::Matrix> std_res_;
    create_std(std_res_);
    for (auto &m : std_res_) {
      std_res.emplace_back();
      AiLearning::MatrixUtils::CopyTo(m, std_res.back());
    }
  }


  for (int e = 0; e < epoch; e++) {
    std::ifstream file(root + data);
    CHECK(file.is_open()) << root + data << " open fail";
    std::string line;
    AiLearning::Matrix input;
    cv::Mat input_cpu;
    while (!file.eof()) {
      std::getline(file, line);
      if (!line.empty()) {
        int number = create_input(line, input);
        AiLearning::MatrixUtils::CopyTo(input, input_cpu);
        work.train(input_cpu, std_res[number]);
      }
    }
    file.close();
  }

  work.write_work(root + "/weight.yaml");
  return work;
}

std::pair<int, scalar> get_res(const cv::Mat& res) {
  const int row = 10;
  CHECK(res.cols == 1 && res.rows == row) << cv::Mat(res).t();
  CHECK(res.type() == CV_TYPE);
  const scalar *data = (scalar *)res.data;
  double all = 0;
  double max = data[0];
  int max_index = 0;

  for (int i = 0; i < row; i++) {
    const scalar cur = data[i];
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
  AiLearning::Matrix input_;
  std::vector<cv::Mat> std_res;
  int right = 0, wrong = 0;

  {
    std::vector<AiLearning::Matrix> std_res_;
    create_std(std_res_);
    for (auto &m : std_res_) {
      std_res.emplace_back();
      AiLearning::MatrixUtils::CopyTo(m, std_res.back());
    }
  }

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      int number = create_input(line, input_);
      AiLearning::MatrixUtils::CopyTo(input_, input);
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
  AiLearning::Matrix input;

  int right = 0, wrong = 0;
  scalar loss = 0;
  std::vector<AiLearning::Matrix> std_res;
  create_std(std_res);

  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      int number = create_input(line, input);
      const AiLearning::Matrix &res = netWork.query(input);
      std::pair<int, scalar> real = std::pair<int, scalar>(number, 0.999);
      cv::Mat cvres;
      AiLearning::MatrixUtils::CopyTo(res, cvres);
      auto detect = get_res(cvres);
      AiLearning::Matrix error;
      AiLearning::MatrixUtils::subtract(std_res[number], res, error);
      double this_loss = AiLearning::MatrixUtils::norml2(error);
      loss += this_loss;

      LOG(INFO) << "[real, possiblity, detect, possiblity], [" << real.first
                 << "," << real.second << "," << detect.first << ","
                 << detect.second << "],"
                 << (detect.first == real.first ? "right" : "wrong") << ","
                 << this_loss;

      if (real.first == detect.first) {
        right++;
      } else {
        wrong++;
      }

      LOG(INFO) << "tar = " << std_res[number];
      LOG(INFO) << "res = " << res;
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
#if false && defined(EIGEN_MODE)
  Eigen::initParallel();
  LOG(ERROR) << "set eigen parallel thread number " << Eigen::nbThreads();
#endif

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

    std::vector<AiLearning::Matrix> std_res;
    create_std(std_res);
    for (int e = 0; e < epoch; e++) {
      scalar loss = 0;
      int train_size = 0;
      std::ifstream file(root + data);
      CHECK(file.is_open()) << root + data << " open fail";
      std::string line;
      AiLearning::Matrix input;

      while (!file.eof()) {
        std::getline(file, line);
        if (!line.empty()) {
          int number = create_input(line, input);
          loss += netWork.train(input, std_res[number], learning_rate);
          train_size++;

        }
//        if (train_size % 1000 == 1) {
//          LOG(ERROR) << "trained data " << train_size;
//        }
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