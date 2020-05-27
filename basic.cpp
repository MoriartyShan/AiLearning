#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <sstream>

void Random(cv::Mat &matrix) {
  CHECK(matrix.type() == CV_32FC1);
#define RAND8BIT() (((uint32_t)rand()) & 0xff)
#define RAND() (RAND8BIT() | (RAND8BIT() << 8) | (RAND8BIT() << 16) | (RAND8BIT() << 24) )
#define RANDOM(a) (((RAND() / (double)0xffffffff) - 0.5) * 2 * (a))
  //LOG(ERROR) << std::hex << RAND() << " " << RANDOM(1) << ", " << (RAND() / (double)0xffffffff) << "," << rand();

  //LOG(ERROR) << "test " << (((0xffffffff / 3.0 * 2 / (double)0xffffffff) - 0.5) * 2 * (1)) << "," << (rand() & 0xff);
  //system("pause");

  float max = -1;
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      matrix.at<float>(i, j) = RANDOM(1);
      if (matrix.at<float>(i, j) > max) {
        max = matrix.at<float>(i, j);
      }
    }
  }
  //LOG(ERROR) << "max value = " << max;
  //system("pause");
}

template<typename T>
void Sigmoid(T *data, const int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1 / (std::exp(-data[i]) + 1.0);
  }
}

void Sigmoid(cv::Mat &matrix) {
  if (matrix.type() == CV_32FC1) {
    Sigmoid<float>((float *)matrix.data, matrix.cols * matrix.rows);
  } else if (matrix.type() == CV_64FC1) {
    Sigmoid<double>((double *)matrix.data, matrix.cols * matrix.rows);
  } else {
    LOG(FATAL) << "Not implemented " << matrix.type();
  }
}

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
  NetWorks(const int inode, const int hnode, const int onode):
      _inode(inode), _hnode(hnode), _onode(onode),
      _Wih(hnode, inode, CV_32FC1),
      _Who(onode, hnode, CV_32FC1){
    Random(_Wih);
    Random(_Who);
    write_work("D:\\train\\init.yaml");
  };

  cv::Mat query(const cv::Mat &in) const {
    cv::Mat processing = _Wih * in;
    Sigmoid(processing);
    LOG(ERROR) << "who:" << _Who.size << " pro:" << processing.size;
    cv::Mat tmp = _Who * processing;
    Sigmoid(tmp);
    return tmp;
  }

  void train(const cv::Mat &in, const cv::Mat &target) {
    cv::Mat hidden_output = _Wih * in;
    Sigmoid(hidden_output);
    cv::Mat final_output = _Who * hidden_output;
    Sigmoid(final_output);
    cv::Mat error = target - final_output;
    cv::Mat hidden_error = _Who.t() * error;
    _Who += learning_rate * ((error.mul(final_output)).mul(1 - final_output)) * hidden_output.t();
    _Wih += learning_rate * ((hidden_error.mul(hidden_output)).mul(1 - hidden_output)) * in.t();
    LOG(ERROR) << "error = " << error.t();
  }

  void write_work(const std::string &path) {
    cv::FileStorage file(path, cv::FileStorage::WRITE);
    CHECK(file.isOpened()) << "path:" << path << " open fail";
    cv::write(file, "_Wih", _Wih);
    cv::write(file, "_Who", _Who);
    cv::write(file, "_inode", _inode);
    cv::write(file, "_hnode", _hnode);
    cv::write(file, "_onode", _onode);
    cv::write(file, "learning_rate", learning_rate);
    file.release();
  }

  void read_work(const std::string &path) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    CHECK(file.isOpened()) << "path:" << path << " open fail";

    cv::read(file["_Wih"], _Wih);
    cv::read(file["_Who"], _Who);
    cv::read(file["_inode"], _inode, 0);
    cv::read(file["_hnode"], _hnode, 0);
    cv::read(file["_onode"], _onode, 0);
    cv::read(file["learning_rate"], learning_rate, 0);
    file.release();
  }

};

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
    target.at<float>(i, 0) = 0.01;
  }
  target.at<float>(cur, 0) = 0.99;

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

NetWorks train() {
  const std::string root = "D:\\train";
  const std::string data = "mnist_train.csv";

  std::ifstream file(root + "/" + data);
  CHECK(file.is_open()) << root + "/" + data << " open fail";

  NetWorks work(784, 100, 10);

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
  work.write_work(root + "/weight.yaml");
  return work;
}

void query(const NetWorks &work) {
  const std::string root = "D:\\train";
  const std::string data = "mnist_test_10.csv";

  std::ifstream file(root + "/" + data);
  CHECK(file.is_open()) << root + "/" + data << " open fail";
  std::string line;
  cv::Mat input, target;
  while (!file.eof()) {
    std::getline(file, line);
    if (!line.empty()) {
      create_input(line, input, target);
      cv::Mat res = work.query(input);
      LOG(ERROR) << "res = " << res.t();
      LOG(ERROR) << "tar = " << target.t();
    }
  }
  file.close();

}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]); 


  srand(time(0));
  NetWorks work = train();
  query(work);

  

  system("pause");
  return 0;
}