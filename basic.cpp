#include "basic.h"

#include <glog/logging.h>
#include <cmath>
#include <fstream>
#include <sstream>
namespace AiLearning {
NetWorks::NetWorks(const int inode, const int hnode, const int onode) :
    _inode(inode), _hnode(hnode), _onode(onode),
    _Wih(hnode, inode, CV_32FC1),
    _Who(onode, hnode, CV_32FC1) {
  Random(_Wih);
  Random(_Who);
  write_work("/home/moriarty/Datasets/init.yaml");
}


cv::Mat NetWorks::query(const cv::Mat &in) const {
  cv::Mat processing = _Wih * in;
  Sigmoid(processing);
  LOG(INFO) << "who:" << _Who.size() << " pro:" << processing.size();
  cv::Mat tmp = _Who * processing;
  Sigmoid(tmp);
  return tmp;
}

void NetWorks::train(const cv::Mat &in, const cv::Mat &target) {
  cv::Mat hidden_output = _Wih * in;
  Sigmoid(hidden_output);
  cv::Mat final_output = _Who * hidden_output;
  Sigmoid(final_output);
  cv::Mat error = target - final_output;
  cv::Mat hidden_error = _Who.t() * error;
  _Who += learning_rate * ((error.mul(final_output)).mul(1 - final_output)) * hidden_output.t();
  _Wih += learning_rate * ((hidden_error.mul(hidden_output)).mul(1 - hidden_output)) * in.t();
  LOG(INFO) << "error = " << error.t();
}

void NetWorks::write_work(const std::string &path) {
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

void NetWorks::read_work(const std::string &path) {
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

} //namespace AiLearning
