//
// Created by moriarty on 6/2/21.
//
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <opencv2/cudaarithm.hpp>
#include <glog/logging.h>

namespace AiLearning {
namespace MatrixUtils {
template<typename T>
struct step_functor : public thrust::unary_function<int, int> {
  int columns;
  int step;
  int channels;

  __host__ __device__ step_functor(int columns_, int step_, int channels_ = 1)
    : columns(columns_), step(step_), channels(channels_) {};

  __host__ step_functor(cv::cuda::GpuMat &mat) {
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    columns = mat.cols;
    step = mat.step / sizeof(T);
    channels = mat.channels();
  }

  __host__ __device__
  int operator()(int x) const {
    int row = x / columns;
    int idx = (row * step) + (x % columns) * channels;
    return idx;
  }
};


/*
    @Brief GpuMatBeginItr returns a thrust compatible iterator to the beginning of a GPU mat's memory.
    @Param mat is the input matrix
    @Param channel is the channel of the matrix that the iterator is accessing.  If set to -1, the iterator will access every element in sequential order
*/
template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatBeginItr(cv::cuda::GpuMat mat, int channel = 0) {
  if (channel == -1) {
    mat = mat.reshape(1);
    channel = 0;
  }
  CHECK(mat.depth() == cv::DataType<T>::depth) << mat.depth() << "," << cv::DataType<T>::depth;
  CV_Assert(channel < mat.channels());
  return thrust::make_permutation_iterator(
    thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
    thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                    step_functor<T>(mat.cols,
                                                    mat.step / sizeof(T),
                                                    mat.channels())));
}

/*
@Brief GpuMatEndItr returns a thrust compatible iterator to the end of a GPU mat's memory.
@Param mat is the input matrix
@Param channel is the channel of the matrix that the iterator is accessing.  If set to -1, the iterator will access every element in sequential order
*/
template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatEndItr(cv::cuda::GpuMat mat, int channel = 0) {
  if (channel == -1) {
    mat = mat.reshape(1);
    channel = 0;
  }
  CV_Assert(mat.depth() == cv::DataType<T>::depth);
  CV_Assert(channel < mat.channels());
  return thrust::make_permutation_iterator(
    thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(mat.rows * mat.cols),
      step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}

template<typename _T>
struct sigmoid_func : public thrust::unary_function<void, _T &> {

  __host__ __device__
  void operator()(_T &p) const {
    if (p > 0) {
      p = 1 / (std::exp(-p) + 1.0);
    } else {
      _T exp = std::exp(p);
      p = exp / (1 + exp);
    }
  }
};

void Sigmoid(cv::cuda::GpuMat &matrix) {
  if (matrix.type() == CV_64FC1) {
    sigmoid_func<double> func;
    auto valueBegin = GpuMatBeginItr<double>(matrix, 0);
    auto valueEnd = GpuMatEndItr<double>(matrix, 0);
    thrust::for_each(valueBegin, valueEnd, func);
  } else if (matrix.type() == CV_32FC1) {
    sigmoid_func<float> func;
    auto valueBegin = GpuMatBeginItr<float>(matrix, 0);
    auto valueEnd = GpuMatEndItr<float>(matrix, 0);
    thrust::for_each(valueBegin, valueEnd, func);
  } else {
    LOG(FATAL) << "invalid input type:" << matrix.type()
               << ", must be CV_64FC1 or CV_32FC1";
  }


  return;
}

}//namespace MatrixUtils
}//namespace AiLearning