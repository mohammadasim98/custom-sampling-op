// kernel_example.h
#ifndef KERNEL_BALL_QUERY_H_
#define KERNEL_BALL_QUERY_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct BallQueryFunctor {
  void operator()(const Device& d, int b, int n, int m, float radius, int nsample, const T *xyz1, const T *xyz2, int *idx, int *pts_cnt);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
