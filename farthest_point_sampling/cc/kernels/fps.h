// kernel_example.h
#ifndef KERNEL_FPS_H_
#define KERNEL_FPS_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct FarthestPointSampleFunctor {
  void operator()(const Device& d, int b, int n, int m, const T * inp, T * temp, int * out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
