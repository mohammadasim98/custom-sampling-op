// kernel_example.h
#ifndef KERNEL_GATHER_H_
#define KERNEL_GATHER_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct GatherPointFunctor {
  void operator()(const Device& d, int b, int n, int m, const T * inp, const int * idx, T * out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
