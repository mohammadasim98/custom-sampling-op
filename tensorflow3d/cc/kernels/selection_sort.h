// kernel_example.h
#ifndef KERNEL_SELECTION_SORT_H_
#define KERNEL_SELECTION_SORT_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct SelectionSortFunctor {
  void operator()(const Device& d, int b, int n, int m, int k, const T *dist, int *outi, T *out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
