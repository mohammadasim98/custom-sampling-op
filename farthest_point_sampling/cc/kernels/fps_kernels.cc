/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "fps.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// // CPU specialization of actual computation.
// template <typename T>
// struct TimeTwoFunctor<CPUDevice, T> {
//   void operator()(const CPUDevice& d, int b, int n, int m, const float * inp, float * temp, int * out) {
//     for (int i = 0; i < size; ++i) {
//       out[i] = 2 * in[i];
//     }
//   }
// };

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context) : OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;
      

      // Grab the input tensor
      const Tensor& inp_tensor=context->input(0);
      int b = inp_tensor.shape().dim_size(0);
      auto inp_flat = inp_tensor.flat<float>();
      const float * inp = &(inp_flat(0));

      // Create an output tensor
      Tensor * out_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, m}, &out_tensor));
      auto out_flat = out_tensor->flat<int>();
      int * out = &(out_flat(0));

      int n = inp_tensor.shape().dim_size(1);

      // Create a temporary tensor
      Tensor temp_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{32, n}, &temp_tensor));
      auto temp_flat = temp_tensor.flat<float>();
      float * temp = &(temp_flat(0));

      // Compute
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      FarthestPointSampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};

// // Register the CPU kernels.
// #define REGISTER_CPU(T)                                          \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("TimeTwo").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
//       TimeTwoOp<CPUDevice, T>);
// REGISTER_CPU(float);
// REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct FarthestPointSampleFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FarthestPointSample").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FarthestPointSampleGpuOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
