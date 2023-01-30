/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "group_point.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void GroupPointKernel(int b, int n, int c, int m, int nsample, const T *points, const int *idx, T *out){
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// output: grad_points (b,n,c)
template <typename T>
__global__ void GroupPointGradKernel(int b, int n, int c, int m, int nsample, const T *grad_out, const int *idx, T *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct GroupPointFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int b, int n, int c, int m, int nsample, const T *points, const int *idx, T *out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int thread_per_block = 256;
    GroupPointGradKernel<T><<<b, thread_per_block, 0, d.stream()>>>(b,n,c,m,nsample,points,idx,out);
  }
};

template <typename T>
struct GroupPointGradFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int b, int n, int c, int m, int nsample, const T *grad_out, const int *idx, T *grad_points) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int thread_per_block = 256;
    GroupPointKernel<T><<<b, thread_per_block, 0, d.stream()>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
  }
};
// Explicitly instantiate functors for the types of OpKernels registered.
template struct GroupPointFunctor<GPUDevice, float>;
template struct GroupPointFunctor<GPUDevice, int32>;

template struct GroupPointGradFunctor<GPUDevice, float>;
template struct GroupPointGradFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
