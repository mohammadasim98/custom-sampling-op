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

#include "ball_query.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void BallQueryKernel(int b, int n, int m, float radius, int nsample, const T *xyz1, const T *xyz2, int *idx, int *pts_cnt){
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
    	    float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct BallQueryFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int b, int n, int m, float radius, int nsample, const T *xyz1, const T *xyz2, int *idx, int *pts_cnt) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int thread_per_block = 256;
    BallQueryKernel<T><<<b, thread_per_block, 0, d.stream()>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct BallQueryFunctor<GPUDevice, float>;
template struct BallQueryFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
