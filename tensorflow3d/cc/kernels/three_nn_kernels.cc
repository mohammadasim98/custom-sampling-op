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

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
     for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
	    float x1=xyz1[j*3+0];
	    float y1=xyz1[j*3+1];
	    float z1=xyz1[j*3+2];
            double best1=1e40; double best2=1e40; double best3=1e40;
            int besti1=0; int besti2=0; int besti3=0;
            for (int k=0;k<m;++k) {
                float x2=xyz2[k*3+0];
	        float y2=xyz2[k*3+1];
	        float z2=xyz2[k*3+2];
		//float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
		double d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                if (d<best1) {
                    best3=best2;
                    besti3=besti2;
                    best2=best1;
                    besti2=besti1;
                    best1=d;
                    besti1=k;
                } else if (d<best2) {
                    best3=best2;
                    besti3=besti2;
                    best2=d;
                    besti2=k;
                } else if (d<best3) {
                    best3=d;
                    besti3=k;
                }
            } 
            dist[j*3]=best1;
            idx[j*3]=besti1;
            dist[j*3+1]=best2;
            idx[j*3+1]=besti2;
            dist[j*3+2]=best3;
            idx[j*3+2]=besti3;
        } 
        xyz1+=n*3;
        xyz2+=m*3;
        dist+=n*3;
        idx+=n*3;
    }
}
class ThreeNNOp : public OpKernel {
    public:
        explicit ThreeNNOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeNN expects (b,n,3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeNN expects (b,m,3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *dist_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,3}, &dist_tensor));
            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,3}, &idx_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto dist_flat = dist_tensor->flat<float>();
            float *dist = &(dist_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            threenn_cpu(b,n,m,xyz1,xyz2,dist,idx);
        }
};
REGISTER_KERNEL_BUILDER(Name("ThreeNN").Device(DEVICE_CPU), ThreeNNOp);