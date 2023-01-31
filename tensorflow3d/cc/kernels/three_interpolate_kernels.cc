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

// input: points (b,m,c), idx (b,n,3), weight (b,n,3)
// output: out (b,n,c)
void threeinterpolate_cpu(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
     float w1,w2,w3;
     int i1,i2,i3;
     for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
            w1=weight[j*3];
            w2=weight[j*3+1];
            w3=weight[j*3+2]; 
            i1=idx[j*3];
            i2=idx[j*3+1];
            i3=idx[j*3+2];
            for (int l=0;l<c;++l) {
                out[j*c+l] = points[i1*c+l]*w1 + points[i2*c+l]*w2 + points[i3*c+l]*w3;
            }
        } 
        points+=m*c;
        idx+=n*3;
        weight+=n*3;
        out+=n*c;
    }
}

// input: grad_out (b,n,c), idx (b,n,3), weight (b,n,3)
// output: grad_points (b,m,c)
void threeinterpolate_grad_cpu(int b, int n, int c, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points) {
     float w1,w2,w3;
     int i1,i2,i3;
     for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
            w1=weight[j*3];
            w2=weight[j*3+1];
            w3=weight[j*3+2]; 
            i1=idx[j*3];
            i2=idx[j*3+1];
            i3=idx[j*3+2];
            for (int l=0;l<c;++l) {
                grad_points[i1*c+l] += grad_out[j*c+l]*w1;
                grad_points[i2*c+l] += grad_out[j*c+l]*w2;
                grad_points[i3*c+l] += grad_out[j*c+l]*w3;
            }
        } 
        grad_out+=n*c;
        idx+=n*3;
        weight+=n*3;
        grad_points+=m*c;
    }
}

class ThreeInterpolateOp: public OpKernel{
    public:
        explicit ThreeInterpolateOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("ThreeInterpolate expects (b,m,c) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int m = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b && idx_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeInterpolate expects (b,n,3) idx shape"));
            int n = idx_tensor.shape().dim_size(1);
            const Tensor& weight_tensor=context->input(2);
            OP_REQUIRES(context,weight_tensor.dims()==3 && weight_tensor.shape().dim_size(0)==b && weight_tensor.shape().dim_size(1)==n && weight_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeInterpolate expects (b,n,3) weight shape"));

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto weight_flat = weight_tensor.flat<float>();
            const float *weight = &(weight_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            threeinterpolate_cpu(b,m,c,n,points,idx,weight,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("ThreeInterpolate").Device(DEVICE_CPU),ThreeInterpolateOp);


class ThreeInterpolateGradOp: public OpKernel{
    public:
        explicit ThreeInterpolateGradOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("ThreeInterpolateGrad expects (b,m,c) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int m = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("ThreeInterpolateGrad expects (b,n,3) idx shape"));
            int n = idx_tensor.shape().dim_size(1);
            const Tensor& weight_tensor=context->input(2);
            OP_REQUIRES(context,weight_tensor.dims()==3 && weight_tensor.shape().dim_size(0)==b && weight_tensor.shape().dim_size(1)==n && weight_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeInterpolateGrad expects (b,n,3) weight shape"));

            const Tensor& grad_out_tensor=context->input(3);
            OP_REQUIRES(context,grad_out_tensor.dims()==3 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==n && grad_out_tensor.shape().dim_size(2)==c, errors::InvalidArgument("ThreeInterpolateGrad expects (b,n,c) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,c}, &grad_points_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto weight_flat = weight_tensor.flat<float>();
            const float *weight = &(weight_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            memset(grad_points, 0, sizeof(float)*b*m*c);
            threeinterpolate_grad_cpu(b,n,c,m,grad_out,idx,weight,grad_points);
        }
};
REGISTER_KERNEL_BUILDER(Name("ThreeInterpolateGrad").Device(DEVICE_CPU),ThreeInterpolateGradOp);
