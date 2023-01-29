#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("FarthestPointSample")
  .Attr("npoint: int")
  .Attr("T: float")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });