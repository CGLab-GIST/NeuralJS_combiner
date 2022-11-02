//  Copyright (c) 2022 CGLab, GIST. All rights reserved.
 
//  Redistribution and use in source and binary forms, with or without modification, 
//  are permitted provided that the following conditions are met:
 
//  - Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.
//  - Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.
 
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// weight_avg.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DeallocateCombiner")  
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	return Status::OK();
  });



REGISTER_OP("Ols")
.Input("img: float")
.Input("denoised: float")
.Input("var_denoised: float")
.Input("albedo: float")
.Input("normal: float")
.Input("depth: float")
.Input("vis: float")
.Output("out: float")
.Attr("win_size: int >= 1")
.Attr("dim_feat: int >= 1")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));	
	return Status::OK();
});

//   - img : [B, Ph, Pw, C]
//   - wgt  : [B, Ph, Pw, K * K]
//   - out  : [B, Ph, Pw, C]
//   - win_size : int
REGISTER_OP("WeightAvg")  
  .Input("img: float")
  .Input("wgt: float")
  .Output("out: float")
  .Attr("win_size: int >= 1")  
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));	
    return Status::OK();
  });
  
//   - grad : [B, Ph, Pw, C]
//   - img  : [B, Ph, Pw, C]
//   - wgt  : [B, Ph, Pw, K * K]
//   - out  : [B, Ph, Pw, C]
//   - win_size : int
REGISTER_OP("WeightAvgGrad")
  .Input("grad: float")
  .Input("img: float")
  .Input("wgt: float")
  .Output("grad_wgt: float")  
  .Attr("win_size: int >= 1")  
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(2));
    return Status::OK();
  });  

REGISTER_OP("CalcShrinkage")
.Input("img: float")
.Input("denoised: float")
.Input("var: float")
.Output("out: float")
.Attr("win_size: int >= 1")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return Status::OK();
});

REGISTER_OP("CalcShrinkageGrad")
.Input("grad: float")
.Input("img: float")
.Input("denoised: float")
.Input("var: float")
.Output("grad_denoised: float")
.Output("grad_var: float")
.Attr("win_size: int >= 1")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(2));
	c->set_output(1, c->input(3));
	return Status::OK();
});

REGISTER_OP("Combiner")
.Input("img: float")
.Input("denoised: float")
.Input("shrinkage: float")
.Output("out: float")
.Attr("win_size: int >= 1")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return Status::OK();
});

REGISTER_OP("CombinerGrad")
.Input("grad: float")
.Input("img: float")
.Input("denoised: float")
.Input("shrinkage: float")
.Output("grad_denoised: float")
.Output("grad_shrinkage: float")
.Attr("win_size: int >= 1")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(2));
	c->set_output(1, c->input(3));
	return Status::OK();
});

void DeallocateCombinerFunction(const GPUDevice &_dev);

void WeightAvgFunction(const GPUDevice &_dev, const float* _img, const float* _wgt, float* _out, int nBatch, int height, int width, int winSize);
void WeightAvgGradFunction(const GPUDevice &_dev, const float* _inGrad, const float* _img, const float* _wgt, float* _out, int nBatch, int height, int width, int winSize);

void CalcShrinkageFunc(const GPUDevice &_dev, const float* _img, const float* _denoised, const float* _var, float* _out, int nBatch, int height, int width, int winSize);
void CalcShrinkageGradFunc(const GPUDevice &_dev, const float* _inGrad, const float* _img, const float* _denoised, const float* _var, float* _gradDenoised, float* _gradVar, int nBatch, int height, int width, int winSize);

void CombinerFunc(const GPUDevice &_dev, const float* _img, const float* _denoised, const float* _shrinkage, float* _out, int nBatch, int height, int width, int winSize);
void CombinerGradFunc(const GPUDevice &_dev, const float* _inGrad, const float* _img, const float* _denoised, const float* _shrinkage, float* _gradDenoised, float* _gradShrinkage, int nBatch, int height, int width, int winSize);

void OlsFunc(const GPUDevice &_dev, const float* _img, const float* _denoised, const float* _varDenoised, const float* _albedo, const float* _normal, const float* _depth, const float* _vis,			 
			 float* _out, int nBatch, int height, int width, int winSize, int dimFeat);

// Deallocate all temp. CUDA memory
class DeallocateCombinerOp : public OpKernel {
 public:
  explicit DeallocateCombinerOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {    	
    DeallocateCombinerFunction(context->eigen_device<GPUDevice>());
  }
};


//   - img : [B, Ph, Pw, C]
//   - wgt  : [B, Ph, Pw, K * K]
//   - out  : [B, Ph, Pw, C]
//   - win_size : int
class WeightAvgOp : public OpKernel {
 public:
  explicit WeightAvgOp(OpKernelConstruction* context) : OpKernel(context) {
	context->GetAttr("win_size", &winSize);
  }

  void Compute(OpKernelContext* context) override {
    
	const Tensor& img = context->input(0);
	const Tensor& wgt = context->input(1);
       
    const TensorShape& img_shape = img.shape();    
    
    TensorShape output_shape;
    output_shape.AddDim(img_shape.dim_size(0));
    output_shape.AddDim(img_shape.dim_size(1));
    output_shape.AddDim(img_shape.dim_size(2));
    output_shape.AddDim(img_shape.dim_size(3));

    Tensor* out_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
    auto out_mat = out_tensor->tensor<float, 4>();

    WeightAvgFunction(
      context->eigen_device<GPUDevice>(),
	  img.flat<float>().data(), wgt.flat<float>().data(), out_mat.data(),
      output_shape.dim_size(0),
      output_shape.dim_size(1),
      output_shape.dim_size(2),
      winSize
      );
  }
  
  private:
    int winSize;
};

//   - grad : [B, Ph, Pw, C]
//   - img  : [B, Ph, Pw, C]
//   - wgt  : [B, Ph, Pw, K * K]
//   - out  : [B, Ph, Pw, C]
//   - win_size : int
class WeightAvgGradOp : public OpKernel {
public:
  
  explicit WeightAvgGradOp(OpKernelConstruction* context) : OpKernel(context) { 
	context->GetAttr("win_size", &winSize);
  }
  
  void Compute(OpKernelContext* context) override {

    const Tensor& grad = context->input(0);
	const Tensor& img = context->input(1);
	const Tensor& wgt = context->input(2);
      
    const TensorShape& wgt_shape = wgt.shape();    

    TensorShape output_shape;
    output_shape.AddDim(wgt_shape.dim_size(0));
    output_shape.AddDim(wgt_shape.dim_size(1));
    output_shape.AddDim(wgt_shape.dim_size(2));
    output_shape.AddDim(wgt_shape.dim_size(3));

    Tensor* out_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
    auto out_mat = out_tensor->tensor<float, 4>();

    WeightAvgGradFunction(
      context->eigen_device<GPUDevice>(), grad.flat<float>().data(),
	  img.flat<float>().data(), wgt.flat<float>().data(), out_mat.data(),
      output_shape.dim_size(0),
      output_shape.dim_size(1),
      output_shape.dim_size(2),
      winSize
	);
  }
  
  private:
    int winSize;
};

class CombinerOp : public OpKernel {
public:
	explicit CombinerOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("win_size", &winSize);
	}
	void Compute(OpKernelContext* context) override {
		const Tensor& img = context->input(0);
		const Tensor& denoised = context->input(1);
		const Tensor& shrinkage = context->input(2);

		const TensorShape& img_shape = img.shape();

		TensorShape output_shape;
		output_shape.AddDim(img_shape.dim_size(0));
		output_shape.AddDim(img_shape.dim_size(1));
		output_shape.AddDim(img_shape.dim_size(2));
		output_shape.AddDim(img_shape.dim_size(3));

		Tensor* out_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
		auto out_mat = out_tensor->tensor<float, 4>();

		CombinerFunc(
			context->eigen_device<GPUDevice>(),
			img.flat<float>().data(), denoised.flat<float>().data(),
			shrinkage.flat<float>().data(), out_mat.data(),
			output_shape.dim_size(0),
			output_shape.dim_size(1),
			output_shape.dim_size(2),
			winSize
		);
	}

private:
	int winSize;
};


class CombinerGradOp : public OpKernel {
public:
	explicit CombinerGradOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("win_size", &winSize);
	}
	void Compute(OpKernelContext* context) override {
		const Tensor& grad = context->input(0);
		const Tensor& img = context->input(1);
		const Tensor& denoised = context->input(2);
		const Tensor& shrinkage = context->input(3);

		const TensorShape& denoised_shape = denoised.shape();
		const TensorShape& shrinkage_shape = shrinkage.shape();		

		TensorShape outShapeGradDenoised, outShapeGradShrinkage;
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(0));
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(1));
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(2));
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(3));
		outShapeGradShrinkage.AddDim(shrinkage_shape.dim_size(0));
		outShapeGradShrinkage.AddDim(shrinkage_shape.dim_size(1));
		outShapeGradShrinkage.AddDim(shrinkage_shape.dim_size(2));
		outShapeGradShrinkage.AddDim(shrinkage_shape.dim_size(3));

		Tensor* outTensorGradDenoised = NULL;
		Tensor* outTensorGradShrinkage = NULL;		

		OP_REQUIRES_OK(context, context->allocate_output(0, outShapeGradDenoised, &outTensorGradDenoised));
		OP_REQUIRES_OK(context, context->allocate_output(1, outShapeGradShrinkage, &outTensorGradShrinkage));
		
		auto out_grad_denoised_mat = outTensorGradDenoised->tensor<float, 4>();
		auto out_grad_shrinkage_mat = outTensorGradShrinkage->tensor<float, 4>();

		CombinerGradFunc(
			context->eigen_device<GPUDevice>(), grad.flat<float>().data(),
			img.flat<float>().data(), denoised.flat<float>().data(), shrinkage.flat<float>().data(), 
			out_grad_denoised_mat.data(), out_grad_shrinkage_mat.data(),
			denoised_shape.dim_size(0),
			denoised_shape.dim_size(1),
			denoised_shape.dim_size(2),
			winSize
		);
	}

private:
	int winSize;
};

class CalcShrinkageOp : public OpKernel {
public:
	explicit CalcShrinkageOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("win_size", &winSize);
	}
	void Compute(OpKernelContext* context) override {
		const Tensor& img = context->input(0);
		const Tensor& denoised = context->input(1);
		const Tensor& var = context->input(2);

		const TensorShape& img_shape = img.shape();

		TensorShape output_shape;
		output_shape.AddDim(img_shape.dim_size(0));
		output_shape.AddDim(img_shape.dim_size(1));
		output_shape.AddDim(img_shape.dim_size(2));
		output_shape.AddDim(img_shape.dim_size(3));

		Tensor* out_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
		auto out_mat = out_tensor->tensor<float, 4>();

		CalcShrinkageFunc(
			context->eigen_device<GPUDevice>(),
			img.flat<float>().data(), denoised.flat<float>().data(),
			var.flat<float>().data(), out_mat.data(),
			output_shape.dim_size(0),
			output_shape.dim_size(1),
			output_shape.dim_size(2),
			winSize
		);
	}

private:
	int winSize;
};


class CalcShrinkageGradOp : public OpKernel {
public:
	explicit CalcShrinkageGradOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("win_size", &winSize);
	}
	void Compute(OpKernelContext* context) override {
		const Tensor& grad = context->input(0);
		const Tensor& img = context->input(1);
		const Tensor& denoised = context->input(2);
		const Tensor& var = context->input(3);

		const TensorShape& denoised_shape = denoised.shape();
		const TensorShape& var_shape = var.shape();
		
		TensorShape outShapeGradDenoised, outShapeGradVar;
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(0));
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(1));
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(2));
		outShapeGradDenoised.AddDim(denoised_shape.dim_size(3));
		outShapeGradVar.AddDim(var_shape.dim_size(0));
		outShapeGradVar.AddDim(var_shape.dim_size(1));
		outShapeGradVar.AddDim(var_shape.dim_size(2));
		outShapeGradVar.AddDim(var_shape.dim_size(3));
		
		Tensor* outTensorGradDenoised = NULL;
		Tensor* outTensorGradVar = NULL;
		
		OP_REQUIRES_OK(context, context->allocate_output(0, outShapeGradDenoised, &outTensorGradDenoised));
		OP_REQUIRES_OK(context, context->allocate_output(1, outShapeGradVar, &outTensorGradVar));

		auto out_grad_denoised_mat = outTensorGradDenoised->tensor<float, 4>();
		auto out_grad_var_mat = outTensorGradVar->tensor<float, 4>();


		CalcShrinkageGradFunc(
			context->eigen_device<GPUDevice>(), grad.flat<float>().data(),
			img.flat<float>().data(), denoised.flat<float>().data(), var.flat<float>().data(), out_grad_denoised_mat.data(), out_grad_var_mat.data(),
			denoised_shape.dim_size(0),
			denoised_shape.dim_size(1),
			denoised_shape.dim_size(2),
			winSize
		);
	}

private:
	int winSize;
};



class OlsOp : public OpKernel {
public:
	explicit OlsOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("win_size", &winSize);
		context->GetAttr("dim_feat", &dimFeat);
	}
	void Compute(OpKernelContext* context) override {
		const Tensor& img = context->input(0);
		const Tensor& denoised = context->input(1);
		const Tensor& var_denoised = context->input(2);
		const Tensor& albedo = context->input(3);
		const Tensor& normal = context->input(4);
		const Tensor& depth = context->input(5);
		const Tensor& vis = context->input(6);

		const TensorShape& img_shape = img.shape();

		TensorShape output_shape;
		output_shape.AddDim(img_shape.dim_size(0));
		output_shape.AddDim(img_shape.dim_size(1));
		output_shape.AddDim(img_shape.dim_size(2));
		output_shape.AddDim(img_shape.dim_size(3));

		Tensor* out_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
		auto out_mat = out_tensor->tensor<float, 4>();

		OlsFunc(
			context->eigen_device<GPUDevice>(), 
			img.flat<float>().data(), denoised.flat<float>().data(), var_denoised.flat<float>().data(), 
			albedo.flat<float>().data(), normal.flat<float>().data(), depth.flat<float>().data(), vis.flat<float>().data(),
			out_mat.data(),
			output_shape.dim_size(0),
			output_shape.dim_size(1),
			output_shape.dim_size(2),
			winSize, dimFeat
		);
	}
private:
	int winSize, dimFeat;
};

// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("WeightAvg").Device(DEVICE_GPU), WeightAvgOp);
REGISTER_KERNEL_BUILDER(Name("WeightAvgGrad").Device(DEVICE_GPU), WeightAvgGradOp);
REGISTER_KERNEL_BUILDER(Name("CalcShrinkage").Device(DEVICE_GPU), CalcShrinkageOp);
REGISTER_KERNEL_BUILDER(Name("CalcShrinkageGrad").Device(DEVICE_GPU), CalcShrinkageGradOp);
REGISTER_KERNEL_BUILDER(Name("Combiner").Device(DEVICE_GPU), CombinerOp);
REGISTER_KERNEL_BUILDER(Name("CombinerGrad").Device(DEVICE_GPU), CombinerGradOp);
REGISTER_KERNEL_BUILDER(Name("Ols").Device(DEVICE_GPU), OlsOp);
REGISTER_KERNEL_BUILDER(Name("DeallocateCombiner").Device(DEVICE_GPU), DeallocateCombinerOp);
