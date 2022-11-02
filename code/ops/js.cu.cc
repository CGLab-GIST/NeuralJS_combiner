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

//#define GOOGLE_CUDA

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "cuda/include/cuda_runtime.h"
#include "cuda/include/vector_types.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

inline int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

#define JS_EP 1e-5f
#define OLS_WGT_EP 0.01f
#define OLS_EP 0.1f
#define OLS_FEAT_DIM 11 // denoised(3), albedo (3), normal (3), depth (1), vis (1)

float4* g_accOut = NULL;
int g_lenAccOut = 0;

__forceinline__ __host__ __device__ float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__forceinline__ __host__ __device__ void operator-=(float4 &a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__forceinline__ __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__forceinline__ __host__ __device__ float4 fmaxf(float4 a, float4 b) {
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

__forceinline__ __device__ float square(const float& a) {
	return (a * a);
}

__forceinline__ __device__ float logNorm2(const float4& a) {
	return __logf(1.f + a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __device__ float norm2(const float& a) {
	return (a * a);
}

__forceinline__ __device__ float Dot(const float4& a, const float4& b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__forceinline__ __device__ float avg(const float4& a) {
	return (a.x + a.y + a.z) / 3.f;
}

__forceinline__ __device__ int getImgIdx(int ix, int iy, int width, int height) {
	int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
	int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
	return y * width + x;
}

__forceinline__ __device__ float4 logTrans(const float4& a) {
	float4 logVal;
	logVal.x = __logf(1.f + a.x);
	logVal.y = __logf(1.f + a.y);
	logVal.z = __logf(1.f + a.z);
	logVal.w = 0.f;
	return logVal;
}

// Input - A (only upper triangle is set!)
__device__ void cholesky(float *A, int P, float *diagL) {
	for (int i = 0; i < P; ++i) {
		for (int j = i; j < P; ++j) {
			float sum = A[i * P + j];
			for (int k = i - 1; k >= 0; --k)
				sum -= A[i * P + k] * A[j * P + k];
			if (i == j) {
				if (sum <= 0.f)
					printf("ERR in cholesky");
				diagL[i] = sqrtf(sum);
			}
			else
				A[j * P + i] = sum / diagL[i];
		}
	}
}

__global__ void WeightAvgKernel(const float* _img, const float* _wgt, float* _out, int iBatch, int height, int width, int winSize) {	   
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;	  
	const int nPix = width * height;       
	const int halfWinSize = winSize / 2; 
	const int winSizeSqr = winSize * winSize;

	float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);	
	int n = 0;
	float sumW = 0.f;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {		
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			
			const float4& iCol = make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float& weight = _wgt[(iBatch * nPix + cIdx) * winSizeSqr + n];
			++n;
			
			accCol += weight * iCol;
			sumW += weight;			
		}
	}		
	float invSumW = 1.f / fmaxf(sumW, JS_EP);
	float4 outCol = invSumW * accCol;
	
	_out[(iBatch * nPix + cIdx) * 3 + 0] = outCol.x;
	_out[(iBatch * nPix + cIdx) * 3 + 1] = outCol.y;
	_out[(iBatch * nPix + cIdx) * 3 + 2] = outCol.z;
}

__global__ void WeightAvgGradKernel(const float* _img, const float* _inGrad, const float* _wgt, float* _out, int iBatch, int height, int width, int winSize) {												
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	int cIdx = cy * width + cx;		
	const int nPix = width * height;       
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;	
		
	float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);		
	int n = 0;	
	float sumW = 0.f;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {		
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			
			const float4& iCol = make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float& weight = _wgt[(iBatch * nPix + cIdx) * winSizeSqr + n];
			++n;
			
			accCol += weight * iCol;
			sumW += weight;			
		}
	}		
	float invSumW = 1.f / fmaxf(sumW, JS_EP);
	float4 outCol = invSumW * accCol;
	
	const float4& inGradCol = make_float4(_inGrad[(iBatch * nPix + cIdx) * 3 + 0], _inGrad[(iBatch * nPix + cIdx) * 3 + 1], _inGrad[(iBatch * nPix + cIdx) * 3 + 2], 0.f);
	n = 0;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {		
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;			
			
			const float4& iCol =  make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f);									
			float4 grad = invSumW * (iCol - outCol);
			_out[(iBatch * nPix + cIdx) * winSizeSqr + n] = Dot(grad, inGradCol);
			++n;
		}
	}
}

__global__ void CombinerKernel(const float* _img, const float* _denoised, const float* _shrinkage, float* _out, int iBatch, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;

	float4 avgRho = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float4& iRho = make_float4(_shrinkage[(iBatch * nPix + idx) * 3 + 0], _shrinkage[(iBatch * nPix + idx) * 3 + 1], _shrinkage[(iBatch * nPix + idx) * 3 + 2], 0.f);			
			avgRho += iRho;
		}
	}
	avgRho = avgRho / (float)winSizeSqr;

	const float4& cImg = make_float4(_img[(iBatch * nPix + cIdx) * 3 + 0], _img[(iBatch * nPix + cIdx) * 3 + 1], _img[(iBatch * nPix + cIdx) * 3 + 2], 0.f);
	const float4& cDenoised = make_float4(_denoised[(iBatch * nPix + cIdx) * 3 + 0], _denoised[(iBatch * nPix + cIdx) * 3 + 1], _denoised[(iBatch * nPix + cIdx) * 3 + 2], 0.f);

	float4 outCol;
	outCol.x = cDenoised.x + (cImg.x - cDenoised.x) * avgRho.x;
	outCol.y = cDenoised.y + (cImg.y - cDenoised.y) * avgRho.y;
	outCol.z = cDenoised.z + (cImg.z - cDenoised.z) * avgRho.z;

	_out[(iBatch * nPix + cIdx) * 3 + 0] = outCol.x;
	_out[(iBatch * nPix + cIdx) * 3 + 1] = outCol.y;
	_out[(iBatch * nPix + cIdx) * 3 + 2] = outCol.z;
}

__global__ void OlsFinalizeKernel(const float* _denoised, float4* _accOut, float* _outCol, int iBatch, int height, int width) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int nPix = width * height;
	const int cIdx = cy * width + cx;

	float4 accCol = _accOut[iBatch * nPix + cIdx];

	if (accCol.w < 1e-5f) {
		_outCol[(iBatch * nPix + cIdx) * 3 + 0] = _denoised[(iBatch * nPix + cIdx) * 3 + 0];
		_outCol[(iBatch * nPix + cIdx) * 3 + 1] = _denoised[(iBatch * nPix + cIdx) * 3 + 1];
		_outCol[(iBatch * nPix + cIdx) * 3 + 2] = _denoised[(iBatch * nPix + cIdx) * 3 + 2];
	}
	else {
		float invW = 1.f / accCol.w;
		_outCol[(iBatch * nPix + cIdx) * 3 + 0] = invW * accCol.x;
		_outCol[(iBatch * nPix + cIdx) * 3 + 1] = invW * accCol.y;
		_outCol[(iBatch * nPix + cIdx) * 3 + 2] = invW * accCol.z;
	}
}

__global__ void OlsKernel(const float* _img, const float* _denoised, const float* _varDenoised, const float* _albedo, const float* _normal, const float* _depth, const float* _vis,						  
						  float4* _accOut, int iBatch, int height, int width, int winSize, int dimFeat) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;

	if (OLS_FEAT_DIM != dimFeat)
		printf("ERR: Check DIM in OlsKernelOp\n");
	
	const int P = OLS_FEAT_DIM + 1;
	float delta[P];
	float A[P * P] = { 0.f, };
	float4 XtB[P];

	for (int i = 0; i < P; ++i)
		XtB[i] = make_float4(0.f, 0.f, 0.f, 0.f);

	const float4& cDenoised = (make_float4(_denoised[(iBatch * nPix + cIdx) * 3 + 0], _denoised[(iBatch * nPix + cIdx) * 3 + 1], _denoised[(iBatch * nPix + cIdx) * 3 + 2], 0.f));
	const float4& cAlbedo = make_float4(_albedo[(iBatch * nPix + cIdx) * 3 + 0], _albedo[(iBatch * nPix + cIdx) * 3 + 1], _albedo[(iBatch * nPix + cIdx) * 3 + 2], 0.f);
	const float4& cNormal = make_float4(_normal[(iBatch * nPix + cIdx) * 3 + 0], _normal[(iBatch * nPix + cIdx) * 3 + 1], _normal[(iBatch * nPix + cIdx) * 3 + 2], 0.f);
	const float   cDepth = _depth[iBatch * nPix + cIdx];
	const float   cShadow = _vis[iBatch * nPix + cIdx];
	const float   cVarDenoised = _varDenoised[iBatch * nPix + cIdx];

	// feature standadization
	float factorDenoised, factorAlbedo, factorNormal, factorDepth, factorShadow; 
	factorDenoised = factorAlbedo = factorNormal = factorDepth = factorShadow = 0.f;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float4& iDenoised = (make_float4(_denoised[(iBatch * nPix + idx) * 3 + 0], _denoised[(iBatch * nPix + idx) * 3 + 1], _denoised[(iBatch * nPix + idx) * 3 + 2], 0.f));
			const float4& iAlbedo = make_float4(_albedo[(iBatch * nPix + idx) * 3 + 0], _albedo[(iBatch * nPix + idx) * 3 + 1], _albedo[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[(iBatch * nPix + idx) * 3 + 0], _normal[(iBatch * nPix + idx) * 3 + 1], _normal[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float   iDepth = _depth[iBatch * nPix + idx];
			const float   iShadow = _vis[iBatch * nPix + idx];
			factorDenoised = fmaxf(factorDenoised, norm2(iDenoised - cDenoised));
			factorAlbedo = fmaxf(factorAlbedo, norm2(iAlbedo - cAlbedo));
			factorNormal = fmaxf(factorNormal, norm2(iNormal - cNormal));
			factorDepth = fmaxf(factorDepth, fabs(iDepth - cDepth));
			factorShadow = fmaxf(factorShadow, fabs(iShadow - cShadow));
		}
	}
	factorDenoised = 1.f / (sqrtf(factorDenoised) + OLS_EP);
	factorAlbedo = 1.f / (sqrtf(factorAlbedo) + OLS_EP);
	factorNormal = 1.f / (sqrtf(factorNormal) + OLS_EP);
	factorDepth = 1.f / (factorDepth + OLS_EP);
	factorShadow = 1.f / (factorShadow + OLS_EP);
		
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;

			const float4& iImg = (make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f));
			const float4& iDenoised = (make_float4(_denoised[(iBatch * nPix + idx) * 3 + 0], _denoised[(iBatch * nPix + idx) * 3 + 1], _denoised[(iBatch * nPix + idx) * 3 + 2], 0.f));
			const float4& iAlbedo = make_float4(_albedo[(iBatch * nPix + idx) * 3 + 0], _albedo[(iBatch * nPix + idx) * 3 + 1], _albedo[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[(iBatch * nPix + idx) * 3 + 0], _normal[(iBatch * nPix + idx) * 3 + 1], _normal[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float   iDepth = _depth[iBatch * nPix + idx];
			const float   iShadow = _vis[iBatch * nPix + idx];
			//const float   iVarDenoised = _varDenoised[iBatch * nPix + idx];

			delta[0] = 1.f;
			delta[1] = (iDenoised.x - cDenoised.x) * factorDenoised;
			delta[2] = (iDenoised.y - cDenoised.y) * factorDenoised;
			delta[3] = (iDenoised.z - cDenoised.z) * factorDenoised;
			delta[4] = (iAlbedo.x - cAlbedo.x) * factorAlbedo;
			delta[5] = (iAlbedo.y - cAlbedo.y) * factorAlbedo;
			delta[6] = (iAlbedo.z - cAlbedo.z) * factorAlbedo;
			delta[7] = (iNormal.x - cNormal.x) * factorNormal;
			delta[8] = (iNormal.y - cNormal.y) * factorNormal;
			delta[9] = (iNormal.z - cNormal.z) * factorNormal;
			delta[10] = (iDepth - cDepth) * factorDepth;
			delta[11] = (iShadow - cShadow) * factorShadow;

			//float weight = __expf(-norm2(iDenoised - cDenoised) / (cVarDenoised + iVarDenoised + OLS_WGT_EP));
			float weight = __expf(-norm2(iDenoised - cDenoised) / (2.f * cVarDenoised + OLS_WGT_EP));

			// only upper triangle of A (XtX)
			for (int row = 0; row < P; ++row) {
				for (int col = row; col < P; ++col) {
					A[row * P + col] += weight * delta[row] * delta[col];
				}
			}
			// XtB
			for (int i = 0; i < P; ++i)
				XtB[i] += weight * delta[i] * iImg;

		}
	}

	for (int row = 0; row < P; ++row)		
		A[row * P + row] += OLS_EP;

	float diagL[P];
	cholesky(A, P, diagL);

	float4 beta[P];
	// L y = XtB
	for (int i = 0; i < P; ++i) {
		float4 sum = XtB[i];
		for (int k = i - 1; k >= 0; --k)
			sum = sum - A[i * P + k] * beta[k];
		beta[i] = sum / diagL[i];
	}
	// L^t \beta = y
	for (int i = P; i >= 0; --i) {
		float4 sum = beta[i];
		for (int k = i + 1; k < P; ++k)
			sum = sum - A[k * P + i] * beta[k];
		beta[i] = sum / diagL[i];
	}
		
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;			
		
			const float4& iDenoised = (make_float4(_denoised[(iBatch * nPix + idx) * 3 + 0], _denoised[(iBatch * nPix + idx) * 3 + 1], _denoised[(iBatch * nPix + idx) * 3 + 2], 0.f));
			const float4& iAlbedo = make_float4(_albedo[(iBatch * nPix + idx) * 3 + 0], _albedo[(iBatch * nPix + idx) * 3 + 1], _albedo[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[(iBatch * nPix + idx) * 3 + 0], _normal[(iBatch * nPix + idx) * 3 + 1], _normal[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float   iDepth = _depth[iBatch * nPix + idx];
			const float   iShadow = _vis[iBatch * nPix + idx];
			//const float   iVarDenoised = _varDenoised[iBatch * nPix + idx];

			delta[0] = 1.f;
			delta[1] = (iDenoised.x - cDenoised.x) * factorDenoised;
			delta[2] = (iDenoised.y - cDenoised.y) * factorDenoised;
			delta[3] = (iDenoised.z - cDenoised.z) * factorDenoised;
			delta[4] = (iAlbedo.x - cAlbedo.x) * factorAlbedo;
			delta[5] = (iAlbedo.y - cAlbedo.y) * factorAlbedo;
			delta[6] = (iAlbedo.z - cAlbedo.z) * factorAlbedo;
			delta[7] = (iNormal.x - cNormal.x) * factorNormal;
			delta[8] = (iNormal.y - cNormal.y) * factorNormal;
			delta[9] = (iNormal.z - cNormal.z) * factorNormal;
			delta[10] = (iDepth - cDepth) * factorDepth;
			delta[11] = (iShadow - cShadow) * factorShadow;

			float4 olsOut = make_float4(0.f, 0.f, 0.f, 0.f);

			// Memory layout: P for red, P for green, P for blue
			for (int i = 0; i < P; ++i) {
				olsOut.x += delta[i] * beta[i].x;
				olsOut.y += delta[i] * beta[i].y;
				olsOut.z += delta[i] * beta[i].z;
			}
			float weight = __expf(-norm2(iDenoised - cDenoised) / (2.f * cVarDenoised + OLS_WGT_EP));
			//float weight = __expf(-norm2(iDenoised - cDenoised) / ((cVarDenoised + iVarDenoised) + OLS_WGT_EP));			

			atomicAdd(&_accOut[iBatch * nPix + idx].x, weight * olsOut.x);
			atomicAdd(&_accOut[iBatch * nPix + idx].y, weight * olsOut.y);
			atomicAdd(&_accOut[iBatch * nPix + idx].z, weight * olsOut.z);
			atomicAdd(&_accOut[iBatch * nPix + idx].w, weight);
		}
	}
}


__global__ void CombinerGradKernel(const float* _grad, const float* _img, const float* _denoised, const float* _shrinkage,
								   float* _gradDenoised, float* _gradShrinkage, int iBatch, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;

	float4 avgRho = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 deriShrinkage = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float4& iRho = make_float4(_shrinkage[(iBatch * nPix + idx) * 3 + 0], _shrinkage[(iBatch * nPix + idx) * 3 + 1], _shrinkage[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iInGrad = make_float4(_grad[(iBatch * nPix + idx) * 3 + 0], _grad[(iBatch * nPix + idx) * 3 + 1], _grad[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iImg = make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iDenoised = make_float4(_denoised[(iBatch * nPix + idx) * 3 + 0], _denoised[(iBatch * nPix + idx) * 3 + 1], _denoised[(iBatch * nPix + idx) * 3 + 2], 0.f);

			avgRho += iRho;
			deriShrinkage += (iImg - iDenoised) * iInGrad;
		}
	}
	avgRho = avgRho / (float)winSizeSqr;
	deriShrinkage = deriShrinkage / (float)winSizeSqr;

	const float4& cInGrad = make_float4(_grad[(iBatch * nPix + cIdx) * 3 + 0], _grad[(iBatch * nPix + cIdx) * 3 + 1], _grad[(iBatch * nPix + cIdx) * 3 + 2], 0.f);	

	_gradDenoised[(iBatch * nPix + cIdx) * 3 + 0] = (1.f - avgRho.x) * cInGrad.x;
	_gradDenoised[(iBatch * nPix + cIdx) * 3 + 1] = (1.f - avgRho.y) * cInGrad.y;
	_gradDenoised[(iBatch * nPix + cIdx) * 3 + 2] = (1.f - avgRho.z) * cInGrad.z;

	_gradShrinkage[(iBatch * nPix + cIdx) * 3 + 0] = deriShrinkage.x;
	_gradShrinkage[(iBatch * nPix + cIdx) * 3 + 1] = deriShrinkage.y;
	_gradShrinkage[(iBatch * nPix + cIdx) * 3 + 2] = deriShrinkage.z;
}

__global__ void CalcShrinkageKernel(const float* _img, const float* _denoised, const float* _var, float* _shrinkage, int iBatch, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;

	float4 sse = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float4& iImg = make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iDenoised = make_float4(_denoised[(iBatch * nPix + idx) * 3 + 0], _denoised[(iBatch * nPix + idx) * 3 + 1], _denoised[(iBatch * nPix + idx) * 3 + 2], 0.f);
			sse += (iImg - iDenoised) * (iImg - iDenoised);
		}
	}
	float df = (float)(winSizeSqr - 2);
	const float4& var = make_float4(_var[(iBatch * nPix + cIdx) * 3 + 0], _var[(iBatch * nPix + cIdx) * 3 + 1], _var[(iBatch * nPix + cIdx) * 3 + 2], 0.f);

	float4 rho;
	rho.x = fmaxf(0.f, 1.f - (df * var.x) / fmaxf(sse.x, JS_EP));
	rho.y = fmaxf(0.f, 1.f - (df * var.y) / fmaxf(sse.y, JS_EP));
	rho.z = fmaxf(0.f, 1.f - (df * var.z) / fmaxf(sse.z, JS_EP));

	_shrinkage[(iBatch * nPix + cIdx) * 3 + 0] = rho.x;
	_shrinkage[(iBatch * nPix + cIdx) * 3 + 1] = rho.y;
	_shrinkage[(iBatch * nPix + cIdx) * 3 + 2] = rho.z;
}

__global__ void CalcShrinkageGradKernel(const float* _grad, const float* _img, const float* _denoised, const float* _var, 
										float* _gradDenoised, float* _gradVar, int iBatch, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;

	float4 sse = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float4& iImg = make_float4(_img[(iBatch * nPix + idx) * 3 + 0], _img[(iBatch * nPix + idx) * 3 + 1], _img[(iBatch * nPix + idx) * 3 + 2], 0.f);
			const float4& iDenoised = make_float4(_denoised[(iBatch * nPix + idx) * 3 + 0], _denoised[(iBatch * nPix + idx) * 3 + 1], _denoised[(iBatch * nPix + idx) * 3 + 2], 0.f);
			sse += (iImg - iDenoised) * (iImg - iDenoised);
		}
	}
	float df = (float)(winSizeSqr - 2);
	const float4& var = make_float4(_var[(iBatch * nPix + cIdx) * 3 + 0], _var[(iBatch * nPix + cIdx) * 3 + 1], _var[(iBatch * nPix + cIdx) * 3 + 2], 0.f);

	float4 rho;
	rho.x = 1.f - (df * var.x) / fmaxf(sse.x, JS_EP);
	rho.y = 1.f - (df * var.y) / fmaxf(sse.y, JS_EP);
	rho.z = 1.f - (df * var.z) / fmaxf(sse.z, JS_EP);

	const float4& cInGrad = make_float4(_grad[(iBatch * nPix + cIdx) * 3 + 0], _grad[(iBatch * nPix + cIdx) * 3 + 1], _grad[(iBatch * nPix + cIdx) * 3 + 2], 0.f);

	const float4& cImg = make_float4(_img[(iBatch * nPix + cIdx) * 3 + 0], _img[(iBatch * nPix + cIdx) * 3 + 1], _img[(iBatch * nPix + cIdx) * 3 + 2], 0.f);
	const float4& cDenoised = make_float4(_denoised[(iBatch * nPix + cIdx) * 3 + 0], _denoised[(iBatch * nPix + cIdx) * 3 + 1], _denoised[(iBatch * nPix + cIdx) * 3 + 2], 0.f);
	float4 deriDenoised;
	deriDenoised.x = (rho.x > 0.f) ? -2.f * df * var.x * (cImg.x - cDenoised.x) / fmaxf(sse.x * sse.x, JS_EP) : 0.f;
	deriDenoised.y = (rho.y > 0.f) ? -2.f * df * var.y * (cImg.y - cDenoised.y) / fmaxf(sse.y * sse.y, JS_EP) : 0.f;
	deriDenoised.z = (rho.z > 0.f) ? -2.f * df * var.z * (cImg.z - cDenoised.z) / fmaxf(sse.z * sse.z, JS_EP) : 0.f;

	_gradDenoised[(iBatch * nPix + cIdx) * 3 + 0] = deriDenoised.x * cInGrad.x;
	_gradDenoised[(iBatch * nPix + cIdx) * 3 + 1] = deriDenoised.y * cInGrad.y;
	_gradDenoised[(iBatch * nPix + cIdx) * 3 + 2] = deriDenoised.z * cInGrad.z;

	float4 deriVar;
	deriVar.x = (rho.x > 0.f) ? -1.f * df / fmaxf(sse.x, JS_EP) : 0.f;
	deriVar.y = (rho.y > 0.f) ? -1.f * df / fmaxf(sse.y, JS_EP) : 0.f;
	deriVar.z = (rho.z > 0.f) ? -1.f * df / fmaxf(sse.z, JS_EP) : 0.f;

	_gradVar[(iBatch * nPix + cIdx) * 3 + 0] = deriVar.x * cInGrad.x;
	_gradVar[(iBatch * nPix + cIdx) * 3 + 1] = deriVar.y * cInGrad.y;
	_gradVar[(iBatch * nPix + cIdx) * 3 + 2] = deriVar.z * cInGrad.z;
}


void DeallocateCombinerFunction(const GPUDevice &_dev) {
	if (g_accOut)
		cudaFree(g_accOut);
	g_accOut = NULL;
	g_lenAccOut = 0;
}

void WeightAvgFunction(const GPUDevice &_dev, const float* _img, const float* _wgt, float* _out, int nBatch, int height, int width, int winSize) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch) 
		WeightAvgKernel <<< grid, threads, 0, _dev.stream() >>>  (_img, _wgt, _out, iBatch, height, width, winSize);	
}

void WeightAvgGradFunction(const GPUDevice &_dev, const float* _inGrad, const float* _img, const float* _wgt, float* _out, int nBatch, int height, int width, int winSize) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
		
	for (int iBatch = 0; iBatch < nBatch; ++iBatch) 
		WeightAvgGradKernel <<< grid, threads, 0, _dev.stream() >>>  (_img, _inGrad, _wgt, _out, iBatch, height, width, winSize);
}  		

void CalcShrinkageFunc(const GPUDevice &_dev, const float* _img, const float* _denoised, const float* _var, float* _out, int nBatch, int height, int width, int winSize) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch)
		CalcShrinkageKernel << < grid, threads, 0, _dev.stream() >> >  (_img, _denoised, _var, _out, iBatch, height, width, winSize);	
}

void CalcShrinkageGradFunc(const GPUDevice &_dev, const float* _inGrad, const float* _img, const float* _denoised, const float* _var, float* _gradDenoised, float* _gradVar, int nBatch, int height, int width, int winSize) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch)
		CalcShrinkageGradKernel << < grid, threads, 0, _dev.stream() >> >  (_inGrad, _img, _denoised, _var, _gradDenoised, _gradVar, iBatch, height, width, winSize);
}

void CombinerFunc(const GPUDevice &_dev, const float* _img, const float* _denoised, const float* _shrinkage, float* _out, int nBatch, int height, int width, int winSize) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch)
		CombinerKernel << < grid, threads, 0, _dev.stream() >> >  (_img, _denoised, _shrinkage, _out, iBatch, height, width, winSize);
}

void CombinerGradFunc(const GPUDevice &_dev, const float* _inGrad, const float* _img, const float* _denoised, const float* _shrinkage, float* _gradDenoised, float* _gradShrinkage, int nBatch, int height, int width, int winSize) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch)
		CombinerGradKernel << < grid, threads, 0, _dev.stream() >> >  (_inGrad, _img, _denoised, _shrinkage, _gradDenoised, _gradShrinkage, iBatch, height, width, winSize);
}

void OlsFunc(const GPUDevice &_dev, const float* _img, const float* _denoised, const float* _varDenoised, const float* _albedo, const float* _normal, const float* _depth, const float* _vis,
			 float* _out, int nBatch, int height, int width, int winSize, int dimFeat) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	if (g_lenAccOut < nBatch * width * height) {
		if (g_accOut)
			cudaFree(g_accOut);
		cudaError_t cudaStatus = cudaMalloc((void **)&g_accOut, nBatch * width * height * sizeof(float4));
		g_lenAccOut = nBatch * width * height;

		if (cudaStatus != cudaSuccess) {
			printf("Err: Malloc failed - Code: %d\n", cudaStatus);
		}
	}
	// blockwise ols
	cudaMemset(g_accOut, 0, nBatch * width * height * sizeof(float4));
	for (int iBatch = 0; iBatch < nBatch; ++iBatch)
		OlsKernel << < grid, threads, 0, _dev.stream() >> >  (_img, _denoised, _varDenoised, _albedo, _normal, _depth, _vis,															
															  g_accOut, iBatch, height, width, winSize, dimFeat);
	for (int iBatch = 0; iBatch < nBatch; ++iBatch)
		OlsFinalizeKernel << < grid, threads, 0, _dev.stream() >> >  (_denoised, g_accOut, _out, iBatch, height, width);
}

#endif  // GOOGLE_CUDA


