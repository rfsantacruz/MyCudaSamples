#include "MyMatrixAdd.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaUtil.h"
#include <stdio.h>
#include <math.h>

//kernel computes each element of matrix per thread
__global__ void matrixAdd_A_Kernel(float* A, float* B, float* C, size_t pitch, int width){

	//compute indexes
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int rowWidthWithPad = pitch/sizeof(float);


	if(row < width && col < width)
		C[row * rowWidthWithPad  + col] = A[row * rowWidthWithPad  + col] + B[row * rowWidthWithPad  + col];

}

//kernel computes each row of matrix per thread
__global__ void matrixAdd_B_Kernel(float* A, float* B, float* C, size_t pitch, int width){

	//compute indexes
	int row = blockIdx.x * blockDim.x + threadIdx.x;


	int rowWidthWithPad = pitch/sizeof(float);

	if(row < width){
		for (int col = 0; col < width; ++col) {
			if(col < width)
				C[row * rowWidthWithPad  + col] = A[row * rowWidthWithPad  + col] + B[row * rowWidthWithPad  + col];
		}
	}


}

//kernel computes each row of matrix per thread
__global__ void matrixAdd_C_Kernel(float* A, float* B, float* C, size_t pitch, int width){

	//compute indexes
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	int rowWidthWithPad = pitch/sizeof(float);

	if(col < width){
		for (int row = 0; row < width; ++row) {
			if(row < width)
				C[row * rowWidthWithPad  + col] = A[row * rowWidthWithPad  + col] + B[row * rowWidthWithPad  + col];
		}
	}


}

void matrixAddHost(const float* A, const float* B, float* C, int width){

	//define variables
	float *devA, *devB, *devC;
	size_t pitch;

	//alocating memory
	CUDA_CHECK(cudaMallocPitch(&devA,&pitch,width * sizeof(float), width));
	CUDA_CHECK(cudaMallocPitch(&devB,&pitch,width * sizeof(float), width));
	CUDA_CHECK(cudaMallocPitch(&devC,&pitch,width * sizeof(float), width));

	//copy memory to the device
	CUDA_CHECK(cudaMemcpy2D(devA, pitch, A, width * sizeof(float),width * sizeof(float), width,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2D(devB, pitch, B, width * sizeof(float),width * sizeof(float), width,cudaMemcpyHostToDevice));

	//configure, Lunch and synchronize kernel
	dim3 blockDim(16, 16,1);
	dim3 gridDim(ceil(width/16.0), ceil(width/16.0), 1);
	matrixAdd_A_Kernel<<<gridDim,blockDim>>>(devA,devB,devC,pitch,width);
	CUDA_CHECK(cudaDeviceSynchronize());

	//copy device memory to host
	CUDA_CHECK(cudaMemcpy2D(C, width * sizeof(float), devC, pitch, width * sizeof(float), width, cudaMemcpyDeviceToHost));

	//free device memory
	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFree(devB));
	CUDA_CHECK(cudaFree(devC));
}
