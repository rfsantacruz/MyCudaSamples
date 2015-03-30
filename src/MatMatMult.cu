/*
 * MatMatMult.h
 *
 *  Created on: Mar 16, 2015
 *      Author: rfsantacruz
 */
#include "MatMatMult.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "CudaUtil.h"

__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int width){

	//compute row and column of the target element to compute
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	//check for safety if target element is within matrix dimensions
	if(row < width && col < width){
		//perform "dot product" line and column
		float sum = 0.0f;
		for (int k = 0; k < width; ++k) {
			sum += d_M[row * width + k] * d_N[k * width + col];
		}
		//assign target element value
		d_P[row * width + col] = sum;
	}
}

void matrixMultHost(const float* h_M, const float* h_N, float* h_P, int width){

	float *dev_M, *dev_N, *dev_P;
	int size = width * width * sizeof(float);

	//allocate device memory
	CUDA_CHECK(cudaMalloc(&dev_M,size));
	CUDA_CHECK(cudaMalloc(&dev_N,size));
	CUDA_CHECK(cudaMalloc(&dev_P,size));

	//copy data to device
	CUDA_CHECK(cudaMemcpy(dev_M, h_M, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_N, h_N, size, cudaMemcpyHostToDevice));

	//configure, lunch and synchronize kernel
	dim3 blockDim(1,1,1);
	dim3 gridDim(1,1,1);

	CUDA_CHECK(cudaDeviceSynchronize());

	//copy data to host
	CUDA_CHECK(cudaMemcpy(h_P, dev_P,size, cudaMemcpyDeviceToHost));

	//free device memory
	CUDA_CHECK(cudaFree(dev_M));
	CUDA_CHECK(cudaFree(dev_N));
	CUDA_CHECK(cudaFree(dev_P));

}
