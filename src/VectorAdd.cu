
#include "VectorAdd.h"
#include "CudaUtil.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>
#include <stdio.h>


__global__ void vectorAddKernel(float* inputA, float* inputB, float* output, int length){

	//compute element index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//add an vector element
	if(idx < length) output[idx] = inputA[idx] + inputB[idx];

}


void vectorAddHost(const float* A, const float* B, float* output, int length){

	float *devA, *devB, *devC;
	int size = length * sizeof(float);

	//allocate device memory
	CUDA_CHECK(cudaMalloc(&devA, size));
	CUDA_CHECK(cudaMalloc(&devB, size));
	CUDA_CHECK(cudaMalloc(&devC, size));

	//copy values from host to device
	CUDA_CHECK(cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice));

	//lunch kernel
	dim3 blockDim(256, 1, 1);
	dim3 gridDim((int)ceil(length/(WARP_SIZE*1.0)), 1, 1);
	vectorAddKernel<<<gridDim,blockDim>>>(devA, devB, devC, length);

	//synchronize with host with async kernel
	CUDA_CHECK(cudaDeviceSynchronize());

	//copy values from device back to host
	CUDA_CHECK(cudaMemcpy(output, devC, size, cudaMemcpyDeviceToHost));

	//free device memory
	CUDA_CHECK(cudaFree(devA));
	CUDA_CHECK(cudaFree(devB));
	CUDA_CHECK(cudaFree(devC));
}
