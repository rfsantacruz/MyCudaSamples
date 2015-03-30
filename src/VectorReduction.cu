#include "VectorReduction.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>

__global__ void reductionKernel(float* vec, int width, double* sumUp){

	//shared memory instantiation
	extern __shared__ float partialSum[];

	//index for global memory
	int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
	//index for shared memory
	int b_idx = threadIdx.x;

	//load shared memory from global memory
	partialSum[b_idx] = g_idx < width ? vec[g_idx] : 0;

	//reduction inside blocks
	for(int stride = blockDim.x/2; stride >= 1 ; stride = stride/2){

		__syncthreads();
		if(b_idx < stride ){
			partialSum[b_idx] = partialSum[b_idx] + partialSum[b_idx + stride];
		}
	}

	//reduction for grid using just thread 0 of each block
	if(b_idx == 0){
		//coppy value back to global memory
		vec[g_idx] = partialSum[b_idx];

		//reduction
		for(int stride = (gridDim.x * blockDim.x)/2; stride>=blockDim.x; stride = stride/2){

			__syncthreads();
			if(g_idx < stride){
				vec[g_idx] = vec[g_idx] + vec[g_idx + stride];
			}
		}
	}

	//save result in output variable
	if(g_idx == 0)
		(*sumUp) = vec[g_idx];
}

void reductionHost(const float* h_a, const int width, double& h_sum){

	float* dev_a;
	double *dev_sumUp, *h_sumUp;
	int size = width * sizeof(float);

	//cuda allocate memory on device
	checkCudaErrors(cudaMalloc(&dev_a, size));
	checkCudaErrors(cudaMalloc(&dev_sumUp, sizeof(double)));
	h_sumUp = new double;

	//copy arguments to device memory
	checkCudaErrors(cudaMemcpy(dev_a,h_a,size,cudaMemcpyHostToDevice));

	printf("Kernel Lunched");
	//configure, lunch and synchronize the kernel
	dim3 blockDim(256,1,1);
	//compute the largest power 2 function
	dim3 gridDim(ceil(width/256.0f),1,1);
	int sharedMemSize = 256 * sizeof(float);
	reductionKernel<<<gridDim,blockDim,sharedMemSize>>>(dev_a, width, dev_sumUp);
	checkCudaErrors(cudaDeviceSynchronize());
	printf("Kernel Finished");

	//copy the result back to host
	checkCudaErrors(cudaMemcpy(h_sumUp,dev_sumUp,sizeof(double),cudaMemcpyDeviceToHost));
	h_sum = (*h_sumUp);

	//free memory
	checkCudaErrors(cudaFree(dev_a));
	checkCudaErrors(cudaFree(dev_sumUp));
	delete h_sumUp;

}
