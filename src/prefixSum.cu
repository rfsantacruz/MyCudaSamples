#include "prefixSum.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>

//this kernel is just didatic and is too slow
__global__ void inefficient_prefixSum(float* in, int in_length, float* out ){

	//shared memory declaration
	extern __shared__ float DSM[];

	//compute index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < in_length){
		//load on shared memory
		DSM[threadIdx.x] = in[idx];

		//compute prefix_sum making sequence of sums
		for(int stride = 1; stride <= threadIdx.x; stride *= 2){
			__syncthreads();

			DSM[threadIdx.x] =  DSM[threadIdx.x] + DSM[threadIdx.x - stride];
		}

		out[idx] = DSM[threadIdx.x];

	}

}

//prefix sum for small array (lunch with just one block)
__global__ void prefixSum_UniqueBlock(float* in, int in_length, float* out ){

	//shared memory declaration
	extern __shared__ float DSM[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//load in shared memory
	if(idx < in_length){
		DSM[threadIdx.x] = in[idx];

		//partial sums phase
		for(int stride = 1; stride <= blockDim.x; stride *= 2){
			__syncthreads();
			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux < blockDim.x)
				DSM[index_aux] += DSM[index_aux - stride];
		}

		//reduction phase
		for(int stride=blockDim.x/4 ; stride > 0 ; stride /= 2){
			__syncthreads();

			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux + stride < blockDim.x)
				DSM[index_aux + stride] += DSM[index_aux];
		}

		__syncthreads();

		out[idx] = DSM[threadIdx.x];

	}

}

//prefix sum for multiple blocks
__global__ void prefixSum_multiBlocks(float* in, int in_length, float* out, float* temp ){

	extern __shared__ float DSM[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//load in shared memory
	if(idx < in_length){
		DSM[threadIdx.x] = in[idx];

		//partial sums phase
		for(int stride = 1; stride <= blockDim.x; stride *= 2){
			__syncthreads();
			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux < blockDim.x)
				DSM[index_aux] += DSM[index_aux - stride];
		}

		//reduction phase
		for(int stride=blockDim.x/4 ; stride > 0 ; stride /= 2){
			__syncthreads();

			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux + stride < blockDim.x)
				DSM[index_aux + stride] += DSM[index_aux];
		}

		__syncthreads();

		//save intermediary values on temp to post combine for multi blocks
		if(threadIdx.x == 0)
			temp[blockIdx.x] = DSM[blockDim.x - 1];

		out[idx] = DSM[threadIdx.x];

	}

}

//combine for multiple blocks
__global__ void prefixsum_combine(float* in, int in_length, float* out, int out_length){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < out_length && blockIdx.x > 0){
		out[idx] += in[blockIdx.x - 1];
	}

}

void prefixSumHost(const float* in, int in_length, float* out ){

	//declaring variables
	float blockSize = 256;
	float *dev_in, *dev_out, *dev_temp;
	int size  = in_length * sizeof(float);
	int tempSize = blockSize * sizeof(float);

	//allocate memory
	checkCudaErrors(cudaMalloc(&dev_in, size));
	checkCudaErrors(cudaMalloc(&dev_out, size));
	checkCudaErrors(cudaMalloc(&dev_temp, tempSize));

	//copy data to device
	checkCudaErrors(cudaMemcpy(dev_in,in,size,cudaMemcpyHostToDevice));


	//lunch kernel to compute prefix sum for each block and save partial block sum on dev_temp
	dim3 blocks(blockSize,1,1);
	dim3 grid(ceil(in_length/blockSize),1,1);
	int shMem = blockSize * sizeof(float);
	prefixSum_multiBlocks<<<grid,blocks,shMem>>>(dev_in, in_length, dev_out, dev_temp);
	checkCudaErrors(cudaDeviceSynchronize());

	//lunch kernel to compute prefix sum on each block total cell stores in dev_temp
	int topBlocksQtd = ceil(in_length/blockSize);
	prefixSum_UniqueBlock<<<1,topBlocksQtd,topBlocksQtd * sizeof(float)>>>(dev_temp,topBlocksQtd,dev_temp);
	checkCudaErrors(cudaDeviceSynchronize());

	//lunch kernel to combine blocks results
	prefixsum_combine<<<grid,blocks>>>(dev_temp, topBlocksQtd, dev_out, in_length);
	checkCudaErrors(cudaDeviceSynchronize());

	//copy data to device
	checkCudaErrors(cudaMemcpy(out,dev_out,size,cudaMemcpyDeviceToHost));

	//free memory space
	checkCudaErrors(cudaFree(dev_in));
	checkCudaErrors(cudaFree(dev_out));

}
