#include "Histogram.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>

//kernel for computing histogram right in memory
__global__ void hist_inGlobal (const int* values, int length, int* hist){

	//compute index and interval
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	//iterate over index and interval since it is less than the total length
	while(idx < length){
		//get value
		int val = values[idx];
		//increment value frequency on histogram using atomic in order to be thread safe
		atomicAdd(&hist[val], 1);
		idx += stride;
	}
}

//computer partial histogram on shared memory and mix them on global memory
__global__ void hist_inShared (const int* values, int length, int* hist){

	//load shared memory
	extern __shared__ int shHist[];
	shHist[threadIdx.x] = 0;
	__syncthreads();

	//compute index and interval
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	//iterate over index and interval since it is less than the total length
	while(idx < length){
		int val = values[idx];
		//increment value frequency on histogram using atomic in order to be thread safe
		atomicAdd(&shHist[val], 1);
		idx += stride;
	}

	//combine partial histogram on shared memory to create a full histogram
	__syncthreads();
	atomicAdd(&hist[threadIdx.x], shHist[threadIdx.x]);
}

void histogramHost(const int* values, int length, int* hist, int hist_length){

	//variables
	int *dev_val, *dev_hist;
	int size_val = length * sizeof(int);
	int size_hist = hist_length * sizeof(int);

	//allocate memory on gpu
	checkCudaErrors(cudaMalloc(&dev_val,size_val));
	checkCudaErrors(cudaMalloc(&dev_hist,size_hist));

	//copy data to gpu
	checkCudaErrors(cudaMemcpy(dev_val,values,size_val,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_hist,hist,size_hist,cudaMemcpyHostToDevice));

	//configure, lunch and synchronize kernel
	dim3 blocks(hist_length,1,1);
	dim3 grid(ceil(length/((float)hist_length)),1,1);
	int shMem = hist_length * sizeof(int);
	hist_inShared<<<grid,blocks,shMem>>>(dev_val, length, dev_hist);
	checkCudaErrors(cudaDeviceSynchronize());

	//copy data back to host
	checkCudaErrors(cudaMemcpy(hist,dev_hist,size_hist,cudaMemcpyDeviceToHost));

	//free GPU memory
	checkCudaErrors(cudaFree(dev_val));
	checkCudaErrors(cudaFree(dev_hist));
}
