#include "VectorReduction.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>

__constant__ float mask_cte[3];

__global__ void conv1D(float* dev_in, int input_width, int mask_width, float* dev_out){

	//index computation
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < input_width){

		float result = 0.0f;
		//start index on the input array
		int start = idx - (mask_width/2);
		//compute convolution
		for(int j = 0; j<mask_width ; j++ ){
			//boundary check
			if((start + j >= 0) && (start + j < input_width)){
				result += dev_in[start + j] * mask_cte[j];
			}
		}
		//store final value
		dev_out[idx] = result;
	}
}

__global__ void tiledConv1D(float* dev_in, int input_width, int mask_width, float* dev_out){

	//index computation
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//load device memory coperatively
	extern __shared__ float ds_in[];
	int n = mask_width/2;

	//load left halo elements uising the las thread in current block
	int haloIdxLeft = (blockIdx.x-1)*blockDim.x + threadIdx.x;
	if(threadIdx.x >= blockDim.x - n)
		ds_in[threadIdx.x - (blockDim.x - n)] = haloIdxLeft < 0 ? 0 : dev_in[haloIdxLeft];

	//load central elements
	ds_in[n + threadIdx.x]

	//load right halo elements


	if(idx < input_width){

		float result = 0.0f;
		//start index on the input array
		int start = idx - (mask_width/2);
		//compute convolution
		for(int j = 0; j<mask_width ; j++ ){
			//boundary check
			if((start + j >= 0) && (start + j < input_width)){
				result += dev_in[start + j] * mask_cte[j];
			}
		}
		//store final value
		dev_out[idx] = result;
	}
}

__global__ void conv2D(float* dev_in, int input_width, float* dev_mask, int mask_width,float* dev_out){

}


void conv1DHost(const float* input, const int input_width, const float* mask, const int mask_width, float* output){

	//variables
	float *dev_in, *dev_mask, *dev_out;
	int size_InOut = input_width * sizeof(float);
	int size_mask = mask_width * sizeof(float);

	checkCudaErrors(cudaMemcpyToSymbol(mask_cte,mask,size_mask));


	//allocate memory on gpu
	checkCudaErrors(cudaMalloc(&dev_in,size_InOut));
	checkCudaErrors(cudaMalloc(&dev_out,size_InOut));
	checkCudaErrors(cudaMalloc(&dev_mask, size_mask));

	//copy data to gpu
	checkCudaErrors(cudaMemcpy(dev_in,input,size_InOut,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_mask,mask,size_mask,cudaMemcpyHostToDevice));

	//configure, lunch and synchronize kernel
	dim3 blocks(256,1,1);
	dim3 grid(ceil(input_width/256.0f),1,1);
	conv1D<<<grid,blocks>>>(dev_in,input_width,mask_width,dev_out);
	checkCudaErrors(cudaDeviceSynchronize());

	//copy data back to host
	checkCudaErrors(cudaMemcpy(output,dev_out,size_InOut,cudaMemcpyDeviceToHost));

	//free memory
	checkCudaErrors(cudaFree(dev_in));
	checkCudaErrors(cudaFree(dev_mask));
	checkCudaErrors(cudaFree(dev_out));

}

void conv2DHost(const float* input, const int input_width, const float* mask, const int mask_width, float* output){}
