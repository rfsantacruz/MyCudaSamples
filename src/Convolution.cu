#include "Convolution.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>

//constant memory should be declared as global variable
__constant__ float MASK_1D_CTE_MEM[MASK_WIDTH];

__constant__ float MASK_2D_CTE_MEM[MASK_WIDTH*MASK_WIDTH];

__global__ void conv1D(float* dev_in, int input_width, float* dev_out){

	//index computation
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < input_width){

		float result = 0.0f;
		//start index on the input array
		int start = idx - (MASK_WIDTH/2);
		//compute convolution
		for(int j = 0; j<MASK_WIDTH ; j++ ){
			//boundary check
			if((start + j >= 0) && (start + j < input_width)){
				result += dev_in[start + j] * MASK_1D_CTE_MEM[j];
			}
		}
		//store final value
		dev_out[idx] = result;
	}
}

__global__ void tiledConv1D(float* dev_in, int input_width, float* dev_out){


	extern __shared__ float N_ds[];

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < input_width){

		int n = MASK_WIDTH/2;

		int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
		if (threadIdx.x >= blockDim.x - n) {
			N_ds[threadIdx.x - (blockDim.x - n)] = halo_index_left < 0 ? 0 : dev_in[halo_index_left];
		}

		N_ds[n + threadIdx.x] = dev_in[blockIdx.x*blockDim.x + threadIdx.x];

		int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
		if (threadIdx.x < n) {
			N_ds[n + blockDim.x + threadIdx.x] = halo_index_right >= input_width ? 0 : dev_in[halo_index_right];
		}

		__syncthreads();

		float Pvalue = 0;
		for(int j = 0; j < MASK_WIDTH; j++) {
			Pvalue += N_ds[threadIdx.x + j]*MASK_1D_CTE_MEM[j];
		}

		dev_out[i] = Pvalue;
	}
}

void conv1DHost(const float* input, const int input_width, const float* mask, float* output){

	//variables
	float *dev_in, *dev_out;
	int size_InOut = input_width * sizeof(float);
	int size_mask = MASK_WIDTH * sizeof(float);

	//allocate memory on gpu
	checkCudaErrors(cudaMalloc(&dev_in,size_InOut));
	checkCudaErrors(cudaMalloc(&dev_out,size_InOut));

	//copy data to gpu
	checkCudaErrors(cudaMemcpy(dev_in,input,size_InOut,cudaMemcpyHostToDevice));
	//copy memory to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(MASK_1D_CTE_MEM,mask,size_mask));

	//configure, lunch and synchronize kernel
	dim3 blocks(256,1,1);
	dim3 grid(ceil(input_width/256.0f),1,1);
	int shMem = (256 + MASK_WIDTH - 1) * sizeof(float);
	tiledConv1D<<<grid,blocks,shMem>>>(dev_in,input_width,dev_out);
	checkCudaErrors(cudaDeviceSynchronize());

	//copy data back to host
	checkCudaErrors(cudaMemcpy(output,dev_out,size_InOut,cudaMemcpyDeviceToHost));

	//free memory
	checkCudaErrors(cudaFree(dev_in));
	checkCudaErrors(cudaFree(dev_out));

}

__global__ void conv2D(float* dev_in, int input_width, float* dev_out){

	//index computation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < input_width && y < input_width){

		float result = 0.0f;
		//start index on the input array
		int startX = x - (MASK_WIDTH/2);
		int startY = y - (MASK_WIDTH/2);
		//compute convolution
		for (int i = 0; i < MASK_WIDTH; ++i) {
			for(int j = 0; j < MASK_WIDTH; j++ ){

				//boundary check
				if((startY + i >= 0) && (startY + i < input_width) && (startX + j >= 0) && (startX + j < input_width)){
					result += dev_in[((startY + i)*input_width) + (startX + j)] * MASK_2D_CTE_MEM[(i * MASK_WIDTH) + j];
				}
			}
		}
		//store final value
		dev_out[(y * input_width) + x] = result;
	}
}

__device__ bool isValid(int x, int y, int limit){
	bool ret = true;

	if(x < 0 || x >= limit)
		ret = false;

	if(y < 0 || y >= limit)
		ret = false;

	return ret;


}

__global__ void tiledConv2D(float* dev_in, int input_width, float* dev_out){

	extern __shared__ float N_ds[];
	int n = MASK_WIDTH/2;
	int shWidth = blockDim.x + 2*n;


	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x < input_width && y < input_width){

		//load left elements
		int hx_l = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
		int hy_l = blockIdx.y*blockDim.y + threadIdx.y;
		if (threadIdx.x >= blockDim.x - n) {
			N_ds[ ((threadIdx.y + n) * shWidth) + (threadIdx.x - (blockDim.x - n))] = !isValid(hx_l,hy_l,input_width) ? 0 : dev_in[hy_l * input_width + hx_l];
		}

		//load top elements
		int hx_t = blockIdx.x*blockDim.x + threadIdx.x;
		int hy_t = (blockIdx.y-1) * blockDim.y + threadIdx.y;
		if (threadIdx.y >= blockDim.y - n) {
			N_ds[ ((threadIdx.y - (blockDim.y - n)) * shWidth) + (threadIdx.x + n)] = !isValid(hx_t,hy_t,input_width) ? 0 : dev_in[hy_t * input_width + hx_t];
		}

		//load right elements
		int hx_r = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
		int hy_r = blockIdx.y * blockDim.y + threadIdx.y;
		if (threadIdx.x < n) {
			N_ds[ ((threadIdx.y + n) * shWidth) + (n + blockDim.x + threadIdx.x)] = !isValid(hx_r,hy_r,input_width) ? 0 : dev_in[hy_r * input_width + hx_r];
		}

		//load bottom elements
		int hx_b = blockIdx.x*blockDim.x + threadIdx.x;
		int hy_b = (blockIdx.y + 1) * blockDim.y + threadIdx.y;
		if (threadIdx.y < n) {
			N_ds[ ((n + blockDim.y + threadIdx.y) * shWidth) + (threadIdx.x + n)] = !isValid(hx_b,hy_b,input_width) ? 0 : dev_in[hy_b * input_width + hx_b];
		}

		//load corners
		//load top left corner
		int hx_tl = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
		int hy_tl = (blockIdx.y - 1)*blockDim.y + threadIdx.y;
		if (threadIdx.x >= blockDim.x - n && threadIdx.y >= blockDim.y - n) {
			N_ds[ ((threadIdx.y - (blockDim.y - n)) * shWidth) + (threadIdx.x - (blockDim.x - n))] = !isValid(hx_tl,hy_tl,input_width) ? 0 : dev_in[hy_tl * input_width + hx_tl];
		}
		//load top right corner
		int hx_tr = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
		int hy_tr = (blockIdx.y - 1)*blockDim.y + threadIdx.y;
		if (threadIdx.x < n && threadIdx.y >= blockDim.y - n) {
			N_ds[((threadIdx.y - (blockDim.y - n)) * shWidth) + (threadIdx.x + n + blockDim.x)] = !isValid(hx_tr,hy_tr,input_width) ? 0 : dev_in[hy_tr * input_width + hx_tr];
		}
		//load bottom right corner
		int hx_br = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
		int hy_br = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
		if (threadIdx.x < n && threadIdx.y < n) {
			N_ds[((threadIdx.y + n + blockDim.y) * shWidth) + (threadIdx.x + n + blockDim.x)] = !isValid(hx_br,hy_br,input_width) ? 0 : dev_in[hy_br * input_width + hx_br];
		}
		//load bottom left corner
		int hx_bl = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
		int hy_bl = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
		if (threadIdx.x >= blockDim.x - n && threadIdx.y < n) {
			N_ds[((threadIdx.y + n + blockDim.y) * shWidth) + (threadIdx.x - (blockDim.x - n))] = !isValid(hx_bl,hy_bl,input_width) ? 0 : dev_in[hy_bl * input_width + hx_bl];
		}
		//load middle
		N_ds[((n + threadIdx.y)*shWidth) + (n + threadIdx.x)] = !isValid(y,x,input_width) ? 0 : dev_in[y * input_width + x];

		__syncthreads();


		float Pvalue = 0;
		for(int i = 0; i < MASK_WIDTH; i++){
			for(int j = 0; j < MASK_WIDTH; j++) {
				Pvalue += N_ds[((threadIdx.y + i)*shWidth) + threadIdx.x + j] * MASK_2D_CTE_MEM[(i * MASK_WIDTH) + j];
			}
		}

		dev_out[(y * input_width) + x] = Pvalue;
	}
}

void conv2DHost(const float* input, const int input_width, const float* mask, float* output){

	//variables
	float *dev_in, *dev_out;
	int size_InOut = input_width * input_width * sizeof(float);
	int size_mask = MASK_WIDTH * MASK_WIDTH  * sizeof(float);


	//allocate memory on gpu
	checkCudaErrors(cudaMalloc(&dev_in,size_InOut));
	checkCudaErrors(cudaMalloc(&dev_out,size_InOut));


	//copy data to gpu
	checkCudaErrors(cudaMemcpy(dev_in,input,size_InOut,cudaMemcpyHostToDevice));
	//copy memory to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(MASK_2D_CTE_MEM, mask, size_mask));

	//configure, lunch and synchronize kernel
	dim3 blocks(16,16,1);
	dim3 grid(ceil(input_width/16.0f), ceil(input_width/16.0f), 1);

	//conv2D<<<grid,blocks>>>(dev_in,input_width,dev_out);
	int shMem = pow(16 + (2*(MASK_WIDTH/2)),2) * sizeof(float);
	tiledConv2D<<<grid,blocks,shMem>>>(dev_in,input_width,dev_out);


	checkCudaErrors(cudaDeviceSynchronize());

	//copy data back to host
	checkCudaErrors(cudaMemcpy(output,dev_out,size_InOut,cudaMemcpyDeviceToHost));

	//free memory
	checkCudaErrors(cudaFree(dev_in));
	checkCudaErrors(cudaFree(dev_out));
}
