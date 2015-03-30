#include "MatrixVectorMult.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaUtil.h"
#include <math.h>
#include <stdio.h>

__global__ void matrixVectorMultKernel(float* fltMatrix, float* vec, float* output, int rows, int columns){

	int row = blockDim.x * blockIdx.x + threadIdx.x;

	if(row < rows){
		float sum = 0.0f;
		for (int col = 0; col < columns; ++col) {
			sum += fltMatrix[row * columns + col] + vec[col];
		}

		output[row] = sum;
	}

}

void MatrixVectorMultHost(const float* fltMatA, const float* vecB, float *output, int rows, int columns){

	float *devFltMat, *devVecB, *devOutput;
	int matEls = rows * columns;
	int vecEls = columns;
	int outPutEls = rows;
	int matSize = matEls * sizeof(float);
	int vecSize = vecEls * sizeof(float);
	int outputSize = outPutEls * sizeof(float);

	//Allocate memory on GPU
	CUDA_CHECK(cudaMalloc(&devFltMat, matSize));
	CUDA_CHECK(cudaMalloc(&devVecB, vecSize));
	CUDA_CHECK(cudaMalloc(&devOutput, outputSize));

	//Copy memory to GPU
	CUDA_CHECK(cudaMemcpy(devFltMat,fltMatA,matSize,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(devVecB,vecB,vecSize,cudaMemcpyHostToDevice));

	//configure, lunch and synchronize kernel
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(ceil(rows/256.0f), 1, 1);
	matrixVectorMultKernel<<<gridDim,blockDim>>>(devFltMat,devVecB,devOutput,rows,columns);
	CUDA_CHECK(cudaDeviceSynchronize());

	//copy memory back to host
	CUDA_CHECK(cudaMemcpy(output, devOutput, outputSize, cudaMemcpyDeviceToHost));

	//free device memory
	CUDA_CHECK(cudaFree(devFltMat));
	CUDA_CHECK(cudaFree(devVecB));
	CUDA_CHECK(cudaFree(devOutput));

}
