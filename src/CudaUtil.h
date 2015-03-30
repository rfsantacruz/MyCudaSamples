/*
 * CudaUtil.h
 *
 *  Created on: Mar 10, 2015
 *      Author: rfsantacruz
 */

#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

#define MAX_THREAD_PERBLOCK 1024
#define MAX_THREAD_PERMP 2048

#define WARP_SIZE 32

#define MAX_BLOCK_X 1024
#define MAX_BLOCK_Y 1024
#define MAX_BLOCK_Z 60

#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65535
#define MAX_GRID_Z 65535

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


#endif /* CUDAUTIL_H_ */
