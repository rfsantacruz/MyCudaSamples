/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef QUERYDEVICE_H
#define QUERYDEVICE_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaUtil.h"

int getDeviceInfo() {

	//get the number of installed devices
	int devQtd;
	cudaGetDeviceCount(&devQtd);

	fprintf(stdout,"Detected %d CUDA capable device(s)\n", devQtd);

	//get device properties
	cudaDeviceProp props;
	for (int devIdx = 0; devIdx < devQtd; ++devIdx) {
		CUDA_CHECK(cudaGetDeviceProperties(&props, devIdx));

		//print device property
		fprintf(stdout,"Device %d : %s\n", devIdx, props.name);
		fprintf(stdout,"CUDA capability %d\n", props.major);
		fprintf(stdout,"Total amount of global memory: %d\n", props.totalGlobalMem);

	}
	cudaDeviceReset();

	return 0;
}


#endif /* QUERYDEVICE_H_ */

