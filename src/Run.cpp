#include "QueryDevice.h"
#include "VectorAdd.h"
#include "MyMatrixAdd.h"
#include "MatrixVectorMult.h"
#include "VectorReduction.h"
#include "Convolution.h"
#include "prefixSum.h"
#include "Histogram.h"
#include <math.h>
#include <iostream>
#include <helper_timer.h>

void testHistogram(){

	int width = 2048;
	int *A, *hist;
	srand (time(NULL));

	A = new int[width];
	for (int el = 0; el < width; el++) {
		A[el] = rand() % 256;
	}
	hist = new int[256];
	for (int el = 0; el < 256; el++) {
			hist[el] = 0;
	}

	histogramHost(A,width,hist,256);

	double sumUp = 0;
	for (int el = 0; el < 256; el++) {
		sumUp += hist[el];
	}

	printf("Histogram check sum: %f\n", sumUp);

	delete[] A; delete[] hist;
}

void testPrefixSum(){

	float *in, *out;
	int length = 1024;

	in = new float[length];
	for (int el = 0; el < length; el++) {
		in[el] = el + 1;
	}

	out = new float[length];
	prefixSumHost(in,length,out);
	float sumUp = 0.0f;
	for (int el = 0; el < length; el++) {
		sumUp += out[el];
	}

	//printf("Prefix Sum Expected Result for 1024 input: %f\n", 32896.0f );
	printf("Prefix Sum Result: %f\n", sumUp);

	delete[] in; delete[] out;

}

void testConvolution2D(){

	float *in, *mask, *out;
	int InOut_width = 512;


	in = new float[InOut_width*InOut_width];
	for (int row = 0; row < InOut_width; row++) {
		for (int col = 0; col < InOut_width; col++) {
			in[(row * InOut_width) + col] = 1;
		}
	}

	mask = new float[MASK_WIDTH * MASK_WIDTH];
	for (int row = 0; row < MASK_WIDTH; row++) {
		for (int col = 0; col < MASK_WIDTH; col++) {
			mask[(row * MASK_WIDTH) + col] = 1;
		}
	}

	out = new float[InOut_width*InOut_width];
	conv2DHost(in,InOut_width,mask,out);
	float sumUp = 0.0f;
	for (int row = 0; row < InOut_width; row++) {
		for (int col = 0; col < InOut_width; col++) {
			sumUp += out[(row * InOut_width) + col];
		}

	}
	printf("Convolution 2D result: %f\n", sumUp);

	delete[] in; delete[] mask; delete[] out;
}

void testConvolution1D(){

	float *in, *mask, *out;
	int InOut_width = 1024;


	in = new float[InOut_width];
	for (int el = 0; el < InOut_width; el++) {
		in[el] = 1;
	}

	mask = new float[MASK_WIDTH];
	for (int el = 0; el < MASK_WIDTH; el++) {
		mask[el] = 1;
	}

	out = new float[InOut_width];
	conv1DHost(in,InOut_width,mask,out);
	float sumUp = 0.0f;
	for (int el = 0; el < InOut_width; el++) {
		sumUp += out[el];
	}

	printf("Convolution result: %f\n", sumUp);

	delete[] in; delete[] mask; delete[] out;
}

void testVetorReduction(){

	int width = 1024;
	float* A;
	double sumUp = 0;

	A = new float[width];
	for (int el = 0; el < width; el++) {
		A[el] = 1;
	}

	reductionHost(A,width,sumUp);

	printf("Vector Reduction final result: %f\n", sumUp);

	delete[] A;

}


void testMatrixVectorMult(){
	// Size of vectors
	int rows = 1000;
	int columns = 100;

	// Host input vectors
	float *A, *B, *C;

	// Allocate and initialize memory for each vector on host
	A = new float[rows * columns];
	B = new float[columns];
	C = new float[rows];


	for( int row = 0; row < rows; row++ ) {
		for (int col = 0; col < columns; ++col) {
			int i = row * columns + col;
			A[i] = sin(col)*sin(col);
		}
	}

	for (int col = 0; col < columns; ++col) {
		B[col] = cos(col)*cos(col);
	}

	//call host function to call kernels
	MatrixVectorMultHost(A, B, C, rows, columns);

	// Sum up vector c and print result divided by n, this should equal 1 within error
	double sum = 0;
	for(int i=0; i < rows; i++)
		sum += C[i];
	printf("Matrix-Vector Multiplication final result: %f\n", sum/(rows*columns));

	// Release host memory
	delete[] A;
	delete[] B;
	delete[] C;
}


void testMatrixAdd(){
	// Size of vectors
	int width = 1000;

	// Host input vectors
	float *A, *B, *C;

	// Allocate and initialize memory for each vector on host
	A = new float[width * width];
	B = new float[width * width];
	C = new float[width * width];

	for( int row = 0; row < width; row++ ) {
		for (int col = 0; col < width; ++col) {
			int i = row * width + col;
			A[i] = sin(i)*sin(i);
			B[i] = cos(i)*cos(i);
		}
	}

	//call host function to call kernels
	matrixAddHost(A, B, C, width);

	// Sum up vector c and print result divided by n, this should equal 1 within error
	double sum = 0;
	for(int i=0; i < width*width; i++)
		sum += C[i];
	printf("Matrix Add final result: %f\n", sum/(width*width));

	// Release host memory
	delete[] A;
	delete[] B;
	delete[] C;
}

void testVecAdd(){
	// Size of vectors
	int length = 100000;

	// Host input vectors
	float *A, *B, *C;

	// Allocate and initialize memory for each vector on host
	A = new float[length];
	B = new float[length];
	C = new float[length];

	for( int i = 0; i < length; i++ ) {
		A[i] = sin(i)*sin(i);
		B[i] = cos(i)*cos(i);
	}

	//call host function to call kernels
	vectorAddHost(A, B, C, length);

	// Sum up vector c and print result divided by n, this should equal 1 within error
	double sum = 0;
	for(int i=0; i<length; i++)
		sum += C[i];

	printf("Vector Add final result: %f\n", sum/length);

	// Release host memory
	delete[] A;
	delete[] B;
	delete[] C;
}

int main(){

	StopWatchInterface *timer_compute = NULL;
	sdkCreateTimer(&timer_compute);
	sdkStartTimer(&timer_compute);

	//getDeviceInfo();

	//testVecAdd();

	//testMatrixAdd();

	//testMatrixVectorMult();

	//testVetorReduction();

	//testConvolution1D();

	//testConvolution2D();

	//testPrefixSum();

	testHistogram();

	sdkStopTimer(&timer_compute);
	printf("CPU timer : %f (ms)\n", sdkGetTimerValue(&timer_compute));
	sdkDeleteTimer(&timer_compute);

}
