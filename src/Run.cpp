#include "QueryDevice.h"
#include "VectorAdd.h"
#include "MyMatrixAdd.h"
#include "MatrixVectorMult.h"
#include "VectorReduction.h"
#include "Convolution.h"
#include <math.h>

void testConvolution(){


	float *in, *mask, *out;
	int InOut_width = 1024, mask_width = 3;


	in = new float[InOut_width];
	for (int el = 0; el < InOut_width; el++) {
		in[el] = 1;
	}

	mask = new float[mask_width];
	for (int el = 0; el < mask_width; el++) {
		mask[el] = 1;
	}

	out = new float[InOut_width];
	conv1DHost(in,InOut_width,mask,mask_width,out);
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

	//getDeviceInfo();

	//testVecAdd();

	//testMatrixAdd();

	//testMatrixVectorMult();

	//testVetorReduction();

	testConvolution();

}
