# MyCudaSamples
Collection of programming exercises based on the book [Programming Massively Parallel Processors: A Hands-on Approach by David Kirk and Wen-mei W. Hwu](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0). In this self-study project, we implement different mathematical operations using C++ and pure CUDA framework.

## Requirements:
- C++
- Cuda Toolkit

## Algorithms:
- [Vector Addition](./src/VectorAdd.cu), [Matrix Vector Multiplication](./src/MatrixVectorMult.cu) and [Matrix Multiplication](./src/MatMatMult.cu): Standard linear algebra operations on CPU and GPU. Check how the GPU implementation becomes faster than the CPU code when we increase the size of the operands.
- [Matrix Addition](./src/MyMatrixAdd.cu): Standard Matrix addition in CPU and GPU. Check how to launch kernels with different configurations and its implications. 
- [Convolution](./src/convolution.cu): Implementation of 1D and 2D convolution in CPU and GPU. We also show how tiling can speed-up the GPU implementation by leveraging data locality.
- [Histogram](./src/histogram.cu): Code to compute histograms using CPU an GPU. We show how to use the shared memory to speed-up the GPU implementation by reducing the global memory access.
- [Vector Reduction](./src/VectorReduction.cu): GPU code to peform vector reduction operations. Here, we see how to synchronize threads in GPU.
- [Prefix Sum](./src/prefixSum.cu): This is the implementation of the cumulative sum which is trivial in CPU code but challenging in GPU. We provide two ways to perform such an algorithm.
- [Run](./src/Run.cpp): Use this file to run the samples described above.

Note that we **do not use** any CUDA library like [cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cuDNN) or [Thrust](https://developer.nvidia.com/thrust) in our project, since we focus on learning how the GPU hardware and programming model works. However, we acknowledge that these libraries are more efficient and very handy for large-scale applications.  
