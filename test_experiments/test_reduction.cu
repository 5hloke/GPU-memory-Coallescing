#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;
#define N 17
__global__ void kernel(float *A, float *x, float *tmp){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // // add shared memory array here
    __shared__ float shared[10*10];
    // __shared__ float shared[10];

    // shared[i] = 0.0f;
    if (i<N)
    {
        // shared[i] = 0.0f;
        __syncthreads();
        for(int j=0; j < N; j++){
            tmp[i] += A[i *N + j] * x[j];
        }
        __syncthreads();

    }
}

void reductionKernel(){
    float *A, *x, *tmp;
    float *d_A, *d_x, *d_tmp;
    int size = N * N * sizeof(float);
    A = (float*)malloc(size);
    x = (float*)malloc(N * sizeof(float));
    tmp = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        tmp[i] = 0.0f;
        for(int j = 0; j < N; j++){
            A[i * N + j] = 1.0f;
        }
    }
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_tmp, N * sizeof(float));
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp, tmp, N * sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    int numBlocks = (N + blockSize - 1) / blockSize;
    dim3 dimGrid(numBlocks, numBlocks);
    kernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_tmp);
    cudaMemcpy(tmp, d_tmp, N * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for(int i = 0; i < N; i++){
        sum += tmp[i];
    }
    cout << "Sum: " << sum << endl;
    free(A);
    free(x);
    free(tmp);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_tmp);
}