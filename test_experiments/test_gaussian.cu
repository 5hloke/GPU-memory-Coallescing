#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N 4 // Size of the matrix

__global__ void gaussianElimination(float *A, float *B) {
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int bidx = blockIdx.x;
    int tidy = threadIdx.y;

    __shared__ float shared_A[N][N+1];
    __shared__ float shared_B[N];

    // Load data into shared memory
    shared_A[idy][idx] = A[bidx * N * N + idy * N + idx];
    shared_B[idy] = B[bidx * N + idy];
    __syncthreads();

    // Gaussian elimination
    for (int k = 0; k < N; k++) {
        if (idy > k) {
            float ratio = shared_A[idy][k] / shared_A[k][k];
            for (int j = k; j < N + 1; j++) {
                shared_A[idy][j] -= ratio * shared_A[k][j];
            }
        }
        __syncthreads();
    }

    // Back substitution
    if (idy == N - 1) {
        shared_B[N-1] = shared_A[N-1][N] / shared_A[N-1][N-1];
        for (int i = N - 2; i >= 0; i--) {
            float sum = 0.0;
            for (int j = i + 1; j < N; j++) {
                sum += shared_A[i][j] * shared_B[j];
            }
            shared_B[i] = (shared_A[i][N] - sum) / shared_A[i][i];
        }
    }
    __syncthreads();

    // Write back the result
    B[bidx * N + tidy] = shared_B[tidy];
}

void create_system(float* matrix, float *b,  int n){

    int i,j;
    float lamda = -0.01;
    float coe[2*n-1];
    float coe_i =0.0;

    for (i=0; i<n; i++){
        coe_i = 10*exp(lamda*i);
        j=n-1+i;     
        coe[j]=coe_i;
        j=n-1-i;     
        coe[j]=coe_i;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n+j]=coe[n-1-i+j];
        }
    }
    for (int i = 0; i < n; i++) {
        b[i] = 0;
    }
}


void gaussianKernel(){

    float *h_A, *h_B; // Host matrices
    float *d_A, *d_B; // Device matrices

    // Allocate memory on the host
    h_A = (float *)malloc(N * N * sizeof(float));
    h_B = (float *)malloc(N * sizeof(float));

    create_system(h_A, h_B, N);

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(N, N);
    dim3 grid(1, 1);

    // Launch kernel
    gaussianElimination<<<grid, block>>>(d_A, d_B);

    // Copy data from device to host
    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
    free(h_A);
    free(h_B);
} 
