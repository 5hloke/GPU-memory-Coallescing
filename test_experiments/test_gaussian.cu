#include <iostream>
#include <math.h>
#include <cuda_runtime.h>


// here create a matrix to apply the gaussian reduction on
__global__ void kernel1 (float *m_cuda, float *a_cuda, int Size, int t){

    if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);

}

__global__ void kernel2 (float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t){
    if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
	
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	if(yidx == 0){
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}

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
            matrix[i * n + j] = pow(sin((i * i) + j), 2) + cos(i - j);
        }
    }
    for (int i = 0; i < n; i++) {
        b[i] = 0;
        for (int j = 0; j < n; j++) {
            b[i] = 1.0;
        }
    }
}


int sampleKernel(){

    int n = 10;

    float* matrix = new float[n * n];
    float* b = new float[n];
    float* m = new float[n * n];

    create_system(matrix, b, n);
    
    //initialize the multiplication matrix
    for (int i = 0; i < n*n; i++){
        m[i] = 0.0;
    }

    // Now CUDA setup
    float* d_matrix;
    float* d_b;
    float* d_m;

    cudaMalloc(&d_matrix, n * n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_m, n * n * sizeof(float));
    cudaMemcpy(d_matrix, matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Block and grid setup 
    int block_size = 64;
    int grid_size = (n + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid(grid_size);

    int d2_block = 4;
    int d2_grid = (n + d2_block - 1) / d2_block;

    dim3 block2(d2_block, d2_block);
    dim3 grid2(d2_grid, d2_grid);

    // Now launch kernels
    for (int t=0; t < (n-1); t++) {

		kernel1<<<grid, block>>>(d_m , d_matrix, n, t);
		cudaThreadSynchronize();
		kernel2<<<grid2,block2>>>(d_m ,d_matrix ,d_b ,n, n-t,t);
		cudaThreadSynchronize();

	}

    // Copy back the results
    cudaMemcpy(m, d_m, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrix, d_matrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_b);
    cudaFree(d_m);

    // Now back propagate the results
    float* x = new float[n];
    for (int i = 0; i < n; i++){
        x[n-i-1]=b[n-i-1];
        for (int j = 0; j < i; j++){
            x[n-i-1] -=* (matrix+n*(n-i-1)+(n-j-1)) * x[n-j-1];
		}
		x[n-i-1]=x[n-i-1]/ *(matrix+n*(n-i-1)+(n-i-1));

    }

    delete[] matrix;
    delete[] b;
    delete[] m;


    

    return 0;
}
