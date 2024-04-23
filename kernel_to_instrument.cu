#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
                         	(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
    	assumed = old;
old = atomicCAS(address_as_ull, assumed,
                    	__double_as_longlong(val +
                           	__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void stencil(double* A, double* B, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= n || j >= n) {
    	return;
	}
	// printf idx idy idx
	if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
    	B[i * n + j] = A[i * n + j];
    	return;
	}

	// the median of 5 values is the value in the middle if the values were sorted
	// (i.e. the third value if the values were sorted in ascending order)
	// Find the median of the 5 values around Ao(i,j), Ao(i+1,j), Ao(i�~H~R1,j), Ao(i,j+1), Ao(i,j�~H~R1)
	// create another array with the 5 values and sort it
	double arr[5] = { A[i * n + j], A[(i + 1) * n + j], A[(i - 1) * n + j], A[i * n + j + 1], A[i * n + j - 1] };
	// sort the array
	for (int i = 0; i < 5; i++) {
    	for (int j = i + 1; j < 5; j++) {
        	if (arr[i] > arr[j]) {
            	double temp = arr[i];
            	arr[i] = arr[j];
            	arr[j] = temp;
        	}
    	}
	}
	// the median is the third value
	B[i * n + j] = arr[2];

   // float median = (A[i * n + j] + A[(i + 1) * n + j] + A[(i - 1) * n + j] + A[i * n + j + 1] + A[i * n + j - 1]) / 5.0f;

}
// A kernel to compute the sum of the final values of the array A and compute the other two values
__global__ void verification(double* A, double* sum, double* a, double* b, int n, int verif){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i == n / 3 && j == n / 3) {
    	*a = A[i * n + j];
	}
	if (i == 19 && j == 37) {
    	*b = A[i * n + j];
	}
	if (i < n && j < n){
    	atomicAdd(sum, A[i * n + j]);
	}
}

int sampleKernel(int arg) {
	int n = 10;
	int t = 10;

	double* A = new double[n * n];
	double* B = new double[n * n];
	double* d_A;
	double* d_B;
	double* d_sum;
	double* d_v1;
	double* d_v2;

	// Initializing A
	for (int i = 0; i < n; i++) {
    	for (int j = 0; j < n; j++) {
        	A[i * n + j] = pow(sin((i * i) + j), 2) + cos(i - j);
    	}
	}

	// Copy A to GPU
	cudaMalloc(&d_sum, sizeof(double));
	cudaMalloc(&d_v1, sizeof(double));
	cudaMalloc(&d_v2, sizeof(double));
	cudaMalloc(&d_A, n * n * sizeof(double));

cudaMalloc(&d_A, n * n * sizeof(double));
	cudaMalloc(&d_B, n * n * sizeof(double));
	cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);

	// Initialize CUDA timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// Do t iterations of stencil updates
	dim3 block(32, 32);
	dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

	for (int i = 0; i < t; i++) {
    	stencil<<<grid, block>>>(d_A, d_B, n);
    	double* temp = d_A;
    	d_A = d_B;
    	d_B = temp;
	}

	// Compute verification values on GPU
	double sum = 0;
	double a = 0;
	double b = 0;
	verification<<<grid, block>>>(d_A, d_sum, d_v1, d_v2, n, int(n/3));
	cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&a, d_v1, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&b, d_v2, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(A, d_A, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	// Stop CUDA timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print elapsed time and verification values

	cout << "Elapsed time: " << milliseconds << " ms" << endl;
	cout << "Sum: " << sum << endl;
	cout << "A(n/3, n/3): " << a << endl;
	cout << "A(19, 37): " << b << endl;

	// Free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_sum);
	cudaFree(d_v1);
	cudaFree(d_v2);
	delete[] A;
	delete[] B;


	return 0;
}


