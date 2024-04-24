#include <cuda_runtime.h>


__global__ void simpleKernel(int *input, int *output, int width, int height, int depth) {
    int value; // Example load operation

    // Example loop for multiple loads
    for (int i = 0; i < 10; ++i) {
        // printf("%d, %d, %d, %d", threadIdx.x, threadIdx.y, threadIdx.z, &input[threadIdx.x + threadIdx.y + threadIdx.z]);
        value =input[threadIdx.x + threadIdx.y + threadIdx.z]; // Repeatedly load from the same position
    }

    // Example loop for multiple stores
    for (int i = 0; i < 10; ++i) {
        // printf("%d, %d, %d, %d", threadIdx.x, threadIdx.y, threadIdx.z, &input[threadIdx.x + threadIdx.y + threadIdx.z]);
        input[threadIdx.x + threadIdx.y + threadIdx.z] = value;
    }

}

int main(int argc, char** argv) {
    // set input to be a 10 x 10 x 10 array with value 1-1000
    int input[10][10][10];
    int output[10][10][10];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                input[i][j][k] = i + j + k;
            }
        }
    }

    // copy input to device
    int *d_input, *d_output;
    cudaMalloc(&d_input, 10 * 10 * 10 * sizeof(int));
    cudaMalloc(&d_output, 10 * 10 * 10 * sizeof(int));
    cudaMemcpy(d_input, input, 10 * 10 * 10 * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block(10, 10, 10);
    simpleKernel<<<1, block>>>(d_input, d_output, 10, 10, 10);

    // copy output back to host
    cudaMemcpy(output, d_output, 10 * 10 * 10 * sizeof(int), cudaMemcpyDeviceToHost);



    return 0;
}