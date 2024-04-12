__global__ void simpleKernel(int *input, int *output, int width, int height, int depth) {
    int value; // Example load operation

    // Example loop for multiple loads
    for (int i = 0; i < 10; ++i) {
        printf("%d, %d, %d, %pj", threadIdx.x, threadIdx.y, threadIdx.z, &input[threadIdx.x + threadIdx.y + threadIdx.z]);
        value =input[threadIdx.x + threadIdx.y + threadIdx.z]; // Repeatedly load from the same position
    }

    // Examåçple loop for multiple stores
    for (int i = 0; i < 10; ++i) {
        // printf("%d, %d, %d, %d", threadIdx.x, threadIdx.y, threadIdx.z, &input[threadIdx.x + threadIdx.y + threadIdx.z]);
	printf("%d", threadIdx.x);	
	printf("%d2st", threadIdx.y);	
	printf("%d3st", threadIdx.z);	
	printf("%p4st", &input[threadIdx.x + threadIdx.y + threadIdx.z]);

        input[threadIdx.x + threadIdx.y + threadIdx.z] = value;
    }

}

int main(int argc, char** argv) {
    // Allocate memory on the host

    return 0;
}
