#include <stdint.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#define long int64_t
#define SIZE 2 << 29 / sizeof(int64_t)
#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}
__global__ void process(long* seeds) {
	long global_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_id >= SIZE - 8) {
		return;
	}
	printf("%lld\n", global_id);
}
int main(void) {
	// Allocate RAM for input
	long *input = (long *)malloc(sizeof(long) * SIZE);
	// Copy file to RAM
	std::ifstream ifs ("input.txt");
	if (ifs.fail()) {
		std::cout << "ERROR::IFSTREAM::FAIL" << std::endl;
		return -1;
	}
	std::string line;
	long i = 0;
	while (ifs >> line && i++ < SIZE) {
		input[i] = std::atol(line.c_str());
	}
	ifs.close();
	// Allocate VRAM for input
	long *seeds;
	CHECK_GPU_ERR(cudaMallocManaged((long **)&seeds, sizeof(long) * SIZE));
	// Copy input to VRAM
	CHECK_GPU_ERR(cudaMemcpy(seeds, input, SIZE, cudaMemcpyHostToDevice));
	process<<<SIZE / 256, 256>>>(seeds);
	cudaFree(seeds);
	return 0;
}