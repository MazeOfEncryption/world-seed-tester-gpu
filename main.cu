#include <stdint.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#define SIZE 1239101157
void read(std::string filename, char *out) {
	// Create input filestream
	std::ifstream ifs (filename);
	// Check for errors
	if (ifs.fail()) {
		std::cout << "ERROR::IFSTREAM::FAIL" << std::endl;
		return;
	}
	// Read SIZE bytes of file into out
	ifs.read(out, SIZE);
	// Close stream
	ifs.close();
}
#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}
__global__ void process(char* seeds) {
	long global_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_id >= SIZE - 8) {
		return;
	}
	// Proof of concept: parse file looking for "10101010"
	if (
		seeds[global_id + 0] == '1' &&
		seeds[global_id + 1] == '0' &&
		seeds[global_id + 2] == '1' &&
		seeds[global_id + 3] == '0' &&
		seeds[global_id + 4] == '1' &&
		seeds[global_id + 5] == '0' &&
		seeds[global_id + 6] == '1' &&
		seeds[global_id + 7] == '0'
	) printf("Found @%ld.\n", global_id);
}
int main(void) {
	// Allocate RAM for input
	char* input = (char *)malloc(SIZE);
	// Copy file to RAM
	read("input.txt", input);
	// Allocate VRAM for input
	char* seeds;
	CHECK_GPU_ERR(cudaMallocManaged((char **)&seeds, SIZE));
	// Copy input to VRAM
	CHECK_GPU_ERR(cudaMemcpy(seeds, input, SIZE, cudaMemcpyHostToDevice));
	process<<<SIZE / 256, 256>>>(seeds);
	cudaFree(seeds);
	return 0;
}