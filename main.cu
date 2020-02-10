#include <stdint.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#define long int64_t
#define WORK_UNIT_SIZE (2 << 20)
#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}
__global__ void process(long* seeds) {
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	
}
int main(void) {
	// Allocate RAM for input
	long *input = (long *)malloc(sizeof(long) * WORK_UNIT_SIZE);
	// Oprn File
	std::ifstream ifs ("input.txt");
	if (ifs.fail()) {
		std::cout << "ERROR::IFSTREAM::FAIL" << std::endl;
		return -1;
	}
	// Allocate VRAM for input
	long *seeds;
	CHECK_GPU_ERR(cudaMallocManaged((long **)&seeds, sizeof(long) * WORK_UNIT_SIZE));
	// Open Inupt Block in RAM, copy to VRAM, and process.

    clock_t lastIteration = clock();
    clock_t startTime = clock();

	std::string line;
	while(std::getline(ifs, line)) {
		// Load Input Block
		// TODO: Fix issue where the last iteration will recheck seeds from the previous
		// iteration if the number of inputs is not evenly divisible by WORK_UNIT_SIZE
		for (long i = 0; ifs >> line && i < WORK_UNIT_SIZE; i++) {
			long val = std::atoll(line.c_str());
			input[i] = val;
		}
		// Copy input to VRAM
		CHECK_GPU_ERR(cudaMemcpy(seeds, input, sizeof(long) * WORK_UNIT_SIZE, cudaMemcpyHostToDevice));
		// Process input
		process<<<WORK_UNIT_SIZE / 256, 256>>>(seeds);

		double iterationTime = (double)(clock() - lastIteration) / CLOCKS_PER_SEC;
        double timeElapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
		lastIteration = clock();
		double speed = (double)WORK_UNIT_SIZE / (double)iterationTime / 1000000.0;
		printf("Uptime: %.1fs. Speed: %.2fm/s.\n", timeElapsed, speed);
	}
	cudaFree(seeds);
	ifs.close();
	return 0;
}