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

#define CHUNK_X 0
#define CHUNK_Y 0
#define TREE_X 2
#define TREE_Z 10
#define TREE_HEIGHT 5

//this.rand.setSeed(integer3 * (this.rand.nextLong() / 2L * 2L + 1L) + integer4 * (this.rand.nextLong() / 2L * 2L + 1L) ^ this.worldObj.randomSeed);
#define RANDOM_MASK (1ULL << 48) - 1
#define setSeed(rand, val) ((rand) = ((val) ^ 0x5DEECE66DLL) & ((1LL << 48) - 1))
__host__ __device__ unsigned int next(long *rand, int bits) {
	*rand = (*rand * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
	return (unsigned int)(*rand >> (48 - bits));
}
__host__ __device__ long nextLong(long *rand) {
	return ((long)next(rand, 32) << 32) + next(rand, 32);
}
__host__ __device__ unsigned int nextIntBound(long *rand, int bound) {
	return (unsigned int)((bound * (long)next(rand, 31)) >> 31);
}
#define advance(rand, multiplier, addend) ((rand) = ((rand) * (multiplier) + (addend)) & (RANDOM_MASK))
#define advance_1(rand) advance(rand, 0x5DEECE66DLL, 0xBLL)
#define advance_16(rand) advance(rand, 0x6DC260740241LL, 0xD0352014D90LL)
#define advance_3760(rand) advance(rand, 0x8C35C76B80C1, 0xD7F102F24F30)

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}
__global__ void process(long* seeds) {
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long seed = seeds[index];
	long rand;
	setSeed(rand, seed);
	setSeed(rand, CHUNK_X * (nextLong(&rand) / 2LL * 2LL + 1LL) + CHUNK_Y * (nextLong(&rand) / 2LL * 2LL + 1LL) ^ seed);
	advance_3760(rand);
	long treeX = nextIntBound(&rand, 16);
	long treeZ = nextIntBound(&rand, 16);
	long treeH = nextIntBound(&rand, 3) + 4;
	if (treeX = TREE_X && treeZ == TREE_Z && treeH == TREE_HEIGHT) printf("%lld\n", seed);
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