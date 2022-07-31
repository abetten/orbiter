#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef gpuErrchk_
#define gpuErrchk_

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__host__ void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(-1);
	}
}

#endif
