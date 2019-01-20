#include "gpuErrchk.h"


__host__ void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}
