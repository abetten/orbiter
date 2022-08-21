/*
 * gpu_table.h
 *
 *  Created on: Jul 22, 2019
 *      Author: sajeeb
 */

#include <cuda.h>
//#include <helper_cuda_drvapi.h>
#include "drvapi_error_string.h"

#ifndef GPU_TABLE_H_
#define GPU_TABLE_H_


namespace gpu_table {

size_t nb_devices () {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount;
}


size_t free_memory () {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free;
}

size_t total_memory () {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return total;
}


}




#endif /* GPU_TABLE_H_ */
