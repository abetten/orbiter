/*
 * arithmetic.h
 *
 *  Created on: Nov 20, 2018
 *      Author: sajeeb
 */

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef ARITHMETIC_H_
#define ARITHMETIC_H_


namespace arithmetic {

	__host__ __device__ int modinv(int a, int b);
	__host__ __device__ void xgcd(long *result, long a, long b);
	__host__ __device__ int mod(int a, int p);

}


#endif /* ARITHMETIC_H_ */
