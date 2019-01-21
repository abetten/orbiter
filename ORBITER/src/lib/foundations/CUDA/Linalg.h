#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

#include "arithmetic.h"
#include "gpuErrchk.h"
#include "Matrix.h"

using arithmetic::mod;

#ifndef LINALG_
#define LINALG_

namespace linalg {

	template <typename Mat>
	__host__
	void transpose(Mat& SRC, Mat& DST) {
		// this function assumes that the destination matrix, DST
		// has the proper dimensions, i.e. if src matrix has dimensions
		// m by n, then the dst matrix should have dimensions n by m,
		// and that the dst matrix and the src matrix class is initialized

		for (size_t i=0; i<SRC.nrows; ++i) {
			for (size_t j=0; j<SRC.ncols; ++j) {
				DST(j, i) = SRC(i, j);
			}
		}
	}

	template <typename Mat>
	__host__
	void transpose(Mat& M) {
		// This function takes in a matrix reference M, and returns
		// the transpose of M

		for (size_t i=0; i<M.nrows; ++i) {
			for (size_t j=i+1; j<M.ncols; ++j) {
			#ifdef __CUDA_ARCH__
				printf("%s:%d:", __FILE__, __LINE__);
				printf("transpose function not implemented for CUDA\n");
			#else
				std::swap(M(i,j), M(j,i));
			#endif
			}
		}

	}

	// CUDA kernels

	template <typename Mat>
	__global__
	void cuda_matrix_matrix_dot_(Mat& A, Mat& B, Mat& C, int p=0, int axis=0) {

		// axis = 0
		//		Do val += A(row, e) * B(e, col);
		//		A * B
		//
		// axis = 1
		//		Do val += A(e, row) * B(e, col);
		//		A.T * B
		//
		// axis = 2
		// 		Do val += A(row, e) * B(col, e);
		//		A * B.T
		//
		// axis = 3
		//		Do val += A(e, row) * B(col, e);
		//		A.T * B.T



		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= C.nrows || col >= C.ncols) return;

		auto val = A(row, 0);
		val = 0;

		if (axis == 0) { // A * B
			for (size_t e = 0; e < A.ncols; ++e) {
				if (p == 0) {
					val += A(row, e) * B(e, col);
				} else {
					val += mod(A(row, e) * B(e, col), p);
				}
			}
		}

		if (axis == 1) { // A.T * B
			for (size_t e = 0; e < A.nrows; ++e) {
				if (p == 0) {
					val += A(e, row) * B(e, col);
				} else {
					val += mod(A(e, row) * B(e, col), p);
				}
			}
		}

		if (axis == 2) { // A * B.T
			for (size_t e = 0; e < A.ncols; ++e) {
				if (p == 0) {
					val += A(row, e) * B(col, e);
				} else {
					val += mod(A(row, e) * B(col, e), p);
				}
			}
		}

		if (axis == 3) { // A.T * B.T
			for (size_t e = 0; e < A.nrows; ++e) {
				if (p == 0) {
					val += A(e, row) * B(col, e);
				} else {
					val += mod(A(e, row) * B(col, e), p);
				}
			}
		}

		if (p == 0) C(row, col) = val;
		else C(row, col) = mod(val, p);

	}

	template <typename Mat>
	__global__
	void cuda_strassen_matrix_matrix_multiply_(Mat& A, Mat& B, Mat& C) {

	}

	template <typename Mat>
	__global__
	void cuda_matrix_matrix_subtract_(Mat& A, Mat& B, Mat& C, int p=0) {

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= A.nrows || col >= A.ncols) return;

		if (p == 0) C(row, col) = A(row, col) - B(row, col);
		else C(row, col) = mod(A(row, col) - B(row, col), p);

	}

	template <typename Mat>
	__global__
	void cuda_matrix_matrix_add_(Mat& A, Mat& B, Mat& C, int p=0) {

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= A.nrows || col >= A.ncols) return;

		if (p == 0) C(row, col) = A(row, col) + B(row, col);
		else C(row, col) = mod(A(row, col) + B(row, col), p);

	}

	template <typename Mat>
	__global__
	void cuda_matrix_matrix_element_multiply_(Mat& A, Mat& B, Mat& C,
												int p=0) {

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= A.nrows || col >= A.ncols) return;

		if (p == 0) C(row, col) = A(row, col) * B(row, col);
		else C(row, col) = mod(A(row, col) * B(row, col), p);

	}

	template <typename Mat>
	__global__
	void cuda_matrix_element_apply_function_(Mat& A, Mat& B,
												double (*func)(double),
												int p=0) {

		// Apply function to elements of A, and store the result in B

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= A.nrows || col >= A.ncols) return;

		B(row, col) = (*func)(A(row, col));

	}

	template <typename Mat, typename scalar>
	__global__
	void cuda_matrix_scale_(scalar a, Mat& src, Mat& dst) {
		// Given a scalar a, multiply the elements of the
		// matrix A by the scalar.

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= src.nrows || col >= src.ncols) return;

		dst(row, col) = a * src(row, col);
	}

	template <typename Mat>
	__global__
	void cuda_norm_(Mat& src, Mat& dst, int p=0) {

		// Apply function to elements of A, and store the result in B

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= src.nrows || col >= src.ncols) return;

		dst(row, col) = pow(src(row, col), double(p));

	}

	// Functions that call CUDA Kernels

	template <typename Mat>
	__device__ __host__ void
	device_dot(Mat& A, Mat& B, Mat& C, int p=0, int axis=0) {
		size_t m = A.nrows;
		size_t n = A.ncols;

		size_t o = B.nrows;
		size_t k = B.ncols;

		size_t e = C.nrows;
		size_t f = C.ncols;

		if (o != n && m != e && k != f && axis == 0) {
			printf("%s:%d:Cannot perform matrix multiplication, ", __FILE__, __LINE__);
			printf("size of matrix column do not match size of vector.\n");
//			exit(-1);
		}

		if (m != o && e != n && f != k && axis == 1) {
			printf("%s:%d:Cannot perform matrix multiplication, ", __FILE__, __LINE__);
			printf("size of matrix column do not match size of vector.\n");
//			exit(-1);
		}

		if (n != k && e != m && f != o && axis == 2) {
			printf("%s:%d:Cannot perform matrix multiplication, ", __FILE__, __LINE__);
			printf("size of matrix column do not match size of vector.\n");
//			exit(-1);
		}

		if (m != k && e != k && f != m && axis == 3) {
			printf("%s:%d:Cannot perform matrix multiplication, ", __FILE__, __LINE__);
			printf("size of matrix column do not match size of vector.\n");
//			exit(-1);
		}

		if (&C == &A || &C == &B) {
			printf("%s:%d:Result of dot(...) cannot be stored in ", __FILE__, __LINE__);
			printf("output matrix.\n");
//			exit(-1);
		}

		int num_threads = C.nrows * C.ncols;
		int block_size = 16;
		int num_blocks = (num_threads + block_size*block_size - 1)/ (block_size*block_size) ;
		int gridDim_x = (C.ncols + block_size - 1) / block_size;
		int gridDim_y = (C.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		if (num_blocks > gridDim_x*gridDim_y || num_threads > gridDim_x*gridDim_y*pow(block_size,2)) {
			printf("Error:%s:%d:number of required blocks is greater than number of blocks set.",__FILE__,__LINE__);
		}

		cuda_matrix_matrix_dot_<<<gridDim, blockDim>>>(A, B, C, p, axis);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat>
	__device__ void
	device_matrix_matrix_substract(Mat& A, Mat& B, Mat& C, int p=0) {

		if (A.nrows != B.nrows || A.ncols != B.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		if (B.nrows != C.nrows || B.ncols != C.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		int num_threads = A.nrows * A.ncols;
		int block_size = 16;
		int num_blocks = (num_threads + block_size*block_size - 1)/ (block_size*block_size) ;
		int gridDim_x = (A.ncols + block_size - 1) / block_size;
		int gridDim_y = (A.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		if (num_blocks > gridDim_x*gridDim_y || num_threads > gridDim_x*gridDim_y*pow(block_size,2)) {
			printf("Error:%s:%d:number of required blocks is greater than number of blocks set.",__FILE__,__LINE__);
		}

		cuda_matrix_matrix_subtract_<<<gridDim, blockDim>>>(A, B, C, p);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat>
	__device__ void
	device_matrix_matrix_add(Mat& A, Mat& B, Mat& C, int p=0) {

		if (A.nrows != B.nrows || A.ncols != B.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		if (B.nrows != C.nrows || B.ncols != C.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		int num_threads = A.nrows * A.ncols;
		int block_size = 16;
		int num_blocks = (num_threads + block_size*block_size - 1)/ (block_size*block_size) ;
		int gridDim_x = (A.ncols + block_size - 1) / block_size;
		int gridDim_y = (A.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		if (num_blocks > gridDim_x*gridDim_y || num_threads > gridDim_x*gridDim_y*pow(block_size,2)) {
			printf("Error:%s:%d:number of required blocks is greater than number of blocks set.",__FILE__,__LINE__);
		}

		cuda_matrix_matrix_add_<<<gridDim, blockDim>>>(A, B, C, p);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat>
	__device__ void
	device_matrix_matrix_element_wise_multiply(Mat& A, Mat& B, Mat& C, int p=0) {

		if (A.nrows != B.nrows || A.ncols != B.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf(":A.nrows= %d", A.nrows);
			printf(":B.nrows= %d", B.nrows);
			printf(":A.ncols= %d", A.ncols);
			printf(":B.ncols= %d", B.ncols);
			printf("\n");
		}

		if (B.nrows != C.nrows || B.ncols != C.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		int num_threads = A.nrows * A.ncols;
		int block_size = 16;
		int num_blocks = (num_threads + block_size*block_size - 1)/ (block_size*block_size) ;
		int gridDim_x = (A.ncols + block_size - 1) / block_size;
		int gridDim_y = (A.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		if (num_blocks > gridDim_x*gridDim_y || num_threads > gridDim_x*gridDim_y*pow(block_size,2)) {
			printf("Error:%s:%d:number of required blocks is greater than number of blocks set.",__FILE__,__LINE__);
		}

		cuda_matrix_matrix_element_multiply_<<<gridDim, blockDim>>>(A, B, C, p);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat>
	__device__ void
	device_element_wise_apply_function( Mat& src, Mat& dst,
										double (*func)(double),
										int p=0 ) {
		// Apply function to elements of A, and store
		// the result in B.

		if (src.nrows != dst.nrows || src.ncols != dst.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		int block_size = 16;
		int gridDim_x = (src.ncols + block_size - 1) / block_size;
		int gridDim_y = (src.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		cuda_matrix_element_apply_function_<<<gridDim, blockDim>>>(src, dst, func, p);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat>
	__device__ void
	device_element_wise_apply_function( Mat& src,
										double (*func)(double),
										int p=0 ) {

		// Apply function to elements of src, and store
		// the result in src.

		int block_size = 16;
		int gridDim_x = (src.ncols + block_size - 1) / block_size;
		int gridDim_y = (src.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		cuda_matrix_element_apply_function_<<<gridDim, blockDim>>>(src, src, func, p);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat, typename scalar>
	__device__ void
	device_matrix_scale(scalar a, Mat& src) {
		int block_size = 16;
		int gridDim_x = (src.ncols + block_size - 1) / block_size;
		int gridDim_y = (src.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		cuda_matrix_scale_<<<gridDim, blockDim>>>(a, src, src);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename Mat, typename scalar>
	__device__ void
	device_matrix_scale(scalar a, Mat& src, Mat& dst) {

		if (src.nrows != dst.nrows) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same rows.");
			printf("\n");
		}

		if (src.ncols != dst.ncols ) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same cols.");
			printf("\n");
		}


		int block_size = 16;
		int gridDim_x = (src.ncols + block_size - 1) / block_size;
		int gridDim_y = (src.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		cuda_matrix_scale_<<<gridDim, blockDim>>>(a, src, dst);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();
	}

	template <typename obj>
	__device__
	double device_norm (obj& src, obj& dst, int p) {

		int block_size = 16;
		int gridDim_x = (src.ncols + block_size - 1) / block_size;
		int gridDim_y = (src.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		if (src.nrows != dst.nrows || src.ncols != dst.ncols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("Input matrices must have the same dimension.");
			printf("\n");
		}

		cuda_norm_<<<gridDim, blockDim>>>(src, dst, p);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();

		double sum = 0;
		for (size_t i=0; i < dst.nrows; ++i) {
			for (size_t j=0; j < dst.ncols; ++j) {
				sum += dst(i, j);
			}
		}

		return pow(sum, double(1)/double(p));

	}

	template <typename obj>
	__device__
	double strassen_matrix_multiply (obj& src, obj& dst) {

		int block_size = 16;
		int gridDim_x = (src.ncols + block_size - 1) / block_size;
		int gridDim_y = (src.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);



	}

	template <typename T>
	void matrix_matrix_subtract(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int p=0) {
		if (A.nrows != B.nrows || A.ncols != B.ncols) {
			printf("%s:%d:Input matrices must have the same dimension.\n", __FILE__, __LINE__);
		}
		if (B.nrows != C.nrows || B.ncols != C.ncols) {
			printf("%s:%d:Input matrices must have the same dimension.\n", __FILE__, __LINE__);
		}

		for (size_t i=0; i<A.nrows; ++i) {
			for (size_t j=0; j<A.ncols; ++j) {
				C(i,j) = A(i,j) - B(i,j);
//				if (p != 0) C(i,j) = mod(int(C(i,j)), p);
			}
		}
	}

	template <typename T>
	void matrix_matrix_add(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int p=0) {
		if (A.nrows != B.nrows || A.ncols != B.ncols) {
			printf("%s:%d:Input matrices must have the same dimension.\n", __FILE__, __LINE__);
		}
		if (B.nrows != C.nrows || B.ncols != C.ncols) {
			printf("%s:%d:Input matrices must have the same dimension.\n", __FILE__, __LINE__);
		}

		for (size_t i=0; i<A.nrows; ++i) {
			for (size_t j=0; j<A.ncols; ++j) {
				C(i,j) = A(i,j) + B(i,j);
//				if (p != 0) C(i,j) = mod(int(C(i,j)), p);
			}
		}
	}

	template <typename T>
	void dot(Matrix<T>& A, Matrix<T>& B, Matrix<T>& M, int p=0) {
		size_t m = A.get_nrows();
		size_t n = A.get_ncols();
		size_t k = B.get_nrows();
		size_t l = B.get_ncols();

		if (n != k) {
			cout << __FILE__ << ":" << __LINE__ << ":";
			cout << "cannot multiply matrices, columns of A "
				 <<	"does not match rows of B." << endl;
			exit(-1);
		}

		Matrix<T> tmp_M(m, l);

		// Implement algorithm for matrix multiplication
		for (size_t i=0; i<m ; ++i) { // iterate over the rows of A
			for (size_t j=0; j<l; ++j) { // iterate over the cols of B
				for (size_t o=0; o<n; ++o) { // iterate over cols of A
					if (p == 0) tmp_M(i, j) += A(i, o) * B(o, j);
//					else tmp_M(i, j) += mod(int(A(i, o) * B(o, j)), p);
				}
//				if (p != 0) tmp_M(i, j) = mod(int(tmp_M(i, j)), p);
			}
		}

		M.Init(tmp_M);
	}

}

#endif
