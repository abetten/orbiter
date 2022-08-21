#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <iomanip>
#include <random>
#include <iostream>
#include <thread>

#include "gpuErrchk.h"
#include "gpu_table.h"

#ifndef _LINALG_
#define _LINALG_

namespace linalg {


#define _cuda_matrix_multiplication_under_mod_p_(X, Y, Z) {			\
		if (p == 0) 												\
			for (size_t e=0; e<X; ++e) val += Y * Z; 				\
		else 														\
			for (size_t e=0; e<X; ++e) val = (val + Y * Z) % p;		\
	}

#define _cpu_matrix_multiply_mod_p_(X, Y, Z, tid, stride) {			\
		if (p == 0) {												\
			for (size_t i=tid; i<C.nrows; i+=stride) {				\
				for (size_t e=0; e<X; ++e) {						\
					for (size_t j=0; j<C.ncols; j+=1) {				\
						C.matrix_[i*C.alloc_cols+j] += Y * Z;		\
					}												\
				}													\
			}														\
		} else {													\
			for (size_t i=0; i<C.nrows; ++i) {								\
				for (size_t e=0; e<X; ++e) {								\
					for (size_t j=tid; j<C.ncols; j+=stride) {		\
						C(i, j) += (Y * Z) % p;						\
					}												\
				}													\
			}														\
		}															\
	}


#include "Matrix.h"


template <typename T>
__host__ __device__
inline void identity(Matrix<T>& M) {
	// Turn the current matrix into an identity matrix
	for (size_t i=0, j=0; i<M.nrows; ++i, j=0) {

		for (; j<i; ++j)
			M.matrix_[i*M.alloc_cols + j] = 0;

		M.matrix_[i*M.alloc_cols + j] = 1;
		++j;

		for (; j<M.ncols; ++j)
			M.matrix_[i*M.alloc_cols + j] = 0;

	}
}


template <typename T>
__host__
inline void print(const Matrix<T>& M) {
	size_t length = 0;
	for (size_t i=0; i<M.nrows*M.ncols; ++i) {
		size_t t = (std::to_string(M.matrix_[i])).length();
		if (t > length) length = t;
	}
	length += 2;

	for (size_t i=0; i<M.nrows; ++i) {
		for (size_t j=0; j<M.ncols; ++j) {
			std::cout << std::setw(length) << M.matrix_[i * M.alloc_cols + j];
		}
		std::cout << std::endl;
	}
}

template <typename T>
__host__ __device__
size_t size_of(Matrix<T>& M) {
	return (sizeof(M) + sizeof(T) * M.alloc_rows * M.alloc_cols);
}



template <typename T>
__host__
void random_fill (linalg::Matrix<T>& M, T min = 0, T max = 1) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uni(min,max);
    for (size_t i=0; i<M.nrows * M.ncols; ++i) {
        M.matrix_[i] = (T)uni(generator);
    }
}


template <typename T>
void matrix_multiply_thread (linalg::Matrix<T>& A, linalg::Matrix<T>& B, linalg::Matrix<T>& C,
															const size_t tid, int p=0, int axis=0) {
	size_t stride = std::thread::hardware_concurrency();

	switch (axis) {
	case 0:
		_cpu_matrix_multiply_mod_p_(A.ncols, A.matrix_[i*A.alloc_cols+e],
											 B.matrix_[e*B.alloc_cols+j], tid, stride);
		break;

	case 1:
		_cpu_matrix_multiply_mod_p_(A.nrows, A.matrix_[e*A.alloc_cols+i],
											 B.matrix_[e*B.alloc_cols+j], tid, stride)
		break;

	case 2:
		_cpu_matrix_multiply_mod_p_(A.ncols, A.matrix_[i*A.alloc_cols+e],
											 B.matrix_[j*B.alloc_cols+e], tid, stride);
		break;

	case 3:
		_cpu_matrix_multiply_mod_p_(A.nrows, A.matrix_[e*A.alloc_cols+i],
											 B.matrix_[j*B.alloc_cols+e], tid, stride);
		break;
	}

}

template <typename T>
__host__
void matrix_tile_cpy (T* dst, size_t m, size_t n, const Matrix<T>& src, size_t i_, size_t j_) {
	size_t h = m;
	size_t w = n;

	if (i_+h > src.nrows) {
		h -= i_+h - src.nrows;
	}
	if (j_+w > src.ncols) {
		w -= j_+w - src.ncols;
	}

	#pragma unroll
	for (size_t ii=0, i=i_; ii<h; ++ii, ++i) {
		memcpy(dst + ii * n,
						src.matrix_ + i * src.alloc_cols + j_,
						w * sizeof(T));
	}
}


template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mat_mul_AB (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t e=0; e<A.ncols; ++e) {
			#pragma unroll
			for (size_t j=0; j<C.ncols; ++j) {
				C.matrix_[i*C.alloc_cols+j] += A.matrix_[i*A.alloc_cols+e] *
												B.matrix_[e*B.alloc_cols+j];
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mod_mat_mul_AB (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int p) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t e=0; e<A.ncols; ++e) {
			#pragma unroll
			for (size_t j=0; j<C.ncols; ++j) {
				C.matrix_[i*C.alloc_cols+j] = ( C.matrix_[i*C.alloc_cols+j] +
												A.matrix_[i*A.alloc_cols+e] *
												B.matrix_[e*B.alloc_cols+j] ) % p;
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mat_mul_ATB (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t e=0; e<A.nrows; ++e) {
			#pragma unroll
			for (size_t j=0; j<C.ncols; ++j) {
				C.matrix_[i*C.alloc_cols+j] += A.matrix_[e*A.alloc_cols+i] *
												B.matrix_[e*B.alloc_cols+j];
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mod_mat_mul_ATB (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int p) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t e=0; e<A.nrows; ++e) {
			#pragma unroll
			for (size_t j=0; j<C.ncols; ++j) {
				C.matrix_[i*C.alloc_cols+j] = ( C.matrix_[i*C.alloc_cols+j] +
				                                A.matrix_[e*A.alloc_cols+i] *
												B.matrix_[e*B.alloc_cols+j] ) % p;
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mat_mul_ABT (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t j=0; j<C.ncols; ++j) {
			#pragma unroll
			for (size_t e=0; e<A.ncols; ++e) {
				C.matrix_[i*C.alloc_cols+j] += A.matrix_[i*A.alloc_cols+e] *
												B.matrix_[j*B.alloc_cols+e];
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mod_mat_mul_ABT (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int p) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t j=0; j<C.ncols; ++j) {
			#pragma unroll
			for (size_t e=0; e<A.ncols; ++e) {
				C.matrix_[i*C.alloc_cols+j] = ( C.matrix_[i*C.alloc_cols+j] +
												A.matrix_[i*A.alloc_cols+e] *
												B.matrix_[j*B.alloc_cols+e] ) % p;
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mat_mul_ATBT (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t j=0; j<C.ncols; ++j) {
			#pragma unroll
			for (size_t e=0; e<A.nrows; ++e) {
				C.matrix_[i*C.alloc_cols+j] += A.matrix_[e*A.alloc_cols+i] *
												B.matrix_[j*B.alloc_cols+e];
			}
		}
	}
}

template <typename T>
__attribute__((always_inline)) __inline__
void cpu_mod_mat_mul_ATBT (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int p) {
	#pragma unroll
	for (size_t i=0; i<C.nrows; ++i) {
		#pragma unroll
		for (size_t e=0; e<A.nrows; ++e) {
			#pragma unroll
			for (size_t j=0; j<C.ncols; ++j) {
				C.matrix_[i*C.alloc_cols+j] = ( C.matrix_[i*C.alloc_cols+j] +
												A.matrix_[e*A.alloc_cols+i] *
												B.matrix_[j*B.alloc_cols+e] ) % p;
			}
		}
	}
}


template <typename T>
void cpu_mat_mul (Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int p=0, int axis=0, bool par=true) {

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


	if (par) {

		size_t maxThreads = std::thread::hardware_concurrency();
		std::thread threads [maxThreads];

		// Initialize the threads
		for (size_t i=0; i<maxThreads; ++i) {
			threads[i] = std::thread(matrix_multiply_thread<T>, std::ref(A),
											std::ref(B), std::ref(C), i, p, axis);
		}

		// join the threads
		for (size_t i=0; i<maxThreads; ++i) threads[i].join();

	} else {

		switch (axis) {

		case 0: {
			(p == 0) ? cpu_mat_mul_AB(A, B, C) : cpu_mod_mat_mul_AB(A, B, C, p);
			break;
		}

		case 1: {
			(p == 0) ? cpu_mat_mul_ATB(A, B, C) : cpu_mod_mat_mul_ATB(A, B, C, p);
			break;
		}

		case 2: {
			(p == 0) ? cpu_mat_mul_ABT(A, B, C) : cpu_mod_mat_mul_ABT(A, B, C, p);
			break;
		}

		case 3: {
			(p == 0) ? cpu_mat_mul_ATBT(A, B, C) : cpu_mod_mat_mul_ATBT(A, B, C, p);
			break;
		}

		}

	}
}

template <typename T>
void mat_mul_block (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	const size_t s = 2048 / sizeof(T);

	size_t imin=std::min(s, C.nrows);
	size_t kmin=std::min(s, A.ncols);
	size_t jmin=std::min(s, C.ncols);

	const size_t m = C.nrows;
	const size_t n = C.ncols;
	const size_t p = A.ncols;

	const size_t c = C.alloc_cols;
	const size_t a = A.alloc_cols;
	const size_t b = B.alloc_cols;

	#pragma unroll
	for (size_t ii=0; ii<m; ii+=s) {
		imin=std::min(ii+s, m);
		#pragma unroll
		for (size_t jj=0; jj<n; jj+=s) {
			jmin=std::min(jj+s, n);
			#pragma unroll
			for (size_t kk=0; kk<p; kk+=s) {
				kmin=std::min(kk+s, p);


				#pragma unroll
				for (size_t i=ii; i<imin; ++i) {
					#pragma unroll
					for (size_t k=kk; k<kmin; ++k) {
						#pragma unroll
						for (size_t j=jj; j<jmin; ++j) {
							C.matrix_[i*c+j] += A.matrix_[i*a+k] * B.matrix_[k*b+j];
						}
					}
				}


			}

		}
	}
}


template <typename T = int>
void mat_mul_block_avx (const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	const size_t s = 2048 / sizeof(T);
	T blockA [s*s] = {0};
	T blockB [s*s] = {0};

	size_t imin=std::min(s, C.nrows);
	size_t kmin=std::min(s, A.ncols);
	size_t jmin=std::min(s, C.ncols);

	for (size_t ii=0; ii<C.nrows; ii+=s) {
		imin=std::min(ii+s, C.nrows);
		for (size_t jj=0; jj<C.ncols; jj+=s) {
			jmin=std::min(jj+s, C.ncols);
			for (size_t kk=0; kk<A.ncols; kk+=s) {
				kmin=std::min(kk+s, A.ncols);


				for (size_t i=ii; i<imin; ++i) {
					for (size_t k=kk; k<kmin; ++k) {
						for (size_t j=jj; j<jmin; ++j) {
							C.matrix_[i*C.alloc_cols+j] += A.matrix_[i*A.alloc_cols+k] *
														   B.matrix_[k*B.alloc_cols+j];
						}
					}
				}


			}

		}
	}
}


template <typename T>
__global__
void cuda_matrix_multiply_kernel(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int p=0, int axis=0) {

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

	T val = C(row, col);

	switch (axis) {
	case 0:
		_cuda_matrix_multiplication_under_mod_p_(A.ncols, A(row, e), B(e, col));
		break;

	case 1:
		_cuda_matrix_multiplication_under_mod_p_(A.nrows, A(e, row), B(e, col))
		break;

	case 2:
		_cuda_matrix_multiplication_under_mod_p_(A.ncols, A(row, e), B(col, e));
		break;

	case 3:
		_cuda_matrix_multiplication_under_mod_p_(A.nrows, A(e, row), B(col, e));
		break;
	}


	C(row, col) = val;

}

template <typename T>
__host__
void tile_matrix_cpy (const Matrix<T>& X, Matrix<T>& M, const size_t i, const size_t j) {



}


template <typename T>
__host__
void cuda_tile_matrix_cpy (const Matrix<T>& X, Matrix<T>& M, const size_t i, const size_t j) {
	size_t h = M.nrows;
	size_t w = M.ncols;

//	printf("%ld, %ld, %ld, %ld, %ld, %ld\n", i,j,w,h, X.nrows, X.ncols);

	if (i+h > X.nrows) {
		h -= i+h - X.nrows;
	}
	if (j+w > X.ncols) {
		w -= j+w - X.ncols;
	}

//	printf("%ld, %ld, %ld, %ld, %ld, %ld\n", i,j,w,h, X.nrows, X.ncols);

	#pragma unroll
	for (size_t ii=0, i_=i; ii<h; ++ii, ++i_) {
		gpuErrchk (
		cudaMemcpy(M.matrix_gpu_ + ii * M.alloc_cols,
						X.matrix_ + i_ * X.alloc_cols + j,
						w * sizeof(T),
						cudaMemcpyHostToDevice)
		);
	}
}


template <typename T>
void cuda_mat_mul_stream (size_t nStreams,
							const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
	/*
	 *
	 * */

	const size_t tile_height = 4096;
	const size_t tile_width  = 4096;

	const size_t A_tiled_nrows = (A.nrows<tile_height) ? A.nrows : tile_height;
	const size_t A_tiled_ncols = (A.ncols<tile_width ) ? A.ncols : tile_width ;

	const size_t B_tiled_nrows = (B.nrows<tile_height) ? B.nrows : tile_height;
	const size_t B_tiled_ncols = (B.ncols<tile_width ) ? B.ncols : tile_width ;

	const size_t C_tiled_nrows = (C.nrows<tile_height) ? C.nrows : tile_height;
	const size_t C_tiled_ncols = (C.ncols<tile_width ) ? C.ncols : tile_width ;

	const size_t req_mem =	sizeof(linalg::Matrix<T>)*3
							+ sizeof(T) * A_tiled_nrows * A_tiled_ncols
							+ sizeof(T) * B_tiled_nrows * B_tiled_ncols
							+ sizeof(T) * C_tiled_nrows * C_tiled_ncols;

	nStreams = req_mem / gpu_table::free_memory();

	cudaStream_t Streams [nStreams];
	linalg::Matrix<T> A_tiled [nStreams];
	linalg::Matrix<T> B_tiled [nStreams];
	linalg::Matrix<T> C_tiled [nStreams];


	#pragma unroll
	for (size_t i=0; i<nStreams; ++i) {
		cudaStreamCreate(Streams+i);
		A_tiled[i].INIT (A_tiled_nrows, A_tiled_ncols);
		B_tiled[i].INIT (B_tiled_nrows, B_tiled_ncols);
		C_tiled[i].INIT (C_tiled_nrows, C_tiled_ncols);
	}


}



template <int block_size = 16, typename T>
void cuda_mod_mat_mul (Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int p=0, int axis=0) {
	size_t device = 0;

	// Find the free memory on each GPU
	const size_t nb_devs = gpu_table::nb_devices();
	size_t free_mem[nb_devs];
	#pragma unroll
	for (size_t i=0; i<nb_devs; ++i) { // get the free memory on each GPU
		cudaSetDevice(i);
		free_mem[i] = gpu_table::free_memory();
	}
	cudaSetDevice(device);

	// Check to see if there is a GPU in the system that can hold all the data
	const size_t req_mem = linalg::size_of(A) + linalg::size_of(B) + linalg::size_of(C);
	for (size_t i=0, j=free_mem[i]; i<nb_devs; ++i) {
		if (j > req_mem) {
			device=i;
			break;
		}
	}

	 if (free_mem[device] <= req_mem) {	// check to see if all data can fit in one GPU
		 printf("Initiating block multiply on device %ld\n", device);

		cudaSetDevice(device);

		size_t tile_height = 4096;
		size_t tile_width  = 4096;

		size_t A_tiled_nrows = (A.nrows<tile_height) ? A.nrows : tile_height;
		size_t A_tiled_ncols = (A.ncols<tile_width ) ? A.ncols : tile_width ;

		size_t B_tiled_nrows = (B.nrows<tile_height) ? B.nrows : tile_height;
		size_t B_tiled_ncols = (B.ncols<tile_width ) ? B.ncols : tile_width ;

		size_t C_tiled_nrows = (C.nrows<tile_height) ? C.nrows : tile_height;
		size_t C_tiled_ncols = (C.ncols<tile_width ) ? C.ncols : tile_width ;

		// TODO: Write code to increase tile height or width, or both

		linalg::Matrix<T> A_tiled (A_tiled_nrows, A_tiled_ncols);
		linalg::Matrix<T> B_tiled (B_tiled_nrows, B_tiled_ncols);
		linalg::Matrix<T> C_tiled (C_tiled_nrows, C_tiled_ncols);

		linalg::Matrix<T>* d_a = A_tiled.InitializeOnGPU();
		linalg::Matrix<T>* d_b = B_tiled.InitializeOnGPU();
		linalg::Matrix<T>* d_c = C_tiled.InitializeOnGPU();


		int gridDim_x = (C_tiled.ncols + block_size - 1) / block_size;
		int gridDim_y = (C_tiled.nrows + block_size - 1) / block_size;
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		for (size_t i=0; i<C.nrows; i+=C_tiled.nrows) {
			for (size_t j=0; j<C.ncols; j+=C_tiled.ncols) {

				gpuErrchk (
				cudaMemset(C_tiled.matrix_gpu_, 0, C_tiled.nrows*C_tiled.ncols*sizeof(int))
				);

				for (size_t o=0; o<A.ncols; o+=A_tiled.ncols) {

					gpuErrchk (
					cudaMemset(A_tiled.matrix_gpu_, 0, A_tiled.nrows*A_tiled.ncols*sizeof(int))
					);
					linalg::cuda_tile_matrix_cpy(A, A_tiled, i, o);


					gpuErrchk (
					cudaMemset(B_tiled.matrix_gpu_, 0, B_tiled.nrows*B_tiled.ncols*sizeof(int))
					);
					linalg::cuda_tile_matrix_cpy(B, B_tiled, o, j);



					linalg::cuda_matrix_multiply_kernel<<<gridDim, blockDim>>>(*d_a, *d_b, *d_c, p, axis);


					if (cudaSuccess != cudaGetLastError()) {
						printf("Error in kernel launch\n");
					}
					cudaDeviceSynchronize();

				}

				C_tiled.copy_matrix_to_host();

				// Copy matrix from C_tiled to M3
				#pragma unroll
				for (size_t k=0, m=i; k<C_tiled.nrows && m<C.nrows; ++k, ++m) {
					#pragma unroll
					for (size_t l=0, n=j; l<C_tiled.ncols && n<C.ncols; ++l, ++n) {
						C(m,n) = C_tiled(k,l);
					}
				}

			}
		}

	} else {


		cudaSetDevice(device);

		auto d_a = A.InitializeOnGPU();
		auto d_b = B.InitializeOnGPU();
		auto d_c = C.InitializeOnGPU();

		int gridDim_x = (C.ncols + block_size - 1) / block_size;
		int gridDim_y = (C.nrows + block_size - 1) / block_size;

		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);

		linalg::cuda_matrix_multiply_kernel<<<gridDim, blockDim>>>(*d_a, *d_b, *d_c, p, axis);

		if (cudaSuccess != cudaGetLastError()) {
			printf("Error in kernel launch\n");
		}
		cudaDeviceSynchronize();

		A.copy_matrix_to_host();
		B.copy_matrix_to_host();
		C.copy_matrix_to_host();

	}
}





}
#endif
