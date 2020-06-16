/*
 * linalg.h
 *
 *  Created on: May 25, 2019
 *      Author: sajeeb
 */

#ifndef LINALG_H_
#define LINALG_H_


#include <iostream>
#include <iomanip>

#ifdef __CUDA_ARCH__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using std::cout;
using std::endl;


namespace linalg {

	/*------------------------------------------------------------------*/
	// Matrix class
	/*------------------------------------------------------------------*/
	template <typename T>
	class Matrix {
	public:

		Matrix (size_t nr, size_t nc) : _nrows(nr), _ncols(nc) {
			data = new T [nr*nc] ();
		}

		~Matrix () {
			if (data) {
				delete [] data;
				data = NULL;
			}
		}

		inline T operator() (size_t i, size_t j) const {
			return data[i*_ncols + j];
		}

		inline T& operator() (size_t i, size_t j) {
			return *(data + i*_ncols + j);
		}

		inline size_t nrows () const { return _nrows; }
		inline size_t ncols () const { return _ncols; }

		void print () {
			size_t length = 0;
			for (size_t i=0; i<_nrows*_ncols; ++i) {
				size_t t = (std::to_string(data[i])).length();
				if (t > length) length = t;
			}
			length += 2;

			for (size_t i=0; i<_nrows; ++i) {
				for (size_t j=0; j<_ncols; ++j) {
					cout << std::setw(length) << data[i*_ncols + j];
				}
				cout << endl;
			}
		}

		T* data = NULL;
		size_t _nrows, _ncols;
	};


	/*------------------------------------------------------------------*/
	// Vector class
	/*------------------------------------------------------------------*/
	template <typename T>
	class Vector : public Matrix<T>{
	public:

	    Vector(size_t nr) : Matrix<T> (nr, 1), _nrows(nr) {

	    }

	    Vector (Vector<T>& V) : Matrix<T>(V._nrows, 1) {
	        _nrows = V._nrows;
	    }

	    ~Vector () {
	    }

	    inline T operator() (size_t i) const {
	        return Matrix<T>::data[i];
	    }

	    void print () {
	        Matrix<T>::print();
	    }

	    size_t _nrows;

	};

	/*------------------------------------------------------------------*/
	// transpose
	/*------------------------------------------------------------------*/
	template <typename T>
	void transpose (linalg::Matrix<T>& M) {
	    for (size_t i=0; i<M.nrows(); ++i) {
	        for (size_t j=i+1; j<M.ncols(); ++j) {
	            std::swap(M(i,j), M(j,i));
	        }
	    }
	    std::swap(M._nrows, M._ncols);
	}

	/*------------------------------------------------------------------*/
	// random fill
	/*------------------------------------------------------------------*/
	template <typename U, typename T>
	void random_fill (linalg::Matrix<T>& M, double low = 0, double high = 1) {
	    for (size_t i=0; i<M.nrows()*M.ncols(); ++i) {
	        double f = (U)rand() / RAND_MAX;
	        f = low + f * (high - low);
	        M.data[i] = f;
	    }
	}

#ifdef __CUDA_ARCH__

	/*------------------------------------------------------------------*/
	// CUDA Kernels
	/*------------------------------------------------------------------*/
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

#endif

}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_LINALG_H_ */
