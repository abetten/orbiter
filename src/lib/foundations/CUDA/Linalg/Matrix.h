/*
 * Matrix.h
 *
 *  Created on: Jul 22, 2019
 *      Author: sajeeb
 */

#ifndef MATRIX_H_
#define MATRIX_H_


template <typename T>
class Matrix {
public:

	__device__
	__host__
	Matrix(const size_t nrows, const size_t ncols) {
		INIT(nrows, ncols);
	}

	__device__
	__host__
	Matrix () {}


	__device__
	__host__
	__inline__
	__attribute__((always_inline))
	void INIT (const size_t nrows, const size_t ncols) {
		// set the number of rows and columns
		this->nrows = this->alloc_rows = nrows;
		this->ncols = this->alloc_cols = ncols;

		// Initialize the matrix
		this->matrix_ = new T [nrows * ncols];
		memset(matrix_, 0, sizeof(T)*nrows*ncols);
	}


	__device__
	__host__
	Matrix(Matrix& m) {
		this->nrows = this->alloc_rows = m.get_nrows();
		this->ncols = this->alloc_cols = m.get_ncols();
		this->matrix_ = new T [nrows * ncols];
		memcpy(matrix_, m.matrix_, sizeof(matrix_[0]) * nrows * ncols);
	}


	__device__
	__host__
	~Matrix() {

		if (matrix_ && dealloc_data_on_destruct)
		{
			delete [] matrix_;
			matrix_ = NULL;
			nrows = ncols = alloc_rows = alloc_cols = 0;
		}

		if (gpu_class && dealloc_gpu_obj_on_destruct) {
			cudaFree(gpu_class);
			gpu_class = NULL;
		}

		if (matrix_gpu_&& dealloc_gpu_data_on_destruct) {
			cudaFree(matrix_gpu_);
			matrix_gpu_ = NULL;
		}

	}

	// overload subscript operator
	__device__
	__host__
	__attribute__((always_inline))
	const T& operator() (size_t i, size_t j) const {
		#ifndef __CUDA_ARCH__
		if (i >= nrows || j >= ncols) {
			printf("%s:%d:index out of range", __FILE__, __LINE__);
			exit(-1);
		}
		#endif

		return matrix_[i*alloc_cols + j];
	}

	__device__
	__host__
	__attribute__((always_inline))
	T& operator() (size_t i, size_t j) {
		#ifndef __CUDA_ARCH__
		if (i >= nrows || j >= ncols) {
			printf("%s:%d:index (%ld, %ld) out of range\n", __FILE__, __LINE__, i, j);
			exit(-1);
		}
		#endif

		return matrix_[i*alloc_cols + j];
	}

	__device__
	__host__
	__attribute__((always_inline))
	bool operator== (Matrix<T>& M) {
		if (M.nrows != nrows && M.ncols != ncols) return false;
		// entry wise check
		for (size_t i=0; i<nrows; ++i) {
			for (size_t j=0; j<ncols; ++j) {
				if (M(i, j) != this->operator()(i, j)) {
//					printf("(%ld, %ld)\n", i,j);
					return false;
				}
			}
		}
		return true;
	}

	// Getters
	__host__
	__device__
	__attribute__((always_inline))
	size_t get_nrows() const { return nrows; }

	__host__
	__device__
	__attribute__((always_inline))
	size_t get_ncols() const { return ncols; }


	__host__
	__device__
	__attribute__((always_inline))
	void swapRows(size_t i,  size_t j) {
	#ifdef __CUDA_ARCH__

		for (size_t c=0; c<ncols; ++c) {
			T a = matrix_[i*alloc_cols+c];
			matrix_[i*alloc_cols+c] = matrix_[j*alloc_cols+c];
			matrix_[j*alloc_cols+c] = a;
		}

	#else

		for (size_t c=0; c<ncols; ++c)
			std::swap(matrix_[i*alloc_cols+c], matrix_[j*alloc_cols+c]);

	#endif
	}

	__device__
	__host__
	__attribute__((always_inline))
	void swapCols(size_t i, size_t j) {
	#ifdef __CUDA_ARCH__

		for (size_t c=0; c<nrows; ++c) {
			T a = matrix_[c*alloc_cols+i];
			matrix_[c*alloc_cols+i] = matrix_[c*alloc_cols+j];
			matrix_[c*alloc_cols+j] = a;
		}

	#else

		for (size_t c=0; c<nrows; ++c)
			std::swap(matrix_[c*alloc_cols+i], matrix_[c*alloc_cols+j]);

	#endif
	}

	__host__
	__attribute__((always_inline))
	void appendRight(Matrix& M) {
		// Append another matrix to the right of the current matrix
		if (M.alloc_rows != alloc_rows) {
			printf("%s:%d:cannot perform appendRight operation on "
					"matricies with unequal rows.\n",
					__FILE__, __LINE__);
			exit(-1);
		}

		size_t new_alloc_col = alloc_cols + M.alloc_cols;

		T* new_matrix = new T [alloc_rows * new_alloc_col]();

		for (size_t i=0; i<alloc_rows; ++i) {
			memcpy( new_matrix+i*new_alloc_col,
					matrix_+i*alloc_cols,
					sizeof(T) * alloc_cols);

			memcpy( new_matrix+i*new_alloc_col+alloc_cols,
					M.matrix_+i*M.alloc_cols,
					sizeof(T) * M.alloc_cols);
		}

		delete [] matrix_;
		matrix_ = new_matrix;
		alloc_cols = new_alloc_col;
		ncols = alloc_cols;
	}

	template <typename Vec>
	__host__
	__attribute__((always_inline))
	void appendRight(Vec& V) {
		if (alloc_rows != V.size()) {
			printf("%s:%d:",__FILE__, __LINE__);
			printf("cannot perform appendRight operation on ");
			printf("matricies with unequal rows.\n");
			exit(-1);
		}

		if (sizeof(V(0)) != sizeof(this->operator ()(0, 0))) {
			printf("%s:%d:",__FILE__, __LINE__);
			printf("cannot perform appendRight operation on ");
			printf("matrix with different data types.\n");
			exit(-1);
		}

		size_t new_alloc_col = alloc_cols + 1;

		T* new_matrix = new T [alloc_rows * new_alloc_col]();

		for (size_t i=0; i<alloc_rows; ++i) {
			memcpy( new_matrix+i*new_alloc_col,
					matrix_+i*alloc_cols,
					sizeof(T) * alloc_cols);

			memcpy( new_matrix+i*new_alloc_col+alloc_cols,
					&V(i),
					sizeof(T) );
		}

		delete [] matrix_;
		matrix_ = new_matrix;
		alloc_cols = new_alloc_col;
	}

	__host__
	__attribute__((always_inline))
	void appendBottom(const Matrix& M)
	{
		// Append another matrix to the bottom of the current matrix
		// This function appends the entire input matrix to the bottom
		// of the current matrix even if the rows and cols of the input
		// matrix is different than the actual allocated size of the
		// input matrix.

		// Note that appending another matrix to the bottom of this
		// matrix does not change the perspective of the current matrix
		// i.e. if the perspective of the current matrix was nxn before
		// performing the append operation with mxn matrix. the
		// perspective of the current matrix will not change to (n+m)xn
		// it will still be nxn unless a reset dimension operation is
		// performed where the perspective of the current matrix is
		// changed to the actual allocated rows and columns.


		if (M.alloc_cols != alloc_cols) {
			printf("%s:%d:", __FILE__, __LINE__);
			printf("cannot perform appendBottom operation on matrices ");
			printf("with unequal columns.\n");
			exit(-1);
		}

		size_t new_alloc_row = alloc_rows + M.alloc_rows;

		T* new_matrix = new T [new_alloc_row * alloc_cols]();

		memcpy( new_matrix,
				matrix_,
				sizeof(matrix_[0]) * alloc_rows * alloc_cols);

		memcpy( new_matrix+(alloc_rows*alloc_cols),
				M.matrix_,
				sizeof(matrix_[0]) * M.alloc_rows * M.alloc_cols);

		delete [] matrix_;
		matrix_ = new_matrix;
		alloc_rows = new_alloc_row;
	}

	__host__
	__attribute__((always_inline))
	void Init(const Matrix<T>& M) {
		if (alloc_rows >= M.nrows && alloc_cols >= M.ncols) {
			memcpy(matrix_, M.matrix_, sizeof(M.matrix_[0])*M.nrows*M.ncols);
			nrows = M.nrows;
			ncols = M.ncols;
		} else {
			std::cout << __FILE__ << ":" << __LINE__ << std::endl;
			std::cout << "Implementation of Matrix init not present for"
			 	 <<	" alloc_rows and alloc_cols less than"
			 	 << " the rows and cols of input matrix." << std::endl;
			exit(-1);
		}
	}

	__host__
	__device__
	__attribute__((always_inline))
	__inline__
	void set_data_reference (T* data, const size_t nr, const size_t nc,
											bool free_current_allocation = true) {
		alloc_rows = nrows = nr;
		alloc_cols = ncols = nc;

		if (matrix_ && free_current_allocation) {
			delete [] matrix_;
		}

		matrix_ = data;
	}


	__host__
	__attribute__((always_inline))
	Matrix<T>*
	InitializeOnGPU()
	{
		// This method initializes this entire class on the GPU

		if (gpu_class) return gpu_class;

		T* matrix_tmp_ = matrix_;


		gpuErrchk( cudaMalloc(  &matrix_,
								sizeof(T)*alloc_rows*alloc_cols  ) );

		gpuErrchk( cudaMemcpy(  matrix_,
								matrix_tmp_,
								sizeof(T)*alloc_rows*alloc_cols,
								cudaMemcpyHostToDevice  ) );





		Matrix<T>* tmp_gpu_class = NULL;
		gpuErrchk( cudaMalloc(&tmp_gpu_class, sizeof(Matrix<T>)) );
		gpuErrchk( cudaMemcpy(tmp_gpu_class, this, sizeof(Matrix<T>), cudaMemcpyHostToDevice) );
		gpu_class = tmp_gpu_class;


		matrix_gpu_ = matrix_;
		matrix_ = matrix_tmp_;

		return gpu_class;
	}

	__host__
	__attribute__((always_inline))
	void
	UninitializeOnGPU()
	{
		// This method uninitializes this class from the GPU
		if (gpu_class) {
			gpuErrchk( cudaFree(gpu_class) );
			gpuErrchk( cudaFree(matrix_gpu_) );
			gpu_class = NULL;
			matrix_gpu_ = NULL;
		}
	}

	__host__
	__attribute__((always_inline))
	void
	copy_matrix_to_host()
	{
		gpuErrchk( cudaMemcpy(  matrix_,
									matrix_gpu_,
									sizeof(T)*alloc_cols*alloc_rows,
									cudaMemcpyDeviceToHost) );
	}

	__host__
	__attribute__((always_inline))
	void
	copy_matrix_to_device()
	{
		gpuErrchk( cudaMemcpy(  matrix_gpu_,
									matrix_,
									sizeof(T)*alloc_cols*alloc_rows,
									cudaMemcpyHostToDevice) );
	}

	__device__
	__host__
	__attribute__((always_inline))
	void reset_dimensions() {
		nrows = alloc_rows;
		ncols = alloc_cols;
	}

	__device__
	__host__
	__attribute__((always_inline))
	void perspective(long int nrows, long int ncols) {
		if (nrows < 0 && ncols > 0) { // keep rows unchanged
			this->ncols = ncols;
		}
		if (nrows > 0 && ncols < 0) { // keep cols unchanged
			this->nrows = nrows;
		}
		if (nrows > 0 && ncols > 0) { // change rows and cols
			this->nrows = nrows;
			this->ncols = ncols;
		}
	}



	__host__
	__attribute__((always_inline))
	void shrink_row_to_fit() {
		if (alloc_rows > nrows) {
			T* new_matrix_ = new T [nrows*ncols];
			memcpy(new_matrix_, matrix_, sizeof(T)*nrows*ncols);
			delete [] matrix_;
			matrix_ = new_matrix_;
			alloc_rows = nrows;
		}
	}

	__host__
	__attribute__((always_inline))
	void shrink_col_to_fit() {
		if (alloc_cols > ncols) {
			T* new_matrix_ = new T [nrows*ncols];
			memcpy(new_matrix_, matrix_, sizeof(T)*nrows*ncols);
			delete [] matrix_;
			matrix_ = new_matrix_;
			alloc_cols = ncols;
		}
	}

	__host__
	__attribute__((always_inline))
	void remove_row(size_t i) {
		if (i >= nrows) {
			printf("%s:%d:index (%ld) out of range\n", __FILE__, __LINE__, i);
			exit(-1);
		}
		if (alloc_rows > nrows) {
			for (size_t j=i; j<nrows-1; ++j) {
				swapRows(j, j+1);
			}
			nrows -= 1;
		} else {

		}
	}

	__host__
	__attribute__((always_inline))
	void expand_row(size_t i=1) {
		if (nrows + i > alloc_rows) {
			T* new_matrix_ = new T [(nrows+i)*ncols];
			memcpy(new_matrix_, matrix_, sizeof(T)*ncols*(nrows+i));
			delete [] matrix_;
			matrix_ = new_matrix_;
			alloc_rows = nrows + i;
		} else {
			nrows += i;
		}
	}

	__host__
	__attribute__((always_inline))
	void reset_entries(T a=0) {
		for (size_t i=0; i<alloc_cols*alloc_rows; ++i)
			matrix_[i] = a;
	}

	__host__
	__attribute__((always_inline))
	void reset_row(size_t j, T a=0) {
		for (size_t i=0; i<ncols; ++i)
			matrix_[j * alloc_cols + i] = a;
	}

	T* matrix_ = NULL;			// store matrix element in row major form
	T* matrix_gpu_ = NULL;		// pointer to matrix on GPU

	size_t nrows = 0;			// num rows of matrix
	size_t ncols = 0;			// num columns of matrix

	size_t alloc_rows = 0;		// allocated rows and columns
	size_t alloc_cols = 0;

	Matrix<T>* gpu_class = NULL;

	bool dealloc_data_on_destruct = true;
	bool dealloc_gpu_obj_on_destruct = true;
	bool dealloc_gpu_data_on_destruct = true;

	#ifndef __CUDA_ARCH__
		std::string name = "";
	#endif

};


#endif /* MATRIX_H_ */
