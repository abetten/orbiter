
// wreath_product.C
//
// Anton Betten, Sajeeb Roy Chowdhury
//
// August 4, 2018
//
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started

void usage(int argc, const char **argv);
int main(int argc, const char **argv);
int wreath_rank_point_func(int *v, void *data);
void wreath_unrank_point_func(int *v, int rk, void *data);
void wreath_product_print_set(ostream &ost, int len, int *S, void *data);
void wreath_product_orbits_CUDA(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int verbosity=0);


typedef class tensor_product tensor_product;

//! classification of tensors under the wreath product group


class tensor_product {
public:
	int argc;
	const char **argv;
	int nb_factors;
	int n;
	int q;

	finite_field *F;
	action *A;
	action *A0;

	strong_generators *SG;
	longinteger_object go;
	wreath_product *W;
	vector_space *VS;
	poset *Poset;
	poset_classification *Gen;
	int vector_space_dimension;
	int *v; // [vector_space_dimension]

	tensor_product();
	~tensor_product();
	void init(int argc, const char **argv,
			int nb_factors, int n, int q, int depth,
			int verbose_level);
};





void usage(int argc, const char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n>                   : verbose level n" << endl;
	cout << "-nb_factors <nb_factors> : set number of factors" << endl;
	cout << "-d <d>                   : set dimension d" << endl;
	cout << "-q <q>                   : set field size q" << endl;
}

/*-------------------------------------------------------*/
// CUDA Stuff
/*-------------------------------------------------------*/
#ifdef __CUDACC__
#include "CUDA/gpuErrchk.h"



// The reason this function exists is because the modulo
// operator in c++ is implementation dependent. This block
// of code works around the implementation dependent modulo
// operator
__device__ __host__
int mod(int a, int p)
{
	if (a < 0) {
		int v = a % p;
		if (v < 0)
			a = p + v;
		else
			return v;
	} else {
		a %= p;
	}
	return a;
}

void xgcd(long *result, long a, long b)
{
	// This block of code has been adapted from:
	// https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
	long aa[2]={1,0}, bb[2]={0,1}, q;
	while(1) {
		q = a / b;
		a = a % b;
		aa[0] = aa[0] - q*aa[1];
		bb[0] = bb[0] - q*bb[1];
		if (a == 0) {
			result[0] = b;
			result[1] = aa[1];
			result[2] = bb[1];
			return;
		};
		q = b / a;
		b = b % a;
		aa[1] = aa[1] - q*aa[0];
		bb[1] = bb[1] - q*bb[0];
		if (b == 0) {
			result[0] = a;
			result[1] = aa[0];
			result[2] = bb[0];
			return;
		};
	};
}

int modinv(int a, int b)
{
	long c[3];
	xgcd(c,a,b);
	long x = c[1];
	return mod(x, b);
}




template <typename T>
class _Vector {
public:

	__device__
	__host__
	_Vector() {}

	__host__
	_Vector(size_t size)
	{
		size_ = size;
		vec_ = new T [size_]();
	}

	__device__
	__host__
	_Vector(T v[], size_t size)
	{
		size_ = size;
		vec_ = v;
	//	vec_ = new unsigned int [size_];
	//	memcpy(vec_, v, sizeof(*vec_) * size_);
	}

	__host__
	_Vector(const _Vector& vec)
	{
		T* new_vec_ = new T [vec.size_];
		memcpy(new_vec_, vec.vec_, sizeof(new_vec_[0]) * vec.size_);
		delete [] vec_;
		vec_ = new_vec_;
	}

	//__device__
	__host__
	~_Vector()
	{
		if (vec_gpu_)
		{
			cudaFree(vec_);
			vec_ = vec_gpu_;
			vec_gpu_ = NULL;
		}

		if (vec_ != NULL) {
			delete [] vec_;
			size_ = 0;
			vec_ = NULL;
		}
	}

	__device__
	__host__
	bool operator== (const _Vector& V) const {
		if (V.size_ != size_) return false;
		for (size_t i=0; i<size_; ++i) {
			if (V(i) != vec_[i]) return false;
		}
		return true;
	}

	__device__
	__host__
	Vector& operator=(const _Vector& v) {
		auto* new_vec_ = new unsigned int [v.size_];
		memcpy(new_vec_, v.vec_, sizeof(*new_vec_) * v.size_);
		delete [] vec_;
		vec_ = new_vec_;
		return *this;
	}

	__device__
	__host__
	bool operator< (const _Vector& V) const {
		return num_rep_ < V.num_rep_;
	}

	__device__
	__host__
	const T& operator() (int i) const {
		if (i >= size_ || i < 0) {
			printf("%s:%d:index out of range\n", __FILE__, __LINE__);
			exit(-1);
		}
		return vec_[i];
	}

	__device__
	__host__
	T& operator[] (size_t i) {
		if (i >= size_) {
			printf("%s:%d:index out of range\n", __FILE__, __LINE__);
			exit(-1);
		}
		return vec_[i];
	}

	__device__
	__host__
	T& operator() (int i) {
	#ifndef __CUDACC__
		if (i >= size_ || i < 0) {
			printf("%s:%d:index out of range\n", __FILE__, __LINE__);
			exit(-1);
		}
	#else
		return vec_[i];
	#endif
	}

	__device__
	__host__
	size_t size() const { return size_; }

	__host__
	void print()
	{	cout << "[";
		for (size_t i=0; i<size_; ++i) {
			cout << this->operator()(i);
			if (i+1 != size_) cout << ", ";
		}
		cout << "]";
	}

	__host__
	__device__
	inline int num_rep() const {return num_rep_;}

	__host__
	void make_str_rep() {
		str = "[";
		for (size_t i=0; i<size_; ++i) {
			str += std::to_string(this->operator ()(i));
			if (i+1 != size_) str += ", ";
		}
		str += "]";
	}

	__device__
	__host__
	void
	InitializeOnGPU()
	{
		T* tmp = vec_;
		gpuErrchk( cudaMalloc(&vec_, sizeof(T)*size_) );
		gpuErrchk( cudaMemcpy(vec_, tmp, sizeof(T)*size_, cudaMemcpyHostToDevice) );
		vec_gpu_ = tmp;
	}

//	template <typename Mat, typename Vec>
//	__device__ __host__
//	friend Vec* cuda_dot(Mat& M, Vec& V);

//private:

	size_t size_ = 0;

	T* vec_ = NULL;
	T* vec_gpu_ = NULL;
	int num_rep_ = 0;

	std::string str = "";

};




template <typename Vec>
void right_normalize_(_Vector<Vec>& x, int p) {
	// last non-zero element made one
	int i, j, a;
	int len = x.size();

	for (i = len - 1; i >= 0; i--) {
		a = x[i];
		if (a) {
			if (a == 1) {
				return;
			}
			a = modinv(a, p);
			x[i] = 1;
			for (j = i - 1; j >= 0; j--) {
				x[j] = mod(x[j] * a, p);
			}
			return;
		}
	}
	cout << __FILE__ << ":" << __LINE__ << endl;
	cout << "PG_element_normalize() zero vector()" << endl;
	exit(1);
}



template <typename T>
class Matrix {
public:

	__device__
	__host__
	Matrix(const size_t nrows, const size_t ncols) {
		// set the number of rows and columns
		this->nrows = this->alloc_rows = nrows;
		this->ncols = this->alloc_cols = ncols;

		// Initialize the matrix
		this->matrix_ = new T [nrows * ncols]();
	}

	__device__
	__host__
	Matrix(Matrix& m) {
		this->nrows = this->alloc_rows = m.get_nrows();
		this->ncols = this->alloc_cols = m.get_ncols();
		this->matrix_ = new T [nrows * ncols];
		memcpy(matrix_, m.matrix_, sizeof(matrix_[0]) * nrows * ncols);
	}

//	Generate matrix from vector
	template <typename Vec>
	__device__
	__host__
	Matrix(Vec& V) {
		nrows = this->alloc_rows = V.size();
		ncols = this->alloc_cols = 1;
		matrix_ = new T [nrows * ncols];
		memcpy(matrix_, V.vec_, sizeof(T)*nrows*ncols);
	}

//	Generate matrix from multiple row vectors by placing each vector in
//	a new column or row of the matrix depending on the axis
	template <typename v>
	__host__
	Matrix(_Vector<v>* const*  V, size_t n, int axis = 0) {
		// n is the number of column vectors in the
		// matrix

		// Check to make sure that the size of every
		// vector is the same
		for (size_t i=0, j=V[0]->size(); i<n; ++i) {
			if ( V[i]->size() != j ) {
				printf("%s:%d:", __FILE__, __LINE__);
				printf("one of the input vectors have size not");
				printf(" equal to %ld \n", j);
				exit(-1);
			}
		}

		if (axis == 0) {
			// Place every vector in a new column of the matrix
			matrix_ = new T [ V[0]->size() * n ];
			nrows = this->alloc_rows = V[0]->size();
			ncols = this->alloc_cols = n;
			for (size_t j=0; j<ncols; ++j) {
				for (size_t i=0; i<nrows; ++i) {
					this->operator ()(i, j) = V[j]->operator() (i);
				}
			}
		}
	}

	__device__
	__host__
	~Matrix() {
		if (matrix_)
		{
#ifdef __CUDA_ARCH__
			cudaFree(matrix_);
#else
			delete [] matrix_;
#endif
			matrix_ = NULL;
			nrows = ncols = this->alloc_rows = this->alloc_cols = 0;
		}
	}

	// overload subscript operator
	__device__
	__host__
	const T& operator() (size_t i, size_t j, bool cuda=false) const {
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
	T& operator() (size_t i, size_t j, bool cuda=false) {
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
	bool operator== (Matrix<T>& M) {
		if (M.nrows == nrows && M.ncols == ncols) { // dimension check
			// entry wise check
			for (size_t i=0; i<nrows; ++i) {
				for (size_t j=0; j<ncols; ++j) {
					if (M(i, j) != this->operator()(i, j)) return false;
				}
			}
		}
		return false;
	}

	__device__
	__host__
	void print(int precision=-1) {
		for (size_t i=0; i<nrows; ++i) {
			for (size_t j=0; j<ncols; ++j) {
#ifndef __CUDA_ARCH__
				if (precision>0) cout.precision(precision);
				cout << std::setw(7) << matrix_[i*ncols + j] << " ";
#else
				printf("%.10f", matrix_[i*alloc_cols + j]);
				printf("   ");
#endif
			}
#ifdef __CUDA_ARCH__
			printf("\n");
#else
			cout << endl;
#endif
		}
	}

	__host__
	void print_row(size_t i) {
		for (size_t j=0; j<ncols; ++j)
			cout << std::setw(5) << this->operator()(i, j);
		cout << endl;
	}

	// Getters
	__host__
	__device__
	size_t get_nrows() const { return nrows; }

	__host__
	__device__
	size_t get_ncols() const { return ncols; }


	__host__
	__device__
	inline void swapRows(size_t i,  size_t j, bool cuda=false) {
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
	inline void swapCols(size_t i, size_t j) {
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
	void identity() {
		// Turn the current matrix into an identity matrix
		for (size_t i=0, j=0; i<nrows; ++i, j=0) {

			for (; j<i; ++j)
				matrix_[i*alloc_cols + j] = 0;

			matrix_[i*alloc_cols + j] = 1;
			j++;

			for (; j<ncols; ++j)
				matrix_[i*alloc_cols + j] = 0;

		}
	}

	__host__
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

	__host__
	T* allocate_new_matrix_memory_(size_t size) {
		return new T [size];
	}

	template <typename Vec>
	__host__
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
	void Init(const Matrix<T>& M) {
		if (alloc_rows >= M.nrows && alloc_cols >= M.ncols) {
			memcpy(matrix_, M.matrix_, sizeof(M.matrix_[0])*M.nrows*M.ncols);
			nrows = M.nrows;
			ncols = M.ncols;
		} else {
			cout << __FILE__ << ":" << __LINE__ << endl;
			cout << "Implementation of Matrix init not present for"
			 	 <<	" alloc_rows and alloc_cols less than"
			 	 << " the rows and cols of input matrix." << endl;
			exit(-1);
		}
	}

	__host__
	Matrix<T>*
	InitializeOnGPU(bool pinned=false)
	{
		// This method initializes this entire class on the GPU

		T* matrix_tmp_ = matrix_;

		if (!pinned) {

			gpuErrchk( cudaMalloc(  &matrix_,
									sizeof(T)*alloc_rows*alloc_cols  ) );

			gpuErrchk( cudaMemcpy(  matrix_,
									matrix_tmp_,
									sizeof(T)*alloc_rows*alloc_cols,
									cudaMemcpyHostToDevice  ) );

			gpuErrchk( cudaMalloc(&gpu_class, sizeof(Matrix<T>)) );
			gpuErrchk( cudaMemcpy(gpu_class, this, sizeof(Matrix<T>), cudaMemcpyHostToDevice) );

			matrix_gpu_ = matrix_;

		} else {
			if (!gpu_class){

				gpuErrchk( cudaMallocManaged(  &matrix_,
										sizeof(T)*alloc_rows*alloc_cols  ) );

				gpuErrchk( cudaMemcpy(  matrix_,
										matrix_tmp_,
										sizeof(T)*alloc_rows*alloc_cols,
										cudaMemcpyHostToDevice  ) );

				gpuErrchk( cudaMallocManaged(&gpu_class, sizeof(Matrix<T>)) );
				gpuErrchk( cudaMemcpy(gpu_class, this, sizeof(Matrix<T>), cudaMemcpyHostToDevice) );

				matrix_gpu_ = matrix_;

			}

		}


		matrix_ = matrix_tmp_;

		return gpu_class;
	}

	__host__
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
	void
	copy_matrix_to_host(bool override = false)
	{
		if (override)
		{
			gpuErrchk( cudaMemcpy(  matrix_,
									matrix_gpu_,
									sizeof(T)*alloc_cols*alloc_rows,
									cudaMemcpyDeviceToHost) );
		}
		else
		{
			return;
		}
	}

	__device__
	__host__
	void
	reset_dimensions() {
		nrows = alloc_rows;
		ncols = alloc_cols;
	}

	__device__
	__host__
	void
	perspective(long int nrows, long int ncols) {
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
	void expand_row(size_t i) {
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
	void reset_entries(T a=0) {
		for (size_t i=0; i<alloc_cols*alloc_rows; ++i)
			matrix_[i] = a;
	}

	__host__
	void reset_row(size_t j, T a=0) {
		for (size_t i=0; i<ncols; ++i)
				this->operator()(j, i) = a;
	}

	T* matrix_ = NULL;			// store matrix element in row major form
	T* matrix_gpu_ = NULL;		// pointer to matrix on GPU

	size_t nrows = 0;			// num rows of matrix
	size_t ncols = 0;			// num columns of matrix

	size_t alloc_rows = 0;		// allocated rows and columns
	size_t alloc_cols = 0;

	Matrix<T>* gpu_class = NULL;

	#ifndef __CUDA_ARCH__
		std::string name = "";
	#endif

};







template <typename Mat>
__global__
void cuda_matrix_matrix_dot_(Mat& A, Mat& B, Mat& C, int p=0, int axis=0, size_t h = 0) {

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



	int col = (blockDim.y * blockIdx.y) + threadIdx.y;
	int row = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (row >= C.nrows || col >= C.ncols) return;

	auto val = A(row, 0);
	val = 0;

	if (axis == 0) { // A * B
		for (size_t e = 0; e < A.ncols; ++e) {
			if (p == 0) {
				val += A(row, e) * B(e, col);
			} else {
				val = mod(val + A(row, e) * B(h * A.ncols + e, col), p);
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

	C(row, col) = val;

}


template <typename T>
__host__
void cuda_dot(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C,
		int* perms, int* result, int q=0, int axis=0,
		bool ea = true, bool eb = true, bool ec = true)
{
	size_t m = A.nrows;
	size_t n = A.ncols;

	size_t o = B.nrows;
	size_t p = B.ncols;

//	if (n != o) {
//		printf("%s:%d:Cannot perform matrix multiplication, ", __FILE__, __LINE__);
//		cout << n << " " << o << endl;
//		printf("size of matrix column do not match size of vector.\n");
//		exit(-1);
//	}


	// Copy matrix A, B and C into device memory
	Matrix<T>* d_A = A.InitializeOnGPU(true);
	Matrix<T>* d_B = B.InitializeOnGPU(true);
	Matrix<T>* d_C = C.InitializeOnGPU(true);

	// Find out how many threads are needed assuming each thread
	// works on one entry of the resultant matrix.
	int num_threads = m * p;

	// Find out how many blocks are needed
	int block_size = 16;
	int num_blocks = (num_threads + block_size * block_size - 1) / (block_size * block_size) ;
	int gridDim_y = (C.ncols + block_size - 1) / block_size;
	int gridDim_x = (C.nrows + block_size - 1) / block_size;
	if (num_blocks > gridDim_x * gridDim_y || num_threads > gridDim_x * gridDim_y * pow(block_size,2)) {
		cout << "Error:" << __FILE__ << ":" << __LINE__ <<
		"number of required blocks is greater than number of blocks set."
		<< endl;
	}
	dim3 blockDim(block_size, block_size, 1);
	dim3 gridDim(gridDim_x, gridDim_y, 1);

	cout << "C.nrows: " << C.nrows << ", C.ncols: " << C.ncols << endl;
	cout << "block_size: " << block_size << ", gridDim_x: " << gridDim_x << ", gridDim_y: " << gridDim_y << endl;

	_Vector<T> V (B.ncols);
	_Vector<T> V2 (B.ncols);
	int a = 0;

	for (size_t h = 0; h < B.nrows / A.ncols; ++h) {

		cout << "at " << __FILE__ << " : " << __LINE__ << " : h=" << h << " / " << B.nrows/A.ncols << endl;

		cuda_matrix_matrix_dot_<<<gridDim, blockDim>>>(*d_A, *d_B, *d_C, q, axis, h);
		// Do some error checking after kernel launch

		cout << "at " << __FILE__ << " : " << __LINE__ << endl;

		gpuErrchk( cudaGetLastError() );
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// After the matrix multiplication is done, copy the matrix into
		// host memory.
		C.copy_matrix_to_host(true);

		for (size_t i = 0; i < C.nrows; ++i) {
			for (size_t j = 0; j < C.ncols; ++j) {
				//a = perms [h * C.nrows + j];
				V(j) = C(i, j);
			}
			for (size_t k = 0; k < B.ncols; ++k) {
				a = perms [h * C.ncols + k];
				V2(a) = V(k);
			}
			make_num_rep(V2, q);
			result [h * A.nrows + i] = V2.num_rep_;

		}

	}



	// Free up all space allocated on the GPU for matrix multiplication.
	if (ea) A.UninitializeOnGPU();
	if (eb) B.UninitializeOnGPU();
	if (ec) C.UninitializeOnGPU();
}



template <typename T>
__host__
void make_num_rep(_Vector<T>& v, unsigned int q) {
	// This function assumes that the vector is already normalized

	int i, j, q_power_j, b, sqj;
	int f_v = false;

	int stride = 1, len = v.size();

	if (len <= 0) {
		cout << "PG_element_rank_modified len <= 0" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
		}
		cout << endl;
	}

	right_normalize_(v, q);

	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
		}
		cout << endl;
	}

	for (i = 0; i < len; i++) {
		if (v[i * stride])
			break;
	}

	if (i == len) {
		cout << "PG_element_rank_modified zero vector" << endl;
		exit(1);
	}

	for (j = i + 1; j < len; j++) {
		if (v[j * stride])
			break;
	}

	if (j == len) {
		// we have the unit vector vector e_i
		v.num_rep_ = i;
		return;
	}

	// test for the all one vector:
	if (i == 0 && v[i * stride] == 1) {
		for (j = i + 1; j < len; j++) {
			if (v[j * stride] != 1)
				break;
		}
		if (j == len) {
			v.num_rep_ = len;
			return;
		}
	}

	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride])
			break;
	}

	if (i < 0) {
		cout << "PG_element_rank_modified zero vector" << endl;
		exit(1);
	}

	if (v[i * stride] != 1) {
		cout << "PG_element_rank_modified vector not normalized" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "i=" << i << endl;
	}

	b = 0;
	q_power_j = 1;
	sqj = 0;

	for (j = 0; j < i; j++) {
		b += q_power_j - 1;
		sqj += q_power_j;
		q_power_j *= q;
	}

	if (f_v) {
		cout << "b=" << b << endl;
		cout << "sqj=" << sqj << endl;
	}

	v.num_rep_ = 0;

	for (j = i - 1; j >= 0; j--) {
		v.num_rep_ += v[j * stride];
		if (j > 0)
			v.num_rep_ *= q;
		if (f_v) {
			cout << "j=" << j << ", a=" << v.num_rep_ << endl;
		}
	}

	if (f_v) {
		cout << "a=" << v.num_rep_ << endl;
	}

	// take care of 1111 vector being left out
	if (i == len - 1) {
		//cout << "sqj=" << sqj << endl;
		if (v.num_rep_ >= sqj)
			v.num_rep_--;
	}

	v.num_rep_ += b;
	v.num_rep_ += len;
}

template <typename T>
__host__
void make_vector_from_number (_Vector<T>& vec, unsigned int number, int q) {
	// Create a new in the heap from the number n
	// and return a pointer to it.

	int len = vec.size_;

	int a = number;
	int stride = 1;
	int n, l, ql, sql, k, j, r, a1 = a;

	n = len;

	if (a < n) {
		// unit vector:
		for (k = 0; k < n; k++) {
			if (k == a) {
				vec.vec_[k * stride] = 1;
			}
			else {
				vec.vec_[k * stride] = 0;
			}
		}
		return;
	}
	a -= n;
	if (a == 0) {
		// all one vector
		for (k = 0; k < n; k++) {
			vec.vec_[k * stride] = 1;
		}
		return;
	}
	a--;

	l = 1;
	ql = q;
	sql = 1;
	// sql = q^0 + q^1 + \cdots + q^{l-1}
	while (l < n) {
		if (a >= ql - 1) {
			a -= (ql - 1);
			sql += ql;
			ql *= q;
			l++;
			continue;
		}
		vec.vec_[l * stride] = 1;
		for (k = l + 1; k < n; k++) {
			vec.vec_[k * stride] = 0;
		}
		a++; // take into account that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++;
			// take int account that the
			// vector 11111 has already been listed
		}
		j = 0;
		while (a != 0) {
			r = a % q;
			vec.vec_[j * stride] = r;
			j++;
			a -= r;
			a /= q;
		}
		for (; j < l; j++) {
			vec.vec_[j * stride] = 0;
		}
		return;
	}
	cout << __FILE__ << ":" << __LINE__ << endl;
	cout << "PG_element_unrank_modified a too large" << endl;
	cout << "len = " << len << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}



template <typename T>
__host__
void PGL_Vector_unrank_Matrix (Matrix<T>& M, size_t vector_size, size_t q, const size_t N) {
	_Vector<T> V (vector_size);

	for (size_t i=0; i<N; ++i) {
		make_vector_from_number (V, i, q);
		for (size_t j=0; j<vector_size; ++j)
			M(i,j) = V(j);
	}
}

template <typename T, typename U>
__host__
void PGL_Vector_unrank_Matrix (Matrix<T>& M, size_t vector_size, size_t q, const vector<U> in) {
	_Vector<T> V (vector_size);

	for (size_t i=0; i<in.size(); ++i) {
		make_vector_from_number (V, in[i], q);
			for (size_t j=0; j<vector_size; ++j)
				M(i,j) = V(j);
	}
}

#endif
/*-------------------------------------------------------*/


int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_nb_factors = FALSE;
	int nb_factors = 0;
	int f_d = FALSE;
	int d = 0;
	int f_q = FALSE;
	int q = 0;
	int f_depth = FALSE;
	int depth = 0;

	t0 = os_ticks();

	//f_memory_debug = TRUE;
	//f_memory_debug_verbose = TRUE;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-h") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-help") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-nb_factors") == 0) {
			f_nb_factors = TRUE;
			nb_factors = atoi(argv[++i]);
			cout << "-nb_factors " << nb_factors << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		}
	if (!f_nb_factors) {
		cout << "please use -nb_factors <nb_factors>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_d) {
		cout << "please use -d <d>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_depth) {
		cout << "please use -depth <depth>" << endl;
		usage(argc, argv);
		exit(1);
		}


	//do_it(argc, argv, nb_factors, d, q, verbose_level);


	tensor_product *T;

	T = NEW_OBJECT(tensor_product);

	T->init(argc, argv, nb_factors, d, q, depth, verbose_level);

	the_end_quietly(t0);

}

tensor_product::tensor_product()
{
	argc= 0;
	argv = NULL;
	nb_factors = 0;
	vector_space_dimension = 0;
	v = NULL;
	n = 0;
	q = 0;
	SG = NULL;
	F = NULL;
	A = NULL;
	A0 = NULL;
	W = NULL;
	VS = NULL;
	Poset = NULL;
	Gen = NULL;
}

tensor_product::~tensor_product()
{

}

void tensor_product::init(int argc, const char **argv,
		int nb_factors, int n, int q, int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *v;
	int i, j, a;

	if (f_v) {
		cout << "tensor_product::init" << endl;
	}
	tensor_product::argc = argc;
	tensor_product::argv = argv;
	tensor_product::nb_factors = nb_factors;
	tensor_product::n = n;
	tensor_product::q = q;

	A = NEW_OBJECT(action);

	//v = NEW_int(n);


	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	A->init_wreath_product_group_and_restrict(nb_factors, n,
			F,
			verbose_level);
	cout << "tensor_product::init after "
			"A->init_wreath_product_group_and_restrict" << endl;

	if (!A->f_has_subaction) {
		cout << "tensor_product::init action "
				"A does not have a subaction" << endl;
		exit(1);
	}
	A0 = A->subaction;

	W = A0->G.wreath_product_group;

	vector_space_dimension = W->dimension_of_tensor_action;

	if (!A0->f_has_strong_generators) {
		cout << "tensor_product::init action A0 does not "
				"have strong generators" << endl;
		exit(1);
		}

	v = NEW_int(vector_space_dimension);

	SG = A0->Strong_gens;
	SG->group_order(go);

	cout << "tensor_product::init The group " << A->label
			<< " has order " << go
			<< " and permutation degree " << A->degree << endl;


#if 0
	i = SG->gens->len - 1;
	cout << "generator " << i << " is: " << endl;


	int h;

	cout << "computing image of 2:" << endl;
	h = A->element_image_of(2,
			SG->gens->ith(i), 10 /*verbose_level - 2*/);


	for (j = 0; j < A->degree; j++) {
		h = A->element_image_of(j,
				SG->gens->ith(i), verbose_level - 2);
		cout << j << " -> " << h << endl;
	}

		A->element_print_as_permutation(SG->gens->ith(i), cout);
	cout << endl;
#endif

	cout << "tensor_product::init Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		if (A->degree < 200) {
			A->element_print_as_permutation_with_offset(
					SG->gens->ith(i), cout,
					0 /* offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree*/,
					TRUE /* f_print_cycles_of_length_one*/,
					0 /* verbose_level*/);
			//A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		} else {
			cout << "too big to print" << endl;
		}
	}
	cout << "tensor_product::init Generators as permutations are:" << endl;



	if (A->degree < 200) {
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		}
	}
	else {
		cout << "too big to print" << endl;
	}
	cout << "tensor_product::init Generators in GAP format are:" << endl;
	if (A->degree < 200) {
		cout << "G := Group([";
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation_with_offset(
					SG->gens->ith(i), cout,
					1 /*offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree */,
					FALSE /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < SG->gens->len - 1) {
				cout << ", " << endl;
			}
		}
		cout << "]);" << endl;
	}
	else {
		cout << "too big to print" << endl;
	}
	cout << "tensor_product::init "
			"Generators in compact permutation form are:" << endl;
	if (A->degree < 200) {
		cout << SG->gens->len << " " << A->degree << endl;
		for (i = 0; i < SG->gens->len; i++) {
			for (j = 0; j < A->degree; j++) {
				a = A->element_image_of(j,
						SG->gens->ith(i), 0 /* verbose_level */);
				cout << a << " ";
				}
			cout << endl;
			}
		cout << "-1" << endl;
	}
	else {
		cout << "too big to print" << endl;
	}


	int* result = NULL;

	cout << "time check: ";
	time_check(cout, t0);
	cout << endl;

	cout << "tensor_product::init "
			"before wreath_product_orbits_CUDA:" << endl;
	cout << __FILE__ << ":" << __LINE__ << endl;

	int nb_gens, degree;

	wreath_product_orbits_CUDA(W, SG, A, result, nb_gens, degree);

	cout << "time check: ";
	time_check(cout, t0);
	cout << endl;

	cout << "tensor_product::init "
			"after wreath_product_orbits_CUDA:" << endl;
	cout << __FILE__ << ":" << __LINE__ << endl;
	cout << "we found " << nb_gens << " generators of degree " << degree << endl;


	if (nb_gens == 0) {
		cout << "Cuda not available" << endl;
		exit(1);
	}

	schreier *Sch;

	Sch = NEW_OBJECT(schreier);

	cout << "before Sch->init_images_only" << endl;
	Sch->init_images_only(nb_gens,
			degree, result, verbose_level);
	cout << "computing point orbits from image table:" << endl;
	Sch->compute_all_point_orbits(verbose_level);

	cout << "time check: ";
	time_check(cout, t0);
	cout << endl;


	cout << "computing point orbits from image table done" << endl;
	cout << "We found " << Sch->nb_orbits << " orbits" << endl;


#if 0
	A->perform_tests(SG, verbose_level);
#endif

	exit(0);


	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "wreath_%d_%d_%d", nb_factors, n, q);


	Gen->depth = depth;

	VS = NEW_OBJECT(vector_space);
	VS->init(F,
			vector_space_dimension /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			wreath_rank_point_func,
			wreath_unrank_point_func,
			this,
			verbose_level - 1);


	Poset = NEW_OBJECT(poset);
	Poset->init_subspace_lattice(
			A0, A,
			SG,
			VS,
			verbose_level);

	if (f_v) {
		cout << "tensor_product::init before Gen->init" << endl;
		}
	Gen->init(Poset, Gen->depth /* sz */, verbose_level);
	if (f_v) {
		cout << "tensor_product::init after Gen->init" << endl;
		}


	Gen->f_print_function = TRUE;
	Gen->print_function = wreath_product_print_set;
	Gen->print_function_data = this;

	int nb_nodes = 1000;

	if (f_v) {
		cout << "tensor_product::init "
				"before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "tensor_product::init "
				"calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 1);

	//int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_debug;

	//schreier_depth = Gen->depth;
	f_use_invariant_subset_if_available = TRUE;
	f_debug = FALSE;

	//int t0 = os_ticks();

	if (f_v) {
		cout << "tensor_product::init before Gen->main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
		}


	//Gen->f_allowed_to_show_group_elements = TRUE;

	Gen->main(t0,
		Gen->depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);

	set_of_sets *SoS;

	SoS = Gen->Schreier_vector_handler->get_orbits_as_set_of_sets(
			Gen->root[0].Schreier_vector, verbose_level);

	SoS->sort_all(verbose_level);
	cout << "orbits at level 1:" << endl;
	SoS->print_table();

	for (i = 0; i < SoS->nb_sets; i++) {
		cout << "Orbit " << i << " has size " << SoS->Set_size[i] << " : ";
		int_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
		cout << endl;
		for (j = 0; j < SoS->Set_size[i]; j++) {
			a = SoS->Sets[i][j];
			cout << j << " : " << a << " : ";
			F->PG_element_unrank_modified(v, 1, vector_space_dimension, a);
			int_vec_print(cout, v, vector_space_dimension);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "tensor_product::init after Gen->main" << endl;
	}
}


int wreath_rank_point_func(int *v, void *data)
{
	tensor_product *T;
	int rk;

	T = (tensor_product *) data;
	//AG_element_rank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	T->F->PG_element_rank_modified(v, 1, T->vector_space_dimension, rk);
	return rk;
}

void wreath_unrank_point_func(int *v, int rk, void *data)
{
	tensor_product *T;

	T = (tensor_product *) data;
	//AG_element_unrank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	T->F->PG_element_unrank_modified(v, 1, T->vector_space_dimension, rk);
}


void wreath_product_print_set(ostream &ost,
		int len, int *S, void *data)
{
	tensor_product *T;
	int i;

	T = (tensor_product *) data;
	cout << "set: ";
	int_vec_print(cout, S, len);
	cout << endl;
	for (i = 0; i < len; i++) {
		T->F->PG_element_unrank_modified(T->v,
				1, T->vector_space_dimension, S[i]);
		cout << S[i] << " : ";
		int_vec_print(cout, T->v, T->vector_space_dimension);
		cout << endl;
	}
}

void wreath_product_orbits_CUDA(wreath_product* W,
								strong_generators* SG,
								action* A,
								int*& result,
								int &nb_gens, int &degree,
								int verbosity) {
#ifdef __CUDACC__

	int *generator_stack;
	int *perms;
	int mtx_n;
	int mtx_n2;

	nb_gens = SG->gens->len;
	degree = W->degree_of_tensor_action;
	mtx_n = W->dimension_of_tensor_action;
	mtx_n2 = mtx_n * mtx_n;

	generator_stack = NEW_int(SG->gens->len * mtx_n2);
	perms = NEW_int(SG->gens->len * mtx_n);
	for (size_t i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		W->create_matrix(SG->gens->ith(i), generator_stack + i * mtx_n2,
		0 /* verbose_level */);
		W->compute_induced_permutation(SG->gens->ith(i), perms + i * mtx_n);
	}
	cout << "generator_stack:" << endl;
	int_matrix_print(generator_stack, SG->gens->len * mtx_n, mtx_n);
	cout << "perms:" << endl;
	int_matrix_print(perms, SG->gens->len, mtx_n);
	cout << "mtx_n=" << mtx_n << endl;
	cout << "SG->gens->len * mtx_n=" << SG->gens->len * mtx_n << endl;


	// matrix M contains in the rows the coordinates of all points
	// of the projective geometry:
	// M has size W->degree_of_tensor_action x mtx_n

	Matrix<int> M (W->degree_of_tensor_action, mtx_n);
	PGL_Vector_unrank_Matrix (M, mtx_n, W->q, W->degree_of_tensor_action);

	// matrix N contains the matrices of all projectivities
	// which generate the group, stacked on top of each other.
	// So, N has size (SG->gens->len * mtx_n) x mtx_n

	Matrix<int> N (SG->gens->len * mtx_n, mtx_n);
	for (size_t i = 0; i < N.nrows; ++i) {
		for (size_t j = 0; j < N.ncols; ++j) {
			N(i, j) = generator_stack [i * N.ncols + j];
		}
	}

	// MN = M * N:

	Matrix<int> MN (M.nrows, N.ncols);


	// result is the ranks of the images.
	// Each row of result is a permutation of the points of projective space
	// So, result is SG->gens->len x W->degree_of_tensor_action

	result = NEW_int(SG->gens->len * W->degree_of_tensor_action);

	// perform the parallel matrix multiplication on the GPU:

	cuda_dot(M, N, MN, perms, result, W->q);

//	cout << "result:" << endl;
//	int_matrix_print(result, SG->gens->len, W->degree_of_tensor_action);


	combinatorics_domain Combi;

	for (size_t i = 0; i < SG->gens->len; i++) {
		cout << "testing result " << i << " / " << SG->gens->len << ": ";
		if (Combi.is_permutation(
				result + i * W->degree_of_tensor_action,
				W->degree_of_tensor_action)) {
			cout << "OK" << endl;
		}
		else {
			cout << "not OK" << endl;
		}
	}
	cout << "We found " << SG->gens->len << " permutations of "
			"degree " << W->degree_of_tensor_action << endl;


	cout << __FILE__ << ":" << __LINE__ << endl;
	//exit(0);

	FREE_int(generator_stack);
	FREE_int(perms);
	cout << "wreath_product_orbits_CUDA done" << endl;
#else
	nb_gens = 0;
	degree = 0;
#endif
}


