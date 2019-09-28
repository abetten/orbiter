
// wreath_product.C
//
// Anton Betten, Sajeeb Roy Chowdhury
//
// August 4, 2018
//
//
//

#include "orbiter.h"

//#include <cstdint>

using namespace std;
using namespace orbiter;
using namespace orbiter::top_level;



// global data:

int t0; // the system time when the program started

void usage(int argc, const char **argv);
int main(int argc, const char **argv);





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
#include "gpuErrchk.h"
#include "CUDA/Linalg/linalg.h"


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


template <typename Mat>
__global__
void cuda_matrix_matrix_dot_(Mat& A, Mat& B, Mat& C, int p=0, size_t h = 0) {

	int col = (blockDim.y * blockIdx.y) + threadIdx.y;
	int row = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (row >= C.nrows || col >= C.ncols) return;

	auto val = A(row, 0);
	val = 0;

	for (size_t e = 0; e < A.ncols; ++e) {
		if (p == 0) {
			val += A(row, e) * B(e, col);
		} else {
			val = (val + A(row, e) * B(h * A.ncols + e, col)) % p;
		}
	}

	C(row, col) = val;

}


template <typename T>
__host__
void cuda_dot(linalg::Matrix<T>& A, linalg::Matrix<T>& B, linalg::Matrix<T>& C,
		int* perms, int* result, int q=0, int axis=0,
		bool ea = true, bool eb = true, bool ec = true)
{
	size_t m = A.nrows;
	size_t n = A.ncols;

	size_t o = B.nrows;
	size_t p = B.ncols;


	// Copy matrix A, B and C into device memory
	linalg::Matrix<T>* d_A = A.InitializeOnGPU();
	linalg::Matrix<T>* d_B = B.InitializeOnGPU();
	linalg::Matrix<T>* d_C = C.InitializeOnGPU();

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

	#pragma unroll
	for (size_t h = 0; h < B.nrows / A.ncols; ++h) {

		cout << "at " << __FILE__ << " : " << __LINE__ << " : h=" << h << " / " << B.nrows/A.ncols << endl;

		cuda_matrix_matrix_dot_<<<gridDim, blockDim>>>(*d_A, *d_B, *d_C, q, h);
		// Do some error checking after kernel launch

		cout << "at " << __FILE__ << " : " << __LINE__ << endl;

		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// After the matrix multiplication is done, copy the matrix into
		// host memory.
		C.copy_matrix_to_host();

		#pragma unroll
		for (size_t i = 0; i < C.nrows; ++i) {
			#pragma unroll
			for (size_t j = 0; j < C.ncols; ++j) {
				//a = perms [h * C.nrows + j];
				V(j) = C(i, j);
			}

			#pragma unroll
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
void PGL_Vector_unrank_Matrix (linalg::Matrix<T>& M, size_t vector_size, size_t q, const size_t N) {
	_Vector<T> V (vector_size);

	for (size_t i=0; i<N; ++i) {
		make_vector_from_number (V, i, q);
		for (size_t j=0; j<vector_size; ++j)
			M(i,j) = V(j);
	}
}

template <typename T, typename U>
__host__
void PGL_Vector_unrank_Matrix (linalg::Matrix<T>& M, size_t vector_size, size_t q, const vector<U> in) {
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
	int f_permutations = FALSE;
	int f_orbits = FALSE;
	int f_orbits_restricted = FALSE;
	const char *orbits_restricted_fname = NULL;
	int f_tensor_ranks = FALSE;
	int f_orbits_restricted_compute = FALSE;
	int f_report = FALSE;
	int f_poset_classify = FALSE;
	int poset_classify_depth = 0;
	os_interface Os;


	t0 = Os.os_ticks();

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
		else if (strcmp(argv[i], "-permutations") == 0) {
			f_permutations = TRUE;
			cout << "-permutations " << endl;
			}
		else if (strcmp(argv[i], "-orbits") == 0) {
			f_orbits = TRUE;
			cout << "-orbits " << endl;
			}
		else if (strcmp(argv[i], "-orbits_restricted") == 0) {
			f_orbits_restricted = TRUE;
			orbits_restricted_fname = argv[++i];
			cout << "-orbits_restricted " << endl;
			}
		else if (strcmp(argv[i], "-tensor_ranks") == 0) {
			f_tensor_ranks = TRUE;
			cout << "-tensor_ranks " << endl;
			}
		else if (strcmp(argv[i], "-orbits_restricted_compute") == 0) {
			f_orbits_restricted_compute = TRUE;
			orbits_restricted_fname = argv[++i];
			cout << "-orbits_restricted_compute " << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
			}
		else if (strcmp(argv[i], "-poset_classify") == 0) {
			f_poset_classify = TRUE;
			poset_classify_depth = atoi(argv[++i]);
			cout << "-poset_classify " << poset_classify_depth << endl;
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




	tensor_classify *T;

	T = NEW_OBJECT(tensor_classify);

	T->init(argc, argv, nb_factors, d, q, depth,
			f_permutations, f_orbits, f_tensor_ranks,
			f_orbits_restricted, orbits_restricted_fname,
			f_orbits_restricted_compute,
			f_report,
			f_poset_classify, poset_classify_depth,
			verbose_level);

	the_end_quietly(t0);

}





