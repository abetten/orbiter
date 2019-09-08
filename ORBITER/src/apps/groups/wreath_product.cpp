
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



// global data:

int t0; // the system time when the program started

void usage(int argc, const char **argv);
int main(int argc, const char **argv);
int wreath_rank_point_func(int *v, void *data);
void wreath_unrank_point_func(int *v, int rk, void *data);
void wreath_product_print_set(ostream &ost, int len, int *S, void *data);
void compute_permutations(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		int verbose_level=0);
void make_fname(char *fname, int nb_factors, int h, int b);
int test_if_file_exists(int nb_factors, int h, int b);
void orbits(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		int verbose_level);
void orbits_restricted(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		const char *orbits_restricted_fname,
		int verbosity);
void orbits_restricted_compute(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		const char *orbits_restricted_fname,
		int verbose_level);
void wreath_product_rank_one_early_test_func_callback(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


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

	action *Ar;
	int nb_points;
	int *points;


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
			int f_permutations, int f_orbits, int f_tensor_ranks,
			int f_orbits_restricted, const char *orbits_restricted_fname,
			int f_orbits_restricted_compute,
			int f_report,
			int f_poset_classify, int poset_classify_depth,
			int verbose_level);
	void classify_poset(int depth,
			int verbose_level);
	void create_restricted_action_on_rank_one_tensors(
			int verbose_level);
	void early_test_func(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
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


	//do_it(argc, argv, nb_factors, d, q, verbose_level);


	tensor_product *T;

	T = NEW_OBJECT(tensor_product);

	T->init(argc, argv, nb_factors, d, q, depth,
			f_permutations, f_orbits, f_tensor_ranks,
			f_orbits_restricted, orbits_restricted_fname,
			f_orbits_restricted_compute,
			f_report,
			f_poset_classify, poset_classify_depth,
			verbose_level);

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
	Ar = NULL;
	nb_points = 0;
	points = NULL;
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
		int f_permutations, int f_orbits, int f_tensor_ranks,
		int f_orbits_restricted, const char *orbits_restricted_fname,
		int f_orbits_restricted_compute,
		int f_report,
		int f_poset_classify, int poset_classify_depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
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



	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

#if 0
	cout << "tensor_product::init before "
			"A->init_wreath_product_group_and_restrict" << endl;
	A->init_wreath_product_group_and_restrict(nb_factors, n,
			F, f_tensor_ranks,
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
#else
	cout << "tensor_product::init before "
			"A->init_wreath_product_group" << endl;
	A->init_wreath_product_group(nb_factors, n,
			F, f_tensor_ranks,
			verbose_level);
	cout << "tensor_product::init after "
			"A->init_wreath_product_group" << endl;

	A0 = A;
	W = A0->G.wreath_product_group;

#if 0
	int nb_points;
	int *points;
	action *Awr;

	cout << "W->degree_of_tensor_action=" << W->degree_of_tensor_action << endl;
	nb_points = W->degree_of_tensor_action;
	points = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = W->perm_offset_i[nb_factors] + i;
	}

	if (f_v) {
		cout << "action::init_wreath_product_group_and_restrict "
				"before A_wreath->restricted_action" << endl;
	}
	Awr = A->restricted_action(points, nb_points,
			verbose_level);
	Awr->f_is_linear = TRUE;
#endif

#endif

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
		if (A->degree < 400) {
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



	if (A->degree < 400) {
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		}
	}
	else {
		cout << "too big to print" << endl;
	}

#if 0
	cout << "tensor_product::init Generators in ASCII format are:" << endl;
		cout << SG->gens->len << endl;
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_for_make_element(
					SG->gens->ith(i), cout);
				cout << endl;
		}
		cout << -1 << endl;
#endif

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

	if (f_poset_classify) {
		classify_poset(poset_classify_depth, verbose_level + 10);
	}

	if (f_report) {
		cout << "report:" << endl;


		file_io Fio;

		{
		char fname[1000];
		char title[1000];
		char author[1000];
		//int f_with_stabilizers = TRUE;

		sprintf(title, "Wreath product $%s$", W->label_tex);
		sprintf(author, "Orbiter");
		sprintf(fname, "WreathProduct_q%d_n%d.tex", W->q, W->nb_factors);

			{
			ofstream fp(fname);
			latex_interface L;

			//latex_head_easy(fp);
			L.head(fp,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);

			fp << "\\section{The field of order " << q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< W->q
					<< "}$ :\\\\" << endl;
			W->F->cheat_sheet(fp, verbose_level);


			W->report(fp, verbose_level);

			fp << "\\section{Generators}" << endl;
			for (i = 0; i < SG->gens->len; i++) {
				fp << "$$" << endl;
				A->element_print_latex(SG->gens->ith(i), fp);
				if (i < SG->gens->len - 1) {
					fp << ", " << endl;
				}
				fp << "$$" << endl;
			}


			fp << "\\section{The Group}" << endl;
			A->report(fp, verbose_level);


			if (f_poset_classify) {


				{
				char fname_poset[1000];

				Gen->draw_poset_fname_base_poset_lvl(fname_poset, poset_classify_depth);
				Gen->draw_poset(fname_poset,
						poset_classify_depth /*depth*/,
						0 /* data1 */,
						FALSE /* f_embedded */,
						FALSE /* f_sideways */,
						verbose_level);
				}


				fp << endl;
				fp << "\\section{Poset Classification}" << endl;
				fp << endl;


				Gen->report(fp);
				fp << "\\subsection*{Orbits at level " << poset_classify_depth << "}" << endl;
				int nb_orbits, orbit_idx;

				nb_orbits = Gen->nb_orbits_at_level(poset_classify_depth);
				for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
					fp << "\\subsubsection*{Orbit " << orbit_idx << " / " << nb_orbits << "}" << endl;

					int *Orbit; // orbit_length * depth
					int orbit_length;

					cout << "before get_whole_orbit orbit_idx=" << orbit_idx << endl;

					Gen->get_whole_orbit(
							poset_classify_depth, orbit_idx,
							Orbit, orbit_length, verbose_level);

					int *data;

					data = NEW_int(orbit_length);

					for (i = 0; i < orbit_length; i++) {

						fp << "set " << i << " / " << orbit_length << " is: ";


						uint32_t a, b;

						a = 0;
						for (j = 0; j < poset_classify_depth; j++) {
							b = W->rank_one_tensors[Orbit[i * poset_classify_depth + j]];
							a ^= b;
						}

						for (j = 0; j < poset_classify_depth; j++) {
							fp << Orbit[i * poset_classify_depth + j];
							if (j < poset_classify_depth - 1) {
								fp << ", ";
							}
						}
						fp << "= ";
						for (j = 0; j < poset_classify_depth; j++) {
							b = W->rank_one_tensors[Orbit[i * poset_classify_depth + j]];
							fp << b;
							if (j < poset_classify_depth - 1) {
								fp << ", ";
							}
						}
						fp << " = " << a;
						data[i] = a;
						fp << "\\\\" << endl;
					}
					sorting Sorting;

					Sorting.int_vec_heapsort(data, orbit_length);

					fp << "$$" << endl;
					print_integer_matrix_tex(fp, data, (orbit_length + 9)/ 10, 10);
					fp << "$$" << endl;

					classify C;

					C.init(data, orbit_length, TRUE, 0);
					fp << "$$";
					C.print_naked_tex(fp, TRUE /* f_backwards */);
					fp << "$$";
					FREE_int(data);
				}
			}

			L.foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}


		cout << "report done" << endl;
	}





	int *result = NULL;

	cout << "time check: ";
	time_check(cout, t0);
	cout << endl;

	cout << "tensor_product::init "
			"before wreath_product_orbits_CUDA:" << endl;
	cout << __FILE__ << ":" << __LINE__ << endl;

	int nb_gens, degree;

	if (f_permutations) {
		compute_permutations(W, SG, A, result, nb_gens, degree, nb_factors, verbose_level);
	}
	//wreath_product_orbits_CUDA(W, SG, A, result, nb_gens, degree, nb_factors, verbose_level);

	if (f_orbits) {
		orbits(W, SG, A, result, nb_gens, degree, nb_factors, verbose_level);
	}
	if (f_orbits_restricted) {
		orbits_restricted(W, SG, A, result, nb_gens, degree, nb_factors, orbits_restricted_fname, verbose_level);

	}
	if (f_orbits_restricted_compute) {
		orbits_restricted_compute(W, SG, A, result, nb_gens, degree, nb_factors, orbits_restricted_fname, verbose_level);

	}

	cout << "time check: ";
	time_check(cout, t0);
	cout << endl;

	cout << "tensor_product::init "
			"after wreath_product_orbits_CUDA:" << endl;
	cout << __FILE__ << ":" << __LINE__ << endl;
	cout << "we found " << nb_gens << " generators of degree " << degree << endl;



//	schreier *Sch;
//
//	Sch = NEW_OBJECT(schreier);
//
//	cout << "before Sch->init_images_only" << endl;
//	Sch->init_images_only(nb_gens,
//			degree, result, verbose_level);
//
//	cout << "nb_gens: " << nb_gens << endl;
//
//	cout << "computing point orbits from image table:" << endl;
//	Sch->compute_all_point_orbits(0);
//
//	Sch->print_orbit_lengths(cout);
//
//	cout << "time check: ";
//	time_check(cout, t0);
//	cout << endl;
//
//
//	cout << "computing point orbits from image table done" << endl;
//	cout << "We found " << Sch->nb_orbits << " orbits" << endl;
//
//
//#if 0
//	A->perform_tests(SG, verbose_level);
//#endif
//
//	exit(0);
//
//
}


void tensor_product::classify_poset(int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tensor_product::classify_poset" << endl;
	}
	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "wreath_%d_%d_%d", nb_factors, n, q);

	Gen->f_max_depth = TRUE;
	Gen->max_depth = depth;
	Gen->depth = depth;

	if (f_v) {
		cout << "tensor_product::classify_poset before create_restricted_action_on_rank_one_tensors" << endl;
	}
	create_restricted_action_on_rank_one_tensors(verbose_level);
	if (f_v) {
		cout << "tensor_product::classify_poset after create_restricted_action_on_rank_one_tensors" << endl;
	}

#if 0
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
#else
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, Ar,
			SG,
			verbose_level);

	if (f_v) {
		cout << "tensor_product::classify_poset before "
				"Poset->add_testing_without_group" << endl;
		}
	Poset->add_testing_without_group(
			wreath_product_rank_one_early_test_func_callback,
			this /* void *data */,
			verbose_level);
#endif

	if (f_v) {
		cout << "tensor_product::classify_poset before Gen->init" << endl;
		}
	Gen->init(Poset, depth /* sz */, verbose_level);
	if (f_v) {
		cout << "tensor_product::classify_poset after Gen->init" << endl;
		}


	Gen->f_print_function = TRUE;
	Gen->print_function = wreath_product_print_set;
	Gen->print_function_data = this;

	int nb_nodes = 1000;

	if (f_v) {
		cout << "tensor_product::classify_poset "
				"before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "tensor_product::classify_poset "
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
		cout << "tensor_product::classify_poset before Gen->main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
		}


	//Gen->f_allowed_to_show_group_elements = TRUE;

	if (f_v) {
		cout << "tensor_product::classify_poset "
				"before Gen->main, verbose_level=" << verbose_level << endl;
		}
	Gen->main(t0,
		depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);
	if (f_v) {
		cout << "tensor_product::classify_poset "
				"after Gen->main" << endl;
		}
	if (f_v) {
		cout << "tensor_product::classify_poset done" << endl;
	}
}

void tensor_product::create_restricted_action_on_rank_one_tensors(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "tensor_product::create_restricted_action_on_rank_one_tensors" << endl;
	}

	nb_points = W->nb_rank_one_tensors;
	points = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		uint32_t a, b;

		a = W->rank_one_tensors[i];
		b = W->affine_rank_to_PG_rank(a);

		points[i] = W->perm_offset_i[nb_factors] + b;
	}

	if (f_v) {
		cout << "tensor_product::create_restricted_action_on_rank_one_tensors "
				"before A->restricted_action" << endl;
	}
	Ar = A->restricted_action(points, nb_points,
			verbose_level);
	Ar->f_is_linear = TRUE;
	if (f_v) {
		cout << "tensor_product::create_restricted_action_on_rank_one_tensors "
				"after A->restricted_action" << endl;
	}
	if (f_v) {
		cout << "tensor_product::create_restricted_action_on_rank_one_tensors done" << endl;
	}
}


void tensor_product::early_test_func(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_OK;
	int i, j, c;

	if (f_v) {
		cout << "tensor_product::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}


	if (len == 0) {
		int_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
		}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "tensor_product::early_test_func before testing" << endl;
			}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "tensor_product::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
				}

			f_OK = TRUE;
			c = candidates[j];

			for (i = 0; i < len; i++) {
				if (S[i] == c) {
					f_OK = FALSE;
					break;
				}
			}



			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
				}
			} // next j
		} // else
	if (f_v) {
		cout << "tensor_product::early_test_func done" << endl;
	}
}


int wreath_rank_point_func(int *v, void *data)
{
	tensor_product *T;
	int rk;

	T = (tensor_product *) data;
	//AG_element_rank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//T->F->PG_element_rank_modified(v, 1, T->vector_space_dimension, rk);
	rk = T->W->tensor_PG_rank(v);

	//uint32_t tensor_PG_rank(int *tensor);

	return rk;
}

void wreath_unrank_point_func(int *v, int rk, void *data)
{
	tensor_product *T;

	T = (tensor_product *) data;
	//AG_element_unrank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//T->F->PG_element_unrank_modified(v, 1, T->vector_space_dimension, rk);
	T->W->tensor_PG_unrank(v, rk);

	//void tensor_PG_unrank(int *tensor, uint32_t PG_rk);


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


//template <typename T = int>
uint32_t root (uint32_t* S, uint32_t i) {
	while (S[i] != i) i = S[i];
	return i;
}


void compute_permutations(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		int verbose_level)
{
	int *generator_stack;
	int **generators_transposed;
	int *perms;
	int mtx_n;
	int mtx_n2;

	nb_gens = SG->gens->len;
	degree = W->degree_of_tensor_action;
	mtx_n = W->dimension_of_tensor_action;
	mtx_n2 = mtx_n * mtx_n;

	generator_stack = NEW_int(SG->gens->len * mtx_n2);
	generators_transposed = NEW_pint(SG->gens->len);
	perms = NEW_int(SG->gens->len * mtx_n);
	for (size_t h = 0; h < SG->gens->len; h++) {
		cout << "generator " << h << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(h), cout);
		A->element_print_as_permutation(SG->gens->ith(h), cout);
		W->create_matrix(SG->gens->ith(h), generator_stack + h * mtx_n2,
				0 /* verbose_level */);
		cout << "matrix:" << endl;
		int_matrix_print(generator_stack + h * mtx_n2, mtx_n, mtx_n);
		generators_transposed[h] = NEW_int(mtx_n2);

		W->F->transpose_matrix(
				generator_stack + h * mtx_n2,
				generators_transposed[h], mtx_n, mtx_n);

		W->compute_induced_permutation(SG->gens->ith(h), perms + h * mtx_n);
	}

	cout << "generator_stack:" << endl;
	int_matrix_print(generator_stack, SG->gens->len, mtx_n * mtx_n);

#if 0
	cout << "generators transposed:" << endl;
	for (size_t h = 0; h < SG->gens->len; h++) {
		int_matrix_print(generators_transposed[h], mtx_n, mtx_n);
	}
#endif
	cout << "perms:" << endl;
	int_matrix_print(perms, SG->gens->len, mtx_n);
	cout << "mtx_n=" << mtx_n << endl;
	cout << "SG->gens->len * mtx_n=" << SG->gens->len * mtx_n << endl;

#if 0
	linalg::Matrix<int> v (mtx_n, 1);


	// matrix N contains the matrices of all projectivities
	// which generate the group, stacked on top of each other.
	// So, N has size (SG->gens->len * mtx_n) x mtx_n


	vector<linalg::Matrix<char>> N (SG->gens->len);
	for (size_t h = 0; h < N.size(); ++h) {
		N[h].INIT(mtx_n, mtx_n);

		for (size_t i=0; i < mtx_n; ++i)
			for (size_t j = 0; j < mtx_n; ++j) {
				N[h].matrix_[i*mtx_n+j] = generator_stack [h * mtx_n2 + i * mtx_n + j];
			}

	}

	// Print the matrices N
	for (size_t h=0; h<N.size(); ++h) {
		printf("=========================================================\n");
		printf("h = %ld\n", h);
		printf("=========================================================\n");

		linalg::print(N[h]);

		printf("=========================================================\n");
	}
#endif


	// result is the ranks of the images.
	// Each row of result is a permutation of the points of projective space
	// So, result is SG->gens->len x W->degree_of_tensor_action

	//result = NEW_int(SG->gens->len * W->degree_of_tensor_action);

	// perform the parallel matrix multiplication on the GPU:


//	int* v = NEW_int (MN.ncols);

	unsigned int w = (unsigned int) W->degree_of_tensor_action - 1;
	long int a;
	a = (long int) w;
	if (a != W->degree_of_tensor_action - 1) {
		cout << "W->degree_of_tensor_action - 1 does not fit into a unsigned int" << endl;
		exit(1);
	}
	else {
		cout << "W->degree_of_tensor_action fits into a unsigned int, this is good" << endl;
	}



	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	cout << "block_size=" << block_size << endl;

	int nb_blocks = (W->degree_of_tensor_action + block_size - 1) / block_size;

	cout << "nb_blocks=" << nb_blocks << endl;


	//cout << "allocating S, an unsigned int array of size " << W->degree_of_tensor_action << endl;

	//unsigned int* S = new unsigned int [W->degree_of_tensor_action];

	//for (unsigned int i=0; i<W->degree_of_tensor_action; ++i) S[i] = i;


	cout << "allocating T, an unsigned int array of size " << block_size << endl;

	unsigned int* T = new unsigned int [block_size];

//	memset(S, -1, sizeof(S)*W->degree_of_tensor_action);





	for (size_t b=0; b<nb_blocks; ++b) {
		cout << "block b=" << b << " / " << nb_blocks << endl;


		int l = std::min((b + 1) * block_size,
				(unsigned long)W->degree_of_tensor_action) - b*block_size;
		cout << "l=" << l << endl;

		//linalg::Matrix<char> M  (l, mtx_n);

		bitmatrix *M;

		M = NEW_OBJECT(bitmatrix);
		M->init(mtx_n, l, 0 /*verbose_level*/);

		cout << "unranking the elements of the PG to the bitmatrix" << endl;
		M->unrank_PG_elements_in_columns_consecutively(
				W->F, (long int) b * (long int) block_size,
				0 /* verbose_level */);


#if 0
		cout << "unranking the elements of the PG" << endl;

		int l1 = l / 100;
		for (size_t i=0; i<l; ++i) {
			if ((i % l1) == 0) {
				cout << "block b=" << b << ", " << i / l1 << " % done unranking" << endl;
			}
			W->F->PG_element_unrank_modified_lint (v.matrix_, 1, mtx_n,
					(long int) b * (long int) block_size + (long int)i) ;
			for (size_t j=0; j<mtx_n; ++j)
				M(i,j) = v(j, 0);
		}
#endif

		cout << "unranking the elements of the PG done" << endl;

		//M->print();

		//linalg::Matrix<char> MN (l, mtx_n);

		bitmatrix *NM;

		NM = NEW_OBJECT(bitmatrix);
		NM->init(mtx_n, l, 0 /*verbose_level*/);


		for (size_t h=0; h < SG->gens->len; ++h) {
			cout << "generator h=" << h << " / " << SG->gens->len << endl;


			if (!test_if_file_exists(nb_factors, h, b)) {


				// Matrix Multiply
				//MN.reset_entries();
				NM->zero_out();



				//cout << "cuda multiplication" << endl;
				//linalg::cuda_mod_mat_mul (M, N[h], MN, W->q);
				//cout << "cuda multiplication done" << endl;
				//M.UninitializeOnGPU();
				//N[h].UninitializeOnGPU();
				//MN.UninitializeOnGPU();


				cout << "CPU multiplication" << endl;
				int t0, t1, dt;
				t0 = os_ticks();
				//linalg::cpu_mod_mat_mul_block_AB(M, N[h], MN, W->q);
				M->mult_int_matrix_from_the_left(
						generators_transposed[h], mtx_n, mtx_n,
						NM, verbose_level);
				cout << "CPU multiplication done" << endl;
				t1 = os_ticks();
				dt = t1 - t0;
				cout << "the multiplication took ";
				time_check_delta(cout, dt);
				cout << endl;

				//cout << "NM:" << endl;
				//NM->print();


				cout << "ranking the elements of the PG" << endl;
				NM->rank_PG_elements_in_columns(
						W->F, perms + h * mtx_n, T,
						verbose_level);

#if 0
				for (size_t i=0; i<l; ++i) {
					if ((i % l1) == 0) {
						cout << "h=" << h << ", b=" << b << ", " << i/l1 << " % done ranking" << endl;
					}
					for (size_t j=0; j<mtx_n; ++j) {
						int a = perms[h * mtx_n + j];
						v.matrix_[a*v.alloc_cols] = MN (i, j);

					}
					long int res;
					W->F->PG_element_rank_modified_lint (v.matrix_, 1, mtx_n, res);
					T [i] = (unsigned int) res;
				}
#endif
				cout << "ranking the elements of the PG done" << endl;


				cout << "writing to file:" << endl;
				char fname[1000];

				make_fname(fname, nb_factors, h, b);
				{
					ofstream fp(fname, ios::binary);

					fp.write((char *) &l, sizeof(int));
					for (int i = 0; i < l; i++) {
						fp.write((char *) &T [i], sizeof(int));
					}
				}
				//file_io Fio;

				cout << "written file " << fname << endl; //" of size " << Fio.file_size(fname) << endl;


			}
			else {
				cout << "the case h=" << h << ", b=" << b << " has already been done" << endl;
			}

		} // next h

		FREE_OBJECT(M);
		FREE_OBJECT(NM);


	} // next b

#if 0
	int nb_orbits = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) ++nb_orbits;
	}
	cout << "nb_orbits: " << nb_orbits << endl;

	long int *orbit_length;
	long int *orbit_rep;

	orbit_length = NEW_lint(nb_orbits);
	orbit_rep = NEW_lint(nb_orbits);

	for (int i = 0; i < nb_orbits; i++) {
		orbit_length[i] = 0;
	}
	int j;
	j = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) {
			orbit_rep[j++] = i;
		}
	}

	cout << "the orbit representatives are: " << endl;
	for (int i = 0; i < nb_orbits; i++) {
		cout << i << " : " << orbit_rep[i] << endl;
	}
#endif

	return;


//	combinatorics_domain Combi;
//
//	for (size_t i = 0; i < SG->gens->len; i++) {
//		cout << "testing result " << i << " / " << SG->gens->len << ": ";
//		if (Combi.is_permutation(
//				result + i * W->degree_of_tensor_action,
//				W->degree_of_tensor_action)) {
//			cout << "OK" << endl;
//		}
//		else {
//			cout << "not OK" << endl;
//		}
//	}
//	cout << "We found " << SG->gens->len << " permutations of "
//			"degree " << W->degree_of_tensor_action << endl;
//
//
//	cout << __FILE__ << ":" << __LINE__ << endl;
//	//exit(0);
//
//	FREE_int(generator_stack);
//	FREE_int(perms);
//	cout << "wreath_product_orbits_CUDA done" << endl;


//#else
//	nb_gens = 0;
//	degree = 0;
//#endif
}

void make_fname(char *fname, int nb_factors, int h, int b)
{
	sprintf(fname, "w%d_h%d_b%d.bin", nb_factors, h, b);
}

int test_if_file_exists(int nb_factors, int h, int b)
{
	char fname[1000];
	file_io Fio;

	make_fname(fname, nb_factors, h, b);
	if (Fio.file_size(fname) > 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void orbits(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		int verbosity)
{

	int mtx_n;

	nb_gens = SG->gens->len;
	degree = W->degree_of_tensor_action;
	mtx_n = W->dimension_of_tensor_action;

	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	cout << "block_size=" << block_size << endl;

	int nb_blocks = (W->degree_of_tensor_action + block_size - 1) / block_size;

	cout << "nb_blocks=" << nb_blocks << endl;


	cout << "allocating S, an unsigned int array of size " << W->degree_of_tensor_action << endl;

	unsigned int* S = new unsigned int [W->degree_of_tensor_action];

	for (unsigned int i=0; i<W->degree_of_tensor_action; ++i) S[i] = i;


	cout << "allocating T, an unsigned int array of size " << W->degree_of_tensor_action << endl;

	unsigned int* T = new unsigned int [W->degree_of_tensor_action];






	for (size_t h=0; h < SG->gens->len; ++h) {
		cout << "generator h=" << h << " / " << SG->gens->len << endl;

		for (size_t b=0; b<nb_blocks; ++b) {
			cout << "block b=" << b << " / " << nb_blocks << endl;


			int l = std::min((b + 1) * block_size,
					(unsigned long)W->degree_of_tensor_action) - b*block_size;
			cout << "l=" << l << endl;





			if (!test_if_file_exists(nb_factors, h, b)) {
				cout << "file does not exist h=" << h << " b=" << b << endl;
				exit(1);
			}
			else {
				char fname[1000];

				make_fname(fname, nb_factors, h, b);
				cout << "reading from file " << fname << endl;
				{
					ifstream fp(fname, ios::binary);

					int l1;
					fp.read((char *) &l1, sizeof(int));
					if (l1 != l) {
						cout << "l1 != l" << endl;
					}
					for (int i = 0; i < l; i++) {
						fp.read((char *) &T [b * block_size + i], sizeof(int));
					}
				}
				//file_io Fio;

				cout << "read file " << fname << endl; //" of size " << Fio.file_size(fname) << endl;


			} // else
		} // next b

		cout << "performing the union-find for generator " << h << " / " << SG->gens->len << ":" << endl;

		for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
			int l1;

			l1 = W->degree_of_tensor_action / 100;

			if ((i % l1) == 0) {
				cout << i/l1 << " % done with union-find" << endl;
			}
			int u = i;
			unsigned int t = T[i];
			unsigned int r1 = root(S, u);
			unsigned int r2 = root(S, t);

			if (r1 != r2) {
				if (r1 < r2) {
					S[r2] = r1;
				}
				else {
					S[r1] = r2;
				}
			}
		} // next i

	} // next h


	cout << "Done with the loop" << endl;
	cout << "Computing the orbit representatives" << endl;



	int nb_orbits = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) {
			nb_orbits++;
		}
	}
	cout << "nb_orbits: " << nb_orbits << endl;

	long int *orbit_length;
	long int *orbit_rep;

	orbit_length = NEW_lint(nb_orbits);
	orbit_rep = NEW_lint(nb_orbits);

	for (int i = 0; i < nb_orbits; i++) {
		orbit_length[i] = 0;
	}
	int j;
	j = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) {
			orbit_rep[j++] = i;
		}
	}

	cout << "the orbit representatives are: " << endl;
	for (int i = 0; i < nb_orbits; i++) {
		cout << i << ", " << orbit_rep[i] << ", " << endl;
	}
	cout << "Path compression:" << endl;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		unsigned int r = root(S, i);
		S[i] = r;
	}
	cout << "Path compression done" << endl;

	uint32_t *Orbit;
	int goi;
	longinteger_object go;


	SG->group_order(go);
	goi = go.as_int();

	cout << "goi=" << goi << endl;


	Orbit = (uint32_t *) NEW_int(goi);

	cout << "determining the orbits: " << endl;
	for (int orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

		unsigned int rep = orbit_rep[orbit_idx];
		uint32_t len = 0;

		cout << "determining orbit " << orbit_idx << " / " << nb_orbits << " with rep " << rep << endl;
		for (unsigned int j=0; j < W->degree_of_tensor_action; ++j) {
			if (S[j] == rep) {
				Orbit[len++] = j;
			}
		}
		orbit_length[orbit_idx] = len;
		cout << "orbit " << orbit_idx << " / " << nb_orbits << " has length " << len << endl;
		char fname_orbit[1000];

		sprintf(fname_orbit, "wreath_q%d_w%d_orbit_%d.bin", W->q, W->nb_factors, orbit_idx);
		cout << "Writing the file " << fname_orbit << endl;
		{
			ofstream fp(fname_orbit, ios::binary);

			fp.write((char *) &len, sizeof(uint32_t));
			for (int i = 0; i < len; i++) {
				fp.write((char *) &Orbit[i], sizeof(uint32_t));
			}
		}
		cout << "We are done writing the file " << fname_orbit << endl;

	}
	FREE_int((int *) Orbit);
	cout << "the orbits are: " << endl;
	for (int orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		cout << orbit_idx << ", " << orbit_rep[orbit_idx] << ", " << orbit_length[orbit_idx] << ", " << endl;
	}
}


void orbits_restricted(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		const char *orbits_restricted_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int mtx_n;

	if (f_v) {
		cout << "orbits_restricted orbits_restricted_fname=" << orbits_restricted_fname << endl;
	}

	file_io Fio;
	sorting Sorting;

	long int *Set;
	long int *Set_in_PG;
	int set_m, set_n;
	int nb_blocks;
	int *restr_first; // [nb_blocks]
	int *restr_length; // [nb_blocks]
	int i, j;

	Fio.lint_matrix_read_csv(orbits_restricted_fname,
			Set, set_m, set_n, verbose_level);

	if (set_n != 1) {
		cout << "orbits_restricted set_n != 1" << endl;
		exit(1);
	}
	cout << "Restricting to a set of size " << set_m << endl;
	cout << "converting points to PG point labels" << endl;

	int *v;
	long int s;
	v = NEW_int(W->dimension_of_tensor_action);
	Set_in_PG = NEW_lint(set_m);
	for (i = 0; i < set_m; i++) {
		s = W->affine_rank_to_PG_rank(Set[i]);
		Set_in_PG[i] = s;
	}
	//FREE_int(v);
	Sorting.lint_vec_heapsort(Set_in_PG, set_m);
	cout << "after sorting, Set_in_PG:" << endl;
	for (i = 0; i < set_m; i++) {
		cout << i << " : " << Set_in_PG[i] << endl;
	}



	nb_gens = SG->gens->len;
	degree = W->degree_of_tensor_action;
	mtx_n = W->dimension_of_tensor_action;

	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	cout << "block_size=" << block_size << endl;

	nb_blocks = (W->degree_of_tensor_action + block_size - 1) / block_size;

	cout << "nb_blocks=" << nb_blocks << endl;

	restr_first = NEW_int(nb_blocks);
	restr_length = NEW_int(nb_blocks);

	for (size_t b = 0; b < nb_blocks; b++) {

		cout << "block b=" << b << " / " << nb_blocks << endl;


		int idx;
		Sorting.lint_vec_search(Set_in_PG, set_m, (long int) b * block_size,
					idx, 0 /*verbose_level*/);

		restr_first[b] = idx;
	}

	for (int b = 0; b < nb_blocks; b++) {
		cout << b << " : " << restr_first[b] << endl;
	}

	for (int b = nb_blocks - 1; b >= 0; b--) {
		cout << "b=" << b << endl;
		if (b == nb_blocks - 1) {
			restr_length[b] = set_m - restr_first[b];
		}
		else {
			restr_length[b] = restr_first[b + 1] - restr_first[b];
		}
	}

	for (int b = 0; b < nb_blocks; b++) {
		cout << b << " : " << restr_first[b] << " : " << restr_length[b] << endl;
	}

	long int *Perms;

	Perms = NEW_lint(set_m * SG->gens->len);



	cout << "allocating T, an unsigned int array of size " << block_size << endl;

	unsigned int* T = new unsigned int [block_size];





	for (int h = 0; h < SG->gens->len; ++h) {
		cout << "generator h=" << h << " / " << SG->gens->len << endl;

		for (int b = 0; b < nb_blocks; ++b) {
			cout << "block b=" << b << " / " << nb_blocks << endl;


			int l = MINIMUM((b + 1) * block_size,
					(unsigned long)W->degree_of_tensor_action) - b * block_size;
			cout << "l=" << l << endl;





			if (!test_if_file_exists(nb_factors, h, b)) {
				cout << "file does not exist h=" << h << " b=" << b << endl;
				exit(1);
			}
			char fname[1000];

			make_fname(fname, nb_factors, h, b);
			cout << "reading from file " << fname << endl;
			{
				ifstream fp(fname, ios::binary);

				int l1;
				fp.read((char *) &l1, sizeof(int));
				if (l1 != l) {
					cout << "l1 != l" << endl;
				}
				for (int i = 0; i < l; i++) {
					fp.read((char *) &T [i], sizeof(int));
				}
			}
			cout << "read file " << fname << endl; //" of size " << Fio.file_size(fname) << endl;

			long int x, y;
			for (long int u = 0; u < restr_length[b]; u++) {
				i = restr_first[b] + u;
				x = Set_in_PG[i];
				if (x < b * block_size) {
					cout << "x < b * block_size" << endl;
					cout << "x=" << x << " b=" << b << endl;
					exit(1);
				}
				if (x >= (b + 1) * block_size) {
					cout << "x >= (b + 1) * block_size" << endl;
					cout << "x=" << x << " b=" << b << endl;
					exit(1);
				}
				y = T[x - b * block_size];

				int idx;
				if (!Sorting.lint_vec_search(Set_in_PG, set_m, y, idx, 0 /*verbose_level*/)) {
					cout << "did not find element y=" << y << " in Set_in_PG "
							"under generator h=" << h << ", something is wrong" << endl;
					cout << "x=" << x << endl;
					W->tensor_PG_unrank(v, x);
					s = W->tensor_affine_rank(v);
					cout << "tensor=";
					int_vec_print(cout, v, W->dimension_of_tensor_action);
					cout << endl;
					cout << "affine rank s=" << s << endl;

					cout << "y=" << y << endl;
					W->tensor_PG_unrank(v, y);
					s = W->tensor_affine_rank(v);
					cout << "tensor=";
					int_vec_print(cout, v, W->dimension_of_tensor_action);
					cout << endl;
					cout << "affine rank s=" << s << endl;

					exit(1);
				}
				j = idx;
				Perms[i * SG->gens->len + h] = j;
			} // next u

		} // next b

	} // next h

	char fname[1000];

	strcpy(fname, orbits_restricted_fname);
	chop_off_extension(fname);

	sprintf(fname + strlen(fname), "_restricted_action.txt");
	Fio.lint_matrix_write_csv(fname,
			Perms, set_m, SG->gens->len);

}

void orbits_restricted_compute(wreath_product* W,
		strong_generators* SG,
		action* A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		const char *orbits_restricted_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_restricted_compute orbits_restricted_fname=" << orbits_restricted_fname << endl;
	}

	file_io Fio;
	sorting Sorting;

	long int *Set;
	long int *Set_in_PG;
	int set_m, set_n;
	int i;

	Fio.lint_matrix_read_csv(orbits_restricted_fname,
			Set, set_m, set_n, verbose_level);

	if (set_n != 1) {
		cout << "orbits_restricted set_n != 1" << endl;
		exit(1);
	}
	cout << "Restricting to a set of size " << set_m << endl;
	cout << "converting points to PG point labels" << endl;

	int *v;
	long int s;
	v = NEW_int(W->dimension_of_tensor_action);
	Set_in_PG = NEW_lint(set_m);
	for (i = 0; i < set_m; i++) {
		s = W->affine_rank_to_PG_rank(Set[i]);
		Set_in_PG[i] = s;
	}
	//FREE_int(v);
	Sorting.lint_vec_heapsort(Set_in_PG, set_m);
	cout << "after sorting, Set_in_PG:" << endl;
#if 0
	for (i = 0; i < set_m; i++) {
		cout << i << " : " << Set_in_PG[i] << endl;
	}
#endif



	nb_gens = SG->gens->len;


	char fname[1000];
	int *Perms;
	int perms_m, perms_n;

	strcpy(fname, orbits_restricted_fname);
	chop_off_extension(fname);

	sprintf(fname + strlen(fname), "_restricted_action.txt");
	Fio.int_matrix_read_csv(fname,
			Perms, perms_m, perms_n, verbose_level - 2);
	if (perms_n != SG->gens->len) {
		cout << "perms_n != SG->gens->len" << endl;
		exit(1);
	}
	if (perms_m != set_m) {
		cout << "perms_m != set_m" << endl;
		exit(1);
	}

	degree = perms_m;




	action *A_perm;
	action *A_perm_matrix;

	A_perm = NEW_OBJECT(action);
	A_perm->init_permutation_representation(A,
			FALSE /* f_stay_in_the_old_action */,
			SG->gens,
			Perms, degree,
			verbose_level);
	cout << "created A_perm = " << A_perm->label << endl;

	A_perm_matrix = NEW_OBJECT(action);
	A_perm_matrix->init_permutation_representation(A,
			TRUE /* f_stay_in_the_old_action */,
			SG->gens,
			Perms, degree,
			verbose_level);
	cout << "created A_perm_matrix = " << A_perm_matrix->label << endl;

	permutation_representation *Permutation_representation;

	Permutation_representation = A_perm->G.Permutation_representation;

	vector_ge *Gens;

	Gens = NEW_OBJECT(vector_ge);

	Gens->init(A_perm, verbose_level - 2);
	Gens->allocate(SG->gens->len, verbose_level - 2);
	for (i = 0; i < SG->gens->len; i++) {
		A_perm->element_move(
				Permutation_representation->Elts
					+ i * A_perm->elt_size_in_int,
				Gens->ith(i),
				verbose_level);
	}

	schreier *Sch;
	longinteger_object go;
	int orbit_idx;

	Sch = NEW_OBJECT(schreier);

	Sch->init(A_perm, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_generators(*Gens, verbose_level - 2);

	cout << "before Sch->compute_all_point_orbits" << endl;
	Sch->compute_all_point_orbits(0 /*verbose_level - 5*/);
	cout << "after Sch->compute_all_point_orbits" << endl;

	Sch->print_orbit_lengths_tex(cout);
	Sch->print_and_list_orbits_tex(cout);

	set_of_sets *Orbits;
	Sch->orbits_as_set_of_sets(Orbits, verbose_level);

	A->group_order(go);
	cout << "Action " << A->label << endl;
	cout << "group order " << go << endl;
	cout << "computing stabilizers:" << endl;



	for (orbit_idx = 0; orbit_idx < Sch->nb_orbits; orbit_idx++) {
		cout << "computing point stabilizer for orbit " << orbit_idx << ":" << endl;

		int orb_rep;
		long int orbit_rep_in_PG;
		uint32_t orbit_rep_in_PG_uint;

		orb_rep = Sch->orbit[Sch->orbit_first[orbit_idx]];

		orbit_rep_in_PG = Set_in_PG[orb_rep];

		orbit_rep_in_PG_uint = W->PG_rank_to_affine_rank(orbit_rep_in_PG);

		int *tensor;

		tensor = NEW_int(W->dimension_of_tensor_action);

		W->tensor_PG_unrank(tensor, orbit_rep_in_PG);

		cout << "orbit representative is " << orb_rep << " = " << orbit_rep_in_PG << " = " << orbit_rep_in_PG_uint << endl;
		cout << "tensor: ";
		int_vec_print(cout, tensor, W->dimension_of_tensor_action);
		cout << endl;
		sims *Stab;

		cout << "before Sch->point_stabilizer in action " << A_perm_matrix->label << endl;
		Sch->point_stabilizer(A_perm_matrix, go,
				Stab, orbit_idx, verbose_level - 5);
		cout << "after Sch->point_stabilizer in action " << A_perm_matrix->label << endl;

		strong_generators *gens;

		gens = NEW_OBJECT(strong_generators);
		gens->init(A_perm_matrix);
		gens->init_from_sims(Stab, verbose_level);


		gens->print_generators_tex(cout);

#if 1
		action *A_on_orbit;

		cout << "computing restricted action on the orbit:" << endl;
		A_on_orbit = A_perm->restricted_action(Orbits->Sets[orbit_idx] + 1, Orbits->Set_size[orbit_idx] - 1,
				verbose_level);

		cout << "generators restricted to the orbit of degree " << Orbits->Set_size[orbit_idx] - 1 << ":" << endl;
		gens->print_generators_MAGMA(A_on_orbit, cout);


		sims *derived_group;
		longinteger_object d_go;

		derived_group = NEW_OBJECT(sims);

		cout << "computing the derived subgroup:" << endl;

		derived_group->init(A_perm_matrix, verbose_level - 2);
		derived_group->init_trivial_group(verbose_level - 1);
		derived_group->build_up_subgroup_random_process(Stab,
				choose_random_generator_derived_group,
				0 /*verbose_level*/);

		derived_group->group_order(d_go);
		cout << "the derived subgroup has order: " << d_go << endl;

		strong_generators *d_gens;

		d_gens = NEW_OBJECT(strong_generators);
		d_gens->init(A_perm_matrix);
		d_gens->init_from_sims(derived_group, 0 /*verbose_level*/);


		d_gens->print_generators_tex(cout);

		schreier *Sch_orbit;

		Sch_orbit = NEW_OBJECT(schreier);
		cout << "computing orbits of stabilizer on the rest of the orbit:" << endl;

		A_on_orbit->all_point_orbits_from_generators(
				*Sch_orbit,
				gens,
				0 /* verbose_level */);

		cout << "Found " << Sch_orbit->nb_orbits << " orbits" << endl;
		Sch_orbit->print_orbit_lengths_tex(cout);
		Sch_orbit->print_and_list_orbits_tex(cout);
#endif

		FREE_OBJECT(gens);
		FREE_OBJECT(Stab);
	}
}

void wreath_product_rank_one_early_test_func_callback(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	tensor_product *T = (tensor_product *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product_rank_one_early_test_func_callback for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	T->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "wreath_product_rank_one_early_test_func_callback done" << endl;
		}
}





