
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

#if 0
	cout << "tensor_product::init before "
			"A->init_wreath_product_group_and_restrict" << endl;
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
#else
	cout << "tensor_product::init before "
			"A->init_wreath_product_group" << endl;
	A->init_wreath_product_group(nb_factors, n,
			F,
			verbose_level);
	cout << "tensor_product::init after "
			"A->init_wreath_product_group" << endl;
	A0 = A;
	W = A0->G.wreath_product_group;
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
//	Gen = NEW_OBJECT(poset_classification);
//
//	Gen->read_arguments(argc, argv, 0);
//
//	//Gen->prefix[0] = 0;
//	sprintf(Gen->fname_base, "wreath_%d_%d_%d", nb_factors, n, q);
//
//
//	Gen->depth = depth;
//
//	VS = NEW_OBJECT(vector_space);
//	VS->init(F,
//			vector_space_dimension /* dimension */,
//			verbose_level - 1);
//	VS->init_rank_functions(
//			wreath_rank_point_func,
//			wreath_unrank_point_func,
//			this,
//			verbose_level - 1);
//
//
//	Poset = NEW_OBJECT(poset);
//	Poset->init_subspace_lattice(
//			A0, A,
//			SG,
//			VS,
//			verbose_level);
//
//	if (f_v) {
//		cout << "tensor_product::init before Gen->init" << endl;
//		}
//	Gen->init(Poset, Gen->depth /* sz */, verbose_level);
//	if (f_v) {
//		cout << "tensor_product::init after Gen->init" << endl;
//		}
//
//
//	Gen->f_print_function = TRUE;
//	Gen->print_function = wreath_product_print_set;
//	Gen->print_function_data = this;
//
//	int nb_nodes = 1000;
//
//	if (f_v) {
//		cout << "tensor_product::init "
//				"before Gen->init_poset_orbit_node" << endl;
//		}
//	Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
//	if (f_v) {
//		cout << "tensor_product::init "
//				"calling Gen->init_root_node" << endl;
//		}
//	Gen->root[0].init_root_node(Gen, verbose_level - 1);
//
//	//int schreier_depth;
//	int f_use_invariant_subset_if_available;
//	int f_debug;
//
//	//schreier_depth = Gen->depth;
//	f_use_invariant_subset_if_available = TRUE;
//	f_debug = FALSE;
//
//	//int t0 = os_ticks();
//
//	if (f_v) {
//		cout << "tensor_product::init before Gen->main" << endl;
//		cout << "A=";
//		A->print_info();
//		cout << "A0=";
//		A0->print_info();
//		}
//
//
//	//Gen->f_allowed_to_show_group_elements = TRUE;
//
//	Gen->main(t0,
//		Gen->depth,
//		f_use_invariant_subset_if_available,
//		f_debug,
//		verbose_level);
//
//	set_of_sets *SoS;
//
//	SoS = Gen->Schreier_vector_handler->get_orbits_as_set_of_sets(
//			Gen->root[0].Schreier_vector, verbose_level);
//
//	SoS->sort_all(verbose_level);
//	cout << "orbits at level 1:" << endl;
//	SoS->print_table();
//
//	for (i = 0; i < SoS->nb_sets; i++) {
//		cout << "Orbit " << i << " has size " << SoS->Set_size[i] << " : ";
//		int_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
//		cout << endl;
//		for (j = 0; j < SoS->Set_size[i]; j++) {
//			a = SoS->Sets[i][j];
//			cout << j << " : " << a << " : ";
//			F->PG_element_unrank_modified(v, 1, vector_space_dimension, a);
//			int_vec_print(cout, v, vector_space_dimension);
//			cout << endl;
//		}
//	}
//
//	if (f_v) {
//		cout << "tensor_product::init after Gen->main" << endl;
//	}
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


//template <typename T = int>
uint32_t root (uint32_t* S, uint32_t i) {
	while (S[i] != i) i = S[i];
	return i;
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



	// result is the ranks of the images.
	// Each row of result is a permutation of the points of projective space
	// So, result is SG->gens->len x W->degree_of_tensor_action

	//result = NEW_int(SG->gens->len * W->degree_of_tensor_action);

	// perform the parallel matrix multiplication on the GPU:


//	int* v = NEW_int (MN.ncols);

	unsigned int w = (unsigned int) W->degree_of_tensor_action;
	int a;
	a = (int) w;
	if (a != W->degree_of_tensor_action) {
		cout << "W->degree_of_tensor_action does not fit into a unsigned int" << endl;
		exit(1);
	}
	else {
		cout << "W->degree_of_tensor_action fits into a unit32_t, this is good" << endl;
	}

	cout << "allocating S, an uint32_t array of size " << W->degree_of_tensor_action << endl;

	unsigned int* S = new unsigned int [W->degree_of_tensor_action];

	cout << "allocating T, an uint32_t array of size " << W->degree_of_tensor_action << endl;

	unsigned int* T = new unsigned int [W->degree_of_tensor_action];

	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	cout << "block_size=" << block_size << endl;

	int nb_blocks = (W->degree_of_tensor_action + block_size - 1) / block_size;

	cout << "nb_blocks=" << nb_blocks << endl;
//	memset(S, -1, sizeof(S)*W->degree_of_tensor_action);


	for (unsigned int i=0; i<W->degree_of_tensor_action; ++i) S[i] = i;



	for (size_t h=0; h < N.size(); ++h) {
		cout << "hh=" << h << endl;

		for (size_t b=0; b<nb_blocks; ++b) {
			cout << "b=" << b << endl;

			const size_t l = std::min((b + 1) * block_size,
										(unsigned long)W->degree_of_tensor_action) - b*block_size;
			cout << "l=" << l << endl;

			linalg::Matrix<char> M  (l, mtx_n);
			linalg::Matrix<char> MN (l, mtx_n);

			cout << "unranking the elements of the PG" << endl;

			int l1 = l / 100;
			for (size_t i=0; i<l; ++i) {
				if ((i % l1) == 0) {
					cout << "h=" << h << ", b=" << b << ", " << i/l1 << " % done unranking" << endl;
				}
				W->F->PG_element_unrank_modified_lint (v.matrix_, 1, mtx_n,
						(long int) b * (long int) block_size + (long int)i) ;
				for (size_t j=0; j<mtx_n; ++j)
					M(i,j) = v(j, 0);
			}
			cout << "unranking the elements of the PG done" << endl;


			// Matrix Multiply
			MN.reset_entries();
//			linalg::cpu_mod_mat_mul_AB (M, N[h], MN, W->q);


			cout << "cuda multiplication" << endl;

			linalg::cuda_mod_mat_mul (M, N[h], MN, W->q);

			cout << "cuda multiplication done" << endl;

			M.UninitializeOnGPU();
			N[h].UninitializeOnGPU();
			MN.UninitializeOnGPU();


			cout << "ranking the elements of the PG" << endl;
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
				T [b * block_size + i] = (unsigned int) res;
			}
			cout << "ranking the elements of the PG done" << endl;

		}

		for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
			unsigned int t = T[i];
			unsigned int r1 = root(S, i);
			unsigned int r2 = root(S, t);

			if (r1 != r2) {
				if (r1 < r2) S[r2] = r1; else S[r1] = r2;
			}
		}

	}

	int nb_orbits = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) ++nb_orbits;
	}
	printf("nb_orbits: %d\n", nb_orbits);
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
#else
	nb_gens = 0;
	degree = 0;
#endif
}


