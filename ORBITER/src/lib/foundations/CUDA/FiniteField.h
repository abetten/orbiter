/*
 * FiniteField.h
 *
 *  Created on: Oct 25, 2018
 *      Author: sajeeb
 */


#include "chrono.h"
#include "Linalg.h"
#include "arithmetic.h"

#include <string.h>
#include <stdio.h>
#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <tuple>
#include <map>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS_PER_BLOCK 1024

#include "Matrix.h"
#include "Vector.h"
#include "PG.h"

using std::cout;
using std::endl;
using std::ostream;
using std::tuple;
using std::vector;
using std::queue;
using std::stack;

using arithmetic::mod;
using arithmetic::modinv;


#ifndef FINITE_FIELD_H_
#define FINITE_FIELD_H_


namespace FiniteField {

	using clock_value_t = long long;
	__device__ void sleep(clock_value_t sleep_cycles);


	template <typename Vec>
	void left_normalize_(Vec& x, int p)
	{
		int non_zero_entry_idx = -1;
		for (size_t i=0; i<x.size(); ++i) {
			if (mod(x(i), p) != 0) {
				non_zero_entry_idx = i;
				break;
			}
		}
		if (non_zero_entry_idx < 0) return;
		unsigned int a = x(non_zero_entry_idx);
		unsigned int b = modinv(a, p);
		for (size_t i=0; i<x.size(); ++i) {
			x(i) = mod(x(i)*b, p);
		}
	}

	template <typename Vec>
	void right_normalize_(Vector<Vec>& x, int p) {
		// last non-zero element made one
		int i, j, a;
		int stride = 1;
		int len = x.size();

		for (i = len - 1; i >= 0; i--) {
			a = x[i * stride];
			if (a) {
				if (a == 1) {
					make_num_rep(x, p);
					return;
				}
				a = modinv(a, p);
				x[i * stride] = 1;
				for (j = i - 1; j >= 0; j--) {
					x[j * stride] = mod(x[j * stride] * a, p);
				}
				make_num_rep(x, p);
				return;
			}
		}
		cout << __FILE__ << ":" << __LINE__ << endl;
		cout << "PG_element_normalize() zero vector()" << endl;
		exit(1);

	}

	template <typename Mat>
	void rref(Mat& M, int p) {
		size_t lead = 0;
		size_t nrows = M.get_nrows();
		size_t ncols = M.get_ncols();

		while (lead < nrows) {

			for (size_t r=lead; r<nrows; ++r) {
				if (M(r, lead) != 0) {
					M.swapRows(lead, r);
					break;
				}
			}

			auto d_inv = modinv(M(lead, lead), p);

			for (size_t c=lead; c<ncols; ++c) {
				M(lead, c) = mod(M(lead, c) * d_inv, p);
			}

			for (size_t r=lead+1; r<nrows; ++r) {
				auto a = M(r, lead);
				for (size_t c=lead; c<ncols; ++c) {
					auto aa = mod(M(lead, c) * a, p);
					auto bb = mod(M(r, c) - aa, p);

					// Mod aa
					// http://www.cplusplus.com/forum/general/19502/

					M(r, c) = bb;
				}
			}

			lead += 1;

		}

		for (int l=nrows-1; l>0; --l) {
			for (int r=l-1; r>-1; --r) {
				auto a = M(r, l);
				for (size_t c=l; c<ncols; ++c) {
					auto aa = mod(M(l, c) * a, p);
					auto bb = mod(M(r, c) - aa, p);
					M(r, c) = bb;
				}
			}
		}
	}

	template <typename Mat>
	void inverse(Mat& M, int p) {
		size_t m = M.nrows, n = M.ncols;

		if (m != n) {
			cout << __FILE__ << ":" << __LINE__ << ":";
			cout << "cannot perform inverse operation on non square matrix" << endl;
			exit(-1);
		}

		// Create an identity matrix
		Mat id(M.nrows, M.ncols);
		id.identity();

		// Augment the current matrix with the
		// identity matrix
		M.appendRight(id);

		// Row reduce the current matrix
		rref(M, p);

		// Extract the non identity matrix from M
		auto* n_mat_ = M.allocate_new_matrix_memory_(id.nrows * id.ncols);
		for (size_t i=0; i<M.nrows; ++i) {
			memcpy(n_mat_+i*id.ncols, &M(i, id.ncols), sizeof(M.matrix_[0]) * id.ncols);
		}

		delete [] M.matrix_;
		M.matrix_ = n_mat_;
		M.alloc_cols = id.ncols;
		M.ncols = id.ncols;
	}


	template <typename Mat>
	Mat* dot(const Mat& A, const Mat& B, int p=0) {
		size_t m = A.get_nrows();
		size_t n = A.get_ncols();
		size_t k = B.get_nrows();
		size_t l = B.get_ncols();

		if (n != k) {
			cout << __FILE__ << ":" << __LINE__ << ":";
			cout << "cannot multiply matricies, columns of A "
				 <<	"does not match rows of B." << endl;
			exit(-1);
		}

		Mat* M = new Mat(m, l);

		chrono C0;

		// Implement algorithm for matrix multiplication
		for (size_t i=0; i<m ; ++i) { // iterate over the rows of A
			for (size_t j=0; j<l; ++j) { // iterate over the cols of B
				for (size_t o=0; o<n; ++o) { // iterate over cols of A
					if (p == 0) M->operator() (i, j) += A(i, o) * B(o, j);
					else M->operator() (i, j) += mod(A(i, o) * B(o, j), p);
				}
				if (p != 0) M->operator() (i, j) = mod(M->operator() (i, j), p);
			}
		}

		return M;
	}

	template <typename Mat>
	void dot(const Mat& A, const Mat& B, Mat& M, int p=0) {
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

		Mat tmp_M(m, l);

		chrono C0;

		// Implement algorithm for matrix multiplication
		for (size_t i=0; i<m ; ++i) { // iterate over the rows of A
			for (size_t j=0; j<l; ++j) { // iterate over the cols of B
				for (size_t o=0; o<n; ++o) { // iterate over cols of A
					if (p == 0) tmp_M(i, j) += A(i, o) * B(o, j);
					else tmp_M(i, j) += mod(A(i, o) * B(o, j), p);
				}
				if (p != 0) tmp_M(i, j) = mod(tmp_M(i, j), p);
			}
		}

		M.Init(tmp_M);

	}

	template <typename Mat, typename Vec>
	Vec* dot(const Mat& M, const Vec& V) {
		size_t m = M.get_nrows();
		size_t n = M.get_ncols();
		size_t l = V.size();

		if (l != n) {
			cout << __FILE__ << ":" << __LINE__ << ":";
			cout << "Cannot perform matrix vector dot product, ";
			cout << "size of matrix column do not match size of vector." << endl;
			exit(-1);
		}

		Vec* rv = new Vec(m);

		for (size_t i=0; i<m; ++i) {
			for (size_t j=0; j<n; ++j) {
				rv->operator ()(i) += V(j) * M(i, j);
			}
		}

		return rv;
	}

	template <typename Mat>
	__host__
	Mat* cuda_dot(Mat& A, Mat& B, int q=0, int axis=0, bool ea=true, bool eb=true, bool ec=true) {
		size_t m = A.nrows;
		size_t n = A.ncols;

		size_t o = B.nrows;
		size_t p = B.ncols;

		if (n != o) {
			printf("%s:%d:Cannot perform matrix multiplication, ", __FILE__, __LINE__);
			cout << n << " " << o << endl;
			printf("size of matrix column do not match size of vector.\n");
			exit(-1);
		}

		// Host matrix C, used to store the result of
		// multiplying matrix A and B.
		Mat& C = *new Mat(m, p);

		// Copy matrix A, B and C into device memory
		Mat* d_A = A.InitializeOnGPU(true);
		Mat* d_B = B.InitializeOnGPU(true);
		Mat* d_C = C.InitializeOnGPU(true);

		// Find out how many threads are needed assuming each thread
		// works on one entry of the resultant matrix.
		int num_threads = m * p;

		// Find out how many blocks are needed
		int block_size = 16;
		int num_blocks = (num_threads + block_size*block_size - 1)/ (block_size*block_size) ;
		int gridDim_x = (C.ncols + block_size - 1) / block_size;
		int gridDim_y = (C.nrows + block_size - 1) / block_size;
		if (num_blocks > gridDim_x*gridDim_y || num_threads > gridDim_x*gridDim_y*pow(block_size,2)) {
			cout << "Error:" << __FILE__ << ":" << __LINE__ << 
			"number of required blocks is greater than number of blocks set."
			<< endl;
		}
		dim3 blockDim(block_size, block_size, 1);
		dim3 gridDim(gridDim_x, gridDim_y, 1);


		linalg::cuda_matrix_matrix_dot_<<<gridDim, blockDim>>>(*d_A, *d_B, *d_C, q, axis);
		// Do some error checking after kernel launch
		gpuErrchk( cudaGetLastError() );
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// After the matrix multiplication is done, copy the matrix into
		// host memory.
		C.copy_matrix_to_host(true);

		// Free up all space allocated on the GPU for matrix multiplication.
		if (ea) A.UninitializeOnGPU();
		if (eb) B.UninitializeOnGPU();
		if (ec) C.UninitializeOnGPU();

		return &C;
	}

	template <typename v>
	tuple<std::unordered_map<int, tuple<int, Matrix<v>*>>, size_t>
	ComputeSchreierTree(const vector<Matrix<v>*>& Generators,
						const vector<Vector<v>*>& X,
						Vector<v>* root_node, // This is assuming that the root node is in X
						int n,
						int q,
						int device = 0) {
		// Generators contains the set of generators
		// X is the set of elements on which the generators of the generating set
		// acts on


		// Put all the matrices in the generating set in a tall matrix
		Matrix<v> M(*Generators[0]);
		for (size_t i=1; i<Generators.size(); ++i)
			M.appendBottom(*Generators[i]);

		M.reset_dimensions();

		// Put all the vectors in the input set X in a column matrix
		Matrix<v> V(X.data(), X.size(), 0);

		// vector<Vec*> nodes_vector;
		Matrix<v>* C = NULL;

		size_t nrows_C = M.nrows, ncols_C = V.ncols;
		if (nrows_C*ncols_C < 1000) device = 0;

		if (device == 0) {
			C = FiniteField::dot(M, V, q);
		} else if (device == 1) {
			C = FiniteField::cuda_dot(M, V, q);
		}


		//---------------------------------------------------------------------------
		// Turn every column vector of C into n vectors
		//---------------------------------------------------------------------------
		// for (size_t col=0; col<C->ncols; ++col) {
		// 	for (size_t row=0; row<C->nrows; row += n) {
		// 		Vec* vec = new Vec (n);
		// 		for (size_t elt=row, i=0; elt<row+n; ++elt, ++i) {
		// 			vec->operator ()(i) = C->operator ()(elt, col);
		// 		}
		// 		vec->make_num_rep(q);
		// 		vec->make_str_rep();
		// 		nodes_vector.push_back(vec);
		// 	}
		// }
		//---------------------------------------------------------------------------


		//---------------------------------------------------------------------------
		// Finally generate the schreier tree
		//---------------------------------------------------------------------------

		// queue containing a list of unexpanded tuples. A queue is used
		// instead of a vector as inserting to the beginning of a queue
		// is constant time, while inserting an element at the beginning
		// of a vector is O(n) time.
		queue<Vector<v>*> unexpanded;

		// The unexpanded_map is here for two reasons: get O(1) lookup
		// time for unexpanded vector, the lookup is only there to check
		// if an entry exists in the unexpanded queue, and to get the
		// parent, edge_label and cost of the numerical representation of
		// the nodes in the unexpanded vector.
		std::unordered_map<int, tuple<Vector<v>*, Matrix<v>*, size_t>> unexpanded_map;

		// This is to be returned at the end of the function. The tree_info
		// map contains a map of the pointer of the vector object with a tuple
		// containing the parent of the vector object and the edge_label between
		// the parent and the vector object.
		std::unordered_map<int, tuple<int, Matrix<v>*>> tree_info;

		// Add the first node to the unexpanded vector and unexpanded_map
		unexpanded_map[root_node->num_rep()] = std::make_tuple((Vector<v>*)NULL, (Matrix<v>*)NULL, 1);
		unexpanded.push(root_node);

		// The expanded set stores the nodes in the partial schreier tree
		std::unordered_set<int> expanded;
		expanded.reserve(5000);

		// This is to be returned at the end of the function call. This variable stores
		// a value that is similar to the cost of generating the tree, in that at every
		// layer, and for every node per layer word_length += 1 is accumulated in this
		// variable.
		size_t word_length = 0;

		vector<Vector<v>*> vector_pointers;

		while (!unexpanded.empty()) {
			// extract the last node from the unexpanded vector, and its parent,
			// edge_label, and cost from the unexpanded_map.
			Vector<v>* node   =  unexpanded.front();
			Vector<v>* parent =  std::get<0>(unexpanded_map.at(node->num_rep()));
			Matrix<v>* lbl    =  std::get<1>(unexpanded_map.at(node->num_rep()));
			int cost    =  std::get<2>(unexpanded_map.at(node->num_rep()));


			// remove the extracted entry from the unexpanded list and from
			// the unexpanded map as unexpanded vector and unexpanded_map are
			// similar in that if an entry is in the unexpanded vector, it will
			// be in the unexpanded_map, if an entry is not in the unexpanded
			// vector, it will not be in the unexpanded_map.
			unexpanded.pop();
			unexpanded_map.erase(node->num_rep());

			// add the extracted entry to the expanded map
			expanded.emplace(node->num_rep());

			// create an entry in tree_info with node as key, and a tuple
			// containing its parent, and the edge_label between the parent
			// and the node, If the node is the root node, then put -1 as its
			// parent, else find the numerical representation of the parent
			// and put that in as the parent of the current node.
			if (parent) tree_info[node->num_rep()] = std::make_tuple(parent->num_rep(), lbl);
			else        tree_info[node->num_rep()] = std::make_tuple(-1, lbl);


			// add the cost of generating this node to word_length
			word_length += cost;

			// Find all the children of node
			for (size_t row=0, g=0, col=node->num_rep(); row<C->nrows; row += n, ++g) {
				Vector<v>* child = new Vector<v> (n);
				Matrix<v>* generator = Generators[g];
				for (size_t elt=row, i=0; elt<row+n; ++elt, ++i) {
					child->operator ()(i) = C->operator ()(elt, col);
				}

				FiniteField::right_normalize_(*child, q);

//				cout << node->num_rep() << ":" << child->num_rep() << ":" << generator->name << endl;


				// check to see if child is in children, expanded map, or
				// in unexpanded map. unexpanded_map contains the exact
				// same entries as unexpanded vector.
				if (expanded.find(child->num_rep())!=expanded.end() ||
					unexpanded_map.find(child->num_rep())!=unexpanded_map.end()) {

					// if the child is in either one of unexpanded, expanded
					// or children, remove the current child, and find the
					// next child.
					delete child;

				} else {
					// if the child is not in unexpanded, expanded or children
					// add child to children, add child to unexpanded vector
					// and map.
					unexpanded.push(child); // append at the beginning because we are doing BFS
					unexpanded_map[child->num_rep()] = std::make_tuple(node, generator, cost+1);
				}
			}
			if (node != root_node) vector_pointers.push_back(node);
		}

		//---------------------------------------------------------------------------


		//---------------------------------------------------------------------------
		// Free memory
		//---------------------------------------------------------------------------
		delete C;
		for (Vector<v>* ptr : vector_pointers) {
			delete ptr;
		}
		//---------------------------------------------------------------------------
		return std::make_tuple(tree_info, word_length);

	}

}

#endif /* FINITE_FIELD_H_ */
