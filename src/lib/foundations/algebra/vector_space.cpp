// vector_space.cpp
//
// Anton Betten
//
// started:  December 2, 2018




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


vector_space::vector_space()
{
	null();
}

vector_space::~vector_space()
{
	freeself();
}

void vector_space::null()
{
	dimension = 0;
	F = NULL;

	rank_point_func = NULL;
	unrank_point_func = NULL;
	rank_point_data = NULL;
	v1 = NULL;
	base_cols = NULL;
	base_cols2 = NULL;
	M1 = NULL;
	M2 = NULL;
}

void vector_space::freeself()
{
	if (v1) {
		FREE_int(v1);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (base_cols2) {
		FREE_int(base_cols2);
	}
	if (M1) {
		FREE_int(M1);
	}
	if (M2) {
		FREE_int(M2);
	}
}

void vector_space::init(finite_field *F, int dimension,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_space::init q=" << F->q
				<< " dimension=" << dimension << endl;
		}
	vector_space::F = F;
	vector_space::dimension = dimension;
	v1 = NEW_int(dimension);
	base_cols = NEW_int(dimension);
	base_cols2 = NEW_int(dimension);
	rank_point_func = vector_space_rank_point_callback;
	unrank_point_func = vector_space_unrank_point_callback;
	rank_point_data = this;
	M1 = NEW_int(dimension * dimension);
	M2 = NEW_int(dimension * dimension);
	if (f_v) {
		cout << "vector_space::init done" << endl;
		}
}

void vector_space::init_rank_functions(
	long int (*rank_point_func)(int *v, void *data),
	void (*unrank_point_func)(int *v, long int rk, void *data),
	void *data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_space::init_rank_functions" << endl;
		}
	vector_space::rank_point_func = rank_point_func;
	vector_space::unrank_point_func = unrank_point_func;
	vector_space::rank_point_data = data;
	if (f_v) {
		cout << "vector_space::init_rank_functions done" << endl;
		}
}

void vector_space::unrank_basis(int *Mtx, long int *set, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		unrank_point(Mtx + i * dimension, set[i]);
	}
}

void vector_space::rank_basis(int *Mtx, long int *set, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		set[i] = rank_point(Mtx + i * dimension);
	}
}

void vector_space::unrank_point(int *v, long int rk)
{
	if (unrank_point_func) {
		(*unrank_point_func)(v, rk, rank_point_data);
	}
	else {
		F->PG_element_unrank_modified_lint(v, 1, dimension, rk);
	}
}

long int vector_space::rank_point(int *v)
{
	long int rk;

	if (rank_point_func) {
		rk = (*rank_point_func)(v, rank_point_data);
	}
	else {
		F->PG_element_rank_modified_lint(v, 1, dimension, rk);
	}
	return rk;
}


int vector_space::RREF_and_rank(int *basis, int k)
{
	int rk;

	rk = F->Linear_algebra->Gauss_simple(
		basis, k, dimension,
		base_cols, 0 /* verbose_level */);
	return rk;
}

int vector_space::is_contained_in_subspace(int *v, int *basis, int k)
{
	int ret;

	ret = F->Linear_algebra->is_contained_in_subspace(k,
					dimension, basis, base_cols,
					v, 0 /*verbose_level*/);
	return ret;
}

int vector_space::compare_subspaces_ranked(
		long int *set1, long int *set2, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r, i;
	int rk1, rk2;

	if (f_v) {
		cout << "vector_space::compare_subspaces_ranked" << endl;
	}
#if 0
	r = F->compare_subspaces_ranked_with_unrank_function(
				set1, set2, k,
				dimension,
				unrank_point_func,
				rank_point_data,
				verbose_level);
#else
	if (k > dimension) {
		cout << "vector_space::compare_subspaces_ranked "
				"k > dimension" << endl;
		exit(1);
	}
	unrank_basis(M1, set1, k);
	unrank_basis(M2, set2, k);
	if (f_v) {
		cout << "matrix1:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M1, k,
				dimension, dimension,
				F->log10_of_q);
		cout << "matrix2:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M2, k,
				dimension, dimension,
				F->log10_of_q);
		}
	rk1 = F->Linear_algebra->Gauss_simple(M1, k,
			dimension, base_cols,
			0/*int verbose_level*/);
	rk2 = F->Linear_algebra->Gauss_simple(M2, k,
			dimension, base_cols2,
			0/*int verbose_level*/);
	if (f_v) {
		cout << "vector_space::compare_subspaces_ranked "
				"after Gauss" << endl;
		cout << "matrix1:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M1, k,
				dimension, dimension,
				F->log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		Orbiter->Int_vec.print(cout, base_cols, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M2, k,
				dimension, dimension,
				F->log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		Orbiter->Int_vec.print(cout, base_cols2, rk2);
		cout << endl;
		}
	if (rk1 != rk2) {
		if (f_v) {
			cout << "vector_space::compare_subspaces_ranked "
					"the ranks differ, "
					"so the subspaces are not equal, "
					"we return 1" << endl;
			}
		r = 1;
		goto ret;
		}
	for (i = 0; i < rk1; i++) {
		if (base_cols[i] != base_cols2[i]) {
			if (f_v) {
				cout << "the base_cols differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
				}
			r = 1;
			goto ret;
			}
		}
	for (i = 0; i < k * dimension; i++) {
		if (M1[i] != M2[i]) {
			if (f_v) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
				}
			r = 1;
			goto ret;
			}
		}
	if (f_v) {
		cout << "the subspaces are equal, we return 0" << endl;
		}
	r = 0;
#endif
	if (f_v) {
		cout << "vector_space::compare_subspaces_ranked "
				"done" << endl;
	}
ret:
	return r;
}

// #############################################################################
// global functions:
// #############################################################################


void vector_space_unrank_point_callback(int *v, long int rk, void *data)
{
	vector_space *VS = (vector_space *) data;

	VS->F->PG_element_unrank_modified_lint(v, 1, VS->dimension, rk);

}

long int vector_space_rank_point_callback(int *v, void *data)
{
	vector_space *VS = (vector_space *) data;
	long int rk;

	VS->F->PG_element_rank_modified_lint(v, 1, VS->dimension, rk);
	return rk;

}

}}


