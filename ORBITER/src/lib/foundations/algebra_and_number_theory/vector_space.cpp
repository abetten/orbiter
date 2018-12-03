// vector_space.C
//
// Anton Betten
//
// started:  December 2, 2018




#include "foundations.h"


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
}

void vector_space::freeself()
{
	if (v1) {
		FREE_int(v1);
	}
	if (base_cols) {
		FREE_int(base_cols);
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
	if (f_v) {
		cout << "vector_space::init done" << endl;
		}
}

void vector_space::init_rank_functions(
	int (*rank_point_func)(int *v, void *data),
	void (*unrank_point_func)(int *v, int rk, void *data),
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

void vector_space::unrank_point(int *v, int rk)
{
	if (unrank_point_func) {
		(*unrank_point_func)(v, rk, rank_point_data);
	}
	else {
		F->PG_element_unrank_modified(v, 1, dimension, rk);
	}
}

int vector_space::rank_point(int *v)
{
	int rk;

	if (rank_point_func) {
		rk = (*rank_point_func)(v, rank_point_data);
	}
	else {
		F->PG_element_rank_modified(v, 1, dimension, rk);
	}
	return rk;
}


int vector_space::RREF_and_rank(int *basis, int k)
{
	int rk;

	rk = F->Gauss_simple(
		basis, k, dimension,
		base_cols, 0 /* verbose_level */);
	return rk;
}

int vector_space::is_contained_in_subspace(int *v, int *basis, int k)
{
	int ret;

	ret = F->is_contained_in_subspace(k,
					dimension, basis, base_cols,
					v, 0 /*verbose_level*/);
	return ret;
}

int vector_space::compare_subspaces_ranked(
		int *set1, int *set2, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "vector_space::compare_subspaces_ranked" << endl;
	}
	ret = F->compare_subspaces_ranked_with_unrank_function(
				set1, set2, k,
				dimension,
				unrank_point_func,
				rank_point_data,
				verbose_level);
	if (f_v) {
		cout << "vector_space::compare_subspaces_ranked done" << endl;
	}
	return ret;
}

