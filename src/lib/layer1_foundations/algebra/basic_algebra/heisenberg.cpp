// heisenberg.cpp
//
// Anton Betten
// 
// April 27, 2017

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace basic_algebra {


heisenberg::heisenberg()
{
	Record_birth();
	q = 0;
	F = NULL;
	n = 0;
	len = 0;
	group_order = 0;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
}

heisenberg::~heisenberg()
{
	Record_death();
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
}

void heisenberg::init(
		field_theory::finite_field *F,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "heisenberg::init n=" << n << " q=" << F->q << endl;
	}
	q = F->q;
	heisenberg::F = F;
	heisenberg::n = n;
	len = 2 * n + 1;
	group_order = NT.i_power_j(q, len);

	Elt1 = NEW_int(len);
	Elt2 = NEW_int(len);
	Elt3 = NEW_int(len);
	Elt4 = NEW_int(len);
	if (f_v) {
		cout << "heisenberg::init done" << endl;
	}
}

void heisenberg::unrank_element(
		int *Elt, long int rk)
{
	geometry::other_geometry::geometry_global Gg;
	Gg.AG_element_unrank(q, Elt, 1, len, rk);
}

long int heisenberg::rank_element(
		int *Elt)
{
	long int rk;
	geometry::other_geometry::geometry_global Gg;
	
	rk = Gg.AG_element_rank(q, Elt, 1, len);
	return rk;
}

void heisenberg::element_add(
		int *Elt1, int *Elt2, int *Elt3, int verbose_level)
{
	int a;

	F->Linear_algebra->add_vector(Elt1, Elt2, Elt3, len);
	a = F->Linear_algebra->dot_product(n, Elt1, Elt2 + n);
	Elt3[2 * n] = F->add(Elt3[2 * n], a);
}

void heisenberg::element_negate(
		int *Elt1, int *Elt2, int verbose_level)
{
	int a;

	F->Linear_algebra->negate_vector(Elt1, Elt2, len);
	a = F->Linear_algebra->dot_product(n, Elt1, Elt1 + n);
	Elt2[2 * n] = F->add(Elt2[2 * n], a);
}


int heisenberg::element_add_by_rank(
		int rk_a, int rk_b, int verbose_level)
{
	int rk;
	
	unrank_element(Elt1, rk_a);
	unrank_element(Elt2, rk_b);
	element_add(Elt1, Elt2, Elt3, 0 /* verbose_level */);
	rk = rank_element(Elt3);
	return rk;
}

int heisenberg::element_negate_by_rank(
		int rk_a, int verbose_level)
{
	int rk;
	
	unrank_element(Elt1, rk_a);
	element_negate(Elt1, Elt2, 0 /* verbose_level */);
	rk = rank_element(Elt2);
	return rk;
}

void heisenberg::group_table(
		int *&Table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, k;

	if (f_v) {
		cout << "heisenberg::group_table" << endl;
	}
	Table = NEW_int(group_order * group_order);
	for (i = 0; i < group_order; i++) {
		unrank_element(Elt1, i);
		for (j = 0; j < group_order; j++) {
			unrank_element(Elt2, j);
			element_add(Elt1, Elt2, Elt3, 0 /* verbose_level */);
			k = rank_element(Elt3);
			Table[i * group_order + j] = k;
		}
	}
	if (f_v) {
		cout << "heisenberg::group_table finished" << endl;
	}
}

void heisenberg::group_table_abv(
		int *&Table_abv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, k;

	if (f_v) {
		cout << "heisenberg::group_table" << endl;
	}
	Table_abv = NEW_int(group_order * group_order);
	for (i = 0; i < group_order; i++) {
		unrank_element(Elt1, i);
		for (j = 0; j < group_order; j++) {
			unrank_element(Elt2, j);
			element_negate(Elt2, Elt3, 0);
			element_add(Elt1, Elt3, Elt4, 0 /* verbose_level */);
			k = rank_element(Elt4);
			Table_abv[i * group_order + j] = k;
		}
	}
	if (f_v) {
		cout << "heisenberg::group_table finished" << endl;
	}
}

void heisenberg::generating_set(
		int *&gens, int &nb_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, cnt, k;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "heisenberg::generating_set" << endl;
	}
	nb_gens = (n * F->e) * 2 + F->e;
	gens = NEW_int(nb_gens);
	cnt = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < F->e; j++) {
			unrank_element(Elt1, 0);
			Elt1[i] = NT.i_power_j(F->p, j);
			k = rank_element(Elt1);
			gens[cnt++] = k;
		}
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < F->e; j++) {
			unrank_element(Elt1, 0);
			Elt1[n + i] = NT.i_power_j(F->p, j);
			k = rank_element(Elt1);
			gens[cnt++] = k;
		}
	}

	for (j = 0; j < F->e; j++) {
		unrank_element(Elt1, 0);
		Elt1[2 * n] = NT.i_power_j(F->p, j);
		k = rank_element(Elt1);
		gens[cnt++] = k;
	}

	if (cnt != nb_gens) {
		cout << "heisenberg::generating_set cnt != nb_gens" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "heisenberg::generating_set finished" << endl;
	}
}

}}}}


