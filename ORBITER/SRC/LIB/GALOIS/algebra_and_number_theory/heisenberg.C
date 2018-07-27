// heisenberg.C
//
// Anton Betten
// 
// April 27, 2017

#include "galois.h"

heisenberg::heisenberg()
{
	null();
}

heisenberg::~heisenberg()
{
	freeself();
}

void heisenberg::null()
{
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
}

void heisenberg::freeself()
{
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	if (Elt3) {
		FREE_INT(Elt3);
		}
	if (Elt4) {
		FREE_INT(Elt4);
		}
}

void heisenberg::init(finite_field *F, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "heisenberg::init n=" << n << " q=" << F->q << endl;
		}
	q = F->q;
	heisenberg::F = F;
	heisenberg::n = n;
	len = 2 * n + 1;
	group_order = i_power_j(q, len);

	Elt1 = NEW_INT(len);
	Elt2 = NEW_INT(len);
	Elt3 = NEW_INT(len);
	Elt4 = NEW_INT(len);
	if (f_v) {
		cout << "heisenberg::init done" << endl;
		}
}

void heisenberg::unrank_element(INT *Elt, INT rk)
{
	AG_element_unrank(q, Elt, 1, len, rk);
}

INT heisenberg::rank_element(INT *Elt)
{
	INT rk;
	
	AG_element_rank(q, Elt, 1, len, rk);
	return rk;
}

void heisenberg::element_add(INT *Elt1, INT *Elt2, INT *Elt3, INT verbose_level)
{
	INT a;

	F->add_vector(Elt1, Elt2, Elt3, len);
	a = F->dot_product(n, Elt1, Elt2 + n);
	Elt3[2 * n] = F->add(Elt3[2 * n], a);
}

void heisenberg::element_negate(INT *Elt1, INT *Elt2, INT verbose_level)
{
	INT a;

	F->negate_vector(Elt1, Elt2, len);
	a = F->dot_product(n, Elt1, Elt1 + n);
	Elt2[2 * n] = F->add(Elt2[2 * n], a);
}


INT heisenberg::element_add_by_rank(INT rk_a, INT rk_b, INT verbose_level)
{
	INT rk;
	
	unrank_element(Elt1, rk_a);
	unrank_element(Elt2, rk_b);
	element_add(Elt1, Elt2, Elt3, 0 /* verbose_level */);
	rk = rank_element(Elt3);
	return rk;
}

INT heisenberg::element_negate_by_rank(INT rk_a, INT verbose_level)
{
	INT rk;
	
	unrank_element(Elt1, rk_a);
	element_negate(Elt1, Elt2, 0 /* verbose_level */);
	rk = rank_element(Elt2);
	return rk;
}

void heisenberg::group_table(INT *&Table, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, k;

	if (f_v) {
		cout << "heisenberg::group_table" << endl;
		}
	Table = NEW_INT(group_order * group_order);
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

void heisenberg::group_table_abv(INT *&Table_abv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, k;

	if (f_v) {
		cout << "heisenberg::group_table" << endl;
		}
	Table_abv = NEW_INT(group_order * group_order);
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

void heisenberg::generating_set(INT *&gens, INT &nb_gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, cnt, k;

	if (f_v) {
		cout << "heisenberg::generating_set" << endl;
		}
	nb_gens = (n * F->e) * 2 + F->e;
	gens = NEW_INT(nb_gens);
	cnt = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < F->e; j++) {
			unrank_element(Elt1, 0);
			Elt1[i] = i_power_j(F->p, j);
			k = rank_element(Elt1);
			gens[cnt++] = k;
			}
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < F->e; j++) {
			unrank_element(Elt1, 0);
			Elt1[n + i] = i_power_j(F->p, j);
			k = rank_element(Elt1);
			gens[cnt++] = k;
			}
		}

	for (j = 0; j < F->e; j++) {
		unrank_element(Elt1, 0);
		Elt1[2 * n] = i_power_j(F->p, j);
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


