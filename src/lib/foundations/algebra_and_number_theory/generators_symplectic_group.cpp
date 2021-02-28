// generators_symplectic_group.cpp
//
// Anton Betten
// March 29, 2016

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



generators_symplectic_group::generators_symplectic_group()
{
	null();
}

generators_symplectic_group::~generators_symplectic_group()
{
	freeself();
}

void generators_symplectic_group::null()
{
	nb_candidates = NULL;
	cur_candidate = NULL;
	candidates = NULL;
	Mtx = NULL;
	v = NULL;
	v2 = NULL;
	w = NULL;
	Points = NULL;
	nb_gens = 0;
	Data = NULL;
	transversal_length = NULL;
}

void generators_symplectic_group::freeself()
{
	int i;
	
	if (nb_candidates) {
		FREE_int(nb_candidates);
		}
	if (cur_candidate) {
		FREE_int(cur_candidate);
		}
	if (candidates) {
		for (i = 0; i < n + 1; i++) {
			FREE_int(candidates[i]);
			}
		FREE_pint(candidates);
		}
	if (Mtx) {
		FREE_int(Mtx);
		}
	if (v) {
		FREE_int(v);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (w) {
		FREE_int(w);
		}
	if (Points) {
		FREE_int(Points);
		}
	if (Data) {
		FREE_int(Data);
		}
	if (transversal_length) {
		FREE_int(transversal_length);
		}
	null();
}

void generators_symplectic_group::init(finite_field *F,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "generators_symplectic_group::init" << endl;
		}
	generators_symplectic_group::F = F;
	generators_symplectic_group::n = n;
	if (ODD(n)) {
		cout << "generators_symplectic_group::init "
				"n must be even" << endl;
		exit(1);
		}
	n_half = n >> 1;
	q = F->q;
	qn = NT.i_power_j(q, n);
	nb_candidates = NEW_int(n + 1);
	cur_candidate = NEW_int(n);
	candidates = NEW_pint(n + 1);
	for (i = 0; i < n + 1; i++) {
		candidates[i] = NEW_int(qn);
		}

	Mtx = NEW_int(n * n);
	v = NEW_int(n);
	v2 = NEW_int(n);
	w = NEW_int(n);
	Points = NEW_int(qn * n);
	for (i = 0; i < qn; i++) {
		Gg.AG_element_unrank(q, Points + i * n, 1, n, i);
		}

	create_first_candidate_set(verbose_level);

	if (f_v) {
		cout << "first candidate set has size "
				<< nb_candidates[0] << endl;
		}

	//backtrack_search(0 /* depth */, verbose_level);
	


	int first_moved = n;
	int nb;

	nb_gens = 0;
	first_moved = n;
	transversal_length = NEW_int(n);
	for (i = 0; i < n; i++) {
		transversal_length[i] = 1;
		}
	count_strong_generators(nb_gens,
			transversal_length, first_moved, 0, verbose_level);

	if (f_v) {
		cout << "We found " << nb_gens << " strong generators" << endl;
		cout << "transversal_length = ";
		Orbiter->Int_vec.print(cout, transversal_length, n);
		cout << endl;
		cout << "group order: ";
		print_longinteger_after_multiplying(cout, transversal_length, n);
		cout << endl;
		}	

	Data = NEW_int(nb_gens * n * n);

	nb = 0;
	first_moved = n;
	get_strong_generators(Data, nb, first_moved, 0, verbose_level);

	if (nb != nb_gens) {
		cout << "nb != nb_gens" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "The strong generators are:" << endl;
		for (i = 0; i < nb_gens; i++) {
			cout << "generator " << i << " / " << nb_gens << ":" << endl;
			int_matrix_print(Data + i * n * n, n, n);
			}
		}


	if (f_v) {
		cout << "generators_symplectic_group::init done" << endl;
		}
}

int generators_symplectic_group::count_strong_generators(int &nb,
		int *transversal_length, int &first_moved, int depth,
		int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int a;
	
	if (depth == n) {
		//cout << "solution " << nb << endl;
		//int_matrix_print(Mtx, n, n);
		if (first_moved < n) {
			transversal_length[first_moved]++;
			}
		nb++;
		return FALSE;
		}
	for (cur_candidate[depth] = 0;
			cur_candidate[depth] < nb_candidates[depth];
			cur_candidate[depth]++) {
		if (cur_candidate[depth] && depth < first_moved) {
			first_moved = depth;
			}	
		a = candidates[depth][cur_candidate[depth]];
		if (FALSE) {
			cout << "depth " << depth << " " << cur_candidate[depth]
				<< " / " << nb_candidates[depth] << " which is " << a << endl;
			}
		Orbiter->Int_vec.copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		if (!count_strong_generators(nb, transversal_length,
			first_moved, depth + 1, verbose_level)
			&& depth > first_moved) {
			return FALSE;
			}
		}
	return TRUE;
}

int generators_symplectic_group::get_strong_generators(int *Data,
		int &nb, int &first_moved, int depth, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int a;
	
	if (depth == n) {
		//cout << "solution " << nb << endl;
		//int_matrix_print(Mtx, n, n);
		Orbiter->Int_vec.copy(Mtx, Data + nb * n * n, n * n);
		nb++;
		return FALSE;
		}
	for (cur_candidate[depth] = 0;
			cur_candidate[depth] < nb_candidates[depth];
			cur_candidate[depth]++) {
		if (cur_candidate[depth] && depth < first_moved) {
			first_moved = depth;
			}	
		a = candidates[depth][cur_candidate[depth]];
		if (FALSE) {
			cout << "depth " << depth << " " << cur_candidate[depth]
				<< " / " << nb_candidates[depth] << " which is " << a << endl;
			}
		Orbiter->Int_vec.copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		if (!get_strong_generators(Data, nb, first_moved,
				depth + 1, verbose_level) && depth > first_moved) {
			return FALSE;
			}
		}
	return TRUE;
}

void generators_symplectic_group::create_first_candidate_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb;

	if (f_v) {
		cout << "generators_symplectic_group::create_first_"
				"candidate_set" << endl;
		}
	nb = 0;
	// skip over the zero vector:
	for (i = 1; i < qn; i++) {
		candidates[0][nb++] = i;
		}
	nb_candidates[0] = nb;
	
	if (f_v) {
		cout << "generators_symplectic_group::create_first_"
				"candidate_set done" << endl;
		}
}

void generators_symplectic_group::create_next_candidate_set(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, ai, nb;

	if (f_v) {
		cout << "generators_symplectic_group::create_next_"
				"candidate_set level=" << level << endl;
		}
	nb = 0;

	if (EVEN(level)) {

		Orbiter->Int_vec.copy(Mtx + level * n, v, n);

		for (i = 0; i < nb_candidates[level]; i++) {
			ai = candidates[level][i];
			Orbiter->Int_vec.copy(Points + ai * n, w, n);
			if (dot_product(v, w) == 1) {
				candidates[level + 1][nb++] = ai;
				}
			}
		}
	else {

		Orbiter->Int_vec.copy(Mtx + (level - 1) * n, v, n);
		Orbiter->Int_vec.copy(Mtx + level * n, v2, n);

		for (i = 0; i < nb_candidates[level - 1]; i++) {
			ai = candidates[level - 1][i];
			Orbiter->Int_vec.copy(Points + ai * n, w, n);
			if (dot_product(v, w) == 0 && dot_product(v2, w) == 0) {
				candidates[level + 1][nb++] = ai;
				}
			}
		}
	nb_candidates[level + 1] = nb;
	
	if (f_v) {
		cout << "generators_symplectic_group::create_next_"
				"candidate_set done, found " << nb_candidates[level + 1]
				<< " candidates at level " << level + 1 << endl;
		}
}


int generators_symplectic_group::dot_product(int *u1, int *u2)
{
	int c;
	int i;

	c = 0;
	for (i = 0; i < n_half; i++) {
		c = F->add(c, F->mult(u1[2 * i + 0], u2[2 * i + 1]));
		c = F->add(c, F->negate(F->mult(u1[2 * i + 1], u2[2 * i + 0])));
		}
	return c;
}

}
}

