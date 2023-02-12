// null_polarity_generator.cpp
//
// Anton Betten
// December 11, 2015

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {


null_polarity_generator::null_polarity_generator()
{
	F = NULL;
	n = q = 0;
	qn = 0;
	nb_candidates = NULL;
	cur_candidate = NULL;
	candidates = NULL;
	Mtx = NULL;
	v = NULL;
	w = NULL;
	Points = NULL;
	nb_gens = 0;
	Data = NULL;
	transversal_length = NULL;
}

null_polarity_generator::~null_polarity_generator()
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
}

void null_polarity_generator::init(
		field_theory::finite_field *F,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "null_polarity_generator::init" << endl;
	}
	null_polarity_generator::F = F;
	null_polarity_generator::n = n;
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
	w = NEW_int(n);
	Points = NEW_int(qn * n);
	for (i = 0; i < qn; i++) {
		Gg.AG_element_unrank(q, Points + i * n, 1, n, i);
	}

	create_first_candidate_set(verbose_level);

	if (f_v) {
		cout << "first candidate set has size " << nb_candidates[0] << endl;
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
	count_strong_generators(nb_gens, transversal_length,
			first_moved, 0, verbose_level);

	if (f_v) {
		cout << "We found " << nb_gens << " strong generators" << endl;
		cout << "transversal_length = ";
		Int_vec_print(cout, transversal_length, n);
		cout << endl;
		cout << "group order: ";

		ring_theory::ring_theory_global R;

		R.print_longinteger_after_multiplying(cout, transversal_length, n);
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
			Int_matrix_print(Data + i * n * n, n, n);
		}
	}


	if (f_v) {
		cout << "null_polarity_generator::init done" << endl;
	}
}

int null_polarity_generator::count_strong_generators(
		int &nb, int *transversal_length, int &first_moved,
		int depth, int verbose_level)
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
				<< " / " << nb_candidates[depth]
				<< " which is " << a << endl;
		}
		Int_vec_copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		if (!count_strong_generators(nb, transversal_length,
				first_moved, depth + 1, verbose_level) &&
				depth > first_moved) {
			return FALSE;
		}
	}
	return TRUE;
}

int null_polarity_generator::get_strong_generators(
		int *Data, int &nb, int &first_moved, int depth,
		int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int a;
	
	if (depth == n) {
		//cout << "solution " << nb << endl;
		//int_matrix_print(Mtx, n, n);
		Int_vec_copy(Mtx, Data + nb * n * n, n * n);
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
				<< " / " << nb_candidates[depth] << " which is "
				<< a << endl;
		}
		Int_vec_copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		if (!get_strong_generators(Data, nb, first_moved,
				depth + 1, verbose_level) && depth > first_moved) {
			return FALSE;
		}
	}
	return TRUE;
}

void null_polarity_generator::backtrack_search(
		int &nb_sol, int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;
	
	if (depth == n) {
		if (f_v) {
			cout << "solution " << nb_sol << endl;
			Int_matrix_print(Mtx, n, n);
		}
		nb_sol++;
		return;
	}
	for (cur_candidate[depth] = 0;
			cur_candidate[depth] < nb_candidates[depth];
			cur_candidate[depth]++) {
		a = candidates[depth][cur_candidate[depth]];
		if (FALSE) {
			cout << "depth " << depth << " "
					<< cur_candidate[depth] << " / "
					<< nb_candidates[depth]
					<< " which is " << a << endl;
		}
		Int_vec_copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		backtrack_search(nb_sol, depth + 1, verbose_level);
	}
}

void null_polarity_generator::create_first_candidate_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb;

	if (f_v) {
		cout << "null_polarity_generator::create_first_candidate_set" << endl;
	}
	nb = 0;
	for (i = 0; i < qn; i++) {
		Int_vec_copy(Points + i * n, v, n);
		if (dot_product(v, v) == 1) {
			candidates[0][nb++] = i;
		}
	}
	nb_candidates[0] = nb;
	
	if (f_v) {
		cout << "null_polarity_generator::create_first_candidate_set done" << endl;
	}
}

void null_polarity_generator::create_next_candidate_set(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, ai, nb;

	if (f_v) {
		cout << "null_polarity_generator::create_next_candidate_set "
				"level=" << level << endl;
	}
	nb = 0;
	Int_vec_copy(Mtx + level * n, v, n);
	for (i = 0; i < nb_candidates[level]; i++) {
		ai = candidates[level][i];
		Int_vec_copy(Points + ai * n, w, n);
		if (dot_product(v, w) == 0) {
			candidates[level + 1][nb++] = ai;
		}
	}
	nb_candidates[level + 1] = nb;
	
	if (f_v) {
		cout << "null_polarity_generator::create_next_candidate_set "
				"done, found " << nb_candidates[level + 1]
				<< " candidates at level " << level + 1 << endl;
	}
}


int null_polarity_generator::dot_product(int *u1, int *u2)
{
	return F->Linear_algebra->dot_product(n, u1, u2);
}

}}}
