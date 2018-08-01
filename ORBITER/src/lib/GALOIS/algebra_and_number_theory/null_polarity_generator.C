// null_polarity_generator.C
//
// Anton Betten
// December 11, 2015

#include "galois.h"



null_polarity_generator::null_polarity_generator()
{
	null();
}

null_polarity_generator::~null_polarity_generator()
{
	freeself();
}

void null_polarity_generator::null()
{
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

void null_polarity_generator::freeself()
{
	INT i;
	
	if (nb_candidates) {
		FREE_INT(nb_candidates);
		}
	if (cur_candidate) {
		FREE_INT(cur_candidate);
		}
	if (candidates) {
		for (i = 0; i < n + 1; i++) {
			FREE_INT(candidates[i]);
			}
		FREE_PINT(candidates);
		}
	if (Mtx) {
		FREE_INT(Mtx);
		}
	if (v) {
		FREE_INT(v);
		}
	if (w) {
		FREE_INT(w);
		}
	if (Points) {
		FREE_INT(Points);
		}
	if (Data) {
		FREE_INT(Data);
		}
	if (transversal_length) {
		FREE_INT(transversal_length);
		}
	null();
}

void null_polarity_generator::init(finite_field *F, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "null_polarity_generator::init" << endl;
		}
	null_polarity_generator::F = F;
	null_polarity_generator::n = n;
	q = F->q;
	qn = i_power_j(q, n);
	nb_candidates = NEW_INT(n + 1);
	cur_candidate = NEW_INT(n);
	candidates = NEW_PINT(n + 1);
	for (i = 0; i < n + 1; i++) {
		candidates[i] = NEW_INT(qn);
		}

	Mtx = NEW_INT(n * n);
	v = NEW_INT(n);
	w = NEW_INT(n);
	Points = NEW_INT(qn * n);
	for (i = 0; i < qn; i++) {
		AG_element_unrank(q, Points + i * n, 1, n, i);
		}

	create_first_candidate_set(verbose_level);

	if (f_v) {
		cout << "first candidate set has size " << nb_candidates[0] << endl;
		}

	//backtrack_search(0 /* depth */, verbose_level);
	


	INT first_moved = n;
	INT nb;

	nb_gens = 0;
	first_moved = n;
	transversal_length = NEW_INT(n);
	for (i = 0; i < n; i++) {
		transversal_length[i] = 1;
		}
	count_strong_generators(nb_gens, transversal_length, first_moved, 0, verbose_level);

	if (f_v) {
		cout << "We found " << nb_gens << " strong generators" << endl;
		cout << "transversal_length = ";
		INT_vec_print(cout, transversal_length, n);
		cout << endl;
		cout << "group order: ";
		print_longinteger_after_multiplying(cout, transversal_length, n);
		cout << endl;
		}	

	Data = NEW_INT(nb_gens * n * n);

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
			INT_matrix_print(Data + i * n * n, n, n);
			}
		}


	if (f_v) {
		cout << "null_polarity_generator::init done" << endl;
		}
}

INT null_polarity_generator::count_strong_generators(INT &nb, INT *transversal_length, INT &first_moved, INT depth, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	INT a;
	
	if (depth == n) {
		//cout << "solution " << nb << endl;
		//INT_matrix_print(Mtx, n, n);
		if (first_moved < n) {
			transversal_length[first_moved]++;
			}
		nb++;
		return FALSE;
		}
	for (cur_candidate[depth] = 0; cur_candidate[depth] < nb_candidates[depth]; cur_candidate[depth]++) {
		if (cur_candidate[depth] && depth < first_moved) {
			first_moved = depth;
			}	
		a = candidates[depth][cur_candidate[depth]];
		if (FALSE) {
			cout << "depth " << depth << " " << cur_candidate[depth] << " / " << nb_candidates[depth] << " which is " << a << endl;
			}
		INT_vec_copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		if (!count_strong_generators(nb, transversal_length, first_moved, depth + 1, verbose_level) && depth > first_moved) {
			return FALSE;
			}
		}
	return TRUE;
}

INT null_polarity_generator::get_strong_generators(INT *Data, INT &nb, INT &first_moved, INT depth, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	INT a;
	
	if (depth == n) {
		//cout << "solution " << nb << endl;
		//INT_matrix_print(Mtx, n, n);
		INT_vec_copy(Mtx, Data + nb * n * n, n * n);
		nb++;
		return FALSE;
		}
	for (cur_candidate[depth] = 0; cur_candidate[depth] < nb_candidates[depth]; cur_candidate[depth]++) {
		if (cur_candidate[depth] && depth < first_moved) {
			first_moved = depth;
			}	
		a = candidates[depth][cur_candidate[depth]];
		if (FALSE) {
			cout << "depth " << depth << " " << cur_candidate[depth] << " / " << nb_candidates[depth] << " which is " << a << endl;
			}
		INT_vec_copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		if (!get_strong_generators(Data, nb, first_moved, depth + 1, verbose_level) && depth > first_moved) {
			return FALSE;
			}
		}
	return TRUE;
}

void null_polarity_generator::backtrack_search(INT &nb_sol, INT depth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a;
	
	if (depth == n) {
		if (f_v) {
			cout << "solution " << nb_sol << endl;
			INT_matrix_print(Mtx, n, n);
			}
		nb_sol++;
		return;
		}
	for (cur_candidate[depth] = 0; cur_candidate[depth] < nb_candidates[depth]; cur_candidate[depth]++) {
		a = candidates[depth][cur_candidate[depth]];
		if (FALSE) {
			cout << "depth " << depth << " " << cur_candidate[depth] << " / " << nb_candidates[depth] << " which is " << a << endl;
			}
		INT_vec_copy(Points + a * n, Mtx + depth * n, n);
		create_next_candidate_set(depth, 0 /* verbose_level */);

		backtrack_search(nb_sol, depth + 1, verbose_level);
		}
}

void null_polarity_generator::create_first_candidate_set(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, nb;

	if (f_v) {
		cout << "null_polarity_generator::create_first_candidate_set" << endl;
		}
	nb = 0;
	for (i = 0; i < qn; i++) {
		INT_vec_copy(Points + i * n, v, n);
		if (dot_product(v, v) == 1) {
			candidates[0][nb++] = i;
			}
		}
	nb_candidates[0] = nb;
	
	if (f_v) {
		cout << "null_polarity_generator::create_first_candidate_set done" << endl;
		}
}

void null_polarity_generator::create_next_candidate_set(INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, ai, nb;

	if (f_v) {
		cout << "null_polarity_generator::create_next_candidate_set level=" << level << endl;
		}
	nb = 0;
	INT_vec_copy(Mtx + level * n, v, n);
	for (i = 0; i < nb_candidates[level]; i++) {
		ai = candidates[level][i];
		INT_vec_copy(Points + ai * n, w, n);
		if (dot_product(v, w) == 0) {
			candidates[level + 1][nb++] = ai;
			}
		}
	nb_candidates[level + 1] = nb;
	
	if (f_v) {
		cout << "null_polarity_generator::create_next_candidate_set done, found " << nb_candidates[level + 1] << " candidates at level " << level + 1 << endl;
		}
}


INT null_polarity_generator::dot_product(INT *u1, INT *u2)
{
#if 0
	INT c;
	INT i;

	c = 0;
	for (i = 0; i < n; i++) {
		c = F->add(c, F->mult(u1[i], u2[i]));
		}
	return c;
#else
	return F->dot_product(n, u1, u2);
#endif
}

