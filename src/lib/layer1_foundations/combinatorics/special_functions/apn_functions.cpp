/*
 * apn_functions.cpp
 *
 *  Created on: Feb 12, 2023
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace special_functions {


apn_functions::apn_functions()
{
	Record_birth();
	F = NULL;
}

apn_functions::~apn_functions()
{
	Record_death();
}

void apn_functions::init(
		algebra::field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "apn_functions::init" << endl;
	}

	apn_functions::F = F;


	if (f_v) {
		cout << "apn_functions::init done" << endl;
	}
}


void apn_functions::search_APN(
		int delta_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	int *f;
	int delta, nb_times;
	int *A_matrix;
	int *B_matrix;
	int *Count_ab;
	int *nb_times_ab;
	int i, j;

	if (f_v) {
		cout << "apn_functions::search_APN" << endl;
	}
	q = F->q;
	delta = delta_max;
	nb_times = 0;
	f = NEW_int(q);

	A_matrix = NEW_int(q * q);
	B_matrix = NEW_int(q * q);
	Count_ab = NEW_int(q * q);
	nb_times_ab = NEW_int(q * q);

	std::vector<std::vector<int> > Solutions;

	Int_vec_zero(A_matrix, q * q);
	Int_vec_zero(B_matrix, q * q);
	Int_vec_zero(Count_ab, q * q);
	Int_vec_zero(nb_times_ab, q * q);

	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			A_matrix[i * q + j] = F->add(i, F->negate(j));
		}
	}
	search_APN_recursion(
			f,
			0 /* depth */,
			true,
			delta, nb_times,
			Solutions,
			A_matrix, B_matrix, Count_ab, nb_times_ab,
			verbose_level);
	if (f_v) {
		cout << "apn_functions::search_APN recursion finished" << endl;
		cout << "delta = " << delta << endl;
		cout << "nb_times = " << nb_times << endl;
	}


	string fname;
	other::orbiter_kernel_system::file_io Fio;

	fname = "APN_functions_q" + std::to_string(F->q) + ".csv";

	Fio.Csv_file_support->vector_matrix_write_csv(
			fname, Solutions);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(A_matrix);
	FREE_int(B_matrix);
	FREE_int(Count_ab);
	FREE_int(nb_times_ab);
	FREE_int(f);

	if (f_v) {
		cout << "apn_functions::search_APN done" << endl;
	}
}


void apn_functions::search_APN_recursion(
		int *f, int depth, int f_normalize,
		int &delta_max, int &nb_times,
		std::vector<std::vector<int> > &Solutions,
		int *A_matrix, int *B_matrix,
		int *Count_ab, int *nb_times_ab,
		int verbose_level)
{
	if (depth == F->q) {
		int delta;

		delta = differential_uniformity(f, nb_times_ab, 0 /* verbose_level */);
		if (delta < delta_max) {
			delta_max = delta;
			nb_times = 1;

			Solutions.clear();

			vector<int> S;
			int i;

			for (i = 0; i < F->q; i++) {
				S.push_back(f[i]);
			}
			Solutions.push_back(S);

			Int_vec_print(cout, f, F->q);
			cout << " delta = " << delta << " nb_times=" << nb_times << endl;
		}
		else if (delta == delta_max) {
			nb_times++;
			int f_do_it;

			if (nb_times > 100) {
				if ((nb_times % 10000) == 0) {
					f_do_it = true;
				}
				else {
					f_do_it = false;
				}
			}
			else {
				f_do_it = true;
			}

			vector<int> S;
			int i;

			for (i = 0; i < F->q; i++) {
				S.push_back(f[i]);
			}
			Solutions.push_back(S);

			if (f_do_it) {
				Int_vec_print(cout, f, F->q);
				cout << " delta = " << delta << " nb_times=" << nb_times << endl;
			}
		}
		return;
	}

	int fxd;
	int f_normalize_below;

	f_normalize_below = f_normalize;

	for (fxd = 0; fxd < F->q; fxd++) {
		if (f_normalize) {
			if (fxd) {
				f_normalize_below = false;
				if (fxd != 1) {
					continue;
				}
			}
		}
		f[depth] = fxd;

		if (search_APN_perform_checks(
				f, depth,
				delta_max,
				A_matrix, B_matrix, Count_ab,
				0 /*verbose_level*/)) {

			search_APN_recursion(
					f, depth + 1, f_normalize_below,
					delta_max, nb_times,
					Solutions,
					A_matrix, B_matrix, Count_ab, nb_times_ab,
					verbose_level);

			search_APN_undo_checks(
					f, depth,
					delta_max,
					A_matrix, B_matrix, Count_ab,
					0 /*verbose_level*/);

		}
		else {
			// cannot choose this value. Continue with the search.
		}
	}
}

int apn_functions::search_APN_perform_checks(
		int *f, int depth,
		int delta_max,
		int *A_matrix, int *B_matrix, int *Count_ab,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "apn_functions::search_APN_perform_checks" << endl;
	}
	for (i = 0; i < depth; i++) {

		if (!perform_single_check(
				f, depth, i, delta_max,
				A_matrix, B_matrix, Count_ab,
				verbose_level)) {

			for (j = i - 1; j >= 0; j--) {
				undo_single_check(
								f, depth, j, delta_max,
								A_matrix, B_matrix, Count_ab,
								verbose_level);
			}

			return false;
		}

	}
	return true;
}

void apn_functions::search_APN_undo_checks(
		int *f, int depth,
		int delta_max,
		int *A_matrix, int *B_matrix, int *Count_ab,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "apn_functions::search_APN_undo_checks" << endl;
	}
	for (i = depth - 1; i >= 0; i--) {

		undo_single_check(
						f, depth, i, delta_max,
						A_matrix, B_matrix, Count_ab,
						verbose_level);

	}
}


int apn_functions::perform_single_check(
		int *f, int depth, int i, int delta_max,
		int *A_matrix, int *B_matrix, int *Count_ab,
		int verbose_level)
{
	int q;
	int a1, b1, a2, b2;

	q = F->q;

	a1 = A_matrix[i * q + depth];
	b1 = F->add(f[depth], F->negate(f[i]));

	if (Count_ab[a1 * q + b1] == delta_max) {
		return false;
	}

	B_matrix[i * q + depth] = b1;
	Count_ab[a1 * q + b1]++;

	a2 = A_matrix[depth * q + i];
	b2 = F->add(f[i], F->negate(f[depth]));

	if (Count_ab[a2 * q + b2] == delta_max) {
		Count_ab[a1 * q + b1]--;
		B_matrix[i * q + depth] = 0;
		return false;
	}

	B_matrix[depth * q + i] = b2;
	Count_ab[a2 * q + b2]++;

	return true;
}

void apn_functions::undo_single_check(
		int *f, int depth, int i, int delta_max,
		int *A_matrix, int *B_matrix, int *Count_ab,
		int verbose_level)
{
	int q;
	int a1, b1, a2, b2;

	q = F->q;

	a1 = A_matrix[i * q + depth];
	b1 = F->add(f[depth], F->negate(f[i]));

	if (Count_ab[a1 * q + b1] == 0) {
		cout << "apn_functions::undo_single_check "
				"Count_ab[a1 * q + b1] == 0" << endl;
		exit(1);
	}

	B_matrix[i * q + depth] = b1;
	Count_ab[a1 * q + b1]--;

	a2 = A_matrix[depth * q + i];
	b2 = F->add(f[i], F->negate(f[depth]));

	if (Count_ab[a2 * q + b2] == 0) {
		cout << "apn_functions::undo_single_check "
				"Count_ab[a2 * q + b2] == 0" << endl;
		exit(1);
	}

	B_matrix[depth * q + i] = b2;
	Count_ab[a2 * q + b2]--;
}

void apn_functions::search_APN_old(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	int *f;
	int delta_min, nb_times;
	int *tmp_qxq;

	if (f_v) {
		cout << "apn_functions::search_APN_old" << endl;
	}
	q = F->q;
	delta_min = INT_MAX;
	nb_times = 0;
	f = NEW_int(q);
	tmp_qxq = NEW_int(q * q);

	std::vector<std::vector<int> > Solutions;

	search_APN_recursion_old(
			f,
			0 /* depth */,
			true,
			delta_min, nb_times, Solutions,
			tmp_qxq,
			verbose_level);
	cout << "nb_times = " << nb_times << endl;
	cout << "delta_min = " << delta_min << endl;
	FREE_int(f);

	string fname;
	other::orbiter_kernel_system::file_io Fio;

	fname = "APN_functions_q" + std::to_string(F->q) + ".csv";

	Fio.Csv_file_support->vector_matrix_write_csv(
			fname, Solutions);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	FREE_int(tmp_qxq);

	if (f_v) {
		cout << "algebra_global::search_APN_old done" << endl;
	}
}

void apn_functions::search_APN_recursion_old(
		int *f, int depth, int f_normalize,
		int &delta_min, int &nb_times,
		std::vector<std::vector<int> > &Solutions,
		int *nb_times_ab,
		int verbose_level)
{
	if (depth == F->q) {
		int delta;

		delta = differential_uniformity(
				f, nb_times_ab,
				0 /* verbose_level */);

		if (delta < delta_min) {

			delta_min = delta;
			nb_times = 1;

			Solutions.clear();

			vector<int> S;
			int i;

			for (i = 0; i < F->q; i++) {
				S.push_back(f[i]);
			}
			Solutions.push_back(S);

			Int_vec_print(cout, f, F->q);
			cout << " delta = " << delta << " nb_times=" << nb_times << endl;

		}

		else if (delta == delta_min) {

			nb_times++;
			int f_do_it;

			if (nb_times > 100) {
				if ((nb_times % 1000) == 0) {
					f_do_it = true;
				}
				else {
					f_do_it = false;
				}
			}
			else {
				f_do_it = true;
			}

			vector<int> S;
			int i;

			for (i = 0; i < F->q; i++) {
				S.push_back(f[i]);
			}
			Solutions.push_back(S);

			if (f_do_it) {
				Int_vec_print(cout, f, F->q);
				cout << " delta = " << delta << " nb_times=" << nb_times << endl;
			}

		}

		return;
	}

	int a;
	int f_normalize_below;

	f_normalize_below = f_normalize;

	for (a = 0; a < F->q; a++) {
		if (f_normalize) {
			if (a) {
				f_normalize_below = false;
				if (a != 1) {
					continue;
				}
			}
		}
		f[depth] = a;
		search_APN_recursion_old(
				f, depth + 1, f_normalize_below,
				delta_min, nb_times, Solutions, nb_times_ab,
				verbose_level);
	}
}

int apn_functions::differential_uniformity(
		int *f, int *nb_times_ab, int verbose_level)
// f[q] is a function from Fq to Fq.
// nb_times_ab[a,b] = number of x in Fq s.t. f(x+a)+f(x) = b, where a neq 0
// delta is the max of all nb_times_ab[a,b], a neq 0.
{
	int f_v = (verbose_level >= 1);
	int q;
	int a, x, b, fx, fxpa, mfx, delta;

	if (f_v) {
		cout << "apn_functions::differential_uniformity" << endl;
	}
	q = F->q;

	Int_vec_zero(nb_times_ab, q * q);

	for (x = 0; x < q; x++) {
		fx = f[x];
		mfx = F->negate(fx);
		for (a = 1; a < q; a++) {
			fxpa = f[F->add(x, a)];
#if 0
			av = F->inverse(a);
			dy = F->add(fxpa, mfx);
			b = F->mult(dy, av);
#else
			b = F->add(fxpa, mfx);
#endif
			nb_times_ab[a * q + b]++;
		}
	}
	delta = 0;
	for (a = 1; a < q; a++) {
		for (b = 0; b < q; b++) {
			delta = MAXIMUM(delta, nb_times_ab[a * q + b]);
		}
	}
	return delta;
}

int apn_functions::differential_uniformity_with_fibre(
		int *f, int *nb_times_ab, int *&Fibre,
		int verbose_level)
// f[q]
{
	int f_v = (verbose_level >= 1);
	int q;
	int a, x, b, fx, fxpa, mfx, delta;

	if (f_v) {
		cout << "apn_functions::differential_uniformity" << endl;
	}
	q = F->q;
	Int_vec_zero(nb_times_ab, q * q);
	for (x = 0; x < q; x++) {
		fx = f[x];
		mfx = F->negate(fx);
		for (a = 1; a < q; a++) {
			fxpa = f[F->add(x, a)];
			b = F->add(fxpa, mfx);
			nb_times_ab[a * q + b]++;
		}
	}
	delta = 0;
	for (a = 1; a < q; a++) {
		for (b = 0; b < q; b++) {
			delta = MAXIMUM(delta, nb_times_ab[a * q + b]);
		}
	}

	Fibre = NEW_int(q * q * delta);
	Int_vec_zero(nb_times_ab, q * q);
	for (x = 0; x < q; x++) {
		fx = f[x];
		mfx = F->negate(fx);
		for (a = 1; a < q; a++) {
			fxpa = f[F->add(x, a)];
			b = F->add(fxpa, mfx);
			Fibre[(a * q + b) * delta + nb_times_ab[a * q + b]] = x;
			nb_times_ab[a * q + b]++;
		}
	}

	return delta;
}



}}}}



