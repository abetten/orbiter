/*
 * orthogonal_rank_unrank.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void orthogonal::unrank_point(
		int *v, int stride, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::unrank_point rk=" << rk
				<< " epsilon=" << epsilon << " n=" << n << endl;
		}
	F->Q_epsilon_unrank(v, stride, epsilon, n - 1,
			form_c1, form_c2, form_c3, rk, verbose_level);
}

int orthogonal::rank_point(int *v, int stride, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "orthogonal::rank_point" << endl;
		}
	// copy the vector since Q_epsilon_rank has side effects
	// (namely, Q_epsilon_rank damages its input vector)

	for (i = 0; i < n; i++)
		rk_pt_v[i] = v[i * stride];

	if (f_v) {
		cout << "orthogonal::rank_point before F->Q_epsilon_rank" << endl;
		}
	ret = F->Q_epsilon_rank(rk_pt_v, 1, epsilon, n - 1,
			form_c1, form_c2, form_c3, verbose_level);
	if (f_v) {
		cout << "orthogonal::rank_point after F->Q_epsilon_rank" << endl;
		}
	if (f_v) {
		cout << "orthogonal::rank_point done" << endl;
		}
	return ret;
}


void orthogonal::unrank_line(int &p1, int &p2,
		long int rk, int verbose_level)
{
	if (epsilon == 1) {
		hyperbolic_unrank_line(p1, p2, rk, verbose_level);
		return;
		}
	else if (epsilon == 0) {
		parabolic_unrank_line(p1, p2, rk, verbose_level);
		return;
		}
	else {
		cout << "orthogonal::unrank_line epsilon = " << epsilon << endl;
		exit(1);
		}
}

long int orthogonal::rank_line(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int ret;

	if (f_v) {
		cout << "orthogonal::rank_line p1=" << p1 << " p2=" << p2 << endl;
	}
	if (epsilon == 1) {
		ret = hyperbolic_rank_line(p1, p2, verbose_level);
		}
	else if (epsilon == 0) {
		ret = parabolic_rank_line(p1, p2, verbose_level);
		}
	else {
		cout << "orthogonal::rank_line epsilon = " << epsilon << endl;
		exit(1);
		}
	if (f_v) {
		cout << "orthogonal::rank_line done" << endl;
	}
	return ret;
}

int orthogonal::line_type_given_point_types(
		int pt1, int pt2, int pt1_type, int pt2_type)
{
	if (epsilon == 1) {
		return hyperbolic_line_type_given_point_types(
				pt1, pt2, pt1_type, pt2_type);
		}
	else if (epsilon == 0) {
		return parabolic_line_type_given_point_types(
				pt1, pt2, pt1_type, pt2_type, FALSE);
		}
	else {
		cout << "type_and_index_to_point_rk "
				"epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "orthogonal::type_and_index_to_point_rk" << endl;
	}
	if (epsilon == 1) {
		if (f_v) {
			cout << "orthogonal::type_and_index_to_point_rk before hyperbolic_type_and_index_to_point_rk" << endl;
		}
		ret = hyperbolic_type_and_index_to_point_rk(
				type, index, verbose_level);
		if (f_v) {
			cout << "orthogonal::type_and_index_to_point_rk after hyperbolic_type_and_index_to_point_rk" << endl;
		}
	}
	else if (epsilon == 0) {
		if (f_v) {
			cout << "orthogonal::type_and_index_to_point_rk before parabolic_type_and_index_to_point_rk" << endl;
		}
		ret = parabolic_type_and_index_to_point_rk(
				type, index, verbose_level);
		if (f_v) {
			cout << "orthogonal::type_and_index_to_point_rk after parabolic_type_and_index_to_point_rk" << endl;
		}
	}
	else {
		cout << "type_and_index_to_point_rk "
				"epsilon = " << epsilon << endl;
		exit(1);
	}
	if (f_v) {
		cout << "orthogonal::type_and_index_to_point_rk done" << endl;
	}
	return ret;
}

void orthogonal::point_rk_to_type_and_index(
		int rk, int &type, int &index,
		int verbose_level)
{
	if (epsilon == 1) {
		hyperbolic_point_rk_to_type_and_index(
				rk, type, index);
		}
	else if (epsilon == 0) {
		parabolic_point_rk_to_type_and_index(
				rk, type, index, verbose_level);
		}
	else {
		cout << "type_and_index_to_point_rk epsilon = " << epsilon << endl;
		exit(1);
		}
}

void orthogonal::canonical_points_of_line(
	int line_type, int pt1, int pt2,
	int &cpt1, int &cpt2, int verbose_level)
{
	if (epsilon == 1) {
		hyperbolic_canonical_points_of_line(line_type,
				pt1, pt2, cpt1, cpt2, verbose_level);
		}
	else if (epsilon == 0) {
		parabolic_canonical_points_of_line(line_type,
				pt1, pt2, cpt1, cpt2, verbose_level);
		}
	else {
		cout << "canonical_points_of_line epsilon = " << epsilon << endl;
		exit(1);
		}
}



void orthogonal::unrank_S(int *v, int stride, int m, int rk)
// m = Witt index
{
	if (m == 0) {
		return;
		}
	F->S_unrank(v, stride, m, rk);
}

int orthogonal::rank_S(int *v, int stride, int m)
// m = Witt index
{
	int rk;

	if (m == 0) {
		return 0;
		}
	F->S_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::unrank_N(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->N_unrank(v, stride, m, rk);
}

int orthogonal::rank_N(int *v, int stride, int m)
// m = Witt index
{
	int rk;

	F->N_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::unrank_N1(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->N1_unrank(v, stride, m, rk);
}

int orthogonal::rank_N1(int *v, int stride, int m)
// m = Witt index
{
	int rk;

	F->N1_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::unrank_Sbar(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->Sbar_unrank(v, stride, m, rk, 0 /* verbose_level */);
}

int orthogonal::rank_Sbar(int *v, int stride, int m)
// m = Witt index
{
	int rk, i;

	for (i = 0; i < 2 * m; i++) {
		v_tmp[i] = v[i * stride];
		}
	F->Sbar_rank(v_tmp, 1, m, rk, 0 /* verbose_level */);
	return rk;
}

void orthogonal::unrank_Nbar(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->Nbar_unrank(v, stride, m, rk);
}

int orthogonal::rank_Nbar(int *v, int stride, int m)
// m = Witt index
{
	int rk;

	F->Nbar_rank(v, stride, m, rk);
	return rk;
}



}}


