/*
 * linear_algebra3.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace linear_algebra {






int linear_algebra::evaluate_conic_form(
		int *six_coeffs, int *v3)
{
	//int a = 2, b = 0, c = 0, d = 4, e = 4, f = 4, val, val1;
	//int a = 3, b = 1, c = 2, d = 4, e = 1, f = 4, val, val1;
	int val, val1;

	val = 0;
	val1 = F->product3(six_coeffs[0], v3[0], v3[0]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[1], v3[1], v3[1]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[2], v3[2], v3[2]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[3], v3[0], v3[1]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[4], v3[0], v3[2]);
	val = F->add(val, val1);
	val1 = F->product3(six_coeffs[5], v3[1], v3[2]);
	val = F->add(val, val1);
	return val;
}

int linear_algebra::evaluate_quadric_form_in_PG_three(
		int *ten_coeffs, int *v4)
{
	int val, val1;

	val = 0;
	val1 = F->product3(ten_coeffs[0], v4[0], v4[0]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[1], v4[1], v4[1]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[2], v4[2], v4[2]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[3], v4[3], v4[3]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[4], v4[0], v4[1]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[5], v4[0], v4[2]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[6], v4[0], v4[3]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[7], v4[1], v4[2]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[8], v4[1], v4[3]);
	val = F->add(val, val1);
	val1 = F->product3(ten_coeffs[9], v4[2], v4[3]);
	val = F->add(val, val1);
	return val;
}

int linear_algebra::Pluecker_12(int *x4, int *y4)
{
	return Pluecker_ij(0, 1, x4, y4);
}

int linear_algebra::Pluecker_21(int *x4, int *y4)
{
	return Pluecker_ij(1, 0, x4, y4);
}

int linear_algebra::Pluecker_13(int *x4, int *y4)
{
	return Pluecker_ij(0, 2, x4, y4);
}

int linear_algebra::Pluecker_31(int *x4, int *y4)
{
	return Pluecker_ij(2, 0, x4, y4);
}

int linear_algebra::Pluecker_14(int *x4, int *y4)
{
	return Pluecker_ij(0, 3, x4, y4);
}

int linear_algebra::Pluecker_41(int *x4, int *y4)
{
	return Pluecker_ij(3, 0, x4, y4);
}

int linear_algebra::Pluecker_23(int *x4, int *y4)
{
	return Pluecker_ij(1, 2, x4, y4);
}

int linear_algebra::Pluecker_32(int *x4, int *y4)
{
	return Pluecker_ij(2, 1, x4, y4);
}

int linear_algebra::Pluecker_24(int *x4, int *y4)
{
	return Pluecker_ij(1, 3, x4, y4);
}

int linear_algebra::Pluecker_42(int *x4, int *y4)
{
	return Pluecker_ij(3, 1, x4, y4);
}

int linear_algebra::Pluecker_34(int *x4, int *y4)
{
	return Pluecker_ij(2, 3, x4, y4);
}

int linear_algebra::Pluecker_43(int *x4, int *y4)
{
	return Pluecker_ij(3, 2, x4, y4);
}

int linear_algebra::Pluecker_ij(
		int i, int j, int *x4, int *y4)
{
	return F->add(F->mult(x4[i], y4[j]),
			F->negate(F->mult(x4[j], y4[i])));
}


int linear_algebra::evaluate_symplectic_form(
		int len, int *x, int *y)
// the form consists of a series of diagonal 2 x 2 blocks
// of the form (0,1,-1,0)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "linear_algebra::evaluate_symplectic_form "
				"len must be even" << endl;
		cout << "len=" << len << endl;
		exit(1);
	}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = F->add(c,
				F->add(
						F->mult(x[2 * i + 0], y[2 * i + 1]),
						F->negate(F->mult(x[2 * i + 1], y[2 * i + 0]))
				));
	}
	return c;
}

int linear_algebra::evaluate_symmetric_form(
		int len, int *x, int *y)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "linear_algebra::evaluate_symmetric_form "
				"len must be even" << endl;
		cout << "len=" << len << endl;
		exit(1);
	}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = F->add(c, F->add(
				F->mult(x[2 * i + 0], y[2 * i + 1]),
				F->mult(x[2 * i + 1], y[2 * i + 0])
				));
	}
	return c;
}

int linear_algebra::evaluate_quadratic_form_x0x3mx1x2(
		int *x)
{
	int a;

	a = F->add(F->mult(x[0], x[3]),
			F->negate(F->mult(x[1], x[2])));
	return a;
}

void linear_algebra::solve_y2py(
		int a, int *Y2, int &nb_sol)
{
	int y, y2py;

	nb_sol = 0;
	for (y = 0; y < F->q; y++) {
		y2py = F->add(F->mult(y, y), y);
		if (y2py == a) {
			Y2[nb_sol++] = y;
		}
	}
	if (nb_sol > 2) {
		cout << "linear_algebra::solve_y2py nb_sol > 2" << endl;
		exit(1);
	}
}

void linear_algebra::find_secant_points_wrt_x0x3mx1x2(
		int *Basis_line, int *Pts4, int &nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u;
	int b0, b1, b2, b3, b4, b5, b6, b7;
	int a, av, b, c, bv, acbv2, cav, t, r, i;

	if (f_v) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2" << endl;
	}
	nb_pts = 0;

#if 0
	u = evaluate_quadratic_form_x0x3mx1x2(Basis_line);
	if (u == 0) {
		Pts4[nb_pts * 2 + 0] = 1;
		Pts4[nb_pts * 2 + 1] = 0;
		nb_pts++;
	}
#endif

	u = evaluate_quadratic_form_x0x3mx1x2(Basis_line + 4);
	if (u == 0) {
		Pts4[nb_pts * 2 + 0] = 0;
		Pts4[nb_pts * 2 + 1] = 1;
		nb_pts++;
	}

	b0 = Basis_line[0];
	b1 = Basis_line[1];
	b2 = Basis_line[2];
	b3 = Basis_line[3];
	b4 = Basis_line[4];
	b5 = Basis_line[5];
	b6 = Basis_line[6];
	b7 = Basis_line[7];
	a = F->add(F->mult(b4, b7), F->negate(F->mult(b5, b6)));
	c = F->add(F->mult(b0, b3), F->negate(F->mult(b1, b2)));
	b = F->add4(F->mult(b0, b7),
			F->mult(b3, b4),
			F->negate(F->mult(b1, b6)),
			F->negate(F->mult(b2, b5)));
	if (f_v) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
				"a=" << a << " b=" << b << " c=" << c << endl;
	}
	if (a == 0) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
				"a == 0" << endl;
		exit(1);
	}
	av = F->inverse(a);
	if (EVEN(F->p)) {
		if (b == 0) {
			cav = F->mult(c, av);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
						"cav=" << cav << endl;
			}
			r = F->frobenius_power(cav, F->e - 1);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
						"r=" << r << endl;
			}
			Pts4[nb_pts * 2 + 0] = 1;
			Pts4[nb_pts * 2 + 1] = r;
			nb_pts++;
		}
		else {
			bv = F->inverse(b);
			acbv2 = F->mult4(a, c, bv, bv);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
						"acbv2=" << acbv2 << endl;
			}
			t = F->absolute_trace(acbv2);
			if (f_v) {
				cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
						"t=" << t << endl;
			}
			if (t == 0) {
				int Y2[2];
				int nb_sol;

				if (f_v) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
							"before solve_y2py" << endl;
				}
				solve_y2py(acbv2, Y2, nb_sol);
				if (f_v) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
							"after solve_y2py nb_sol= " << nb_sol << endl;
					Int_vec_print(cout, Y2, nb_sol);
					cout << endl;
				}
				if (nb_sol + nb_pts > 2) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
							"nb_sol + nb_pts > 2" << endl;
					exit(1);
				}
				for (i = 0; i < nb_sol; i++) {
					r = F->mult3(b, Y2[i], av);
					if (f_v) {
						cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
								"solution " << i << " r=" << r << endl;
					}
					Pts4[nb_pts * 2 + 0] = 1;
					Pts4[nb_pts * 2 + 1] = r;
					nb_pts++;
				}
			}
			else {
				// no solution
				if (f_v) {
					cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
							"no solution" << endl;
				}
				nb_pts = 0;
			}
		}
	}
	else {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 "
				"odd characteristic not yet implemented" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "linear_algebra::find_secant_points_wrt_x0x3mx1x2 done" << endl;
	}
}

int linear_algebra::is_totally_isotropic_wrt_symplectic_form(
		int k, int n, int *Basis)
{
	int i, j;

	for (i = 0; i < k; i++) {
		for (j = i + 1; j < k; j++) {
			if (evaluate_symplectic_form(n, Basis + i * n, Basis + j * n)) {
				return false;
			}
		}
	}
	return true;
}

int linear_algebra::evaluate_monomial(
		int *monomial,
		int *variables, int nb_vars)
{
	int i, j, a, b, x;

	a = 1;
	for (i = 0; i < nb_vars; i++) {
		b = monomial[i];
		x = variables[i];
		for (j = 0; j < b; j++) {
			a = F->mult(a, x);
		}
	}
	return a;
}





}}}

