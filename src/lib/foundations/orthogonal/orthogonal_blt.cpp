/*
 * orthogonal_blt.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {



#if 0
void orthogonal::create_Linear_BLT_set(long int *set, int *ABC, int verbose_level)
// a(t)= 1, b(t) = t, c(t) = t^2, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int i, a, b, c;

	int q = F->q;

	if (f_v) {
		cout << "orthogonal::create_Linear_BLT_set" << endl;
	}
	int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = F->power(i, 2);
		c = F->power(i, 3);
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
		}
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
		F->create_BLT_point(v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
		}
	}
	int_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		int_vec_print(cout, v, 5);
		cout << endl;
	}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
	}
	if (f_v) {
		cout << "orthogonal::create_Linear_BLT_set done" << endl;
	}
}

#endif


void orthogonal::create_FTWKB_BLT_set(long int *set, int *ABC, int verbose_level)
// for q congruent 2 mod 3
// a(t)= t, b(t) = 3*t^2, c(t) = 3*t^3, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int r, i, a, b, c;
	geometry::geometry_global Gg;

	int q = F->q;

	if (q <= 5) {
		cout << "orthogonal::create_FTWKB_BLT_set q <= 5" << endl;
		exit(1);
	}
	r = q % 3;
	if (r != 2) {
		cout << "orthogonal::create_FTWKB_BLT_set q mod 3 must be 2" << endl;
		exit(1);
	}
	Int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = F->mult(3, F->power(i, 2));
		c = F->mult(3, F->power(i, 3));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
		}
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
		Gg.create_BLT_point(F, v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
		}
	}
	orbiter_kernel_system::Orbiter->Int_vec->init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		Int_vec_print(cout, v, 5);
		cout << endl;
	}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
	}
	if (f_v) {
		cout << "orthogonal::create_FTWKB_BLT_set the BLT set FTWKB is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
}

void orthogonal::create_K1_BLT_set(long int *set, int *ABC, int verbose_level)
// for a nonsquare m, and q=p^e
// a(t)= t, b(t) = 0, c(t) = -m*t^p, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int i, m, minus_one, exponent, a, b, c;
	int q;
	geometry::geometry_global Gg;

	q = F->q;
	m = F->p; // the primitive element is a nonsquare
	exponent = F->p;
	minus_one = F->negate(1);
	if (f_v) {
		cout << "m=" << m << endl;
		cout << "exponent=" << exponent << endl;
		cout << "minus_one=" << minus_one << endl;
		}
	Int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = 0;
		c = F->mult(minus_one, F->mult(m, F->power(i, exponent)));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
			}
		Gg.create_BLT_point(F, v, a, b, c, verbose_level - 2);
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	orbiter_kernel_system::Orbiter->Int_vec->init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		Int_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "orthogonal::create_K1_BLT_set the BLT set K1 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_K2_BLT_set(long int *set, int *ABC, int verbose_level)
// for q congruent 2 or 3 mod 5
// a(t)= t, b(t) = 5*t^3, c(t) = 5*t^5, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int five, r, i, a, b, c;
	int q;
	geometry::geometry_global Gg;

	q = F->q;
	if (q <= 5) {
		cout << "orthogonal::create_K2_BLT_set q <= 5" << endl;
		return;
		}
	r = q % 5;
	if (r != 2 && r != 3) {
		cout << "orthogonal::create_K2_BLT_set "
				"q mod 5 must be 2 or 3" << endl;
		return;
		}
	five = 5 % F->p;
	Int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = F->mult(five, F->power(i, 3));
		c = F->mult(five, F->power(i, 5));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
			}
		Gg.create_BLT_point(F, v, a, b, c, verbose_level - 2);
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	orbiter_kernel_system::Orbiter->Int_vec->init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		Int_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "orthogonal::create_K2_BLT_set "
				"the BLT set K2 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_LP_37_72_BLT_set(
		long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,20,1,33,5,
		1,6,23,19,23,
		1,32,11,35,17,
		1,33,12,14,23,
		1,25,8,12,6,
		1,16,6,1,22,
		1,23,8,5,6,
		1,8,6,13,8,
		1,22,19,20,13,
		1,21,23,16,23,
		1,28,6,9,8,
		1,2,26,7,13,
		1,5,9,36,35,
		1,12,23,10,17,
		1,14,16,25,23,
		1,9,8,26,35,
		1,1,11,8,19,
		1,19,12,11,17,
		1,18,27,22,22,
		1,24,36,17,35,
		1,26,27,23,5,
		1,27,25,24,22,
		1,36,21,32,35,
		1,7,16,31,8,
		1,35,5,15,5,
		1,10,36,6,13,
		1,30,4,3,5,
		1,4,3,30,19,
		1,17,13,2,19,
		1,11,28,18,17,
		1,13,16,27,22,
		1,29,12,28,6,
		1,15,10,34,19,
		1,3,30,4,13,
		1,31,9,21,8,
		1,34,9,29,6
		};
	int q;

	q = F->q;
	if (q != 37) {
		cout << "orthogonal::create_LP_37_72_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_LP_37_72_BLT_set "
				"the BLT set LP_37_72 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_LP_37_4a_BLT_set(long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,9,16,8,5,
		1,13,20,26,2,
		1,4,12,14,22,
		1,19,23,5,5,
		1,24,17,19,32,
		1,18,18,10,14,
		1,2,4,36,23,
		1,7,5,24,29,
		1,36,20,22,29,
		1,14,10,13,14,
		1,28,22,7,23,
		1,32,28,20,19,
		1,30,27,23,24,
		1,3,30,28,15,
		1,1,20,31,13,
		1,11,36,33,6,
		1,29,22,30,15,
		1,20,10,4,5,
		1,8,14,32,29,
		1,25,15,9,31,
		1,26,13,18,29,
		1,23,19,6,19,
		1,35,11,15,20,
		1,22,11,25,32,
		1,10,16,2,20,
		1,17,18,27,31,
		1,15,29,16,29,
		1,31,18,1,15,
		1,12,34,35,15,
		1,33,23,17,20,
		1,27,23,21,14,
		1,34,22,3,6,
		1,21,11,11,18,
		1,5,33,12,35,
		1,6,22,34,15,
		1,16,31,29,18
		};
	int q;

	q = F->q;
	if (q != 37) {
		cout << "orthogonal::create_LP_37_4a_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_LP_37_4a_BLT_set "
				"the BLT set LP_37_4a is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_LP_37_4b_BLT_set(long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,3,7,25,24,
		1,35,30,32,15,
		1,4,10,30,2,
		1,14,8,17,31,
		1,30,18,2,23,
		1,19,0,10,32,
		1,8,18,12,24,
		1,34,2,20,19,
		1,28,34,15,15,
		1,2,21,23,31,
		1,13,29,36,23,
		1,23,13,8,17,
		1,25,12,35,17,
		1,1,14,4,22,
		1,17,2,19,6,
		1,12,17,1,32,
		1,27,23,3,19,
		1,20,2,21,20,
		1,33,30,22,2,
		1,11,16,31,32,
		1,29,6,13,31,
		1,16,17,7,6,
		1,6,25,14,31,
		1,32,27,29,8,
		1,15,8,9,23,
		1,5,17,24,35,
		1,18,13,33,14,
		1,7,36,26,2,
		1,21,34,28,32,
		1,10,22,16,22,
		1,26,34,27,29,
		1,31,13,34,35,
		1,9,13,18,2,
		1,22,28,5,31,
		1,24,3,11,23,
		1,36,27,6,17
		};
	int q;

	q = F->q;
	if (q != 37) {
		cout << "orthogonal::create_LP_37_4b_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_LP_37_4b_BLT_set "
				"the BLT set LP_37_4b is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_Law_71_BLT_set(
		long int *set, int verbose_level)
// This example can be found in Maska Law's thesis on page 115.
// Maska Law: Flocks, generalised quadrangles
// and translatrion planes from BLT-sets,
// The University of Western Australia, 2003.
// Note the coordinates here are different (for an unknown reason).
// Law suggests to construct an infinite family
// starting from the subgroup A_4 of
// the stabilizer of the Fisher/Thas/Walker/Kantor examples.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
#if 1
		0,0,0,0,1,
		1,0,0,0,0,
		1,20,1,33,5,
		1,6,23,19,23,
		1,32,11,35,17,
		1,33,12,14,23,
		1,25,8,12,6,
		1,16,6,1,22,
		1,23,8,5,6,
		1,8,6,13,8,
		1,22,19,20,13,
		1,21,23,16,23,
		1,28,6,9,8,
		1,2,26,7,13,
		1,5,9,36,35,
		1,12,23,10,17,
		1,14,16,25,23,
		1,9,8,26,35,
		1,1,11,8,19,
		1,19,12,11,17,
		1,18,27,22,22,
		1,24,36,17,35,
		1,26,27,23,5,
		1,27,25,24,22,
		1,36,21,32,35,
		1,7,16,31,8,
		1,35,5,15,5,
		1,10,36,6,13,
		1,30,4,3,5,
		1,4,3,30,19,
		1,17,13,2,19,
		1,11,28,18,17,
		1,13,16,27,22,
		1,29,12,28,6,
		1,15,10,34,19,
		1,3,30,4,13,
		1,31,9,21,8,
		1,34,9,29,6
#endif
		};
	int q;

	q = F->q;
	if (q != 71) {
		cout << "orthogonal::create_Law_71_BLT_set q = 71" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_Law_71_BLT_set "
				"the BLT set LP_71 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
		}
}


int orthogonal::BLT_test_full(int size, long int *set, int verbose_level)
{
	if (!collinearity_test(size, set, 0/*verbose_level - 2*/)) {
		return FALSE;
		}
	if (!BLT_test(size, set, verbose_level)) {
		return FALSE;
		}
	return TRUE;
}

int orthogonal::BLT_test(int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, x, y, z, a;
	int f_OK = TRUE;
	int fxy, fxz, fyz, l1, l2, l3;
	int two;
	int m1[5], m3[5];

	if (size <= 2)
		return TRUE;
	if (f_v) {
		cout << "BLT_test for" << endl;
		Lint_vec_print(cout, set, size);
		if (f_vv) {
			for (i = 0; i < size; i++) {
				unrank_point(v1, 1, set[i], verbose_level - 1);
				cout << i << " : " << set[i] << " : ";
				Int_vec_print(cout, v1, n);
				cout << endl;
				}
			}
		}
	x = set[0];
	z = set[size - 1];
	two = F->add(1, 1);
	unrank_point(v1, 1, x, verbose_level - 1);
	unrank_point(v3, 1, z, verbose_level - 1);

	m1[0] = F->mult(two, v1[0]);
	m1[1] = v1[2];
	m1[2] = v1[1];
	m1[3] = v1[4];
	m1[4] = v1[3];

	//fxz = evaluate_bilinear_form(v1, v3, 1);
	// too slow !!!
	fxz = F->add5(
			F->mult(m1[0], v3[0]),
			F->mult(m1[1], v3[1]),
			F->mult(m1[2], v3[2]),
			F->mult(m1[3], v3[3]),
			F->mult(m1[4], v3[4])
		);

	m3[0] = F->mult(two, v3[0]);
	m3[1] = v3[2];
	m3[2] = v3[1];
	m3[3] = v3[4];
	m3[4] = v3[3];


	if (f_vv) {
		l1 = F->log_alpha(fxz);
		cout << "fxz=" << fxz << " (log " << l1 << ") ";
		if (EVEN(l1))
			cout << "+";
		else
			cout << "-";
		cout << endl;
		}

	for (i = 1; i < size - 1; i++) {

		y = set[i];

		unrank_point(v2, 1, y, verbose_level - 1);

		//fxy = evaluate_bilinear_form(v1, v2, 1);
		fxy = F->add5(
				F->mult(m1[0], v2[0]),
				F->mult(m1[1], v2[1]),
				F->mult(m1[2], v2[2]),
				F->mult(m1[3], v2[3]),
				F->mult(m1[4], v2[4])
			);

		//fyz = evaluate_bilinear_form(v2, v3, 1);
		fyz = F->add5(
				F->mult(m3[0], v2[0]),
				F->mult(m3[1], v2[1]),
				F->mult(m3[2], v2[2]),
				F->mult(m3[3], v2[3]),
				F->mult(m3[4], v2[4])
			);

		a = F->product3(fxy, fxz, fyz);
		if (f_vv) {
			l2 = F->log_alpha(fxy);
			l3 = F->log_alpha(fyz);
			cout << "i=" << i << " fxy=" << fxy << " (log=" << l2
				<< ") fyz=" << fyz << " (log=" << l3
				<< ") a=" << a << endl;
			}


		if (f_is_minus_square[a]) {
			f_OK = FALSE;
			if (f_v) {
				l1 = F->log_alpha(fxz);
				l2 = F->log_alpha(fxy);
				l3 = F->log_alpha(fyz);
				cout << "not OK; i=" << i << endl;
				cout << "{x,y,z}={" << x << "," << y
						<< "," << z << "}" << endl;
				Int_vec_print(cout, v1, n);
				cout << endl;
				Int_vec_print(cout, v2, n);
				cout << endl;
				Int_vec_print(cout, v3, n);
				cout << endl;
				cout << "fxz=" << fxz << " ";
				if (EVEN(l1))
					cout << "+";
				else
					cout << "-";
				cout << " (log=" << l1 << ")" << endl;
				cout << "fxy=" << fxy << " ";
				if (EVEN(l2))
					cout << "+";
				else
					cout << "-";
				cout << " (log=" << l2 << ")" << endl;
				cout << "fyz=" << fyz << " ";
				if (EVEN(l3))
					cout << "+";
				else
					cout << "-";
				cout << " (log=" << l3 << ")" << endl;
				cout << "a=" << a << "(log=" << F->log_alpha(a)
						<< ") is the negative of a square" << endl;
				print_minus_square_tables();
				}
			break;
			}
		}

	if (f_v) {
		if (!f_OK) {
			cout << "BLT_test fails" << endl;
			}
		else {
			cout << endl;
			}
		}
	return f_OK;
}

int orthogonal::collinearity_test(int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, x, y;
	int f_OK = TRUE;
	int fxy;

	if (f_v) {
		cout << "collinearity test for" << endl;
		for (i = 0; i < size; i++) {
			unrank_point(v1, 1, set[i], verbose_level - 1);
			//Q_epsilon_unrank(*M->GFq, u, 1, epsilon, k,
				//form_c1, form_c2, form_c3, line[i]);
			Int_vec_print(cout, v1, 5);
			cout << endl;
			}
		}
	y = set[size - 1];
	//Q_epsilon_unrank(*M->GFq, v, 1, epsilon, k,
	//form_c1, form_c2, form_c3, y);
	unrank_point(v1, 1, y, verbose_level - 1);

	for (i = 0; i < size - 1; i++) {
		x = set[i];
		unrank_point(v2, 1, x, verbose_level - 1);
		//Q_epsilon_unrank(*M->GFq, u, 1, epsilon, k,
		//form_c1, form_c2, form_c3, x);

		//fxy = evaluate_bilinear_form(*M->GFq, u, v, d, Gram);
		fxy = evaluate_bilinear_form(v1, v2, 1);

		if (fxy == 0) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y
						<< "} are collinear" << endl;
				Int_vec_print(cout, v1, 5);
				cout << endl;
				Int_vec_print(cout, v2, 5);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}

	if (f_v) {
		if (!f_OK) {
			cout << "collinearity test fails" << endl;
			}
		}
	return f_OK;
}

int orthogonal::triple_is_collinear(long int pt1, long int pt2, long int pt3)
{
	int verbose_level = 0;
	int rk;
	int *base_cols;

	base_cols = NEW_int(n);
	unrank_point(T1, 1, pt1, verbose_level - 1);
	unrank_point(T1 + n, 1, pt2, verbose_level - 1);
	unrank_point(T1 + 2 * n, 1, pt3, verbose_level - 1);
	rk = F->Linear_algebra->Gauss_int(T1,
			FALSE /* f_special */,
			FALSE /* f_complete */,
			base_cols,
			FALSE /* f_P */, NULL, 3, n, 0,
			0 /* verbose_level */);
	FREE_int(base_cols);
	if (rk < 2) {
		cout << "orthogonal::triple_is_collinear rk < 2" << endl;
		exit(1);
		}
	if (rk == 2) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int orthogonal::is_minus_square(int i)
{
	if (DOUBLYEVEN(q - 1)) {
		if (EVEN(i)) {
			return TRUE;
			}
		else {
			return FALSE;
			}
		}
	else {
		if (EVEN(i)) {
			return FALSE;
			}
		else {
			return TRUE;
			}
		}
}

void orthogonal::print_minus_square_tables()
{
	int i;

	cout << "field element indices and f_minus_square:" << endl;
	for (i = 0; i < q; i++) {
			cout << i << " : "
			<< setw(3) << index_minus_square[i] << ","
			<< setw(3) << index_minus_square_without[i] << ","
			<< setw(3) << index_minus_nonsquare[i] << " : "
			<< setw(3) << f_is_minus_square[i] << endl;
		}
}

// formerly DISCRETA/extras.cpp
//
// Anton Betten
// Sept 17, 2010

// plane_invariant started 2/23/09


void orthogonal::plane_invariant(unusual_model *U,
	int size, int *set,
	int &nb_planes, int *&intersection_matrix,
	int &Block_size, int *&Blocks,
	int verbose_level)
// using hash values
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *Mtx;
	int *Hash;
	int rk, H, log2_of_q, n_choose_k;
	int f_special = FALSE;
	int f_complete = TRUE;
	int base_col[1000];
	int subset[1000];
	int level = 3;
	int n = 5;
	int cnt;
	int i;
	int q;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	data_structures::algorithms Algo;



	q = F->q;
	n_choose_k = Combi.int_n_choose_k(size, level);
	log2_of_q = NT.int_log2(q);

	Mtx = NEW_int(level * n);
	Hash = NEW_int(n_choose_k);

	Combi.first_k_subset(subset, size, level);
	cnt = -1;

	if (f_v) {
		cout << "computing planes spanned by 3-subsets" << endl;
		cout << "n_choose_k=" << n_choose_k << endl;
		cout << "log2_of_q=" << log2_of_q << endl;
		}
	while (TRUE) {
		cnt++;

		for (i = 0; i < level; i++) {
			F->Orthogonal_indexing->Q_unrank(Mtx + i * n, 1, n - 1, set[subset[i]], 0 /* verbose_level */);
			}
		if (f_vvv) {
			cout << "subset " << setw(5) << cnt << " : ";
			Int_vec_print(cout, subset, level);
			cout << " : "; // << endl;
			}
		//print_integer_matrix_width(cout, Mtx, level, n, n, 3);
		rk = F->Linear_algebra->Gauss_int(Mtx, f_special, f_complete,
				base_col, FALSE, NULL, level, n, n, 0);
		if (f_vvv) {
			cout << "after Gauss, rank = " << rk << endl;
			Int_vec_print_integer_matrix_width(cout, Mtx, level, n, n, 3);
			}
		H = 0;
		for (i = 0; i < level * n; i++) {
			H = Algo.hashing_fixed_width(H, Mtx[i], log2_of_q);
			}
		if (f_vvv) {
			cout << "hash =" << setw(10) << H << endl;
			}
		Hash[cnt] = H;
		if (!Combi.next_k_subset(subset, size, level)) {
			break;
			}
		}
	int *Hash_sorted, *sorting_perm, *sorting_perm_inv,
		nb_types, *type_first, *type_len;

	Sorting.int_vec_classify(n_choose_k, Hash, Hash_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);


	if (f_v) {
		cout << nb_types << " types of planes" << endl;
		}
	if (f_vvv) {
		for (i = 0; i < nb_types; i++) {
			cout << setw(3) << i << " : "
				<< setw(4) << type_first[i] << " : "
				<< setw(4) << type_len[i] << " : "
				<< setw(10) << Hash_sorted[type_first[i]] << endl;
			}
		}
	int *type_len_sorted, *sorting_perm2, *sorting_perm_inv2,
		nb_types2, *type_first2, *type_len2;

	Sorting.int_vec_classify(nb_types, type_len, type_len_sorted,
		sorting_perm2, sorting_perm_inv2,
		nb_types2, type_first2, type_len2);

	if (f_v) {
		cout << "multiplicities:" << endl;
		for (i = 0; i < nb_types2; i++) {
			//cout << setw(3) << i << " : "
			//<< setw(4) << type_first2[i] << " : "
			cout << setw(4) << type_len2[i] << " x "
				<< setw(10) << type_len_sorted[type_first2[i]] << endl;
			}
		}
	int f, ff, ll, j, u, ii, jj, idx;

	f = type_first2[nb_types2 - 1];
	nb_planes = type_len2[nb_types2 - 1];
	if (f_v) {
		if (nb_planes == 1) {
			cout << "there is a unique plane that appears "
					<< type_len_sorted[f]
					<< " times among the 3-sets of points" << endl;
			}
		else {
			cout << "there are " << nb_planes
					<< " planes that each appear "
					<< type_len_sorted[f]
					<< " times among the 3-sets of points" << endl;
			for (i = 0; i < nb_planes; i++) {
				j = sorting_perm_inv2[f + i];
				cout << "The " << i << "-th plane, which is " << j
						<< ", appears " << type_len_sorted[f + i]
						<< " times" << endl;
				}
			}
		}
	if (f_vvv) {
		cout << "these planes are:" << endl;
		for (i = 0; i < nb_planes; i++) {
			cout << "plane " << i << endl;
			j = sorting_perm_inv2[f + i];
			ff = type_first[j];
			ll = type_len[j];
			for (u = 0; u < ll; u++) {
				cnt = sorting_perm_inv[ff + u];
				Combi.unrank_k_subset(cnt, subset, size, level);
				cout << "subset " << setw(5) << cnt << " : ";
				Int_vec_print(cout, subset, level);
				cout << " : " << endl;
				}
			}
		}

	//return;

	//int *Blocks;
	int *Block;
	//int Block_size;


	Block = NEW_int(size);
	Blocks = NEW_int(nb_planes * size);

	for (i = 0; i < nb_planes; i++) {
		j = sorting_perm_inv2[f + i];
		ff = type_first[j];
		ll = type_len[j];
		if (f_vv) {
			cout << setw(3) << i << " : " << setw(3) << " : "
				<< setw(4) << ff << " : "
				<< setw(4) << ll << " : "
				<< setw(10) << Hash_sorted[type_first[j]] << endl;
			}
		Block_size = 0;
		for (u = 0; u < ll; u++) {
			cnt = sorting_perm_inv[ff + u];
			Combi.unrank_k_subset(cnt, subset, size, level);
			if (f_vvv) {
				cout << "subset " << setw(5) << cnt << " : ";
				Int_vec_print(cout, subset, level);
				cout << " : " << endl;
				}
			for (ii = 0; ii < level; ii++) {
				F->Orthogonal_indexing->Q_unrank(Mtx + ii * n, 1, n - 1, set[subset[ii]], 0 /* verbose_level */);
				}
			for (ii = 0; ii < level; ii++) {
				if (!Sorting.int_vec_search(Block, Block_size, subset[ii], idx)) {
					for (jj = Block_size; jj > idx; jj--) {
						Block[jj] = Block[jj - 1];
						}
					Block[idx] = subset[ii];
					Block_size++;
					}
				}
			rk = F->Linear_algebra->Gauss_int(Mtx, f_special,
					f_complete, base_col, FALSE, NULL, level, n, n, 0);
			if (f_vvv)  {
				cout << "after Gauss, rank = " << rk << endl;
				Int_vec_print_integer_matrix_width(cout, Mtx, level, n, n, 3);
				}

			H = 0;
			for (ii = 0; ii < level * n; ii++) {
				H = Algo.hashing_fixed_width(H, Mtx[ii], log2_of_q);
				}
			if (f_vvv) {
				cout << "hash =" << setw(10) << H << endl;
				}
			}
		if (f_vv) {
			cout << "found Block ";
			Int_vec_print(cout, Block, Block_size);
			cout << endl;
			}
		for (u = 0; u < Block_size; u++) {
			Blocks[i * Block_size + u] = Block[u];
			}
		}
	if (f_vv) {
		cout << "Incidence structure between points "
				"and high frequency planes:" << endl;
		if (nb_planes < 30) {
			Int_vec_print_integer_matrix_width(cout, Blocks,
					nb_planes, Block_size, Block_size, 3);
			}
		}

	int *Incma, *Incma_t, *IIt, *ItI;
	int a;

	Incma = NEW_int(size * nb_planes);
	Incma_t = NEW_int(nb_planes * size);
	IIt = NEW_int(size * size);
	ItI = NEW_int(nb_planes * nb_planes);


	for (i = 0; i < size * nb_planes; i++) {
		Incma[i] = 0;
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < Block_size; j++) {
			a = Blocks[i * Block_size + j];
			Incma[a * nb_planes + i] = 1;
			}
		}
	if (f_vv) {
		cout << "Incidence matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Incma,
				size, nb_planes, nb_planes, 1);
		}
	for (i = 0; i < size; i++) {
		for (j = 0; j < nb_planes; j++) {
			Incma_t[j * size + i] = Incma[i * nb_planes + j];
			}
		}
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] * Incma_t[u * size + j];
				}
			IIt[i * size + j] = a;
			}
		}
	if (f_vv) {
		cout << "I * I^\\top = " << endl;
		Int_vec_print_integer_matrix_width(cout, IIt, size, size, size, 2);
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < size; u++) {
				a += Incma[u * nb_planes + i] * Incma[u * nb_planes + j];
				}
			ItI[i * nb_planes + j] = a;
			}
		}
	if (f_v) {
		cout << "I^\\top * I = " << endl;
		Int_vec_print_integer_matrix_width(cout, ItI,
				nb_planes, nb_planes, nb_planes, 3);
		}

	intersection_matrix = NEW_int(nb_planes * nb_planes);
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			intersection_matrix[i * nb_planes + j] = ItI[i * nb_planes + j];
			}
		}

#if 0
	{
		char fname[1000];

		snprintf(fname, 1000, "plane_invariant_%d_%d.txt", q, k);

		ofstream fp(fname);
		fp << nb_planes << endl;
		for (i = 0; i < nb_planes; i++) {
			for (j = 0; j < nb_planes; j++) {
				fp << ItI[i * nb_planes + j] << " ";
				}
			fp << endl;
			}
		fp << -1 << endl;
		fp << "# Incidence structure between points "
				"and high frequency planes:" << endl;
		fp << l << " " << Block_size << endl;
		print_integer_matrix_width(fp,
				Blocks, nb_planes, Block_size, Block_size, 3);
		fp << -1 << endl;

	}
#endif

	FREE_int(Mtx);
	FREE_int(Hash);
	FREE_int(Block);
	//FREE_int(Blocks);
	FREE_int(Incma);
	FREE_int(Incma_t);
	FREE_int(IIt);
	FREE_int(ItI);


	FREE_int(Hash_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);



	FREE_int(type_len_sorted);
	FREE_int(sorting_perm2);
	FREE_int(sorting_perm_inv2);
	FREE_int(type_first2);
	FREE_int(type_len2);



}




}}}

