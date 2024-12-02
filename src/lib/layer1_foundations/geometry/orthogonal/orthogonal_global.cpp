/*
 * orthogonal_global.cpp
 *
 *  Created on: Sep 30, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace orthogonal_geometry {


orthogonal_global::orthogonal_global()
{
	Record_birth();

}

orthogonal_global::~orthogonal_global()
{
	Record_death();

}



#if 0
void orthogonal_global::create_Linear_BLT_set(long int *set, int *ABC, int verbose_level)
// a(t)= 1, b(t) = t, c(t) = t^2, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int i, a, b, c;

	int q = F->q;

	if (f_v) {
		cout << "orthogonal_global::create_Linear_BLT_set" << endl;
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
		cout << "orthogonal_global::create_Linear_BLT_set done" << endl;
	}
}

#endif


void orthogonal_global::create_BLT_set_from_flock(
		orthogonal *O,
		long int *set, int *ABC, int verbose_level)
// output: set[q + 1]
// input: ABC[q * 3]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int i, a, b, c;
	geometry::other_geometry::geometry_global Gg;


	if (f_v) {
		cout << "orthogonal_global::create_BLT_set_from_flock" << endl;
	}

	int q = O->F->q;

	for (i = 0; i < q; i++) {

		a = ABC[i * 3 + 0];
		b = ABC[i * 3 + 1];
		c = ABC[i * 3 + 2];

		Gg.create_BLT_point_from_flock(O->F, v, a, b, c, verbose_level - 2);


		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = O->Hyperbolic_pair->rank_point(v, 1, 0);

	}
	other::orbiter_kernel_system::Orbiter->Int_vec->init5(v, 0, 1, 0, 0, 0);
	if (f_vv) {
		cout << "point : ";
		Int_vec_print(cout, v, 5);
		cout << endl;
	}
	set[q] = O->Hyperbolic_pair->rank_point(v, 1, 0);

	if (f_v) {
		cout << "orthogonal_global::create_BLT_set_from_flock done" << endl;
	}
}


void orthogonal_global::create_FTWKB_flock(
		orthogonal *O,
		int *ABC, int verbose_level)
// for q congruent 2 mod 3
// a(t)= t, b(t) = 3*t^2, c(t) = 3*t^3, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int v[5];
	int r, i, a, b, c;
	//geometry::geometry_global Gg;

	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock" << endl;
	}

	int q = O->F->q;

	if (q <= 5) {
		cout << "orthogonal_global::create_FTWKB_flock "
				"q <= 5" << endl;
		exit(1);
	}
	r = q % 3;
	if (r != 2) {
		cout << "orthogonal_global::create_FTWKB_flock "
				"q mod 3 must be 2" << endl;
		exit(1);
	}
	Int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = O->F->mult(3, O->F->power(i, 2));
		c = O->F->mult(3, O->F->power(i, 3));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
		}
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
	}

	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock done" << endl;
	}
}

void orthogonal_global::create_K1_flock(
		orthogonal *O,
		int *ABC, int verbose_level)
// for a nonsquare m, and q=p^e
// a(t)= t, b(t) = 0, c(t) = -m*t^p, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int v[5];
	int i, m, minus_one, exponent, a, b, c;
	int q;
	//geometry::geometry_global Gg;

	if (f_v) {
		cout << "orthogonal_global::create_K1_BLT_set" << endl;
	}
	q = O->F->q;
	m = O->F->p; // the primitive element is a nonsquare
	exponent = O->F->p;
	minus_one = O->F->negate(1);
	if (f_v) {
		cout << "m=" << m << endl;
		cout << "exponent=" << exponent << endl;
		cout << "minus_one=" << minus_one << endl;
	}
	Int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = 0;
		c = O->F->mult(minus_one, O->F->mult(m, O->F->power(i, exponent)));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
		}
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
	}

	if (f_v) {
		cout << "orthogonal_global::create_K1_BLT_set done" << endl;
	}
}

void orthogonal_global::create_K2_flock(
		orthogonal *O,
		int *ABC, int verbose_level)
// for q congruent 2 or 3 mod 5
// a(t)= t, b(t) = 5*t^3, c(t) = 5*t^5, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int v[5];
	int five, r, i, a, b, c;
	int q;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "orthogonal_global::create_K2_flock" << endl;
	}
	q = O->F->q;
	if (q <= 5) {
		cout << "orthogonal_global::create_K2_flock q <= 5" << endl;
		return;
	}
	r = q % 5;
	if (r != 2 && r != 3) {
		cout << "orthogonal_global::create_K2_flock "
				"q mod 5 must be 2 or 3" << endl;
		return;
	}
	five = 5 % O->F->p;
	Int_vec_zero(ABC, 3 * (q + 1));
	for (i = 0; i < q; i++) {
		a = i;
		b = O->F->mult(five, O->F->power(i, 3));
		c = O->F->mult(five, O->F->power(i, 5));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
		}
		ABC[i * 3 + 0] = a;
		ABC[i * 3 + 1] = b;
		ABC[i * 3 + 2] = c;
	}
	if (f_v) {
		cout << "orthogonal_global::create_K2_flock done" << endl;
	}
}


void orthogonal_global::create_FTWKB_flock_and_BLT_set(
		orthogonal *O,
		long int *set, int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set" << endl;
	}
	int q;

	q = O->F->q;

	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set "
				"before create_FTWKB_flock" << endl;
	}
	create_FTWKB_flock(O,
			ABC, verbose_level);
	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set "
				"after create_FTWKB_flock" << endl;
	}

	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set "
				"before create_BLT_set_from_flock" << endl;
	}
	create_BLT_set_from_flock(O,
			set, ABC, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set "
				"after create_BLT_set_from_flock" << endl;
	}

	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set "
				"the BLT set K1 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_global::create_FTWKB_flock_and_BLT_set done" << endl;
	}

}



void orthogonal_global::create_K1_flock_and_BLT_set(
		orthogonal *O,
		long int *set, int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set" << endl;
	}
	int q;

	q = O->F->q;

	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set "
				"before create_K1_flock" << endl;
	}
	create_K1_flock(O,
			ABC, verbose_level);
	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set "
				"after create_K1_flock" << endl;
	}

	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set "
				"before create_BLT_set_from_flock" << endl;
	}
	create_BLT_set_from_flock(O,
			set, ABC, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set "
				"after create_BLT_set_from_flock" << endl;
	}

	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set "
				"the BLT set K1 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_global::create_K1_flock_and_BLT_set done" << endl;
	}

}


void orthogonal_global::create_K2_flock_and_BLT_set(
		orthogonal *O,
		long int *set, int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set" << endl;
	}
	int q;

	q = O->F->q;

	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set "
				"before create_K2_flock" << endl;
	}
	create_K2_flock(O,
			ABC, verbose_level);
	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set "
				"after create_K2_flock" << endl;
	}

	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set "
				"before create_BLT_set_from_flock" << endl;
	}
	create_BLT_set_from_flock(O,
			set, ABC, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set "
				"after create_BLT_set_from_flock" << endl;
	}

	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set the BLT set K2 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_global::create_K2_flock_and_BLT_set done" << endl;
	}

}

void orthogonal_global::create_LP_37_72_BLT_set(
		orthogonal *O,
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

	q = O->F->q;
	if (q != 37) {
		cout << "orthogonal_global::create_LP_37_72_BLT_set q = 37" << endl;
		return;
	}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		other::orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = O->Hyperbolic_pair->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal_global::create_LP_37_72_BLT_set "
				"the BLT set LP_37_72 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
}

void orthogonal_global::create_LP_37_4a_BLT_set(
		orthogonal *O,
		long int *set, int verbose_level)
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

	q = O->F->q;
	if (q != 37) {
		cout << "orthogonal_global::create_LP_37_4a_BLT_set q = 37" << endl;
		return;
	}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		other::orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = O->Hyperbolic_pair->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal_global::create_LP_37_4a_BLT_set "
				"the BLT set LP_37_4a is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
}

void orthogonal_global::create_LP_37_4b_BLT_set(
		orthogonal *O,
		long int *set, int verbose_level)
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

	q = O->F->q;
	if (q != 37) {
		cout << "orthogonal_global::create_LP_37_4b_BLT_set q = 37" << endl;
		return;
	}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		other::orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = O->Hyperbolic_pair->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal_global::create_LP_37_4b_BLT_set "
				"the BLT set LP_37_4b is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
}

void orthogonal_global::create_Law_71_BLT_set(
		orthogonal *O,
		long int *set, int verbose_level)
// This example can be found in Maska Law's thesis on page 115.
// Maska Law: Flocks, generalised quadrangles
// and translation planes from BLT-sets,
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

	q = O->F->q;
	if (q != 71) {
		cout << "orthogonal_global::create_Law_71_BLT_set q = 71" << endl;
		return;
	}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		other::orbiter_kernel_system::Orbiter->Int_vec->init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << endl;
		}
		set[i] = O->Hyperbolic_pair->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal_global::create_Law_71_BLT_set "
				"the BLT set LP_71 is ";
		Lint_vec_print(cout, set, q + 1);
		cout << endl;
	}
}


int orthogonal_global::BLT_test_full(
		orthogonal *O,
		int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::BLT_test_full" << endl;
	}
	if (!collinearity_test(O, size, set, 0/*verbose_level - 2*/)) {
		if (f_v) {
			cout << "orthogonal_global::BLT_test_full "
					"fails due to collinearity_test" << endl;
		}
		return false;
	}
	if (!BLT_test(O, size, set, verbose_level)) {
		if (f_v) {
			cout << "orthogonal_global::BLT_test_full "
					"fails due to BLT_test" << endl;
		}
		return false;
	}
	if (f_v) {
		cout << "orthogonal_global::BLT_test_full "
				"passes" << endl;
	}
	return true;
}

int orthogonal_global::BLT_test(
		orthogonal *O,
		int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, x, y, z, a;
	int f_OK = true;
	int fxy, fxz, fyz, l1, l2, l3;
	int two;
	int m1[5], m3[5];

	if (size <= 2) {
		return true;
	}
	if (f_v) {
		cout << "orthogonal_global::BLT_test BLT_test "
				"for" << endl;
		Lint_vec_print(cout, set, size);
		cout << endl;
		if (f_vv) {
			cout << "orthogonal_global::BLT_test "
					"the set of points is:" << endl;
			for (i = 0; i < size; i++) {
				O->Hyperbolic_pair->unrank_point(
						O->Hyperbolic_pair->v1, 1, set[i], 0 /*verbose_level - 1*/);
				cout << i << " : " << set[i] << " : ";
				Int_vec_print(cout, O->Hyperbolic_pair->v1, O->Quadratic_form->n);
				cout << endl;
			}
		}
	}
	x = set[0];
	z = set[size - 1];
	two = O->F->add(1, 1);
	O->Hyperbolic_pair->unrank_point(
			O->Hyperbolic_pair->v1, 1, x, 0 /*verbose_level - 1*/);
	O->Hyperbolic_pair->unrank_point(
			O->Hyperbolic_pair->v3, 1, z, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "v1=";
		Int_vec_print(cout, O->Hyperbolic_pair->v1, O->Quadratic_form->n);
		cout << endl;
		cout << "v3=";
		Int_vec_print(cout, O->Hyperbolic_pair->v3, O->Quadratic_form->n);
		cout << endl;
	}

	m1[0] = O->F->mult(two, O->Hyperbolic_pair->v1[0]);
	m1[1] = O->Hyperbolic_pair->v1[2];
	m1[2] = O->Hyperbolic_pair->v1[1];
	m1[3] = O->Hyperbolic_pair->v1[4];
	m1[4] = O->Hyperbolic_pair->v1[3];

	//fxz = evaluate_bilinear_form(v1, v3, 1);
	// too slow !!!
	fxz = O->F->add5(
			O->F->mult(m1[0], O->Hyperbolic_pair->v3[0]),
			O->F->mult(m1[1], O->Hyperbolic_pair->v3[1]),
			O->F->mult(m1[2], O->Hyperbolic_pair->v3[2]),
			O->F->mult(m1[3], O->Hyperbolic_pair->v3[3]),
			O->F->mult(m1[4], O->Hyperbolic_pair->v3[4])
		);

	m3[0] = O->F->mult(two, O->Hyperbolic_pair->v3[0]);
	m3[1] = O->Hyperbolic_pair->v3[2];
	m3[2] = O->Hyperbolic_pair->v3[1];
	m3[3] = O->Hyperbolic_pair->v3[4];
	m3[4] = O->Hyperbolic_pair->v3[3];


	if (f_vv) {
		l1 = O->F->log_alpha(fxz);
		cout << "fxz=" << fxz << " (log " << l1 << ") ";
		if (EVEN(l1))
			cout << "+";
		else
			cout << "-";
		cout << endl;
	}

	for (i = 1; i < size - 1; i++) {

		y = set[i];

		O->Hyperbolic_pair->unrank_point(
				O->Hyperbolic_pair->v2, 1, y, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "i=" << i << " v2=";
			Int_vec_print(cout, O->Hyperbolic_pair->v2, O->Quadratic_form->n);
			cout << endl;
		}

		//fxy = evaluate_bilinear_form(v1, v2, 1);
		fxy = O->F->add5(
				O->F->mult(m1[0], O->Hyperbolic_pair->v2[0]),
				O->F->mult(m1[1], O->Hyperbolic_pair->v2[1]),
				O->F->mult(m1[2], O->Hyperbolic_pair->v2[2]),
				O->F->mult(m1[3], O->Hyperbolic_pair->v2[3]),
				O->F->mult(m1[4], O->Hyperbolic_pair->v2[4])
			);

		//fyz = evaluate_bilinear_form(v2, v3, 1);
		fyz = O->F->add5(
				O->F->mult(m3[0], O->Hyperbolic_pair->v2[0]),
				O->F->mult(m3[1], O->Hyperbolic_pair->v2[1]),
				O->F->mult(m3[2], O->Hyperbolic_pair->v2[2]),
				O->F->mult(m3[3], O->Hyperbolic_pair->v2[3]),
				O->F->mult(m3[4], O->Hyperbolic_pair->v2[4])
			);

		a = O->F->product3(fxy, fxz, fyz);
		if (f_vv) {
			l2 = O->F->log_alpha(fxy);
			l3 = O->F->log_alpha(fyz);
			cout << "i=" << i << " fxy=" << fxy << " (log=" << l2
				<< ") fyz=" << fyz << " (log=" << l3
				<< ") a=" << a << " log=" << O->F->log_alpha(a)
				<< " is_minus_square=" << O->SN->f_is_minus_square[a] << endl;
		}


		if (O->SN->f_is_minus_square[a]) {
			f_OK = false;
			if (f_v) {
				l1 = O->F->log_alpha(fxz);
				l2 = O->F->log_alpha(fxy);
				l3 = O->F->log_alpha(fyz);
				cout << "not OK; i=" << i << endl;
				cout << "{x,y,z}={" << x << "," << y
						<< "," << z << "}" << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v1, O->Quadratic_form->n);
				cout << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v2, O->Quadratic_form->n);
				cout << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v3, O->Quadratic_form->n);
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
				cout << "a=" << a << "(log=" << O->F->log_alpha(a)
						<< ") is the negative of a square" << endl;
				O->SN->print_minus_square_tables();
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

int orthogonal_global::collinearity_test(
		orthogonal *O,
		int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, x, y;
	int f_OK = true;
	int fxy;

	if (f_v) {
		cout << "orthogonal_global::collinearity_test "
				"collinearity test for" << endl;
		for (i = 0; i < size; i++) {
			O->Hyperbolic_pair->unrank_point(
					O->Hyperbolic_pair->v1, 1, set[i], verbose_level - 1);
			//Q_epsilon_unrank(*M->GFq, u, 1, epsilon, k,
				//form_c1, form_c2, form_c3, line[i]);
			Int_vec_print(cout, O->Hyperbolic_pair->v1, 5);
			cout << endl;
		}
	}
	y = set[size - 1];
	//Q_epsilon_unrank(*M->GFq, v, 1, epsilon, k,
	//form_c1, form_c2, form_c3, y);
	O->Hyperbolic_pair->unrank_point(
			O->Hyperbolic_pair->v1, 1, y, verbose_level - 1);

	for (i = 0; i < size - 1; i++) {
		x = set[i];
		O->Hyperbolic_pair->unrank_point(
				O->Hyperbolic_pair->v2, 1, x, verbose_level - 1);
		//Q_epsilon_unrank(*M->GFq, u, 1, epsilon, k,
		//form_c1, form_c2, form_c3, x);

		//fxy = evaluate_bilinear_form(*M->GFq, u, v, d, Gram);
		fxy = O->Quadratic_form->evaluate_bilinear_form(
				O->Hyperbolic_pair->v1, O->Hyperbolic_pair->v2, 1);

		if (fxy == 0) {
			f_OK = false;
			if (f_v) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y
						<< "} are collinear" << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v1, 5);
				cout << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v2, 5);
				cout << endl;
				cout << "fxy=" << fxy << endl;
			}
			break;
		}
	}

	if (f_v) {
		if (!f_OK) {
			cout << "orthogonal_global::collinearity_test "
					"collinearity test fails" << endl;
		}
	}
	return f_OK;
}



void orthogonal_global::create_Fisher_BLT_set(
		long int *Fisher_BLT, int *ABC,
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::create_Fisher_BLT_set" << endl;
	}
	unusual_model U;

	U.setup(FQ, Fq, verbose_level);
	U.create_Fisher_BLT_set(Fisher_BLT, ABC, verbose_level);
	if (f_v) {
		cout << "orthogonal_global::create_Fisher_BLT_set done" << endl;
	}

}

void orthogonal_global::create_Linear_BLT_set(
		long int *BLT, int *ABC,
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::create_Linear_BLT_set" << endl;
	}
	unusual_model U;

	U.setup(FQ, Fq, verbose_level);
	U.create_Linear_BLT_set(BLT, ABC, verbose_level);
	if (f_v) {
		cout << "orthogonal_global::create_Linear_BLT_set done" << endl;
	}

}

void orthogonal_global::create_Mondello_BLT_set(
		long int *BLT, int *ABC,
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_global::create_Mondello_BLT_set" << endl;
	}
	unusual_model U;

	U.setup(FQ, Fq, verbose_level);
	U.create_Mondello_BLT_set(BLT, ABC, verbose_level);
	if (f_v) {
		cout << "orthogonal_global::create_Mondello_BLT_set done" << endl;
	}

}


}}}}



