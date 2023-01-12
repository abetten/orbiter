/*
 * finite_field_implementation_wo_tables.cpp
 *
 *  Created on: Sep 6, 2021
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {



finite_field_implementation_wo_tables::finite_field_implementation_wo_tables()
{
	F = NULL;
	v1 = v2 = v3 = NULL;
	GFp = NULL;
	FX = NULL;
	m = NULL;
	factor_polynomial_degree = 0;
	factor_polynomial_coefficients_negated = NULL;
	Fq = NULL;
	Alpha = NULL;
}

finite_field_implementation_wo_tables::~finite_field_implementation_wo_tables()
{
	if (v1) {
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
	if (v3) {
		FREE_int(v3);
	}
	if (factor_polynomial_coefficients_negated) {
		FREE_int(factor_polynomial_coefficients_negated);
	}
}

void finite_field_implementation_wo_tables::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init" << endl;
	}

	finite_field_implementation_wo_tables::F = F;


	v1 = NEW_int(F->e);
	v2 = NEW_int(F->e);
	v3 = NEW_int(F->e);



	if (F->e > 1) {

		if (f_v) {
			cout << "finite_field_implementation_wo_tables::init "
					"The field is an extension field" << endl;
		}
		GFp = NEW_OBJECT(finite_field);

		// we assiume that the prime field is small enough for tables to be created:

		if (f_v) {
			cout << "finite_field_implementation_wo_tables::init "
					"before GFp->finite_field_init_small_order" << endl;
		}
		GFp->finite_field_init_small_order(F->p,
				FALSE /* f_without_tables */,
				verbose_level - 1);
		if (f_v) {
			cout << "finite_field_implementation_wo_tables::init "
					"after GFp->finite_field_init_small_order" << endl;
		}

		if (f_v) {
			cout << "finite_field_implementation_wo_tables::init "
					"before init_extension_field" << endl;
		}
		init_extension_field(verbose_level - 1);
		if (f_v) {
			cout << "finite_field_implementation_wo_tables::init "
					"after init_extension_field" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "finite_field_implementation_wo_tables::init "
					"the field is a prime field. Nothing to do for now." << endl;
		}

	}

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init done" << endl;
	}
}

void finite_field_implementation_wo_tables::init_extension_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field" << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"the field is an extension field "
				"of degree " << F->e << " over the prime field." << endl;
	}

	FX = NEW_OBJECT(ring_theory::unipoly_domain);


	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"we will be using the polynomial " << F->my_poly << endl;
	}


	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"before FX->init_basic" << endl;
	}

	FX->init_basic(GFp, verbose_level - 2);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"after FX->init_basic" << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"before FX->create_object_by_rank_string" << endl;
	}
	FX->create_object_by_rank_string(m, F->my_poly, verbose_level - 2);
	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"after FX->create_object_by_rank_string" << endl;
	}
	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field m=";
		FX->print_object(m, cout);
		cout << endl;
	}

	Fq = NEW_OBJECT(ring_theory::unipoly_domain);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"before Fq->init_factorring" << endl;
	}
	Fq->init_factorring(GFp, m, verbose_level - 1);
	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"after Fq->init_factorring" << endl;
	}

	int i;

	factor_polynomial_degree = Fq->degree(m);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"factor_polynomial_degree = " << factor_polynomial_degree << endl;
	}


	factor_polynomial_coefficients_negated =
			NEW_int(factor_polynomial_degree + 1);

	for (i = 0; i <= factor_polynomial_degree; i++) {
		factor_polynomial_coefficients_negated[i] = F->negate(Fq->s_i(m, i));
	}


	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"before Fq->create_object_by_rank" << endl;
	}
	Fq->create_object_by_rank(Alpha, F->alpha,
			__FILE__, __LINE__, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field "
				"after Fq->create_object_by_rank" << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::init_extension_field done" << endl;
	}
}

int finite_field_implementation_wo_tables::mult(int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::mult" << endl;
	}

	long int k;

	if (F->e > 1) {
		ring_theory::unipoly_object a, b, c;

		Fq->create_object_by_rank(a, i,
				__FILE__, __LINE__, 0 /*verbose_level - 2*/);
		Fq->create_object_by_rank(b, j,
				__FILE__, __LINE__, 0 /*verbose_level - 2*/);
		Fq->create_object_of_degree_no_test(c, factor_polynomial_degree);

		if (f_v) {
			cout << "a=" << endl;
			Fq->print_object(a, cout);
			cout << endl;
			cout << "b=" << endl;
			Fq->print_object(b, cout);
			cout << endl;
			cout << "finite_field_implementation_wo_tables::mult "
					"before Fq->mult_mod_negated" << endl;
		}

		Fq->mult_mod_negated(a, b, c,
				factor_polynomial_degree,
				factor_polynomial_coefficients_negated,
				verbose_level);

		if (f_v) {
			cout << "finite_field_implementation_wo_tables::mult "
					"after Fq->mult_mod_negated" << endl;
		}

		k = Fq->rank(c);
		if (f_v) {
			cout << "c=" << endl;
			Fq->print_object(c, cout);
			cout << endl;
			cout << "i=" << i << ", j=" << j << " k=" << k << endl;
		}

		Fq->delete_object(a);
		Fq->delete_object(b);
		Fq->delete_object(c);
	}
	else {

		number_theory::number_theory_domain NT;

		k = NT.mult_mod(i, j, F->p);
		// possibility of int overflow is eliminated in
		// number_theory::number_theory_domain class
		// by using longinteger objects.

		// To test, use large primes below a power of two as recorded in
		// https://primes.utm.edu/lists/2small/0bit.html
		// for primes less than 2 to the power n - 1.


	}
	if (f_v) {
		cout << "finite_field_implementation_wo_tables::mult done" << endl;
	}
	return k;
}

int finite_field_implementation_wo_tables::inverse(int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::inverse" << endl;
	}

	long int k;

	if (F->e > 1) {
		ring_theory::unipoly_object a, u, v, g;

		Fq->create_object_by_rank(a, i,
				__FILE__, __LINE__, 0 /*verbose_level - 2*/);
		Fq->create_object_of_degree_no_test(u, factor_polynomial_degree);
		Fq->create_object_of_degree_no_test(v, factor_polynomial_degree);
		Fq->create_object_of_degree_no_test(g, factor_polynomial_degree);

		if (f_v) {
			cout << "a=" << endl;
			Fq->print_object(a, cout);
			cout << endl;
			cout << "finite_field_implementation_wo_tables::inverse "
					"before Fq->extended_gcd" << endl;
		}

		Fq->extended_gcd(m, a, u, v, g, verbose_level);

		if (f_v) {
			cout << "finite_field_implementation_wo_tables::inverse "
					"after Fq->extended_gcd" << endl;
		}

		k = Fq->rank(v);
		if (f_v) {
			cout << "v=" << endl;
			Fq->print_object(v, cout);
			cout << endl;
			cout << "i=" << i << ", k=" << k << endl;
		}

		Fq->delete_object(a);
		Fq->delete_object(u);
		Fq->delete_object(v);
		Fq->delete_object(g);
	}
	else {
		number_theory::number_theory_domain NT;

		k = NT.inverse_mod(i, F->p);

	}

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::inverse done" << endl;
	}
	return k;
}

int finite_field_implementation_wo_tables::negate(int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::negate" << endl;
	}

	long int k;
	number_theory::number_theory_domain NT;

	if (F->e > 1) {
		geometry::geometry_global Gg;

		if (i < 0 || i >= F->q) {
			cout << "finite_field_implementation_wo_tables::negate "
					"out of range, i = " << i << endl;
			exit(1);
		}
		long int l;

		Gg.AG_element_unrank(F->p, v1, 1, F->e, i);
		for (l = 0; l < F->e; l++) {
			// v2[l] = (F->p - v1[l]) % F->p;
			v2[l] = NT.int_negate(v1[l], F->p);
		}
		k = Gg.AG_element_rank(F->p, v2, 1, F->e);
	}
	else {

		k = NT.int_negate(i, F->p);

	}
	if (f_v) {
		cout << "finite_field_implementation_wo_tables::negate done" << endl;
	}
	return k;
}

int finite_field_implementation_wo_tables::add(int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::add" << endl;
	}

	geometry::geometry_global Gg;

	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_wo_tables::add "
				"out of range, i = " << i << endl;
		exit(1);
	}
	if (j < 0 || j >= F->q) {
		cout << "finite_field_implementation_wo_tables::add "
				"out of range, j = " << j << endl;
		exit(1);
	}

	long int k;
	number_theory::number_theory_domain NT;

	if (F->e > 1) {
		long int l;

		Gg.AG_element_unrank(F->p, v1, 1, F->e, i);
		Gg.AG_element_unrank(F->p, v2, 1, F->e, j);
		for (l = 0; l < F->e; l++) {
			//v3[l] = (v1[l] + v2[l]) % F->p;
			v3[l] = NT.add_mod(v1[l], v2[l], F->p);
		}
		k = Gg.AG_element_rank(F->p, v3, 1, F->e);
	}
	else {

		k = NT.add_mod(i, j, F->p);
	}

	if (f_v) {
		cout << "finite_field_implementation_wo_tables::add done" << endl;
	}
	return k;
}



}}}




