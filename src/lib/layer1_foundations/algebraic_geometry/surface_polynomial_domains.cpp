/*
 * surface_polynomial_domains.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {





surface_polynomial_domains::surface_polynomial_domains()
{
	Surf = NULL;

	nb_monomials = 0;

	Poly1 = NULL;
	Poly2 = NULL;
	Poly3 = NULL;
	Poly1_x123 = NULL;
	Poly2_x123 = NULL;
	Poly3_x123 = NULL;
	Poly4_x123 = NULL;
	Poly1_4 = NULL;
	Poly2_4 = NULL;
	Poly3_4 = NULL;

	Partials = NULL;

	f_has_large_polynomial_domains = false;
	Poly2_27 = NULL;
	Poly4_27 = NULL;
	Poly6_27 = NULL;
	Poly3_24 = NULL;

	nb_monomials2 = nb_monomials4 = nb_monomials6 = 0;
	nb_monomials3 = 0;

	Clebsch_Pij = NULL;
	Clebsch_P = NULL;
	Clebsch_P3 = NULL;
	Clebsch_coeffs = NULL;
	CC = NULL;
}

surface_polynomial_domains::~surface_polynomial_domains()
{
	if (Poly1) {
		FREE_OBJECT(Poly1);
	}
	if (Poly2) {
		FREE_OBJECT(Poly2);
	}
	if (Poly3) {
		FREE_OBJECT(Poly3);
	}
	if (Poly1_x123) {
		FREE_OBJECT(Poly1_x123);
	}
	if (Poly2_x123) {
		FREE_OBJECT(Poly2_x123);
	}
	if (Poly3_x123) {
		FREE_OBJECT(Poly3_x123);
	}
	if (Poly4_x123) {
		FREE_OBJECT(Poly4_x123);
	}
	if (Poly1_4) {
		FREE_OBJECT(Poly1_4);
	}
	if (Poly2_4) {
		FREE_OBJECT(Poly2_4);
	}
	if (Poly3_4) {
		FREE_OBJECT(Poly3_4);
	}

	if (Partials) {
		FREE_OBJECTS(Partials);
	}

	if (f_has_large_polynomial_domains) {
		if (Poly2_27) {
			FREE_OBJECT(Poly2_27);
		}
		if (Poly4_27) {
			FREE_OBJECT(Poly4_27);
		}
		if (Poly6_27) {
			FREE_OBJECT(Poly6_27);
		}
		if (Poly3_24) {
			FREE_OBJECT(Poly3_24);
		}
	}
	if (Clebsch_Pij) {
		FREE_int(Clebsch_Pij);
	}
	if (Clebsch_P) {
		FREE_pint(Clebsch_P);
	}
	if (Clebsch_P3) {
		FREE_pint(Clebsch_P3);
	}
	if (Clebsch_coeffs) {
		FREE_int(Clebsch_coeffs);
	}
	if (CC) {
		FREE_pint(CC);
	}
}


void surface_polynomial_domains::init(
		surface_domain *Surf, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::init" << endl;
	}

	surface_polynomial_domains::Surf = Surf;

	Poly1 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly1->init" << endl;
	}
	Poly1->init(Surf->F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly1->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly2->init" << endl;
	}
	Poly2->init(Surf->F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly2->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly3->init" << endl;
	}
	Poly3->init(Surf->F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly3->init" << endl;
	}

	Poly1_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly4_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly1_x123->init" << endl;
	}
	Poly1_x123->init(Surf->F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly1_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly2_x123->init" << endl;
	}
	Poly2_x123->init(Surf->F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly2_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly3_x123->init" << endl;
	}
	Poly3_x123->init(Surf->F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly3_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly4_x123->init" << endl;
	}
	Poly4_x123->init(Surf->F,
			3 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly4_x123->init" << endl;
	}


	label_variables_3(Poly1, verbose_level);
	label_variables_3(Poly2, verbose_level);
	label_variables_3(Poly3, verbose_level);

	label_variables_x123(Poly1_x123, verbose_level);
	label_variables_x123(Poly2_x123, verbose_level);
	label_variables_x123(Poly3_x123, verbose_level);
	label_variables_x123(Poly4_x123, verbose_level);

	Poly1_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly1_4->init" << endl;
	}
	Poly1_4->init(Surf->F,
			4 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly1_4->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly2_4->init" << endl;
	}
	Poly2_4->init(Surf->F,
			4 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly2_4->init" << endl;
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before Poly3_4->init" << endl;
	}
	Poly3_4->init(Surf->F,
			4 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after Poly3_4->init" << endl;
	}

	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"before label_variables_4" << endl;
	}
	label_variables_4(Poly1_4, 0 /* verbose_level */);
	label_variables_4(Poly2_4, 0 /* verbose_level */);
	label_variables_4(Poly3_4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"after label_variables_4" << endl;
	}

	nb_monomials = Poly3_4->get_nb_monomials();

	if (f_v) {
		cout << "Poly3_4->nb_monomials = " << nb_monomials << endl;
		cout << "Poly2_4->nb_monomials = " << Poly2_4->get_nb_monomials() << endl;
	}



	Partials = NEW_OBJECTS(ring_theory::partial_derivative, 4);

	int i;

	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"initializing partials" << endl;
	}
	for (i = 0; i < 4; i++) {
		Partials[i].init(Poly3_4, Poly2_4, i, verbose_level);
	}
	if (f_v) {
		cout << "surface_polynomial_domains::init_polynomial_domains "
				"initializing partials done" << endl;
	}


	//init_large_polynomial_domains(verbose_level);


	if (f_v) {
		cout << "surface_polynomial_domains::init done" << endl;
	}

}

void surface_polynomial_domains::init_large_polynomial_domains(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::init_large_polynomial_domains" << endl;
	}
	f_has_large_polynomial_domains = true;
	Poly2_27 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly4_27 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly6_27 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3_24 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	Poly2_27->init(Surf->F,
			27 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	Poly4_27->init(Surf->F,
			27 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);
	Poly6_27->init(Surf->F,
			27 /* nb_vars */, 6 /* degree */,
			t_PART,
			verbose_level);
	Poly3_24->init(Surf->F,
			24 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);

	nb_monomials2 = Poly2_27->get_nb_monomials();
	nb_monomials4 = Poly4_27->get_nb_monomials();
	nb_monomials6 = Poly6_27->get_nb_monomials();
	nb_monomials3 = Poly3_24->get_nb_monomials();

	label_variables_27(Poly2_27, 0 /* verbose_level */);
	label_variables_27(Poly4_27, 0 /* verbose_level */);
	label_variables_27(Poly6_27, 0 /* verbose_level */);
	label_variables_24(Poly3_24, 0 /* verbose_level */);

	if (f_v) {
		cout << "nb_monomials2 = " << nb_monomials2 << endl;
		cout << "nb_monomials4 = " << nb_monomials4 << endl;
		cout << "nb_monomials6 = " << nb_monomials6 << endl;
		cout << "nb_monomials3 = " << nb_monomials3 << endl;
	}

	if (f_v) {
		cout << "surface_polynomial_domains::init_large_polynomial_domains "
				"before clebsch_cubics" << endl;
	}
	clebsch_cubics(verbose_level - 1);
	if (f_v) {
		cout << "surface_polynomial_domains::init_large_polynomial_domains "
				"after clebsch_cubics" << endl;
	}

	if (f_v) {
		cout << "surface_polynomial_domains::init_large_polynomial_domains done" << endl;
	}
}

void surface_polynomial_domains::label_variables_3(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_3" << endl;
	}
	if (HPD->nb_variables != 3) {
		cout << "surface_polynomial_domains::label_variables_3 HPD->nb_variables != 3" << endl;
		exit(1);
	}

	string s1, s2;

	s1.assign("y%d");
	s2.assign("y_{%d}");
	HPD->remake_symbols(0 /* symbol_offset */,
			s1, s2, verbose_level);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_3 done" << endl;
	}
}

void surface_polynomial_domains::label_variables_x123(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_x123" << endl;
	}
	if (HPD->nb_variables != 3) {
		cout << "surface_polynomial_domains::label_variables_x123 "
				"HPD->nb_variables != 3" << endl;
		exit(1);
	}


	string s1, s2;

	s1.assign("x%d");
	s2.assign("x_{%d}");
	HPD->remake_symbols(1 /* symbol_offset */,
			s1, s2, verbose_level);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_x123 done" << endl;
	}
}

void surface_polynomial_domains::label_variables_4(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, l;

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_4" << endl;
		}
	if (HPD->nb_variables != 4) {
		cout << "surface_polynomial_domains::label_variables_4 HPD->nb_variables != 4" << endl;
		exit(1);
		}

	string s1, s2;

	s1.assign("X%d");
	s2.assign("X_{%d}");

	HPD->remake_symbols(0 /* symbol_offset */,
			s1, s2, verbose_level);


	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_4 done" << endl;
		}

}

void surface_polynomial_domains::label_variables_27(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_27" << endl;
	}
	if (HPD->nb_variables != 27) {
		cout << "surface_polynomial_domains::label_variables_27 HPD->n != 27" << endl;
		exit(1);
	}

	string s1, s2;

	s1.assign("y%d");
	s2.assign("y_{%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			0, 3,
			s1, s2,
			verbose_level);
	s1.assign("f_0%d");
	s2.assign("f_{0%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			3, 4,
			s1, s2,
			verbose_level);
	s1.assign("f_1%d");
	s2.assign("f_{1%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			7, 4,
			s1, s2,
			verbose_level);
	s1.assign("f_2%d");
	s2.assign("f_{2%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			11, 4,
			s1, s2,
			verbose_level);
	s1.assign("g_0%d");
	s2.assign("g_{0%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			15, 4,
			s1, s2,
			verbose_level);
	s1.assign("g_1%d");
	s2.assign("g_{1%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			19, 4,
			s1, s2,
			verbose_level);
	s1.assign("g_2%d");
	s2.assign("g_{2%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			23, 4,
			s1, s2,
			verbose_level);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_27 done" << endl;
	}
}

void surface_polynomial_domains::label_variables_24(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_24" << endl;
	}
	if (HPD->nb_variables != 24) {
		cout << "surface_polynomial_domains::label_variables_24 "
				"HPD->n != 24" << endl;
		exit(1);
	}

	string s1, s2;

	s1.assign("f_0%d");
	s2.assign("f_{0%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			0, 4,
			s1, s2,
			verbose_level);
	s1.assign("f_1%d");
	s2.assign("f_{1%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			4, 4,
			s1, s2,
			verbose_level);
	s1.assign("f_2%d");
	s2.assign("f_{2%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			8, 4,
			s1, s2,
			verbose_level);
	s1.assign("g_0%d");
	s2.assign("g_{0%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			12, 4,
			s1, s2,
			verbose_level);
	s1.assign("g_1%d");
	s2.assign("g_{1%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			16, 4,
			s1, s2,
			verbose_level);
	s1.assign("g_2%d");
	s2.assign("g_{2%d}");
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			20, 4,
			s1, s2,
			verbose_level);

	if (f_v) {
		cout << "surface_polynomial_domains::label_variables_24 done" << endl;
	}
}


int surface_polynomial_domains::index_of_monomial(
		int *v)
{
	return Poly3_4->index_of_monomial(v);
}

void surface_polynomial_domains::multiply_conic_times_linear(
		int *six_coeff,
	int *three_coeff, int *ten_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_conic_times_linear" << endl;
	}


	Int_vec_zero(ten_coeff, 10);
	for (i = 0; i < 6; i++) {
		a = six_coeff[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 3; j++) {
			b = three_coeff[j];
			if (b == 0) {
				continue;
			}
			c = Surf->F->mult(a, b);

			for (u = 0; u < 3; u++) {
				M[u] = Poly2->get_monomial(i, u) + Poly1->get_monomial(j, u);
			}
			idx = Poly3->index_of_monomial(M);
			if (idx >= 10) {
				cout << "surface_polynomial_domains::multiply_conic_times_linear "
						"idx >= 10" << endl;
				exit(1);
				}
			ten_coeff[idx] = Surf->F->add(ten_coeff[idx], c);
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::multiply_conic_times_linear done" << endl;
	}
}

void surface_polynomial_domains::multiply_linear_times_linear_times_linear(
	int *three_coeff1, int *three_coeff2, int *three_coeff3,
	int *ten_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_linear_times_linear_times_linear" << endl;
	}

	Int_vec_zero(ten_coeff, 10);
	for (i = 0; i < 3; i++) {
		a = three_coeff1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 3; j++) {
			b = three_coeff2[j];
			if (b == 0) {
				continue;
			}
			for (k = 0; k < 3; k++) {
				c = three_coeff3[k];
				if (c == 0) {
					continue;
				}
				d = Surf->F->mult3(a, b, c);
				for (u = 0; u < 3; u++) {
					M[u] = Poly1->get_monomial(i, u)
							+ Poly1->get_monomial(j, u)
							+ Poly1->get_monomial(k, u);
				}
				idx = Poly3->index_of_monomial(M);
				if (idx >= 10) {
					cout << "surface_polynomial_domains::multiply_linear_times_"
							"linear_times_linear idx >= 10" << endl;
					exit(1);
					}
				ten_coeff[idx] = Surf->F->add(ten_coeff[idx], d);
			}
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::multiply_linear_times_linear_times_linear done" << endl;
	}
}

void surface_polynomial_domains::multiply_linear_times_linear_times_linear_in_space(
	int *four_coeff1, int *four_coeff2, int *four_coeff3,
	int *twenty_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx, u;
	int M[4];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_linear_times_linear_times_linear_in_space" << endl;
	}

	Int_vec_zero(twenty_coeff, 20);
	for (i = 0; i < 4; i++) {
		a = four_coeff1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < 4; j++) {
			b = four_coeff2[j];
			if (b == 0) {
				continue;
			}
			for (k = 0; k < 4; k++) {
				c = four_coeff3[k];
				if (c == 0) {
					continue;
				}
				d = Surf->F->mult3(a, b, c);
				for (u = 0; u < 4; u++) {
					M[u] = Poly1_4->get_monomial(i, u)
							+ Poly1_4->get_monomial(j, u)
							+ Poly1_4->get_monomial(k, u);
				}
				idx = index_of_monomial(M);
				if (idx >= 20) {
					cout << "surface_polynomial_domains::multiply_linear_times_linear_"
							"times_linear_in_space idx >= 20" << endl;
					exit(1);
					}
				twenty_coeff[idx] = Surf->F->add(twenty_coeff[idx], d);
			}
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::multiply_linear_times_linear_times_linear_in_space done" << endl;
	}
}

void surface_polynomial_domains::multiply_Poly2_3_times_Poly2_3(
	int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_Poly2_3_times_Poly2_3" << endl;
	}

	Int_vec_zero(result, Poly4_x123->get_nb_monomials());
	for (i = 0; i < Poly2->get_nb_monomials(); i++) {
		a = input1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < Poly2->get_nb_monomials(); j++) {
			b = input2[j];
			if (b == 0) {
				continue;
			}
			c = Surf->F->mult(a, b);

			for (u = 0; u < 3; u++) {
				M[u] = Poly2->get_monomial(i, u) + Poly2->get_monomial(j, u);
			}
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = Surf->F->add(result[idx], c);
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::multiply_Poly2_3_times_Poly2_3 done" << endl;
	}
}

void surface_polynomial_domains::multiply_Poly1_3_times_Poly3_3(
		int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx, u;
	int M[3];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_Poly1_3_times_Poly3_3" << endl;
	}

	Int_vec_zero(result, Poly4_x123->get_nb_monomials());
	for (i = 0; i < Poly1->get_nb_monomials(); i++) {
		a = input1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < Poly3->get_nb_monomials(); j++) {
			b = input2[j];
			if (b == 0) {
				continue;
			}
			c = Surf->F->mult(a, b);
			for (u = 0; u < 3; u++) {
				M[u] = Poly1->get_monomial(i, u) + Poly3->get_monomial(j, u);
			}
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = Surf->F->add(result[idx], c);
		}
	}

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_Poly1_3_times_Poly3_3 done" << endl;
	}
}

void surface_polynomial_domains::clebsch_cubics(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::clebsch_cubics f_has_large_"
				"polynomial_domains is false" << endl;
		exit(1);
	}
	int Monomial[27];

	int i, j, idx;

	Clebsch_Pij = NEW_int(3 * 4 * nb_monomials2);
	Clebsch_P = NEW_pint(3 * 4);
	Clebsch_P3 = NEW_pint(3 * 3);

	Int_vec_zero(Clebsch_Pij, 3 * 4 * nb_monomials2);


	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			Clebsch_P[i * 4 + j] =
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
		}
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Clebsch_P3[i * 3 + j] =
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
		}
	}
	int coeffs[] = {
		1, 15, 2, 11,
		1, 16, 2, 12,
		1, 17, 2, 13,
		1, 18, 2, 14,
		0, 3, 2, 19,
		0, 4, 2, 20,
		0, 5, 2, 21,
		0, 6, 2, 22,
		0, 23, 1, 7,
		0, 24, 1, 8,
		0, 25, 1, 9,
		0, 26, 1, 10
	};
	int c0, c1;

	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics "
				"Setting up the matrix P:" << endl;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "i=" << i << " j=" << j << endl;
			Int_vec_zero(Monomial, 27);
			c0 = coeffs[(i * 4 + j) * 4 + 0];
			c1 = coeffs[(i * 4 + j) * 4 + 1];
			Int_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			c0 = coeffs[(i * 4 + j) * 4 + 2];
			c1 = coeffs[(i * 4 + j) * 4 + 3];
			Int_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics the matrix "
				"Clebsch_P is:" << endl;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(cout, Clebsch_P[i * 4 + j]);
			cout << endl;
		}
	}

	int *Cubics;
	int *Adjugate;
	int *Ad[3 * 3];
	int *C[4];
	int m1;


	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics "
				"allocating cubics" << endl;
	}

	Cubics = NEW_int(4 * nb_monomials6);
	Int_vec_zero(Cubics, 4 * nb_monomials6);

	Adjugate = NEW_int(3 * 3 * nb_monomials4);
	Int_vec_zero(Adjugate, 3 * 3 * nb_monomials4);

	for (i = 0; i < 4; i++) {
		C[i] = Cubics + i * nb_monomials6;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Ad[i * 3 + j] = Adjugate + (i * 3 + j) * nb_monomials4;
		}
	}

	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics "
				"computing C[3] = the determinant" << endl;
	}
	// compute C[3] as the negative of the determinant
	// of the matrix of the first three columns:
	//int_vec_zero(C[3], nb_monomials6);
	m1 = Surf->F->negate(1);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[2 * 4 + 2], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[2 * 4 + 0], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[2 * 4 + 1], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[0 * 4 + 2], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[0 * 4 + 0], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[0 * 4 + 1], 1, C[3],
			0 /* verbose_level*/);

	int I[3];
	int J[3];
	int size_complement, scalar;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics "
				"computing adjoint" << endl;
	}
	// compute adjoint:
	for (i = 0; i < 3; i++) {
		I[0] = i;
		Combi.set_complement(I, 1, I + 1, size_complement, 3);
		for (j = 0; j < 3; j++) {
			J[0] = j;
			Combi.set_complement(J, 1, J + 1, size_complement, 3);

			if ((i + j) % 2) {
				scalar = m1;
			}
			else {
				scalar = 1;
			}
			minor22(
					Clebsch_P3,
					I[1], I[2], J[1], J[2],
					scalar,
					Ad[j * 3 + i],
					0 /* verbose_level */);
		}
	}

	// multiply adjoint * last column:
	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics "
				"multiply adjoint times last column" << endl;
	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			multiply42_and_add(Ad[i * 3 + j],
					Clebsch_P[j * 4 + 3], C[i], 0 /* verbose_level */);
		}
	}

	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics "
				"We have computed the Clebsch cubics" << endl;
	}


	int Y[3];
	int M24[24];
	int h;
	data_structures::sorting Sorting;

	Clebsch_coeffs = NEW_int(4 * Poly3->get_nb_monomials() * nb_monomials3);
	Int_vec_zero(Clebsch_coeffs,
			4 * Poly3->get_nb_monomials() * nb_monomials3);
	CC = NEW_pint(4 * Poly3->get_nb_monomials());
	for (i = 0; i < 4; i++) {
		for (j = 0; j < Poly3->get_nb_monomials(); j++) {
			CC[i * Poly3->get_nb_monomials() + j] =
				Clebsch_coeffs + (i * Poly3->get_nb_monomials() + j) * nb_monomials3;
		}
	}
	for (i = 0; i < Poly3->get_nb_monomials(); i++) {
		Int_vec_copy(Poly3->get_monomial_pointer(i), Y, 3);
		for (j = 0; j < nb_monomials6; j++) {
			if (Sorting.int_vec_compare(Y, Poly6_27->get_monomial_pointer(j), 3) == 0) {
				Int_vec_copy(Poly6_27->get_monomial_pointer(j) + 3, M24, 24);
				idx = Poly3_24->index_of_monomial(M24);
				for (h = 0; h < 4; h++) {
					CC[h * Poly3->get_nb_monomials() + i][idx] =
							Surf->F->add(CC[h * Poly3->get_nb_monomials() + i][idx], C[h][j]);
				}
			}
		}
	}

	if (f_v) {
		print_clebsch_cubics(cout);
	}

	FREE_int(Cubics);
	FREE_int(Adjugate);

	if (f_v) {
		cout << "surface_polynomial_domains::clebsch_cubics done" << endl;
	}
}

void surface_polynomial_domains::multiply_222_27_and_add(
		int *M1, int *M2, int *M3,
	int scalar, int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply_222_27_and_add" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::multiply_222_27_and_add "
				"f_has_large_polynomial_domains is false" << endl;
		exit(1);
	}
	//int_vec_zero(MM, nb_monomials6);
	for (i = 0; i < nb_monomials2; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
			}
			for (k = 0; k < nb_monomials2; k++) {
				c = M3[k];
				if (c == 0) {
					continue;
				}
				d = Surf->F->mult3(a, b, c);
				orbiter_kernel_system::Orbiter->Int_vec->add3(
						Poly2_27->get_monomial_pointer(i),
					Poly2_27->get_monomial_pointer(j),
					Poly2_27->get_monomial_pointer(k),
					M, 27);
				idx = Poly6_27->index_of_monomial(M);
				if (idx >= nb_monomials6) {
					cout << "surface_polynomial_domains::multiply_222_27_and_add "
							"idx >= nb_monomials6" << endl;
					exit(1);
					}
				d = Surf->F->mult(scalar, d);
				MM[idx] = Surf->F->add(MM[idx], d);
			}
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::multiply_222_27_and_add done" << endl;
	}
}

void surface_polynomial_domains::minor22(
		int **P3, int i1, int i2, int j1, int j2,
	int scalar, int *Ad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_polynomial_domains::minor22" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::minor22 "
				"f_has_large_polynomial_domains is false" << endl;
		exit(1);
	}
	Int_vec_zero(Ad, nb_monomials4);
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i1 * 3 + j1][i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i2 * 3 + j2][j];
			if (b == 0) {
				continue;
			}
			d = Surf->F->mult(a, b);
			orbiter_kernel_system::Orbiter->Int_vec->add(Poly2_27->get_monomial_pointer(i),
				Poly2_27->get_monomial_pointer(j),
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
			}
			d = Surf->F->mult(scalar, d);
			Ad[idx] = Surf->F->add(Ad[idx], d);
		}
	}
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i2 * 3 + j1][i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i1 * 3 + j2][j];
			if (b == 0) {
				continue;
			}
			d = Surf->F->mult(a, b);
			orbiter_kernel_system::Orbiter->Int_vec->add(Poly2_27->get_monomial_pointer(i),
				Poly2_27->get_monomial_pointer(j),
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_polynomial_domains::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
			}
			d = Surf->F->mult(scalar, d);
			d = Surf->F->negate(d);
			Ad[idx] = Surf->F->add(Ad[idx], d);
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::minor22 done" << endl;
	}
}

void surface_polynomial_domains::multiply42_and_add(
		int *M1, int *M2,
		int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_polynomial_domains::multiply42_and_add" << endl;
	}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::multiply42_and_add "
				"f_has_large_polynomial_domains is false" << endl;
		exit(1);
	}
	for (i = 0; i < nb_monomials4; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
		}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
			}
			d = Surf->F->mult(a, b);
			orbiter_kernel_system::Orbiter->Int_vec->add(Poly4_27->get_monomial_pointer(i),
				Poly2_27->get_monomial_pointer(j),
				M, 27);
			idx = Poly6_27->index_of_monomial(M);
			if (idx >= nb_monomials6) {
				cout << "surface_polynomial_domains::multiply42_and_add "
						"idx >= nb_monomials6" << endl;
				exit(1);
			}
			MM[idx] = Surf->F->add(MM[idx], d);
		}
	}

	if (f_v) {
		cout << "surface_polynomial_domains::multiply42_and_add done" << endl;
	}
}

void surface_polynomial_domains::split_nice_equation(
		int *nice_equation,
	int *&f1, int *&f2, int *&f3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::split_nice_equation" << endl;
	}
	int M[4];
	int i, a, idx;

	f1 = NEW_int(Poly1->get_nb_monomials());
	f2 = NEW_int(Poly2->get_nb_monomials());
	f3 = NEW_int(Poly3->get_nb_monomials());
	Int_vec_zero(f1, Poly1->get_nb_monomials());
	Int_vec_zero(f2, Poly2->get_nb_monomials());
	Int_vec_zero(f3, Poly3->get_nb_monomials());

	for (i = 0; i < 20; i++) {
		a = nice_equation[i];
		if (a == 0) {
			continue;
		}
		Int_vec_copy(Poly3_4->get_monomial_pointer(i), M, 4);
		if (M[0] == 3) {
			cout << "surface_polynomial_domains::split_nice_equation "
					"the x_0^3-term is supposed to be zero" << endl;
			exit(1);
		}
		else if (M[0] == 2) {
			idx = Poly1->index_of_monomial(M + 1);
			f1[idx] = a;
		}
		else if (M[0] == 1) {
			idx = Poly2->index_of_monomial(M + 1);
			f2[idx] = a;
		}
		else if (M[0] == 0) {
			idx = Poly3->index_of_monomial(M + 1);
			f3[idx] = a;
		}
	}
	if (f_v) {
		cout << "surface_polynomial_domains::split_nice_equation done" << endl;
	}
}

void surface_polynomial_domains::assemble_polar_hypersurface(
	int *f1, int *f2, int *f3,
	int *&polar_hypersurface, int verbose_level)
// 2*x_0*f_1 + f_2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::assemble_polar_hypersurface" << endl;
	}
	int M[4];
	int i, a, idx, two;


	two = Surf->F->add(1, 1);
	polar_hypersurface = NEW_int(Poly2_4->get_nb_monomials());
	Int_vec_zero(polar_hypersurface, Poly2_4->get_nb_monomials());

	for (i = 0; i < Poly1->get_nb_monomials(); i++) {
		a = f1[i];
		if (a == 0) {
			continue;
		}
		Int_vec_copy(Poly1->get_monomial_pointer(i), M + 1, 3);
		M[0] = 1;
		idx = Poly2_4->index_of_monomial(M);
		polar_hypersurface[idx] = Surf->F->add(
				polar_hypersurface[idx], Surf->F->mult(two, a));
	}

	for (i = 0; i < Poly2->get_nb_monomials(); i++) {
		a = f2[i];
		if (a == 0) {
			continue;
		}
		Int_vec_copy(Poly2->get_monomial_pointer(i), M + 1, 3);
		M[0] = 0;
		idx = Poly2_4->index_of_monomial(M);
		polar_hypersurface[idx] = Surf->F->add(polar_hypersurface[idx], a);
	}

	if (f_v) {
		cout << "surface_polynomial_domains::assemble_polar_hypersurface done" << endl;
	}
}

void surface_polynomial_domains::compute_gradient(
		int *equation20, int *&gradient, int verbose_level)
// gradient[4]
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surface_polynomial_domains::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "surface_polynomial_domains::compute_gradient "
				"Poly2_4->get_nb_monomials() = " << Poly2_4->get_nb_monomials() << endl;
	}

	gradient = NEW_int(4 * Poly2_4->get_nb_monomials());

	for (i = 0; i < 4; i++) {
		if (f_v) {
			cout << "surface_polynomial_domains::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "surface_polynomial_domains::compute_gradient "
					"eqn_in=";
			Int_vec_print(cout, equation20, 20);
			cout << " = " << endl;
			Poly3_4->print_equation(cout, equation20);
			cout << endl;
		}
		Partials[i].apply(equation20,
				gradient + i * Poly2_4->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "surface_polynomial_domains::compute_gradient "
					"partial=";
			Int_vec_print(cout, gradient + i * Poly2_4->get_nb_monomials(),
					Poly2_4->get_nb_monomials());
			cout << " = ";
			Poly2_4->print_equation(cout,
					gradient + i * Poly2_4->get_nb_monomials());
			cout << endl;
		}
	}


	if (f_v) {
		cout << "surface_polynomial_domains::compute_gradient done" << endl;
	}
}

long int surface_polynomial_domains::compute_tangent_plane(
		int *pt_coords,
		int *gradient,
		int verbose_level)
// gradient[4 * Poly2_4->get_nb_monomials()]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_eqns = 4;
	int i;
	int w[4];

	if (f_v) {
		cout << "surface_polynomial_domains::compute_tangent_plane" << endl;
	}

	for (i = 0; i < nb_eqns; i++) {
		if (f_vv) {
			cout << "surface_polynomial_domains::compute_tangent_plane "
					"gradient i=" << i << " / " << nb_eqns << endl;
		}
		if (false) {
			cout << "surface_polynomial_domains::compute_tangent_plane "
					"gradient " << i << " = ";
			Int_vec_print(cout,
					gradient + i * Poly2_4->get_nb_monomials(),
					Poly2_4->get_nb_monomials());
			cout << endl;
		}
		w[i] = Poly2_4->evaluate_at_a_point(
				gradient + i * Poly2_4->get_nb_monomials(), pt_coords);
		if (f_vv) {
			cout << "surface_polynomial_domains::compute_tangent_plane "
					"value = " << w[i] << endl;
		}
	}

	if (Int_vec_is_zero(w, nb_eqns)) {
		cout << "surface_polynomial_domains::compute_tangent_plane "
				"the point is singular" << endl;
		exit(1);
	}
	long int plane_rk;

	plane_rk = Surf->P->Solid->plane_rank_using_dual_coordinates_in_three_space(
			w /* eqn4 */,
			0 /* verbose_level*/);

	return plane_rk;
}

long int surface_polynomial_domains::compute_special_bitangent(
		geometry::projective_space *P2,
		int *gradient,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_polynomial_domains::compute_special_bitangent" << endl;
	}

	int v[4];

	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	v[3] = 0;

	long int plane_rk;
	long int special_bitangent_rk;

	if (f_v) {
		cout << "surface_polynomial_domains::compute_special_bitangent "
				"before compute_tangent_plane" << endl;
	}

	plane_rk = compute_tangent_plane(
			v, gradient, verbose_level);


	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_from "
				"after compute_tangent_plane" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_from "
				"plane_rk = " << plane_rk << endl;
	}
	int Basis12[12];

	Surf->unrank_plane(Basis12, plane_rk);
	if (f_v) {
		cout << "surface_polynomial_domains::compute_special_bitangent "
				"Basis12=" << endl;
		Int_matrix_print(Basis12, 3, 4);
	}
	int Basis6[6];
	int j;

	// we assume that the first row of Basis12 is v = (1,0,0,0)

	if (Int_vec_compare(v, Basis12, 4)) {
		cout << "surface_polynomial_domains::compute_special_bitangent "
				"the first row of Basis12 is not 1,0,0,0" << endl;
		exit(1);
	}

	for (j = 0; j < 2; j++) {
		Int_vec_copy(Basis12 + (j + 1) * 4 + 1, Basis6 + j * 3, 3);
	}
	special_bitangent_rk =
			P2->Subspaces->Grass_lines->rank_lint_here(
					Basis6, 0);
	if (f_v) {
		cout << "surface_polynomial_domains::compute_special_bitangent "
				"special_bitangent_rk = " << special_bitangent_rk << endl;
	}
	return special_bitangent_rk;

}

void surface_polynomial_domains::print_clebsch_P(
		std::ostream &ost)
{
	int h, i, f_first;

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::print_clebsch_P "
				"f_has_large_polynomial_domains is false" << endl;
		//exit(1);
		return;
	}
	ost << "\\clearpage" << endl;
	ost << "\\subsection*{The Clebsch system $P$}" << endl;

	ost << "$$" << endl;
	print_clebsch_P_matrix_only(ost);
	ost << "\\cdot \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "x_0\\\\" << endl;
	ost << "x_1\\\\" << endl;
	ost << "x_2\\\\" << endl;
	ost << "x_3\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "= \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;


	ost << "\\begin{align*}" << endl;
	for (h = 0; h < 4; h++) {
		ost << "x_" << h << " &= C_" << h
				<< "(y_0,y_1,y_2)=\\\\" << endl;
		f_first = true;
		for (i = 0; i < Poly3->get_nb_monomials(); i++) {

			if (Poly3_24->is_zero(CC[h * Poly3->get_nb_monomials() + i])) {
				continue;
			}
			ost << "&";

			if (f_first) {
				f_first = false;
			}
			else {
				ost << "+";
			}
			ost << "\\Big(";
			Poly3_24->print_equation_with_line_breaks_tex(
					ost, CC[h * Poly3->get_nb_monomials() + i],
					6, "\\\\\n&");
			ost << "\\Big)" << endl;

			ost << "\\cdot" << endl;

			Poly3->print_monomial(ost, i);
			ost << "\\\\" << endl;
		}
	}
	ost << "\\end{align*}" << endl;
}

void surface_polynomial_domains::print_clebsch_P_matrix_only(
		std::ostream &ost)
{
	int i, j;

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::print_clebsch_P_matrix_only "
				"f_has_large_polynomial_domains is false" << endl;
		exit(1);
	}
	ost << "\\left[" << endl;
	ost << "\\begin{array}{cccc}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			//cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(ost, Clebsch_P[i * 4 + j]);
			if (j < 4 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void surface_polynomial_domains::print_clebsch_cubics(
		std::ostream &ost)
{
	int i, h;

	if (!f_has_large_polynomial_domains) {
		cout << "surface_polynomial_domains::print_clebsch_cubics "
				"f_has_large_polynomial_domains is false" << endl;
		exit(1);
	}
	ost << "The Clebsch coefficients are:" << endl;
	for (h = 0; h < 4; h++) {
		ost << "C[" << h << "]:" << endl;
		for (i = 0; i < Poly3->get_nb_monomials(); i++) {

			if (Poly3_24->is_zero(CC[h * Poly3->get_nb_monomials() + i])) {
				continue;
			}

			Poly3->print_monomial(ost, i);
			ost << " \\cdot \\Big(";
			Poly3_24->print_equation(ost, CC[h * Poly3->get_nb_monomials() + i]);
			ost << "\\Big)" << endl;
		}
	}
}

void surface_polynomial_domains::print_system(
		std::ostream &ost, int *system)
{
	int i, j;

	//ost << "The system:\\\\";
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	ost << "\\begin{array}{cccc}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			int *p = system + (i * 4 + j) * 3;
			Poly1->print_equation(ost, p);
			if (j < 4 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "\\cdot \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "x_0\\\\" << endl;
	ost << "x_1\\\\" << endl;
	ost << "x_2\\\\" << endl;
	ost << "x_3\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "= \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;
}

void surface_polynomial_domains::print_polynomial_domains_latex(
		std::ostream &ost)
{
	ost << "The polynomial domain Poly3\\_4 is\\\\" << endl;
	Poly3_4->print_monomial_ordering_latex(ost);

	ost << "The polynomial domain Poly1\\_x123 is\\\\" << endl;
	Poly1_x123->print_monomial_ordering_latex(ost);

	ost << "The polynomial domain Poly2\\_x123 is\\\\" << endl;
	Poly2_x123->print_monomial_ordering_latex(ost);

	ost << "The polynomial domain Poly3\\_x123 is\\\\" << endl;
	Poly3_x123->print_monomial_ordering_latex(ost);

	ost << "The polynomial domain Poly4\\_x123 is\\\\" << endl;
	Poly4_x123->print_monomial_ordering_latex(ost);

}

void surface_polynomial_domains::print_equation_in_trihedral_form(
		std::ostream &ost,
	int *the_six_plane_equations,
	int lambda, int *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = F_0F_1F_2 + \\lambda G_0G_1G_2\\\\" << endl;
	ost << "& = " << endl;
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 0 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 1 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 2 * 4);
	ost << "\\Big)";
	ost << "+ " << lambda;
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 3 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 4 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 5 * 4);
	ost << "\\Big)\\\\";
	ost << "& \\equiv " << endl;
	Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}


}}}

