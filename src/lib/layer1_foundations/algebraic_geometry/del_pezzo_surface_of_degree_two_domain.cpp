/*
 * del_pezzo_surface_of_degree_two_domain.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


del_pezzo_surface_of_degree_two_domain::del_pezzo_surface_of_degree_two_domain()
{
	F = NULL;
	P3 = NULL;
	P2 = NULL;
	Gr = NULL;
	Gr3 = NULL;
	nb_lines_PG_3 = 0;
	Poly4_3 = NULL;
}


del_pezzo_surface_of_degree_two_domain::~del_pezzo_surface_of_degree_two_domain()
{
	if (P2) {
		FREE_OBJECT(P2);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
	if (Gr3) {
		FREE_OBJECT(Gr3);
	}
}

void del_pezzo_surface_of_degree_two_domain::init(
		geometry::projective_space *P3,
		ring_theory::homogeneous_polynomial_domain *Poly4_3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_domain::init" << endl;
	}

	if (P3->n != 3) {
		cout << "del_pezzo_surface_of_degree_two_domain::init "
				"projective space must have dimension 3" << endl;
		exit(1);
	}
	del_pezzo_surface_of_degree_two_domain::P3 = P3;
	del_pezzo_surface_of_degree_two_domain::Poly4_3 = Poly4_3;
	F = P3->F;

	P2 = NEW_OBJECT(geometry::projective_space);
	P2->projective_space_init(2, F,
		FALSE /*f_init_incidence_structure*/,
		verbose_level);

	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(4 /*n*/, 2, F, 0 /* verbose_level */);

	nb_lines_PG_3 = Gr->nCkq->as_lint();

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_domain::init "
				"nb_lines_PG_3 = " << nb_lines_PG_3 << endl;
	}

	Gr3 = NEW_OBJECT(geometry::grassmann);
	Gr3->init(4, 3, F, 0 /* verbose_level*/);


	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_domain::init done" << endl;
	}
}

void del_pezzo_surface_of_degree_two_domain::enumerate_points(int *coeff,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_domain::enumerate_points" << endl;
	}

	//Poly4_3->enumerate_points(coeff, Pts, verbose_level);
	long int rk, rk_pt;
	int a;
	int v3[3];
	int v4[4];

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_domain::enumerate_points "
				"P2->N_points=" << P2->N_points << endl;
#if 0
		print_equation_with_line_breaks_tex(cout,
				coeff, 8 /* nb_terms_per_line*/,
				"\\\\\n");
		cout << endl;
#endif
	}

	for (rk = 0; rk < P2->N_points; rk++) {
		P2->unrank_point(v3, rk);
		a = Poly4_3->evaluate_at_a_point(coeff, v3);

		if (f_v) {
			cout << a << " : ";
			Int_vec_print(cout, v3, 3);
			cout << " : " << a << " : ";
		}

		Int_vec_copy(v3, v4, 3);

		if (a == 0) {
			v4[3] = 0;
			rk_pt = P3->rank_point(v4);

			Pts.push_back(rk_pt);
			if (f_v) {
				 cout << " one point" << endl;
			}
		}
		else {
			int nb_roots;
			int roots[2];
			int i;

			F->all_square_roots(a, nb_roots, roots);
			if (f_v) {
				 cout << nb_roots << " points" << endl;
			}
			for (i = 0; i < nb_roots; i++) {
				Int_vec_copy(v3, v4, 3);
				v4[3] = roots[i];
				rk_pt = P3->rank_point(v4);

				Pts.push_back(rk_pt);

			}
		}
	}

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_domain::enumerate_points done" << endl;
	}
}

void del_pezzo_surface_of_degree_two_domain::print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "w^2=";
	Poly4_3->print_equation_with_line_breaks_tex(
			ost, coeffs, 15 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
	//ost << "=0" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}

void del_pezzo_surface_of_degree_two_domain::unrank_point(int *v, long int rk)
{
	P3->unrank_point(v, rk);
}

long int del_pezzo_surface_of_degree_two_domain::rank_point(int *v)
{
	long int rk;

	rk = P3->rank_point(v);
	return rk;
}




}}}

