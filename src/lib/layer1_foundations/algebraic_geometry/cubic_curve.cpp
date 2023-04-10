/*
 * cubic_curve.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


cubic_curve::cubic_curve()
{
	q = 0;
	F = NULL;
	P = NULL;


	nb_monomials = 0;


	Poly = NULL;
	Poly2 = NULL;

	Partials = NULL;

	gradient = NULL;

}

cubic_curve::~cubic_curve()
{
	int f_v = false;

	if (f_v) {
		cout << "cubic_curve::~cubic_curve" << endl;
	}
	if (P) {
		FREE_OBJECT(P);
	}
	if (Poly) {
		FREE_OBJECT(Poly);
	}
	if (Poly2) {
		FREE_OBJECT(Poly2);
	}
	if (Partials) {
		FREE_OBJECTS(Partials);
	}
	if (gradient) {
		FREE_int(gradient);
	}
}

void cubic_curve::init(
		field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "cubic_curve::init" << endl;
	}

	cubic_curve::F = F;
	q = F->q;
	if (f_v) {
		cout << "cubic_curve::init q = " << q << endl;
	}

	P = NEW_OBJECT(geometry::projective_space);
	if (f_v) {
		cout << "cubic_curve::init before P->projective_space_init" << endl;
	}
	P->projective_space_init(2, F,
		true /*f_init_incidence_structure */,
		verbose_level - 2);
	if (f_v) {
		cout << "cubic_curve::init after P->projective_space_init" << endl;
	}

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	Poly->init(F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);

	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	Poly2->init(F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);

	nb_monomials = Poly->get_nb_monomials();
	if (f_v) {
		cout << "cubic_curve::init nb_monomials = " << nb_monomials << endl;
	}

	Partials = NEW_OBJECTS(ring_theory::partial_derivative, 3);
	for (i = 0; i < 3; i++) {
		Partials[i].init(Poly, Poly2, i, verbose_level);
	}

	gradient = NEW_int(3 * Poly2->get_nb_monomials());

	if (f_v) {
		cout << "cubic_curve::init done" << endl;
	}
}


int cubic_curve::compute_system_in_RREF(
		int nb_pts, long int *pt_list, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i, j, r;
	int *Pts;
	int *System;
	int *base_cols;

	if (f_v) {
		cout << "cubic_curve::compute_system_in_RREF" << endl;
	}
	Pts = NEW_int(nb_pts * 3);
	System = NEW_int(nb_pts * nb_monomials);
	base_cols = NEW_int(nb_monomials);

	if (false) {
		cout << "cubic_curve::compute_system_in_RREF list of "
				"covered points by lines:" << endl;
		Lint_matrix_print(pt_list, nb_pts, P->Subspaces->k);
	}
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(Pts + i * 3, pt_list[i]);
	}
	if (f_v && false) {
		cout << "cubic_curve::compute_system_in_RREF list of "
				"covered points in coordinates:" << endl;
		Int_matrix_print(Pts, nb_pts, 3);
	}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
					Poly->evaluate_monomial(j, Pts + i * 3);
		}
	}
	if (f_v && false) {
		cout << "cubic_curve::compute_system_in_RREF "
				"The system:" << endl;
		Int_matrix_print(System, nb_pts, nb_monomials);
	}
	r = F->Linear_algebra->Gauss_simple(System, nb_pts, nb_monomials,
		base_cols, 0 /* verbose_level */);
	if (false) {
		cout << "cubic_curve::compute_system_in_RREF "
				"The system in RREF:" << endl;
		Int_matrix_print(System, nb_pts, nb_monomials);
	}
	if (f_v) {
		cout << "cubic_curve::compute_system_in_RREF "
				"The system has rank " << r << endl;
	}
	FREE_int(Pts);
	FREE_int(System);
	FREE_int(base_cols);
	return r;
}

void cubic_curve::compute_gradient(int *eqn_in, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "cubic_curve::compute_gradient" << endl;
	}
	for (i = 0; i < 3; i++) {
		if (f_v) {
			cout << "cubic_curve::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "cubic_curve::compute_gradient eqn_in=";
			Int_vec_print(cout, eqn_in, Poly->get_nb_monomials());
			cout << " = ";
			Poly->print_equation(cout, eqn_in);
			cout << endl;
		}
		Partials[i].apply(eqn_in,
				gradient + i * Poly2->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "cubic_curve::compute_gradient partial=";
			Int_vec_print(cout, gradient + i * Poly2->get_nb_monomials(),
					Poly2->get_nb_monomials());
			cout << " = ";
			Poly2->print_equation(cout, gradient + i * Poly2->get_nb_monomials());
			cout << endl;
		}
	}
	if (f_v) {
		cout << "cubic_curve::compute_gradient done" << endl;
	}
}

void cubic_curve::compute_singular_points(
		int *eqn_in,
		long int *Pts_on_curve, int nb_pts_on_curve,
		long int *Pts, int &nb_pts,
		int verbose_level)
// a singular point is a point where all partials vanish
// We compute the set of singular points into Pts[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int h, i, a, rk;
	int nb_eqns = 3;
	int v[3];

	if (f_v) {
		cout << "cubic_curve::compute_singular_points" << endl;
	}
	compute_gradient(eqn_in, verbose_level);

	nb_pts = 0;

	for (h = 0; h < nb_pts_on_curve; h++) {
		if (f_vv) {
			cout << "cubic_curve::compute_singular_points "
					"h=" << h << " / " << nb_pts_on_curve << endl;
		}
		rk = Pts_on_curve[h];
		if (f_vv) {
			cout << "cubic_curve::compute_singular_points "
					"rk=" << rk << endl;
		}
		Poly->unrank_point(v, rk);
		if (f_vv) {
			cout << "cubic_curve::compute_singular_points "
					"v=";
			Int_vec_print(cout, v, 3);
			cout << endl;
		}
		for (i = 0; i < nb_eqns; i++) {
			if (f_vv) {
				cout << "cubic_curve::compute_singular_points "
						"gradient i=" << i << " / " << nb_eqns << endl;
			}
			if (f_vv) {
				cout << "cubic_curve::compute_singular_points "
						"gradient " << i << " = ";
				Int_vec_print(cout,
						gradient + i * Poly2->get_nb_monomials(),
						Poly2->get_nb_monomials());
				cout << endl;
			}
			a = Poly2->evaluate_at_a_point(
					gradient + i * Poly2->get_nb_monomials(), v);
			if (f_vv) {
				cout << "cubic_curve::compute_singular_points "
						"value = " << a << endl;
			}
			if (a) {
				break;
			}
		}
		if (i == nb_eqns) {
			Pts[nb_pts++] = rk;
		}
	}
	if (f_v) {
		cout << "cubic_curve::compute_singular_points done" << endl;
	}
}

void cubic_curve::compute_inflexion_points(
		int *eqn_in,
		long int *Pts_on_curve, int nb_pts_on_curve,
		long int *Pts, int &nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h, a;
	int T[3];
	int v[3];
	int w[3];
	int Basis[9];
	int Basis2[6];
	int eqn_restricted_to_line[4];

	if (f_v) {
		cout << "cubic_curve::compute_inflexion_points" << endl;
	}
	compute_gradient(eqn_in, verbose_level - 2);


	nb_pts = 0;

	for (h = 0; h < nb_pts_on_curve; h++) {
		a = Pts_on_curve[h];
		Poly->unrank_point(v, a);

		if (f_v) {
			cout << "cubic_curve::compute_inflexion_points testing point "
					<< h << " / " << nb_pts_on_curve << " = " << a << " = ";
			Int_vec_print(cout, v, 3);
			cout << endl;
		}
		for (i = 0; i < 3; i++) {
			T[i] = Poly2->evaluate_at_a_point(
					gradient + i * Poly2->get_nb_monomials(), v);
		}
		for (i = 0; i < 3; i++) {
			if (T[i]) {
				break;
			}
		}
		if (i < 3) {
			// now we know that the tangent line at point a exists:

			//int_vec_copy(v, Basis, 3);
			Int_vec_copy(T, Basis, 3);
			if (f_v) {
				cout << "cubic_curve::compute_inflexion_points "
						"before F->perp_standard:" << endl;
				Int_matrix_print(Basis, 1, 3);
			}
			F->Linear_algebra->perp_standard(3, 1, Basis, 0 /*verbose_level*/);
			if (f_v) {
				cout << "cubic_curve::compute_inflexion_points "
						"after F->perp_standard:" << endl;
				Int_matrix_print(Basis, 3, 3);
			}
			// test if the first basis vector is a multiple of v:
			Int_vec_copy(v, Basis2, 3);
			Int_vec_copy(Basis + 3, Basis2 + 3, 3);
			if (F->Linear_algebra->rank_of_rectangular_matrix(Basis2,
					2, 3, 0 /*verbose_level*/) == 1) {
				Int_vec_copy(Basis + 6, w, 3);
			}
			else {
				Int_vec_copy(Basis + 3, w, 3);
			}
			Int_vec_copy(v, Basis2, 3);
			Int_vec_copy(w, Basis2 + 3, 3);
			if (F->Linear_algebra->rank_of_rectangular_matrix(Basis,
					2, 3, 0 /*verbose_level*/) != 2) {
				cout << "cubic_curve::compute_inflexion_points rank of "
						"line spanned by v and w is not two" << endl;
				exit(1);
			}
			Poly->substitute_line(
					eqn_in, eqn_restricted_to_line,
					v /*int *Pt1_coeff*/, w /*int *Pt2_coeff*/,
					0 /*verbose_level*/);
				// coeff_in[nb_monomials], coeff_out[degree + 1]
			if (f_v) {
				cout << "cubic_curve::compute_inflexion_points "
						"after Poly->substitute_line:" << endl;
				Int_vec_print(cout, eqn_restricted_to_line, 4);
				cout << endl;
			}
			if (eqn_restricted_to_line[0] == 0 &&
					eqn_restricted_to_line[1] == 0 &&
					eqn_restricted_to_line[2] == 0) {
				if (f_v) {
					cout << "cubic_curve::compute_inflexion_points "
							"found an inflexion point " << a << endl;
				}
				Pts[nb_pts++] = a;
			}
		}
		else {
			if (f_v) {
				cout << "cubic_curve::compute_inflexion_points "
						"the tangent line does not exist" << endl;
			}
		}
	} // next h

	if (f_v) {
		cout << "cubic_curve::compute_inflexion_points "
				"we found " << nb_pts << " inflexion points" << endl;
	}

	if (f_v) {
		cout << "cubic_curve::compute_inflexion_points done" << endl;
		}
}



}}}

