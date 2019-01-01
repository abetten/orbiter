// determine_cubic.C
//
// Anton Betten
// December 28, 2018
//
// computes the equation of a cubic through 9 given points
// in PG(2, q)




#include "orbiter.h"

int compute_system_in_RREF(
		projective_space *P,
		homogeneous_polynomial_domain *Poly_3_3,
		int nb_pts, int *Pts, int *coeff10,
		int verbose_level);

int main(int argc, char **argv)
{
	int verbose_level = 1;
	int i;
	int q = -1;
	int f_has_pts = FALSE;
	const char *pts_string = NULL;
	int f_poly = FALSE;
	const char *override_poly = NULL;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			override_poly = argv[++i];
			cout << "-poly " << override_poly << endl;
			}
		else if (strcmp(argv[i], "-pts") == 0) {
			f_has_pts = TRUE;
			pts_string = argv[++i];
			cout << "-pts " << pts_string << endl;
			}
		}

	int f_v = (verbose_level >= 1);
	finite_field *F;

	if (q == -1) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	F->init_override_polynomial(q, override_poly, verbose_level);

	int *input_pts;
	int nb_pts = 0;

	if (f_has_pts) {
		int_vec_scan(pts_string, input_pts, nb_pts);
	} else {
		cout << "please use -pts to specify the points" << endl;
		exit(1);
	}

	if (nb_pts < 9) {
		cout << "need at least 9 points" << endl;
		exit(1);
		}

	cout << "There are " << nb_pts << " input points: ";
	int_vec_print(cout, input_pts, nb_pts);
	cout << endl;


	projective_space * P;

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "determine_cubic before P->init" << endl;
		}
	P->init(2, F,
		FALSE,
		verbose_level - 2/*MINIMUM(2, verbose_level)*/);

	if (f_v) {
		cout << "determine_cubic after P->init" << endl;
		}

	homogeneous_polynomial_domain *Poly_3_3;

	Poly_3_3 = NEW_OBJECT(homogeneous_polynomial_domain);


	Poly_3_3->init(F,
			3 /* n */, 3 /* degree */, FALSE /* f_init_incidence_structure */,
			verbose_level);

	int coeff10[10];


	compute_system_in_RREF(
			P,
			Poly_3_3,
			nb_pts, input_pts, coeff10,
			verbose_level);

	cout << "The equation is:" << endl;
	Poly_3_3->print_equation(cout, coeff10);
	cout << endl;
}


int compute_system_in_RREF(
		projective_space *P,
		homogeneous_polynomial_domain *Poly_3_3,
		int nb_pts, int *Pts, int *coeff10,
		int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i, j, r, d;
	int *Pt_coord;
	int *System;
	int *base_cols;

	if (f_v) {
		cout << "compute_system_in_RREF" << endl;
		}
	d = P->n + 1;
	Pt_coord = NEW_int(nb_pts * d);
	System = NEW_int(nb_pts * Poly_3_3->nb_monomials);
	base_cols = NEW_int(Poly_3_3->nb_monomials);

	if (f_v) {
		cout << "compute_system_in_RREF list of "
				"points:" << endl;
		int_vec_print(cout, Pts, nb_pts);
		cout << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(Pt_coord + i * d, Pts[i]);
		}
	if (f_v) {
		cout << "compute_system_in_RREF matrix of "
				"point coordinates:" << endl;
		int_matrix_print(Pt_coord, nb_pts, d);
		}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < Poly_3_3->nb_monomials; j++) {
			System[i * Poly_3_3->nb_monomials + j] =
				P->F->evaluate_monomial(
					Poly_3_3->Monomials + j * d,
					Pt_coord + i * d, d);
			}
		}
	if (f_v) {
		cout << "compute_system_in_RREF "
				"The system:" << endl;
		int_matrix_print(System, nb_pts, Poly_3_3->nb_monomials);
		}
	r = P->F->Gauss_simple(System, nb_pts, Poly_3_3->nb_monomials,
		base_cols, 0 /* verbose_level */);
	if (f_v) {
		cout << "compute_system_in_RREF "
				"The system in RREF:" << endl;
		int_matrix_print(System, r, Poly_3_3->nb_monomials);
		}
	if (f_v) {
		cout << "compute_system_in_RREF "
				"The system has rank " << r << endl;
		}

	if (r != 9) {
		cout << "r != 9" << endl;
		exit(1);
	}
	int kernel_m, kernel_n;

	P->F->matrix_get_kernel(System, r, Poly_3_3->nb_monomials,
		base_cols, r,
		kernel_m, kernel_n, coeff10);


	FREE_int(Pt_coord);
	FREE_int(System);
	FREE_int(base_cols);
	return r;
}
