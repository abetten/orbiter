/*
 * orthogonal_group.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {


orthogonal_group::orthogonal_group()
{
	O = NULL;

	Quadratic_form_stack = NULL;

	find_root_x = NULL;
	find_root_y = NULL;
	find_root_z = NULL;

	Sv1 = NULL;
	Sv2 = NULL;
	Sv3 = NULL;
	Sv4 = NULL;
	Gram2 = NULL;
	ST_N1 = NULL;
	ST_N2 = NULL;
	ST_w = NULL;
	STr_B = STr_Bv = STr_w = STr_z = STr_x = NULL;
}

orthogonal_group::~orthogonal_group()
{
	if (Quadratic_form_stack) {
		FREE_OBJECTS(Quadratic_form_stack);
	}
	if (find_root_x) {
		FREE_int(find_root_x);
	}
	if (find_root_y) {
		FREE_int(find_root_y);
	}
	if (find_root_z) {
		FREE_int(find_root_z);
	}
	if (Sv1) {
		FREE_int(Sv1);
	}
	if (Sv2) {
		FREE_int(Sv2);
	}
	if (Sv3) {
		FREE_int(Sv3);
	}
	if (Sv4) {
		FREE_int(Sv4);
	}
	if (Gram2) {
		FREE_int(Gram2);
	}
	if (ST_N1) {
		FREE_int(ST_N1);
	}
	if (ST_N2) {
		FREE_int(ST_N2);
	}
	if (ST_w) {
		FREE_int(ST_w);
	}
	if (STr_B) {
		FREE_int(STr_B);
	}
	if (STr_Bv) {
		FREE_int(STr_Bv);
	}
	if (STr_w) {
		FREE_int(STr_w);
	}
	if (STr_z) {
		FREE_int(STr_z);
	}
	if (STr_x) {
		FREE_int(STr_x);
	}
}

void orthogonal_group::init(
		orthogonal *O, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_group::init" << endl;
	}

	orthogonal_group::O = O;

	int i;

	if (f_v) {
		cout << "orthogonal_group::init creating Quadratic_form_stack" << endl;
	}
	Quadratic_form_stack = NEW_OBJECTS(quadratic_form, O->Quadratic_form->m + 1);
	for (i = 1; i <= O->Quadratic_form->m; i++) {
		Quadratic_form_stack[i].init(1, 2 * i, O->F, verbose_level);
	}
	if (f_v) {
		cout << "orthogonal_group::init creating Quadratic_form_stack finished" << endl;
	}

	find_root_x = NEW_int(O->Quadratic_form->n);
	find_root_y = NEW_int(O->Quadratic_form->n);
	find_root_z = NEW_int(O->Quadratic_form->n);
	// for Siegel transformations:
	Sv1 = NEW_int(O->Quadratic_form->n);
	Sv2 = NEW_int(O->Quadratic_form->n);
	Sv3 = NEW_int(O->Quadratic_form->n);
	Sv4 = NEW_int(O->Quadratic_form->n);
	Gram2 = NEW_int(O->Quadratic_form->n * O->Quadratic_form->n);
	ST_N1 = NEW_int(O->Quadratic_form->n * O->Quadratic_form->n);
	ST_N2 = NEW_int(O->Quadratic_form->n * O->Quadratic_form->n);
	ST_w = NEW_int(O->Quadratic_form->n);
	STr_B = NEW_int(O->Quadratic_form->n * O->Quadratic_form->n);
	STr_Bv = NEW_int(O->Quadratic_form->n * O->Quadratic_form->n);
	STr_w = NEW_int(O->Quadratic_form->n);
	STr_z = NEW_int(O->Quadratic_form->n);
	STr_x = NEW_int(O->Quadratic_form->n);

	if (f_v) {
		cout << "orthogonal_group::init done" << endl;
	}
}

long int orthogonal_group::find_root(
		long int rk2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int ret;

	if (f_v) {
		cout << "orthogonal_group::find_root" << endl;
	}
	if (O->Quadratic_form->epsilon == 1) {
		ret = O->Hyperbolic_pair->find_root_hyperbolic(
				rk2, O->Quadratic_form->m, verbose_level);
		}
	else if (O->Quadratic_form->epsilon == 0) {
		ret = O->Hyperbolic_pair->find_root_parabolic(
				rk2, verbose_level);
		}
	else {
		cout << "orthogonal_group::find_root "
				"epsilon = " << O->Quadratic_form->epsilon << endl;
		exit(1);
		}
	if (f_v) {
		cout << "orthogonal_group::find_root done" << endl;
	}
	return ret;
}


void orthogonal_group::Siegel_map_between_singular_points(
		int *T,
		long int rk_from, long int rk_to,
		long int root, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_group::Siegel_map_between_singular_points" << endl;
	}
	O->Quadratic_form->Siegel_map_between_singular_points(T,
		rk_from, rk_to, root,
		//O->Quadratic_form->epsilon, O->Quadratic_form->n,
		//O->Quadratic_form->form_c1, O->Quadratic_form->form_c2, O->Quadratic_form->form_c3,
		//O->Quadratic_form->Gram_matrix,
		verbose_level);
	if (f_v) {
		cout << "orthogonal_group::Siegel_map_between_singular_points done" << endl;
	}
}

void orthogonal_group::Siegel_map_between_singular_points_hyperbolic(
		int *T,
	long int rk_from, long int rk_to,
	long int root, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *Gram;

	if (f_v) {
		cout << "orthogonal_group::Siegel_map_between_singular_points_hyperbolic" << endl;
	}
#if 0
	O->F->Linear_algebra->Gram_matrix(
			1, 2 * m - 1, 0,0,0, Gram, verbose_level - 1);
	O->Quadratic_form->Siegel_map_between_singular_points(T,
		rk_from, rk_to, root,
		//O->Quadratic_form->epsilon, 2 * m,
		//0, 0, 0, Gram,
		verbose_level);
	FREE_int(Gram);
#else
	if (f_v) {
		cout << "orthogonal_group::Siegel_map_between_singular_points_hyperbolic "
				"before Quadratic_form_stack[m].Siegel_map_between_singular_points" << endl;
	}
	Quadratic_form_stack[m].Siegel_map_between_singular_points(T,
			rk_from, rk_to, root, verbose_level);
	if (f_v) {
		cout << "orthogonal_group::Siegel_map_between_singular_points_hyperbolic "
				"after Quadratic_form_stack[m].Siegel_map_between_singular_points" << endl;
	}
#endif
	if (f_v) {
		cout << "orthogonal_group::Siegel_map_between_singular_points_hyperbolic done" << endl;
	}
}

void orthogonal_group::Siegel_Transformation(
		int *T,
	long int rk_from, long int rk_to, long int root,
	int verbose_level)
// root is not perp to from and to.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation" << endl;
	}
	if (f_vv) {
		cout << "orthogonal_group::Siegel_Transformationrk_from=" << rk_from
				<< " rk_to=" << rk_to << " root=" << root << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation "
				"before Siegel_Transformation2" << endl;
	}
	Siegel_Transformation2(T,
		rk_from, rk_to, root,
		STr_B, STr_Bv, STr_w, STr_z, STr_x,
		verbose_level);
	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation "
				"after Siegel_Transformation2" << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation done" << endl;
	}
}

void orthogonal_group::Siegel_Transformation2(
		int *T,
	long int rk_from, long int rk_to, long int root,
	int *B, int *Bv, int *w, int *z, int *x,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *From, *To, *Root;

	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation2" << endl;
	}
	From = NEW_int(O->Quadratic_form->n);
	To = NEW_int(O->Quadratic_form->n);
	Root = NEW_int(O->Quadratic_form->n);

	O->Hyperbolic_pair->unrank_point(Root, 1, root, verbose_level - 1);
	O->Hyperbolic_pair->unrank_point(From, 1, rk_from, verbose_level - 1);
	O->Hyperbolic_pair->unrank_point(To, 1, rk_to, verbose_level - 1);

	if (f_vv) {
		cout << "root: ";
		Int_vec_print(cout, Root, O->Quadratic_form->n);
		cout << endl;
		cout << "rk_from: ";
		Int_vec_print(cout, From, O->Quadratic_form->n);
		cout << endl;
		cout << "rk_to: ";
		Int_vec_print(cout, To, O->Quadratic_form->n);
		cout << endl;
	}

	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation2 "
				"before Siegel_Transformation3" << endl;
	}
	Siegel_Transformation3(T,
		From, To, Root,
		B, Bv, w, z, x,
		verbose_level - 1);
	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation2 "
				"after Siegel_Transformation3" << endl;
	}

	FREE_int(From);
	FREE_int(To);
	FREE_int(Root);

	if (f_vv) {
		cout << "the Siegel transformation is:" << endl;
		Int_vec_print_integer_matrix(cout, T, O->Quadratic_form->n, O->Quadratic_form->n);
	}

	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation2 done" << endl;
	}
}

void orthogonal_group::Siegel_Transformation3(
		int *T,
	int *from, int *to, int *root,
	int *B, int *Bv, int *w, int *z, int *x,
	int verbose_level)
{
	int i, j, a, b, av, bv, minus_one;
	//int k;
	//int *Gram;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation3" << endl;
	}
	//k = n - 1;
	//Gram = O->Quadratic_form->Gram_matrix;
#if 0
	if (f_vv) {
		cout << "n=" << O->Quadratic_form->n << endl;
		cout << "Gram matrix:" << endl;
		Combi.print_int_matrix(cout, Gram, O->Quadratic_form->n, O->Quadratic_form->n);
	}
#endif

	//Q_epsilon_unrank(*F, B, 1, epsilon, k,
	//form_c1, form_c2, form_c3, root);
	//Q_epsilon_unrank(*F, B + d, 1, epsilon, k,
	//form_c1, form_c2, form_c3, rk_from);
	//Q_epsilon_unrank(*F, w, 1, epsilon, k,
	//form_c1, form_c2, form_c3, rk_to);

	for (i = 0; i < O->Quadratic_form->n; i++) {
		B[i] = root[i];
		B[O->Quadratic_form->n + i] = from[i];
		w[i] = to[i];
	}
	if (f_vv) {
		cout << "root: ";
		Int_vec_print(cout, B, O->Quadratic_form->n);
		cout << endl;
		cout << "from: ";
		Int_vec_print(cout, B + O->Quadratic_form->n, O->Quadratic_form->n);
		cout << endl;
		cout << "to: ";
		Int_vec_print(cout, w, O->Quadratic_form->n);
		cout << endl;
	}

	a = O->Quadratic_form->evaluate_bilinear_form(B, B + O->Quadratic_form->n, 1);
	b = O->Quadratic_form->evaluate_bilinear_form(B, w, 1);
	av = O->F->inverse(a);
	bv = O->F->inverse(b);

	for (i = 0; i < O->Quadratic_form->n; i++) {
		B[O->Quadratic_form->n + i] = O->F->mult(B[O->Quadratic_form->n + i], av);
		w[i] = O->F->mult(w[i], bv);
	}

	for (i = 2; i < O->Quadratic_form->n; i++) {
		for (j = 0; j < O->Quadratic_form->n; j++) {
			B[i * O->Quadratic_form->n + j] = 0;
		}
	}

	if (f_vv) {
		cout << "before perp, the matrix B is:" << endl;
		Int_vec_print_integer_matrix(cout,
				B, O->Quadratic_form->n, O->Quadratic_form->n);
	}
	O->F->Linear_algebra->perp(
			O->Quadratic_form->n, 2, B, O->Quadratic_form->Gram_matrix,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix B is:" << endl;
		Int_vec_print_integer_matrix(cout,
				B, O->Quadratic_form->n, O->Quadratic_form->n);
	}
	O->F->Linear_algebra->invert_matrix(
			B, Bv, O->Quadratic_form->n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix Bv is:" << endl;
		Int_vec_print_integer_matrix(cout,
				B, O->Quadratic_form->n, O->Quadratic_form->n);
	}
	O->F->Linear_algebra->mult_matrix_matrix(
			w, Bv, z, 1, O->Quadratic_form->n, O->Quadratic_form->n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the coefficient vector z is:" << endl;
		Int_vec_print_integer_matrix(cout,
				z, 1, O->Quadratic_form->n);
	}
	z[0] = 0;
	z[1] = 0;
	if (f_vv) {
		cout << "the coefficient vector z is:" << endl;
		Int_vec_print_integer_matrix(cout,
				z, 1, O->Quadratic_form->n);
	}
	O->F->Linear_algebra->mult_matrix_matrix(
			z, B, x, 1, O->Quadratic_form->n, O->Quadratic_form->n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x is:" << endl;
		Int_vec_print_integer_matrix(cout,
				x, 1, O->Quadratic_form->n);
	}
	minus_one = O->F->negate(1);
	for (i = 0; i < O->Quadratic_form->n; i++) {
		x[i] = O->F->mult(x[i], minus_one);
	}
	if (f_vv) {
		cout << "the vector -x is:" << endl;
		Int_vec_print_integer_matrix(cout,
				x, 1, O->Quadratic_form->n);
	}
	make_Siegel_Transformation(
			T, x, B,
			O->Quadratic_form->n, O->Quadratic_form->Gram_matrix,
			false);
	if (f_vv) {
		cout << "the Siegel transformation is:" << endl;
		Int_vec_print_integer_matrix(cout,
				T, O->Quadratic_form->n, O->Quadratic_form->n);
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_Transformation3 done" << endl;
	}
}

void orthogonal_group::random_generator_for_orthogonal_group(
	int f_action_is_semilinear,
	int f_siegel,
	int f_reflection,
	int f_similarity,
	int f_semisimilarity,
	int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "orthogonal_group::random_generator_for_orthogonal_group" << endl;
		cout << "f_action_is_semilinear=" << f_action_is_semilinear << endl;
		cout << "f_siegel=" << f_siegel << endl;
		cout << "f_reflection=" << f_reflection << endl;
		cout << "f_similarity=" << f_similarity << endl;
		cout << "f_semisimilarity=" << f_semisimilarity << endl;
	}


	while (true) {
		r = Os.random_integer(4);
		if (r == 0 && f_siegel) {
			break;
		}
		else if (r == 1 && f_reflection) {
			break;
		}
		else if (r == 2 && f_similarity) {
			break;
		}
		else if (r == 3 && f_semisimilarity) {
			if (!f_action_is_semilinear) {
				continue;
			}
			break;
		}
	}

	if (r == 0) {
		if (f_vv) {
			cout << "orthogonal_group::random_generator_for_orthogonal_group "
					"choosing Siegel_transformation" << endl;
		}
		create_random_Siegel_transformation(
				Mtx, verbose_level /*- 2 */);
		if (f_action_is_semilinear) {
			Mtx[O->Quadratic_form->n * O->Quadratic_form->n] = 0;
		}
	}
	else if (r == 1) {
		if (f_vv) {
			cout << "orthogonal_group::random_generator_for_orthogonal_group "
					"choosing orthogonal reflection" << endl;
		}

		create_random_orthogonal_reflection(
				Mtx, verbose_level - 2);
		if (f_action_is_semilinear) {
			Mtx[O->Quadratic_form->n * O->Quadratic_form->n] = 0;
		}
	}
	else if (r == 2) {
		if (f_vv) {
			cout << "orthogonal_group::random_generator_for_orthogonal_group "
					"choosing similarity" << endl;
		}
		create_random_similarity(
				Mtx, verbose_level - 2);
		if (f_action_is_semilinear) {
			Mtx[O->Quadratic_form->n * O->Quadratic_form->n] = 0;
		}
	}
	else if (r == 3) {
		if (f_vv) {
			cout << "orthogonal_group::random_generator_for_orthogonal_group "
					"choosing random similarity" << endl;
		}
		create_random_semisimilarity(
				Mtx, verbose_level - 2);
	}
	if (f_v) {
		cout << "orthogonal_group::random_generator_for_orthogonal_group "
				"done" << endl;
	}
}


void orthogonal_group::create_random_Siegel_transformation(
		int *Mtx, int verbose_level)
// Makes an n x n matrix only. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk_u, alpha, i;
	int nb_pts; //, nb_pts_affine;
	//int k = O->Quadratic_form->m; // the Witt index, previously orthogonal_k;
	int d = O->Quadratic_form->n;
	int *u, *v;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "orthogonal_group::create_random_Siegel_transformation" << endl;
	}

	u = NEW_int(d);
	v = NEW_int(d);

	nb_pts = O->Hyperbolic_pair->nb_points; //nb_pts_Qepsilon(epsilon, d - 1, q);
	//nb_pts_affine = i_power_j(q, d);

	if (f_v) {
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"q=" << O->Quadratic_form->q << endl;
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"d=" << d << endl;
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"Witt index m=" << O->Quadratic_form->m << endl;
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"nb_pts=" << nb_pts << endl;
		//cout << "orthogonal::create_random_Siegel_transformation "
		//		"nb_pts_affine=" << nb_pts_affine << endl;
	}

	rk_u = Os.random_integer(nb_pts);
	if (f_v) {
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"rk_u=" << rk_u << endl;
	}
	O->Hyperbolic_pair->unrank_point(
			u, 1, rk_u, 0 /* verbose_level*/);
	//Q_epsilon_unrank(*F, u, 1 /*stride*/, epsilon, d - 1,
	// form_c1, form_c2, form_c3, rk_u);

	while (true) {

#if 0
		rk_v = random_integer(nb_pts_affine);
		if (f_v) {
			cout << "orthogonal_group::create_random_Siegel_transformation "
					"trying rk_v=" << rk_v << endl;
			}
		AG_element_unrank(q, v, 1 /* stride */, d, rk_v);
#else
		for (i = 0; i < d; i++) {
			v[i] = Os.random_integer(O->Quadratic_form->q);
		}

#endif

		alpha = O->Quadratic_form->evaluate_bilinear_form(u, v, 1);

		if (alpha == 0) {
			if (f_v) {
				cout << "orthogonal_group::create_random_Siegel_transformation "
						"it works" << endl;
			}
			break;
		}
		if (f_v) {
			cout << "orthogonal_group::create_random_Siegel_transformation "
					"fail, try again" << endl;
		}
	}
	if (f_vv) {
		cout << "rk_u = " << rk_u << " : ";
		Int_vec_print(cout, u, d);
		cout << endl;
		//cout << "rk_v = " << rk_v << " : ";
		cout << "v=";
		Int_vec_print(cout, v, d);
		cout << endl;
		}

	if (f_v) {
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"before O->Quadratic_form->Siegel_Transformation" << endl;
	}
	O->Quadratic_form->Siegel_Transformation(
			Mtx, v, u, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"after O->Quadratic_form->Siegel_Transformation" << endl;
	}

	if (f_vv) {
		cout << "\\rho_{";
		Int_vec_print(cout, u, d);
		cout << ",";
		Int_vec_print(cout, v, d);
		cout << "}=" << endl;
		Int_matrix_print(Mtx, d, d);
	}
	FREE_int(u);
	FREE_int(v);
	if (f_v) {
		cout << "orthogonal_group::create_random_Siegel_transformation "
				"done" << endl;
	}
}


void orthogonal_group::create_random_semisimilarity(
		int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int d = O->Quadratic_form->n;
	int i, a, b, c, k;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "orthogonal_group::create_random_semisimilarity" << endl;
	}

#if 0
	for (i = 0; i < d * d; i++) {
		Mtx[i] = 0;
	}
	for (i = 0; i < d; i++) {
		Mtx[i * d + i] = 1;
	}
#else

	O->F->Linear_algebra->identity_matrix(Mtx, d);

#endif


#if 0
	if (!f_semilinear) {
		return;
		}
#endif

	if (O->Quadratic_form->epsilon == 1) {
		Mtx[d * d] = Os.random_integer(O->F->e);
	}
	else if (O->Quadratic_form->epsilon == 0) {
		Mtx[d * d] = Os.random_integer(O->F->e);
	}
	else if (O->Quadratic_form->epsilon == -1) {
		if (O->Quadratic_form->q == 4) {
			int u, v, w, x;

			Mtx[d * d] = 1;
			for (i = 0; i < d - 2; i++) {
				if (EVEN(i)) {
					Mtx[i * d + i] = 3;
					Mtx[(i + 1) * d + i + 1] = 2;
				}
			}
			u = 1;
			v = 0;
			w = 3;
			x = 1;
			Mtx[(d - 2) * d + d - 2] = u;
			Mtx[(d - 2) * d + d - 1] = v;
			Mtx[(d - 1) * d + d - 2] = w;
			Mtx[(d - 1) * d + d - 1] = x;
		}
		else if (EVEN(O->Quadratic_form->q)) {
			cout << "orthogonal_group::create_random_semisimilarity "
					"semisimilarity for even characteristic and "
					"q != 4 not yet implemented" << endl;
			exit(1);
		}
		else {
			k = (O->F->p - 1) >> 1;
			a = O->F->primitive_element();
			b = O->F->power(a, k);
			c = O->F->frobenius_power(b, O->F->e - 1);
			Mtx[d * d - 1] = c;
			Mtx[d * d] = 1;
			cout << "orthogonal_group::create_random_semisimilarity "
					"k=(p-1)/2=" << k << " a=prim elt=" << a
					<< " b=a^k=" << b << " c=b^{p^{h-1}}=" << c << endl;

		}
	}

	if (f_v) {
		cout << "orthogonal_group::create_random_semisimilarity done" << endl;
	}
}


void orthogonal_group::create_random_similarity(
		int *Mtx, int verbose_level)
// Makes an n x n matrix only.
// Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = O->Quadratic_form->n;
	int i, r, r2;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "orthogonal_group::create_random_similarity" << endl;
	}
#if 0
	for (i = 0; i < d * d; i++) {
		Mtx[i] = 0;
	}
#if 0
	if (f_semilinear) {
		Mtx[d * d] = 0;
	}
#endif
	for (i = 0; i < d; i++) {
		Mtx[i * d + i] = 1;
	}
#else

	O->F->Linear_algebra->identity_matrix(Mtx, d);

#endif

	r = Os.random_integer(O->Quadratic_form->q - 1) + 1;
	if (f_vv) {
		cout << "orthogonal_group::create_random_similarity "
				"r=" << r << endl;
	}
	if (O->Quadratic_form->epsilon == 1) {
		for (i = 0; i < d; i++) {
			if (EVEN(i)) {
				Mtx[i * d + i] = r;
			}
		}
	}
	else if (O->Quadratic_form->epsilon == 0) {
		r2 = O->F->mult(r, r);
		if (f_vv) {
			cout << "orthogonal_group::create_random_similarity "
					"r2=" << r2 << endl;
		}
		Mtx[0 * d + 0] = r;
		for (i = 1; i < d; i++) {
			if (EVEN(i - 1)) {
				Mtx[i * d + i] = r2;
			}
		}
	}
	else if (O->Quadratic_form->epsilon == -1) {

		r2 = O->F->mult(r, r);

		for (i = 0; i < d - 2; i++) {
			if (EVEN(i)) {
				Mtx[i * d + i] = r2;
			}
		}
		i = d - 2;
		Mtx[i * d + i] = r;
		i = d - 1;
		Mtx[i * d + i] = r;
	}
	if (f_v) {
		cout << "orthogonal_group::create_random_similarity done" << endl;
	}
}

void orthogonal_group::create_random_orthogonal_reflection(
		int *Mtx, int verbose_level)
// Makes an n x n matrix only. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int alpha;
	int i;
	//int rk_z;
	//int nb_pts_affine;
	int d = O->Quadratic_form->n;
	int cnt;
	int *z;
	orbiter_kernel_system::os_interface Os;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "orthogonal_group::create_random_orthogonal_reflection" << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

	z = NEW_int(d);

#if 0
	nb_pts_affine = i_power_j(q, d);
	if (f_v) {
		cout << "orthogonal_group::create_random_orthogonal_reflection" << endl;
		cout << "nb_pts_affine=" << nb_pts_affine << endl;
	}
#endif

	cnt = 0;
	while (true) {
		if (f_v) {
			cout << "orthogonal_group::create_random_orthogonal_reflection "
					"iteration = " << cnt << endl;
		}

#if 0
		rk_z = random_integer(nb_pts_affine);
		if (f_v) {
			cout << "orthogonal_group::create_random_orthogonal_reflection "
					"iteration = " << cnt
					<< " trying rk_z=" << rk_z << endl;
		}

		AG_element_unrank(q, z, 1 /* stride */, d, rk_z);
#else
		for (i = 0; i < d; i++) {
			z[i] = Os.random_integer(O->Quadratic_form->q);
		}
#endif

		if (f_v) {
			cout << "orthogonal_group::create_random_orthogonal_reflection "
					"trying ";
			Int_vec_print(cout, z, d);
			cout << endl;
		}

		alpha = O->Quadratic_form->evaluate_quadratic_form(z, 1 /* stride */);
		if (f_v) {
			cout << "orthogonal_group::create_random_orthogonal_reflection "
					"value of the quadratic form is " << alpha << endl;
		}
		if (alpha) {
			break;
		}
		cnt++;
	}
	if (f_vv) {
		cout << "orthogonal_group::create_random_orthogonal_reflection "
				"cnt=" << cnt
				//"rk_z = " << rk_z
				<< " alpha = " << alpha << " : ";
		Int_vec_print(cout, z, d);
		cout << endl;
	}

	if (f_v) {
		cout << "orthogonal_group::create_random_orthogonal_reflection "
				"before make_orthogonal_reflection" << endl;
	}

	make_orthogonal_reflection(Mtx, z, verbose_level - 1);

	if (f_v) {
		cout << "orthogonal_group::create_random_orthogonal_reflection "
				"after make_orthogonal_reflection" << endl;
	}



	{
		int *new_Gram;
		new_Gram = NEW_int(d * d);

		if (f_v) {
			cout << "orthogonal_group::create_random_orthogonal_reflection "
					"before transform_form_matrix" << endl;
		}

		O->F->Linear_algebra->transform_form_matrix(
				Mtx,
				O->Quadratic_form->Gram_matrix, new_Gram, d,
				0 /* verbose_level */);

		if (f_v) {
			cout << "orthogonal_group::create_random_orthogonal_reflection "
					"after transform_form_matrix" << endl;
		}

		if (Sorting.int_vec_compare(
				O->Quadratic_form->Gram_matrix,
				new_Gram, d * d) != 0) {

			cout << "create_random_orthogonal_reflection "
					"The Gram matrix is not preserved" << endl;

			cout << "Gram matrix:" << endl;
			Int_vec_print_integer_matrix_width(
					cout, O->Quadratic_form->Gram_matrix,
					d, d, d, O->F->log10_of_q);

			cout << "transformed Gram matrix:" << endl;
			Int_vec_print_integer_matrix_width(
					cout, new_Gram,
					d, d, d, O->F->log10_of_q);
			exit(1);

		}
		FREE_int(new_Gram);
	}

	FREE_int(z);
	if (f_v) {
		cout << "orthogonal_group::create_random_orthogonal_reflection "
				"done" << endl;
	}

}


void orthogonal_group::make_orthogonal_reflection(
		int *M, int *z, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int Qz, Qzv, i, j;

	if (f_v) {
		cout << "orthogonal_group::make_orthogonal_reflection" << endl;
	}
	Qz = O->Quadratic_form->evaluate_quadratic_form(z, 1);
	Qzv = O->F->inverse(Qz);
	Qzv = O->F->negate(Qzv);

	O->F->Linear_algebra->mult_vector_from_the_right(
			O->Quadratic_form->Gram_matrix, z, ST_w,
			O->Quadratic_form->n, O->Quadratic_form->n);

	for (i = 0; i < O->Quadratic_form->n; i++) {
		for (j = 0; j < O->Quadratic_form->n; j++) {

			M[i * O->Quadratic_form->n + j] =
					O->F->mult(Qzv, O->F->mult(ST_w[i], z[j]));

			if (i == j) {
				M[i * O->Quadratic_form->n + j] =
						O->F->add(1, M[i * O->Quadratic_form->n + j]);
			}
		}
	}

	if (f_vv) {
		cout << "orthogonal_group::make_orthogonal_reflection created:" << endl;
		Int_vec_print_integer_matrix(
				cout, M, O->Quadratic_form->n, O->Quadratic_form->n);
	}
	if (f_v) {
		cout << "orthogonal_group::make_orthogonal_reflection done" << endl;
	}
}

void orthogonal_group::make_Siegel_Transformation(
		int *M, int *v, int *u,
	int n, int *Gram, int verbose_level)
// if u is singular and v \in \la u \ra^\perp, then
// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
// is called Siegel transform (see Taylor p. 148)
// Here Q is the quadratic form and \beta is
// the corresponding bilinear form
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, Qv, e;

	if (f_v) {
		cout << "orthogonal_group::make_Siegel_Transformation" << endl;
	}
	Qv = O->Quadratic_form->evaluate_quadratic_form(v, 1 /*stride*/);

	O->F->Linear_algebra->identity_matrix(M, n);


	// compute w^T := Gram * v^T

	O->F->Linear_algebra->mult_vector_from_the_right(Gram, v, ST_w, n, n);


	// M := M + w^T * u
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = O->F->mult(ST_w[i], u[j]);
			M[i * n + j] = O->F->add(M[i * n + j], e);
		}
	}

	// compute w^T := Gram * u^T
	O->F->Linear_algebra->mult_vector_from_the_right(Gram, u, ST_w, n, n);



	// M := M - w^T * v
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = O->F->mult(ST_w[i], v[j]);
			M[i * n + j] = O->F->add(M[i * n + j], O->F->negate(e));
		}
	}

	// M := M - Q(v) * w^T * u

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = O->F->mult(ST_w[i], u[j]);
			M[i * n + j] = O->F->add(M[i * n + j],
					O->F->mult(O->F->negate(e), Qv));
		}
	}
	if (f_vv) {
		cout << "Siegel matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, n, n, n, 2);
		O->F->Linear_algebra->transform_form_matrix(M,
				Gram, Gram2, n, 0 /* verbose_level */);
		cout << "transformed Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Gram2, n, n, n, 2);
		cout << endl;
	}

	if (f_v) {
		cout << "orthogonal_group::make_Siegel_Transformation done" << endl;
	}

}

void orthogonal_group::Siegel_move_forward_by_index(
		long int rk1, long int rk2, int *v, int *w,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward_by_index" << endl;
	}
	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_forward_by_index "
				"rk1=" << rk1 << " rk2=" << rk2 << endl;
	}
	if (rk1 == rk2) {
		for (i = 0; i < O->Quadratic_form->n; i++) {
			w[i] = v[i];
		}
		return;
	}
	O->Hyperbolic_pair->unrank_point(Sv1, 1, rk1, verbose_level - 1);
	O->Hyperbolic_pair->unrank_point(Sv2, 1, rk2, verbose_level - 1);

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_forward_by_index" << endl;
		cout << rk1 << " : ";
		Int_vec_print(cout, Sv1, O->Quadratic_form->n);
		cout << endl;
		cout << rk2 << " : ";
		Int_vec_print(cout, Sv2, O->Quadratic_form->n);
		cout << endl;
	}

	Siegel_move_forward(Sv1, Sv2, v, w, verbose_level);

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_forward_by_index "
				"moving forward: ";
		Int_vec_print(cout, v, O->Quadratic_form->n);
		cout << endl;
		cout << "            to: ";
		Int_vec_print(cout, w, O->Quadratic_form->n);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward_by_index done" << endl;
	}
}

void orthogonal_group::Siegel_move_backward_by_index(
		long int rk1, long int rk2, int *w, int *v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_backward_by_index" << endl;
	}
	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_backward_by_index "
				"rk1=" << rk1 << " rk2=" << rk2 << endl;
	}

	if (rk1 == rk2) {
		for (i = 0; i < O->Quadratic_form->n; i++) {
			v[i] = w[i];
		}
		return;
	}

	O->Hyperbolic_pair->unrank_point(
			Sv1, 1, rk1, verbose_level - 1);
	O->Hyperbolic_pair->unrank_point(
			Sv2, 1, rk2, verbose_level - 1);

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_backward_by_index" << endl;
		cout << rk1 << " : ";
		Int_vec_print(cout, Sv1, O->Quadratic_form->n);
		cout << endl;
		cout << rk2 << " : ";
		Int_vec_print(cout, Sv2, O->Quadratic_form->n);
		cout << endl;
	}

	Siegel_move_backward(Sv1, Sv2, w, v, verbose_level);

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_backward_by_index "
				"moving backward: ";
		Int_vec_print(cout, w, O->Quadratic_form->n);
		cout << endl;
		cout << "              to ";
		Int_vec_print(cout, v, O->Quadratic_form->n);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_move_backward_by_index done" << endl;
	}
}

void orthogonal_group::Siegel_move_forward(
		int *v1, int *v2, int *v3, int *v4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk1_subspace, rk2_subspace, root, i;

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward" << endl;
	}
	if (f_vv) {
		Int_vec_print(cout, v1, O->Quadratic_form->n);
		cout << endl;
		Int_vec_print(cout, v2, O->Quadratic_form->n);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward "
				"before subspace->rank_point(v1)" << endl;
	}
	rk1_subspace = O->subspace->Hyperbolic_pair->rank_point(
			v1, 1, verbose_level - 1);
	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward "
				"before subspace->rank_point(v2)" << endl;
	}
	rk2_subspace = O->subspace->Hyperbolic_pair->rank_point(
			v2, 1, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_forward "
				"rk1_subspace=" << rk1_subspace << endl;
		cout << "orthogonal_group::Siegel_move_forward "
				"rk2_subspace=" << rk2_subspace << endl;
	}
	if (rk1_subspace == rk2_subspace) {
		for (i = 0; i < O->Quadratic_form->n; i++) {
			v4[i] = v3[i];
		}
		return;
	}

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward "
				"before subspace->find_root_parabolic" << endl;
	}
	root = O->subspace->Hyperbolic_pair->find_root_parabolic(
			rk2_subspace, verbose_level - 2);

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_forward "
				"root=" << root << endl;
	}

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward "
				"before subspace->Siegel_Transformation" << endl;
	}
	O->subspace->Orthogonal_group->Siegel_Transformation(
			O->T1,
			rk1_subspace, rk2_subspace, root,
			verbose_level - 2);

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward "
				"before mult_matrix_matrix" << endl;
	}
	O->F->Linear_algebra->mult_matrix_matrix(
			v3, O->T1, v4, 1, O->Quadratic_form->n - 2, O->Quadratic_form->n - 2,
			0 /* verbose_level */);

	v4[O->Quadratic_form->n - 2] = v3[O->Quadratic_form->n - 2];
	v4[O->Quadratic_form->n - 1] = v3[O->Quadratic_form->n - 1];
	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_forward "
				"moving: ";
		Int_vec_print(cout, v3, O->Quadratic_form->n);
		cout << endl;
		cout << "     to ";
		Int_vec_print(cout, v4, O->Quadratic_form->n);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_move_forward "
				"done" << endl;
	}
}

void orthogonal_group::Siegel_move_backward(
		int *v1, int *v2, int *v3, int *v4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int rk1_subspace, rk2_subspace;
	long int root;
	int i;

	if (f_v) {
		cout << "orthogonal_group::Siegel_move_backward" << endl;
	}
	if (f_vv) {
		Int_vec_print(cout, v1, O->Quadratic_form->n);
		cout << endl;
		Int_vec_print(cout, v2, O->Quadratic_form->n);
		cout << endl;
	}
	rk1_subspace = O->subspace->Hyperbolic_pair->rank_point(
			v1, 1, verbose_level - 1);

	rk2_subspace = O->subspace->Hyperbolic_pair->rank_point(
			v2, 1, verbose_level - 1);

	if (f_vv) {
		cout << "rk1_subspace=" << rk1_subspace << endl;
		cout << "rk2_subspace=" << rk2_subspace << endl;
	}
	if (rk1_subspace == rk2_subspace) {
		for (i = 0; i < O->Quadratic_form->n; i++) {
			v4[i] = v3[i];
		}
		return;
	}

	root = O->subspace->Hyperbolic_pair->find_root_parabolic(
			rk2_subspace, verbose_level - 2);

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_backward "
				"root=" << root << endl;
		cout << "orthogonal_group::Siegel_move_backward "
				"image, to be moved back: " << endl;
		Int_vec_print(cout, v4, O->Quadratic_form->n);
		cout << endl;
	}

	O->subspace->Orthogonal_group->Siegel_Transformation(
			O->T1,
			rk1_subspace, rk2_subspace, root,
			verbose_level - 2);

	O->F->Linear_algebra->invert_matrix(
			O->T1, O->T2, O->Quadratic_form->n - 2,
			0 /* verbose_level */);

	O->F->Linear_algebra->mult_matrix_matrix(
			v3, O->T2, v4, 1,
			O->Quadratic_form->n - 2, O->Quadratic_form->n - 2,
			0 /* verbose_level */);

	v4[O->Quadratic_form->n - 2] = v3[O->Quadratic_form->n - 2];
	v4[O->Quadratic_form->n - 1] = v3[O->Quadratic_form->n - 1];

	if (f_vv) {
		cout << "orthogonal_group::Siegel_move_backward moving: ";
		Int_vec_print(cout, v3, O->Quadratic_form->n);
		cout << endl;
		cout << "     to ";
		Int_vec_print(cout, v4, O->Quadratic_form->n);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal_group::Siegel_move_backward done" << endl;
	}
}



void orthogonal_group::move_points_by_ranks_in_place(
	long int pt_from, long int pt_to,
	int nb, long int *ranks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_group::move_points_by_ranks_in_place" << endl;
	}
	int *input_coords, *output_coords;
	int i;

	input_coords = NEW_int(nb * O->Quadratic_form->n);
	output_coords = NEW_int(nb * O->Quadratic_form->n);

	for (i = 0; i < nb; i++) {
		O->Hyperbolic_pair->unrank_point(
				input_coords + i * O->Quadratic_form->n, 1, ranks[i],
				verbose_level - 1);
	}

	move_points(
			pt_from, pt_to,
		nb, input_coords, output_coords,
		verbose_level);

	for (i = 0; i < nb; i++) {
		ranks[i] = O->Hyperbolic_pair->rank_point(
				output_coords + i * O->Quadratic_form->n, 1,
				verbose_level - 1);
	}

	FREE_int(input_coords);
	FREE_int(output_coords);

	if (f_v) {
		cout << "orthogonal_group::move_points_by_ranks_in_place done" << endl;
	}
}

void orthogonal_group::move_points_by_ranks(
		long int pt_from, long int pt_to,
	int nb, long int *input_ranks, long int *output_ranks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_group::move_points_by_ranks" << endl;
	}
	int *input_coords, *output_coords;
	int i;

	input_coords = NEW_int(nb * O->Quadratic_form->n);
	output_coords = NEW_int(nb * O->Quadratic_form->n);

	for (i = 0; i < nb; i++) {
		O->Hyperbolic_pair->unrank_point(
				input_coords + i * O->Quadratic_form->n, 1,
				input_ranks[i],
				0 /*verbose_level - 1*/);
	}

	move_points(
			pt_from, pt_to,
		nb, input_coords, output_coords,
		verbose_level);

	for (i = 0; i < nb; i++) {
		output_ranks[i] = O->Hyperbolic_pair->rank_point(
				output_coords + i * O->Quadratic_form->n, 1,
				0 /*verbose_level - 1*/);
	}

	FREE_int(input_coords);
	FREE_int(output_coords);
	if (f_v) {
		cout << "orthogonal_group::move_points_by_ranks done" << endl;
	}
}

void orthogonal_group::move_points(
		long int pt_from, long int pt_to,
	int nb, int *input_coords, int *output_coords,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_group::move_points" << endl;
	}
	long int root;
	int i;
	int *tmp_coords = NULL;
	int *input_coords2;
	int *T;

	if (pt_from == pt_to) {
		for (i = 0; i < nb * O->Quadratic_form->n; i++) {
			output_coords[i] = input_coords[i];
		}
		return;
	}

	T = NEW_int(O->Quadratic_form->n * O->Quadratic_form->n);
	if (pt_from != 0) {

		tmp_coords = NEW_int(O->Quadratic_form->n * nb);
		root = find_root(
				pt_from, verbose_level - 2);

		Siegel_Transformation(T,
				pt_from /* from */,
				0 /* to */,
				root /* root */,
				verbose_level - 2);

		O->F->Linear_algebra->mult_matrix_matrix(
				input_coords,
				T, tmp_coords, nb, O->Quadratic_form->n, O->Quadratic_form->n,
				0 /* verbose_level */);
		input_coords2 = tmp_coords;
	}
	else {
		input_coords2 = input_coords;
	}

	if (f_v) {
		cout << "orthogonal_group::move_points "
				"before find_root" << endl;
	}
	root = find_root(
			pt_to, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_group::move_points "
				"after find_root" << endl;
	}

	if (f_v) {
		cout << "orthogonal_group::move_points "
				"before Siegel_Transformation" << endl;
	}
	Siegel_Transformation(T,
			0 /* from */,
			pt_to /* to */,
			root /* root */,
			verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_group::move_points "
				"after Siegel_Transformation" << endl;
	}

	if (f_v) {
		cout << "orthogonal_group::move_points "
				"before mult_matrix_matrix" << endl;
	}
	O->F->Linear_algebra->mult_matrix_matrix(
			input_coords2, T, output_coords, nb, 5, 5,
			0 /* verbose_level */);
	if (f_v) {
		cout << "orthogonal_group::move_points "
				"before mult_matrix_matrix" << endl;
	}

	if (tmp_coords) {
		FREE_int(tmp_coords);
	}

	FREE_int(T);
	if (f_v) {
		cout << "orthogonal_group::move_points done" << endl;
	}
}


void orthogonal_group::test_Siegel(
		int index, int verbose_level)
{
	int rk1, rk2, rk1_subspace, rk2_subspace, root, j, rk3, cnt, u, t2;

	rk1 = O->Hyperbolic_pair->type_and_index_to_point_rk(
			5, 0, verbose_level);
	cout << 0 << " : " << rk1 << " : ";
	O->Hyperbolic_pair->unrank_point(
			O->Hyperbolic_pair->v1, 1, rk1,
			verbose_level - 1);
	Int_vec_print(cout, O->Hyperbolic_pair->v1, O->Quadratic_form->n);
	cout << endl;

	rk2 = O->Hyperbolic_pair->type_and_index_to_point_rk(
			5, index, verbose_level);
	cout << index << " : " << rk2 << " : ";
	O->Hyperbolic_pair->unrank_point(
			O->Hyperbolic_pair->v2, 1, rk2,
			verbose_level - 1);
	Int_vec_print(cout, O->Hyperbolic_pair->v2, O->Quadratic_form->n);
	cout << endl;

	rk1_subspace = O->subspace->Hyperbolic_pair->rank_point(
			O->Hyperbolic_pair->v1, 1,
			verbose_level - 1);
	rk2_subspace = O->subspace->Hyperbolic_pair->rank_point(
			O->Hyperbolic_pair->v2, 1,
			verbose_level - 1);
	cout << "rk1_subspace=" << rk1_subspace << endl;
	cout << "rk2_subspace=" << rk2_subspace << endl;

	root = O->subspace->Hyperbolic_pair->find_root_parabolic(
			rk2_subspace,
			verbose_level);
	O->subspace->Orthogonal_group->Siegel_Transformation(O->T1,
			rk1_subspace, rk2_subspace, root,
			verbose_level);

	cout << "Siegel map takes 1st point to" << endl;
	O->F->Linear_algebra->mult_matrix_matrix(
			O->Hyperbolic_pair->v1, O->T1,
			O->Hyperbolic_pair->v3, 1,
			O->Quadratic_form->n - 2, O->Quadratic_form->n - 2,
			0 /* verbose_level */);
	Int_vec_print(
			cout, O->Hyperbolic_pair->v3, O->Quadratic_form->n - 2);
	cout << endl;

	cnt = 0;

	t2 = 1;
	for (j = 0; j < O->subspace->Hyperbolic_pair->P[t2 - 1]; j++) {
		if (O->Quadratic_form->f_even) {
			cout << "f_even" << endl;
			exit(1);
		}
		O->Hyperbolic_pair->parabolic_neighbor51_odd_unrank(
				j, O->Hyperbolic_pair->v3, false);
		//rk3 = type_and_index_to_point_rk(t2, j);
		//unrank_point(v3, 1, rk3);
		rk3 = O->Hyperbolic_pair->rank_point(
				O->Hyperbolic_pair->v3, 1,
				verbose_level - 1);

		u = O->Quadratic_form->evaluate_bilinear_form(
				O->Hyperbolic_pair->v1, O->Hyperbolic_pair->v3, 1);
		if (u) {
			cout << "error, u not zero" << endl;
		}

		//if (test_if_minimal_on_line(v3, v1, v_tmp)) {


		cout << "Siegel map takes 2nd point ";
		cout << cnt << " : " << j << " : " << rk3 << " : ";
		Int_vec_print(cout, O->Hyperbolic_pair->v3, O->Quadratic_form->n);
		cout << " to ";
		O->F->Linear_algebra->mult_matrix_matrix(
				O->Hyperbolic_pair->v3, O->T1,
				O->Hyperbolic_pair->v_tmp, 1,
				O->Quadratic_form->n - 2, O->Quadratic_form->n - 2,
				0 /* verbose_level */);


		O->Hyperbolic_pair->v_tmp[O->Quadratic_form->n - 2] = O->Hyperbolic_pair->v3[O->Quadratic_form->n - 2];
		O->Hyperbolic_pair->v_tmp[O->Quadratic_form->n - 1] = O->Hyperbolic_pair->v3[O->Quadratic_form->n - 1];
		Int_vec_print(
				cout, O->Hyperbolic_pair->v_tmp, O->Quadratic_form->n);


		//cout << "find_minimal_point_on_line " << endl;
		//find_minimal_point_on_line(v_tmp, v2, v4);

		//cout << " minrep: ";
		//int_vec_print(cout, v4, n);

		//normalize_point(v4, 1);
		//cout << " normalized: ";
		//int_vec_print(cout, v4, n);

		cout << endl;

		cnt++;
		//}
	}
	cout << endl;
}


}}}


