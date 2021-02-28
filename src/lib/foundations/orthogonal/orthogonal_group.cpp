/*
 * orthogonal_group.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


long int orthogonal::find_root(long int rk2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int ret;

	if (f_v) {
		cout << "orthogonal::find_root" << endl;
	}
	if (epsilon == 1) {
		ret = find_root_hyperbolic(rk2, m, verbose_level);
		}
	else if (epsilon == 0) {
		ret = find_root_parabolic(rk2, verbose_level);
		}
	else {
		cout << "orthogonal::find_root epsilon = " << epsilon << endl;
		exit(1);
		}
	if (f_v) {
		cout << "orthogonal::find_root done" << endl;
	}
	return ret;
}


void orthogonal::Siegel_map_between_singular_points(int *T,
		long int rk_from, long int rk_to, long int root, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::Siegel_map_between_singular_points" << endl;
	}
	F->Siegel_map_between_singular_points(T,
		rk_from, rk_to, root,
		epsilon, n,
		form_c1, form_c2, form_c3, Gram_matrix,
		verbose_level);
	if (f_v) {
		cout << "orthogonal::Siegel_map_between_singular_points done" << endl;
	}
}

void orthogonal::Siegel_map_between_singular_points_hyperbolic(int *T,
	long int rk_from, long int rk_to, long int root, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Gram;

	if (f_v) {
		cout << "orthogonal::Siegel_map_between_singular_points_hyperbolic" << endl;
	}
	F->Gram_matrix(
			1, 2 * m - 1, 0,0,0, Gram, verbose_level - 1);
	F->Siegel_map_between_singular_points(T,
		rk_from, rk_to, root,
		epsilon, 2 * m,
		0, 0, 0, Gram,
		verbose_level);
	FREE_int(Gram);
	if (f_v) {
		cout << "orthogonal::Siegel_map_between_singular_points_hyperbolic done" << endl;
	}
}

void orthogonal::Siegel_Transformation(int *T,
	long int rk_from, long int rk_to, long int root,
	int verbose_level)
// root is not perp to from and to.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "Siegel_Transformation rk_from=" << rk_from
				<< " rk_to=" << rk_to << " root=" << root << endl;
	}
	Siegel_Transformation2(T,
		rk_from, rk_to, root,
		STr_B, STr_Bv, STr_w, STr_z, STr_x,
		verbose_level);
	if (f_v) {
		cout << "Siegel_Transformation done" << endl;
	}
}

void orthogonal::Siegel_Transformation2(int *T,
	long int rk_from, long int rk_to, long int root,
	int *B, int *Bv, int *w, int *z, int *x,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *From, *To, *Root;

	if (f_v) {
		cout << "orthogonal::Siegel_Transformation2" << endl;
	}
	From = NEW_int(n);
	To = NEW_int(n);
	Root = NEW_int(n);
	unrank_point(Root, 1, root, verbose_level - 1);
	unrank_point(From, 1, rk_from, verbose_level - 1);
	unrank_point(To, 1, rk_to, verbose_level - 1);
	if (f_vv) {
		cout << "root: ";
		Orbiter->Int_vec.print(cout, Root, n);
		cout << endl;
		cout << "rk_from: ";
		Orbiter->Int_vec.print(cout, From, n);
		cout << endl;
		cout << "rk_to: ";
		Orbiter->Int_vec.print(cout, To, n);
		cout << endl;
		}

	Siegel_Transformation3(T,
		From, To, Root,
		B, Bv, w, z, x,
		verbose_level - 1);
	FREE_int(From);
	FREE_int(To);
	FREE_int(Root);
	if (f_vv) {
		cout << "the Siegel transformation is:" << endl;
		print_integer_matrix(cout, T, n, n);
	}
	if (f_v) {
		cout << "orthogonal::Siegel_Transformation2 done" << endl;
	}
}

void orthogonal::Siegel_Transformation3(int *T,
	int *from, int *to, int *root,
	int *B, int *Bv, int *w, int *z, int *x,
	int verbose_level)
{
	int i, j, a, b, av, bv, minus_one;
	//int k;
	int *Gram;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	combinatorics_domain Combi;

	if (f_v) {
		cout << "orthogonal::Siegel_Transformation3" << endl;
	}
	//k = n - 1;
	Gram = Gram_matrix;
	if (f_vv) {
		cout << "n=" << n << endl;
		cout << "Gram matrix:" << endl;
		Combi.print_int_matrix(cout, Gram, n, n);
	}

	//Q_epsilon_unrank(*F, B, 1, epsilon, k,
	//form_c1, form_c2, form_c3, root);
	//Q_epsilon_unrank(*F, B + d, 1, epsilon, k,
	//form_c1, form_c2, form_c3, rk_from);
	//Q_epsilon_unrank(*F, w, 1, epsilon, k,
	//form_c1, form_c2, form_c3, rk_to);

	for (i = 0; i < n; i++) {
		B[i] = root[i];
		B[n + i] = from[i];
		w[i] = to[i];
	}
	if (f_vv) {
		cout << "root: ";
		Orbiter->Int_vec.print(cout, B, n);
		cout << endl;
		cout << "from: ";
		Orbiter->Int_vec.print(cout, B + n, n);
		cout << endl;
		cout << "to: ";
		Orbiter->Int_vec.print(cout, w, n);
		cout << endl;
	}

	a = F->evaluate_bilinear_form(B, B + n, n, Gram);
	b = F->evaluate_bilinear_form(B, w, n, Gram);
	av = F->inverse(a);
	bv = F->inverse(b);
	for (i = 0; i < n; i++) {
		B[n + i] = F->mult(B[n + i], av);
		w[i] = F->mult(w[i], bv);
	}
	for (i = 2; i < n; i++) {
		for (j = 0; j < n; j++) {
			B[i * n + j] = 0;
		}
	}

	if (f_vv) {
		cout << "before perp, the matrix B is:" << endl;
		print_integer_matrix(cout, B, n, n);
	}
	F->perp(n, 2, B, Gram, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix B is:" << endl;
		print_integer_matrix(cout, B, n, n);
	}
	F->invert_matrix(B, Bv, n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "the matrix Bv is:" << endl;
		print_integer_matrix(cout, B, n, n);
	}
	F->mult_matrix_matrix(w, Bv, z, 1, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the coefficient vector z is:" << endl;
		print_integer_matrix(cout, z, 1, n);
	}
	z[0] = 0;
	z[1] = 0;
	if (f_vv) {
		cout << "the coefficient vector z is:" << endl;
		print_integer_matrix(cout, z, 1, n);
	}
	F->mult_matrix_matrix(z, B, x, 1, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x is:" << endl;
		print_integer_matrix(cout, x, 1, n);
	}
	minus_one = F->negate(1);
	for (i = 0; i < n; i++) {
		x[i] = F->mult(x[i], minus_one);
	}
	if (f_vv) {
		cout << "the vector -x is:" << endl;
		print_integer_matrix(cout, x, 1, n);
	}
	make_Siegel_Transformation(T, x, B, n, Gram, FALSE);
	if (f_vv) {
		cout << "the Siegel transformation is:" << endl;
		print_integer_matrix(cout, T, n, n);
	}
	if (f_v) {
		cout << "orthogonal::Siegel_Transformation3 done" << endl;
	}
}

void orthogonal::random_generator_for_orthogonal_group(
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
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::random_generator_for_orthogonal_group" << endl;
		cout << "f_action_is_semilinear=" << f_action_is_semilinear << endl;
		cout << "f_siegel=" << f_siegel << endl;
		cout << "f_reflection=" << f_reflection << endl;
		cout << "f_similarity=" << f_similarity << endl;
		cout << "f_semisimilarity=" << f_semisimilarity << endl;
	}


	while (TRUE) {
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
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing Siegel_transformation" << endl;
		}
		create_random_Siegel_transformation(Mtx, verbose_level /*- 2 */);
		if (f_action_is_semilinear) {
			Mtx[n * n] = 0;
		}
	}
	else if (r == 1) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing orthogonal reflection" << endl;
		}

		create_random_orthogonal_reflection(Mtx, verbose_level - 2);
		if (f_action_is_semilinear) {
			Mtx[n * n] = 0;
		}
	}
	else if (r == 2) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing similarity" << endl;
		}
		create_random_similarity(Mtx, verbose_level - 2);
		if (f_action_is_semilinear) {
			Mtx[n * n] = 0;
		}
	}
	else if (r == 3) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing random similarity" << endl;
		}
		create_random_semisimilarity(Mtx, verbose_level - 2);
	}
	if (f_v) {
		cout << "orthogonal::random_generator_for_orthogonal_group "
				"done" << endl;
	}
}


void orthogonal::create_random_Siegel_transformation(
		int *Mtx, int verbose_level)
// Only makes a n x n matrix. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk_u, alpha, i;
	int nb_pts; //, nb_pts_affine;
	int k = m; // the Witt index, previously orthogonal_k;
	int d = n;
	int *u, *v;
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation" << endl;
	}

	u = NEW_int(d);
	v = NEW_int(d);

	nb_pts = nb_points; //nb_pts_Qepsilon(epsilon, d - 1, q);
	//nb_pts_affine = i_power_j(q, d);

	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation "
				"q=" << q << endl;
		cout << "orthogonal::create_random_Siegel_transformation "
				"d=" << d << endl;
		cout << "orthogonal::create_random_Siegel_transformation "
				"Witt index k=" << k << endl;
		cout << "orthogonal::create_random_Siegel_transformation "
				"nb_pts=" << nb_pts << endl;
		//cout << "orthogonal::create_random_Siegel_transformation "
		//		"nb_pts_affine=" << nb_pts_affine << endl;
	}

	rk_u = Os.random_integer(nb_pts);
	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation "
				"rk_u=" << rk_u << endl;
	}
	unrank_point(u, 1, rk_u, 0 /* verbose_level*/);
	//Q_epsilon_unrank(*F, u, 1 /*stride*/, epsilon, d - 1,
	// form_c1, form_c2, form_c3, rk_u);

	while (TRUE) {

#if 0
		rk_v = random_integer(nb_pts_affine);
		if (f_v) {
			cout << "orthogonal::create_random_Siegel_transformation "
					"trying rk_v=" << rk_v << endl;
			}
		AG_element_unrank(q, v, 1 /* stride */, d, rk_v);
#else
		for (i = 0; i < d; i++) {
			v[i] = Os.random_integer(q);
		}

#endif

		alpha = F->evaluate_bilinear_form(
				u, v, d, Gram_matrix);
		if (alpha == 0) {
			if (f_v) {
				cout << "orthogonal::create_random_Siegel_transformation "
						"it works" << endl;
			}
			break;
		}
		if (f_v) {
			cout << "orthogonal::create_random_Siegel_transformation "
					"fail, try again" << endl;
		}
	}
	if (f_vv) {
		cout << "rk_u = " << rk_u << " : ";
		Orbiter->Int_vec.print(cout, u, d);
		cout << endl;
		//cout << "rk_v = " << rk_v << " : ";
		cout << "v=";
		Orbiter->Int_vec.print(cout, v, d);
		cout << endl;
		}

	F->Siegel_Transformation(
			epsilon, d - 1,
			form_c1, form_c2, form_c3,
			Mtx, v, u, verbose_level - 1);

	if (f_vv) {
		cout << "form_c1=" << form_c1 << endl;
		cout << "form_c2=" << form_c2 << endl;
		cout << "form_c3=" << form_c3 << endl;
		cout << "\\rho_{";
		Orbiter->Int_vec.print(cout, u, d);
		cout << ",";
		Orbiter->Int_vec.print(cout, v, d);
		cout << "}=" << endl;
		int_matrix_print(Mtx, d, d);
	}
	FREE_int(u);
	FREE_int(v);
	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation "
				"done" << endl;
	}
}


void orthogonal::create_random_semisimilarity(int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int d = n;
	int i, a, b, c, k;
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::create_random_semisimilarity" << endl;
	}
	for (i = 0; i < d * d; i++) {
		Mtx[i] = 0;
	}
	for (i = 0; i < d; i++) {
		Mtx[i * d + i] = 1;
	}

#if 0
	if (!f_semilinear) {
		return;
		}
#endif

	if (epsilon == 1) {
		Mtx[d * d] = Os.random_integer(F->e);
	}
	else if (epsilon == 0) {
		Mtx[d * d] = Os.random_integer(F->e);
	}
	else if (epsilon == -1) {
		if (q == 4) {
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
		else if (EVEN(q)) {
			cout << "orthogonal::create_random_semisimilarity "
					"semisimilarity for even characteristic and "
					"q != 4 not yet implemented" << endl;
			exit(1);
		}
		else {
			k = (F->p - 1) >> 1;
			a = F->primitive_element();
			b = F->power(a, k);
			c = F->frobenius_power(b, F->e - 1);
			Mtx[d * d - 1] = c;
			Mtx[d * d] = 1;
			cout << "orthogonal::create_random_semisimilarity "
					"k=(p-1)/2=" << k << " a=prim elt=" << a
					<< " b=a^k=" << b << " c=b^{p^{h-1}}=" << c << endl;

		}
	}

	if (f_v) {
		cout << "orthogonal::create_random_semisimilarity done" << endl;
	}
}


void orthogonal::create_random_similarity(int *Mtx, int verbose_level)
// Only makes a n x n matrix. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n;
	int i, r, r2;
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::create_random_similarity" << endl;
	}
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
	r = Os.random_integer(q - 1) + 1;
	if (f_vv) {
		cout << "orthogonal::create_random_similarity "
				"r=" << r << endl;
	}
	if (epsilon == 1) {
		for (i = 0; i < d; i++) {
			if (EVEN(i)) {
				Mtx[i * d + i] = r;
			}
		}
	}
	else if (epsilon == 0) {
		r2 = F->mult(r, r);
		if (f_vv) {
			cout << "orthogonal::create_random_similarity "
					"r2=" << r2 << endl;
		}
		Mtx[0 * d + 0] = r;
		for (i = 1; i < d; i++) {
			if (EVEN(i - 1)) {
				Mtx[i * d + i] = r2;
			}
		}
	}
	else if (epsilon == -1) {
		r2 = F->mult(r, r);
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
		cout << "orthogonal::create_random_similarity done" << endl;
	}
}

void orthogonal::create_random_orthogonal_reflection(
		int *Mtx, int verbose_level)
// Only makes a n x n matrix. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int alpha;
	int i;
	//int rk_z;
	//int nb_pts_affine;
	int d = n;
	int cnt;
	int *z;
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}

	z = NEW_int(d);

#if 0
	nb_pts_affine = i_power_j(q, d);
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection" << endl;
		cout << "nb_pts_affine=" << nb_pts_affine << endl;
		}
#endif

	cnt = 0;
	while (TRUE) {
		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"iteration = " << cnt << endl;
			}

#if 0
		rk_z = random_integer(nb_pts_affine);
		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"iteration = " << cnt
					<< " trying rk_z=" << rk_z << endl;
			}

		AG_element_unrank(q, z, 1 /* stride */, d, rk_z);
#else
		for (i = 0; i < d; i++) {
			z[i] = Os.random_integer(q);
		}
#endif

		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"trying ";
			Orbiter->Int_vec.print(cout, z, d);
			cout << endl;
		}

		alpha = evaluate_quadratic_form(z, 1 /* stride */);
		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"value of the quadratic form is " << alpha << endl;
		}
		if (alpha) {
			break;
			}
		cnt++;
		}
	if (f_vv) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"cnt=" << cnt
				//"rk_z = " << rk_z
				<< " alpha = " << alpha << " : ";
		Orbiter->Int_vec.print(cout, z, d);
		cout << endl;
		}

	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"before make_orthogonal_reflection" << endl;
		}

	make_orthogonal_reflection(Mtx, z, verbose_level - 1);

	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"after make_orthogonal_reflection" << endl;
		}



	{
		int *new_Gram;
		new_Gram = NEW_int(d * d);

		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"before transform_form_matrix" << endl;
			}

		F->transform_form_matrix(Mtx, Gram_matrix, new_Gram, d, 0 /* verbose_level */);

		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"after transform_form_matrix" << endl;
			}

		if (int_vec_compare(Gram_matrix, new_Gram, d * d) != 0) {
			cout << "create_random_orthogonal_reflection "
					"The Gram matrix is not preserved" << endl;
			cout << "Gram matrix:" << endl;
			print_integer_matrix_width(cout, Gram_matrix,
					d, d, d, F->log10_of_q);
			cout << "transformed Gram matrix:" << endl;
			print_integer_matrix_width(cout, new_Gram,
					d, d, d, F->log10_of_q);
			exit(1);
			}
		FREE_int(new_Gram);
	}

	FREE_int(z);
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"done" << endl;
		}

}


void orthogonal::make_orthogonal_reflection(
		int *M, int *z, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int Qz, Qzv, i, j;

	if (f_v) {
		cout << "orthogonal::make_orthogonal_reflection" << endl;
	}
	Qz = evaluate_quadratic_form(z, 1);
	Qzv = F->inverse(Qz);
	Qzv = F->negate(Qzv);

	F->mult_vector_from_the_right(Gram_matrix, z, ST_w, n, n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			M[i * n + j] = F->mult(Qzv, F->mult(ST_w[i], z[j]));
			if (i == j) {
				M[i * n + j] = F->add(1, M[i * n + j]);
			}
		}
	}

	if (f_vv) {
		cout << "orthogonal::make_orthogonal_reflection created:" << endl;
		print_integer_matrix(cout, M, n, n);
	}
	if (f_v) {
		cout << "orthogonal::make_orthogonal_reflection done" << endl;
	}
}

void orthogonal::make_Siegel_Transformation(int *M, int *v, int *u,
	int n, int *Gram, int verbose_level)
// if u is singular and v \in \la u \ra^\perp, then
// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
// is called the Siegel transform (see Taylor p. 148)
// Here Q is the quadratic form and \beta is
// the corresponding bilinear form
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, Qv, e;

	if (f_v) {
		cout << "orthogonal::make_Siegel_Transformation" << endl;
	}
	Qv = F->evaluate_quadratic_form(
			v, 1 /*stride*/,
			epsilon, n - 1,
			form_c1, form_c2, form_c3);
	F->identity_matrix(M, n);


	// compute w^T := Gram * v^T

	F->mult_vector_from_the_right(Gram, v, ST_w, n, n);


	// M := M + w^T * u
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = F->mult(ST_w[i], u[j]);
			M[i * n + j] = F->add(M[i * n + j], e);
		}
	}

	// compute w^T := Gram * u^T
	F->mult_vector_from_the_right(Gram, u, ST_w, n, n);



	// M := M - w^T * v
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = F->mult(ST_w[i], v[j]);
			M[i * n + j] = F->add(M[i * n + j], F->negate(e));
		}
	}

	// M := M - Q(v) * w^T * u

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = F->mult(ST_w[i], u[j]);
			M[i * n + j] = F->add(M[i * n + j],
					F->mult(F->negate(e), Qv));
		}
	}
	if (f_vv) {
		cout << "Siegel matrix:" << endl;
		print_integer_matrix_width(cout, M, n, n, n, 2);
		F->transform_form_matrix(M, Gram, Gram2, n, 0 /* verbose_level */);
		cout << "transformed Gram matrix:" << endl;
		print_integer_matrix_width(cout, Gram2, n, n, n, 2);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal::make_Siegel_Transformation done" << endl;
	}

}

void orthogonal::Siegel_move_forward_by_index(
		long int rk1, long int rk2, int *v, int *w, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "orthogonal::Siegel_move_forward_by_index" << endl;
	}
	if (f_vv) {
		cout << "orthogonal::Siegel_move_forward_by_index "
				"rk1=" << rk1 << " rk2=" << rk2 << endl;
		}
	if (rk1 == rk2) {
		for (i = 0; i < n; i++)
			w[i] = v[i];
		return;
		}
	unrank_point(Sv1, 1, rk1, verbose_level - 1);
	unrank_point(Sv2, 1, rk2, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_forward_by_index" << endl;
		cout << rk1 << " : ";
		Orbiter->Int_vec.print(cout, Sv1, n);
		cout << endl;
		cout << rk2 << " : ";
		Orbiter->Int_vec.print(cout, Sv2, n);
		cout << endl;
		}
	Siegel_move_forward(Sv1, Sv2, v, w, verbose_level);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_forward_by_index moving forward: ";
		Orbiter->Int_vec.print(cout, v, n);
		cout << endl;
		cout << "            to: ";
		Orbiter->Int_vec.print(cout, w, n);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::Siegel_move_forward_by_index done" << endl;
	}
}

void orthogonal::Siegel_move_backward_by_index(
		long int rk1, long int rk2, int *w, int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "orthogonal::Siegel_move_backward_by_index" << endl;
	}
	if (f_vv) {
		cout << "orthogonal::Siegel_move_backward_by_index "
				"rk1=" << rk1 << " rk2=" << rk2 << endl;
		}
	if (rk1 == rk2) {
		for (i = 0; i < n; i++)
			v[i] = w[i];
		return;
		}
	unrank_point(Sv1, 1, rk1, verbose_level - 1);
	unrank_point(Sv2, 1, rk2, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_backward_by_index" << endl;
		cout << rk1 << " : ";
		Orbiter->Int_vec.print(cout, Sv1, n);
		cout << endl;
		cout << rk2 << " : ";
		Orbiter->Int_vec.print(cout, Sv2, n);
		cout << endl;
		}
	Siegel_move_backward(Sv1, Sv2, w, v, verbose_level);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_backward_by_index moving backward: ";
		Orbiter->Int_vec.print(cout, w, n);
		cout << endl;
		cout << "              to ";
		Orbiter->Int_vec.print(cout, v, n);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::Siegel_move_backward_by_index done" << endl;
	}
}

void orthogonal::Siegel_move_forward(
		int *v1, int *v2, int *v3, int *v4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk1_subspace, rk2_subspace, root, i;

	if (f_v) {
		cout << "orthogonal::Siegel_move_forward" << endl;
	}
	if (f_vv) {
		Orbiter->Int_vec.print(cout, v1, n);
		cout << endl;
		Orbiter->Int_vec.print(cout, v2, n);
		cout << endl;
		}
	rk1_subspace = subspace->rank_point(v1, 1, verbose_level - 1);
	rk2_subspace = subspace->rank_point(v2, 1, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_forward rk1_subspace=" << rk1_subspace << endl;
		cout << "orthogonal::Siegel_move_forward rk2_subspace=" << rk2_subspace << endl;
		}
	if (rk1_subspace == rk2_subspace) {
		for (i = 0; i < n; i++)
			v4[i] = v3[i];
		return;
		}

	root = subspace->find_root_parabolic(rk2_subspace, verbose_level - 2);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_forward root=" << root << endl;
		}
	subspace->Siegel_Transformation(T1,
			rk1_subspace, rk2_subspace, root, verbose_level - 2);
	F->mult_matrix_matrix(v3, T1, v4, 1, n - 2, n - 2,
			0 /* verbose_level */);
	v4[n - 2] = v3[n - 2];
	v4[n - 1] = v3[n - 1];
	if (f_vv) {
		cout << "orthogonal::Siegel_move_forward moving: ";
		Orbiter->Int_vec.print(cout, v3, n);
		cout << endl;
		cout << "     to ";
		Orbiter->Int_vec.print(cout, v4, n);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::Siegel_move_forward done" << endl;
	}
}

void orthogonal::Siegel_move_backward(
		int *v1, int *v2, int *v3, int *v4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int rk1_subspace, rk2_subspace;
	long int root;
	int i;

	if (f_v) {
		cout << "orthogonal::Siegel_move_backward" << endl;
	}
	if (f_vv) {
		Orbiter->Int_vec.print(cout, v1, n);
		cout << endl;
		Orbiter->Int_vec.print(cout, v2, n);
		cout << endl;
		}
	rk1_subspace = subspace->rank_point(v1, 1, verbose_level - 1);
	rk2_subspace = subspace->rank_point(v2, 1, verbose_level - 1);
	if (f_vv) {
		cout << "rk1_subspace=" << rk1_subspace << endl;
		cout << "rk2_subspace=" << rk2_subspace << endl;
		}
	if (rk1_subspace == rk2_subspace) {
		for (i = 0; i < n; i++)
			v4[i] = v3[i];
		return;
		}

	root = subspace->find_root_parabolic(
			rk2_subspace, verbose_level - 2);
	if (f_vv) {
		cout << "orthogonal::Siegel_move_backward root=" << root << endl;
		cout << "orthogonal::Siegel_move_backward image, to be moved back: " << endl;
		Orbiter->Int_vec.print(cout, v4, n);
		cout << endl;
		}
	subspace->Siegel_Transformation(T1,
			rk1_subspace, rk2_subspace, root, verbose_level - 2);
	F->invert_matrix(T1, T2, n - 2, 0 /* verbose_level */);
	F->mult_matrix_matrix(v3, T2, v4, 1, n - 2, n - 2,
			0 /* verbose_level */);
	v4[n - 2] = v3[n - 2];
	v4[n - 1] = v3[n - 1];
	if (f_vv) {
		cout << "orthogonal::Siegel_move_backward moving: ";
		Orbiter->Int_vec.print(cout, v3, n);
		cout << endl;
		cout << "     to ";
		Orbiter->Int_vec.print(cout, v4, n);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::Siegel_move_backward done" << endl;
	}
}



void orthogonal::move_points_by_ranks_in_place(
	long int pt_from, long int pt_to,
	int nb, long int *ranks, int verbose_level)
{
	int *input_coords, *output_coords;
	int i;

	input_coords = NEW_int(nb * n);
	output_coords = NEW_int(nb * n);
	for (i = 0; i < nb; i++) {
		unrank_point(
				input_coords + i * n, 1, ranks[i],
				verbose_level - 1);
		}

	move_points(pt_from, pt_to,
		nb, input_coords, output_coords, verbose_level);

	for (i = 0; i < nb; i++) {
		ranks[i] = rank_point(
				output_coords + i * n, 1, verbose_level - 1);
		}

	FREE_int(input_coords);
	FREE_int(output_coords);
}

void orthogonal::move_points_by_ranks(long int pt_from, long int pt_to,
	int nb, long int *input_ranks, long int *output_ranks,
	int verbose_level)
{
	int *input_coords, *output_coords;
	int i;

	input_coords = NEW_int(nb * n);
	output_coords = NEW_int(nb * n);
	for (i = 0; i < nb; i++) {
		unrank_point(input_coords + i * n, 1,
				input_ranks[i], verbose_level - 1);
		}

	move_points(pt_from, pt_to,
		nb, input_coords, output_coords, verbose_level);

	for (i = 0; i < nb; i++) {
		output_ranks[i] = rank_point(
				output_coords + i * n, 1, verbose_level - 1);
		}

	FREE_int(input_coords);
	FREE_int(output_coords);
}

void orthogonal::move_points(long int pt_from, long int pt_to,
	int nb, int *input_coords, int *output_coords,
	int verbose_level)
{
	long int root;
	int i;
	int *tmp_coords = NULL;
	int *input_coords2;
	int *T;

	if (pt_from == pt_to) {
		for (i = 0; i < nb * n; i++) {
			output_coords[i] = input_coords[i];
			}
		return;
		}

	T = NEW_int(n * n);
	if (pt_from != 0) {

		tmp_coords = NEW_int(n * nb);
		root = find_root(pt_from, verbose_level - 2);
		Siegel_Transformation(T,
				pt_from /* from */,
				0 /* to */,
				root /* root */,
				verbose_level - 2);
		F->mult_matrix_matrix(input_coords,
				T, tmp_coords, nb, n, n,
				0 /* verbose_level */);
		input_coords2 = tmp_coords;
		}
	else {
		input_coords2 = input_coords;
		}

	root = find_root(pt_to, verbose_level - 2);
	Siegel_Transformation(T,
			0 /* from */,
			pt_to /* to */,
			root /* root */,
			verbose_level - 2);
	F->mult_matrix_matrix(input_coords2, T, output_coords, nb, 5, 5,
			0 /* verbose_level */);

	if (tmp_coords) FREE_int(tmp_coords);

	FREE_int(T);
}


void orthogonal::test_Siegel(int index, int verbose_level)
{
	int rk1, rk2, rk1_subspace, rk2_subspace, root, j, rk3, cnt, u, t2;

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	cout << 0 << " : " << rk1 << " : ";
	unrank_point(v1, 1, rk1, verbose_level - 1);
	Orbiter->Int_vec.print(cout, v1, n);
	cout << endl;

	rk2 = type_and_index_to_point_rk(5, index, verbose_level);
	cout << index << " : " << rk2 << " : ";
	unrank_point(v2, 1, rk2, verbose_level - 1);
	Orbiter->Int_vec.print(cout, v2, n);
	cout << endl;

	rk1_subspace = subspace->rank_point(v1, 1, verbose_level - 1);
	rk2_subspace = subspace->rank_point(v2, 1, verbose_level - 1);
	cout << "rk1_subspace=" << rk1_subspace << endl;
	cout << "rk2_subspace=" << rk2_subspace << endl;

	root = subspace->find_root_parabolic(
			rk2_subspace, verbose_level);
	subspace->Siegel_Transformation(T1,
			rk1_subspace, rk2_subspace, root, verbose_level);

	cout << "Siegel map takes 1st point to" << endl;
	F->mult_matrix_matrix(v1, T1, v3, 1, n - 2, n - 2,
			0 /* verbose_level */);
	Orbiter->Int_vec.print(cout, v3, n - 2);
	cout << endl;

	cnt = 0;

	t2 = 1;
	for (j = 0; j < subspace->P[t2 - 1]; j++) {
		if (f_even) {
			cout << "f_even" << endl;
			exit(1);
			}
		parabolic_neighbor51_odd_unrank(j, v3, FALSE);
		//rk3 = type_and_index_to_point_rk(t2, j);
		//unrank_point(v3, 1, rk3);
		rk3 = rank_point(v3, 1, verbose_level - 1);

		u = evaluate_bilinear_form(v1, v3, 1);
		if (u) {
			cout << "error, u not zero" << endl;
			}

		//if (test_if_minimal_on_line(v3, v1, v_tmp)) {


		cout << "Siegel map takes 2nd point ";
		cout << cnt << " : " << j << " : " << rk3 << " : ";
		Orbiter->Int_vec.print(cout, v3, n);
		cout << " to ";
		F->mult_matrix_matrix(v3, T1, v_tmp, 1, n - 2, n - 2,
				0 /* verbose_level */);


		v_tmp[n - 2] = v3[n - 2];
		v_tmp[n - 1] = v3[n - 1];
		Orbiter->Int_vec.print(cout, v_tmp, n);


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


}}


