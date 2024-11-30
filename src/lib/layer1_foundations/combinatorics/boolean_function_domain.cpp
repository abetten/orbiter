/*
 * boolean_function_domain.cpp
 *
 *  Created on: Nov 7, 2020
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


boolean_function_domain::boolean_function_domain()
{
	n = n2 = Q = bent = near_bent = 0;
	NN = NULL;
	N = 0;
	Fq = NULL;
	//FQ = NULL;
	Poly = NULL;
	A_poly = NULL;
	B_poly = NULL;
	Kernel = NULL;
	dim_kernel = 0;
	affine_points = NULL;
	v = v1 = w = f = f2 = F = T = W = f_proj = f_proj2 = NULL;
}

boolean_function_domain::~boolean_function_domain()
{
	int degree;

	if (NN) {
		FREE_OBJECT(NN);
	}
#if 0
	if (Fq) {
		FREE_OBJECT(Fq);
	}
	if (FQ) {
		FREE_OBJECT(FQ);
	}
#endif
	if (Poly) {
		FREE_OBJECTS(Poly);
	}
	if (A_poly) {
		for (degree = 1; degree <= n; degree++) {
			FREE_int(A_poly[degree]);
			FREE_int(B_poly[degree]);
		}
		FREE_pint(A_poly);
		FREE_pint(B_poly);
	}
	if (Kernel) {
		FREE_int(Kernel);
	}
	if (affine_points) {
		FREE_lint(affine_points);
	}
	if (v) {
		FREE_int(v);
	}
	if (v1) {
		FREE_int(v1);
	}
	if (w) {
		FREE_int(w);
	}
	if (f) {
		FREE_int(f);
	}
	if (f2) {
		FREE_int(f2);
	}
	if (F) {
		FREE_int(F);
	}
	if (T) {
		FREE_int(T);
	}
	if (W) {
		FREE_int(W);
	}
	if (f_proj) {
		FREE_int(f_proj);
	}
	if (f_proj2) {
		FREE_int(f_proj2);
	}
}

void boolean_function_domain::init(
		field_theory::finite_field *F2,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::other_geometry::geometry_global Gg;
	algebra::algebra_global Algebra;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "boolean_function_domain::init" << endl;
	}

	if (f_v) {
		cout << "boolean_function_domain::init n=" << n << endl;
		cout << "boolean_function_domain::init q=" << F2->q << endl;
	}
#if 0
	if (ODD(n)) {
		cout << "n must be even" << endl;
		exit(1);
	}
#endif

	boolean_function_domain::Fq = F2;

	if (Fq->q != 2) {
		cout << "boolean_function_domain::init "
				"order of the field must be equal to 2" << endl;
		exit(1);
	}

	boolean_function_domain::n = n;
	n2 = n >> 1;
	Q = 1 << n;
	bent = 1 << (n2);
	near_bent = 1 << ((n + 1) >> 1);
	//NN = 1 << Q;
	NN = NEW_OBJECT(ring_theory::longinteger_object);
	NN->create(2);
	D.power_int(*NN, Q - 1);
	N = Gg.nb_PG_elements(n, 2);
	if (f_v) {
		cout << "boolean_function_domain::init n=" << n << endl;
		cout << "boolean_function_domain::init n2=" << n2 << endl;
		cout << "boolean_function_domain::init Q=" << Q << endl;
		cout << "boolean_function_domain::init bent=" << bent << endl;
		cout << "boolean_function_domain::init near_bent=" << near_bent << endl;
		cout << "boolean_function_domain::init NN=" << *NN << endl;
		cout << "boolean_function_domain::init N=" << N << endl;
	}

	//Fq = NEW_OBJECT(field_theory::finite_field);
	//Fq->finite_field_init(2, false /* f_without_tables */, 0);

	//FQ = NEW_OBJECT(finite_field);
	//FQ->finite_field_init(Q, 0);

	affine_points = NEW_lint(Q);

	v = NEW_int(n + 1);
	v1 = NEW_int(n + 1);
	w = NEW_int(n);
	f = NEW_int(Q);
	f2 = NEW_int(Q);
	F = NEW_int(Q);
	T = NEW_int(Q);
	//W = NEW_int(Q * Q);
	f_proj = NEW_int(N);
	f_proj2 = NEW_int(N);

	int i;
	long int a;

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v1, 1, n, i);
		v1[n] = 1;
		Fq->Projective_space_basic->PG_element_rank_modified_lint(
				v1, 1, n + 1, a);
		affine_points[i] = a;
	}
	if (false) {
		cout << "affine_points" << endl;
		for (i = 0; i < Q; i++) {
			Gg.AG_element_unrank(2, v1, 1, n, i);
			cout << i << " : " << affine_points[i] << " : ";
			Int_vec_print(cout, v1, n);
			cout << endl;
		}
	}

	// setup the Walsh matrix:

	if (f_v) {
		cout << "boolean_function_domain::init "
				"before Gg.Walsh_matrix" << endl;
	}
	if (n <= 10) {
		Algebra.Walsh_matrix(Fq, n, W, verbose_level);
	}
	else {
		cout << "Walsh matrix is too big" << endl;
	}
	if (f_v) {
		cout << "boolean_function_domain::init "
				"after Gg.Walsh_matrix" << endl;
	}


	if (f_v) {
		cout << "boolean_function_domain::init "
				"before setup_polynomial_rings" << endl;
	}
	setup_polynomial_rings(verbose_level);
	if (f_v) {
		cout << "boolean_function_domain::init "
				"after setup_polynomial_rings" << endl;
	}



	if (f_v) {
		cout << "boolean_function_domain::init done" << endl;
	}
}

void boolean_function_domain::setup_polynomial_rings(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_vars;
	int degree;

	if (f_v) {
		cout << "boolean_function_domain::setup_polynomial_rings" << endl;
	}
	nb_vars = n + 1;
		// We need one more variable to capture the constants
		// So, we are really making homogeneous polynomials
		// for projective space PG(n,2) with n+1 variables.

	Poly = NEW_OBJECTS(ring_theory::homogeneous_polynomial_domain, n + 1);

	A_poly = NEW_pint(n + 1);
	B_poly = NEW_pint(n + 1);
	for (degree = 1; degree <= n; degree++) {
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"setting up polynomial ring of degree " << degree << endl;
		}
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"before Poly[degree].init" << endl;
		}
		Poly[degree].init(Fq, nb_vars, degree,
				t_PART,
				0 /* verbose_level */);
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"after Poly[degree].init" << endl;
		}
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"before allocating A_poly" << endl;
		}
		A_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"after allocating A_poly" << endl;
		}
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"before allocating B_poly" << endl;
		}
		B_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
		if (f_v) {
			cout << "boolean_function_domain::setup_polynomial_rings "
					"after allocating B_poly" << endl;
		}
	}

	if (f_v) {
		cout << "boolean_function_domain::setup_polynomial_rings "
				"before Poly[n].affine_evaluation_kernel" << endl;
	}
	Poly[n].affine_evaluation_kernel(
			Kernel, dim_kernel, verbose_level);
	if (f_v) {
		cout << "boolean_function_domain::setup_polynomial_rings "
				"after Poly[n].affine_evaluation_kernel" << endl;
	}

	if (false) {
		cout << "Kernel of evaluation map:" << endl;
		Int_matrix_print(Kernel, dim_kernel, 2);
	}

	if (f_v) {
		cout << "boolean_function_domain::setup_polynomial_rings done" << endl;
	}
}

void boolean_function_domain::compute_polynomial_representation(
		int *func, int *coeff, int verbose_level)
// called from
// boolean_function_classify::search_for_bent_functions
// action_on_forms::orbits_on_functions
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	geometry::other_geometry::geometry_global Gg;
	int s, i, u, v, a, b, c, h, idx;
	int N;
	int degree = n + 1;
	int *vec;
	int *mon;

	if (f_v) {
		cout << "boolean_function_domain::compute_polynomial_representation" << endl;
	}
	N = 1 << n;
	if (f_v) {
		cout << "func=" << endl;
		for (s = 0; s < N; s++) {
			cout << s << " : " << func[s] << endl;
		}
		cout << "Poly[n].nb_monomials=" << Poly[n].get_nb_monomials() << endl;
	}
	vec = NEW_int(n);
	mon = NEW_int(degree);
	Int_vec_zero(coeff, Poly[n].get_nb_monomials());
	if (f_v) {
		cout << "boolean_function_domain::compute_polynomial_representation "
				"looping over all values, N=" << N << endl;
	}
	for (s = 0; s < N; s++) {

		// we are making the complement of the function,
		// so we are skipping all entries which are zero!

		if (f_v) {
			cout << "boolean_function_domain::compute_polynomial_representation "
					"s=" << s << " / " << N << endl;
		}

		if (func[s]) {
			continue;
		}

		if (f_vv) {
			cout << "the function value at s=" << s << " is " << func[s] << endl;
			cout << "func=" << endl;
			for (h = 0; h < N; h++) {
				cout << h << " : " << func[h] << endl;
			}
		}
		Gg.AG_element_unrank(2, vec, 1, n, s);


		// create the polynomial
		// \prod_{i=0}^{n-1} (x_i+(vec[i]+1)*x_n)
		// which is one exactly if x_i = vec[i] for i=0..n-1 and x_n = 1.
		// and zero otherwise.
		// So this polynomial agrees with the boolean function
		// on the affine space x_n = 1.

		for (i = 0; i < n; i++) {


			if (f_vv) {
				cout << "s=" << s << " i=" << i << endl;
			}

			// create the polynomial (x_i+(vec[i]+1)*x_n)
			// note that x_n stands for the constants
			// because we are in affine space
			Int_vec_zero(A_poly[1], Poly[1].get_nb_monomials());
			A_poly[1][n] = Fq->add(1, vec[i]);
			A_poly[1][i] = 1;

			if (f_v) {
				cout << "created the polynomial ";
				Poly[1].print_equation(cout, A_poly[1]);
				cout << endl;
			}


			if (i == 0) {
				Int_vec_copy(A_poly[1], B_poly[1], Poly[1].get_nb_monomials());
			}
			else {
				// B_poly[i + 1] = A_poly[1] * B_poly[i]
				Int_vec_zero(B_poly[i + 1], Poly[i + 1].get_nb_monomials());
				for (u = 0; u < Poly[1].get_nb_monomials(); u++) {
					a = A_poly[1][u];
					if (a == 0) {
						continue;
					}
					for (v = 0; v < Poly[i].get_nb_monomials(); v++) {

						// multiply  the term u in A_poly[1] with the term v in B_poly[i].
						// To do so, the exponent vectors of the terms are added.
						//
						// the coefficient is a * b and will be added to the existing coefficient:

						b = B_poly[i][v];
						if (b == 0) {
							continue;
						}
						c = Fq->mult(a, b);
						Int_vec_zero(mon, n + 1);
						for (h = 0; h < n + 1; h++) {
							mon[h] = Poly[1].get_monomial(u, h) +
									Poly[i].get_monomial(v, h);
						}
						idx = Poly[i + 1].index_of_monomial(mon);
						B_poly[i + 1][idx] = Fq->add(B_poly[i + 1][idx], c);
					} // next v
				} // next u
			} // else
		} // next i
		if (f_v) {
			cout << "s=" << s << " / " << N << " : ";
			Poly[n].print_equation(cout, B_poly[n]);
			cout << endl;
		}
		for (h = 0; h < Poly[n].get_nb_monomials(); h++) {
			coeff[h] = Fq->add(coeff[h], B_poly[n][h]);
		}
	} // next s
	if (f_v) {
		cout << "boolean_function_domain::compute_polynomial_representation "
				"looping over all values done" << endl;
	}

	if (f_v) {
		cout << "preliminary result : ";
		Poly[n].print_equation(cout, coeff);
		cout << endl;

		int *f;
		int f_error = false;

		f = NEW_int(Q);
		evaluate(coeff, f);

		for (h = 0; h < Q; h++) {
			cout << h << " : " << func[h] << " : " << f[h];
#if 0
			if (func[h] == f[h]) {
				cout << "error";
				f_error = true;
			}
#endif
			cout << endl;
		}
		if (f_error) {
			cout << "an error has occurred" << endl;
			exit(1);
		}
		FREE_int(f);
	}

	Int_vec_zero(mon, n + 1);
	mon[n] = n;
	idx = Poly[n].index_of_monomial(mon);
	coeff[idx] = Fq->add(coeff[idx], 1);

	if (f_v) {
		cout << "result : ";
		Poly[n].print_equation(cout, coeff);
		cout << endl;


		int *f;
		int f_error = false;

		f = NEW_int(Q);
		evaluate(coeff, f);

		for (h = 0; h < Q; h++) {
			cout << h << " : " << func[h] << " : " << f[h];
			if (func[h] != f[h]) {
				cout << "error";
				f_error = true;
			}
			cout << endl;
		}
		if (f_error) {
			cout << "an error has occurred" << endl;
			exit(1);
		}
		FREE_int(f);


	}

	FREE_int(vec);
	FREE_int(mon);

	if (f_v) {
		cout << "boolean_function_domain::compute_polynomial_representation done" << endl;
	}
}

void boolean_function_domain::evaluate_projectively(
		int *coeff, int *f)
{
	int i;

	for (i = 0; i < N; i++) {
		f[i] = Poly[n].evaluate_at_a_point_by_rank(coeff, i);
	}

}

void boolean_function_domain::evaluate(
		int *coeff, int *f)
{
	int i;
	geometry::other_geometry::geometry_global Gg;

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v1, 1, n, i);
		v1[n] = 1;
		f[i] = Poly[n].evaluate_at_a_point(coeff, v1);
	}

}

void boolean_function_domain::raise(
		int *in, int *out)
{
	int i;

	for (i = 0; i < Q; i++) {
		if (in[i]) {
			out[i] = -1;
		}
		else {
			out[i] = 1;
		}
	}
}

void boolean_function_domain::apply_Walsh_transform(
		int *in, int *out)
{
	int i, j;

	Int_vec_zero(out, Q);
	for (i = 0; i < Q; i++) {
		for (j = 0; j < Q; j++) {
			out[i] += W[i * Q + j] * in[j];
		}
	}
}

int boolean_function_domain::is_bent(
		int *T)
{
	int i;

	for (i = 0; i < Q; i++) {
		if (ABS(T[i]) != bent) {
			//cout << "ABS(T[i]) != bent, T[i] = " << T[i] << " bent=" << bent << endl;
			break;
		}
	}
	if (i == Q) {
		return true;
	}
	else {
		return false;
	}
}

int boolean_function_domain::is_near_bent(
		int *T)
{
	int i;

	for (i = 0; i < Q; i++) {
		if (T[i] == 0) {
			continue;
		}
		if (ABS(T[i]) != near_bent) {
			//cout << "ABS(T[i]) != near_bent, T[i] = " << T[i] << " near_bent=" << near_bent << endl;
			break;
		}
	}
	if (i == Q) {
		return true;
	}
	else {
		return false;
	}
}



}}}



