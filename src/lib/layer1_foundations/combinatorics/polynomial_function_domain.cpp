/*
 * polynomial_function_domain.cpp
 *
 *  Created on: Oct 22, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


polynomial_function_domain::polynomial_function_domain()
{

	Fq = NULL;
	q = 0;

	n = Q = 0;
	max_degree = 0;

	Poly = NULL;
	A_poly = NULL;
	B_poly = NULL;
	C_poly = NULL;
	Kernel = NULL;

	dim_kernel = 0;
	affine_points = NULL;
	v = v1 = w = f = f2 = NULL;
}

polynomial_function_domain::~polynomial_function_domain()
{
	int degree;

	if (Poly) {
		FREE_OBJECTS(Poly);
	}
	if (A_poly) {
		for (degree = 1; degree <= max_degree; degree++) {
			FREE_int(A_poly[degree]);
			FREE_int(B_poly[degree]);
			FREE_int(C_poly[degree]);
		}
		FREE_pint(A_poly);
		FREE_pint(B_poly);
		FREE_pint(C_poly);
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
}

void polynomial_function_domain::init(field_theory::finite_field *Fq, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;
	//algebra::algebra_global Algebra;
	//ring_theory::longinteger_domain D;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "polynomial_function_domain::init" << endl;
	}


	polynomial_function_domain::Fq = Fq;
	q = Fq->q;

	polynomial_function_domain::n = n;
	max_degree = n * (q - 1);

	if (f_v) {
		cout << "polynomial_function_domain::init n=" << n << endl;
	}
	polynomial_function_domain::n = n;
	Q = NT.i_power_j_lint(q, n);

	if (f_v) {
		cout << "polynomial_function_domain::init n=" << n << endl;
		cout << "polynomial_function_domain::init q=" << q << endl;
		cout << "polynomial_function_domain::init max_degree=" << max_degree << endl;
		cout << "polynomial_function_domain::init Q=" << Q << endl;
	}

	affine_points = NEW_lint(Q);

	v = NEW_int(n);
	v1 = NEW_int(n);
	w = NEW_int(n);
	f = NEW_int(Q);
	f2 = NEW_int(Q);

	int i;
	long int a;

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v1, 1, n, i);
		v1[n] = 1;
		Fq->PG_element_rank_modified_lint(v1, 1, n + 1, a);
		affine_points[i] = a;
	}
	if (FALSE) {
		cout << "affine_points" << endl;
		for (i = 0; i < Q; i++) {
			Gg.AG_element_unrank(2, v1, 1, n, i);
			cout << i << " : " << affine_points[i] << " : ";
			Int_vec_print(cout, v1, n);
			cout << endl;
		}
	}


	if (f_v) {
		cout << "polynomial_function_domain::init before setup_polynomial_rings" << endl;
	}
	setup_polynomial_rings(verbose_level);
	if (f_v) {
		cout << "polynomial_function_domain::init after setup_polynomial_rings" << endl;
	}



	if (f_v) {
		cout << "polynomial_function_domain::init done" << endl;
	}
}

void polynomial_function_domain::setup_polynomial_rings(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_vars;
	int degree;

	if (f_v) {
		cout << "polynomial_function_domain::setup_polynomial_rings" << endl;
	}
	nb_vars = n + 1;
		// We need one more variable to capture the constants
		// So, we are really making homogeneous polynomials
		// for projective space PG(n,2) with n+1 variables.

	Poly = NEW_OBJECTS(ring_theory::homogeneous_polynomial_domain, max_degree + 1);

	A_poly = NEW_pint(max_degree + 1);
	B_poly = NEW_pint(max_degree + 1);
	C_poly = NEW_pint(max_degree + 1);
	for (degree = 1; degree <= max_degree; degree++) {
		if (f_v) {
			cout << "polynomial_function_domain::setup_polynomial_rings "
					"setting up polynomial ring of degree " << degree << endl;
		}
		Poly[degree].init(Fq, nb_vars, degree,
				t_PART,
				0 /* verbose_level */);
		A_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
		B_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
		C_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
	}

	if (f_v) {
		cout << "polynomial_function_domain::setup_polynomial_rings "
				"before Poly[max_degree].affine_evaluation_kernel" << endl;
	}
	Poly[max_degree].affine_evaluation_kernel(
			Kernel, dim_kernel, verbose_level);
	if (f_v) {
		cout << "polynomial_function_domain::setup_polynomial_rings "
				"after Poly[max_degree].affine_evaluation_kernel" << endl;
		cout << "polynomial_function_domain::setup_polynomial_rings "
				"dim_kernel = " << dim_kernel << endl;
	}

	if (FALSE) {
		cout << "Kernel of evaluation map:" << endl;
		Int_matrix_print(Kernel, dim_kernel, 2);
	}

	if (f_v) {
		cout << "polynomial_function_domain::setup_polynomial_rings done" << endl;
	}
}

void polynomial_function_domain::compute_polynomial_representation(
		int *func, int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	geometry::geometry_global Gg;
	int s, i, h, idx; //idx_last
	int *vec;
	int *mon;
	int m1; // minus one

	if (f_v) {
		cout << "polynomial_function_domain::compute_polynomial_representation" << endl;
	}
	if (f_v) {
		cout << "func=" << endl;
		for (s = 0; s < Q; s++) {
			cout << s << " : " << func[s] << endl;
		}
		cout << "Poly[n].nb_monomials=" << Poly[n].get_nb_monomials() << endl;
	}
	m1 = Fq->negate(1); // minus one

	vec = NEW_int(n);
	mon = NEW_int(n);
	Int_vec_zero(coeff, Poly[max_degree].get_nb_monomials());

	Int_vec_zero(mon, n + 1);
	mon[n] = max_degree;
	//idx_last = Poly[max_degree].index_of_monomial(mon);

	if (f_v) {
		cout << "polynomial_function_domain::compute_polynomial_representation "
				"looping over all values, Q=" << Q << endl;
	}
	for (s = 0; s < Q; s++) {


		if (f_v) {
			cout << "polynomial_function_domain::compute_polynomial_representation "
					"s=" << s << " / " << Q << endl;
		}

		if (func[s] == 0) {
			continue;
		}

		Gg.AG_element_unrank(q, vec, 1, n, s);

		if (f_v) {
			cout << "the function value at s=" << s << " is " << func[s] << endl;
			cout << "vec=" << endl;
			Int_vec_print(cout, vec, n);
			cout << endl;
		}


		// create the polynomial
		// \prod_{i=0}^{n-1} (1-(x_i-vec[i]*x_n)^{q-1})
		// \prod_{i=0}^{n-1} (x_n^{(q-1)}-(x_i-vec[i]*x_n)^{q-1})
		// which is one exactly if x_i = vec[i] for all i=0..n-1 and x_n = 1.
		// and zero otherwise.
		// So this polynomial agrees with the q-ary function
		// on the affine space x_n = 1.

		for (i = 0; i < n; i++) {


			if (f_vv) {
				cout << "s=" << s << " i=" << i << endl;
			}


			// create x_i-vec[i]*x_n
			Int_vec_zero(A_poly[1], Poly[1].get_nb_monomials());


			Int_vec_zero(mon, n + 1);
			mon[i] = 1;
			idx = Poly[1].index_of_monomial(mon);
			A_poly[1][idx] = 1;

			Int_vec_zero(mon, n + 1);
			mon[n] = 1;
			idx = Poly[1].index_of_monomial(mon);
			A_poly[1][idx] = Fq->negate(vec[i]);


			if (f_v) {
				cout << "created the polynomial x_i-vec[i]*x_n = ";
				Poly[1].print_equation(cout, A_poly[1]);
				cout << endl;
			}

			Int_vec_copy(A_poly[1], B_poly[1], Poly[1].get_nb_monomials());

			int j;

			for (j = 2; j <= q - 1; j++) {
				multiply_i_times_j(
							1, j - 1,
							A_poly[1], B_poly[j - 1], B_poly[j],
							0 /*verbose_level*/);
			}
			if (f_v) {
				cout << "after raising to the power q-1: ";
				Poly[q - 1].print_equation(cout, B_poly[q - 1]);
				cout << endl;
			}


			// multiply B_poly[q - 1] by -1:

			for (h = 0; h < Poly[(q - 1)].get_nb_monomials(); h++) {
				B_poly[q - 1][h] = Fq->mult(m1, B_poly[q - 1][h]);
			}


			// need to add x_n^{(q - 1)}:
			Int_vec_zero(mon, n + 1);
			mon[n] = (q - 1);
			idx = Poly[q - 1].index_of_monomial(mon);

			B_poly[q - 1][idx] = Fq->add(B_poly[q - 1][idx], 1);


			if (i == 0) {
				Int_vec_copy(B_poly[q - 1], C_poly[q - 1], Poly[q - 1].get_nb_monomials());
			}
			else {
				// C_poly[(i + 1) * (q - 1)] = B_poly[q - 1] * C_poly[i * (q - 1)]

				multiply_i_times_j(
							q - 1, i * (q - 1),
							B_poly[q - 1], C_poly[i * (q - 1)], C_poly[(i + 1) * (q - 1)],
							0 /*verbose_level*/);
			}


#if 0
			// multiply C_poly[(i + 1) * (q - 1)] by -1:

			for (h = 0; h < Poly[(i + 1) * (q - 1)].get_nb_monomials(); h++) {
				C_poly[(i + 1) * (q - 1)][h] = Fq->mult(m1, C_poly[(i + 1) * (q - 1)][h]);
			}

			// need to add x_n^{(i + 1) * (q - 1)}:
			Int_vec_zero(mon, n + 1);
			mon[n] = (i + 1) * (q - 1);
			idx = Poly[(i + 1) * (q - 1)].index_of_monomial(mon);

			C_poly[(i + 1) * (q - 1)][idx] = Fq->add(C_poly[(i + 1) * (q - 1)][idx], 1);

#endif

			if (f_v) {
				cout << "s=" << s << " / " << Q << " : ";
				Poly[(i + 1) * (q - 1)].print_equation(cout, C_poly[(i + 1) * (q - 1)]);
				cout << endl;
			}


		} // next i
		if (f_v) {
			cout << "s=" << s << " / " << Q << " : ";
			Poly[max_degree].print_equation(cout, C_poly[max_degree]);
			cout << endl;
		}

		// need to multiply by func[s]:

		for (h = 0; h < Poly[max_degree].get_nb_monomials(); h++) {
			C_poly[max_degree][h] = Fq->mult(func[s], C_poly[max_degree][h]);
		}

		// add on top of what we already have:

		for (h = 0; h < Poly[max_degree].get_nb_monomials(); h++) {
			coeff[h] = Fq->add(coeff[h], C_poly[max_degree][h]);
		}


	} // next s

	if (f_v) {
		cout << "polynomial_function_domain::compute_polynomial_representation "
				"looping over all values done" << endl;
	}

	if (f_v) {
		cout << "preliminary result : ";
		Poly[max_degree].print_equation(cout, coeff);
		cout << endl;

		int *f;
		int f_error = FALSE;

		f = NEW_int(Q);
		evaluate(coeff, f);

		for (h = 0; h < Q; h++) {
			cout << h << " : " << func[h] << " : " << f[h];
#if 0
			if (func[h] != f[h]) {
				cout << "error";
				f_error = TRUE;
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

#if 0
	Int_vec_zero(mon, n + 1);
	mon[n] = n;
	idx = Poly[max_degree].index_of_monomial(mon);
	coeff[idx] = Fq->add(coeff[idx], 1);

	if (f_v) {
		cout << "result : ";
		Poly[n].print_equation(cout, coeff);
		cout << endl;


		int *f;
		int f_error = FALSE;

		f = NEW_int(Q);
		evaluate(coeff, f);

		for (h = 0; h < Q; h++) {
			cout << h << " : " << func[h] << " : " << f[h];
			if (func[h] != f[h]) {
				cout << "error";
				f_error = TRUE;
			}
			cout << endl;
		}
		if (f_error) {
			cout << "an error has occurred" << endl;
			exit(1);
		}
		FREE_int(f);


	}
#endif

	FREE_int(vec);
	FREE_int(mon);

	if (f_v) {
		cout << "polynomial_function_domain::compute_polynomial_representation done" << endl;
	}
}

void polynomial_function_domain::evaluate_projectively(int *coeff, int *f)
{
	int i;

	for (i = 0; i < Q; i++) {
		f[i] = Poly[n].evaluate_at_a_point_by_rank(coeff, i);
	}

}

void polynomial_function_domain::evaluate(int *coeff, int *f)
{
	int i;
	geometry::geometry_global Gg;

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(q, v1, 1, n, i);
		v1[n] = 1;
		f[i] = Poly[max_degree].evaluate_at_a_point(coeff, v1);
	}

}

void polynomial_function_domain::raise(int *in, int *out)
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

void polynomial_function_domain::multiply_i_times_j(
		int i, int j,
		int *A_eqn, int *B_eqn, int *C_eqn,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ipj, a, b, c, idx, u, v, w;
	int *M;

	if (f_v) {
		cout << "polynomial_function_domain::multiply_i_times_j" << endl;
	}

	M = NEW_int(n + 1);
	ipj = i + j;

	Int_vec_zero(C_eqn, Poly[ipj].get_nb_monomials());

	for (u = 0; u < Poly[i].get_nb_monomials(); u++) {
		a = A_eqn[u];
		if (a == 0) {
			continue;
		}
		for (v = 0; v < Poly[j].get_nb_monomials(); v++) {
			b = B_eqn[v];
			if (b == 0) {
				continue;
			}
			c = Fq->mult(a, b);
			for (w = 0; w <= n; w++) {
				M[w] = Poly[i].get_monomial(u, w)
						+ Poly[j].get_monomial(v, w);
			}
			idx = Poly[ipj].index_of_monomial(M);
			C_eqn[idx] = Fq->add(C_eqn[idx], c);
		}
	}
	FREE_int(M);

	if (f_v) {
		cout << "polynomial_function_domain::multiply_i_times_j done" << endl;
	}
}


}}}




