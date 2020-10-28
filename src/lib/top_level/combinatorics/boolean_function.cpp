/*
 * boolean_function.cpp
 *
 *  Created on: Nov 06, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



boolean_function::boolean_function()
{
	n = n2 = Q = Q2 = N = 0;
	Fq = NULL;
	FQ = NULL;
	Poly = NULL;
	A_poly = NULL;
	B_poly = NULL;
	Kernel = NULL;
	dim_kernel = 0;
	A = NULL;
	nice_gens = NULL;
	AonHPD = NULL;
	SG = NULL;
	// go;
	affine_points = NULL;
	A_affine = NULL;
	v = v1 = w = f = f2 = F = T = W = f_proj = f_proj2 = NULL;
}

boolean_function::~boolean_function()
{
	int degree;

	if (Fq) {
		FREE_OBJECT(Fq);
	}
	if (FQ) {
		FREE_OBJECT(FQ);
	}
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
	if (A) {
		FREE_OBJECT(A);
	}
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
	if (SG) {
		FREE_OBJECT(SG);
	}
	if (affine_points) {
		FREE_lint(affine_points);
	}
	if (A_affine) {
		FREE_OBJECT(A_affine);
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

void boolean_function::init(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry_global Gg;
	longinteger_domain D;

	if (f_v) {
		cout << "boolean_function::init" << endl;
	}

	if (f_v) {
		cout << "do_it n=" << n << endl;
	}
#if 0
	if (ODD(n)) {
		cout << "n must be even" << endl;
		exit(1);
	}
#endif
	boolean_function::n = n;
	n2 = n >> 1;
	Q = 1 << n;
	Q2 = 1 << n2;
	//NN = 1 << Q;
	NN.create(2, __FILE__, __LINE__);
	D.power_int(NN, Q - 1);
	N = Gg.nb_PG_elements(n, 2);
	if (f_v) {
		cout << "do_it n=" << n << endl;
		cout << "do_it n2=" << n2 << endl;
		cout << "do_it Q=" << Q << endl;
		cout << "do_it Q2=" << Q2 << endl;
		cout << "do_it NN=" << NN << endl;
		cout << "do_it N=" << N << endl;
	}

	Fq = NEW_OBJECT(finite_field);
	Fq->init(2, 0);

	FQ = NEW_OBJECT(finite_field);
	FQ->init(Q, 0);

	affine_points = NEW_lint(Q);

	v = NEW_int(n);
	v1 = NEW_int(n);
	w = NEW_int(n);
	f = NEW_int(Q);
	f2 = NEW_int(Q);
	F = NEW_int(Q);
	T = NEW_int(Q);
	W = NEW_int(Q * Q);
	f_proj = NEW_int(N);
	f_proj2 = NEW_int(N);

	int i, j, a;

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v1, 1, n, i);
		v1[n] = 1;
		Fq->PG_element_rank_modified(v1, 1, n + 1, a);
		affine_points[i] = a;
	}
	if (f_v) {
		cout << "affine_points" << endl;
		for (i = 0; i < Q; i++) {
			Gg.AG_element_unrank(2, v1, 1, n, i);
			cout << i << " : " << affine_points[i] << " : ";
			int_vec_print(cout, v1, n);
			cout << endl;
		}
	}

	// setup the Wash matrix:
	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v, 1, n, i);
		for (j = 0; j < Q; j++) {
			Gg.AG_element_unrank(2, w, 1, n, j);
			a = Fq->dot_product(n, v, w);
			if (a) {
				W[i * Q + j] = -1;
			}
			else {
				W[i * Q + j] = 1;
			}
		}
	}

	setup_polynomial_rings(verbose_level);

	init_group(verbose_level);


	if (f_v) {
		cout << "boolean_function::init done" << endl;
	}
}

void boolean_function::init_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "boolean_function::init_group" << endl;
	}

	int degree = n + 1;

	A = NEW_OBJECT(action);

	if (f_v) {
		cout << "boolean_function::init_group "
				"before init_projective_group" << endl;
	}
	A->init_projective_group(degree, Fq,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level);

	AonHPD = NEW_OBJECT(action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "boolean_function::init_group "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(A, &Poly[n], verbose_level);
	if (f_v) {
		cout << "boolean_function::init_group "
				"after AonHPD->init" << endl;
	}


	SG = NEW_OBJECT(strong_generators);

	matrix_group *Mtx;

	Mtx = A->get_matrix_group();

	if (f_v) {
		cout << "boolean_function::init_group "
				"before generators_for_parabolic_subgroup" << endl;
	}
	SG->generators_for_parabolic_subgroup(A,
			Mtx, degree - 1, verbose_level);
	if (f_v) {
		cout << "boolean_function::init_group "
				"after generators_for_parabolic_subgroup" << endl;
	}

	SG->print_generators_tex(cout);

	SG->group_order(go);
	if (f_v) {
		cout << "boolean_function::init_group "
				"go=" << go << endl;
	}

	if (f_v) {
		cout << "boolean_function::init_group "
				"before A->restricted_action" << endl;
	}
	A_affine = A->restricted_action(affine_points, Q,
			verbose_level);
	if (f_v) {
		cout << "boolean_function::init_group "
				"after A->restricted_action" << endl;
	}

	if (f_v) {
		cout << "Generators in the induced action:" << endl;
		SG->print_with_given_action(
			cout, A_affine);
	}


#if 0
	SG->init(A);
	if (f_v) {
		cout << "boolean_function::init_group "
				"before init_transposed_group" << endl;
	}
	SG->init_transposed_group(SGt, verbose_level);
	if (f_v) {
		cout << "boolean_function::init_group "
				"after init_transposed_group" << endl;
	}
#endif

	if (f_v) {
		cout << "boolean_function::init_group done" << endl;
	}
}

void boolean_function::setup_polynomial_rings(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_vars;
	int degree;

	if (f_v) {
		cout << "boolean_function::setup_polynomial_rings" << endl;
	}
	nb_vars = n + 1;
		// We need one more variable to capture the constants
		// So, we are really making homogeneous polynomials
		// for projective space PG(n,2) with n+1 variables.

	Poly = NEW_OBJECTS(homogeneous_polynomial_domain, n + 1);

	A_poly = NEW_pint(n + 1);
	B_poly = NEW_pint(n + 1);
	for (degree = 1; degree <= n; degree++) {
		Poly[degree].init(Fq, nb_vars, degree,
				FALSE /* f_init_incidence_structure */,
				t_PART,
				0 /* verbose_level */);
		A_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
		B_poly[degree] = NEW_int(Poly[degree].get_nb_monomials());
	}

	Poly[n].affine_evaluation_kernel(
			Kernel, dim_kernel, verbose_level);

	if (f_v) {
		cout << "Kernel of evaluation map:" << endl;
		int_matrix_print(Kernel, dim_kernel, 2);
	}

	if (f_v) {
		cout << "boolean_function::setup_polynomial_rings done" << endl;
	}
}

void boolean_function::compute_polynomial_representation(
		int *func, int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	geometry_global Gg;
	int s, i, u, v, a, b, c, h, idx;
	int N;
	int degree = n + 1;
	int *vec;
	int *mon;

	if (f_v) {
		cout << "boolean_function::compute_polynomial_representation" << endl;
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
	int_vec_zero(coeff, Poly[n].get_nb_monomials());
	for (s = 0; s < N; s++) {

		// we are making the complement of the function,
		// so we are skipping all entries which are zero!

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
		// which is one exacly if x_i = vec[i] for i=0..n-1 and x_n = 1.
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
			int_vec_zero(A_poly[1], Poly[1].get_nb_monomials());
			A_poly[1][n] = Fq->add(1, vec[i]);
			A_poly[1][i] = 1;

			if (f_v) {
				cout << "created the polynomial ";
				Poly[1].print_equation(cout, A_poly[1]);
				cout << endl;
			}


			if (i == 0) {
				int_vec_copy(A_poly[1], B_poly[1], Poly[1].get_nb_monomials());
			}
			else {
				// B_poly[i + 1] = A_poly[1] * B_poly[i]
				int_vec_zero(B_poly[i + 1], Poly[i + 1].get_nb_monomials());
				for (u = 0; u < Poly[1].get_nb_monomials(); u++) {
					a = A_poly[1][u];
					if (a == 0) {
						continue;
					}
					for (v = 0; v < Poly[i].get_nb_monomials(); v++) {
						b = B_poly[i][v];
						if (b == 0) {
							continue;
						}
						c = Fq->mult(a, b);
						int_vec_zero(mon, n + 1);
						for (h = 0; h <= n + 1; h++) {
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
		cout << "preliminary result : ";
		Poly[n].print_equation(cout, coeff);
		cout << endl;

		int *f;
		int f_error = FALSE;

		f = NEW_int(Q);
		evaluate(coeff, f);

		for (h = 0; h < Q; h++) {
			cout << h << " : " << func[h] << " : " << f[h];
			if (func[h] == f[h]) {
				cout << "error";
				f_error = TRUE;
			}
			cout << endl;
		}
		if (f_error) {
			cout << "an error has occured" << endl;
			exit(1);
		}
		FREE_int(f);
	}

	int_vec_zero(mon, n + 1);
	mon[n] = n;
	idx = Poly[n].index_of_monomial(mon);
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
			cout << "an error has occured" << endl;
			exit(1);
		}
		FREE_int(f);


	}

	FREE_int(vec);
	FREE_int(mon);

	if (f_v) {
		cout << "boolean_function::compute_polynomial_representation done" << endl;
	}
}

void boolean_function::evaluate_projectively(int *coeff, int *f)
{
	int i;

	for (i = 0; i < N; i++) {
		f[i] = Poly[n].evaluate_at_a_point_by_rank(coeff, i);
	}

}

void boolean_function::evaluate(int *coeff, int *f)
{
	int i;
	geometry_global Gg;

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v1, 1, n, i);
		v1[n] = 1;
		f[i] = Poly[n].evaluate_at_a_point(coeff, v1);
	}

}

void boolean_function::raise(int *in, int *out)
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

void boolean_function::apply_Walsh_transform(int *in, int *out)
{
	int i, j;

	int_vec_zero(out, Q);
	for (i = 0; i < Q; i++) {
		for (j = 0; j < Q; j++) {
			out[i] += W[i * Q + j] * in[j];
		}
	}
}

int boolean_function::is_bent(int *T)
{
	int i;

	for (i = 0; i < Q; i++) {
		if (ABS(T[i]) != Q2) {
			break;
		}
	}
	if (i == Q) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void boolean_function::search_for_bent_functions(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *poly;
	int i, j;
	longinteger_object a;
	int nb_sol = 0;
	int nb_orbits = 0;
	uint32_t h;
	geometry_global Gg;
	longinteger_domain D;
	data_structures_global Data;
	vector<int> orbit_first;
	vector<int> orbit_length;

	vector<vector<int> > Bent_function_table;
	vector<vector<int> > Equation_table;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.


	if (f_v) {
		cout << "bent_function_classify::search_for_bent_functions" << endl;
	}



	poly = NEW_int(Poly[n].get_nb_monomials());

	a.create(0, __FILE__, __LINE__);
	while (D.is_less_than(a, NN)) {

		Gg.AG_element_unrank_longinteger(2, f, 1, Q, a);
		//Gg.AG_element_unrank(2, f, 1, Q, a);
		cout << a << " / " << NN << " : ";
		int_vec_print(cout, f, Q);
		//cout << endl;

		raise(f, F);

		apply_Walsh_transform(F, T);

		cout << " : ";
		int_vec_print(cout, T, Q);

		if (is_bent(T)) {
			cout << " is bent " << nb_sol;
			nb_sol++;

			h = Data.int_vec_hash(f, Q);

		    map<uint32_t, int>::iterator itr, itr1, itr2;
		    int pos, f_found;

		    itr1 = Hashing.lower_bound(h);
		    itr2 = Hashing.upper_bound(h);
		    f_found = FALSE;
		    for (itr = itr1; itr != itr2; ++itr) {
		        pos = itr->second;
		        for (j = 0; j < Q; j++) {
		        	if (f[j] != Bent_function_table[pos][j]) {
		        		break;
		        	}
		        }
		        if (j == Q) {
		        	f_found = TRUE;
		        	break;
		        }
		    }


		    if (!f_found) {

				cout << " NEW orbit " << nb_orbits << endl;

				compute_polynomial_representation(f, poly, 0 /*verbose_level*/);
				cout << " : ";
				Poly[n].print_equation(cout, poly);
				cout << " : ";
				//evaluate_projectively(poly, f_proj);
				evaluate(poly, f_proj);
				int_vec_print(cout, f_proj, Q);
				cout << endl;

				orbit_of_equations *Orb;

				Orb = NEW_OBJECT(orbit_of_equations);

				Orb->f_has_print_function = TRUE;
				Orb->print_function = boolean_function_print_function;
				Orb->print_function_data = this;

				Orb->f_has_reduction = TRUE;
				Orb->reduction_function = boolean_function_reduction_function;
				Orb->reduction_function_data = this;

				cout << "orbit " << nb_orbits << ", computing orbit of bent function:" << endl;
				Orb->init(A, Fq,
					AonHPD,
					SG /* A->Strong_gens*/, poly,
					0 /*verbose_level*/);
				cout << "found an orbit of length " << Orb->used_length << endl;

				strong_generators *Stab_gens;

				cout << "orbit " << nb_orbits << ", computing stabilizer:" << endl;
				Stab_gens = Orb->stabilizer_orbit_rep(
						go, verbose_level);
				Stab_gens->print_generators_tex(cout);

				orbit_first.push_back(Bent_function_table.size());
				orbit_length.push_back(Orb->used_length);

				int *coeff;

				for (i = 0; i < Orb->used_length; i++) {
					coeff = Orb->Equations[i] + 1;
					evaluate(coeff, f_proj);
					vector<int> v;
					for (j = 0; j < Q; j++) {
						v.push_back(f_proj[j]);
					}
					vector<int> w;
					for (j = 0; j < Poly[n].get_nb_monomials(); j++) {
						w.push_back(Orb->Equations[i][1 + j]);
					}

					h = Data.int_vec_hash(f_proj, Q);
					Hashing.insert(pair<uint32_t, int>(h, Bent_function_table.size()));

					Bent_function_table.push_back(v);
					Equation_table.push_back(w);
				}

				//int idx = 3;
				int idx = 0;

				if (n == 4) {
					if (nb_orbits == 0) {
						idx = 12;
					}
					else if (nb_orbits == 1) {
						idx = 180;
					}
				}

				if (Orb->used_length > idx) {
					cout << "orbit " << nb_orbits << ", computing stabilizer of element " << idx << endl;

					coeff = Orb->Equations[idx] + 1;
					evaluate(coeff, f_proj);
					cout << "orbit " << nb_orbits << ", function: ";
					int_vec_print(cout, f_proj, Q);
					cout << endl;
					cout << "orbit " << nb_orbits << ", equation: ";
					int_vec_print(cout, coeff, Poly[n].get_nb_monomials());
					cout << endl;

					strong_generators *Stab_gens_clean;


					Stab_gens_clean = Orb->stabilizer_any_point(
							go, idx,
							verbose_level);
					Stab_gens_clean->print_generators_tex(cout);
					cout << "orbit " << nb_orbits << ", induced action:" << endl;
					Stab_gens_clean->print_with_given_action(
							cout, A_affine);

					FREE_OBJECT(Stab_gens_clean);
				}

				FREE_OBJECT(Stab_gens);
				FREE_OBJECT(Orb);

				nb_orbits++;

		    }
		    else {
		    	cout << "The bent function has been found earlier already" << endl;
		    }
		}
		else {
			cout << endl;
		}
		a.increment();
		cout << "after increment: a=" << a << endl;
	}
	cout << "We found " << nb_sol << " bent functions" << endl;
	cout << "We have " << Bent_function_table.size() << " bent functions in the table" << endl;
	cout << "They fall into " << orbit_first.size() << " orbits:" << endl;

	int fst, len, t;

	for (h = 0; h < orbit_first.size(); h++) {
		fst = orbit_first[h];
		len = orbit_length[h];
		cout << "Orbit " << h << " / " << orbit_first.size() << " has length " << len << ":" << endl;
		for (t = 0; t < len; t++) {
			i = fst + t;
			cout << i << " : " << t << " / " << len << " : ";
			for (j = 0; j < Q; j++) {
				f[j] = Bent_function_table[i][j];
			}
			for (j = 0; j < Poly[n].get_nb_monomials(); j++) {
				poly[j] = Equation_table[i][j];
			}

			int_vec_copy(f, f2, Q);
			Gg.AG_element_rank_longinteger(2, f2, 1, Q, a);

			int_vec_print(cout, f, Q);
			cout << " : " << a << " : ";
			int_vec_print(cout, poly, Poly[n].get_nb_monomials());
			cout << " : ";
			Poly[n].print_equation(cout, poly);
			cout << endl;
		}
	}
#if 0
	for (i = 0; i < Bent_function_table.size(); i++) {
		cout << i << " : ";
		for (j = 0; j < Q; j++) {
			f[j] = Bent_function_table[i][j];
		}
		for (j = 0; j < Poly[n].get_nb_monomials(); j++) {
			poly[j] = Equation_table[i][j];
		}
		int_vec_print(cout, f, Q);
		cout << " : ";
		int_vec_print(cout, poly, Poly[n].get_nb_monomials());
		cout << " : ";
		Poly[n].print_equation(cout, poly);
		cout << endl;
	}
#endif



	for (h = 0; h < orbit_first.size(); h++) {
		cout << "orbit " << h << " / " << orbit_first.size() << " has length " << orbit_length[h] << endl;
	}

	FREE_int(poly);

	if (f_v) {
		cout << "boolean_function::search_for_bent_functions done" << endl;
	}
}



void boolean_function_print_function(int *poly, int sz, void *data)
{
	boolean_function *BFC = (boolean_function *) data;
	geometry_global Gg;
	longinteger_object a;

	BFC->evaluate(poly + 1, BFC->f_proj);
	int_vec_copy(BFC->f_proj, BFC->f_proj2, BFC->Q);
	Gg.AG_element_rank_longinteger(2, BFC->f_proj2, 1, BFC->Q, a);

	cout << " : ";
	int_vec_print(cout, BFC->f_proj, BFC->Q);
	cout << " : rk=" << a;

}

void boolean_function_reduction_function(int *poly, void *data)
{
	boolean_function *BFC = (boolean_function *) data;

	if (BFC->dim_kernel) {
		int i, i1, i2;
		int a, ma;

		for (i = 0; i < BFC->dim_kernel; i++) {
			i1 = BFC->Kernel[i * 2 + 0];
			i2 = BFC->Kernel[i * 2 + 1];
			a = poly[i1];
			if (a) {
				ma = BFC->Fq->negate(a);
				poly[i1] = 0;
				poly[i2] = BFC->Fq->add(poly[i2], ma);
			}

		}
	}
#if 0
	// c_0 = c_4:
	a = poly[0];
	if (a) {
		ma = BFC->Fq->negate(a);
		poly[0] = 0;
		poly[4] = BFC->Fq->add(poly[4], ma);
	}
	// c_1 = c_5:
	a = poly[1];
	if (a) {
		ma = BFC->Fq->negate(a);
		poly[1] = 0;
		poly[5] = BFC->Fq->add(poly[5], ma);
	}
#endif
	//BFC->evaluate(poly + 1, BFC->f_proj);
	//int_vec_print(cout, BFC->f_proj, BFC->Q);

}



}}

