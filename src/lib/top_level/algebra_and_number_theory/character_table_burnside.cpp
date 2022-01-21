/*
 * character_table_burnside.cpp
 *
 *  Created on: Nov 9, 2019
 *      Author: anton
 *
 *      originally started on February 26, 2015
 *
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::foundations;

namespace orbiter {
namespace top_level {
namespace apps_algebra {


void character_table_burnside::do_it(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "character_table_burnside::do_it" << endl;
	}

	algebra::a_domain *D;

	D = NEW_OBJECT(algebra::a_domain);
	D->init_integer_fractions(verbose_level);


	actions::action *A;
	ring_theory::longinteger_object go;
	int goi;
	int *Elt;
	int i, j;
	int f_no_base = FALSE;

	A = NEW_OBJECT(actions::action);
	A->init_symmetric_group(n, f_no_base, verbose_level);
	A->group_order(go);

	goi = go.as_int();
	cout << "Created group Sym(" << n << ") of size " << goi << endl;

	Elt = NEW_int(A->elt_size_in_int);

	groups::sims *S;

	S = A->Sims;

	for (i = 0; i < goi; i++) {
		S->element_unrank_lint(i, Elt);
		cout << "element " << i << " is ";
		A->element_print_quick(Elt, cout);
		cout << endl;
		}

	actions::action *Aconj;

	Aconj = NEW_OBJECT(actions::action);

	cout << "Creating action by conjugation" << endl;

	Aconj->induced_action_by_conjugation(S,
		S, FALSE /* f_ownership */, FALSE /* f_basis */, verbose_level);

	cout << "Creating action by conjugation done" << endl;

	induced_actions::action_by_conjugation *ABC;

	ABC = Aconj->G.ABC;

	groups::schreier *Sch;
	groups::strong_generators *SG;

	Sch = NEW_OBJECT(groups::schreier);

	Sch->init(Aconj, verbose_level - 2);


	SG = NEW_OBJECT(groups::strong_generators);

	SG->init_from_sims(S, 0);

#if 0
	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
		}
#endif

	Sch->init_generators(*SG->gens, verbose_level - 2);

	cout << "Computing conjugacy classes:" << endl;
	Sch->compute_all_point_orbits(verbose_level);


	int nb_classes;
	int *class_size;

	nb_classes = Sch->nb_orbits;

	class_size = NEW_int(nb_classes);

	for (i = 0; i < nb_classes; i++) {
		class_size[i] = Sch->orbit_len[i];
		}
	cout << "class sizes : ";
	Orbiter->Int_vec->print(cout, class_size, nb_classes);
	cout << endl;




	int *N;
	int r, r0;


	compute_multiplication_constants_center_of_group_ring(A,
		ABC,
		Sch, nb_classes, N, verbose_level);


	for (r = 0; r < nb_classes; r++) {
		cout << "N_" << r << ":" << endl;
		Orbiter->Int_vec->matrix_print(N + r * nb_classes * nb_classes, nb_classes, nb_classes);
		cout << endl;
		}


	r0 = compute_r0(N, nb_classes, verbose_level);


	if (r0 == -1) {
		cout << "Did not find a matrix with the right number "
				"of distinct eigenvalues" << endl;
		exit(1);
		}


	cout << "r0=" << r0 << endl;

	int *N0;

	N0 = N + r0 * nb_classes * nb_classes;




	int *Lambda;
	int nb_lambda;
	int *Mu;
	int *Mu_mult;
	int nb_mu;

	cout << "N_" << r0 << ":" << endl;

	integral_eigenvalues(N0, nb_classes,
		Lambda,
		nb_lambda,
		Mu,
		Mu_mult,
		nb_mu,
		0 /*verbose_level*/);

	cout << "Has " << nb_mu << " distinct eigenvalues" << endl;


	cout << "We found " << nb_lambda << " integer roots, they are: " << endl;
	Orbiter->Int_vec->print(cout, Lambda, nb_lambda);
	cout << endl;
	cout << "We found " << nb_mu << " distinct integer roots, they are: " << endl;
	for (i = 0; i < nb_mu; i++) {
		cout << Mu[i] << " with multiplicity " << Mu_mult[i] << endl;
		}

	int *Omega;


	compute_omega(D, N0, nb_classes, Mu, nb_mu, Omega, verbose_level);



	cout << "Omega:" << endl;
	D->print_matrix(Omega, nb_classes, nb_classes);
	//double_matrix_print(Omega, nb_classes, nb_classes);




	int *character_degree;


	compute_character_degrees(D, goi, nb_classes, Omega, class_size,
		character_degree, verbose_level);


	cout << "character degrees : ";
	Orbiter->Int_vec->print(cout, character_degree, nb_classes);
	cout << endl;


	int *character_table;


	compute_character_table(D, nb_classes, Omega,
		character_degree, class_size,
		character_table, verbose_level);





	cout << "character table:" << endl;
	Orbiter->Int_vec->matrix_print(character_table, nb_classes, nb_classes);

	int f_special = TRUE;
	int **Gens;
	int nb_gens;
	int t_max;
	int *Distribution;

	t_max = character_degree[0];
	for (i = 0; i < nb_classes; i++) {
		if (character_degree[i] > t_max) {
			t_max = character_degree[i];
			}
		}

	cout << "t_max=" << t_max << endl;

	cout << "creating generators:" << endl;
	create_generators(A, n, Gens, nb_gens, f_special, verbose_level);


	compute_Distribution_table(A, ABC,
		Sch, nb_classes,
		Gens,nb_gens, t_max, Distribution, verbose_level);


	cout << "Distribution table:" << endl;
	Orbiter->Int_vec->matrix_print(Distribution + nb_classes, t_max, nb_classes);


	for (i = 0; i < nb_classes; i++) {

		cout << "character " << i << " / " << nb_classes << ":" << endl;
		Orbiter->Int_vec->print(cout, character_table + i * nb_classes, nb_classes);
		cout << endl;


		int *S, a, t;

		S = NEW_int(t_max + 1);
		Orbiter->Int_vec->zero(S, t_max + 1);

		for (t = 0; t <= t_max; t++) {
			S[t] = 0;
			for (j = 0; j < nb_classes; j++) {
				a = Distribution[t * nb_classes + j];
				if (a == 0) {
					continue;
					}
				S[t] += a * character_table[i * nb_classes + j];
				}
			}
		cout << "S=";
		Orbiter->Int_vec->print(cout, S + 1, t_max);
		cout << endl;


		discreta_matrix M;

		int /*n,*/ deg;

		//n = character_degree[i];

		create_matrix(M, i, S, nb_classes,
			character_degree, class_size,
			verbose_level);

		cout << "M=" << endl;
		cout << M << endl;

		unipoly p;


		M.determinant(p, 0 /*verbose_level*/);
		if (f_v) {
			cout << "determinant:" << p << endl;
			}

		deg = p.degree();
		if (f_v) {
			cout << "has degree " << deg << endl;
			}




		FREE_int(S);
		}


	cout << "character table:" << endl;
	Orbiter->Int_vec->matrix_print(character_table, nb_classes, nb_classes);


	latex_interface L;

	L.print_integer_matrix_tex(cout, character_table, nb_classes, nb_classes);




	FREE_int(Distribution);
	for (i = 0; i < nb_gens; i++) {
		FREE_int(Gens[i]);
		}
	FREE_pint(Gens);


	FREE_int(character_table);
	FREE_int(character_degree);

	FREE_int(Omega);

	FREE_int(Lambda);
	FREE_int(Mu);
	FREE_int(Mu_mult);


	FREE_int(N);
	FREE_OBJECT(SG);
	FREE_OBJECT(Sch);
	FREE_OBJECT(Aconj);

	FREE_int(Elt);
	FREE_OBJECT(A);
	FREE_OBJECT(D);
	if (f_v) {
		cout << "character_table_burnside::do_it" << endl;
	}
}

void character_table_burnside::create_matrix(discreta_matrix &M, int i, int *S, int nb_classes,
	int *character_degree, int *class_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, ii, j;


	if (f_v) {
		cout << "character_table_burnside::create_matrix" << endl;
		}
	n = character_degree[i];
	if (f_v) {
		cout << "n=" << n << endl;
		}
	M.m_mn_n(n + 1, n + 1);

	M.elements_to_unipoly();

	for (j = 0; j <= n; j++) {

		{
		unipoly p;

		p.x_to_the_i(j);
		M.s_ij(0, n - j) = p;
		}

		}
	for (ii = 1; ii <= n; ii++) {

		cout << "ii=" << ii << endl;

		for (j = 0; j <= ii; j++) {
			unipoly p;

			p.one();
			if (j == 0) {
				p.m_ii(0, ii);
				}
			else {
				p.m_ii(0, S[j]);
				}
			cout << "j=" << j << " p=" << p << endl;

			M.s_ij(ii, ii - j) = p;
			}
		}

	if (f_v) {
		cout << "character_table_burnside::create_matrix done" << endl;
		}
}


void character_table_burnside::compute_character_table(
		algebra::a_domain *D, int nb_classes, int *Omega,
	int *character_degree, int *class_size,
	int *&character_table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	int i, j, w;

	if (f_v) {
		cout << "character_table_burnside::compute_character_table" << endl;
	}

	character_table = NEW_int(nb_classes * nb_classes);

	for (i = 0; i < nb_classes; i++) {

		for (j = 0; j < nb_classes; j++) {

			if (f_vv) {
				cout << "i=" << i << " j=" << j
					<< " character_degree[i]=" << character_degree[i]
					<< " omega_ij="
					<< D->as_int(D->offset(Omega, j * nb_classes + i), 0)
					<< " class_size[j]=" << class_size[j] << endl;
			}

			w = character_degree[i] * D->as_int(D->offset(Omega, j * nb_classes + i), 0);
			if (w % class_size[j]) {
				cout << "class size does not divide w" << endl;
				exit(1);
			}
			character_table[i * nb_classes + j] = w / class_size[j];
		}
	}


	if (f_v) {
		cout << "character_table_burnside::compute_character_table done" << endl;
	}
}

void character_table_burnside::compute_character_degrees(
		algebra::a_domain *D,
	int goi, int nb_classes, int *Omega, int *class_size,
	int *&character_degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, r, d, f;
	int *A, *B, *C, *Cv, *G, *S, *Sv, *E, *F;

	if (f_v) {
		cout << "character_table_burnside::compute_character_degrees" << endl;
		}

	character_degree = NEW_int(nb_classes);
	A = NEW_int(D->size_of_instance_in_int);
	B = NEW_int(D->size_of_instance_in_int);
	C = NEW_int(D->size_of_instance_in_int);
	Cv = NEW_int(D->size_of_instance_in_int);
	G = NEW_int(D->size_of_instance_in_int);
	S = NEW_int(D->size_of_instance_in_int);
	Sv = NEW_int(D->size_of_instance_in_int);
	E = NEW_int(D->size_of_instance_in_int);
	F = NEW_int(D->size_of_instance_in_int);

	for (i = 0; i < nb_classes; i++) {


		D->make_zero(S, 0);

		for (r = 0; r < nb_classes; r++) {
			D->copy(D->offset(Omega, r * nb_classes + i), A, 0);

			D->mult(A, A, B, 0);


			D->make_integer(C, class_size[r], 0);
			D->inverse(C, Cv, 0);


			D->mult(B, Cv, E, 0);

			D->add_apply(S, E, 0);
			}

		D->inverse(S, Sv, 0);

		D->make_integer(G, goi, 0);
		D->mult(G, Sv, F, 0);


		f = D->as_int(F, 0);
		d = sqrt(f);

		if (d * d != f) {
			cout << "f is not a perfect square" << endl;
			exit(1);
			}

		if (f_vv) {
			cout << "i=" << i << " d=" << d << endl;
			}

		character_degree[i] = d;
		}
	FREE_int(A);
	FREE_int(B);
	FREE_int(C);
	FREE_int(Cv);
	FREE_int(G);
	FREE_int(S);
	FREE_int(Sv);
	FREE_int(E);
	FREE_int(F);
	if (f_v) {
		cout << "character_table_burnside::compute_character_degrees done" << endl;
		}
}

void character_table_burnside::compute_omega(
		algebra::a_domain *D, int *N0, int nb_classes,
		int *Mu, int nb_mu, int *&Omega, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M;
	int *base_cols;
	int h, x, rk, i, j, a;

	if (f_v) {
		cout << "character_table_burnside::compute_omega" << endl;
		}
	Omega = NEW_int(nb_classes * nb_classes * D->size_of_instance_in_int);
	M = NEW_int(nb_classes * nb_classes * D->size_of_instance_in_int);
	base_cols = NEW_int(nb_classes);

	for (h = 0; h < nb_mu; h++) {

		x = Mu[h];
		if (f_v) {
			cout << "eigenvalue " << h << " / " << nb_mu
					<< " is " << x << ":" << endl;
			}
		for (i = 0; i < nb_classes; i++) {
			for (j = 0; j < nb_classes; j++) {
				a = N0[i * nb_classes + j];
				if (i == j) {
					a -= x;
					}
				D->make_integer(D->offset(M, i * nb_classes + j), a, 0);
				}
			}
		if (f_vv) {
			cout << "before get_image_and_kernel:" << endl;
			D->print_matrix(M, nb_classes, nb_classes);
			//double_matrix_print(M, nb_classes, nb_classes);
			}

		D->get_image_and_kernel(M, nb_classes, rk, verbose_level);

		//rk = double_Gauss(M, nb_classes, nb_classes,
		//base_cols, 0 /*verbose_level */);

		if (f_vv) {
			cout << "after get_image_and_kernel:" << endl;
			//double_matrix_print(M, nb_classes, nb_classes);
			D->print_matrix(M, nb_classes, nb_classes);

			cout << "after get_image_and_kernel, rk=" << rk << endl;
			}

		if (rk != nb_classes - 1) {
			cout << "rk != nb_classes - 1" << endl;
			exit(1);
			}

		int *b, *c;

		b = NEW_int(D->size_of_instance_in_int);
		c = NEW_int(D->size_of_instance_in_int);
		D->copy(D->offset(M, (nb_classes - 1) * nb_classes), b, 0);
		D->inverse(b, c, 0);

		cout << "c=";
		D->print(c);
		cout << endl;

		for (i = 0; i < nb_classes; i++) {
			D->mult_apply(D->offset(M,
					(nb_classes - 1) * nb_classes + i), c, 0);
			}

		if (f_vv) {
			cout << "after rescaling:" << endl;
			D->print_matrix(M, nb_classes, nb_classes);
			}
		for (i = 0; i < nb_classes; i++) {
			D->copy(D->offset(M, (nb_classes - 1) * nb_classes + i),
					D->offset(Omega, i * nb_classes + h), 0);
			}
		FREE_int(b);
		FREE_int(c);




		}


	if (f_vv) {
		cout << "Omega:" << endl;
		D->print_matrix(Omega, nb_classes, nb_classes);
		}

	FREE_int(M);
	FREE_int(base_cols);
	if (f_v) {
		cout << "character_table_burnside::compute_omega done" << endl;
		}
}

int character_table_burnside::compute_r0(int *N, int nb_classes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	int r, r0, i;

	if (f_v) {
		cout << "character_table_burnside::compute_r0" << endl;
		}
	r0 = -1;

	for (r = 0; r < nb_classes; r++) {

		int *Lambda;
		int nb_lambda;
		int *Mu;
		int *Mu_mult;
		int nb_mu;

		if (f_vv) {
			cout << "N_" << r << ":" << endl;
			}

		integral_eigenvalues(N + r * nb_classes * nb_classes, nb_classes,
			Lambda,
			nb_lambda,
			Mu,
			Mu_mult,
			nb_mu,
			0 /*verbose_level*/);


		if (f_vv) {
			cout << "Has " << nb_mu << " distinct eigenvalues" << endl;


			cout << "We found " << nb_lambda
					<< " integer roots, they are: " << endl;
			Orbiter->Int_vec->print(cout, Lambda, nb_lambda);
			cout << endl;
			cout << "We found " << nb_mu
					<< " distinct integer roots, they are: " << endl;
			for (i = 0; i < nb_mu; i++) {
				cout << Mu[i] << " with multiplicity " << Mu_mult[i] << endl;
				}
			}

		if (nb_mu == nb_classes) {
			r0 = r;
			}


		FREE_int(Lambda);
		FREE_int(Mu);
		FREE_int(Mu_mult);


		}
	if (f_v) {
		cout << "character_table_burnside::compute_r0 done" << endl;
		}
	return r0;
}

void character_table_burnside::compute_multiplication_constants_center_of_group_ring(
		actions::action *A,
		induced_actions::action_by_conjugation *ABC,
	groups::schreier *Sch, int nb_classes, int *&N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r, rl, rf, s, sl, sf, i, a, j, b, c, idx, t, tf; //, tl;


	if (f_v) {
		cout << "character_table_burnside::compute_multiplication_constants_center_of_group_ring" << endl;
		}

	N = NEW_int(nb_classes * nb_classes * nb_classes);
	Orbiter->Int_vec->zero(N, nb_classes * nb_classes * nb_classes);


	for (r = 0; r < nb_classes; r++) {
		rl = Sch->orbit_len[r];
		rf = Sch->orbit_first[r];

		for (s = 0; s < nb_classes; s++) {
			sl = Sch->orbit_len[s];
			sf = Sch->orbit_first[s];


			for (i = 0; i < rl; i++) {
				a = Sch->orbit[rf + i];

				for (j = 0; j < sl; j++) {
					b = Sch->orbit[sf + j];

					c = ABC->multiply(A, a, b, 0 /*verbose_level*/);


					idx = Sch->orbit_inv[c];

					t = Sch->orbit_number(c); //Sch->orbit_no[idx];

					tf = Sch->orbit_first[t];
					//tl = Sch->orbit_len[t];

					if (idx == tf) {
						N[r * nb_classes * nb_classes + s * nb_classes + t]++;
						}
					}
				}
			}
		}
	if (f_v) {
		cout << "character_table_burnside::compute_multiplication_constants_center_of_group_ring done" << endl;
		}
}

void character_table_burnside::compute_Distribution_table(
		actions::action *A, induced_actions::action_by_conjugation *ABC,
		groups::schreier *Sch, int nb_classes,
	int **Gens, int nb_gens, int t_max, int *&Distribution, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *Elt1;
	int *Elt2;
	int *Choice;
	int *Nb;
	int t, h, i, /*idx,*/ j;
	number_theory::number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "character_table_burnside::compute_Distribution_table" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	Choice = NEW_int(t_max);
	Distribution = NEW_int((t_max + 1) * nb_classes);
	Orbiter->Int_vec->zero(Distribution, (t_max + 1) * nb_classes);
	Nb = NEW_int(t_max + 1);

	for (t = 1; t <= t_max; t++) {
		Nb[t] = NT.i_power_j(nb_gens, t);
		}

	if (f_v) {
		cout << "Nb : ";
		Orbiter->Int_vec->print(cout, Nb + 1, t_max);
		cout << endl;
		}

	for (t = 1; t <= t_max; t++) {
		cout << "t=" << t << " Nb[t]=" << Nb[t] << endl;
		for (h = 0; h < Nb[t]; h++) {
			Gg.AG_element_unrank(nb_gens, Choice, 1, t, h);

			if (f_vvv) {
				cout << "h=" << h << " Choice=";
				Orbiter->Int_vec->print(cout, Choice, t);
				cout << endl;
				}

			multiply_word(A, Gens, Choice, t, Elt1, Elt2, verbose_level);

			i = ABC->rank(Elt1);


			//idx = Sch->orbit_inv[i];

			j = Sch->orbit_number(i); // Sch->orbit_no[idx];


			if (f_vvv) {
				cout << "word:";
				A->element_print(Elt1, cout);
				cout << " has rank " << i << " and belongs to class " << j;
				cout << endl;
				}

			Distribution[t * nb_classes + j]++;
			}

		if (f_v) {
			cout << "after t=" << t << " Distribution:" << endl;
			Orbiter->Int_vec->matrix_print(Distribution, t + 1, nb_classes);
			}
		}

	FREE_int(Choice);
	FREE_int(Nb);
	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "character_table_burnside::compute_Distribution_table done" << endl;
		}
}


void character_table_burnside::multiply_word(actions::action *A, int **Gens, int *Choice, int t, int *Elt1, int *Elt2, int verbose_level)
{
	int i;

	A->element_move(Gens[Choice[0]], Elt1, 0);
	for (i = 1; i < t; i++) {
		A->element_mult(Elt1, Gens[Choice[i]], Elt2, 0);
		A->element_move(Elt2, Elt1, 0);
		}
}

void character_table_burnside::create_generators(actions::action *A, int n, int **&Elt, int &nb_gens, int f_special, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int *v;

	if (f_v) {
		cout << "character_table_burnside::create_generators" << endl;
		}
	nb_gens = n - 1;
	Elt = NEW_pint(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		Elt[i] = NEW_int(A->elt_size_in_int);
		}
	v = NEW_int(n);


	if (f_special) {
		for (i = 0; i < nb_gens; i++) {
			for (j = 0; j < n; j++) {
				v[j] = j;
				}
			v[0] = i + 1;
			v[i + 1] = 0;
			A->make_element(Elt[i], v, 0 /* verbose_level */);
			}
		}
	else {
		for (i = 0; i < nb_gens; i++) {
			for (j = 0; j < n; j++) {
				v[j] = j;
				}
			v[i] = i + 1;
			v[i + 1] = i;
			A->make_element(Elt[i], v, 0 /* verbose_level */);
			}
		}
	cout << "generators:" << endl;
	for (i = 0; i < nb_gens; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(Elt[i], cout);
		cout << endl;
		}

	FREE_int(v);

}



void character_table_burnside::integral_eigenvalues(int *M, int n,
	int *&Lambda,
	int &nb_lambda,
	int *&Mu,
	int *&Mu_mult,
	int &nb_mu,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly charpoly;
	int *A;
	int *B;
	int i, deg, a, x;

	if (f_v) {
		cout << "character_table_burnside::integral_eigenvalues" << endl;
		}
	characteristic_poly(M, n, charpoly, 0 /*verbose_level*/);
	if (f_v) {
		cout << "characteristic polynomial:" << charpoly << endl;
		}

	deg = charpoly.degree();
	if (f_v) {
		cout << "has degree " << deg << endl;
		}

	A = NEW_int(deg + 1);
	B = NEW_int(deg + 1);

	for (i = 0; i <= deg; i++) {
		A[i] = charpoly.s_ii(i);
		}
	if (f_v) {
		cout << "coeffs : ";
		Orbiter->Int_vec->print(cout, A, deg + 1);
		cout << endl;
		}


	Lambda = NEW_int(deg);
	Mu = NEW_int(deg);
	Mu_mult = NEW_int(deg);
	nb_lambda = 0;
	nb_mu = 0;

	for (x = -100; x < 100; x++) {
		a = A[deg];
		for (i = deg - 1; i >= 0; i--) {
			a *= x;
			a += A[i];
			}
		if (a == 0) {
			if (f_v) {
				cout << "Found integer root " << x << endl;
				}
			Lambda[nb_lambda++] = x;
			if (nb_mu && Mu[nb_mu - 1] == x) {
				if (f_v) {
					cout << "The root is a multiple root" << endl;
					}
				Mu_mult[nb_mu - 1]++;
				}
			else {
				Mu[nb_mu] = x;
				Mu_mult[nb_mu] = 1;
				nb_mu++;
				}

			for (i = deg - 1; i >= 0; i--) {
				B[i] = A[i + 1];
				A[i] = A[i] + x * B[i];
				if (i == 0 && A[0]) {
					cout << "division unsuccessful" << endl;
					exit(1);
					}
				}
			Orbiter->Int_vec->copy(B, A, deg);
			deg--;
			if (f_v) {
				cout << "after dividing off, the polynomial is: ";
				Orbiter->Int_vec->print(cout, A, deg + 1);
				cout << endl;
				}

			x--; // try x again
			}
		}


	if (f_v) {
		cout << "after dividing off integer roots, the polynomial is: ";
		Orbiter->Int_vec->print(cout, A, deg + 1);
		cout << endl;
		}

	if (f_v) {
		cout << "We found " << nb_lambda << " integer roots, they are: " << endl;
		Orbiter->Int_vec->print(cout, Lambda, nb_lambda);
		cout << endl;
		cout << "We found " << nb_mu << " distinct integer roots, they are: " << endl;
		for (i = 0; i < nb_mu; i++) {
			cout << Mu[i] << " with multiplicity " << Mu_mult[i] << endl;
			}
		}

	FREE_int(A);
	FREE_int(B);
}

void character_table_burnside::characteristic_poly(int *N, int size, unipoly &charpoly, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, a;
	discreta_matrix M, M1, P, Pv, Q, Qv, S, T;

	if (f_v) {
		cout << "character_table_burnside::characteristic_poly" << endl;
		}
	M.m_mn(size, size);
	k = 0;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = N[k++];
			M.m_iji(i, j, a);
			}
		}
	if (f_vv) {
		cout << "M=" << endl;
		cout << M << endl;
		}


	M.elements_to_unipoly();
	M.minus_X_times_id();


#if 0
	M1 = M;
	cout << "M - x * Id=" << endl << M << endl;
	M.smith_normal_form(P, Pv, Q, Qv, verbose_level);

	cout << "the Smith normal form is:" << endl;
	cout << M << endl;

	S.mult(P, Pv);
	cout << "P * Pv=" << endl << S << endl;

	S.mult(Q, Qv);
	cout << "Q * Qv=" << endl << S << endl;

	S.mult(P, M1);
	cout << "T.mult(S, Q):" << endl;
	T.mult(S, Q);
	cout << "T=" << endl << T << endl;
#endif

	//int deg;
	//int l, lv, b;


	M.determinant(charpoly, verbose_level);
	//charpoly = M.s_ij(size - 1, size - 1);

	if (f_v) {
		cout << "characteristic polynomial:" << charpoly << endl;
		}
	//deg = charpoly.degree();
	//cout << "has degree " << deg << endl;

#if 0
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		if (b > q2) {
			b -= q;
			}
		//c = Fq.mult(b, lv);
		charpoly.m_ii(i, b);
		}

	cout << "characteristic polynomial:" << charpoly << endl;
#endif

	if (f_v) {
		cout << "character_table_burnside::characteristic_poly done" << endl;
		}
}



void character_table_burnside::double_swap(double &a, double &b)
{
	double c;

	c = a;
	a = b;
	b = c;
}

int character_table_burnside::double_Gauss(double *A, int m, int n, int *base_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	double pivot, pivot_inv, z, f, a, b, c, p;
	int i, j, k, jj, rank, idx;

	if (f_v) {
		cout << "character_table_burnside::double_Gauss" << endl;
		}
	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
			double_matrix_print(A, m, n);
			}
		// search for pivot element:
		idx = -1;
		for (k = i; k < m; k++) {
			if (idx == -1) {
				p = A[k * n + j];
				idx = k;
				}
			else {
				if (double_abs(A[k * n + j]) > double_abs(p)) {
					p = A[k * n + j];
					idx = k;
					}
				}
			} // next k
		if (f_v) {
			cout << "column " << i << " pivot is " << p << " in row " << idx << endl;
			}

		if (idx == -1 || double_abs(p) < 0.00001) { // no pivot found
			if (f_v) {
				cout << "no pivot found" << endl;
				}
			continue; // increase j, leave i constant
			}
		else {
			k = idx;
			// pivot element found:
			if (k != i) {
				for (jj = j; jj < n; jj++) {
					double_swap(A[i * n + jj], A[k * n + jj]);
					}
				}
			}

		if (f_vv) {
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
			double_matrix_print(A, m, n);
			}

		base_cols[i] = j;
		//if (FALSE) {
		//	cout << "."; cout.flush();
		//	}

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
			}
		//pivot_inv = inv_table[pivot];
		pivot_inv = 1. / pivot;
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
			}
		// make pivot to 1:
		for (jj = j; jj < n; jj++) {
			A[i * n + jj] *= pivot_inv;
			}
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
				<< " made to one: " << A[i * n + j] << endl;
			double_matrix_print(A, m, n);
			}


		// do the gaussian elimination:

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row " << i + 1 << " to row " << m - 1 << ":" << endl;
			}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (double_abs(z) < 0.0000000001) {
				continue;
				}
			f = z;
			//A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = j; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special
				//      b - z * pivot_inv * a  if f_special
				c = b - f * a;
				A[k * n + jj] = c;
				}
			if (f_vv) {
				double_matrix_print(A, m, n);
				}
			}
		i++;
		} // next j
	rank = i;


	for (i = rank - 1; i >= 0; i--) {
		if (f_v) {
			cout << "."; cout.flush();
			}
		j = base_cols[i];
		a = A[i * n + j];

		// do the gaussian elimination in the upper part:
		for (k = i - 1; k >= 0; k--) {
			z = A[k * n + j];
			if (z == 0) {
				continue;
				}
			//A[k * n + j] = 0;
			for (jj = j; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				c = - z * a;
				c += b;
				A[k * n + jj] = c;
				}
			} // next k
		} // next i

	return rank;
}

void character_table_burnside::double_matrix_print(double *A, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << setw(10) << A[i * n + j] << " ";
			}
		cout << endl;
		}
}

double character_table_burnside::double_abs(double x)
{
	if (x < 0) {
		return - x;
		}
	else {
		return x;
		}
}

void character_table_burnside::kernel_columns(int n, int nb_base_cols, int *base_cols, int *kernel_cols)
{
	int i, j, k;

	j = k = 0;
	for (i = 0; i < n; i++) {
		if (j < nb_base_cols && i == base_cols[j]) {
			j++;
			continue;
			}
		kernel_cols[k++] = i;
		}
}

void character_table_burnside::matrix_get_kernel(double *M, int m, int n, int *base_cols, int nb_base_cols,
	int &kernel_m, int &kernel_n, double *kernel)
	// kernel must point to the appropriate amount of memory! (at least n * (n - nb_base_cols) int's)
{
	int r, k, i, j, ii, iii, a, b;
	int *kcol;

	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;

	kcol = NEW_int(k);

	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
		}
	else {
		b = -1;
		}
	for (i = 0; i < n; i++) {
		if (i == b) {
			j++;
			if (j < r) {
				b = base_cols[j];
				}
			else {
				b = -1;
				}
			}
		else {
			kcol[ii] = i;
			ii++;
			}
		}
	if (ii != k) {
		cout << "character_table_burnside::matrix_get_kernel ii != k" << endl;
		exit(1);
		}
	//cout << "kcol = " << kcol << endl;
	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
		}
	else {
		b = -1;
		}
	for (i = 0; i < n; i++) {
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kcol[iii];
				kernel[i * kernel_n + iii] = M[j * n + a];
				}
			j++;
			if (j < r) {
				b = base_cols[j];
				}
			else {
				b = -1;
				}
			}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					kernel[i * kernel_n + iii] = -1.;
					}
				else {
					kernel[i * kernel_n + iii] = 0;
					}
				}
			ii++;
			}
		}
	FREE_int(kcol);
}


int character_table_burnside::double_as_int(double x)
{
	int a;
	double a1, a2;

	a = (int) (x);
	a1 = (double)a - 0.000001;
	a2 = (double)a + 0.000001;
	if (a1 < a && a < a2) {
		return a;
		}
	cout << "error in double_as_int" << endl;
	exit(1);
}




}}}

