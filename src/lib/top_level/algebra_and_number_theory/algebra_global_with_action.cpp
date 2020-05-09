/*
 * algebra_global_with_action.cpp
 *
 *  Created on: Dec 15, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::foundations;

namespace orbiter {
namespace top_level {



void algebra_global_with_action::classes_GL(int q, int d, int f_no_eigenvalue_one, int verbose_level)
{
	gl_classes C;
	gl_class_rep *R;
	int nb_classes;
	finite_field *F;
	int i;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	C.init(d, F, verbose_level);

	C.make_classes(R, nb_classes, f_no_eigenvalue_one, verbose_level);

	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;
	int a;
	int *Mtx;
	int *Elt;



	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);

	Mtx = NEW_int(d * d);
	Elt = NEW_int(A->elt_size_in_int);


	for (i = 0; i < nb_classes; i++) {

		C.make_matrix_from_class_rep(Mtx, R + i, 0 /*verbose_level - 1 */);

		A->make_element(Elt, Mtx, 0);

		a = A->Sims->element_rank_lint(Elt);

		cout << "Representative of class " << i << " / "
				<< nb_classes << " has rank " << a << endl;
		int_matrix_print(Elt, d, d);

		C.print_matrix_and_centralizer_order_latex(
				cout, R + i);

		}


	char fname[1000];

	sprintf(fname, "Class_reps_GL_%d_%d.tex", d, q);
	{
	ofstream fp(fname);
	latex_interface L;

	L.head_easy(fp);
	C.report(fp, verbose_level);
	L.foot(fp);
	}

	//make_gl_classes(d, q, f_no_eigenvalue_one, verbose_level);

	FREE_int(Mtx);
	FREE_int(Elt);
	FREE_OBJECTS(R);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
}

void algebra_global_with_action::do_normal_form(int q, int d,
		int f_no_eigenvalue_one, int *data, int data_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form" << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form before C.init" << endl;
		}
	C.init(d, F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form after C.init" << endl;
		}

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form before C.make_classes" << endl;
		}
	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form after C.make_classes" << endl;
		}



	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;


	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */, TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	int class_rep;

	int *Elt, *Basis;

	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	//go = Go.as_int();

	cout << "Making element from data ";
	int_vec_print(cout, data, data_sz);
	cout << endl;

	//A->Sims->element_unrank_int(elt_idx, Elt);
	A->make_element(Elt, data, verbose_level);

	cout << "Looking at element:" << endl;
	int_matrix_print(Elt, d, d);


	gl_class_rep *R1;

	R1 = NEW_OBJECT(gl_class_rep);

	C.identify_matrix(Elt, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes, R1,
			0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

	FREE_OBJECT(R1);




	FREE_int(Elt);
	FREE_int(Basis);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	FREE_OBJECTS(Reps);
}


void algebra_global_with_action::do_identify_one(int q, int d,
		int f_no_eigenvalue_one, int elt_idx,
		int verbose_level)
{
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;


	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	int class_rep;

	int *Elt, *Basis;

	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	//int go;
	//go = Go.as_int();

	cout << "Looking at element " << elt_idx << ":" << endl;

	A->Sims->element_unrank_lint(elt_idx, Elt);
	int_matrix_print(Elt, d, d);


	gl_class_rep *R1;

	R1 = NEW_OBJECT(gl_class_rep);

	C.identify_matrix(Elt, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes, R1, 0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

	FREE_OBJECT(R1);




	FREE_int(Elt);
	FREE_int(Basis);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	FREE_OBJECTS(Reps);
}

void algebra_global_with_action::do_identify_all(int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
{
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	action *A;
	longinteger_object Go;
	int *Class_count;
	vector_ge *nice_gens;


	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	int i, go, class_rep;

	int *Elt, *Basis;

	Class_count = NEW_int(nb_classes);
	int_vec_zero(Class_count, nb_classes);
	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	go = Go.as_int();
	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << ":" << endl;

		A->Sims->element_unrank_lint(i, Elt);
		int_matrix_print(Elt, d, d);


		gl_class_rep *R1;

		R1 = NEW_OBJECT(gl_class_rep);

		C.identify_matrix(Elt, R1, Basis, verbose_level);

		class_rep = C.find_class_rep(Reps,
				nb_classes, R1, 0 /* verbose_level */);

		cout << "class = " << class_rep << endl;

		Class_count[class_rep]++;

		FREE_OBJECT(R1);
		}

	cout << "class : count" << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << setw(3) << i << " : " << setw(10)
				<< Class_count[i] << endl;
		}



	FREE_int(Class_count);
	FREE_int(Elt);
	FREE_int(Basis);
	FREE_OBJECTS(Reps);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
}

void algebra_global_with_action::do_random(int q, int d, int f_no_eigenvalue_one, int verbose_level)
{
	//gl_random_matrix(d, q, verbose_level);

	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);

	int *Mtx;
	int *Basis;
	int class_rep;


	Mtx = NEW_int(d * d);
	Basis = NEW_int(d * d);

	C.F->random_invertible_matrix(Mtx, d, verbose_level - 2);


	gl_class_rep *R1;

	R1 = NEW_OBJECT(gl_class_rep);

	C.identify_matrix(Mtx, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes,
			R1, 0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

	FREE_OBJECT(R1);

	FREE_int(Mtx);
	FREE_int(Basis);
	FREE_OBJECTS(Reps);
	FREE_OBJECT(F);
}


void algebra_global_with_action::group_table(int q, int d, int f_poly, const char *poly,
		int f_no_eigenvalue_one, int verbose_level)
{
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	int *Class_rep;
	int *List;
	int list_sz, a, b, j, h;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	if (f_poly) {
		F->init_override_polynomial(q, poly, 0);
		}
	else {
		F->init(q, 0);
		}

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);


	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;


	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */,
			F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	int i, go, class_rep;
	int eval;

	int *Elt;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Basis;

	Elt = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);




	go = Go.as_int();
	List = NEW_int(go);
	list_sz = 0;
	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << ":" << endl;

		A->Sims->element_unrank_lint(i, Elt);
		int_matrix_print(Elt, d, d);

		{
		unipoly_domain U(C.F);
		unipoly_object char_poly;



		U.create_object_by_rank(char_poly, 0, __FILE__, __LINE__, verbose_level);

		U.characteristic_polynomial(Elt,
				d, char_poly, verbose_level - 2);

		cout << "The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;

		eval = U.substitute_scalar_in_polynomial(char_poly,
				1 /* scalar */, 0 /* verbose_level */);
		U.delete_object(char_poly);


		}

		if (eval) {
			List[list_sz++] = i;
			}

		} // next i

	cout << "Found " << list_sz
			<< " elements without eigenvalue one" << endl;


	Class_rep = NEW_int(list_sz);

	for (i = 0; i < list_sz; i++) {
		a = List[i];

		cout << "Looking at element " << a << ":" << endl;

		A->Sims->element_unrank_lint(a, Elt);
		int_matrix_print(Elt, d, d);


		gl_class_rep *R1;

		R1 = NEW_OBJECT(gl_class_rep);

		C.identify_matrix(Elt, R1, Basis, verbose_level);

		class_rep = C.find_class_rep(Reps,
				nb_classes, R1, 0 /* verbose_level */);


		FREE_OBJECT(R1);


		cout << "class = " << class_rep << endl;
		Class_rep[i] = class_rep;
		}

	int *Group_table;
	int *Table;

	Group_table = NEW_int(list_sz * list_sz);
	int_vec_zero(Group_table, list_sz * list_sz);
	for (i = 0; i < list_sz; i++) {
		a = List[i];
		A->Sims->element_unrank_lint(a, Elt1);
		for (j = 0; j < list_sz; j++) {
			b = List[j];
			A->Sims->element_unrank_lint(b, Elt2);
			A->element_mult(Elt1, Elt2, Elt3, 0);
			h = A->Sims->element_rank_lint(Elt3);
			Group_table[i * list_sz + j] = h;
			}
		}
	int L_sz = list_sz + 1;
	Table = NEW_int(L_sz * L_sz);
	int_vec_zero(Table, L_sz * L_sz);
	for (i = 0; i < list_sz; i++) {
		Table[0 * L_sz + 1 + i] = List[i];
		Table[(i + 1) * L_sz + 0] = List[i];
		}
	for (i = 0; i < list_sz; i++) {
		for (j = 0; j < list_sz; j++) {
			Table[(i + 1) * L_sz + 1 + j] =
					Group_table[i * list_sz + j];
			}
		}
	cout << "extended group table:" << endl;
	int_matrix_print(Table, L_sz, L_sz);


	const char *fname = "group_table.tex";

	{
	ofstream fp(fname);
	latex_interface L;

	L.head(fp, FALSE /* f_book */, FALSE /* f_title */,
		"" /*const char *title */, "" /*const char *author */,
		FALSE /* f_toc */, FALSE /* f_landscape */, FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */, FALSE /* f_pagenumbers */,
		NULL /* extra_praeamble */);


	L.print_integer_matrix_tex_block_by_block(fp, Table, L_sz, L_sz, 15);



	L.foot(fp);

	}


	FREE_int(List);
	FREE_int(Class_rep);
	FREE_int(Elt);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Basis);
	FREE_OBJECTS(Reps);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
}

void algebra_global_with_action::centralizer_brute_force(int q, int d,
		int elt_idx, int verbose_level)
{
	action *A;
	longinteger_object Go;
	finite_field *F;
	vector_ge *nice_gens;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	int i, go;

	int *Elt;
	int *Eltv;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *List;
	int sz;

	Elt = NEW_int(A->elt_size_in_int);
	Eltv = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);




	go = Go.as_int();
	List = NEW_int(go);
	sz = 0;



	A->Sims->element_unrank_lint(elt_idx, Elt);

	cout << "Computing centralizer of element "
			<< elt_idx << ":" << endl;
	int_matrix_print(Elt, d, d);

	A->element_invert(Elt, Eltv, 0);

	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << " / " << go << endl;

		A->Sims->element_unrank_lint(i, Elt1);
		//int_matrix_print(Elt1, d, d);


		A->element_invert(Elt1, Elt2, 0);
		A->element_mult(Elt2, Elt, Elt3, 0);
		A->element_mult(Elt3, Elt1, Elt2, 0);
		A->element_mult(Elt2, Eltv, Elt3, 0);
		if (A->is_one(Elt3)) {
			List[sz++] = i;
			}
		}

	cout << "The centralizer has order " << sz << endl;

	int a;
	vector_ge *gens;
	vector_ge *SG;
	int *tl;

	gens = NEW_OBJECT(vector_ge);
	SG = NEW_OBJECT(vector_ge);
	tl = NEW_int(A->base_len());
	gens->init(A, verbose_level - 2);
	gens->allocate(sz, verbose_level - 2);

	for (i = 0; i < sz; i++) {
		a = List[i];

		cout << "Looking at element " << i << " / " << sz
				<< " which is " << a << endl;

		A->Sims->element_unrank_lint(a, Elt1);
		int_matrix_print(Elt1, d, d);

		A->element_move(Elt1, gens->ith(i), 0);
		}

	sims *Cent;

	Cent = A->create_sims_from_generators_with_target_group_order_int(
			gens, sz, 0 /* verbose_level */);
	Cent->extract_strong_generators_in_order(*SG, tl,
			0 /* verbose_level */);
	cout << "strong generators for the centralizer are:" << endl;
	for (i = 0; i < SG->len; i++) {

		A->element_move(SG->ith(i), Elt1, 0);
		a = A->Sims->element_rank_lint(Elt1);

		cout << "Element " << i << " / " << SG->len
				<< " which is " << a << endl;

		int_matrix_print(Elt1, d, d);

		}



	FREE_int(Elt);
	FREE_int(Eltv);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
}


void algebra_global_with_action::centralizer(int q, int d,
		int elt_idx, int verbose_level)
{
	finite_field *F;
	action *A_PGL;
	action *A_GL;
	longinteger_object Go;
	vector_ge *nice_gens;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	A_PGL = NEW_OBJECT(action);
	A_PGL->init_projective_group(d /* n */, F,
		FALSE /* f_semilinear */,
		TRUE /* f_basis */, TRUE /* f_init_sims */,
		nice_gens,
		0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A_PGL->print_base();
	A_PGL->group_order(Go);

	A_GL = NEW_OBJECT(action);
	A_GL->init_general_linear_group(d /* n */, F,
		FALSE /* f_semilinear */,
		TRUE /* f_basis */, TRUE /* f_init_sims */,
		nice_gens,
		0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A_GL->print_base();
	A_GL->group_order(Go);

	int *Elt;

	Elt = NEW_int(A_PGL->elt_size_in_int);


	//go = Go.as_int();

	cout << "Looking at element " << elt_idx << ":" << endl;

	A_PGL->Sims->element_unrank_lint(elt_idx, Elt);
	int_matrix_print(Elt, d, d);

	strong_generators *Cent;
	strong_generators *Cent_GL;
	longinteger_object go, go1;

	Cent = NEW_OBJECT(strong_generators);
	Cent_GL = NEW_OBJECT(strong_generators);

	cout << "before Cent->init_centralizer_of_matrix" << endl;
	Cent->init_centralizer_of_matrix(A_PGL, Elt, verbose_level);
	cout << "before Cent->init_centralizer_of_matrix" << endl;

	cout << "before Cent_GL->init_centralizer_of_matrix_general_linear" << endl;
	Cent_GL->init_centralizer_of_matrix_general_linear(
			A_PGL, A_GL, Elt, verbose_level);
	cout << "after Cent_GL->init_centralizer_of_matrix_general_linear" << endl;



	Cent->group_order(go);
	Cent_GL->group_order(go1);

	cout << "order of centralizer in PGL: " << go << " in GL: " << go1 << endl;
	FREE_int(Elt);
	FREE_OBJECT(Cent);
	FREE_OBJECT(Cent_GL);
	FREE_OBJECT(A_GL);
	FREE_OBJECT(A_PGL);
	FREE_OBJECT(F);

}

void algebra_global_with_action::centralizer(int q, int d, int verbose_level)
{
	action *A;
	finite_field *F;
	longinteger_object Go;
	vector_ge *nice_gens;
	int go, i;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);

	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);


	go = Go.as_int();

	for (i = 0; i < go; i++) {
		cout << "Looking at element " << i << ":" << endl;

		A->Sims->element_unrank_lint(i, Elt);
		int_matrix_print(Elt, d, d);

		sims *Cent;
		longinteger_object cent_go;

		Cent = A->create_sims_for_centralizer_of_matrix(
				Elt, verbose_level);
		Cent->group_order(cent_go);

		cout << "Looking at element " << i
				<< ", the centralizer has order " << cent_go << endl;



		FREE_OBJECT(Cent);

		}



	FREE_int(Elt);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
}


void algebra_global_with_action::analyze_group(action *A, sims *S,
		vector_ge *SG, vector_ge *gens2, int verbose_level)
{
	int *Elt1;
	int *Elt2;
	int i, goi;
	longinteger_object go;
	int *perm;
	int *primes;
	int *exponents;
	int factorization_length;
	int nb_primes, nb_gens2;
	number_theory_domain NT;


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);


	S->group_order(go);
	goi = go.as_int();

	factorization_length = NT.factor_int(goi, primes, exponents);
	cout << "analyzing a group of order " << goi << " = ";
	NT.print_factorization(factorization_length, primes, exponents);
	cout << endl;

	nb_primes = 0;
	for (i = 0; i < factorization_length; i++) {
		nb_primes += exponents[i];
		}
	cout << "nb_primes=" << nb_primes << endl;
	gens2->init(A, verbose_level - 2);
	gens2->allocate(nb_primes, verbose_level - 2);

	compute_regular_representation(A, S, SG, perm, verbose_level);

	int *center;
	int size_center;

	center = NEW_int(goi);

	S->center(*SG, center, size_center, verbose_level);

	cout << "the center is:" << endl;
	for (i = 0; i < size_center; i++) {
		cout << i << " element has rank " << center[i] << endl;
		S->element_unrank_lint(center[i], Elt1);
		A->print(cout, Elt1);
		//A->print_as_permutation(cout, Elt1);
		cout << endl;
		}
	cout << endl << endl;

	S->element_unrank_lint(center[1], Elt1);
	A->move(Elt1, gens2->ith(0));
	nb_gens2 = 1;

	cout << "chosen generator " << nb_gens2 - 1 << endl;
	A->print(cout, gens2->ith(nb_gens2 - 1));

	factor_group *FactorGroup;

	FactorGroup = NEW_OBJECT(factor_group);

	create_factor_group(A, S, goi, size_center, center,
			FactorGroup, verbose_level);

	cout << "FactorGroup created" << endl;
	cout << "Order of FactorGroup is " <<
			FactorGroup->goi_factor_group << endl;


	cout << "computing the regular representation of degree "
			<< FactorGroup->goi_factor_group << ":" << endl;


	for (i = 0; i < SG->len; i++) {
		FactorGroup->FactorGroup->print_as_permutation(cout, SG->ith(i));
		cout << endl;
		}
	cout << endl;


#if 0
	cout << "now listing all elements:" << endl;
	for (i = 0; i < FactorGroup->goi_factor_group; i++) {
		FactorGroup->FactorGroup->Sims->element_unrank_int(i, Elt1);
		cout << "element " << i << ":" << endl;
		A->print(cout, Elt1);
		FactorGroup->FactorGroupConjugated->print_as_permutation(cout, Elt1);
		cout << endl;
		}
	cout << endl << endl;
#endif




	sims H1, H2, H3;
	longinteger_object goH1, goH2, goH3;
	vector_ge SGH1, SGH2, SGH3;
	int *tl1, *tl2, *tl3, *tlF1, *tlF2;

	tl1 = NEW_int(A->base_len());
	tl2 = NEW_int(A->base_len());
	tl3 = NEW_int(A->base_len());
	tlF1 = NEW_int(A->base_len());
	tlF2 = NEW_int(A->base_len());


	// now we compute H1, the derived group


	H1.init(FactorGroup->FactorGroup, verbose_level - 2);
	H1.init_trivial_group(verbose_level - 1);
	H1.build_up_subgroup_random_process(FactorGroup->FactorGroup->Sims,
		choose_random_generator_derived_group, verbose_level - 1);
	H1.group_order(goH1);
	cout << "the commutator subgroup has order " << goH1 << endl << endl;
	H1.extract_strong_generators_in_order(SGH1, tl1, verbose_level - 2);
	for (i = 0; i < SGH1.len; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, SGH1.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH1.ith(i));
		//cout << endl;
		}
	cout << endl << endl;


	int size_H1;
	int *elts_H1;

	size_H1 = goH1.as_int();
	elts_H1 = NEW_int(size_H1);


	FactorGroup->FactorGroup->Sims->element_ranks_subgroup(
			&H1, elts_H1, verbose_level);
	cout << "the ranks of elements in H1 are:" << endl;
	int_vec_print(cout, elts_H1, size_H1);
	cout << endl;

	factor_group *ModH1;

	ModH1 = NEW_OBJECT(factor_group);

	create_factor_group(FactorGroup->FactorGroupConjugated,
		FactorGroup->FactorGroup->Sims,
		FactorGroup->goi_factor_group,
		size_H1, elts_H1, ModH1, verbose_level);



	cout << "ModH1 created" << endl;
	cout << "Order of ModH1 is " << ModH1->goi_factor_group << endl;



	cout << "the elements of ModH1 are:" << endl;
	for (i = 0; i < ModH1->goi_factor_group; i++) {
		cout << "element " << i << ":" << endl;
		ModH1->FactorGroup->Sims->element_unrank_lint(i, Elt1);
		A->print(cout, Elt1);
		cout << endl;
		cout << "in the factor group mod H1" << endl;
		ModH1->FactorGroupConjugated->print_as_permutation(cout, Elt1);
		cout << endl;
		cout << "in the factor group mod center" << endl;
		FactorGroup->FactorGroupConjugated->print_as_permutation(
				cout, Elt1);
		cout << endl;
		}




	// now we compute H2, the second derived group


	H2.init(FactorGroup->FactorGroup, verbose_level - 2);
	H2.init_trivial_group(verbose_level - 1);
	H2.build_up_subgroup_random_process(&H1,
		choose_random_generator_derived_group, verbose_level - 1);
	H2.group_order(goH2);
	cout << "the second commutator subgroup has order "
			<< goH2 << endl << endl;
	H2.extract_strong_generators_in_order(SGH2, tl2, verbose_level - 2);
	for (i = 0; i < SGH2.len; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, SGH2.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH2.ith(i));
		//cout << endl;

		A->move(SGH2.ith(i), gens2->ith(nb_gens2));
		nb_gens2++;
		cout << "chosen generator " << nb_gens2 - 1 << endl;
		A->print(cout, gens2->ith(nb_gens2 - 1));

		}
	cout << endl << endl;

	int size_H2;
	int *elts_H2;

	size_H2 = goH2.as_int();
	elts_H2 = NEW_int(size_H1);


	H1.element_ranks_subgroup(&H2, elts_H2, verbose_level);
	cout << "the ranks of elements in H2 are:" << endl;
	int_vec_print(cout, elts_H2, size_H2);
	cout << endl;

	factor_group *ModH2;

	ModH2 = NEW_OBJECT(factor_group);

	create_factor_group(FactorGroup->FactorGroupConjugated,
		&H1,
		size_H1,
		size_H2, elts_H2, ModH2, verbose_level);



	cout << "ModH2 created" << endl;
	cout << "Order of ModH2 is " << ModH2->goi_factor_group << endl;

	cout << "the elements of ModH2 are:" << endl;
	for (i = 0; i < ModH2->goi_factor_group; i++) {
		cout << "element " << i << ":" << endl;
		ModH2->FactorGroup->Sims->element_unrank_lint(i, Elt1);
		A->print(cout, Elt1);
		cout << endl;
		cout << "in the factor group mod H2" << endl;
		ModH2->FactorGroupConjugated->print_as_permutation(cout, Elt1);
		cout << endl;
		//cout << "in the factor group mod center" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, Elt1);
		//cout << endl;
		}

	vector_ge SG_F1, SG_F2;

	ModH2->FactorGroup->Sims->extract_strong_generators_in_order(
			SG_F2, tlF2, verbose_level - 2);
	for (i = 0; i < SG_F2.len; i++) {
		cout << "generator " << i << " for ModH2:" << endl;
		A->print(cout, SG_F2.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH2.ith(i));
		//cout << endl;

		A->move(SG_F2.ith(i), gens2->ith(nb_gens2));
		nb_gens2++;
		cout << "chosen generator " << nb_gens2 - 1 << endl;
		A->print(cout, gens2->ith(nb_gens2 - 1));

		}
	cout << endl << endl;

	ModH1->FactorGroup->Sims->extract_strong_generators_in_order(
			SG_F1, tlF1, verbose_level - 2);
	for (i = 0; i < SG_F1.len; i++) {
		cout << "generator " << i << " for ModH1:" << endl;
		A->print(cout, SG_F1.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH2.ith(i));
		//cout << endl;

		A->move(SG_F1.ith(i), gens2->ith(nb_gens2));
		nb_gens2++;
		cout << "chosen generator " << nb_gens2 - 1 << endl;
		A->print(cout, gens2->ith(nb_gens2 - 1));

		}
	cout << endl << endl;

	cout << "we found " << nb_gens2 << " generators:" << endl;
	for (i = 0; i < nb_gens2; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, gens2->ith(i));
		}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(perm);
	FREE_int(tl1);
	FREE_int(tl2);
	FREE_int(tl3);
	FREE_int(tlF1);
	FREE_int(tlF2);
}

void algebra_global_with_action::compute_regular_representation(action *A, sims *S,
		vector_ge *SG, int *&perm, int verbose_level)
{
	longinteger_object go;
	int goi, i;
	combinatorics_domain Combi;

	S->group_order(go);
	goi = go.as_int();
	cout << "computing the regular representation of degree "
			<< go << ":" << endl;
	perm = NEW_int(SG->len * goi);

	for (i = 0; i < SG->len; i++) {
		S->regular_representation(SG->ith(i),
				perm + i * goi, verbose_level);
		}
	cout << endl;
	for (i = 0; i < SG->len; i++) {
		Combi.perm_print_offset(cout,
			perm + i * goi, goi, 1 /* offset */,
			FALSE /* f_print_cycles_of_length_one */,
			FALSE /* f_cycle_length */, FALSE, 0,
			TRUE /* f_orbit_structure */,
			NULL, NULL);
		cout << endl;
		}
}

void algebra_global_with_action::presentation(action *A, sims *S, int goi,
		vector_ge *gens, int *primes, int verbose_level)
{
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int i, j, jj, k, l, a, b;
	int word[100];
	int *word_list;
	int *inverse_word_list;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);

	word_list = NEW_int(goi);
	inverse_word_list = NEW_int(goi);

	l = gens->len;

	cout << "presentation of length " << l << endl;
	cout << "primes: ";
	int_vec_print(cout, primes, l);
	cout << endl;

#if 0
	// replace g5 by  g5 * g3:
	A->mult(gens->ith(5), gens->ith(3), Elt1);
	A->move(Elt1, gens->ith(5));

	// replace g7 by  g7 * g4:
	A->mult(gens->ith(7), gens->ith(4), Elt1);
	A->move(Elt1, gens->ith(7));
#endif



	for (i = 0; i < goi; i++) {
		inverse_word_list[i] = -1;
		}
	for (i = 0; i < goi; i++) {
		A->one(Elt1);
		j = i;
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
			}
		for (k = 0; k < l; k++) {
			b = word[k];
			while (b) {
				A->mult(Elt1, gens->ith(k), Elt2);
				A->move(Elt2, Elt1);
				b--;
				}
			}
		A->move(Elt1, Elt2);
		a = S->element_rank_lint(Elt2);
		word_list[i] = a;
		inverse_word_list[a] = i;
		cout << "word " << i << " = ";
		int_vec_print(cout, word, 9);
		cout << " gives " << endl;
		A->print(cout, Elt1);
		cout << "which is element " << word_list[i] << endl;
		cout << endl;
		}
	cout << "i : word_list[i] : inverse_word_list[i]" << endl;
	for (i = 0; i < goi; i++) {
		cout << setw(5) << i << " : " << setw(5) << word_list[i]
			<< " : " << setw(5) << inverse_word_list[i] << endl;
		}



	for (i = 0; i < l; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, gens->ith(i));
		cout << endl;
		}
	for (i = 0; i < l; i++) {
		A->move(gens->ith(i), Elt1);
		A->element_power_int_in_place(Elt1, primes[i], 0);
		a = S->element_rank_lint(Elt1);
		cout << "generator " << i << " to the power " << primes[i]
			<< " is elt " << a << " which is word "
			<< inverse_word_list[a];
		j = inverse_word_list[a];
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
			}
		int_vec_print(cout, word, l);
		cout << " :" << endl;
		A->print(cout, Elt1);
		cout << endl;
		}


	for (i = 0; i < l; i++) {
		A->move(gens->ith(i), Elt1);
		A->invert(Elt1, Elt2);
		for (j = 0; j < i; j++) {
			A->mult(Elt2, gens->ith(j), Elt3);
			A->mult(Elt3, Elt1, Elt4);
			cout << "g_" << j << "^{g_" << i << "} =" << endl;
			a = S->element_rank_lint(Elt4);
			cout << "which is element " << a << " which is word "
				<< inverse_word_list[a] << " = ";
			jj = inverse_word_list[a];
			for (k = 0; k < l; k++) {
				b = jj % primes[k];
				word[k] = b;
				jj = jj - b;
				jj = jj / primes[k];
				}
			int_vec_print(cout, word, l);
			cout << endl;
			A->print(cout, Elt4);
			cout << endl;
			}
		cout << endl;
		}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);

	FREE_int(word_list);
	FREE_int(inverse_word_list);
}


void algebra_global_with_action::do_eigenstuff(int q, int size, int *Data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	discreta_matrix M;
	int i, j, k, a, p, h;
	finite_field Fq;
	//unipoly_domain U;
	//unipoly_object char_poly;
	number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff" << endl;
	}
	M.m_mn(size, size);
	k = 0;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = Data[k++];
			M.m_iji(i, j, a);
		}
	}

	if (f_v) {
		cout << "M=" << endl;
		cout << M << endl;
	}

	if (!NT.is_prime_power(q, p, h)) {
		cout << "q is not prime power, we need a prime power" << endl;
		exit(1);
	}
#if 0
	if (h > 1) {
		cout << "prime powers are not implemented yet" << endl;
		exit(1);
	}
#endif
	Fq.init(q, verbose_level);

	//domain d(q);
	domain d(&Fq);
	with w(&d);

#if 0

	matrix M2;
	M2 = M;
	for (i = 0; i < size; i++) {
		unipoly mue;
		M2.KX_module_order_ideal(i, mue, verbose_level - 1);
		cout << "order ideal " << i << ":" << endl;
		cout << mue << endl;
		}
#endif

	// This part uses DISCRETA data structures:

	discreta_matrix M1, P, Pv, Q, Qv, S, T;

	M.elements_to_unipoly();
	M.minus_X_times_id();
	M1 = M;
	cout << "M - x * Id has been computed" << endl;
	//cout << "M - x * Id =" << endl << M << endl;

	if (f_v) {
		cout << "M - x * Id = " << endl;
		cout << M << endl;
	}


	cout << "before M.smith_normal_form" << endl;
	M.smith_normal_form(P, Pv, Q, Qv, verbose_level);
	cout << "after M.smith_normal_form" << endl;

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


	unipoly charpoly;
	int deg;
	int l, lv, b, c;

	charpoly = M.s_ij(size - 1, size - 1);

	cout << "characteristic polynomial:" << charpoly << endl;
	deg = charpoly.degree();
	cout << "has degree " << deg << endl;
	l = charpoly.s_ii(deg);
	cout << "leading coefficient " << l << endl;
	lv = Fq.inverse(l);
	cout << "leading coefficient inverse " << lv << endl;
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		c = Fq.mult(b, lv);
		charpoly.m_ii(i, c);
	}
	cout << "monic characteristic polynomial:" << charpoly << endl;

	integer x, y;
	int *roots;
	int nb_roots = 0;

	roots = new int[q];

	for (a = 0; a < q; a++) {
		x.m_i(a);
		charpoly.evaluate_at(x, y);
		if (y.s_i() == 0) {
			cout << "root " << a << endl;
			roots[nb_roots++] = a;
		}
	}
	cout << "we found the following eigenvalues: ";
	int_vec_print(cout, roots, nb_roots);
	cout << endl;

	int eigenvalue, eigenvalue_negative;

	for (h = 0; h < nb_roots; h++) {
		eigenvalue = roots[h];
		cout << "looking at eigenvalue " << eigenvalue << endl;
		int *A, *B, *Bt;
		eigenvalue_negative = Fq.negate(eigenvalue);
		A = new int[size * size];
		B = new int[size * size];
		Bt = new int[size * size];
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				A[i * size + j] = Data[i * size + j];
			}
		}
		cout << "A:" << endl;
		print_integer_matrix_width(cout, A,
				size, size, size, Fq.log10_of_q);
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				a = A[i * size + j];
				if (j == i) {
					a = Fq.add(a, eigenvalue_negative);
				}
				B[i * size + j] = a;
			}
		}
		cout << "B = A - eigenvalue * I:" << endl;
		print_integer_matrix_width(cout, B,
				size, size, size, Fq.log10_of_q);

		cout << "B transposed:" << endl;
		Fq.transpose_matrix(B, Bt, size, size);
		print_integer_matrix_width(cout, Bt,
				size, size, size, Fq.log10_of_q);

		int f_special = FALSE;
		int f_complete = TRUE;
		int *base_cols;
		int nb_base_cols;
		int f_P = FALSE;
		int kernel_m, kernel_n, *kernel;

		base_cols = new int[size];
		kernel = new int[size * size];

		nb_base_cols = Fq.Gauss_int(Bt,
			f_special, f_complete, base_cols,
			f_P, NULL, size, size, size,
			verbose_level - 1);
		cout << "rank = " << nb_base_cols << endl;

		Fq.matrix_get_kernel(Bt, size, size, base_cols, nb_base_cols,
			kernel_m, kernel_n, kernel);
		cout << "kernel = left eigenvectors:" << endl;
		print_integer_matrix_width(cout, kernel,
				size, kernel_n, kernel_n, Fq.log10_of_q);

		int *vec1, *vec2;
		vec1 = new int[size];
		vec2 = new int[size];
		for (i = 0; i < size; i++) {
			vec1[i] = kernel[i * kernel_n + 0];
			}
		int_vec_print(cout, vec1, size);
		cout << endl;
		Fq.PG_element_normalize_from_front(vec1, 1, size);
		int_vec_print(cout, vec1, size);
		cout << endl;
		Fq.PG_element_rank_modified(vec1, 1, size, a);
		cout << "has rank " << a << endl;


		cout << "computing xA" << endl;

		Fq.mult_vector_from_the_left(vec1, A, vec2, size, size);
		int_vec_print(cout, vec2, size);
		cout << endl;
		Fq.PG_element_normalize_from_front(vec2, 1, size);
		int_vec_print(cout, vec2, size);
		cout << endl;
		Fq.PG_element_rank_modified(vec2, 1, size, a);
		cout << "has rank " << a << endl;

		delete [] vec1;
		delete [] vec2;

		delete [] A;
		delete [] B;
		delete [] Bt;
	}
}


// a5_in_PSL.cpp
//
// Anton Betten, Evi Haberberger
// 10.06.2000
//
// moved here from D2: 3/18/2010

void algebra_global_with_action::A5_in_PSL_(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int p, f;
	discreta_matrix A, B, D; //, B1, B2, C, D, A2, A3, A4;
	number_theory_domain NT;


	NT.factor_prime_power(q, p, f);
	domain *dom;

	if (f_v) {
		cout << "a5_in_psl.out: "
				"q=" << q << ", p=" << p << ", f=" << f << endl;
		}
	dom = allocate_finite_field_domain(q, verbose_level);

	A5_in_PSL_2_q(q, A, B, dom, verbose_level);

	{
	with w(dom);
	D.mult(A, B);

	if (f_v) {
		cout << "finished with A5_in_PSL_2_q()" << endl;
		cout << "A=\n" << A << endl;
		cout << "B=\n" << B << endl;
		cout << "AB=\n" << D << endl;
		int AA[4], BB[4], DD[4];
		matrix_convert_to_numerical(A, AA, q);
		matrix_convert_to_numerical(B, BB, q);
		matrix_convert_to_numerical(D, DD, q);
		cout << "A=" << endl;
		print_integer_matrix_width(cout, AA, 2, 2, 2, 7);
		cout << "B=" << endl;
		print_integer_matrix_width(cout, BB, 2, 2, 2, 7);
		cout << "AB=" << endl;
		print_integer_matrix_width(cout, DD, 2, 2, 2, 7);
		}

	int oA, oB, oD;

	oA = proj_order(A);
	oB = proj_order(B);
	oD = proj_order(D);
	if (f_v) {
		cout << "projective order of A = " << oA << endl;
		cout << "projective order of B = " << oB << endl;
		cout << "projective order of AB = " << oD << endl;
		}


	}
	free_finite_field_domain(dom);
}

void algebra_global_with_action::A5_in_PSL_2_q(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level)
{
	if (((q - 1) % 5) == 0) {
		A5_in_PSL_2_q_easy(q, A, B, dom_GFq, verbose_level);
		}
	else if (((q + 1) % 5) == 0) {
		A5_in_PSL_2_q_hard(q, A, B, dom_GFq, verbose_level);
		}
	else {
		cout << "either q + 1 or q - 1 must be divisible by 5!" << endl;
		exit(1);
		}
}

void algebra_global_with_action::A5_in_PSL_2_q_easy(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, r;
	integer zeta5, zeta5v, b, c, d, b2, e;

	if (f_v) {
		cout << "A5_in_PSL_2_q_easy verbose_level=" << verbose_level << endl;
		}
	with w(dom_GFq);

	i = (q - 1) / 5;
	r = finite_field_domain_primitive_root();
	zeta5.m_i(r);
	zeta5.power_int(i);
	zeta5v = zeta5;
	zeta5v.power_int(4);

	if (f_v) {
		cout << "zeta5=" << zeta5 << endl;
		cout << "zeta5v=" << zeta5v << endl;
		}

	A.m_mn_n(2, 2);
	B.m_mn_n(2, 2);
	A[0][0] = zeta5;
	A[0][1].zero();
	A[1][0].zero();
	A[1][1] = zeta5v;

	if (f_v) {
		cout << "A=\n" << A << endl;
		}

	// b := (zeta5 - zeta5^{-1})^{-1}:
	b = zeta5v;
	b.negate();
	b += zeta5;
	b.invert();

	// determine c, d such that $-b^2 -cd = 1$:
	b2 = b;
	b2 *= b;
	b2.negate();
	e.m_one();
	e += b2;
	c.one();
	d = e;
	B[0][0] = b;
	B[0][1] = c;
	B[1][0] = d;
	B[1][1] = b;
	B[1][1].negate();

	if (f_v) {
		cout << "B=\n" << B << endl;
		}
}


void algebra_global_with_action::A5_in_PSL_2_q_hard(int q,
		discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	with w(dom_GFq);
	unipoly m;
	int i, q2;
	discreta_matrix S, Sv, E, /*Sbart, SSbart,*/ AA, BB;
	integer a, b, m1;
	int norm_alpha, l;

#if 0
	m.get_an_irreducible_polynomial(2, verbose_level);
#else
	m.Singer(q, 2, verbose_level);
#endif
	cout << "m=" << m << endl;
	norm_alpha = m.s_ii(0);
	cout << "norm_alpha=" << norm_alpha << endl;

	domain GFq2(&m, dom_GFq);
	with ww(&GFq2);
	q2 = q * q;

	if (f_v) {
		cout << "searching for element of norm -1:" << endl;
		}
	S.m_mn_n(2, 2);
	m1.m_one();
	if (f_v) {
		cout << "-1=" << m1 << endl;
		}
#if 0
	for (i = q; i < q2; i++) {
		// cout << "i=" << i;
		a.m_i(i);
		b = a;
		b.power_int(q + 1);
		cout << i << ": (" << a << ")^" << q + 1 << " = " << b << endl;
		if (b.is_m_one())
			break;
		}
	if (i == q2) {
		cout << "A5_in_PSL_2_q_hard() couldn't find element of norm -1" << endl;
		exit(1);
		}
#else
	a.m_i(q); // alpha
	a.power_int((q - 1) >> 1);
	b = a;
	b.power_int(q + 1);
	cout << "(" << a << ")^" << q + 1 << " = " << b << endl;
	if (!b.is_m_one()) {
		cout << "fatal: element a does not have norm -1" << endl;
		exit(1);
		}
#endif
	if (f_v) {
		cout << "element of norm -1:" << a << endl;
		}
#if 1
	S[0][0] = a;
	S[0][1].one();
	S[1][0].one();
	S[1][0].negate();
	S[1][1] = a;
#else
	// Huppert I page 105 (does not work!)
	S[0][0].one();
	S[0][1] = a;
	S[1][0].one();
	S[1][1] = a;
	S[1][1].negate();
#endif
	if (f_v) {
		cout << "S=\n" << S << endl;
		}
	Sv = S;
	Sv.invert();
	E.mult(S, Sv);
	if (f_v) {
		cout << "S^{-1}=\n" << Sv << endl;
		cout << "S \\cdot S^{-1}=\n" << E << endl;
		}

#if 0
	Sbart = S;
	elementwise_power_int(Sbart, q);
	Sbart.transpose();
	SSbart.mult(S, Sbart);
	if (f_v) {
		cout << "\\bar{S}^\\top=\n" << Sbart << endl;
		cout << "S \\cdot \\bar{S}^\\top=\n" << SSbart << endl;
		}
#endif

	int r;
	integer zeta5, zeta5v;

	i = (q2 - 1) / 5;
	r = finite_field_domain_primitive_root();
	zeta5.m_i(r);
	zeta5.power_int(i);
	zeta5v = zeta5;
	zeta5v.power_int(4);

	if (f_v) {
		cout << "zeta5=" << zeta5 << endl;
		cout << "zeta5v=" << zeta5v << endl;
		}

	AA.m_mn_n(2, 2);
	BB.m_mn_n(2, 2);
	AA[0][0] = zeta5;
	AA[0][1].zero();
	AA[1][0].zero();
	AA[1][1] = zeta5v;

	if (f_v) {
		cout << "AA=\n" << AA << endl;
		}

	integer bb, c, d, e, f, c1, b1;

	// b := (zeta5 - zeta5^{-1})^{-1}:
	b = zeta5v;
	b.negate();
	b += zeta5;
	b.invert();

	if (f_v) {
		cout << "b=" << b << endl;
		}

	// compute $c$ with $N(c) = c \cdot \bar{c} = 1 - N(b) = 1 - b \cdot \bar{b}$:
	b1 = b;
	b1.power_int(q);

	bb.mult(b, b1);
	bb.negate();
	e.one();
	e += bb;
	if (f_v) {
		cout << "1 - b \\cdot \\bar{b}=" << e << endl;
		}
#if 1
	for (l = 0; l < q; l++) {
		c.m_i(norm_alpha);
		f = c;
		f.power_int(l);
		if (f.compare_with(e) == 0)
			break;
		}
	if (f_v) {
		cout << "the discrete log with respect to " << norm_alpha << " is " << l << endl;
		}
	c.m_i(q);
	c.power_int(l);

	f = c;
	f.power_int(q + 1);
	if (f.compare_with(e) != 0) {
		cout << "fatal: norm of " << c << " is not " << e << endl;
		exit(1);
		}
#else
	for (i = q; i < q2; i++) {
		c.m_i(i);
		f = c;
		f.power_int(q + 1);
		if (f.compare_with(e) == 0)
			break;
		}
	if (i == q2) {
		cout << "A5_in_PSL_2_q_hard() couldn't find element c" << endl;
		exit(1);
		}
#endif
	if (f_v) {
		cout << "element c=" << c << endl;
		}
	c1 = c;
	c1.power_int(q);

	BB[0][0] = b;
	BB[0][1] = c;
	BB[1][0] = c1;
	BB[1][0].negate();
	BB[1][1] = b1;
	if (f_v) {
		cout << "BB=\n" << BB << endl;
		}
	A.mult(S, AA);
	A *= Sv;
	B.mult(S, BB);
	B *= Sv;

	if (f_v) {
		cout << "A=\n" << A << endl;
		cout << "B=\n" << B << endl;
		}
}

int algebra_global_with_action::proj_order(discreta_matrix &A)
{
	discreta_matrix B;
	int m, n;
	int ord;

	m = A.s_m();
	n = A.s_n();
	if (m != n)
	{
		cout << "matrix::proj_order_mod() m != n" << endl;
		exit(1);
	}
	if (A.is_zero())
	{
		ord = 0;
		cout << "is zero matrix!" << endl;
	}
	else
	{
		B = A;
		ord = 1;
		while (is_in_center(B) == FALSE)
		{
			ord++;
			B *= A;
		}
	}
	return ord;
}

void algebra_global_with_action::trace(discreta_matrix &A, discreta_base &tr)
{
	int i, m, n;

	m = A.s_m();
	n = A.s_n();
	if (m != n)
	{
		cout << "ERROR: matrix::trace(): no square matrix!" << endl;
		exit(1);
	}
	tr = A[0][0];
	for (i = 1; i < m; i++)
	{
		tr += A[i][i];
	}
}

void algebra_global_with_action::elementwise_power_int(discreta_matrix &A, int k)
{
	int i, j, m, n;

	m = A.s_m();
	n = A.s_n();

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			A[i][j].power_int(k);
		}
	}
}

int algebra_global_with_action::is_in_center(discreta_matrix &B)
{
	int m, n, i, j;
	discreta_matrix A;
	integer c;

	m = B.s_m();
	n = B.s_n();
	A = B;
	c = A[0][0];
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			integer e;

			e = A[i][j];
			if (i != j && !e.is_zero())
			{
				return FALSE;
			}
			if (i == j && e.s_i() != c.s_i())
			{
				return FALSE;
			}
		}
	}
	return TRUE;
}


void algebra_global_with_action::matrix_convert_to_numerical(discreta_matrix &A, int *AA, int q)
{
	int m, n, i, j, /*h, l,*/ val;

	m = A.s_m();
	n = A.s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {

			//cout << "i=" << i << " j=" << j << endl;
			discreta_base a;

			A[i][j].copyobject_to(a);

			//cout << "a=" << a << endl;
			//a.printobjectkindln(cout);

			val = a.s_i_i();
#if 0
			l = a.as_unipoly().s_l();
			cout << "degree=" << l << endl;
			for (h = l - 1; h >= 0; h--) {
				val *= q;
				cout << "coeff=" << a.as_unipoly().s_ii(h) << endl;
				val += a.as_unipoly().s_ii(h);
				}
#endif
			//cout << "val=" << val << endl;
			AA[i * n + j] = val;
			}
		}
}


void algebra_global_with_action::classify_surfaces(
		finite_field *F, linear_group *LG,
		poset_classification_control *Control,
		surface_domain *&Surf, surface_with_action *&Surf_A,
		surface_classify_wedge *&SCW,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, verbose_level - 3);
	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces after Surf->init" << endl;
	}


	Surf_A = NEW_OBJECT(surface_with_action);


	int f_semilinear;

	f_semilinear = LG->A2->is_semilinear_matrix_group();

	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, f_semilinear, verbose_level - 3);
	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces after Surf_A->init" << endl;
	}



	SCW = NEW_OBJECT(surface_classify_wedge);

	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces before SCW->init" << endl;
	}

	SCW->init(F, LG,
			f_semilinear, Surf_A,
			Control,
			verbose_level - 1);

	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces after SCW->init" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces before SCW->do_classify_double_sixes" << endl;
	}
	SCW->do_classify_double_sixes(verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces after SCW->do_classify_double_sixes" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces before SCW->do_classify_surfaces" << endl;
	}
	SCW->do_classify_surfaces(verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces after SCW->do_classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::classify_surfaces done" << endl;
	}

}


void algebra_global_with_action::young_symmetrizer(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer" << endl;
	}

	young *Y;

	Y = NEW_OBJECT(young);

	Y->init(n, verbose_level);



	int *elt1, *elt2, *h_alpha, *elt4, *elt5, *elt6, *elt7;

	Y->group_ring_element_create(Y->A, Y->S, elt1);
	Y->group_ring_element_create(Y->A, Y->S, elt2);
	Y->group_ring_element_create(Y->A, Y->S, h_alpha);
	Y->group_ring_element_create(Y->A, Y->S, elt4);
	Y->group_ring_element_create(Y->A, Y->S, elt5);
	Y->group_ring_element_create(Y->A, Y->S, elt6);
	Y->group_ring_element_create(Y->A, Y->S, elt7);



	int *part;
	int *parts;

	int *Base;
	int *Base_inv;
	int *Fst;
	int *Len;
	int cnt, s, i, j;
	combinatorics_domain Combi;


	part = NEW_int(n);
	parts = NEW_int(n);
	Fst = NEW_int(Y->goi);
	Len = NEW_int(Y->goi);
	Base = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	Base_inv = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	s = 0;
	Fst[0] = 0;

		// create the first partition in exponential notation:
	Combi.partition_first(part, n);
	cnt = 0;


	while (TRUE) {
		int nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = n - 1; i >= 0; i--) {
			for (j = 0; j < part[i]; j++) {
				parts[nb_parts++] = i + 1;
				}
			}

		cout << "partition ";
		int_vec_print(cout, parts, nb_parts);
		cout << endl;


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		int *tableau;

		tableau = NEW_int(n);
		for (i = 0; i < n; i++) {
			tableau[i] = i;
			}
		Y->young_symmetrizer(parts, nb_parts, tableau, elt1, elt2, h_alpha, verbose_level);
		FREE_int(tableau);


		cout << "h_alpha =" << endl;
		Y->group_ring_element_print(Y->A, Y->S, h_alpha);
		cout << endl;


		Y->group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		Y->group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		cout << "h_alpha * h_alpha=" << endl;
		Y->group_ring_element_print(Y->A, Y->S, elt5);
		cout << endl;

		int *Module_Base;
		int *base_cols;
		int rk;


		Y->create_module(h_alpha,
			Module_Base, base_cols, rk,
			verbose_level);

		cout << "Module_Basis=" << endl;
		Y->D->print_matrix(Module_Base, rk, Y->goi);


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(Y->D->offset(Module_Base, i * Y->goi + j), Y->D->offset(Base, s * Y->goi + j), 0);
				}
			s++;
			}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_int(Module_Base);
		FREE_int(base_cols);


			// create the next partition in exponential notation:
		if (!Combi.partition_next(part, n)) {
			break;
			}
		cnt++;
		}

	cout << "Basis of submodule=" << endl;
	Y->D->print_matrix(Base, s, Y->goi);


	FREE_int(part);
	FREE_int(parts);
	FREE_int(Fst);
	FREE_int(Len);
	cout << "before freeing Base" << endl;
	FREE_int(Base);
	FREE_int(Base_inv);
	cout << "before freeing Y" << endl;
	FREE_OBJECT(Y);
	cout << "before freeing elt1" << endl;
	FREE_int(elt1);
	FREE_int(elt2);
	FREE_int(h_alpha);
	FREE_int(elt4);
	FREE_int(elt5);
	FREE_int(elt6);
	FREE_int(elt7);
	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer done" << endl;
	}
}

void algebra_global_with_action::young_symmetrizer_sym_4(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer_sym_4" << endl;
	}
	young *Y;
	int n = 4;

	Y = NEW_OBJECT(young);

	Y->init(n, verbose_level);



	int *elt1, *elt2, *h_alpha, *elt4, *elt5, *elt6, *elt7;

	Y->group_ring_element_create(Y->A, Y->S, elt1);
	Y->group_ring_element_create(Y->A, Y->S, elt2);
	Y->group_ring_element_create(Y->A, Y->S, h_alpha);
	Y->group_ring_element_create(Y->A, Y->S, elt4);
	Y->group_ring_element_create(Y->A, Y->S, elt5);
	Y->group_ring_element_create(Y->A, Y->S, elt6);
	Y->group_ring_element_create(Y->A, Y->S, elt7);



	int *part;
	int *parts;

	int *Base;
	int *Base_inv;
	int *Fst;
	int *Len;
	int cnt, s, i, j;

	part = NEW_int(n);
	parts = NEW_int(n);
	Fst = NEW_int(Y->goi);
	Len = NEW_int(Y->goi);
	Base = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	Base_inv = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	s = 0;
	Fst[0] = 0;

		// create the first partition in exponential notation:
	//partition_first(part, n);
	cnt = 0;

	int Part[10][5] = {
		{4, -1, 0, 0, 0},
		{3, 1, -1, 0, 0},
		{3, 1, -1, 0, 0},
		{3, 1, -1, 0, 0},
		{2, 2, -1, 0, 0},
		{2, 2, -1, 0, 0},
		{2, 1, 1, -1, 0},
		{2, 1, 1, -1, 0},
		{2, 1, 1, -1, 0},
		{1, 1, 1, 1, -1},
			};
	int Tableau[10][4] = {
		{0,1,2,3},
		{0,1,2,3}, {0,1,3,2}, {0,2,3,1},
		{0,1,2,3}, {0,2,1,3},
		{0,1,2,3}, {0,2,1,3}, {0,3,1,2},
		{0,1,2,3}
		};

	for(cnt = 0; cnt < 10; cnt++) {
		int nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = 0; i < 4; i++) {
			parts[nb_parts] = Part[cnt][i];
			if (parts[nb_parts] == -1) {
				break;
				}
			nb_parts++;
			}

		cout << "partition ";
		int_vec_print(cout, parts, nb_parts);
		cout << endl;


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		Y->young_symmetrizer(parts, nb_parts, Tableau[cnt], elt1, elt2, h_alpha, verbose_level);


		cout << "h_alpha =" << endl;
		Y->group_ring_element_print(Y->A, Y->S, h_alpha);
		cout << endl;


		Y->group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		Y->group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		cout << "h_alpha * h_alpha=" << endl;
		Y->group_ring_element_print(Y->A, Y->S, elt5);
		cout << endl;

		int *Module_Base;
		int *base_cols;
		int rk;


		Y->create_module(h_alpha,
			Module_Base, base_cols, rk,
			verbose_level);

		cout << "Module_Basis=" << endl;
		Y->D->print_matrix(Module_Base, rk, Y->goi);


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(Y->D->offset(Module_Base, i * Y->goi + j), Y->D->offset(Base, s * Y->goi + j), 0);
				}
			s++;
			}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_int(Module_Base);
		FREE_int(base_cols);


		}

	cout << "Basis of submodule=" << endl;
	//Y->D->print_matrix(Base, s, Y->goi);
	Y->D->print_matrix_for_maple(Base, s, Y->goi);

	FREE_int(part);
	FREE_int(parts);
	FREE_int(Fst);
	FREE_int(Len);
	cout << "before freeing Base" << endl;
	FREE_int(Base);
	FREE_int(Base_inv);
	cout << "before freeing Y" << endl;
	FREE_OBJECT(Y);
	cout << "before freeing elt1" << endl;
	FREE_int(elt1);
	FREE_int(elt2);
	FREE_int(h_alpha);
	FREE_int(elt4);
	FREE_int(elt5);
	FREE_int(elt6);
	FREE_int(elt7);
	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer_sym_4 done" << endl;
	}
}

void algebra_global_with_action::classify_surfaces_through_arcs_and_trihedral_pairs(
		surface_with_action *Surf_A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface_domain *Surf;
	finite_field *F;
	action *A;
	int i, j, arc_idx;

	char fname_arc_lifting[10000];


	if (f_v) {
		cout << "classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}
	F = Surf_A->F;
	q = F->q;
	Surf = Surf_A->Surf;

	A = NEW_OBJECT(action);

	vector_ge *nice_gens;

	int f_semilinear = TRUE;
	number_theory_domain NT;

	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}


	if (f_v) {
		cout << "before A->init_projective_group" << endl;
	}
	A->init_projective_group(3, F,
			f_semilinear,
			TRUE /*f_basis*/, TRUE /* f_init_sims */,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "after A->init_projective_group" << endl;
	}


	six_arcs_not_on_a_conic *Six_arcs;

	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);


	// classify six arcs not on a conic:

	if (f_v) {
		cout << "before Six_arcs->init" << endl;
	}
	Six_arcs->init(F,
			A,
			Surf->P2,
		//argc, argv,
		verbose_level - 2);
	if (f_v) {
		cout << "after Six_arcs->init" << endl;
	}



	if (f_v) {
		cout << "before report" << endl;
	}
	{
		char title[10000];
		char author[10000];
		sprintf(title, "Arc lifting over GF(%d) ", q);
		sprintf(author, "");

		sprintf(fname_arc_lifting, "arc_lifting_q%d.tex", q);
		ofstream fp(fname_arc_lifting);
		latex_interface L;


		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);


		if (f_v) {
			cout << "classify_surfaces_through_arcs_and_trihedral_pairs q=" << q << endl;
		}



		Six_arcs->report_latex(fp);

		if (f_v) {
			Surf->print_polynomial_domains(fp);
			Surf->print_line_labelling(fp);


			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"before Surf->print_Steiner_and_Eckardt" << endl;
			Surf->print_Steiner_and_Eckardt(fp);
			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"after Surf->print_Steiner_and_Eckardt" << endl;

			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"before Surf->print_clebsch_P" << endl;
			Surf->print_clebsch_P(fp);
			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"after Surf->print_clebsch_P" << endl;



			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"before Surf_A->list_orbits_on_trihedra_type1" << endl;
			Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type1(fp);

			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"before Surf_A->list_orbits_on_trihedra_type2" << endl;
			Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type2(fp);

			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"before Surf_A->print_trihedral_pairs no stabs" << endl;
			Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
					FALSE /* f_with_stabilizers */);

			cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
					"before Surf_A->print_trihedral_pairs with stabs" << endl;
			Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
					TRUE /* f_with_stabilizers */);

		}



		char fname_base[1000];
		sprintf(fname_base, "arcs_q%d", q);

		if (q < 20) {
			cout << "before Gen->gen->draw_poset_full" << endl;
			Six_arcs->Gen->gen->draw_poset(
				fname_base,
				6 /* depth */, 0 /* data */,
				TRUE /* f_embedded */,
				FALSE /* f_sideways */,
				verbose_level);
		}


		int *f_deleted; // [Six_arcs->nb_arcs_not_on_conic]
		int *Arc_identify; //[Six_arcs->nb_arcs_not_on_conic *
					// Six_arcs->nb_arcs_not_on_conic]
		int *Arc_identify_nb; // [Six_arcs->nb_arcs_not_on_conic]
		long int Arc6[6];
		int nb_surfaces;

		nb_surfaces = 0;

		f_deleted = NEW_int(Six_arcs->nb_arcs_not_on_conic);
		Arc_identify = NEW_int(Six_arcs->nb_arcs_not_on_conic *
				Six_arcs->nb_arcs_not_on_conic);
		Arc_identify_nb = NEW_int(Six_arcs->nb_arcs_not_on_conic);

		int_vec_zero(f_deleted, Six_arcs->nb_arcs_not_on_conic);
		int_vec_zero(Arc_identify_nb, Six_arcs->nb_arcs_not_on_conic);

		for (arc_idx = 0;
				arc_idx < Six_arcs->nb_arcs_not_on_conic;
				arc_idx++) {


			if (f_deleted[arc_idx]) {
				continue;
			}


			if (f_v) {
				cout << "classify_surfaces_through_arcs_and_trihedral_pairs extending arc "
						<< arc_idx << " / "
						<< Six_arcs->nb_arcs_not_on_conic << ":" << endl;
			}

			fp << "\\clearpage\n\\section{Extending arc " << arc_idx
					<< " / " << Six_arcs->nb_arcs_not_on_conic << "}" << endl;

			Six_arcs->Gen->gen->get_set_by_level(
					6 /* level */,
					Six_arcs->Not_on_conic_idx[arc_idx],
					Arc6);

			{
				set_and_stabilizer *The_arc;

				The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
						6 /* level */,
						Six_arcs->Not_on_conic_idx[arc_idx],
						0 /* verbose_level */);


				fp << "Arc " << arc_idx << " / "
						<< Six_arcs->nb_arcs_not_on_conic << " is: ";
				fp << "$$" << endl;
				//int_vec_print(fp, Arc6, 6);
				The_arc->print_set_tex(fp);
				fp << "$$" << endl;

				F->display_table_of_projective_points(fp,
					The_arc->data, 6, 3);


				fp << "The stabilizer is the following group:\\\\" << endl;
				The_arc->Strong_gens->print_generators_tex(fp);

				FREE_OBJECT(The_arc);
			}

			char arc_label[1000];
			char arc_label_short[1000];

			sprintf(arc_label, "%d / %d",
					arc_idx, Six_arcs->nb_arcs_not_on_conic);
			sprintf(arc_label_short, "Arc%d", arc_idx);

			if (f_v) {
				cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
						"before arc_lifting_and_classify_using_trihedral_pairs" << endl;
			}

			Surf_A->arc_lifting_and_classify_using_trihedral_pairs(
				TRUE /* f_log_fp */,
				fp,
				Arc6,
				arc_label,
				arc_label_short,
				nb_surfaces,
				Six_arcs,
				Arc_identify_nb,
				Arc_identify,
				f_deleted,
				verbose_level);

			if (f_v) {
				cout << "classify_surfaces_through_arcs_and_trihedral_pairs "
						"after arc_lifting_and_classify_using_trihedral_pairs" << endl;
			}



			nb_surfaces++;
		} // next arc_idx

		cout << "We found " << nb_surfaces << " surfaces" << endl;


		cout << "decomposition matrix:" << endl;
		for (i = 0; i < nb_surfaces; i++) {
			for (j = 0; j < Arc_identify_nb[i]; j++) {
				cout << Arc_identify[i * Six_arcs->nb_arcs_not_on_conic + j];
				if (j < Arc_identify_nb[i] - 1) {
					cout << ", ";
				}
			}
			cout << endl;
		}
		int *Decomp;
		int a;

		Decomp = NEW_int(Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
		int_vec_zero(Decomp, Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
		for (i = 0; i < nb_surfaces; i++) {
			for (j = 0; j < Arc_identify_nb[i]; j++) {
				a = Arc_identify[i * Six_arcs->nb_arcs_not_on_conic + j];
				Decomp[a * nb_surfaces + i]++;
			}
		}

		cout << "decomposition matrix:" << endl;
		cout << "$$" << endl;
		L.print_integer_matrix_with_standard_labels(cout, Decomp,
				Six_arcs->nb_arcs_not_on_conic, nb_surfaces,
				TRUE /* f_tex */);
		cout << "$$" << endl;

		fp << "Decomposition matrix:" << endl;
		//fp << "$$" << endl;
		//print_integer_matrix_with_standard_labels(fp, Decomp,
		//nb_arcs_not_on_conic, nb_surfaces, TRUE /* f_tex */);
		L.print_integer_matrix_tex_block_by_block(fp, Decomp,
				Six_arcs->nb_arcs_not_on_conic, nb_surfaces, 25);
		//fp << "$$" << endl;



		FREE_int(Decomp);
		FREE_int(f_deleted);
		FREE_int(Arc_identify);
		FREE_int(Arc_identify_nb);

		L.foot(fp);
	} // fp

	file_io Fio;

	cout << "Written file " << fname_arc_lifting << " of size "
			<< Fio.file_size(fname_arc_lifting) << endl;
	//delete Gen;
	//delete F;

	if (f_v) {
		cout << "classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}
}





}}
