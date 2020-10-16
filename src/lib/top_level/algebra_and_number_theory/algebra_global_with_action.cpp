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



void algebra_global_with_action::conjugacy_classes_based_on_normal_forms(action *A,
		sims *override_Sims,
		std::string &label,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string prefix;
	string fname_output;
	file_io Fio;
	int d;
	finite_field *F;


	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_classes_based_on_normal_forms" << endl;
	}

	prefix.assign(label);
	fname_output.assign(label);


	d = A->matrix_group_dimension();
	F = A->matrix_group_finite_field();

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_classes_based_on_normal_forms d=" << d << endl;
		cout << "algebra_global_with_action::conjugacy_classes_based_on_normal_forms q=" << F->q << endl;
	}

	gl_classes C;
	gl_class_rep *R;
	int nb_classes;
	int *Mtx;
	int *Elt;
	int i, order;
	long int a;

	char str[1000];

	sprintf(str, "_classes_based_on_normal_forms_%d_%d.tex", d, F->q);
	fname_output.append("_classes_normal_form.tex");

	C.init(d, F, verbose_level);

	if (f_v) {
		cout << "before C.make_classes" << endl;
	}
	C.make_classes(R, nb_classes, FALSE /*f_no_eigenvalue_one*/, verbose_level);
	if (f_v) {
		cout << "after C.make_classes" << endl;
	}

	Mtx = NEW_int(d * d + 1);
	Elt = NEW_int(A->elt_size_in_int);

	int *Order;

	Order = NEW_int(nb_classes);

	for (i = 0; i < nb_classes; i++) {

		if (f_v) {
			cout << "class " << i << " / " << nb_classes << ":" << endl;
		}

		int_vec_zero(Mtx, d * d + 1);
		C.make_matrix_from_class_rep(Mtx, R + i, verbose_level - 1);

		A->make_element(Elt, Mtx, 0);

		if (f_v) {
			cout << "before override_Sims->element_rank_lint" << endl;
		}
		a = override_Sims->element_rank_lint(Elt);
		if (f_v) {
			cout << "after override_Sims->element_rank_lint" << endl;
		}

		cout << "Representative of class " << i << " / "
				<< nb_classes << " has rank " << a << "\\\\" << endl;
		int_matrix_print(Elt, d, d);

		if (f_v) {
			cout << "before C.print_matrix_and_centralizer_order_latex" << endl;
		}
		C.print_matrix_and_centralizer_order_latex(
				cout, R + i);
		if (f_v) {
			cout << "after C.print_matrix_and_centralizer_order_latex" << endl;
		}

		if (f_v) {
			cout << "before A->element_order" << endl;
		}
		order = A->element_order(Elt);
		if (f_v) {
			cout << "after A->element_order" << endl;
		}

		cout << "The element order is : " << order << "\\\\" << endl;

		Order[i] = order;

	}

	tally T_order;

	T_order.init(Order, nb_classes, FALSE, 0);


	{
		ofstream ost(fname_output);
		latex_interface L;

		L.head_easy(ost);
		//C.report(fp, verbose_level);


		ost << "The distribution of element orders is:" << endl;
#if 0
		ost << "$$" << endl;
		T_order.print_file_tex_we_are_in_math_mode(ost, FALSE /* f_backwards */);
		ost << "$$" << endl;
#endif

		//ost << "$" << endl;
		T_order.print_file_tex(ost, FALSE /* f_backwards */);
		ost << "\\\\" << endl;

		ost << "$$" << endl;
		T_order.print_array_tex(ost, FALSE /* f_backwards */);
		ost << "$$" << endl;



		int t, f, l, a, h, c;

		for (t = 0; t < T_order.nb_types; t++) {
			f = T_order.type_first[t];
			l = T_order.type_len[t];
			a = T_order.data_sorted[f];

			if (f_v) {
				cout << "class type " << t << " / " << T_order.nb_types << ":" << endl;
			}

			ost << "\\section{The Classes of Elements of Order $" << a << "$}" << endl;


			ost << "There are " << l << " classes of elements of order " << a << "\\\\" << endl;

			for (h = 0; h < l; h++) {

				c = f + h;

				i = T_order.sorting_perm_inv[c];

				if (f_v) {
					cout << "class " << h << " / " << l << " of elements of order " << a << ":" << endl;
				}

				int_vec_zero(Mtx, d * d + 1);
				C.make_matrix_from_class_rep(Mtx, R + i, verbose_level - 1);

				A->make_element(Elt, Mtx, 0);

				if (f_v) {
					cout << "before override_Sims->element_rank_lint" << endl;
				}
				a = override_Sims->element_rank_lint(Elt);
				if (f_v) {
					cout << "after override_Sims->element_rank_lint" << endl;
				}

				ost << "Representative of class " << i << " / "
						<< nb_classes << " has rank " << a << "\\\\" << endl;
				int_matrix_print(Elt, d, d);

				if (f_v) {
					cout << "before C.print_matrix_and_centralizer_order_latex" << endl;
				}
				C.print_matrix_and_centralizer_order_latex(ost, R + i);
				if (f_v) {
					cout << "after C.print_matrix_and_centralizer_order_latex" << endl;
				}

				if (f_v) {
					cout << "before A->element_order" << endl;
				}
				order = A->element_order(Elt);
				if (f_v) {
					cout << "after A->element_order" << endl;
				}

				ost << "The element order is : " << order << "\\\\" << endl;


			}

		}
		L.foot(ost);
	}
	cout << "Written file " << fname_output << " of size "
			<< Fio.file_size(fname_output) << endl;

	FREE_int(Mtx);
	FREE_int(Elt);
	FREE_OBJECTS(R);

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_classes_based_on_normal_forms done" << endl;
	}
}



void algebra_global_with_action::classes_GL(int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
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


void algebra_global_with_action::group_table(int q, int d, int f_poly, std::string &poly,
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

	Cent = A->create_sims_from_generators_with_target_group_order_lint(
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
			kernel_m, kernel_n, kernel, 0 /* verbose_level */);
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
	if (m != n) {
		cout << "matrix::proj_order_mod m != n" << endl;
		exit(1);
	}
	if (A.is_zero()) {
		ord = 0;
		cout << "is zero matrix!" << endl;
	}
	else {
		B = A;
		ord = 1;
		while (is_in_center(B) == FALSE) {
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
	if (m != n) {
		cout << "ERROR: matrix::trace not a square matrix!" << endl;
		exit(1);
	}
	tr = A[0][0];
	for (i = 1; i < m; i++) {
		tr += A[i][i];
	}
}

void algebra_global_with_action::elementwise_power_int(discreta_matrix &A, int k)
{
	int i, j, m, n;

	m = A.s_m();
	n = A.s_n();

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
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
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			integer e;

			e = A[i][j];
			if (i != j && !e.is_zero()) {
				return FALSE;
			}
			if (i == j && e.s_i() != c.s_i()) {
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
	Surf_A->init(Surf, LG, verbose_level - 3);
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



void algebra_global_with_action::report_tactical_decomposition_by_automorphism_group(
		ostream &ost, projective_space *P,
		action *A_on_points, action *A_on_lines,
		strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::report_tactical_decomposition_by_automorphism_group" << endl;
	}
	int *Mtx;
	int i, j, h;
	incidence_structure *Inc;
	Inc = NEW_OBJECT(incidence_structure);

	Mtx = NEW_int(P->N_points * P->N_lines);
	int_vec_zero(Mtx, P->N_points * P->N_lines);

	for (j = 0; j < P->N_lines; j++) {
		for (h = 0; h < P->k; h++) {
			i = P->Lines[j * P->k + h];
			Mtx[i * P->N_lines + j] = 1;
		}
	}

	Inc->init_by_matrix(P->N_points, P->N_lines, Mtx, 0 /* verbose_level*/);


	partitionstack S;

	int N;

	if (f_v) {
		cout << "algebra_global_with_action::report_tactical_decomposition_by_automorphism_group "
				"allocating partitionstack" << endl;
	}
	N = Inc->nb_points() + Inc->nb_lines();

	S.allocate(N, 0);
	// split off the column class:
	S.subset_continguous(Inc->nb_points(), Inc->nb_lines());
	S.split_cell(0);

	#if 0
	// ToDo:
	S.split_cell_front_or_back(data, target_size,
			TRUE /* f_front */, 0 /* verbose_level*/);
	#endif


	int TDO_depth = N;
	//int TDO_ht;


	if (f_v) {
		cout << "algebra_global_with_action::report_tactical_decomposition_by_automorphism_group "
				"before Inc->compute_TDO_safe" << endl;
	}
	Inc->compute_TDO_safe(S, TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;


	if (S.ht < size_limit_for_printing) {
		ost << "The TDO decomposition is" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				ost, TRUE /* f_enter_math */,
				TRUE /* f_print_subscripts */, S);
	}
	else {
		ost << "The TDO decomposition is very large (with "
				<< S.ht<< " classes).\\\\" << endl;
	}


	{
		schreier *Sch_points;
		schreier *Sch_lines;
		Sch_points = NEW_OBJECT(schreier);
		Sch_points->init(A_on_points, verbose_level - 2);
		Sch_points->initialize_tables();
		Sch_points->init_generators(*gens->gens /* *generators */, verbose_level - 2);
		Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "found " << Sch_points->nb_orbits
					<< " orbits on points" << endl;
		}
		Sch_lines = NEW_OBJECT(schreier);
		Sch_lines->init(A_on_lines, verbose_level - 2);
		Sch_lines->initialize_tables();
		Sch_lines->init_generators(*gens->gens /* *generators */, verbose_level - 2);
		Sch_lines->compute_all_point_orbits(0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "found " << Sch_lines->nb_orbits
					<< " orbits on lines" << endl;
		}
		S.split_by_orbit_partition(Sch_points->nb_orbits,
			Sch_points->orbit_first, Sch_points->orbit_len, Sch_points->orbit,
			0 /* offset */,
			verbose_level - 2);
		S.split_by_orbit_partition(Sch_lines->nb_orbits,
			Sch_lines->orbit_first, Sch_lines->orbit_len, Sch_lines->orbit,
			Inc->nb_points() /* offset */,
			verbose_level - 2);
		FREE_OBJECT(Sch_points);
		FREE_OBJECT(Sch_lines);
	}

	if (S.ht < size_limit_for_printing) {
		ost << "The TDA decomposition is" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				ost, TRUE /* f_enter_math */,
				TRUE /* f_print_subscripts */, S);
	}
	else {
		ost << "The TDA decomposition is very large (with "
				<< S.ht<< " classes).\\\\" << endl;
	}

	FREE_int(Mtx);
	FREE_OBJECT(gens);
	FREE_OBJECT(Inc);

	if (f_v) {
		cout << "algebra_global_with_action::report_tactical_decomposition_by_automorphism_group done" << endl;
	}
}

void algebra_global_with_action::linear_codes_with_bounded_minimum_distance(
		poset_classification_control *Control, linear_group *LG,
		int d, int target_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance" << endl;
	}

	poset *Poset;
	poset_classification *PC;


	Control->f_depth = TRUE;
	Control->depth = target_depth;


	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance group set up, "
				"calling gen->init" << endl;
		cout << "LG->A2->A->f_has_strong_generators="
				<< LG->A2->f_has_strong_generators << endl;
	}

	Poset = NEW_OBJECT(poset);

	Poset->init_subset_lattice(LG->A_linear, LG->A_linear,
			LG->Strong_gens,
			verbose_level);


	int independence_value = d - 1;

	Poset->add_independence_condition(
			independence_value,
			verbose_level);

#if 0
	Poset->f_print_function = FALSE;
	Poset->print_function = print_code;
	Poset->print_function_data = this;
#endif

	PC = NEW_OBJECT(poset_classification);
	PC->initialize_and_allocate_root_node(Control, Poset,
			target_depth, verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance before gen->main" << endl;
	}

	int t0;
	os_interface Os;
	int depth;

	t0 = Os.os_ticks();
	depth = PC->main(t0,
			target_depth /*schreier_depth*/,
		TRUE /*f_use_invariant_subset_if_available*/,
		FALSE /*f_debug */,
		verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance depth = " << depth << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance done" << endl;
	}
}

void algebra_global_with_action::packing_init(
		poset_classification_control *Control, linear_group *LG,
		int dimension_of_spread_elements,
		int f_select_spread, std::string &select_spread_text,
		std::string &path_to_spread_tables,
		packing_classify *&P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::packing_init "
				"dimension_of_spread_elements=" << dimension_of_spread_elements << endl;
	}
	action *A;
	int n, q;
	matrix_group *Mtx;
	spread_classify *T;


	A = LG->A2;
	n = A->matrix_group_dimension();
	Mtx = A->get_matrix_group();
	q = Mtx->GFq->q;
	if (f_v) {
		cout << "algebra_global_with_action::packing_init n=" << n
				<< " k=" << dimension_of_spread_elements << " q=" << q << endl;
	}


	T = NEW_OBJECT(spread_classify);


	if (f_v) {
		cout << "algebra_global_with_action::packing_init before T->init" << endl;
	}


	T->init(LG, dimension_of_spread_elements, Control, verbose_level - 1);

	if (f_v) {
		cout << "algebra_global_with_action::packing_init after T->init" << endl;
	}


	spread_table_with_selection *Spread_table_with_selection;

	Spread_table_with_selection = NEW_OBJECT(spread_table_with_selection);

	if (f_v) {
		cout << "algebra_global_with_action::packing_init "
				"before Spread_table_with_selection->init" << endl;
	}
	Spread_table_with_selection->init(T,
		f_select_spread,
		select_spread_text,
		path_to_spread_tables,
		verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::packing_init "
				"after Spread_table_with_selection->init" << endl;
	}



	P = NEW_OBJECT(packing_classify);


	if (f_v) {
		cout << "algebra_global_with_action::packing_init before P->init" << endl;
	}
	P->init(Spread_table_with_selection,
		TRUE, // ECA->f_lex,
		verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::packing_init after P->init" << endl;
	}

#if 0
	cout << "before IA->init" << endl;
	IA->init(T->A, P->A_on_spreads, P->gen,
		P->size_of_packing, P->prefix_with_directory, ECA,
		callback_packing_report,
		NULL /*callback_subset_orbits*/,
		P,
		verbose_level);
	cout << "after IA->init" << endl;
#endif

	if (f_v) {
		cout << "algebra_global_with_action::packing_init before P->compute_spread_table" << endl;
	}
	P->Spread_table_with_selection->compute_spread_table(verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::packing_init after P->compute_spread_table" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::packing_init done" << endl;
	}


}

void algebra_global_with_action::centralizer_of_element(
		action *A, sims *S,
		const char *element_description,
		const char *label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element label=" << label
				<< " element_description=" << element_description << endl;
	}

	prefix.assign("element_");
	prefix.append(label);

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}
#if 0
	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element Matrix:" << endl;
		int_matrix_print(data, 4, 4);
	}
#endif

	A->make_element(Elt, data, 0 /* verbose_level */);

	int o;

	o = A->element_order(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element Elt:" << endl;
		A->element_print_quick(Elt, cout);
		cout << "algebra_global_with_action::centralizer_of_element on points:" << endl;
		A->element_print_as_permutation(Elt, cout);
		//cout << "algebra_global_with_action::centralizer_of_element on lines:" << endl;
		//A2->element_print_as_permutation(Elt, cout);
	}

	cout << "algebra_global_with_action::centralizer_of_element "
			"the element has order " << o << endl;



	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element "
				"before centralizer_using_MAGMA" << endl;
	}

	strong_generators *gens;

	A->centralizer_using_MAGMA(prefix,
			S, Elt, gens, verbose_level);


	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element "
				"after centralizer_using_MAGMA" << endl;
	}


	cout << "generators for the centralizer are:" << endl;
	gens->print_generators_tex();


	FREE_int(data);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element done" << endl;
	}
}

void algebra_global_with_action::normalizer_of_cyclic_subgroup(
		action *A, sims *S,
		const char *element_description,
		const char *label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup label=" << label
				<< " element_description=" << element_description << endl;
	}

	prefix.assign("element_");
	prefix.append(label);

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}
#if 0
	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup Matrix:" << endl;
		int_matrix_print(data, 4, 4);
	}
#endif

	A->make_element(Elt, data, 0 /* verbose_level */);

	int o;

	o = A->element_order(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup Elt:" << endl;
		A->element_print_quick(Elt, cout);
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup on points:" << endl;
		A->element_print_as_permutation(Elt, cout);
		//cout << "algebra_global_with_action::centralizer_of_element on lines:" << endl;
		//A2->element_print_as_permutation(Elt, cout);
	}

	cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
			"the element has order " << o << endl;



	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
				"before normalizer_of_cyclic_group_using_MAGMA" << endl;
	}

	strong_generators *gens;

	A->normalizer_of_cyclic_group_using_MAGMA(prefix,
			S, Elt, gens, verbose_level);



	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
				"after normalizer_of_cyclic_group_using_MAGMA" << endl;
	}


	cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
			"generators for the normalizer are:" << endl;
	gens->print_generators_tex();


	FREE_int(data);
	FREE_int(Elt);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup done" << endl;
	}
}

void algebra_global_with_action::find_subgroups(
		action *A, sims *S,
		int subgroup_order,
		std::string &label,
		int &nb_subgroups,
		strong_generators *&H_gens,
		strong_generators *&N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string prefix;
	char str[1000];

	if (f_v) {
		cout << "algebra_global_with_action::find_subgroups label=" << label
				<< " subgroup_order=" << subgroup_order << endl;
	}
	prefix.assign(label);
	sprintf(str, "_find_subgroup_of_order_%d", subgroup_order);
	prefix.append(str);



	if (f_v) {
		cout << "algebra_global_with_action::find_subgroups "
				"before find_subgroup_using_MAGMA" << endl;
	}


	A->find_subgroups_using_MAGMA(prefix,
			S, subgroup_order,
			nb_subgroups, H_gens, N_gens, verbose_level);


	if (f_v) {
		cout << "algebra_global_with_action::find_subgroups "
				"after find_subgroup_using_MAGMA" << endl;
	}


	//cout << "generators for the subgroup are:" << endl;
	//gens->print_generators_tex();


	if (f_v) {
		cout << "algebra_global_with_action::find_subgroups done" << endl;
	}
}


void algebra_global_with_action::relative_order_vector_of_cosets(
		action *A, strong_generators *SG,
		vector_ge *cosets, int *&relative_order_table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	int *Elt2;
	//int *Elt3;
	sims *S;
	int i, drop_out_level, image, order;

	if (f_v) {
		cout << "algebra_global_with_action::relative_order_vector_of_cosets" << endl;
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	//Elt3 = NEW_int(A->elt_size_in_int);

	relative_order_table = NEW_int(cosets->len);

	S = SG->create_sims(0 /*verbose_level */);
	for (i = 0; i < cosets->len; i++) {
		A->element_move(cosets->ith(i), Elt1, 0);
		order = 1;
		while (TRUE) {
			if (S->strip(Elt1, Elt2, drop_out_level, image, 0 /*verbose_level*/)) {
				break;
			}
			A->element_mult(cosets->ith(i), Elt1, Elt2, 0);
			A->element_move(Elt2, Elt1, 0);
			order++;
		}
		relative_order_table[i] = order;
	}


	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "algebra_global_with_action::relative_order_vector_of_cosets done" << endl;
	}
}

void algebra_global_with_action::orbits_on_polynomials(
		linear_group *LG,
		int degree_of_poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_stabilizer = TRUE;
	//int f_draw_tree = TRUE;


	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_polynomials" << endl;
	}
	//linear_group_description *Descr;
	//linear_group *LG;

#if 0
	int f_linear = FALSE;
	int q;
	int f_degree = FALSE;
	int degree = 0;
	int nb_identify = 0;
	const char *Identify_label[1000];
	const char *Identify_coeff_text[1000];
	int f_stabilizer = FALSE;
	int f_draw_tree = FALSE;
	int f_test_orbit = FALSE;
	int test_orbit_idx = 0;


	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-degree") == 0) {
			f_degree = TRUE;
			degree = atoi(argv[++i]);
			cout << "-degree " << degree << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = new linear_group_description;
			i += Descr->read_arguments(argc - (i - 1),
					argv + i, verbose_level);

			cout << "after Descr->read_arguments" << endl;
			}
		else if (strcmp(argv[i], "-identify") == 0) {
			Identify_label[nb_identify] = argv[++i];
			Identify_coeff_text[nb_identify] = argv[++i];
			cout << "-identify " << Identify_label[nb_identify]
				<< " " << Identify_coeff_text[nb_identify] << endl;
			nb_identify++;

			}
		else if (strcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			cout << "-stabilizer " << endl;
			}
		else if (strcmp(argv[i], "-draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-draw_tree " << endl;
			}
		else if (strcmp(argv[i], "-test_orbit") == 0) {
			f_test_orbit = TRUE;
			test_orbit_idx = atoi(argv[++i]);
			cout << "-test_orbit " << test_orbit_idx << endl;
			}
		}

	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}
	if (!f_degree) {
		cout << "please use option -degree <degree>" << endl;
		exit(1);
		}

	//int f_v = (verbose_level >= 1);


	F = NEW_OBJECT(finite_field);
	F->init(Descr->input_q, 0);

	Descr->F = F;
	q = Descr->input_q;



	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "before LG->init, creating the group" << endl;
		}

	LG->init(Descr, verbose_level);

	cout << "after LG->init" << endl;
#endif

	finite_field *F;
	action *A;
	//matrix_group *M;
	int n;
	//int degree;

	A = LG->A_linear;
	F = A->matrix_group_finite_field();

	n = A->matrix_group_dimension();

	cout << "n = " << n << endl;

	cout << "strong generators:" << endl;
	//A->Strong_gens->print_generators();
	A->Strong_gens->print_generators_tex();

	homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);


	monomial_ordering_type Monomial_ordering_type = t_PART;


	HPD->init(F, n /* nb_var */, degree_of_poly,
			TRUE /* f_init_incidence_structure */,
			Monomial_ordering_type,
			verbose_level);

	action *A2;

	A2 = NEW_OBJECT(action);
	A2->induced_action_on_homogeneous_polynomials(A,
		HPD,
		FALSE /* f_induce_action */, NULL,
		verbose_level);

	cout << "created action A2" << endl;
	A2->print_info();


	int *Elt1;
	int *Elt2;
	int *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);



#if 0
	action_on_homogeneous_polynomials *OnHP;

	OnHP = A2->G.OnHP;
#endif
	schreier *Sch;
	longinteger_object full_go;

	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);

	cout << "computing orbits:" << endl;

	Sch = A->Strong_gens->orbits_on_points_schreier(A2, verbose_level);

	orbit_transversal *T;

	A->group_order(full_go);
	T = NEW_OBJECT(orbit_transversal);

	cout << "before T->init_from_schreier" << endl;

	T->init_from_schreier(
			Sch,
			A,
			full_go,
			verbose_level);

	cout << "after T->init_from_schreier" << endl;

	Sch->print_orbit_reps(cout);

	{
	int pt = 814083;
	if (Sch->degree >= pt) {
		int orbit_idx;
		orbit_idx = Sch->orbit_number(pt);
		cout << "point " << pt << " lies in orbit " << orbit_idx << endl;
	}
	}
	{
	int pt = 577680;
	if (Sch->degree >= pt) {
		int orbit_idx;
		orbit_idx = Sch->orbit_number(pt);
		cout << "point " << pt << " lies in orbit " << orbit_idx << endl;
	}
	}
	{
	int pt = 288258;
	if (Sch->degree >= pt) {
		int orbit_idx;
		orbit_idx = Sch->orbit_number(pt);
		cout << "point " << pt << " lies in orbit " << orbit_idx << endl;
	}
	}
	{
	int pt = 308226;
	if (Sch->degree >= pt) {
		int orbit_idx;
		orbit_idx = Sch->orbit_number(pt);
		cout << "point " << pt << " lies in orbit " << orbit_idx << endl;
	}
	}
	{
	int pt = 271362;
	if (Sch->degree >= pt) {
		int orbit_idx;
		orbit_idx = Sch->orbit_number(pt);
		cout << "point " << pt << " lies in orbit " << orbit_idx << endl;
	}
	}

	cout << "orbit reps:" << endl;

	char fname[1000];
	char title[1000];
	char author[1000];

	sprintf(fname, "poly_orbits_d%d_n%d_q%d.tex", degree_of_poly, n - 1, F->q);
	sprintf(title, "Varieties of degree %d in PG(%d,%d)", degree_of_poly, n - 1, F->q);
	sprintf(author, "Orbiter");
	{
		ofstream f(fname);
		latex_interface L;

		L.head(f,
				FALSE /* f_book*/,
				TRUE /* f_title */,
				title, author,
				FALSE /* f_toc */,
				FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extra_praeamble */);

		f << "\\small" << endl;
		f << "\\arraycolsep=2pt" << endl;
		f << "\\parindent=0pt" << endl;
		f << "$q = " << F->q << "$\\\\" << endl;
		f << "$n = " << n - 1 << "$\\\\" << endl;
		f << "degree of poly $ = " << degree_of_poly << "$\\\\" << endl;

		f << "\\clearpage" << endl << endl;
		f << "\\section{The Varieties of degree $" << degree_of_poly
				<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, summary}" << endl;

#if 0
		T->print_table_latex(
				f,
				TRUE /* f_has_callback */,
				polynomial_orbits_callback_print_function2,
				HPD /* callback_data */,
				TRUE /* f_has_callback */,
				polynomial_orbits_callback_print_function,
				HPD /* callback_data */,
				verbose_level);
#else
		int *coeff;
		int i;
		int *Nb_pts;

		coeff = NEW_int(HPD->get_nb_monomials());
		Nb_pts = NEW_int(T->nb_orbits);

		f << "orbit : rep : go : nb pts : poly\\\\" << endl;
		for (i = 0; i < T->nb_orbits; i++) {

			longinteger_object go;
			T->Reps[i].Strong_gens->group_order(go);

			f << i << " : ";
			lint_vec_print(f, T->Reps[i].data, T->Reps[i].sz);
			f << " : ";
			f << go;

			f << " : ";

			HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

			vector<long int> Points;
			int nb_pts;

			HPD->enumerate_points(coeff, Points, verbose_level);

			nb_pts = Points.size();
			Nb_pts[i] = nb_pts;

			f << nb_pts;
			f << " : ";

			f << T->Reps[i].data[0] << "=$";
			HPD->print_equation_tex(f, coeff);
			//int_vec_print(f, coeff, HPD->get_nb_monomials());
			//cout << " = ";
			//HPD->print_equation_str(ost, coeff);

			//f << " & ";
			//Reps[i].Strong_gens->print_generators_tex(f);
			f << "$";
			f << "\\\\" << endl;
		}

#endif

		FREE_int(coeff);


		tally T1;

		T1.init(Nb_pts, T->nb_orbits, FALSE, 0);
		f << "Distribution of the number of points: $";
		T1.print_naked_tex(f, TRUE);
		f << "$\\\\" << endl;

		f << "\\section{The Varieties of degree $" << degree_of_poly
				<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, "
						"detailed listing}" << endl;
		{
			int fst, l, a, r;
			longinteger_object go, go1;
			longinteger_domain D;
			int *coeff;
			int *line_type;
			long int *Pts;
			int nb_pts;
			int *Kernel;
			int *v;
			int i;
			//int h, pt, orbit_idx;

			A->group_order(go);
			Pts = NEW_lint(HPD->get_P()->N_points);
			coeff = NEW_int(HPD->get_nb_monomials());
			line_type = NEW_int(HPD->get_P()->N_lines);
			Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
			v = NEW_int(n);

			for (i = 0; i < Sch->nb_orbits; i++) {
				f << "\\subsection*{Orbit " << i << " / "
						<< Sch->nb_orbits << "}" << endl;
				fst = Sch->orbit_first[i];
				l = Sch->orbit_len[i];

				D.integral_division_by_int(go, l, go1, r);
				a = Sch->orbit[fst];
				HPD->unrank_coeff_vector(coeff, a);


				vector<long int> Points;

				HPD->enumerate_points(coeff, Points, verbose_level);

				nb_pts = Points.size();
				Pts = NEW_lint(nb_pts);
				for (int u = 0; u < nb_pts; u++) {
					Pts[u] = Points[u];
				}

				f << "stab order " << go1 << "\\\\" << endl;
				f << "orbit length = " << l << "\\\\" << endl;
				f << "orbit rep = " << a << "\\\\" << endl;
				f << "number of points = " << nb_pts << "\\\\" << endl;

				f << "$";
				int_vec_print(f, coeff, HPD->get_nb_monomials());
				f << " = ";
				HPD->print_equation(f, coeff);
				f << "$\\\\" << endl;


				cout << "We found " << nb_pts << " points on the curve" << endl;
				cout << "They are : ";
				lint_vec_print(cout, Pts, nb_pts);
				cout << endl;
				HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);

				F->display_table_of_projective_points(
					f, Pts, nb_pts, n);

				HPD->get_P()->line_intersection_type(Pts, nb_pts,
						line_type, 0 /* verbose_level */);

				f << "The line type is: ";

				stringstream sstr;
				int_vec_print_classified_str(sstr,
						line_type, HPD->get_P()->N_lines, TRUE /* f_backwards*/);
				string s = sstr.str();
				f << "$" << s << "$\\\\" << endl;
				//int_vec_print_classified(line_type, HPD->P->N_lines);
				//cout << "after int_vec_print_classified" << endl;

				f << "The stabilizer is generated by:" << endl;
				T->Reps[i].Strong_gens->print_generators_tex(f);
			} // next i

			FREE_lint(Pts);
			FREE_int(coeff);
			FREE_int(line_type);
			FREE_int(Kernel);
			FREE_int(v);
			}
		L.foot(f);

	}
	file_io Fio;

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


#if 0
	int f, l, a, r;
	longinteger_object go, go1;
	longinteger_domain D;
	int *coeff;
	int *line_type;
	long int *Pts;
	int nb_pts;
	int *Kernel;
	//int h, pt, orbit_idx;
	sorting Sorting;
	int i;


	Pts = NEW_lint(HPD->get_P()->N_points);
	coeff = NEW_int(HPD->get_nb_monomials());
	line_type = NEW_int(HPD->get_P()->N_lines);
	Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());


	A->group_order(go);

	for (i = 0; i < Sch->nb_orbits; i++) {
		cout << "Orbit " << i << " / " << Sch->nb_orbits << ":" << endl;
		f = Sch->orbit_first[i];
		l = Sch->orbit_len[i];

		D.integral_division_by_int(go, l, go1, r);

		cout << "stab order " << go1 << endl;
		a = Sch->orbit[f];
		cout << "orbit length = " << l << endl;
		cout << "orbit rep = " << a << endl;

		HPD->unrank_coeff_vector(coeff, a);
		int_vec_print(cout, coeff, HPD->get_nb_monomials());
		cout << " = ";
		HPD->print_equation(cout, coeff);
		cout << endl;


		vector<long int> Points;

		HPD->enumerate_points(coeff, Points, verbose_level);

		nb_pts = Points.size();
		Pts = NEW_lint(nb_pts);
		for (int u = 0; u < nb_pts; u++) {
			Pts[u] = Points[u];
		}


		cout << "We found " << nb_pts << " points on the curve" << endl;
		cout << "They are : ";
		lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);

		HPD->get_P()->line_intersection_type(Pts, nb_pts,
				line_type, 0 /* verbose_level */);

		cout << "The line type is: ";
		int_vec_print_classified(line_type, HPD->get_P()->N_lines);
		cout << "after int_vec_print_classified" << endl;

		cout << "The stabilizer is generated by:" << endl;
		T->Reps[i].Strong_gens->print_generators_tex(cout);


#if 0
		HPD->vanishing_ideal(Pts, nb_pts, r, Kernel, 0 /*verbose_level */);

		cout << "The system has rank " << r << endl;
		cout << "The ideal has dimension " << HPD->nb_monomials - r << endl;
		cout << "and is generated by:" << endl;
		int_matrix_print(Kernel, HPD->nb_monomials - r, HPD->nb_monomials);
		cout << "corresponding to the following basis of polynomials:" << endl;
		for (h = 0; h < HPD->nb_monomials - r; h++) {
			HPD->print_equation(cout, Kernel + h * HPD->nb_monomials);
			cout << endl;
			}
#endif



		if (f_stabilizer) {
			strong_generators *SG;
			longinteger_object stab_go;
			int canonical_pt;

			cout << "before set_stabilizer_in_projective_space" << endl;
			SG = A->set_stabilizer_in_projective_space(
				HPD->get_P(),
				Pts, nb_pts,
				canonical_pt, NULL,
				//FALSE, NULL,
				0 /*verbose_level */);




			SG->group_order(stab_go);

			cout << "stabilizer order = " << stab_go << endl;
			cout << "stabilizer generators:" << endl;
			SG->print_generators_tex();
			FREE_OBJECT(SG);
			}


		if (f_draw_tree) {
			{

			char label[1000];
			char fname[1000];
			int xmax = 1000000;
			int ymax =  500000;
#if 0
			int f_circletext = TRUE;
			int rad = 30000;
#else
			int f_circletext = FALSE;
			int rad = 2000;
#endif

			cout << "before Sch->draw_tree" << endl;
			sprintf(label, "orbit_tree_q%d_d%d_%d", F->q, n, i);
			sprintf(fname, "orbit_tree_q%d_d%d_%d_tables.tex", F->q, n, i);
			Sch->draw_tree(label,
				i, xmax, ymax,
				f_circletext, rad,
				TRUE /* f_embedded */, FALSE /* f_sideways */,
				0.3 /* scale */, 1. /* line_width */,
				FALSE, NULL,
				0 /*verbose_level */);

			{
			ofstream fp(fname);
			latex_interface L;

			L.head_easy(fp);


			A->Strong_gens->print_generators_tex(fp);

			Sch->print_and_list_orbits_tex(fp);

			Sch->print_tables_latex(fp, TRUE /* f_with_cosetrep */);

			L.foot(fp);
			}
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;

			}
			}



		}
#endif

#if 0
	if (nb_identify) {
		cout << "starting the identification:" << endl;
		for (h = 0; h < nb_identify; h++) {



			cout << "Trying to identify " << h << " / "
				<< nb_identify << endl;

			cout << "which is "
				<< Identify_label[h] << " : "
				<< Identify_coeff_text[h] << endl;

			int *coeff_list;
			int nb_coeffs;
			int a, b;

			int_vec_scan(Identify_coeff_text[h],
					coeff_list, nb_coeffs);
			if (ODD(nb_coeffs)) {
				cout << "number of coefficients must be even" << endl;
				exit(1);
				}

			int_vec_zero(coeff, HPD->nb_monomials);
			for (i = 0; i < nb_coeffs >> 1; i++) {
				a = coeff_list[2 * i + 0];
				b = coeff_list[2 * i + 1];
				if (b >= HPD->nb_monomials) {
					cout << "b >= HPD->nb_monomials" << endl;
					exit(1);
					}
				coeff[b] = a;
				}

			cout << "The equation of the input surface is:" << endl;
			int_vec_print(cout, coeff, HPD->nb_monomials);
			cout << endl;


			pt = HPD->rank_coeff_vector(coeff);
			cout << "pt=" << pt << endl;


			HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);
			cout << "We found " << nb_pts << " points on the curve" << endl;
			cout << "They are : ";
			lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			HPD->P->print_set_numerical(cout, Pts, nb_pts);

			char fname[1000];

			sprintf(fname, "point_set_identify_%d", h);
			//draw_point_set_in_plane(HPD, Pts, nb_pts,
			//fname, TRUE /* f_with_points */);

			cout << "before HPD->P->draw_point_set_in_plane" << endl;
			HPD->P->draw_point_set_in_plane(fname,
					Pts, nb_pts,
					TRUE /* f_with_points */,
					FALSE /* f_point_labels */,
					TRUE /* f_embedded */,
					FALSE /* f_sideways */,
					17000 /* rad */,
					verbose_level);
			cout << "after HPD->P->draw_point_set_in_plane" << endl;

			strong_generators *SG;
			longinteger_object ago;
			cout << "before Sch->transporter_from_point_to_orbit_rep" << endl;
			Sch->transporter_from_point_to_orbit_rep(pt,
					orbit_idx, Elt1, 5 /* verbose_level */);
			cout << "after Sch->transporter_from_point_to_orbit_rep" << endl;



			SG = NEW_OBJECT(strong_generators);
			SG->init(A);
			cout << "before SG->init_point_stabilizer_of_arbitrary_"
					"point_through_schreier" << endl;
			SG->init_point_stabilizer_of_arbitrary_point_through_schreier(
				Sch,
				pt, orbit_idx, go /*full_group_order */,
				0 /* verbose_level */);
			cout << "The given pt " << pt << " lies in orbit " << orbit_idx << endl;
			cout << "orbit has length " << Sch->orbit_len[orbit_idx] << endl;
			cout << "A transporter from the point to the orbit rep is:" << endl;
			A->element_print(Elt1, cout);

			Sch->print_generators();

			char label[1000];
			int xmax = 1000000;
			int ymax =  500000;
			int f_circletext = FALSE;
			int rad = 2000;

			sprintf(label, "orbit_tree_q%d_d%d_%d", q, degree, orbit_idx);
			cout << "before Sch->draw_tree" << endl;
			Sch->draw_tree(label,
				orbit_idx, xmax, ymax, f_circletext, rad,
				TRUE /* f_embedded */, FALSE /* f_sideways */,
				0.3 /* scale */, 1. /* line_width */,
				FALSE, NULL,
				0 /*verbose_level */);


			SG->group_order(ago);
			cout << "The stabilizer is a group of order " << ago << endl;

			cout << "generators for the stabilizer of "
					"pt = " << pt << " are:" << endl;
			SG->print_generators();


			cout << "before set_stabilizer_in_projective_space" << endl;
			strong_generators *SG2;
			longinteger_object stab_go;
			int canonical_pt;
			SG2 = A->set_stabilizer_in_projective_space(
				HPD->P,
				Pts, nb_pts, canonical_pt, NULL,
				FALSE, NULL,
				verbose_level + 3);

			SG2->group_order(stab_go);

			cout << "stabilizer order = " << stab_go << endl;
			cout << "stabilizer generators:" << endl;
			SG2->print_generators_tex();
			FREE_OBJECT(SG2);

			//sims *Stab;
			//action *AR;
			int *perm;
			int *Elt;

			Elt = NEW_int(A->elt_size_in_int);
			perm = NEW_int(nb_pts);
			//Stab = SG->create_sims(0 /*verbose_level*/);
			cout << "creating restricted action on the curve:" << endl;
			//AR = A->restricted_action(Pts, nb_pts, 0 /* verbose_level */);


			int a1, a2, a3, a4, a6;
			//int A6[6];

			if (!HPD->test_weierstrass_form(pt,
				a1, a2, a3, a4, a6, 0 /* verbose_level */)) {
				cout << "Not in Weierstrass form" << endl;
				exit(1);
				}
			cout << "The curve is in Weierstrass form: "
					"a1=" << a1
					<< " a2=" << a2
					<< " a3=" << a3
					<< " a4=" << a4
					<< " a6=" << a6 << endl;
#if 0
			A6[0] = a1;
			A6[1] = a2;
			A6[2] = a3;
			A6[3] = a4;
			A6[4] = 0;
			A6[5] = a6;
#endif

			//c = HPD->P->elliptic_curve_addition(A6,
			//Pts[0], Pts[1], 2 /*verbose_level*/);
			//cout << "c=" << c << endl;

#if 0
			int *Table;
			cout << "before HPD->P->elliptic_curve_addition_table" << endl;
			HPD->P->elliptic_curve_addition_table(A6,
					Pts, nb_pts, Table, verbose_level);
			cout << "The addition table:" << endl;
			int_matrix_print(Table, nb_pts, nb_pts);
			int_matrix_print_tex(cout, Table, nb_pts, nb_pts);


			cout << "The group elements acting on the curve:" << endl;
			Stab->print_all_group_elements_as_permutations_in_special_action(AR);

			int order, k, idx;
			int *Idx;

			Idx = NEW_int(nb_pts);
			for (k = 0; k < nb_pts; k++) {
				Idx[k] = -1;
				}

			order = ago.as_int();
			for (i = 0; i < order; i++) {
				cout << "elt " << i << " / " << order << " : ";
				Stab->element_as_permutation(AR, i, perm, 0 /*verbose_level */);
				for (j = 0; j < nb_pts; j++) {
					cout << perm[j] << ", ";
					}
				cout << " ; ";
				for (k = 0; k < nb_pts; k++) {
					if (int_vec_compare(perm, Table + k * nb_pts, nb_pts) == 0) {
						cout << k << " ";
						Idx[k] = i;
						}
					}
				cout << endl;
				}

			for (i = 0; i < nb_pts; i++) {
				idx = Idx[i];
				cout << "point " << i << " / " << nb_pts
						<< " corresponds to element " << idx << ":" << endl;
				Stab->element_unrank_int(idx, Elt);
				cout << "which is:" << endl;
				A->element_print(Elt, cout);
				cout << endl;
				}
#endif

			FREE_OBJECT(SG);
			FREE_int(perm);
			FREE_int(Elt);

			}
	}

	if (f_test_orbit) {
		cout << "test_orbit_idx = " << test_orbit_idx << endl;

		f = Sch->orbit_first[test_orbit_idx];
		l = Sch->orbit_len[test_orbit_idx];

		D.integral_division_by_int(go, l, go1, r);

		cout << "stab order " << go1 << endl;
		a = Sch->orbit[f];
		cout << "orbit length = " << l << endl;
		cout << "orbit rep = " << a << endl;

		HPD->unrank_coeff_vector(coeff, a);

		cout << "orbit rep is " << a << " which is ";
		int_vec_print(cout, coeff, HPD->nb_monomials);
		cout << " = ";
		HPD->print_equation(cout, coeff);
		cout << endl;

		long int *Pts2;
		long int *Pts3;
		int nb_pts2;
		int *coeff2;
		int r, b, orbit_idx;

		Pts2 = NEW_lint(HPD->P->N_points);
		Pts3 = NEW_lint(HPD->P->N_points);
		coeff2 = NEW_int(HPD->nb_monomials);

		HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);
		cout << "We found " << nb_pts << " points on the curve" << endl;

		Sorting.lint_vec_heapsort(Pts, nb_pts);

		cout << "They are : ";
		lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->P->print_set_numerical(cout, Pts, nb_pts);

		//r = random_integer(l);
		//cout << "Picking random integer " << r << endl;


		for (r = 0; r < l; r++) {
			cout << "orbit element " << r << " / " << l << endl;

			b = Sch->orbit[f + r];
			cout << "b=" << b << endl;

			HPD->unrank_coeff_vector(coeff2, b);


			cout << "orbit element " << b << " is ";
			int_vec_print(cout, coeff2, HPD->nb_monomials);
			cout << " = ";
			HPD->print_equation(cout, coeff2);
			cout << endl;



			HPD->enumerate_points(coeff2, Pts2, nb_pts2, verbose_level);
			cout << "We found " << nb_pts2
					<< " points on the curve" << endl;

			Sorting.lint_vec_heapsort(Pts2, nb_pts);

			cout << "They are : ";
			lint_vec_print(cout, Pts2, nb_pts2);
			cout << endl;
			HPD->P->print_set_numerical(cout, Pts2, nb_pts2);
			if (nb_pts2 != nb_pts) {
				cout << "nb_pts2 != nb_pts" << endl;
				exit(1);
				}


			Sch->transporter_from_orbit_rep_to_point(b /* pt */,
					orbit_idx, Elt1, verbose_level);
			cout << "transporter = " << endl;
			A->element_print_quick(Elt1, cout);

			A->element_invert(Elt1, Elt2, 0);

			cout << "transporter inv = " << endl;
			A->element_print_quick(Elt2, cout);

			A->map_a_set_and_reorder(Pts, Pts3, nb_pts, Elt1,
					verbose_level);
			Sorting.lint_vec_heapsort(Pts3, nb_pts);

			cout << "after apply : ";
			lint_vec_print(cout, Pts3, nb_pts);
			cout << endl;


			if (lint_vec_compare(Pts3, Pts2, nb_pts)) {
				cout << "The sets are different" << endl;
				exit(1);
				}

			cout << "The element maps the orbit representative "
					"to the new set, which is good" << endl;
			}

		FREE_lint(Pts2);
		FREE_lint(Pts3);
		FREE_int(coeff2);
		}
#endif


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

#if 0
	FREE_lint(Pts);
	FREE_int(coeff);
	FREE_int(line_type);
	FREE_int(Kernel);
#endif

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_polynomials done" << endl;
	}
}

#if 0
void polynomial_orbits_callback_print_function(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;

	coeff = NEW_int(HPD->get_nb_monomials());
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
}

void polynomial_orbits_callback_print_function2(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;
	std::vector<long int> Points;
	//long int *Pts;
	//int nb_pts;

	//Pts = NEW_lint(HPD->P->N_points);
	coeff = NEW_int(HPD->get_nb_monomials());
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	HPD->enumerate_points(coeff, Points,  0 /*verbose_level*/);
	ost << Points.size();
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	//HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
	//FREE_lint(Pts);
}
#endif






}}
