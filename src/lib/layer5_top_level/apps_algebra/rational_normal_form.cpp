/*
 * rational_normal_form.cpp
 *
 *  Created on: Mar 20, 2025
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {



rational_normal_form::rational_normal_form()
{
	Record_birth();
}


rational_normal_form::~rational_normal_form()
{
	Record_death();

}


void rational_normal_form::make_classes_GL(
		algebra::field_theory::finite_field *F,
		int d, int f_no_eigenvalue_one, int verbose_level)
// called from interface_algebra
// creates an object of type action
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::make_classes_GL" << endl;
	}

	algebra::linear_algebra::gl_classes C;
	algebra::linear_algebra::gl_class_rep *R;
	int nb_classes;
	int i;


	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"before C.init" << endl;
	}
	C.init(d, F, verbose_level - 2);
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"after C.init" << endl;
	}

	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"before C.make_classes" << endl;
	}
	C.make_classes(R, nb_classes, f_no_eigenvalue_one, verbose_level);
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"after C.make_classes" << endl;
	}

	actions::action *A;
	algebra::ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;
	int a;
	int *Mtx;
	int *Elt;



	A = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"before A->Known_groups->init_projective_group" << endl;
	}
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"after A->Known_groups->init_projective_group" << endl;
	}

	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	actions::action *A_on_lines;

	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"before A->Induced_action->induced_action_on_grassmannian" << endl;
	}
	A_on_lines = A->Induced_action->induced_action_on_grassmannian(
			2, verbose_level);
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"after A->Induced_action->induced_action_on_grassmannian" << endl;
	}

	Mtx = NEW_int(d * d);
	Elt = NEW_int(A->elt_size_in_int);

	int order, nb_fixpoints, nb_fixlines;

	std::string Col_headings[7];

	Col_headings[0] = "Line";
	Col_headings[1] = "order";
	Col_headings[2] = "nb_fixpoints";
	Col_headings[3] = "nb_fixlines";
	Col_headings[4] = "centralizer_order";
	Col_headings[5] = "class_size";
	Col_headings[6] = "matrix";

	std::string *Table;
	int nb_rows, nb_cols;

	nb_rows = nb_classes;
	nb_cols = 7;
	Table = new string [nb_rows * nb_cols];


	for (i = 0; i < nb_classes; i++) {

		C.make_matrix_from_class_rep(
				Mtx, R + i, 0 /*verbose_level - 1 */);

		A->Group_element->make_element(Elt, Mtx, 0);

		a = A->Sims->element_rank_lint(Elt);

		cout << "Representative of class " << i << " / "
				<< nb_classes << " has rank " << a << endl;
		Int_matrix_print(Elt, d, d);


		order = A->Group_element->element_order(Elt);

		nb_fixpoints = A->Group_element->count_fixed_points(
				Elt, 0 /*  verbose_level */);


		nb_fixlines = A_on_lines->Group_element->count_fixed_points(
				Elt, 0 /*  verbose_level */);

		algebra::ring_theory::longinteger_object go, co, cl;
		int *Mtx2;

		C.get_matrix_and_centralizer_order(
				go,
				co,
				cl,
				Mtx2,
				R + i);


		FREE_int(Mtx2);


		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = std::to_string(order);
		Table[i * nb_cols + 2] = std::to_string(nb_fixpoints);
		Table[i * nb_cols + 3] = std::to_string(nb_fixlines);
		Table[i * nb_cols + 4] = co.stringify();
		Table[i * nb_cols + 5] = cl.stringify();
		Table[i * nb_cols + 6] = "\"" + Int_vec_stringify(Elt, d * d) + "\"";

		C.print_matrix_and_centralizer_order_latex(
				cout, R + i);

	}

	other::orbiter_kernel_system::file_io Fio;


	string fname;

	{
		fname = "Class_reps_GL_" + std::to_string(d)
				+ "_" + std::to_string(F->q) + "_classes.csv";

		other::orbiter_kernel_system::file_io Fio;


		Fio.Csv_file_support->write_table_of_strings_with_col_headings(
				fname,
				nb_rows, nb_cols, Table,
				Col_headings,
				verbose_level);

	}
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	delete [] Table;






	fname = "Class_reps_GL_" + std::to_string(d)
			+ "_" + std::to_string(F->q) + ".tex";
	{
		ofstream fp(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(fp);
		C.report(fp, verbose_level);
		L.foot(fp);
	}
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	//make_gl_classes(d, q, f_no_eigenvalue_one, verbose_level);

	FREE_int(Mtx);
	FREE_int(Elt);
	FREE_OBJECTS(R);
	FREE_OBJECT(A);
	FREE_OBJECT(A_on_lines);
	if (f_v) {
		cout << "rational_normal_form::make_classes_GL done" << endl;
	}
}


#if 0
// please use action_global::rational_normal_form
void rational_normal_form::compute_rational_normal_form(
		algebra::field_theory::finite_field *F,
		int d,
		int *matrix_data,
		int *Basis, int *Rational_normal_form,
		int verbose_level)
// opens a projective group action to make the element.
// matrix_data[d * d]
// list could go in level 3
// compare action_global::rational_normal_form
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form" << endl;
	}

	algebra::linear_algebra::gl_classes C;
	algebra::linear_algebra::gl_class_rep *Reps;
	int nb_classes;

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"before C.init" << endl;
	}
	C.init(d, F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"after C.init" << endl;
	}

	int f_no_eigenvalue_one = false;

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"before C.make_classes" << endl;
	}
	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"after C.make_classes" << endl;
	}



	actions::action *A;
	algebra::ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"before A->Known_groups->init_projective_group" << endl;
	}
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */,
			true /* f_init_sims */,
			nice_gens,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"after A->Known_groups->init_projective_group" << endl;
	}

	FREE_OBJECT(nice_gens);

	A->print_base();
	A->group_order(Go);


	int class_rep;

	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);

	//go = Go.as_int();

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"Making element from data ";
		Int_vec_print(cout, matrix_data, d * d);
		cout << endl;
	}

	//A->Sims->element_unrank_int(elt_idx, Elt);
	A->Group_element->make_element(Elt, matrix_data, verbose_level);

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"Looking at element:" << endl;
		Int_matrix_print(Elt, d, d);
	}


	algebra::linear_algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(algebra::linear_algebra::gl_class_rep);

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"before C.identify_matrix" << endl;
	}
	C.identify_matrix(Elt, R1, Basis, Rational_normal_form, verbose_level);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"after C.identify_matrix" << endl;
	}

	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"before C.find_class_rep" << endl;
	}
	class_rep = C.find_class_rep(
			Reps, nb_classes, R1,
			verbose_level);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form "
				"after C.find_class_rep" << endl;
	}

	if (f_v) {
		cout << "class = " << class_rep << endl;
	}

	FREE_OBJECT(R1);




	FREE_int(Elt);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	FREE_OBJECTS(Reps);
	if (f_v) {
		cout << "rational_normal_form::compute_rational_normal_form done" << endl;
	}
}
#endif


#if 0
void rational_normal_form::do_identify_one(
		int q, int d,
		int f_no_eigenvalue_one, int elt_idx,
		int verbose_level)
// ToDo: move this
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::do_identify_one" << endl;
	}
	linear_algebra::gl_classes C;
	linear_algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes,
			f_no_eigenvalue_one, verbose_level);



	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
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

	if (f_v) {
		cout << "Looking at element " << elt_idx << ":" << endl;
	}

	A->Sims->element_unrank_lint(elt_idx, Elt);
	if (f_v) {
		Int_matrix_print(Elt, d, d);
	}


	linear_algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(linear_algebra::gl_class_rep);

	C.identify_matrix(Elt, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(
			Reps, nb_classes, R1,
			0 /* verbose_level */);

	if (f_v) {
		cout << "class = " << class_rep << endl;
	}

	FREE_OBJECT(R1);




	FREE_int(Elt);
	FREE_int(Basis);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	FREE_OBJECTS(Reps);
	if (f_v) {
		cout << "rational_normal_form::do_identify_one done" << endl;
	}
}

void rational_normal_form::do_identify_all(
		int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
// ToDo: move this
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::do_identify_all" << endl;
	}
	linear_algebra::gl_classes C;
	linear_algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	actions::action *A;
	ring_theory::longinteger_object Go;
	int *Class_count;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */,
			true /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	int i, go, class_rep;

	int *Elt, *Basis;

	Class_count = NEW_int(nb_classes);
	Int_vec_zero(Class_count, nb_classes);
	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	go = Go.as_int();
	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << ":" << endl;

		A->Sims->element_unrank_lint(i, Elt);
		Int_matrix_print(Elt, d, d);


		linear_algebra::gl_class_rep *R1;

		R1 = NEW_OBJECT(linear_algebra::gl_class_rep);

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
	if (f_v) {
		cout << "rational_normal_form::do_identify_all done" << endl;
	}
}

void rational_normal_form::do_random(
		int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
// ToDo: move this
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::do_random" << endl;
	}
	//gl_random_matrix(d, q, verbose_level);

	linear_algebra::gl_classes C;
	linear_algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);
	C.init(d, F, verbose_level);

	C.make_classes(
			Reps, nb_classes, f_no_eigenvalue_one,
			verbose_level);

	int *Mtx;
	int *Basis;
	int class_rep;


	Mtx = NEW_int(d * d);
	Basis = NEW_int(d * d);

	C.F->Linear_algebra->random_invertible_matrix(
			Mtx, d, verbose_level - 2);


	linear_algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(linear_algebra::gl_class_rep);

	C.identify_matrix(
			Mtx, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(
			Reps, nb_classes,
			R1, 0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

	FREE_OBJECT(R1);

	FREE_int(Mtx);
	FREE_int(Basis);
	FREE_OBJECTS(Reps);
	FREE_OBJECT(F);
	if (f_v) {
		cout << "rational_normal_form::do_random done" << endl;
	}
}


void rational_normal_form::group_table(
		int q, int d, int f_poly, std::string &poly,
		int f_no_eigenvalue_one, int verbose_level)
// This function does too many things!
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::group_table" << endl;
	}
	linear_algebra::gl_classes C;
	linear_algebra::gl_class_rep *Reps;
	int nb_classes;
	int *Class_rep;
	int *List;
	int list_sz, a, b, j, h;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	if (f_poly) {
		F->init_override_polynomial_small_order(
				q, poly,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
	}
	else {
		F->finite_field_init_small_order(
				q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
	}

	C.init(d, F, verbose_level);

	C.make_classes(
			Reps, nb_classes, f_no_eigenvalue_one,
			verbose_level);


	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(
			d /* n */,
			F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
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
		Int_matrix_print(Elt, d, d);

		{
			ring_theory::unipoly_domain U(C.F);
			ring_theory::unipoly_object char_poly;



			U.create_object_by_rank(
					char_poly, 0, verbose_level);

			U.characteristic_polynomial(
					Elt,
					d, char_poly, verbose_level - 2);

			cout << "The characteristic polynomial is ";
			U.print_object(char_poly, cout);
			cout << endl;

			eval = U.substitute_scalar_in_polynomial(
					char_poly,
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
		Int_matrix_print(Elt, d, d);


		linear_algebra::gl_class_rep *R1;

		R1 = NEW_OBJECT(linear_algebra::gl_class_rep);

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
	Int_vec_zero(Group_table, list_sz * list_sz);
	for (i = 0; i < list_sz; i++) {
		a = List[i];
		A->Sims->element_unrank_lint(a, Elt1);
		for (j = 0; j < list_sz; j++) {
			b = List[j];
			A->Sims->element_unrank_lint(b, Elt2);
			A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
			h = A->Sims->element_rank_lint(Elt3);
			Group_table[i * list_sz + j] = h;
			}
		}
	int L_sz = list_sz + 1;
	Table = NEW_int(L_sz * L_sz);
	Int_vec_zero(Table, L_sz * L_sz);
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
	Int_matrix_print(Table, L_sz, L_sz);


	{


		string fname, title, author, extra_praeamble;

		fname.assign("group_table.tex");


		ofstream fp(fname);
		l1_interfaces::latex_interface L;

		L.head(fp, false /* f_book */, false /* f_title */,
			title /*const char *title */, author /*const char *author */,
			false /* f_toc */, false /* f_landscape */, false /* f_12pt */,
			false /* f_enlarged_page */, false /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);


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
	if (f_v) {
		cout << "rational_normal_form::group_table done" << endl;
	}
}

void rational_normal_form::centralizer_in_PGL_d_q_single_element_brute_force(
		int q, int d,
		int elt_idx, int verbose_level)
// this function is not used anywhere
// ToDo: move this
// problem elt_idx does not describe the group element uniquely.
// Reason: the sims chain is not canonical.
// creates a finite_field object and an action object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::centralizer_in_PGL_d_q_single_element_brute_force" << endl;
	}
	actions::action *A;
	ring_theory::longinteger_object Go;
	field_theory::finite_field *F;
	data_structures_groups::vector_ge *nice_gens;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */,
			true /* f_init_sims */,
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
	Int_matrix_print(Elt, d, d);

	A->Group_element->element_invert(Elt, Eltv, 0);

	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << " / " << go << endl;

		A->Sims->element_unrank_lint(i, Elt1);
		//int_matrix_print(Elt1, d, d);


		A->Group_element->element_invert(Elt1, Elt2, 0);
		A->Group_element->element_mult(Elt2, Elt, Elt3, 0);
		A->Group_element->element_mult(Elt3, Elt1, Elt2, 0);
		A->Group_element->element_mult(Elt2, Eltv, Elt3, 0);
		if (A->Group_element->is_one(Elt3)) {
			List[sz++] = i;
			}
		}

	cout << "The centralizer has order " << sz << endl;

	int a;
	data_structures_groups::vector_ge *gens;
	data_structures_groups::vector_ge *SG;
	int *tl;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	SG = NEW_OBJECT(data_structures_groups::vector_ge);
	tl = NEW_int(A->base_len());
	gens->init(A, verbose_level - 2);
	gens->allocate(sz, verbose_level - 2);

	for (i = 0; i < sz; i++) {
		a = List[i];

		cout << "Looking at element " << i << " / " << sz
				<< " which is " << a << endl;

		A->Sims->element_unrank_lint(a, Elt1);
		Int_matrix_print(Elt1, d, d);

		A->Group_element->element_move(Elt1, gens->ith(i), 0);
		}

	groups::sims *Cent;

	Cent = A->create_sims_from_generators_with_target_group_order_lint(
			gens, sz, 0 /* verbose_level */);
	Cent->extract_strong_generators_in_order(*SG, tl,
			0 /* verbose_level */);
	cout << "strong generators for the centralizer are:" << endl;
	for (i = 0; i < SG->len; i++) {

		A->Group_element->element_move(SG->ith(i), Elt1, 0);
		a = A->Sims->element_rank_lint(Elt1);

		cout << "Element " << i << " / " << SG->len
				<< " which is " << a << endl;

		Int_matrix_print(Elt1, d, d);

		}



	FREE_int(Elt);
	FREE_int(Eltv);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	if (f_v) {
		cout << "rational_normal_form::centralizer_in_PGL_d_q_single_element_brute_force done" << endl;
	}
}


void rational_normal_form::centralizer_in_PGL_d_q_single_element(
		int q, int d,
		int elt_idx, int verbose_level)
// this function is not used anywhere
// ToDo: move this
// creates a finite_field, and two actions
// using init_projective_group and init_general_linear_group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::centralizer_in_PGL_d_q_single_element" << endl;
	}
	field_theory::finite_field *F;
	actions::action *A_PGL;
	actions::action *A_GL;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	A_PGL = NEW_OBJECT(actions::action);
	A_PGL->Known_groups->init_projective_group(
			d /* n */, F,
		false /* f_semilinear */,
		true /* f_basis */, true /* f_init_sims */,
		nice_gens,
		0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A_PGL->print_base();
	A_PGL->group_order(Go);

	A_GL = NEW_OBJECT(actions::action);
	A_GL->Known_groups->init_general_linear_group(
			d /* n */, F,
		false /* f_semilinear */,
		true /* f_basis */, true /* f_init_sims */,
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
	Int_matrix_print(Elt, d, d);

	groups::strong_generators *Cent;
	groups::strong_generators *Cent_GL;
	ring_theory::longinteger_object go, go1;

	Cent = NEW_OBJECT(groups::strong_generators);
	Cent_GL = NEW_OBJECT(groups::strong_generators);

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
	if (f_v) {
		cout << "rational_normal_form::centralizer_in_PGL_d_q_single_element done" << endl;
	}

}

void rational_normal_form::compute_centralizer_of_all_elements_in_PGL_d_q(
		int q, int d, int verbose_level)
// this function is not used anywhere
// ToDo: move this
// creates a finite_field, and an action
// using init_projective_group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::compute_centralizer_of_all_elements_in_PGL_d_q" << endl;
	}
	actions::action *A;
	field_theory::finite_field *F;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;
	int go, i;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(
			q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);
	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
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
		Int_matrix_print(Elt, d, d);

		groups::sims *Cent;
		ring_theory::longinteger_object cent_go;

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
	if (f_v) {
		cout << "rational_normal_form::compute_centralizer_of_all_elements_in_PGL_d_q done" << endl;
	}
}
#endif


void rational_normal_form::do_eigenstuff_with_coefficients(
		algebra::field_theory::finite_field *F, int n, std::string &coeffs_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_with_coefficients" << endl;
	}
	int *Data;
	int len;

	Int_vec_scan(coeffs_text, Data, len);
	if (len != n * n) {
		cout << "len != n * n " << len << endl;
		exit(1);
	}

	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_with_coefficients "
				"before do_eigenstuff" << endl;
	}
	do_eigenstuff(F, n, Data, verbose_level);
	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_with_coefficients "
				"after do_eigenstuff" << endl;
	}

	FREE_int(Data);
	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_with_coefficients done" << endl;
	}
}

void rational_normal_form::do_eigenstuff_from_file(
		algebra::field_theory::finite_field *F,
		int n, std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_from_file" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	int *Data;
	int mtx_m, mtx_n;

	Fio.Csv_file_support->int_matrix_read_csv(
			fname, Data, mtx_m, mtx_n, verbose_level - 1);
	if (mtx_m != n) {
		cout << "mtx_m != n" << endl;
		exit(1);
	}
	if (mtx_n != n) {
		cout << "mtx_n != n" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_from_file "
				"before do_eigenstuff" << endl;
	}
	do_eigenstuff(F, n, Data, verbose_level);
	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_from_file "
				"after do_eigenstuff" << endl;
	}


	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff_from_file done" << endl;
	}
}



void rational_normal_form::do_eigenstuff(
		algebra::field_theory::finite_field *F,
		int size, int *Data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	typed_objects::discreta_matrix M;
	int i, j, k, h;
	long int a;
	//unipoly_domain U;
	//unipoly_object char_poly;
	algebra::number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff" << endl;
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

	typed_objects::domain d(F);
	typed_objects::with w(&d);

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

	typed_objects::discreta_matrix M1, P, Pv, Q, Qv, S, T;

	M.elements_to_unipoly();
	M.minus_X_times_id();
	M1 = M;
	if (f_v) {
		cout << "M - x * Id has been computed" << endl;
		//cout << "M - x * Id =" << endl << M << endl;
	}
	if (f_v) {
		cout << "M - x * Id = " << endl;
		cout << M << endl;
	}


	if (f_v) {
		cout << "before M.smith_normal_form" << endl;
	}
	M.smith_normal_form(
			P, Pv, Q, Qv, verbose_level);
	if (f_v) {
		cout << "after M.smith_normal_form" << endl;
	}

	if (f_v) {
		cout << "the Smith normal form is:" << endl;
		cout << M << endl;
	}

	S.mult(P, Pv, verbose_level);

	if (f_v) {
		cout << "P * Pv=" << endl << S << endl;
	}

	S.mult(Q, Qv, verbose_level);

	if (f_v) {
		cout << "Q * Qv=" << endl << S << endl;
	}

	S.mult(P, M1, verbose_level);
	T.mult(S, Q, verbose_level);

	if (f_v) {
		cout << "T.mult(S, Q):" << endl;
		cout << "T=" << endl << T << endl;
	}


	typed_objects::unipoly charpoly;
	int deg;
	int l, lv, b, c;

	charpoly = M.s_ij(size - 1, size - 1);

	if (f_v) {
		cout << "characteristic polynomial:" << charpoly << endl;
	}
	deg = charpoly.degree();
	if (f_v) {
		cout << "has degree " << deg << endl;
	}
	l = charpoly.s_ii(deg);
	if (f_v) {
		cout << "leading coefficient " << l << endl;
	}
	lv = F->inverse(l);
	if (f_v) {
		cout << "leading coefficient inverse " << lv << endl;
	}
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		c = F->mult(b, lv);
		charpoly.m_ii(i, c);
	}
	if (f_v) {
		cout << "monic characteristic polynomial:" << charpoly << endl;
	}

	typed_objects::integer x, y;
	int *roots;
	int nb_roots = 0;

	roots = new int[F->q];

	for (a = 0; a < F->q; a++) {
		x.m_i(a);
		charpoly.evaluate_at(x, y);
		if (y.s_i() == 0) {
			if (f_v) {
				cout << "root " << a << endl;
			}
			roots[nb_roots++] = a;
		}
	}
	if (f_v) {
		cout << "we found the following eigenvalues: ";
		Int_vec_print(cout, roots, nb_roots);
		cout << endl;
	}

	int eigenvalue, eigenvalue_negative;

	for (h = 0; h < nb_roots; h++) {
		eigenvalue = roots[h];
		if (f_v) {
			cout << "looking at eigenvalue " << eigenvalue << endl;
		}
		int *A, *B, *Bt;
		eigenvalue_negative = F->negate(eigenvalue);
		A = new int[size * size];
		B = new int[size * size];
		Bt = new int[size * size];
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				A[i * size + j] = Data[i * size + j];
			}
		}
		if (f_v) {
			cout << "A:" << endl;
			Int_vec_print_integer_matrix_width(
					cout, A,
					size, size, size, F->log10_of_q);
		}
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				a = A[i * size + j];
				if (j == i) {
					a = F->add(a, eigenvalue_negative);
				}
				B[i * size + j] = a;
			}
		}
		if (f_v) {
			cout << "B = A - eigenvalue * I:" << endl;
			Int_vec_print_integer_matrix_width(
					cout, B,
					size, size, size, F->log10_of_q);
		}

		F->Linear_algebra->transpose_matrix(
				B, Bt, size, size);

		if (f_v) {
			cout << "B transposed:" << endl;
			Int_vec_print_integer_matrix_width(
					cout, Bt,
					size, size, size, F->log10_of_q);
		}

		int f_special = false;
		int f_complete = true;
		int *base_cols;
		int nb_base_cols;
		int f_P = false;
		int kernel_m, kernel_n, *kernel;

		base_cols = new int[size];
		kernel = new int[size * size];

		nb_base_cols = F->Linear_algebra->Gauss_int(
				Bt,
			f_special, f_complete, base_cols,
			f_P, NULL, size, size, size,
			verbose_level - 1);


		if (f_v) {
			cout << "rank = " << nb_base_cols << endl;
		}

		F->Linear_algebra->matrix_get_kernel(
				Bt, size, size, base_cols, nb_base_cols,
			kernel_m, kernel_n, kernel,
			0 /* verbose_level */);

		if (f_v) {
			cout << "kernel = left eigenvectors:" << endl;
			Int_vec_print_integer_matrix_width(
					cout, kernel,
					size, kernel_n, kernel_n, F->log10_of_q);
		}

		int *vec1, *vec2;
		vec1 = new int[size];
		vec2 = new int[size];
		for (i = 0; i < size; i++) {
			vec1[i] = kernel[i * kernel_n + 0];


		}
		if (f_v) {
			Int_vec_print(cout, vec1, size);
			cout << endl;
		}


		F->Projective_space_basic->PG_element_normalize_from_front(
				vec1, 1, size);


		if (f_v) {
			Int_vec_print(cout, vec1, size);
			cout << endl;
		}


		F->Projective_space_basic->PG_element_rank_modified(
				vec1, 1, size, a);


		if (f_v) {
			cout << "has rank " << a << endl;
		}


		if (f_v) {
			cout << "computing xA" << endl;
		}

		F->Linear_algebra->mult_vector_from_the_left(
				vec1, A, vec2, size, size);


		if (f_v) {
			Int_vec_print(cout, vec2, size);
			cout << endl;
		}


		F->Projective_space_basic->PG_element_normalize_from_front(
				vec2, 1, size);


		if (f_v) {
			Int_vec_print(cout, vec2, size);
			cout << endl;
		}


		F->Projective_space_basic->PG_element_rank_modified(
				vec2, 1, size, a);


		if (f_v) {
			cout << "has rank " << a << endl;
		}

		delete [] vec1;
		delete [] vec2;

		delete [] A;
		delete [] B;
		delete [] Bt;
	}
	if (f_v) {
		cout << "rational_normal_form::do_eigenstuff done" << endl;
	}
}





}}}


