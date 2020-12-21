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


void algebra_global_with_action::orbits_under_conjugation(
		long int *the_set, int set_size, sims *S,
		strong_generators *SG,
		vector_ge *Transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation" << endl;
	}
	action A_conj;
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"before A_conj.induced_action_by_conjugation" << endl;
	}
	A_conj.induced_action_by_conjugation(S, S,
			FALSE /* f_ownership */, FALSE /* f_basis */,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"created action by conjugation" << endl;
	}

	action *A_conj_restricted;

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"before A_conj.restricted_action" << endl;
	}

	A_conj_restricted = A_conj.restricted_action(the_set, set_size,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"after A_conj.restricted_action" << endl;
	}



	schreier Classes;
	Classes.init(A_conj_restricted, verbose_level - 2);
	Classes.init_generators(*SG->gens, verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"before Classes.compute_all_point_orbits" << endl;
	}
	Classes.compute_all_point_orbits(1 /*verbose_level - 1*/);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"after Classes.compute_all_point_orbits" << endl;
		cout << "found " << Classes.nb_orbits << " conjugacy classes" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"before create_subgroups" << endl;
	}
	create_subgroups(
			the_set, set_size, S, &A_conj,
			&Classes,
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"after create_subgroups" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation done" << endl;
	}
}

void algebra_global_with_action::create_subgroups(
		long int *the_set, int set_size, sims *S, action *A_conj,
		schreier *Classes,
		vector_ge *Transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::create_subgroups" << endl;
	}

	int i, j;
	int f, l, rep;
	long int *the_set_sorted;
	long int *position;
	sorting Sorting;

	the_set_sorted = NEW_lint(set_size);
	position = NEW_lint(set_size);
	lint_vec_copy(the_set, the_set_sorted, set_size);
	//Sorting.lint_vec_heapsort(the_set_sorted, set_size);
	for (i = 0; i < set_size; i++) {
		position[i] = i;
	}
	Sorting.lint_vec_heapsort_with_log(the_set_sorted, position, set_size);

	for (i = 0; i < Classes->nb_orbits; i++) {

		f = Classes->orbit_first[i];
		l = Classes->orbit_len[i];
		rep = Classes->orbit[f];
		if (f_v) {
			cout << "Orbit " << i << " has length " << l << " representative is " << rep << " = " << the_set[rep] << endl;
		}
	}

	long int rk0;
	long int rk1;
	long int rk2;
	int idx;
	int *Elt0;
	int *Elt1;
	int *Elt2;
	int nb_flag_orbits;
	long int *Flags;
	int *SO;
	int *SOL;

	Elt0 = NEW_int(S->A->elt_size_in_int);
	Elt1 = NEW_int(S->A->elt_size_in_int);
	Elt2 = NEW_int(S->A->elt_size_in_int);

	f = Classes->orbit_first[0];
	l = Classes->orbit_len[0];
	if (l != 1) {
		cout << "algebra_global_with_action::create_subgroups l != 1" << endl;
		exit(1);
	}
	rep = Classes->orbit[f];
	rk0 = the_set[rep];

	S->element_unrank_lint(rk0, Elt0);

	nb_flag_orbits = 0;
	Flags = NEW_lint(Classes->nb_orbits * 3);
	SO = NEW_int(Classes->nb_orbits);
	SOL = NEW_int(Classes->nb_orbits);

	for (j = 1; j < Classes->nb_orbits; j++) {



		f = Classes->orbit_first[j];
		l = Classes->orbit_len[j];
		rep = Classes->orbit[f];
		rk1 = the_set[rep];
		rk2 = S->mult_by_rank(rk0, rk1, 0 /*verbose_level*/);

		if (Sorting.lint_vec_search(the_set_sorted, set_size, rk2, idx, 0 /*verbose_level*/)) {
			cout << "flag orbit " << nb_flag_orbits << " : " << j << " l=" << l << " : " << rk0 << "," << rk1 << "," << rk2 << endl;

			S->element_unrank_lint(rk1, Elt1);
			S->element_unrank_lint(rk2, Elt2);
			S->A->element_print_quick(Elt0, cout);
			S->A->element_print_quick(Elt1, cout);
			S->A->element_print_quick(Elt2, cout);

			Flags[nb_flag_orbits * 3 + 0] = rk0;
			Flags[nb_flag_orbits * 3 + 1] = rk1;
			Flags[nb_flag_orbits * 3 + 2] = rk2;
			SO[nb_flag_orbits] = j;
			SOL[nb_flag_orbits] = l;
			nb_flag_orbits++;
		}

	}

	if (f_v) {
		cout << "We found " << nb_flag_orbits << " flag orbits" << endl;
	}

	int flag;
	int nb_iso;
	int *upstep_transversal_size;
	int *iso_type_of_flag_orbit;
	int *f_is_definition;
	int *flag_orbit_of_iso_type;
	int *f_fused;
	long int cur_flag[3];
	long int cur_flag_mapped1[3];
	int h, pt;

	upstep_transversal_size = NEW_int(nb_flag_orbits);
	iso_type_of_flag_orbit = NEW_int(nb_flag_orbits);
	flag_orbit_of_iso_type = NEW_int(nb_flag_orbits);
	f_is_definition = NEW_int(nb_flag_orbits);
	f_fused = NEW_int(nb_flag_orbits);
	int_vec_zero(f_is_definition, nb_flag_orbits);
	int_vec_zero(f_fused, nb_flag_orbits);

	nb_iso = 0;
	for (flag = 0; flag < nb_flag_orbits; flag++) {
		if (f_fused[flag]) {
			continue;
		}
		f_is_definition[flag] = TRUE;
		iso_type_of_flag_orbit[flag] = nb_iso;
		flag_orbit_of_iso_type[nb_iso] = flag;
		upstep_transversal_size[nb_iso] = 1;

		for (h = 1; h < 3; h++) {
			if (h == 1) {
				cur_flag[0] = Flags[flag * 3 + 1];
				cur_flag[1] = Flags[flag * 3 + 0];
				cur_flag[2] = Flags[flag * 3 + 2];
			}
			else {
				cur_flag[0] = Flags[flag * 3 + 2];
				cur_flag[1] = Flags[flag * 3 + 1];
				cur_flag[2] = Flags[flag * 3 + 0];
			}

			// move cur_flag[0] to the_set[0] using the inverse of Transporter

			if (!Sorting.lint_vec_search(the_set_sorted, set_size, cur_flag[0], idx, 0 /*verbose_level*/)) {
				cout << "cannot find cur_flag[0] in the_set_sorted" << endl;
				exit(1);
			}
			pt = position[idx];
			S->A->element_invert(Transporter->ith(pt), Elt0, 0);
			for (int u = 0; u < 3; u++) {
				cur_flag_mapped1[u] = A_conj->element_image_of(cur_flag[u], Elt0, 0);
			}
			if (cur_flag_mapped1[0] != rk0) {
				cout << "cur_flag_mapped1[u] != rk0" << endl;
				exit(1);
			}



			if (!Sorting.lint_vec_search(the_set_sorted, set_size, cur_flag_mapped1[1], idx, 0 /*verbose_level*/)) {
				cout << "cannot find cur_flag[1] in the_set_sorted" << endl;
				exit(1);
			}
			pt = position[idx];
			j = Classes->orbit_number(pt);
			if (j == SO[flag]) {
				cout << "found an automorphism" << endl;
				upstep_transversal_size[nb_iso]++;
			}
			else {
				if (!Sorting.int_vec_search(SO, nb_flag_orbits, j, idx)) {
					cout << "cannot find j in SO" << endl;
					exit(1);
				}
				f_fused[idx] = TRUE;
			}
		}

		nb_iso++;
	}

	cout << "We found " << nb_iso << " conjugacy classes of subgroups" << endl;
	for (i = 0; i < nb_iso; i++) {
		flag = flag_orbit_of_iso_type[i];
		rk0 = Flags[flag * 3 + 0];
		rk1 = Flags[flag * 3 + 1];
		rk2 = Flags[flag * 3 + 2];
		cout << i << " : " << flag << " : " <<  " : " << SO[flag] << " l=" << SOL[flag]
				<< " : " << rk0 << "," << rk1 << "," << rk2 << " : "
				<< upstep_transversal_size[i] << endl;
	}

	FREE_int(upstep_transversal_size);
	FREE_int(iso_type_of_flag_orbit);
	FREE_int(f_is_definition);
	FREE_int(f_fused);
	FREE_int(flag_orbit_of_iso_type);
	FREE_lint(Flags);
	FREE_int(SO);
	FREE_int(Elt0);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_lint(the_set_sorted);
	FREE_lint(position);
	if (f_v) {
		cout << "algebra_global_with_action::create_subgroups done" << endl;
	}
}

void algebra_global_with_action::orbits_on_set_from_file(
		long int *the_set, int set_size,
		action *A1, action *A2,
		vector_ge *gens,
		std::string &label_set,
		std::string &label_group,
		long int *&Table,
		int &orbit_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_set_from_file" << endl;
	}

	orbit_of_sets *OS;

	OS = NEW_OBJECT(orbit_of_sets);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_set_from_file before OS->init" << endl;
	}
	OS->init(A1, A2, the_set, set_size, gens, verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_set_from_file after OS->init" << endl;
	}

	if (f_v) {
		cout << "Found an orbit of length " << OS->used_length << endl;
	}

	int set_size1;

	if (f_v) {
		cout << "before OS->get_table_of_orbits" << endl;
	}
	OS->get_table_of_orbits_and_hash_values(Table,
			orbit_length, set_size1, verbose_level - 2);
	if (f_v) {
		cout << "after OS->get_table_of_orbits" << endl;
	}

	if (f_v) {
		cout << "before OS->get_table_of_orbits" << endl;
	}
	OS->get_table_of_orbits(Table,
			orbit_length, set_size, verbose_level);
	if (f_v) {
		cout << "after OS->get_table_of_orbits" << endl;
	}


	// write transporter as csv file:

	string fname;

	vector_ge *Coset_reps;

	if (f_v) {
		cout << "before OS->make_table_of_coset_reps" << endl;
	}
	OS->make_table_of_coset_reps(Coset_reps, verbose_level);
	if (f_v) {
		cout << "after OS->make_table_of_coset_reps" << endl;
	}

	fname.assign(label_set);
	fname.append("_orbit_under_");
	fname.append(label_group);
	fname.append("_transporter.csv");

	Coset_reps->write_to_csv_file_coded(fname, verbose_level);

	// testing Coset_reps

	if (f_v) {
		cout << "testing Coset_reps " << endl;
	}

	long int rk0 = the_set[0];
	long int rk1;

	for (int i = 0; i < orbit_length; i++) {
		rk1 = A2->element_image_of(rk0, Coset_reps->ith(i), 0);
		if (rk1 != Table[i * set_size + 0]) {
			cout << "rk1 != Table[i * set_size + 0], i=" << i << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "testing Coset_reps passes" << endl;
	}

	// write as csv file:


	fname.assign(label_set);
	fname.append("_orbit_under_");
	fname.append(label_group);
	fname.append(".csv");

	if (f_v) {
		cout << "Writing orbit to file " << fname << endl;
	}
	file_io Fio;

	Fio.lint_matrix_write_csv(fname, Table, orbit_length, set_size);
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	// write as txt file:


	fname.assign(label_set);
	fname.append("_orbit_under_");
	fname.append(label_group);
	fname.append(".txt");

	if (f_v) {
		cout << "Writing table to file " << fname << endl;
	}
	{
		ofstream ost(fname);
		int i;
		for (i = 0; i < orbit_length; i++) {
			ost << set_size;
			for (int j = 0; j < set_size; j++) {
				ost << " " << Table[i * set_size + j];
			}
			ost << endl;
		}
		ost << -1 << " " << orbit_length << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "before FREE_OBJECT(OS)" << endl;
	}
	FREE_OBJECT(OS);
	if (f_v) {
		cout << "after FREE_OBJECT(OS)" << endl;
	}
	FREE_OBJECT(Coset_reps);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_set_from_file done" << endl;
	}
}


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



void algebra_global_with_action::classes_GL(finite_field *F, int d,
		int f_no_eigenvalue_one, int verbose_level)
{
	gl_classes C;
	gl_class_rep *R;
	int nb_classes;
	int i;


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

	sprintf(fname, "Class_reps_GL_%d_%d.tex", d, F->q);
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
	F->finite_field_init(q, 0);

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
	F->finite_field_init(q, 0);

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
	F->finite_field_init(q, 0);

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
	F->finite_field_init(q, 0);
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
		F->finite_field_init(q, 0);
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
// problem elt_idx does not describe the group element uniquely.
// Reason: the sims chain is not canonical.
{
	action *A;
	longinteger_object Go;
	finite_field *F;
	vector_ge *nice_gens;

	F = NEW_OBJECT(finite_field);
	F->finite_field_init(q, 0);

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
	F->finite_field_init(q, 0);

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
	F->finite_field_init(q, 0);
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

void algebra_global_with_action::compute_regular_representation(
		action *A, sims *S,
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

void algebra_global_with_action::presentation(
		action *A, sims *S, int goi,
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


void algebra_global_with_action::do_eigenstuff(finite_field *F, int size, int *Data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	discreta_matrix M;
	int i, j, k, a, h;
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

	//domain d(q);
	domain d(F);
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
	lv = F->inverse(l);
	cout << "leading coefficient inverse " << lv << endl;
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		c = F->mult(b, lv);
		charpoly.m_ii(i, c);
	}
	cout << "monic characteristic polynomial:" << charpoly << endl;

	integer x, y;
	int *roots;
	int nb_roots = 0;

	roots = new int[F->q];

	for (a = 0; a < F->q; a++) {
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
		eigenvalue_negative = F->negate(eigenvalue);
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
				size, size, size, F->log10_of_q);
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				a = A[i * size + j];
				if (j == i) {
					a = F->add(a, eigenvalue_negative);
				}
				B[i * size + j] = a;
			}
		}
		cout << "B = A - eigenvalue * I:" << endl;
		print_integer_matrix_width(cout, B,
				size, size, size, F->log10_of_q);

		cout << "B transposed:" << endl;
		F->transpose_matrix(B, Bt, size, size);
		print_integer_matrix_width(cout, Bt,
				size, size, size, F->log10_of_q);

		int f_special = FALSE;
		int f_complete = TRUE;
		int *base_cols;
		int nb_base_cols;
		int f_P = FALSE;
		int kernel_m, kernel_n, *kernel;

		base_cols = new int[size];
		kernel = new int[size * size];

		nb_base_cols = F->Gauss_int(Bt,
			f_special, f_complete, base_cols,
			f_P, NULL, size, size, size,
			verbose_level - 1);
		cout << "rank = " << nb_base_cols << endl;

		F->matrix_get_kernel(Bt, size, size, base_cols, nb_base_cols,
			kernel_m, kernel_n, kernel, 0 /* verbose_level */);
		cout << "kernel = left eigenvectors:" << endl;
		print_integer_matrix_width(cout, kernel,
				size, kernel_n, kernel_n, F->log10_of_q);

		int *vec1, *vec2;
		vec1 = new int[size];
		vec2 = new int[size];
		for (i = 0; i < size; i++) {
			vec1[i] = kernel[i * kernel_n + 0];
			}
		int_vec_print(cout, vec1, size);
		cout << endl;
		F->PG_element_normalize_from_front(vec1, 1, size);
		int_vec_print(cout, vec1, size);
		cout << endl;
		F->PG_element_rank_modified(vec1, 1, size, a);
		cout << "has rank " << a << endl;


		cout << "computing xA" << endl;

		F->mult_vector_from_the_left(vec1, A, vec2, size, size);
		int_vec_print(cout, vec2, size);
		cout << endl;
		F->PG_element_normalize_from_front(vec2, 1, size);
		int_vec_print(cout, vec2, size);
		cout << endl;
		F->PG_element_rank_modified(vec2, 1, size, a);
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
	Surf_A->init(Surf, LG, TRUE /* f_recoordinatize */, verbose_level - 3);
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

void algebra_global_with_action::centralizer_of_element(
		action *A, sims *S,
		std::string &element_description,
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element label=" << label
				<< " element_description=" << element_description << endl;
	}

	prefix.assign(A->label);
	prefix.append("_elt_");
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


	string fname;

	fname.assign(prefix);
	fname.append("_centralizer.tex");


	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Centralizer of element %s", label.c_str());
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::centralizer_of_element before report" << endl;
			}
			gens->print_generators_tex(ost);

			if (f_v) {
				cout << "algebra_global_with_action::centralizer_of_element after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	FREE_int(data);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element done" << endl;
	}
}

void algebra_global_with_action::normalizer_of_cyclic_subgroup(
		action *A, sims *S,
		std::string &element_description,
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup label=" << label
				<< " element_description=" << element_description << endl;
	}

	prefix.assign("normalizer_of_");
	prefix.append(label);
	prefix.append("_in_");
	prefix.append(A->label);

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
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup label=" << label
				<< " element order=" << o << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup Elt:" << endl;
		A->element_print_quick(Elt, cout);
		cout << endl;
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


	string fname;

	fname.assign(prefix);
	fname.append(".tex");


	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Normalizer of cyclic subgroup %s", label.c_str());
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);

			longinteger_object go;
			gens->group_order(go);
			ost << "The subgroup generated by " << endl;
			ost << "$$" << endl;
			A->element_print_latex(Elt, ost);
			ost << "$$" << endl;
			ost << "has order " << o << "\\\\" << endl;
			ost << "The normalizer has order " << go << "\\\\" << endl;
			if (f_v) {
				cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup before report" << endl;
			}
			gens->print_generators_tex(ost);

			if (f_v) {
				cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}





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

void algebra_global_with_action::do_orbits_on_polynomials(
		linear_group *LG,
		int degree_of_poly,
		int f_recognize, std::string &recognize_text,
		int f_draw_tree, int draw_tree_idx, layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "algebra_global_with_action::do_orbits_on_polynomials" << endl;
	}

	orbits_on_polynomials *O;

	O = NEW_OBJECT(orbits_on_polynomials);

	O->init(LG,
			degree_of_poly,
			f_recognize, recognize_text,
			verbose_level);

	if (f_draw_tree) {

		string fname;
		char str[1000];


		sprintf(str, "_orbit_%d_tree", draw_tree_idx);

		fname.assign(O->fname_base);
		fname.append(str);

		O->Sch->draw_tree(fname,
				Opt,
				draw_tree_idx,
				FALSE /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);
	}

	O->report(verbose_level);

	FREE_OBJECT(O);


	if (f_v) {
		cout << "algebra_global_with_action::do_orbits_on_polynomials done" << endl;
	}
}

void algebra_global_with_action::representation_on_polynomials(
		linear_group *LG,
		int degree_of_poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_stabilizer = TRUE;
	//int f_draw_tree = TRUE;


	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials" << endl;
	}


	finite_field *F;
	action *A;
	//matrix_group *M;
	int n;
	//int degree;
	longinteger_object go;

	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

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


	action_on_homogeneous_polynomials *A_on_HPD;
	int *M;
	int nb_gens;
	int i;

	A_on_HPD = A2->G.OnHP;

	if (LG->f_has_nice_gens) {
		if (f_v) {
			cout << "algebra_global_with_action::representation_on_polynomials using nice generators" << endl;
		}
		LG->nice_gens->matrix_representation(A_on_HPD, M, nb_gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "algebra_global_with_action::representation_on_polynomials using strong generators" << endl;
		}
		LG->Strong_gens->gens->matrix_representation(A_on_HPD, M, nb_gens, verbose_level);
	}

	for (i = 0; i < nb_gens; i++) {
		cout << "matrix " << i << " / " << nb_gens << ":" << endl;
		int_matrix_print(M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);
	}

	for (i = 0; i < nb_gens; i++) {
		string fname;
		char str[1000];
		file_io Fio;

		fname.assign(LG->label);
		sprintf(str, "_rep_%d_%d.csv", degree_of_poly, i);
		fname.append(str);
		Fio.int_matrix_write_csv(fname, M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials done" << endl;
	}
}



void algebra_global_with_action::do_eigenstuff_with_coefficients(
		finite_field *F, int n, std::string &coeffs_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients" << endl;
	}
	int *Data;
	int len;

	int_vec_scan(coeffs_text, Data, len);
	if (len != n * n) {
		cout << "len != n * n " << len << endl;
		exit(1);
	}

	algebra_global_with_action A;

	A.do_eigenstuff(F, n, Data, verbose_level);

	FREE_int(Data);
	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients done" << endl;
	}
}

void algebra_global_with_action::do_eigenstuff_from_file(
		finite_field *F, int n, std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_from_file" << endl;
	}

	file_io Fio;
	int *Data;
	int mtx_m, mtx_n;

	Fio.int_matrix_read_csv(fname, Data, mtx_m, mtx_n, verbose_level - 1);
	if (mtx_m != n) {
		cout << "mtx_m != n" << endl;
		exit(1);
	}
	if (mtx_n != n) {
		cout << "mtx_n != n" << endl;
		exit(1);
	}

	algebra_global_with_action A;

	A.do_eigenstuff(F, n, Data, verbose_level);


	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_from_file done" << endl;
	}
}


void algebra_global_with_action::do_cheat_sheet_for_decomposition_by_element_PG(finite_field *F,
		int n,
		int decomposition_by_element_power,
		std::string &decomposition_by_element_data, std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "algebra_global_with_action::do_cheat_sheet_for_decomposition_by_element_PG verbose_level="
				<< verbose_level << endl;
	}



	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "algebra_global_with_action::do_cheat_sheet_for_decomposition_by_element_PG before PA->init" << endl;
	}
	PA->init(F, n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "algebra_global_with_action::do_cheat_sheet_for_decomposition_by_element_PG after PA->init" << endl;
	}



	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];
		//int f_with_group = FALSE;
		//int f_semilinear = FALSE;
		//int f_basis = TRUE;
		//int q = F->q;

		snprintf(str, 1000, "PG_%d_%d.tex", n, F->q);
		fname.assign(str);
		snprintf(title, 1000, "Cheat Sheet PG($%d,%d$)", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::do_cheat_sheet_for_decomposition_by_element_PG f_decomposition_by_element" << endl;
			}

			int *Elt;

			Elt = NEW_int(PA->A->elt_size_in_int);


			PA->A->make_element_from_string(Elt,
					decomposition_by_element_data, verbose_level);


			PA->A->element_power_int_in_place(Elt,
					decomposition_by_element_power, verbose_level);

			PA->report_decomposition_by_single_automorphism(
					Elt, ost, fname_base,
					verbose_level);

			FREE_int(Elt);


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	FREE_OBJECT(PA);

	if (f_v) {
		cout << "algebra_global_with_action::do_cheat_sheet_for_decomposition_by_element_PG done" << endl;
	}

}

void algebra_global_with_action::do_canonical_form_PG(finite_field *F,
		projective_space_object_classifier_description *Canonical_form_PG_Descr,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "algebra_global_with_action::do_canonical_form_PG" << endl;
	}



	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	PA->init(F, n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);



	projective_space_object_classifier *OC;

	OC = NEW_OBJECT(projective_space_object_classifier);

	if (f_v) {
		cout << "algebra_global_with_action::do_canonical_form_PG before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_PG_Descr,
			PA,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_canonical_form_PG after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);
	FREE_OBJECT(PA);



	if (f_v) {
		cout << "algebra_global_with_action::do_canonical_form_PG done" << endl;
	}
}

void algebra_global_with_action::do_study_surface(finite_field *F, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_study_surface" << endl;
	}

	surface_study *study;

	study = NEW_OBJECT(surface_study);

	cout << "before study->init" << endl;
	study->init(F, nb, verbose_level);
	cout << "after study->init" << endl;

	cout << "before study->study_intersection_points" << endl;
	study->study_intersection_points(verbose_level);
	cout << "after study->study_intersection_points" << endl;

	cout << "before study->study_line_orbits" << endl;
	study->study_line_orbits(verbose_level);
	cout << "after study->study_line_orbits" << endl;

	cout << "before study->study_group" << endl;
	study->study_group(verbose_level);
	cout << "after study->study_group" << endl;

	cout << "before study->study_orbits_on_lines" << endl;
	study->study_orbits_on_lines(verbose_level);
	cout << "after study->study_orbits_on_lines" << endl;

	cout << "before study->study_find_eckardt_points" << endl;
	study->study_find_eckardt_points(verbose_level);
	cout << "after study->study_find_eckardt_points" << endl;

#if 0
	if (study->nb_Eckardt_pts == 6) {
		cout << "before study->study_surface_with_6_eckardt_points" << endl;
		study->study_surface_with_6_eckardt_points(verbose_level);
		cout << "after study->study_surface_with_6_eckardt_points" << endl;
		}
#endif

	if (f_v) {
		cout << "algebra_global_with_action::do_study_surface done" << endl;
	}
}

void algebra_global_with_action::do_cubic_surface_properties(
		linear_group *LG,
		std::string fname_csv, int defining_q,
		int column_offset,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties" << endl;
	}

	int i;
	finite_field *F0;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	projective_space_with_action *PA;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;
	int f_semilinear;



	F0 = NEW_OBJECT(finite_field);
	F0->finite_field_init(defining_q, 0);

	F = LG->F;

	f_semilinear = LG->A_linear->is_semilinear_matrix_group();


	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, verbose_level - 1);
	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, LG, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"after Surf_A->init" << endl;
	}

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"before PA->init" << endl;
	}
	PA->init(
		F, 3 /*n*/, f_semilinear,
		TRUE /* f_init_incidence_structure */,
		verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"after PA->init" << endl;
	}



	long int *M;
	int nb_orbits, n;

	Fio.lint_matrix_read_csv(fname_csv, M, nb_orbits, n, verbose_level);

	if (n != 3 + column_offset) {
		cout << "algebra_global_with_action::do_cubic_surface_properties "
				"n != 3 + column_offset" << endl;
		exit(1);
	}

	int orbit_idx;

	long int *Orbit;
	long int *Rep;
	long int *Stab_order;
	long int *Orbit_length;
	long int *Nb_pts;
	long int *Nb_lines;
	long int *Nb_Eckardt_points;
	long int *Nb_singular_pts;
	long int *Nb_Double_points;
	long int *Ago;

	Orbit = NEW_lint(nb_orbits);
	Rep = NEW_lint(nb_orbits);
	Stab_order = NEW_lint(nb_orbits);
	Orbit_length = NEW_lint(nb_orbits);
	Nb_pts = NEW_lint(nb_orbits);
	Nb_lines = NEW_lint(nb_orbits);
	Nb_Eckardt_points = NEW_lint(nb_orbits);
	Nb_singular_pts = NEW_lint(nb_orbits);
	Nb_Double_points = NEW_lint(nb_orbits);
	Ago = NEW_lint(nb_orbits);

	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (f_v) {
			cout << "algebra_global_with_action::do_cubic_surface_properties "
					"orbit_idx = " << orbit_idx << " / " << nb_orbits << endl;
		}
		int coeff20[20];
		char str[1000];


		Orbit[orbit_idx] = M[orbit_idx * n + 0];
		Rep[orbit_idx] = M[orbit_idx * n + column_offset + 0];
		Stab_order[orbit_idx] = M[orbit_idx * n + column_offset + 1];
		Orbit_length[orbit_idx] = M[orbit_idx * n + column_offset + 2];

		cout << "Rep=" << Rep[orbit_idx] << endl;
		F0->PG_element_unrank_modified_lint(coeff20, 1, 20, Rep[orbit_idx]);
		cout << "coeff20=";
		int_vec_print(cout, coeff20, 20);
		cout << endl;

		surface_create_description *Descr;

		Descr = NEW_OBJECT(surface_create_description);
		Descr->f_q = TRUE;
		Descr->q = F->q;
		Descr->f_by_coefficients = TRUE;
		sprintf(str, "%d,0", coeff20[0]);
		Descr->coefficients_text.assign(str);
		for (i = 1; i < 20; i++) {
			sprintf(str, ",%d,%d", coeff20[i], i);
			Descr->coefficients_text.append(str);
		}
		cout << "Descr->coefficients_text = " << Descr->coefficients_text << endl;


		surface_create *SC;
		SC = NEW_OBJECT(surface_create);

		if (f_v) {
			cout << "algebra_global_with_action::do_cubic_surface_properties "
					"before SC->init" << endl;
		}
		SC->init(Descr, Surf_A, verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::do_cubic_surface_properties "
					"after SC->init" << endl;
		}


		if (SC->F->e == 1) {
			SC->F->f_print_as_exponentials = FALSE;
		}

		SC->F->PG_element_normalize(SC->SO->eqn, 1, 20);

		if (f_v) {
			cout << "algebra_global_with_action::do_cubic_surface_properties "
					"We have created the following surface:" << endl;
			cout << "$$" << endl;
			SC->Surf->print_equation_tex(cout, SC->SO->eqn);
			cout << endl;
			cout << "$$" << endl;

			cout << "$$" << endl;
			int_vec_print(cout, SC->SO->eqn, 20);
			cout << endl;
			cout << "$$" << endl;
		}


		// compute the group of the surface if we are over a small field.
		// Otherwise we don't, because it would take too long.


		if (F->q <= 8) {
			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties "
						"before SC->compute_group" << endl;
			}
			SC->compute_group(PA, verbose_level);
			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties "
						"after SC->compute_group" << endl;
			}
			Ago[orbit_idx] = SC->Sg->group_order_as_lint();
		}
		else {
			Ago[orbit_idx] = 0;
		}


		Nb_pts[orbit_idx] = SC->SO->nb_pts;
		Nb_lines[orbit_idx] = SC->SO->nb_lines;
		Nb_Eckardt_points[orbit_idx] = SC->SO->SOP->nb_Eckardt_points;
		Nb_singular_pts[orbit_idx] = SC->SO->SOP->nb_singular_pts;
		Nb_Double_points[orbit_idx] = SC->SO->SOP->nb_Double_points;

		//SC->SO->SOP->print_everything(ost, verbose_level);






		FREE_OBJECT(SC);
		FREE_OBJECT(Descr);


	}


	string fname_data;

	fname_data.assign(fname_csv);
	chop_off_extension(fname_data);

	char str[1000];
	sprintf(str, "_F%d.csv", F->q);
	fname_data.append(str);

	long int *Vec[10];
	char str_A[1000];
	char str_P[1000];
	char str_L[1000];
	char str_E[1000];
	char str_S[1000];
	char str_D[1000];
	sprintf(str_A, "Ago-%d", F->q);
	sprintf(str_P, "Nb_P-%d", F->q);
	sprintf(str_L, "Nb_L-%d", F->q);
	sprintf(str_E, "Nb_E-%d", F->q);
	sprintf(str_S, "Nb_S-%d", F->q);
	sprintf(str_D, "Nb_D-%d", F->q);
	const char *column_label[] = {
			"Orbit_idx",
			"Rep",
			"StabOrder",
			"OrbitLength",
			str_A,
			str_P,
			str_L,
			str_E,
			str_S,
			str_D,
	};

	Vec[0] = Orbit;
	Vec[1] = Rep;
	Vec[2] = Stab_order;
	Vec[3] = Orbit_length;
	Vec[4] = Ago;
	Vec[5] = Nb_pts;
	Vec[6] = Nb_lines;
	Vec[7] = Nb_Eckardt_points;
	Vec[8] = Nb_singular_pts;
	Vec[9] = Nb_Double_points;

	Fio.lint_vec_array_write_csv(10 /* nb_vecs */, Vec, nb_orbits,
			fname_data, column_label);

	if (f_v) {
		cout << "Written file " << fname_data << " of size "
				<< Fio.file_size(fname_data) << endl;
	}



	FREE_lint(M);
	FREE_OBJECT(PA);
	FREE_OBJECT(F0);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties done" << endl;
	}
}


struct cubic_surface_data_set {

	int orbit_idx;
	long int Orbit_idx;
	long int Rep;
	long int Stab_order;
	long int Orbit_length;
	long int Ago;
	long int Nb_pts;
	long int Nb_lines;
	long int Nb_Eckardt_points;
	long int Nb_singular_pts;
	long int Nb_Double_points;

};

void algebra_global_with_action::do_cubic_surface_properties_analyze(
		linear_group *LG,
		std::string fname_csv, int defining_q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze" << endl;
	}

	//int i;
	finite_field *F0;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	projective_space_with_action *PA;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;
	int f_semilinear;



	F0 = NEW_OBJECT(finite_field);
	F0->finite_field_init(defining_q, 0);

	F = LG->F;

	f_semilinear = LG->A_linear->is_semilinear_matrix_group();


	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
				"before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, verbose_level - 1);
	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, LG, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
				"after Surf_A->init" << endl;
	}

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
				"before PA->init" << endl;
	}
	PA->init(
		F, 3 /*n*/, f_semilinear,
		TRUE /* f_init_incidence_structure */,
		verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
				"after PA->init" << endl;
	}

	int nb_orbits, n;
	int orbit_idx;
	struct cubic_surface_data_set *Data;

	{
		long int *M;

		Fio.lint_matrix_read_csv(fname_csv, M, nb_orbits, n, verbose_level);

		if (n != 10) {
			cout << "algebra_global_with_action::do_cubic_surface_properties_analyze n != 10" << endl;
			exit(1);
		}





		Data = new struct cubic_surface_data_set [nb_orbits];

		for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
			Data[orbit_idx].orbit_idx = orbit_idx;
			Data[orbit_idx].Orbit_idx = M[orbit_idx * n + 0];
			Data[orbit_idx].Rep = M[orbit_idx * n + 1];
			Data[orbit_idx].Stab_order = M[orbit_idx * n + 2];
			Data[orbit_idx].Orbit_length = M[orbit_idx * n + 3];
			Data[orbit_idx].Ago = M[orbit_idx * n + 4];
			Data[orbit_idx].Nb_pts = M[orbit_idx * n + 5];
			Data[orbit_idx].Nb_lines = M[orbit_idx * n + 6];
			Data[orbit_idx].Nb_Eckardt_points = M[orbit_idx * n + 7];
			Data[orbit_idx].Nb_singular_pts = M[orbit_idx * n + 8];
			Data[orbit_idx].Nb_Double_points = M[orbit_idx * n + 9];
		}
		FREE_lint(M);
	}
	long int *Nb_singular_pts;

	Nb_singular_pts = NEW_lint(nb_orbits);
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		Nb_singular_pts[orbit_idx] = Data[orbit_idx].Nb_singular_pts;
	}


	tally T_S;

	T_S.init_lint(Nb_singular_pts, nb_orbits, FALSE, 0);

	cout << "Classification by the number of singular points:" << endl;
	T_S.print(TRUE /* f_backwards */);

	{
		string fname_report;
		fname_report.assign(fname_csv);
		chop_off_extension(fname_report);
		fname_report.append("_report.tex");
		latex_interface L;
		file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

#if 0
			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
						"before get_A()->report" << endl;
			}

			if (!Descr->f_draw_options) {
				cout << "please use -draw_options" << endl;
				exit(1);
			}
			PA->A->report(ost,
					FALSE /* f_sims */,
					NULL, //A1/*LG->A_linear*/->Sims,
					FALSE /* f_strong_gens */,
					NULL,
					Descr->draw_options,
					verbose_level - 1);

			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
						"after LG->A_linear->report" << endl;
			}
#endif

			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
						"before report" << endl;
			}


			ost << "\\section{Surfaces over ${\\mathbb F}_{" << F->q << "}$}" << endl;


			ost << "Number of surfaces: " << nb_orbits << "\\\\" << endl;
			ost << "Classification by the number of singular points:" << endl;
			ost << "$$" << endl;
			T_S.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
			ost << "$$" << endl;


			ost << "\\section{Singular Surfaces}" << endl;

			report_singular_surfaces(ost, Data, nb_orbits, verbose_level);

			ost << "\\section{Nonsingular Surfaces}" << endl;

			report_non_singular_surfaces(ost, Data, nb_orbits, verbose_level);



			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
						"after report" << endl;
			}

			L.foot(ost);
		}
		cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}





	FREE_OBJECT(PA);
	FREE_OBJECT(F0);

	if (f_v) {
		cout << "algebra_global_with_action::do_cubic_surface_properties_analyze done" << endl;
	}
}

void algebra_global_with_action::report_singular_surfaces(std::ostream &ost,
		struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::report_singular_surfaces" << endl;
	}

	struct cubic_surface_data_set *Data_S;
	int nb_S, h, orbit_idx;


	nb_S = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts) {
			nb_S++;
		}
	}


	Data_S = new struct cubic_surface_data_set [nb_S];

	h = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts) {
			Data_S[h] = Data[orbit_idx];
			h++;
		}
	}
	if (h != nb_S) {
		cout << "h != nb_S" << endl;
		exit(1);
	}

	long int *Selected_Nb_lines;


	Selected_Nb_lines = NEW_lint(nb_S);


	for (h = 0; h < nb_S; h++) {
		Selected_Nb_lines[h] = Data_S[h].Nb_lines;
	}

	tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_S, FALSE, 0);

	ost << "Number of surfaces: " << nb_S << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
	ost << "$$" << endl;

	report_surfaces_by_lines(ost, Data_S, T_L, verbose_level);



	FREE_lint(Selected_Nb_lines);
	delete [] Data_S;

	if (f_v) {
		cout << "algebra_global_with_action::report_singular_surfaces done" << endl;
	}
}


void algebra_global_with_action::report_non_singular_surfaces(std::ostream &ost,
		struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::report_non_singular_surfaces" << endl;
	}

	struct cubic_surface_data_set *Data_NS;
	int nb_NS, h, orbit_idx;


	nb_NS = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts == 0) {
			nb_NS++;
		}
	}


	Data_NS = new struct cubic_surface_data_set [nb_NS];

	h = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts == 0) {
			Data_NS[h] = Data[orbit_idx];
			h++;
		}
	}
	if (h != nb_NS) {
		cout << "h != nb_NS" << endl;
		exit(1);
	}

	long int *Selected_Nb_lines;


	Selected_Nb_lines = NEW_lint(nb_NS);


	for (h = 0; h < nb_NS; h++) {
		Selected_Nb_lines[h] = Data_NS[h].Nb_lines;
	}

	for (h = 0; h < nb_NS; h++) {
		cout << h << " : " << Data_NS[h].orbit_idx << " : " << Data_NS[h].Nb_lines << endl;
	}

	tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_NS, FALSE, 0);

	ost << "Number of surfaces: " << nb_NS << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
	ost << "$$" << endl;


	report_surfaces_by_lines(ost, Data_NS, T_L, verbose_level);


	FREE_lint(Selected_Nb_lines);
	delete [] Data_NS;

	if (f_v) {
		cout << "algebra_global_with_action::report_non_singular_surfaces done" << endl;
	}
}

void algebra_global_with_action::report_surfaces_by_lines(std::ostream &ost,
		struct cubic_surface_data_set *Data, tally &T, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::report_surfaces_by_lines" << endl;
	}

	int i, j, f, l, a, idx;

	for (i = T.nb_types - 1; i >= 0; i--) {
		f = T.type_first[i];
		l = T.type_len[i];
		a = T.data_sorted[f];

		int nb_L;
		struct cubic_surface_data_set *Data_L;

		nb_L = l;

		Data_L = new struct cubic_surface_data_set [nb_L];

		ost << "The number of surfaces with exactly " << a << " lines is " << nb_L << ": \\\\" << endl;

		for (j = 0; j < l; j++) {
			idx = T.sorting_perm_inv[f + j];
			Data_L[j] = Data[idx];

		}


		for (j = 0; j < l; j++) {
			ost << j
					<< " : i=" << Data_L[j].orbit_idx
					<< " : id=" << Data_L[j].Orbit_idx
					<< " : P=" << Data_L[j].Nb_pts
					<< " : S=" << Data_L[j].Nb_singular_pts
					<< " : E=" << Data_L[j].Nb_Eckardt_points
					<< " : D=" << Data_L[j].Nb_Double_points
					<< " : ago=" << Data_L[j].Ago
					<< " : Rep=" << Data_L[j].Rep
				<< "\\\\" << endl;
		}

		delete [] Data_L;
	}
	if (f_v) {
		cout << "algebra_global_with_action::report_surfaces_by_lines done" << endl;
	}

}


void algebra_global_with_action::orbits_on_points(
		linear_group *LG,
		action *A2,
		int f_load_save,
		std::string &prefix,
		orbits_on_something *&Orb,
		//int f_stabilizer, int f_export_trees, int f_shallow_tree, int f_report,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points" << endl;
	}
	//cout << "computing orbits on points:" << endl;


#if 1

	//orbits_on_something *Orb;

	Orb = NEW_OBJECT(orbits_on_something);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points before Orb->init" << endl;
	}
	Orb->init(
			A2,
			LG->Strong_gens,
			f_load_save,
			prefix,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points after Orb->init" << endl;
	}



#else

	schreier *Sch;
	Sch = NEW_OBJECT(schreier);

	cout << "Strong generators are:" << endl;
	LG->Strong_gens->print_generators(cout);
	cout << "Strong generators in tex are:" << endl;
	LG->Strong_gens->print_generators_tex(cout);
	cout << "Strong generators as permutations are:" << endl;
	LG->Strong_gens->print_generators_as_permutations();




	//A->all_point_orbits(*Sch, verbose_level);
	A2->all_point_orbits_from_generators(*Sch,
			LG->Strong_gens,
			verbose_level);

	longinteger_object go;
	int orbit_idx;

	LG->Strong_gens->group_order(go);
	cout << "Computing stabilizers. Group order = " << go << endl;
	if (f_stabilizer) {
		for (orbit_idx = 0; orbit_idx < Sch->nb_orbits; orbit_idx++) {

			strong_generators *SG;

			SG = Sch->stabilizer_orbit_rep(
					LG->A_linear /*default_action*/,
					go,
					orbit_idx, 0 /*verbose_level*/);

			cout << "orbit " << orbit_idx << " / " << Sch->nb_orbits << ":" << endl;
			SG->print_generators_tex(cout);

		}
	}


	cout << "computing orbits on points done." << endl;


	if (f_report) {

	}
	{
		string fname;
		file_io Fio;
		int *orbit_reps;
		int i;


		fname.assign(A2->label);
		fname.append("_orbit_reps.csv");

		orbit_reps = NEW_int(Sch->nb_orbits);


		for (i = 0; i < Sch->nb_orbits; i++) {
			orbit_reps[i] = Sch->orbit[Sch->orbit_first[i]];
		}


		Fio.int_vec_write_csv(orbit_reps, Sch->nb_orbits, fname, "OrbRep");

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	{
		string fname;
		file_io Fio;
		int *orbit_reps;
		int i;


		fname.assign(A2->label);
		fname.append("_orbit_length.csv");

		orbit_reps = NEW_int(Sch->nb_orbits);


		for (i = 0; i < Sch->nb_orbits; i++) {
			orbit_reps[i] = Sch->orbit_len[i];
		}


		Fio.int_vec_write_csv(orbit_reps, Sch->nb_orbits, fname, "OrbLen");

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}



	cout << "before Sch->print_and_list_orbits." << endl;
	if (A2->degree < 1000) {
		Sch->print_and_list_orbits(cout);
	}
	else {
		cout << "The degree is too large." << endl;
	}

	string fname_orbits;
	file_io Fio;

	fname_orbits.assign(A2->label);
	fname_orbits.append("_orbits.tex");


	Sch->latex(fname_orbits);
	cout << "Written file " << fname_orbits << " of size "
			<< Fio.file_size(fname_orbits) << endl;



	if (f_export_trees) {
		string fname_tree_mask;

		fname_tree_mask.assign(A2->label);
		fname_tree_mask.append("_%d.layered_graph");

		for (orbit_idx = 0; orbit_idx < Sch->nb_orbits; orbit_idx++) {
			cout << "orbit " << orbit_idx << " / " <<  Sch->nb_orbits
					<< " before Sch->export_tree_as_layered_graph" << endl;
			Sch->export_tree_as_layered_graph(0 /* orbit_no */,
					fname_tree_mask,
					verbose_level - 1);
		}
	}

	if (f_shallow_tree) {
		orbit_idx = 0;
		schreier *shallow_tree;
		string fname_schreier_tree_mask;

		cout << "computing shallow Schreier tree for orbit " << orbit_idx << endl;

	#if 0
		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				//shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;
	#endif
		int f_randomized = TRUE;

		Sch->shallow_tree_generators(orbit_idx,
				f_randomized,
				shallow_tree,
				verbose_level);

		cout << "computing shallow Schreier tree done." << endl;

		fname_schreier_tree_mask.assign(A2->label);
		fname_schreier_tree_mask.append("_%d_shallow.layered_graph");

		shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_schreier_tree_mask,
				verbose_level - 1);
	}
#endif

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points done" << endl;
	}
}

void algebra_global_with_action::find_singer_cycle(linear_group *LG,
		action *A1, action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::find_singer_cycle" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;
	int i, d, q, cnt, ord, order;
	number_theory_domain NT;

	if (!A1->is_matrix_group()) {
		cout << "group_theoretic_activity::find_singer_cycle needs matrix group" << endl;
		exit(1);
	}
	matrix_group *M;

	M = A1->get_matrix_group();
	q = M->GFq->q;
	d = A1->matrix_group_dimension();

	if (A1->is_projective()) {
		order = (NT.i_power_j(q, d) - 1) / (q - 1);
	}
	else {
		order = NT.i_power_j(q, d) - 1;
	}
	if (f_v) {
		cout << "algebra_global_with_action::find_singer_cycle looking for an "
				"element of order " << order << endl;
	}

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->element_order(Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order) {
			continue;
		}
		if (!M->has_shape_of_singer_cycle(Elt)) {
			continue;
		}
		cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << " = " << cnt << ":" << endl;
		A2->element_print(Elt, cout);
		cout << endl;
		A2->element_print_as_permutation(Elt, cout);
		cout << endl;
		cnt++;
	}
	cout << "we found " << cnt << " group elements of order " << order << endl;

	FREE_int(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::find_singer_cycle done" << endl;
	}
}

void algebra_global_with_action::search_element_of_order(linear_group *LG,
		action *A1, action *A2,
		int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::search_element_of_order" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;
	int i, cnt, ord;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->element_order(Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order) {
			continue;
		}
		cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << " = " << cnt << ":" << endl;
		A2->element_print(Elt, cout);
		cout << endl;
		A2->element_print_as_permutation(Elt, cout);
		cout << endl;
		cnt++;
	}
	cout << "we found " << cnt << " group elements of order " << order << endl;

	FREE_int(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::search_element_of_order done" << endl;
	}
}

void algebra_global_with_action::element_rank(linear_group *LG,
		action *A1,
		std::string &elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::element_rank" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	cout << "creating element " << elt_data << endl;
	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);
	A1->make_element_from_string(Elt, elt_data, 0);

	cout << "Element :" << endl;
	A1->element_print(Elt, cout);
	cout << endl;

	longinteger_object a;
	H->element_rank(a, Elt);

	cout << "The rank of the element is " << a << endl;


	FREE_int(Elt);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "algebra_global_with_action::element_rank done" << endl;
	}
}

void algebra_global_with_action::element_unrank(linear_group *LG,
		action *A1,
		std::string &rank_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::element_unrank" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);


	longinteger_object a;

	a.create_from_base_10_string(rank_string.c_str(), 0 /*verbose_level*/);

	cout << "Creating element of rank " << a << endl;

	H->element_unrank(a, Elt);

	cout << "Element :" << endl;
	A1->element_print(Elt, cout);
	cout << endl;


	FREE_int(Elt);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "algebra_global_with_action::element_unrank done" << endl;
	}
}



}}
