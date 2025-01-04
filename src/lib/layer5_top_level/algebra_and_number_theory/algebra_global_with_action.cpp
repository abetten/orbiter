/*
 * algebra_global_with_action.cpp
 *
 *  Created on: Dec 15, 2019
 *      Author: betten
 */



#include "orbiter.h"


using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {

algebra_global_with_action::algebra_global_with_action()
{
	Record_birth();

}

algebra_global_with_action::~algebra_global_with_action()
{
	Record_death();

}

void algebra_global_with_action::orbits_under_conjugation(
		long int *the_set, int set_size,
		groups::sims *S,
		groups::strong_generators *SG,
		data_structures_groups::vector_ge *Transporter,
		int verbose_level)
// this is related to Betten, Topalova, Zhelezova 2021,
// packings in PG(3,4) invariant under an elementary group of order 4
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation" << endl;
	}
	actions::action *A_conj;
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"before create_induced_action_by_conjugation" << endl;
	}
	A_conj = S->A->Induced_action->create_induced_action_by_conjugation(S,
			false /* f_ownership */, false /* f_basis */, S,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"after create_induced_action_by_conjugation" << endl;
	}

	actions::action *A_conj_restricted;
	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_on_group");
	label_of_set_tex.assign("\\_on\\_group");

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"before A_conj->restricted_action" << endl;
	}

	A_conj_restricted = A_conj->Induced_action->restricted_action(
			the_set, set_size,
			label_of_set, label_of_set_tex,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"after A_conj->restricted_action" << endl;
	}



	groups::schreier Classes;
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
			SG,
			the_set, set_size, S, A_conj,
			&Classes,
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation "
				"after create_subgroups" << endl;
	}

	FREE_OBJECT(A_conj);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_under_conjugation done" << endl;
	}
}

void algebra_global_with_action::create_subgroups(
		groups::strong_generators *SG,
		long int *the_set, int set_size,
		groups::sims *S,
		actions::action *A_conj,
		groups::schreier *Classes,
		data_structures_groups::vector_ge *Transporter,
		int verbose_level)
// this is related to Betten, Topalova, Zhelezova 2021,
// packings in PG(3,4) invariant under an elementary abelian group of order 4
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::create_subgroups" << endl;
	}

	int i, j;
	int f, l, rep;
	long int *the_set_sorted;
	long int *position;
	other::data_structures::sorting Sorting;
	algebra::ring_theory::longinteger_object go;

	SG->group_order(go);
	if (f_v) {
		cout << "The group has order " << go << endl;
	}

	the_set_sorted = NEW_lint(set_size);
	position = NEW_lint(set_size);
	Lint_vec_copy(the_set, the_set_sorted, set_size);
	//Sorting.lint_vec_heapsort(the_set_sorted, set_size);
	for (i = 0; i < set_size; i++) {
		position[i] = i;
	}
	Sorting.lint_vec_heapsort_with_log(the_set_sorted, position, set_size);


	cout << "There are " << Classes->nb_orbits << " orbits "
			"on the given set of elements by conjugation" << endl;
	for (i = 0; i < Classes->nb_orbits; i++) {

		f = Classes->orbit_first[i];
		l = Classes->orbit_len[i];
		rep = Classes->orbit[f];
		if (f_v) {
			cout << "Orbit " << i << " has length " << l
					<< " representative is " << rep << " = " << the_set[rep] << endl;
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
	groups::strong_generators **Flag_stab;
	int *SO;
	int *SOL;
	int nb_reject;

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
	Flag_stab = new groups::pstrong_generators [Classes->nb_orbits];
	SO = NEW_int(Classes->nb_orbits);
	SOL = NEW_int(Classes->nb_orbits);

	nb_reject = 0;

	for (j = 1; j < Classes->nb_orbits; j++) {



		f = Classes->orbit_first[j];
		l = Classes->orbit_len[j];
		rep = Classes->orbit[f];
		rk1 = the_set[rep];
		rk2 = S->mult_by_rank(rk0, rk1, 0 /*verbose_level*/);

		if (Sorting.lint_vec_search(
				the_set_sorted, set_size, rk2, idx,
				0 /*verbose_level*/)) {
			cout << "flag orbit " << nb_flag_orbits << " : "
					<< j << " l=" << l << " : "
					<< rk0 << "," << rk1 << "," << rk2 << endl;

			S->element_unrank_lint(rk1, Elt1);
			S->element_unrank_lint(rk2, Elt2);
			S->A->Group_element->element_print_quick(Elt0, cout);
			S->A->Group_element->element_print_quick(Elt1, cout);
			S->A->Group_element->element_print_quick(Elt2, cout);

			Flags[nb_flag_orbits * 3 + 0] = rk0;
			Flags[nb_flag_orbits * 3 + 1] = rk1;
			Flags[nb_flag_orbits * 3 + 2] = rk2;
			SO[nb_flag_orbits] = j;
			SOL[nb_flag_orbits] = l;

			if (f_v) {
				cout << "algebra_global_with_action::create_subgroups "
						"before Classes->stabilizer_orbit_rep" << endl;
			}
			Flag_stab[nb_flag_orbits] = Classes->stabilizer_orbit_rep(
					S->A,
					go,
					j /* orbit_idx */, 0 /*verbose_level*/);
			if (f_v) {
				cout << "algebra_global_with_action::create_subgroups "
						"after Classes->stabilizer_orbit_rep" << endl;
			}


			nb_flag_orbits++;
		}
		else {
			cout << "Class " << j << " is rejected because the third element "
					"does not belong to the same class." << endl;
			nb_reject++;
		}

	}

	if (f_v) {
		cout << "We found " << nb_flag_orbits << " flag orbits, with "
				<< nb_reject << " may rejected" << endl;

		int h;

		for (h = 0; h < nb_flag_orbits; h++) {
			algebra::ring_theory::longinteger_object go1;

			cout << "flag orbit " << h << " / " << nb_flag_orbits << ":" << endl;
			Flag_stab[h]->group_order(go1);
			rk0 = Flags[h * 3 + 0];
			rk1 = Flags[h * 3 + 1];
			rk2 = Flags[h * 3 + 2];
			S->element_unrank_lint(rk0, Elt0);
			S->element_unrank_lint(rk1, Elt1);
			S->element_unrank_lint(rk2, Elt2);
			cout << h << " : " << SO[h] << " : " << SOL[h]
				<< " : (" << rk0 << "," << rk1 << "," << rk2 << ") : " << go1
				<< endl;
			cout << "The subgroup consists of the following three "
					"non-identity elements:" << endl;
			S->A->Group_element->element_print_quick(Elt0, cout);
			S->A->Group_element->element_print_quick(Elt1, cout);
			S->A->Group_element->element_print_quick(Elt2, cout);

			Flag_stab[h]->print_generators_tex(cout);

		}
	}

	int flag;
	int nb_iso;
	int *upstep_transversal_size;
	int *iso_type_of_flag_orbit;
	int *f_is_definition;
	int *flag_orbit_of_iso_type;
	int *f_fused;
	groups::strong_generators **Aut;
	long int cur_flag[3];
	long int cur_flag_mapped1[3];
	int h, pt;

	upstep_transversal_size = NEW_int(nb_flag_orbits);
	iso_type_of_flag_orbit = NEW_int(nb_flag_orbits);
	flag_orbit_of_iso_type = NEW_int(nb_flag_orbits);
	f_is_definition = NEW_int(nb_flag_orbits);
	f_fused = NEW_int(nb_flag_orbits);
	Int_vec_zero(f_is_definition, nb_flag_orbits);
	Int_vec_zero(f_fused, nb_flag_orbits);

	Aut = new groups::pstrong_generators [nb_flag_orbits];


	nb_iso = 0;
	for (flag = 0; flag < nb_flag_orbits; flag++) {

		if (f_v) {
			cout << "upstep: considering flag orbit " << flag << " / " << nb_flag_orbits
					<< " with a flag stabilizer of order "
					<< Flag_stab[flag]->group_order_as_lint() << endl;
		}

		if (f_fused[flag]) {
			if (f_v) {
				cout << "upstep: flag orbit " << flag << " / " << nb_flag_orbits
						<< " has been fused, skipping" << endl;
			}
			continue;
		}
		f_is_definition[flag] = true;
		iso_type_of_flag_orbit[flag] = nb_iso;
		flag_orbit_of_iso_type[nb_iso] = flag;
		upstep_transversal_size[nb_iso] = 1;

		data_structures_groups::vector_ge *transversal;

		transversal = NEW_OBJECT(data_structures_groups::vector_ge);
		transversal->init(S->A, 0);
		transversal->allocate(6, 0);
		for (h = 0; h < 6; h++) {
			S->A->Group_element->element_one(transversal->ith(h), 0);
		}

		for (h = 1; h < 6; h++) {
			if (h == 1) {
				cur_flag[0] = Flags[flag * 3 + 0];
				cur_flag[1] = Flags[flag * 3 + 2];
				cur_flag[2] = Flags[flag * 3 + 1];
			}
			else if (h == 2) {
				cur_flag[0] = Flags[flag * 3 + 1];
				cur_flag[1] = Flags[flag * 3 + 0];
				cur_flag[2] = Flags[flag * 3 + 2];
			}
			else if (h == 3) {
				cur_flag[0] = Flags[flag * 3 + 1];
				cur_flag[1] = Flags[flag * 3 + 2];
				cur_flag[2] = Flags[flag * 3 + 0];
			}
			else if (h == 4) {
				cur_flag[0] = Flags[flag * 3 + 2];
				cur_flag[1] = Flags[flag * 3 + 0];
				cur_flag[2] = Flags[flag * 3 + 1];
			}
			else if (h == 5) {
				cur_flag[0] = Flags[flag * 3 + 2];
				cur_flag[1] = Flags[flag * 3 + 1];
				cur_flag[2] = Flags[flag * 3 + 0];
			}

			// move cur_flag[0] to the_set[0] using the inverse of Transporter

			if (!Sorting.lint_vec_search(the_set_sorted,
					set_size, cur_flag[0], idx, 0 /*verbose_level*/)) {
				cout << "cannot find cur_flag[0] in the_set_sorted" << endl;
				exit(1);
			}
			pt = position[idx];
			S->A->Group_element->element_invert(Transporter->ith(pt), Elt0, 0);
			for (int u = 0; u < 3; u++) {
				cur_flag_mapped1[u] = A_conj->Group_element->element_image_of(
						cur_flag[u], Elt0, 0);
			}
			if (cur_flag_mapped1[0] != rk0) {
				cout << "cur_flag_mapped1[u] != rk0" << endl;
				exit(1);
			}



			if (!Sorting.lint_vec_search(
					the_set_sorted,
					set_size, cur_flag_mapped1[1], idx,
					0 /*verbose_level*/)) {
				cout << "cannot find cur_flag[1] in the_set_sorted" << endl;
				exit(1);
			}
			pt = position[idx];
			j = Classes->orbit_number(pt);


			if (j == SO[flag]) {
				cout << "flag " << flag << " coset " << h << ", found an automorphism" << endl;

				int orbit_idx;

				Classes->transporter_from_point_to_orbit_rep(
						pt,
						orbit_idx, Elt1, verbose_level);

				S->A->Group_element->element_mult(Elt0, Elt1, Elt2, 0);
				S->A->Group_element->element_print_quick(Elt2, cout);

				for (int u = 0; u < 3; u++) {
					cur_flag_mapped1[u] = A_conj->Group_element->element_image_of(
							cur_flag[u], Elt2, 0);
				}
				cout << "which maps as follows:" << endl;
				for (int u = 0; u < 3; u++) {
					cout << cur_flag[u] << " -> " << cur_flag_mapped1[u] << endl;
				}

				S->A->Group_element->element_move(
						Elt2, transversal->ith(
						upstep_transversal_size[nb_iso]), 0);

				upstep_transversal_size[nb_iso]++;
			}
			else {
				if (!Sorting.int_vec_search(SO, nb_flag_orbits, j, idx)) {
					cout << "cannot find j in SO" << endl;
					exit(1);
				}
				cout << "flag " << flag << " coset " << h
						<< ", fusing with flag " << idx << endl;
				f_fused[idx] = true;
			}

		}

		groups::strong_generators *aut;

		aut = NEW_OBJECT(groups::strong_generators);

		if (f_v) {
			cout << "flag " << flag << " stab order = "
					<< Flag_stab[flag]->group_order_as_lint()
					<< " upstep ransversal length = "
					<< upstep_transversal_size[nb_iso] << endl;
			cout << "before aut->init_group_extension" << endl;
		}
		aut->init(S->A);
		aut->init_group_extension(Flag_stab[flag],
				transversal, upstep_transversal_size[nb_iso] /* index */,
				verbose_level);

		cout << "created a stabilizer of order " << aut->group_order_as_lint() << endl;

		Aut[nb_iso] = aut;

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
				<< upstep_transversal_size[i] << " : "
				<< Aut[i]->group_order_as_lint() << endl;

		S->element_unrank_lint(rk0, Elt0);
		S->element_unrank_lint(rk1, Elt1);
		S->element_unrank_lint(rk2, Elt2);

		cout << "The subgroup consists of the following three "
				"non-identity elements:" << endl;
		S->A->Group_element->element_print_quick(Elt0, cout);
		S->A->Group_element->element_print_quick(Elt1, cout);
		S->A->Group_element->element_print_quick(Elt2, cout);

		Aut[i]->print_generators_tex(cout);

	}


	{
		string fname;
		string title, author, extra_praeamble;

		fname = "subgroups_of_order_4.tex";
		title = "Subgroups of order 4";


		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::create_subgroups "
						"before report" << endl;
			}
#if 1
			int h;
			ost << "There are " << nb_flag_orbits << " flag orbits:\\\\" << endl;
			for (h = 0; h < nb_flag_orbits; h++) {
				algebra::ring_theory::longinteger_object go1;

				ost << "Flag orbit " << h << " / " << nb_flag_orbits << ":\\\\" << endl;
				Flag_stab[h]->group_order(go1);
				rk0 = Flags[h * 3 + 0];
				rk1 = Flags[h * 3 + 1];
				rk2 = Flags[h * 3 + 2];
				S->element_unrank_lint(rk0, Elt0);
				S->element_unrank_lint(rk1, Elt1);
				S->element_unrank_lint(rk2, Elt2);
				cout << h << " : " << SO[h] << " : " << SOL[h]
					<< " : (" << rk0 << "," << rk1 << "," << rk2 << ") : " << go1 << endl;
				ost << "The subgroup consists of the following three "
						"non-identity elements:\\\\" << endl;
				ost << "$$" << endl;
				S->A->Group_element->element_print_latex(Elt0, ost);
				S->A->Group_element->element_print_latex(Elt1, ost);
				S->A->Group_element->element_print_latex(Elt2, ost);
				ost << "$$" << endl;
				ost << "The flag stabilizer is the following group:\\\\" << endl;
				Flag_stab[h]->print_generators_tex(ost);

			}

			ost << "\\bigskip" << endl;
#endif

			ost << "We found " << nb_iso
					<< " conjugacy classes of subgroups\\\\" << endl;
			ost << "Subgroup  : Order of normalizer\\\\" << endl;
			for (i = 0; i < nb_iso; i++) {
				ost << i << " : " << Aut[i]->group_order_as_lint() << "\\\\" << endl;
			}

			ost << "\\bigskip" << endl;

			for (i = 0; i < nb_iso; i++) {
				ost << "Subgroup " << i << " / " << nb_iso << ":\\\\" << endl;
				flag = flag_orbit_of_iso_type[i];
				rk0 = Flags[flag * 3 + 0];
				rk1 = Flags[flag * 3 + 1];
				rk2 = Flags[flag * 3 + 2];
				cout << i << " : " << flag << " : "
						<<  " : " << SO[flag] << " l=" << SOL[flag]
						<< " : " << rk0 << "," << rk1 << "," << rk2 << " : "
						<< upstep_transversal_size[i] << " : "
						<< Aut[i]->group_order_as_lint() << endl;

				S->element_unrank_lint(rk0, Elt0);
				S->element_unrank_lint(rk1, Elt1);
				S->element_unrank_lint(rk2, Elt2);

				ost << "The subgroup consists of the following three "
						"non-identity elements:\\\\" << endl;
				ost << "$$" << endl;
				S->A->Group_element->element_print_latex(Elt0, ost);
				S->A->Group_element->element_print_latex(Elt1, ost);
				S->A->Group_element->element_print_latex(Elt2, ost);
				ost << "$$" << endl;
				S->A->Group_element->element_print_for_make_element(Elt0, ost);
				ost << "\\\\" << endl;
				S->A->Group_element->element_print_for_make_element(Elt1, ost);
				ost << "\\\\" << endl;
				S->A->Group_element->element_print_for_make_element(Elt2, ost);
				ost << "\\\\" << endl;
				ost << "The normalizer is the following group:\\\\" << endl;
				Aut[i]->print_generators_tex(ost);

				ost << "\\bigskip" << endl;

			}


			if (f_v) {
				cout << "algebra_global_with_action::create_subgroups after report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
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




void algebra_global_with_action::classes_GL(
		algebra::field_theory::finite_field *F,
		int d, int f_no_eigenvalue_one, int verbose_level)
// called from interface_algebra
// creates an object of type action
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::classes_GL" << endl;
	}

	algebra::linear_algebra::gl_classes C;
	algebra::linear_algebra::gl_class_rep *R;
	int nb_classes;
	int i;


	C.init(d, F, verbose_level);

	C.make_classes(R, nb_classes, f_no_eigenvalue_one, verbose_level);

	actions::action *A;
	algebra::ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;
	int a;
	int *Mtx;
	int *Elt;



	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);


	actions::action *A_on_lines;

	A_on_lines = A->Induced_action->induced_action_on_grassmannian(
			2, verbose_level);

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
		cout << "algebra_global_with_action::classes_GL "
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
		cout << "algebra_global_with_action::classes_GL "
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
		cout << "algebra_global_with_action::classes_GL done" << endl;
	}
}


#if 0
void algebra_global_with_action::do_normal_form(
		int q, int d,
		int f_no_eigenvalue_one, int *data, int data_sz,
		int verbose_level)
// ToDo: move this
// not called from anywhere at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form" << endl;
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

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form "
				"before C.init" << endl;
	}
	C.init(d, F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form "
				"after C.init" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form "
				"before C.make_classes" << endl;
	}
	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form "
				"after C.make_classes" << endl;
	}



	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_projective_group(
			d /* n */, F,
			false /* f_semilinear */,
			true /* f_basis */,
			true /* f_init_sims */,
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
	Int_vec_print(cout, data, data_sz);
	cout << endl;

	//A->Sims->element_unrank_int(elt_idx, Elt);
	A->Group_element->make_element(Elt, data, verbose_level);

	cout << "Looking at element:" << endl;
	Int_matrix_print(Elt, d, d);


	linear_algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(linear_algebra::gl_class_rep);

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
	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form done" << endl;
	}
}


void algebra_global_with_action::do_identify_one(
		int q, int d,
		int f_no_eigenvalue_one, int elt_idx,
		int verbose_level)
// ToDo: move this
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_identify_one" << endl;
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
		cout << "algebra_global_with_action::do_identify_one done" << endl;
	}
}

void algebra_global_with_action::do_identify_all(
		int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
// ToDo: move this
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_identify_all" << endl;
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
		cout << "algebra_global_with_action::do_identify_all done" << endl;
	}
}

void algebra_global_with_action::do_random(
		int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
// ToDo: move this
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_random" << endl;
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
		cout << "algebra_global_with_action::do_random done" << endl;
	}
}


void algebra_global_with_action::group_table(
		int q, int d, int f_poly, std::string &poly,
		int f_no_eigenvalue_one, int verbose_level)
// This function does too many things!
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::group_table" << endl;
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
		cout << "algebra_global_with_action::group_table done" << endl;
	}
}

void algebra_global_with_action::centralizer_in_PGL_d_q_single_element_brute_force(
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
		cout << "algebra_global_with_action::centralizer_in_PGL_d_q_single_element_brute_force" << endl;
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
		cout << "algebra_global_with_action::centralizer_in_PGL_d_q_single_element_brute_force done" << endl;
	}
}


void algebra_global_with_action::centralizer_in_PGL_d_q_single_element(
		int q, int d,
		int elt_idx, int verbose_level)
// this function is not used anywhere
// ToDo: move this
// creates a finite_field, and two actions
// using init_projective_group and init_general_linear_group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_in_PGL_d_q_single_element" << endl;
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
		cout << "algebra_global_with_action::centralizer_in_PGL_d_q_single_element done" << endl;
	}

}

void algebra_global_with_action::compute_centralizer_of_all_elements_in_PGL_d_q(
		int q, int d, int verbose_level)
// this function is not used anywhere
// ToDo: move this
// creates a finite_field, and an action
// using init_projective_group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::compute_centralizer_of_all_elements_in_PGL_d_q" << endl;
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
		cout << "algebra_global_with_action::compute_centralizer_of_all_elements_in_PGL_d_q done" << endl;
	}
}
#endif


void algebra_global_with_action::compute_regular_representation(
		actions::action *A, groups::sims *S,
		data_structures_groups::vector_ge *SG, int *&perm,
		int verbose_level)
// allocates perm[SG->len * goi]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::compute_regular_representation" << endl;
	}
	algebra::ring_theory::longinteger_object go;
	int goi, i;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	S->group_order(go);
	goi = go.as_int();

	if (f_v) {
	cout << "computing the regular representation of degree "
			<< go << ":" << endl;
	}

	perm = NEW_int(SG->len * goi);

	for (i = 0; i < SG->len; i++) {
		S->regular_representation(SG->ith(i),
				perm + i * goi, verbose_level);
	}

	if (f_v) {
		cout << endl;
		for (i = 0; i < SG->len; i++) {
			Combi.Permutations->perm_print_offset(
					cout,
				perm + i * goi, goi, 1 /* offset */,
				false /* f_print_cycles_of_length_one */,
				false /* f_cycle_length */, false, 0,
				true /* f_orbit_structure */,
				NULL, NULL);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "algebra_global_with_action::compute_regular_representation done" << endl;
	}
}

#if 0
void algebra_global_with_action::presentation(
		actions::action *A, groups::sims *S, int goi,
		data_structures_groups::vector_ge *gens, int *primes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::presentation" << endl;
	}
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

	if (f_v) {
		cout << "presentation of length " << l << endl;
		cout << "primes: ";
		Int_vec_print(cout, primes, l);
		cout << endl;
	}

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
		A->Group_element->one(Elt1);
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
				A->Group_element->mult(Elt1, gens->ith(k), Elt2);
				A->Group_element->move(Elt2, Elt1);
				b--;
			}
		}

		A->Group_element->move(Elt1, Elt2);
		a = S->element_rank_lint(Elt2);
		word_list[i] = a;
		inverse_word_list[a] = i;

		if (f_v) {
			cout << "word " << i << " = ";
			Int_vec_print(cout, word, 9);
			cout << " gives " << endl;
			A->Group_element->print(cout, Elt1);
			cout << "which is element " << word_list[i] << endl;
			cout << endl;
		}
	}
	if (f_v) {
		cout << "i : word_list[i] : inverse_word_list[i]" << endl;
		for (i = 0; i < goi; i++) {
			cout << setw(5) << i << " : " << setw(5) << word_list[i]
				<< " : " << setw(5) << inverse_word_list[i] << endl;
		}
	}



	if (f_v) {
		for (i = 0; i < l; i++) {
			cout << "generator " << i << ":" << endl;
			A->Group_element->print(cout, gens->ith(i));
			cout << endl;
		}
	}
	for (i = 0; i < l; i++) {
		A->Group_element->move(gens->ith(i), Elt1);
		A->Group_element->element_power_int_in_place(Elt1, primes[i], 0);

		a = S->element_rank_lint(Elt1);

		j = inverse_word_list[a];
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
		}

		if (f_v) {
			cout << "generator " << i << " to the power " << primes[i]
				<< " is elt " << a << " which is word "
				<< inverse_word_list[a];
			Int_vec_print(cout, word, l);
			cout << " :" << endl;
			A->Group_element->print(cout, Elt1);
			cout << endl;
		}
	}


	for (i = 0; i < l; i++) {
		A->Group_element->move(gens->ith(i), Elt1);
		A->Group_element->invert(Elt1, Elt2);
		for (j = 0; j < i; j++) {
			A->Group_element->mult(Elt2, gens->ith(j), Elt3);
			A->Group_element->mult(Elt3, Elt1, Elt4);

			a = S->element_rank_lint(Elt4);
			jj = inverse_word_list[a];
			for (k = 0; k < l; k++) {
				b = jj % primes[k];
				word[k] = b;
				jj = jj - b;
				jj = jj / primes[k];
			}

			if (f_v) {
				cout << "g_" << j << "^{g_" << i << "} =" << endl;
				cout << "which is element " << a << " which is word "
					<< inverse_word_list[a] << " = ";
				Int_vec_print(cout, word, l);
				cout << endl;
				A->Group_element->print(cout, Elt4);
				cout << endl;
			}
		}
		if (f_v) {
			cout << endl;
		}
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);

	FREE_int(word_list);
	FREE_int(inverse_word_list);
	if (f_v) {
		cout << "algebra_global_with_action::presentation done" << endl;
	}
}
#endif



void algebra_global_with_action::do_eigenstuff_with_coefficients(
		algebra::field_theory::finite_field *F, int n, std::string &coeffs_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients" << endl;
	}
	int *Data;
	int len;

	Int_vec_scan(coeffs_text, Data, len);
	if (len != n * n) {
		cout << "len != n * n " << len << endl;
		exit(1);
	}

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients "
				"before do_eigenstuff" << endl;
	}
	do_eigenstuff(F, n, Data, verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients "
				"after do_eigenstuff" << endl;
	}

	FREE_int(Data);
	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients done" << endl;
	}
}

void algebra_global_with_action::do_eigenstuff_from_file(
		algebra::field_theory::finite_field *F,
		int n, std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_from_file" << endl;
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
		cout << "algebra_global_with_action::do_eigenstuff_from_file before do_eigenstuff" << endl;
	}
	do_eigenstuff(F, n, Data, verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_from_file after do_eigenstuff" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_from_file done" << endl;
	}
}



void algebra_global_with_action::do_eigenstuff(
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
		cout << "algebra_global_with_action::do_eigenstuff done" << endl;
	}
}






void algebra_global_with_action::young_symmetrizer(
		int n, int verbose_level)
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
	combinatorics::other_combinatorics::combinatorics_domain Combi;


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


	while (true) {
		int nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = n - 1; i >= 0; i--) {
			for (j = 0; j < part[i]; j++) {
				parts[nb_parts++] = i + 1;
			}
		}

		if (f_v) {
			cout << "partition ";
			Int_vec_print(cout, parts, nb_parts);
			cout << endl;
		}


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		int *tableau;

		tableau = NEW_int(n);
		for (i = 0; i < n; i++) {
			tableau[i] = i;
		}
		Y->young_symmetrizer(parts, nb_parts, tableau, elt1, elt2, h_alpha, verbose_level);
		FREE_int(tableau);


		if (f_v) {
			cout << "h_alpha =" << endl;
			Y->group_ring_element_print(Y->A, Y->S, h_alpha);
			cout << endl;
		}


		Y->group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		Y->group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		if (f_v) {
			cout << "h_alpha * h_alpha=" << endl;
			Y->group_ring_element_print(Y->A, Y->S, elt5);
			cout << endl;
		}

		int *Module_Base;
		int *base_cols;
		int rk;


		Y->create_module(h_alpha,
			Module_Base, base_cols, rk,
			verbose_level);

		if (f_v) {
			cout << "Module_Basis=" << endl;
			Y->D->print_matrix(Module_Base, rk, Y->goi);
		}


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(Y->D->offset(Module_Base, i * Y->goi + j),
						Y->D->offset(Base, s * Y->goi + j), 0);
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

	if (f_v) {
		cout << "Basis of submodule=" << endl;
		Y->D->print_matrix(Base, s, Y->goi);
	}


	FREE_int(part);
	FREE_int(parts);
	FREE_int(Fst);
	FREE_int(Len);
	if (f_v) {
		cout << "before freeing Base" << endl;
	}
	FREE_int(Base);
	FREE_int(Base_inv);
	if (f_v) {
		cout << "before freeing Y" << endl;
	}
	FREE_OBJECT(Y);
	if (f_v) {
		cout << "before freeing elt1" << endl;
	}
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

void algebra_global_with_action::young_symmetrizer_sym_4(
		int verbose_level)
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

	for (cnt = 0; cnt < 10; cnt++) {
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

		if (f_v) {
			cout << "partition ";
			Int_vec_print(cout, parts, nb_parts);
			cout << endl;
		}


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		Y->young_symmetrizer(parts, nb_parts,
				Tableau[cnt], elt1, elt2, h_alpha,
				verbose_level);


		if (f_v) {
			cout << "h_alpha =" << endl;
			Y->group_ring_element_print(Y->A, Y->S, h_alpha);
			cout << endl;
		}


		Y->group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		Y->group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		if (f_v) {
			cout << "h_alpha * h_alpha=" << endl;
			Y->group_ring_element_print(Y->A, Y->S, elt5);
			cout << endl;
		}

		int *Module_Base;
		int *base_cols;
		int rk;


		Y->create_module(h_alpha,
			Module_Base, base_cols, rk,
			verbose_level);

		if (f_v) {
			cout << "Module_Basis=" << endl;
			Y->D->print_matrix(Module_Base, rk, Y->goi);
		}


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(
						Y->D->offset(Module_Base, i * Y->goi + j),
						Y->D->offset(Base, s * Y->goi + j), 0);
			}
			s++;
		}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_int(Module_Base);
		FREE_int(base_cols);


	}

	if (f_v) {
		cout << "Basis of submodule=" << endl;
		//Y->D->print_matrix(Base, s, Y->goi);
		Y->D->print_matrix_for_maple(Base, s, Y->goi);
	}

	FREE_int(part);
	FREE_int(parts);
	FREE_int(Fst);
	FREE_int(Len);
	if (f_v) {
		cout << "before freeing Base" << endl;
	}
	FREE_int(Base);
	FREE_int(Base_inv);
	if (f_v) {
		cout << "before freeing Y" << endl;
	}
	FREE_OBJECT(Y);
	if (f_v) {
		cout << "before freeing elt1" << endl;
	}
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




void algebra_global_with_action::linear_codes_with_bounded_minimum_distance(
		poset_classification::poset_classification_control *Control,
		group_constructions::linear_group *LG,
		int d, int target_depth,
		poset_classification::poset_classification *&PC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance" << endl;
	}

	poset_classification::poset_with_group_action *Poset;


	Control->f_depth = true;
	Control->depth = target_depth;


	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance "
				"group set up, calling gen->init" << endl;
		cout << "LG->A2->A->f_has_strong_generators="
				<< LG->A2->f_has_strong_generators << endl;
	}

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);

	Poset->init_subset_lattice(
			LG->A_linear, LG->A_linear,
			LG->Strong_gens,
			verbose_level);


	int independence_value = d - 1;

	Poset->add_independence_condition(
			independence_value,
			verbose_level);

#if 0
	Poset->f_print_function = false;
	Poset->print_function = print_code;
	Poset->print_function_data = this;
#endif

	PC = NEW_OBJECT(poset_classification::poset_classification);
	PC->initialize_and_allocate_root_node(Control, Poset,
			target_depth, verbose_level);

	int t0;
	other::orbiter_kernel_system::os_interface Os;
	int depth;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance "
				"before PC->main" << endl;
	}

	depth = PC->main(t0,
			target_depth /*schreier_depth*/,
		true /*f_use_invariant_subset_if_available*/,
		false /*f_debug */,
		verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance "
				"after PC->main" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance "
				"depth = " << depth << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance "
				"done" << endl;
	}
}



void algebra_global_with_action::permutation_representation_of_element(
		actions::action *A,
		std::string &element_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "algebra_global_with_action::permutation_representation_of_element "
				"element_description=" << element_description << endl;
	}

	prefix = A->label + "_elt";

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	Int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}

	A->Group_element->make_element(Elt, data, 0 /* verbose_level */);




	{
		string fname, title, author, extra_praeamble;

		fname = prefix + "_permutation.tex";
		title = "Permutation representation of element";


		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::permutation_representation_of_element "
						"before report" << endl;
			}

			ost << "$$" << endl;
			A->Group_element->element_print_latex(Elt, ost);
			ost << "$$" << endl;

			ost << "$$" << endl;
			A->Group_element->element_print_as_permutation(Elt, ost);
			ost << "$$" << endl;

			if (f_v) {
				cout << "algebra_global_with_action::permutation_representation_of_element "
						"after report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(data);




	if (f_v) {
		cout << "algebra_global_with_action::permutation_representation_of_element done" << endl;
	}
}






void algebra_global_with_action::representation_on_polynomials(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
// creates an action object for the induced action on polynomials
// assumes a linear group
{
	int f_v = (verbose_level >= 1);
	//int f_stabilizer = true;
	//int f_draw_tree = true;


	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials" << endl;
	}


	//field_theory::finite_field *F;
	actions::action *A;
	//matrix_group *M;
	int n;
	//int degree;
	algebra::ring_theory::longinteger_object go;

	A = LG->A_linear;
	//F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();
	int degree_of_poly;

	degree_of_poly = HPD->degree;

	if (f_v) {
		cout << "n = " << n << endl;
		cout << "degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}

#if 0
	//ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);


	monomial_ordering_type Monomial_ordering_type = t_PART;


	HPD->init(F, n /* nb_var */, degree_of_poly,
			Monomial_ordering_type,
			verbose_level);
#endif

	actions::action *A2;

	//A2 = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials "
				"before A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}
	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials "
				"after A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	induced_actions::action_on_homogeneous_polynomials *A_on_HPD;
	int *M;
	int nb_gens;
	int i;

	A_on_HPD = A2->G.OnHP;

	if (LG->f_has_nice_gens) {
		if (f_v) {
			cout << "algebra_global_with_action::representation_on_polynomials "
					"using nice generators" << endl;
		}
		LG->nice_gens->matrix_representation(
				A_on_HPD, M, nb_gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "algebra_global_with_action::representation_on_polynomials "
					"using strong generators" << endl;
		}
		LG->Strong_gens->gens->matrix_representation(
				A_on_HPD, M, nb_gens, verbose_level);
	}

	for (i = 0; i < nb_gens; i++) {
		cout << "matrix " << i << " / " << nb_gens << ":" << endl;
		Int_matrix_print(
				M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);
	}

	for (i = 0; i < nb_gens; i++) {
		string fname;
		other::orbiter_kernel_system::file_io Fio;

		fname = LG->label + "_rep_" + std::to_string(degree_of_poly) + "_" + std::to_string(i) + ".csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		fname = LG->label + "_rep_" + std::to_string(degree_of_poly) + "_" + std::to_string(i) + ".gap";

		Fio.int_matrix_write_cas_friendly(
				fname, M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



	}
	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials done" << endl;
	}
}








void algebra_global_with_action::search_element_of_order(
		groups::any_group *Any_group,
		actions::action *A1, actions::action *A2,
		int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::search_element_of_order" << endl;
	}
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = SG->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	algebra::ring_theory::longinteger_object go;
	int i, cnt, ord;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->Group_element->element_order(Elt);

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
		if (f_v) {
			cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
			A2->Group_element->element_print(Elt, cout);
			cout << endl;
			A2->Group_element->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
		cnt++;
	}
	if (f_v) {
		cout << "we found " << cnt << " group elements of order " << order << endl;
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::search_element_of_order done" << endl;
	}
}

void algebra_global_with_action::find_standard_generators(
		groups::any_group *Any_group,
		actions::action *A1, actions::action *A2,
		int order_a, int order_b, int order_ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::find_standard_generators" << endl;
	}
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	H = SG->create_sims(verbose_level);

	cout << "algebra_global_with_action::find_standard_generators "
			"group order H = " << H->group_order_lint() << endl;

	int *Elt_a;
	int *Elt_b;
	int *Elt_ab;
	algebra::ring_theory::longinteger_object go;
	int i, j, cnt, ord;

	Elt_a = NEW_int(A1->elt_size_in_int);
	Elt_b = NEW_int(A1->elt_size_in_int);
	Elt_ab = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt_a);


		ord = A2->Group_element->element_order(Elt_a);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order_a) {
			continue;
		}

		for (j = 0; j < go.as_int(); j++) {
			H->element_unrank_lint(j, Elt_b);


			ord = A2->Group_element->element_order(Elt_b);

			if (ord != order_b) {
				continue;
			}

			A2->Group_element->element_mult(Elt_a, Elt_b, Elt_ab, 0);

			ord = A2->Group_element->element_order(Elt_ab);

			if (ord != order_ab) {
				continue;
			}

			if (f_v) {
				cout << "algebra_global_with_action::find_standard_generators "
						"a = " << setw(5) << i << ", b=" << setw(5) << j
						<< " : " << cnt << ":" << endl;
				cout << "a=" << endl;
				A2->Group_element->element_print(Elt_a, cout);
				cout << endl;
				A2->Group_element->element_print_as_permutation(Elt_a, cout);
				cout << endl;
				cout << "b=" << endl;
				A2->Group_element->element_print(Elt_b, cout);
				cout << endl;
				A2->Group_element->element_print_as_permutation(Elt_b, cout);
				cout << endl;
				cout << "ab=" << endl;
				A2->Group_element->element_print(Elt_ab, cout);
				cout << endl;
				A2->Group_element->element_print_as_permutation(Elt_ab, cout);
				cout << endl;
			}
			cnt++;
		}
	}
	if (f_v) {
		cout << "algebra_global_with_action::find_standard_generators "
				"we found " << cnt << " group elements with "
				"ord_a = " << order_a << " ord_b  = " << order_b
				<< " and ord_ab = " << order_ab << endl;
	}

	FREE_int(Elt_a);
	FREE_int(Elt_b);
	FREE_int(Elt_ab);
	if (f_v) {
		cout << "algebra_global_with_action::find_standard_generators done" << endl;
	}
}

void algebra_global_with_action::do_character_table_symmetric_group(
		int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group" << endl;
		cout << "deg=" << deg << endl;
	}

	apps_algebra::character_table_burnside *CTB;

	CTB = NEW_OBJECT(apps_algebra::character_table_burnside);

	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group "
				"before CTB->do_it" << endl;
	}
	CTB->do_it(deg, verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group "
				"after CTB->do_it" << endl;
	}

	FREE_OBJECT(CTB);

	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group done" << endl;
	}
}

void algebra_global_with_action::group_of_automorphisms_by_images_of_generators(
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int m, int n,
		groups::any_group *AG,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators" << endl;
	}

	int *Perms;
	long int go;



	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before automorphism_by_generator_images" << endl;
	}

	automorphism_by_generator_images(
			label,
			AG->A,
			AG->Subgroup_gens,
			AG->Subgroup_sims,
			Elements_ge,
			Images, m, n,
			Perms, go,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after automorphism_by_generator_images" << endl;
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"we found " << m << " permutations of degree " << go << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before AG->automorphism_by_generator_images_save" << endl;
	}

	AG->automorphism_by_generator_images_save(
			Images, m, n,
			Perms, go,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after AG->automorphism_by_generator_images_save" << endl;
	}



	actions::action *A_perm;

	A_perm = NEW_OBJECT(actions::action);


	algebra::ring_theory::longinteger_object target_go;
	int f_target_go = true;

	target_go.create(m);


	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before A_perm->Known_groups->init_permutation_group" << endl;
	}
	A_perm->Known_groups->init_permutation_group(
			go /* degree */, false /* f_no_base */,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after A_perm->Known_groups->init_permutation_group" << endl;
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A_perm, Perms,
			m /*nb_elements*/, A_perm->make_element_size,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after gens->init_from_data" << endl;
	}



#if 0
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before A_perm->Known_groups->init_permutation_group_from_generators" << endl;
	}
	A_perm->Known_groups->init_permutation_group_from_generators(
			go /* degree */,
		true /* f_target_go */, target_go,
		m /* nb_gens */, Perms,
		0 /* given_base_length */, NULL /* long int *given_base */,
		false /* f_given_base */,
		verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after A_perm->Known_groups->init_permutation_group_from_generators" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"create_automorphism_group_of_incidence_structure: created action ";
		A_perm->print_info();
		cout << endl;
	}
#endif

	groups::sims *Sims;

	{
		groups::schreier_sims *ss;

		ss = NEW_OBJECT(groups::schreier_sims);

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"before ss->init" << endl;
		}
		ss->init(A_perm, verbose_level - 1);
		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"after ss->init" << endl;
		}

		//ss->interested_in_kernel(A_subaction, verbose_level - 1);

		if (f_target_go) {
			if (f_v) {
				cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
						"before ss->init_target_group_order" << endl;
			}
			ss->init_target_group_order(target_go, verbose_level - 1);
			if (f_v) {
				cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
						"after ss->init_target_group_order" << endl;
			}
		}

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"before ss->init_generators" << endl;
		}
		ss->init_generators(gens, verbose_level - 2);
		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"after ss->init_generators" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"before ss->create_group" << endl;
		}
		ss->create_group(verbose_level - 2);
		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"after ss->create_group" << endl;
		}
		Sims = ss->G;
		ss->G = NULL;
		//*this = *ss->G;

		//ss->G->null();

		//cout << "create_sims_from_generators_randomized
		// before FREE_OBJECT ss" << endl;
		FREE_OBJECT(ss);
	}


	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"transversal lengths:" << endl;
		Sims->print_transversal_lengths();
	}

	groups::strong_generators *Strong_gens;

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(Sims, verbose_level - 1);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"Strong_gens:" << endl;
		Strong_gens->print_generators_for_make_element(cout);

		std::string fname;


		fname = label + ".gap";

		{
			ofstream ost(fname);

			Strong_gens->print_generators_gap(
					ost, verbose_level);
		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}
	}

}

void algebra_global_with_action::automorphism_by_generator_images(
		std::string &label,
		actions::action *A,
		groups::strong_generators *Subgroup_gens,
		groups::sims *Subgroup_sims,
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int m, int n,
		int *&Perms, long int &go,
		int verbose_level)
// An automorphism of a group is determined by the images of the generators.
// Here, we assume that we have a certain set of standard generators, and that
// the images of these generators are known.
// Using the right regular representation and a Schreier tree,
// we can then compute the automorphisms associated to the Images.
// Any automorphism is computed as a permutation of the elements
// in the ordering defined by the sims object Subgroup_sims
// The images in Images[] and the generators
// in Subgroup_gens->gens must correspond.
// This means that n must equal Subgroup_gens->gens->len
//
// We use orbits_schreier::orbit_of_sets for the Schreier tree.
// We need Subgroup_sims to set up action by right multiplication
// output: Perms[m * go]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images" << endl;
	}

	go = Subgroup_sims->group_order_lint();
	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images go = " << go << endl;
	}


	actions::action *A_rm;
	// action by right multiplication

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before A->Induced_action->induced_action_by_right_multiplication" << endl;
	}
	A_rm = A->Induced_action->induced_action_by_right_multiplication(
			false /* f_basis */, NULL /* old_G */,
			Subgroup_sims /*Base_group*/, false /* f_ownership */,
			verbose_level - 2);

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"after A->Induced_action->induced_action_by_right_multiplication" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"generators:" << endl;
		Subgroup_gens->gens->print_quick(cout);
		cout << endl;
	}

	if (Subgroup_gens->gens->len != n) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"Subgroup_gens->gens->len != n" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before Orb.init" << endl;
	}


	orbits_schreier::orbit_of_sets Orb;
	long int set[1];
	//orbiter_kernel_system::file_io Fio;

	set[0] = 0;

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before Orb.init" << endl;
	}
	Orb.init(A, A_rm,
			set, 1 /* sz */,
			Subgroup_gens->gens,
			verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"Found an orbit of size " << Orb.used_length << endl;
	}
	if (Orb.used_length != go) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"orbit length != go" << endl;
		exit(1);
	}


	string fname;

	fname = label + "_tree.layered_graph";

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before Orb.export_tree_as_layered_graph_to_file" << endl;
	}

	Orb.export_tree_as_layered_graph_to_file(
			fname,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"after Orb.export_tree_as_layered_graph_to_file" << endl;
	}



	int h;
	int *Elt;
	int *perm;

	Elt = NEW_int(A->elt_size_in_int);
	perm = NEW_int(go);


	Perms = NEW_int(m * go);


	for (h = 0; h < m; h++) {

		if (f_vv) {
			cout << "algebra_global_with_action::automorphism_by_generator_images "
					"h=" << h << " : ";
			Int_vec_print(cout, Images + h * n, n);
			cout << endl;
		}

		int verbose_level1;

		if (h == 0) {
			verbose_level1 = verbose_level;
		}
		else {
			verbose_level1 = 0;
		}
		create_permutation(
				A,
				Subgroup_gens,
				Subgroup_sims,
				&Orb,
				Elements_ge,
				Images + h * n, n, h,
				Elt,
				perm, go,
				verbose_level1);

		Int_vec_copy(perm, Perms + h * go, go);

	}


	FREE_int(Elt);
	FREE_int(perm);

	FREE_OBJECT(A_rm);



	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images done" << endl;
	}
}


void algebra_global_with_action::create_permutation(
		actions::action *A,
		groups::strong_generators *Subgroup_gens,
		groups::sims *Subgroup_sims,
		orbits_schreier::orbit_of_sets *Orb,
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int n, int h,
		int *Elt,
		int *perm, long int go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "algebra_global_with_action::create_permutation" << endl;
	}

	if (f_vv) {
		cout << "algebra_global_with_action::create_permutation h=" << h << " : ";
		Int_vec_print(cout, Images, n);
		cout << endl;
	}

	int in, out;
	int i, a, b, c;
	long int new_set[1];
	int pos;

	int nb_gens;

	nb_gens = Subgroup_gens->gens->len;

	if (f_vv) {
		for (i = 0; i < n; i++) {
			cout << "generator image " << i << " : " << Images[i] << " is: ";
			A->Group_element->print(cout, Elements_ge->ith(Images[i]));
			cout << endl;
		}
	}

	for (in = 0; in < go; in++) {

		uint32_t hash;
		std::vector<int> path;

		new_set[0] = in;

		if (!Orb->find_set(
				new_set, pos, hash)) {
			cout << "algebra_global_with_action::create_permutation !find_set" << endl;
			exit(1);
		}
		Orb->get_path(
				path,
				pos);

		if (f_vvv) {
			cout << "algebra_global_with_action::create_permutation "
					"in=" << in << " pos=" << pos << " path=";
			Int_vec_stl_print(cout, path);
			cout << " of length " << path.size();
			cout << endl;
		}


		int *path1;
		int *word;

		path1 = NEW_int(path.size());
		word = NEW_int(path.size());

		for (i = 0; i < path.size(); i++) {
			a = path[i];
			b = nb_gens - 1 - a; // reverse ordering because the Coxeter generators are listed in reverse
			path1[i] = b;
			c = Images[b];

			word[i] = c;
		}

		if (f_vvv) {
			cout << "algebra_global_with_action::create_permutation in=" << in << " path=";
			Int_vec_stl_print(cout, path);
			cout << " -> ";
			Int_vec_print(cout, path1, path.size());
			cout << " -> ";
			Int_vec_print(cout, word, path.size());
			cout << endl;
		}


		A->Group_element->evaluate_word(
				Elt, word, path.size(),
				Elements_ge,
				verbose_level - 3);

		if (f_vvv) {
			cout << "The word evaluates to" << endl;
			A->Group_element->element_print_quick(Elt, cout);
			cout << endl;
			cout << "in latex:" << endl;
			A->Group_element->element_print_latex(Elt, cout);
			cout << endl;
		}

		algebra::ring_theory::longinteger_object rk_out;

		Subgroup_sims->element_rank(rk_out, Elt);

		out = rk_out.as_int();

		perm[in] = out;

		if (f_vvv) {
			cout << "algebra_global_with_action::create_permutation "
					"in=" << in << " -> " << out << endl;
		}


		FREE_int(path1);
		FREE_int(word);

	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;


	int f_is_perm;

	f_is_perm = Combi.Permutations->is_permutation(
			perm, go);
	if (f_is_perm) {
		if (f_vv) {
			cout << "algebra_global_with_action::create_permutation "
					"h = " << h << " output is a permutation" << endl;
		}
	}
	else {
		cout << "algebra_global_with_action::create_permutation "
				"h = " << h << " output is not a permutation" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "algebra_global_with_action::create_permutation "
				"h = " << h << ", perm = ";
		Combi.Permutations->perm_print_list(
					cout, perm, go);
		cout << endl;
		cout << "in cycle notation: ";
		Combi.Permutations->perm_print(
				cout, perm, go);
		cout << endl;

		int *cycles;
		int nb_cycles;

		cycles = NEW_int(go);

		Combi.Permutations->perm_cycle_type(
				perm, go, cycles, nb_cycles);
		cout << "The cycle type is: ";

		other::data_structures::tally T;

		T.init(cycles,
				nb_cycles, false /* f_second */, 0 /* verbose_level*/);

		T.print(true /* f_backwards*/);

		FREE_int(cycles);
		cout << endl;

	}
	if (f_v) {
		cout << "algebra_global_with_action::create_permutation done" << endl;
	}

}




void algebra_global_with_action::do_orbits_on_subspaces(
		groups::any_group *Any_group,
		poset_classification::poset_classification_control *Control,
		orbits::orbits_on_subspaces *&OoS,
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_orbits_on_subspaces" << endl;
	}




	if (!Any_group->A->f_is_linear) {
		cout << "algebra_global_with_action::do_orbits_on_subspaces !A->f_is_linear" << endl;
		cout << "group:" << endl;
		Any_group->print();
		exit(1);
	}

	if (f_v) {
		cout << "group:" << endl;
		Any_group->print();
	}


	//orbits_on_subspaces *OoS;

	OoS = NEW_OBJECT(orbits::orbits_on_subspaces);

	OoS->init(Any_group, Control, depth, verbose_level);


	//finite_field *F;

	//F = LG->F;


	//FREE_OBJECT(OoS);


	if (f_v) {
		cout << "algebra_global_with_action::do_orbits_on_subspaces done" << endl;
	}
}

void algebra_global_with_action::do_tensor_classify(
		groups::any_group *Any_group,
		std::string &control_label,
		apps_geometry::tensor_classify *&T,
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_classify" << endl;
	}

	if (!Any_group->f_linear_group) {
		cout << "algebra_global_with_action::do_tensor_classify !Any_group->f_linear_group" << endl;
		exit(1);
	}

	algebra::field_theory::finite_field *F;

	F = Any_group->LG->F;

	poset_classification::poset_classification_control *Control =
			Get_poset_classification_control(control_label);



	//apps_geometry::tensor_classify *T;

	T = NEW_OBJECT(apps_geometry::tensor_classify);

	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_classify before T->init" << endl;
	}
	T->init(F, Any_group->LG, verbose_level - 1);
	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_classify after T->init" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_classify before classify_poset" << endl;
	}
	T->classify_poset(depth,
			Control,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_classify after classify_poset" << endl;
	}



	//FREE_OBJECT(T);

	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_classify done" << endl;
	}
}


void algebra_global_with_action::do_tensor_permutations(
		groups::any_group *Any_group,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_permutations" << endl;
	}

	if (!Any_group->f_linear_group) {
		cout << "algebra_global_with_action::do_tensor_permutations "
				"!Any_group->f_linear_group" << endl;
		exit(1);
	}
	algebra::field_theory::finite_field *F;

	F = Any_group->LG->F;


	apps_geometry::tensor_classify *T;

	T = NEW_OBJECT(apps_geometry::tensor_classify);

	T->init(F, Any_group->LG, verbose_level - 1);


	FREE_OBJECT(T);

	if (f_v) {
		cout << "algebra_global_with_action::do_tensor_permutations done" << endl;
	}
}


void algebra_global_with_action::do_linear_codes(
		groups::any_group *Any_group,
		std::string &control_label,
		int minimum_distance,
		int target_size,
		poset_classification::poset_classification *&PC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_linear_codes" << endl;
	}

	if (!Any_group->f_linear_group) {
		cout << "algebra_global_with_action::do_linear_codes !f_linear_group" << endl;
		exit(1);
	}

	poset_classification::poset_classification_control *Control =
			Get_poset_classification_control(control_label);

	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "algebra_global_with_action::do_linear_codes before "
				"Algebra.linear_codes_with_bounded_minimum_distance" << endl;
	}

	Algebra.linear_codes_with_bounded_minimum_distance(
			Control, Any_group->LG,
			minimum_distance, target_size,
			PC,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::do_linear_codes after "
				"Algebra.linear_codes_with_bounded_minimum_distance" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::do_linear_codes done" << endl;
	}
}

void algebra_global_with_action::do_classify_ovoids(
		groups::any_group *Any_group,
		apps_geometry::ovoid_classify_description
			*Ovoid_classify_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_classify_ovoids" << endl;
	}

	if (!Any_group->f_linear_group) {
		cout << "algebra_global_with_action::do_classify_ovoids !Any_group->f_linear_group" << endl;
		exit(1);
	}

	apps_geometry::ovoid_classify *Ovoid_classify;


	Ovoid_classify = NEW_OBJECT(apps_geometry::ovoid_classify);


	Ovoid_classify->init(Ovoid_classify_description,
			Any_group->LG,
			verbose_level);

	FREE_OBJECT(Ovoid_classify);

	if (f_v) {
		cout << "algebra_global_with_action::do_classify_ovoids done" << endl;
	}
}


void algebra_global_with_action::print_action_on_surface(
		groups::any_group *Any_group,
		std::string &surface_label,
		std::string &label_of_elements,
		data_structures_groups::vector_ge *Elements,
		//int *element_data, int nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface" << endl;
	}

	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

	SC = Get_object_of_cubic_surface(surface_label);

	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface "
				"before SC->SOG->print_action_on_surface" << endl;
	}
	SC->SOG->print_action_on_surface(
			label_of_elements,
			Elements,
			//element_data, nb_elements,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface "
				"after SC->SOG->print_action_on_surface" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface done" << endl;
	}

}

void algebra_global_with_action::element_processing(
		groups::any_group *Any_group,
		element_processing_description *element_processing_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::element_processing" << endl;
	}

#if 0
	int *element_data = NULL;
	int nb_elements;
	int n;
#endif

	apps_algebra::vector_ge_builder *Elements_builder;
	data_structures_groups::vector_ge *Elements;


	if (!element_processing_descr->f_input) {

		cout << "please use -input <label> to define input elements" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "algebra_global_with_action::element_processing getting input" << endl;
	}

	Elements_builder = Get_object_of_type_vector_ge(element_processing_descr->input_label);

	Elements = Elements_builder->V;

#if 0
	Get_matrix(element_processing_descr->input_label,
			element_data, nb_elements, n);
#endif



	if (element_processing_descr->f_print) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing f_print" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->print_given_elements_tex" << endl;
		}

		Any_group->print_given_elements_tex(
				element_processing_descr->input_label,
				Elements,
				//element_data, nb_elements,
				element_processing_descr->f_with_permutation,
				element_processing_descr->f_with_fix_structure,
				verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->print_given_elements_tex" << endl;
		}

	}
	else if (element_processing_descr->f_apply_isomorphism_wedge_product_4to6) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_apply_isomorphism_wedge_product_4to6" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->apply_isomorphism_wedge_product_4to6" << endl;
		}
		Any_group->apply_isomorphism_wedge_product_4to6(
				element_processing_descr->input_label,
				Elements,
				//element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->apply_isomorphism_wedge_product_4to6" << endl;
		}


	}
	else if (element_processing_descr->f_order_of_products_of_pairs) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_order_of_products_of_pairs" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->order_of_products_of_pairs" << endl;
		}
		Any_group->order_of_products_of_pairs(
				element_processing_descr->input_label,
				Elements,
				//element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->order_of_products_of_pairs" << endl;
		}


	}
	else if (element_processing_descr->f_conjugate) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_conjugate" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->conjugate" << endl;
		}
		Any_group->conjugate(
				element_processing_descr->input_label,
				element_processing_descr->conjugate_data,
				Elements,
				//element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->conjugate" << endl;
		}


	}

	else if (element_processing_descr->f_print_action_on_surface) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_print_action_on_surface" << endl;
		}


		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before print_action_on_surface" << endl;
		}
		print_action_on_surface(
				Any_group,
				element_processing_descr->print_action_on_surface_label,
				element_processing_descr->input_label,
				Elements,
				//element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after print_action_on_surface" << endl;
		}


	}



	if (f_v) {
		cout << "algebra_global_with_action::element_processing done" << endl;
	}
}


void algebra_global_with_action::subgroup_lattice_identify_subgroup(
		groups::any_group *Any_group,
		std::string &group_label,
		int &go, int &layer_idx,
		int &orb_idx, int &group_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup" << endl;
	}

	if (!Any_group->f_has_subgroup_lattice) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup "
				"subgroup lattice is not available" << endl;
		exit(1);
	}

	groups::any_group *Subgroup;

	groups::strong_generators *Subgroup_strong_gens;

	Subgroup = Get_any_group(group_label);

	Subgroup_strong_gens = Subgroup->Subgroup_gens;

	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup "
				"before Any_group->Subgroup_lattice->identify_subgroup" << endl;
	}
	Any_group->Subgroup_lattice->identify_subgroup(
			Subgroup_strong_gens,
			go, layer_idx, orb_idx, group_idx,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup "
				"after Any_group->Subgroup_lattice->identify_subgroup" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroup "
				"found subgroup of order " << go << " in layer " << layer_idx
				<< " in orbit " << orb_idx << " at position " << group_idx << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup done" << endl;
	}
}


void algebra_global_with_action::modified_group_init(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		int verbose_level)
// Some group modifications need to be performed at level 5.
// This is because they rely on stuff that is in the application layer.
// Examples: Orbit computations, variety stabilizer
// So, this function serves as a front end for the actual function in
// group_constructions::modified_group_create
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_init" << endl;
	}
	Modified_group_create->Descr = Descr;

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_init "
				"initializing group" << endl;
	}


	if (Descr->f_point_stabilizer) {

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"before create_point_stabilizer_subgroup" << endl;
		}

		create_point_stabilizer_subgroup(Modified_group_create, Descr, verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"after create_point_stabilizer_subgroup" << endl;
		}
	}

	else if (Descr->f_set_stabilizer) {

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"before create_set_stabilizer_subgroup" << endl;
		}

		create_set_stabilizer_subgroup(Modified_group_create, Descr, verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"after create_set_stabilizer_subgroup" << endl;
		}
	}
	else if (Descr->f_stabilizer_of_variety) {

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"f_stabilizer_of_variety" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"before modified_group_create_stabilizer_of_variety" << endl;
		}
		modified_group_create_stabilizer_of_variety(
				Modified_group_create,
				Descr,
				Descr->stabilizer_of_variety_label,
				verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"after modified_group_create_stabilizer_of_variety" << endl;
		}

		// output in A_modified

	}




	else {

		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"before Modified_group_create->modified_group_init" << endl;
		}
		Modified_group_create->modified_group_init(Descr, verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"after Modified_group_create->modified_group_init" << endl;
		}
		//cout << "algebra_global_with_action::modified_group_init "
		//		"unknown operation" << endl;
		//exit(1);

	}

	if (f_v) {

		algebra::ring_theory::longinteger_object go;

		Modified_group_create->Strong_gens->group_order(go);

		cout << "algebra_global_with_action::modified_group_init "
				"created a group of order " << go
				<< " and degree " << Modified_group_create->A_modified->degree << endl;

	}



	if (f_v) {
		cout << "algebra_global_with_action::modified_group_init done" << endl;
	}
}




void algebra_global_with_action::create_point_stabilizer_subgroup(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	groups::any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	Modified_group_create->A_base = AG->A_base;
	Modified_group_create->A_previous = AG->A;

	Modified_group_create->label = AG->label;
	Modified_group_create->label_tex = AG->label_tex;

	if (f_v) {
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->A_modified = Modified_group_create->A_previous; // ToDo!



	Modified_group_create->f_has_strong_generators = true;
	if (f_v) {
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}

	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
					"before Orbits.orbits_on_points" << endl;
		}

		orbits::orbits_global Orbits;

		Orbits.orbits_on_points(AG, Orb, verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
					"after Orbits.orbits_on_points" << endl;
		}

		Orb->stabilizer_any_point(
				Descr->point_stabilizer_point,
				Modified_group_create->Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}



	if (f_v) {
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
				"action A_modified created: ";
		Modified_group_create->A_modified->print_info();
	}


	Modified_group_create->label += "_Stab" + std::to_string(Descr->point_stabilizer_point);
	Modified_group_create->label_tex += "{\\rm Stab " + std::to_string(Descr->point_stabilizer_point) + "}";



	if (f_v) {
		cout << "algebra_global_with_action::create_point_stabilizer_subgroup "
				"done" << endl;
	}
}


void algebra_global_with_action::create_set_stabilizer_subgroup(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::create_set_stabilizer_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	groups::any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	Modified_group_create->A_base = AG->A_base;
	Modified_group_create->A_previous = AG->A;

	Modified_group_create->label = AG->label;
	Modified_group_create->label_tex = AG->label_tex;

	if (f_v) {
		cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->A_modified = Modified_group_create->A_previous; // ToDo !!!




#if 0
	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"before Orbits.orbits_on_points" << endl;
		}

		orbits::orbits_global Orbits;

		Orbits.orbits_on_points(AG, Orb, verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"after Orbits.orbits_on_points" << endl;
		}

		Orb->stabilizer_any_point(
				Descr->point_stabilizer_point,
				Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}
#endif
	{

		orbits::orbits_global Orbits_global;
		poset_classification::poset_classification_control *Control;
		long int *the_set;
		long int *canonical_set;
		int *Elt1;
		int the_set_sz;
		int local_idx;

		Lint_vec_scan(Descr->set_stabilizer_the_set, the_set, the_set_sz);


		canonical_set = NEW_lint(the_set_sz);
		Elt1 = NEW_int(Modified_group_create->A_base->elt_size_in_int);

		Control = Get_poset_classification_control(Descr->set_stabilizer_control);

		poset_classification::poset_classification *PC;

		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"before Orbits_global.orbits_on_subsets" << endl;
		}
		Orbits_global.orbits_on_subsets(
				AG,
				Control,
				PC,
				the_set_sz,
				verbose_level - 2);
		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"after Orbits_global.orbits_on_subsets" << endl;
		}


		// trace the subset:

		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"before trace_set" << endl;
		}


		local_idx = PC->trace_set(
				the_set, the_set_sz, the_set_sz,
				canonical_set, Elt1,
			verbose_level - 2);


		// Elt1 maps the_set to canonical_set.


		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"after trace_set local_idx=" << local_idx << endl;
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"canonical_set=";
			Lint_vec_print(cout, canonical_set, the_set_sz);
			cout << endl;
		}

		groups::strong_generators *stab_gens_canonical_set;

		PC->get_stabilizer_generators_cleaned_up(
				stab_gens_canonical_set,
				the_set_sz, local_idx, verbose_level - 2);

		groups::group_theory_global Group_theory_global;

		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"before Group_theory_global.strong_generators_conjugate_aGav" << endl;
		}

		Group_theory_global.strong_generators_conjugate_aGav(
				stab_gens_canonical_set,
				Elt1,
				Modified_group_create->Strong_gens,
				verbose_level - 2);

		if (f_v) {
			cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
					"after Group_theory_global.strong_generators_conjugate_aGav" << endl;
		}



		FREE_OBJECT(stab_gens_canonical_set);
		FREE_OBJECT(PC);
		FREE_lint(canonical_set);
		FREE_int(Elt1);
	}

	Modified_group_create->f_has_strong_generators = true;

	if (f_v) {
		cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
				"strong generators created" << endl;
	}


	Modified_group_create->label += "_SetStab" + Descr->set_stabilizer_the_set;
	Modified_group_create->label_tex += "{\\rm SetStab " + Descr->set_stabilizer_the_set + "}";



	if (f_v) {
		cout << "algebra_global_with_action::create_set_stabilizer_subgroup "
				"done" << endl;
	}
}


void algebra_global_with_action::modified_group_create_stabilizer_of_variety(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		std::string &variety_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety" << endl;
	}

	canonical_form::variety_object_with_action *Input_Variety;

	Input_Variety = Get_variety(variety_label);

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"Input_Variety = " << Input_Variety->Variety_object->label_txt << endl;
	}

	std::string fname_base;

	fname_base = Input_Variety->Variety_object->label_txt + "_c";


	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);


	if (!Descr->f_nauty_control) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"Please use -nauty_control" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			1 /*nb_input_Vo*/,
			Input_Variety,
			fname_base,
			Descr->f_nauty_control,
			Descr->Nauty_interface_control,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"after Classifier->init_direct" << endl;
	}

	canonical_form::canonical_form_classifier_description *Descr1;

	Descr1 = NEW_OBJECT(canonical_form::canonical_form_classifier_description);



	//Descr1->f_save_nauty_input_graphs = true;

	Classifier->set_description(Descr1);



	canonical_form::canonical_form_global Canonical_form_global;
	canonical_form::classification_of_varieties_nauty *Classification_of_varieties_nauty;

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"before Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}
	Canonical_form_global.compute_group_and_tactical_decomposition(
			Classifier,
			Input_Variety,
			Classification_of_varieties_nauty,
			//fname_base,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"after Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"before FREE_OBJECT(Classification_of_varieties_nauty)" << endl;
	}

	//FREE_OBJECT(Classification_of_varieties_nauty);
	// Classification_of_varieties_nauty is freed in FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"after FREE_OBJECT(Classification_of_varieties_nauty)" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"before FREE_OBJECT(Descr1)" << endl;
	}
	FREE_OBJECT(Descr1);
	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"after FREE_OBJECT(Descr1)" << endl;
	}
	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"before FREE_OBJECT(Classifier)" << endl;
	}
	FREE_OBJECT(Classifier);
	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"after FREE_OBJECT(Classifier)" << endl;
	}


	//Input_Vo[0].Stab_gens;


	Modified_group_create->A_base = Input_Variety->PA->A;
	Modified_group_create->A_previous = Input_Variety->PA->A;





	Modified_group_create->label = Input_Variety->PA->A->label + "_stab_of_" + Input_Variety->Variety_object->label_txt;
	Modified_group_create->label_tex = Input_Variety->PA->A->label_tex + "{\\rm \\_stab\\_of\\_}" + Input_Variety->Variety_object->label_tex;
	if (f_v) {
		cout << "algebra_global_with_action::do_stabilizer_of_variety "
				"label = " << Modified_group_create->label << endl;
		cout << "algebra_global_with_action::do_stabilizer_of_variety "
				"label_tex = " << Modified_group_create->label_tex << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->f_has_strong_generators = true;

	groups::strong_generators *Strong_gens_temp;
	Strong_gens_temp = Input_Variety->Stab_gens->create_copy(verbose_level - 4);

	actions::action_global Action_global;

	Modified_group_create->A_modified = Action_global.init_subgroup_from_strong_generators(
			Modified_group_create->A_base,
			Strong_gens_temp,
			verbose_level - 1);

	Modified_group_create->A_modified->label = Modified_group_create->label;
	Modified_group_create->A_modified->label_tex = Modified_group_create->label_tex;

	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"A_modified->label = " << Modified_group_create->A_modified->label << endl;
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety "
				"A_modified->label_tex = " << Modified_group_create->A_modified->label_tex << endl;
	}


	// Strong_gens should be in the new action.

	Modified_group_create->f_has_strong_generators = true;
	Modified_group_create->Strong_gens = Modified_group_create->A_modified->Strong_gens->create_copy(verbose_level - 4);
	//Strong_gens = AG->class_data->Conjugacy_class[orbit_index]->gens->create_copy(verbose_level - 4);



	if (f_v) {
		cout << "algebra_global_with_action::modified_group_create_stabilizer_of_variety done" << endl;
	}
}


void algebra_global_with_action::conjugacy_class_of(
		groups::any_group *Any_group,
		std::string &label_of_class,
		std::string &elt_data,
		int verbose_level)
// uses orbits_schreier::orbit_of_sets
// needs Subgroup_sims to set up action by conjugation
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of" << endl;
	}

#if 0
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;
#endif

	int *Elt;

	Elt = NEW_int(Any_group->A->elt_size_in_int);

	algebra::ring_theory::longinteger_object a, b;

#if 1
	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of creating element " << elt_data << endl;
	}

	Any_group->A->Group_element->make_element_from_string(Elt, elt_data, 0);

	Any_group->Subgroup_sims->element_rank(a, Elt);

	a.assign_to(b);


#else


	a.create_from_base_10_string(rank_string.c_str(), 0 /*verbose_level*/);

	cout << "Creating element of rank " << a << endl;

	a.assign_to(b);

	H->element_unrank(a, Elt);

#endif

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of Element :" << endl;
		Any_group->A->Group_element->element_print(Elt, cout);
		cout << endl;
	}


	actions::action *A_conj;

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"before A->Induced_action->create_induced_action_by_conjugation" << endl;
	}
	A_conj = Any_group->A->Induced_action->create_induced_action_by_conjugation(
			Any_group->Subgroup_sims /*Base_group*/, false /* f_ownership */,
			false /* f_basis */, NULL /* old_G */,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"after A->Induced_action->create_induced_action_by_conjugation" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"created action A_conj of degree " << A_conj->degree << endl;
	}



	orbits_schreier::orbit_of_sets Orb;
	long int set[1];
	other::orbiter_kernel_system::file_io Fio;

	set[0] = b.as_lint();

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"before Orb.init" << endl;
	}
	Orb.init(Any_group->A, A_conj,
			set, 1 /* sz */,
			Any_group->Subgroup_gens->gens, //LG->Strong_gens->gens,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"Found an orbit of size " << Orb.used_length << endl;
	}

	std::vector<long int> Orbit;

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"before Orb.get_orbit_of_points" << endl;
	}
	Orb.get_orbit_of_points(Orbit, verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"Found an orbit of size " << Orbit.size() << endl;
	}

	int *data;
	int i;
	int nb_r, nb_c;
	std::string *Header_cols;
	std::string *Table;

	nb_r = Orbit.size();
	nb_c = 1;
	data = NEW_int(Any_group->A->make_element_size);
	Table = new std::string [nb_r * nb_c];
	Header_cols = new std::string [nb_c];

	Header_cols[0] = "elements";

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"collecting orbit elements" << endl;
	}

	for (i = 0; i < Orbit.size(); i++) {
		Any_group->Subgroup_sims->element_unrank_lint(Orbit[i], Elt);

		Any_group->A->Group_element->element_code_for_make_element(Elt, data);

		//Int_vec_copy(Elt, M + i * A->make_element_size, A->make_element_size);
		Table[i] = "\"" + Int_vec_stringify(data, Any_group->A->make_element_size) + "\"";
#if 0
		for (j = 0; j < A->make_element_size; j++) {
			M[i * A->make_element_size + j] = Elt[j];
		}
		//M[i] = Orbit[i];
#endif
	}

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"collecting orbit elements done" << endl;
	}

	string fname;

	fname = Any_group->label + "_class_of_" + label_of_class + ".csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_r, nb_c, Table,
			Header_cols,
			verbose_level);

	delete [] Header_cols;
	delete [] Table;

#if 0
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, Orbit.size(), A->make_element_size);
#endif

	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	//FREE_int(M);


	FREE_OBJECT(A_conj);
	//FREE_OBJECT(H);



	FREE_int(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::conjugacy_class_of done" << endl;
	}
}






}}}

