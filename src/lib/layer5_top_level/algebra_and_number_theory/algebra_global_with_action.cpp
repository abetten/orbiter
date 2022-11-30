/*
 * algebra_global_with_action.cpp
 *
 *  Created on: Dec 15, 2019
 *      Author: betten
 */



#include "orbiter.h"


//#include "orbiter.h"

using namespace std;

//using namespace orbiter::foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


void algebra_global_with_action::orbits_under_conjugation(
		long int *the_set, int set_size, groups::sims *S,
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
	actions::action A_conj;
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

	actions::action *A_conj_restricted;

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
	create_subgroups(SG,
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
		groups::strong_generators *SG,
		long int *the_set, int set_size, groups::sims *S, actions::action *A_conj,
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
	data_structures::sorting Sorting;
	ring_theory::longinteger_object go;

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


	cout << "There are " << Classes->nb_orbits << " orbits on the given set of elements by conjugation" << endl;
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
			ring_theory::longinteger_object go1;

			cout << "flag orbit " << h << " / " << nb_flag_orbits << ":" << endl;
			Flag_stab[h]->group_order(go1);
			rk0 = Flags[h * 3 + 0];
			rk1 = Flags[h * 3 + 1];
			rk2 = Flags[h * 3 + 2];
			S->element_unrank_lint(rk0, Elt0);
			S->element_unrank_lint(rk1, Elt1);
			S->element_unrank_lint(rk2, Elt2);
			cout << h << " : " << SO[h] << " : " << SOL[h] << " : (" << rk0 << "," << rk1 << "," << rk2 << ") : " << go1 << endl;
			cout << "The subgroup consists of the following three "
					"non-identity elements:" << endl;
			S->A->element_print_quick(Elt0, cout);
			S->A->element_print_quick(Elt1, cout);
			S->A->element_print_quick(Elt2, cout);

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
		f_is_definition[flag] = TRUE;
		iso_type_of_flag_orbit[flag] = nb_iso;
		flag_orbit_of_iso_type[nb_iso] = flag;
		upstep_transversal_size[nb_iso] = 1;

		data_structures_groups::vector_ge *transversal;

		transversal = NEW_OBJECT(data_structures_groups::vector_ge);
		transversal->init(S->A, 0);
		transversal->allocate(6, 0);
		for (h = 0; h < 6; h++) {
			S->A->element_one(transversal->ith(h), 0);
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
			S->A->element_invert(Transporter->ith(pt), Elt0, 0);
			for (int u = 0; u < 3; u++) {
				cur_flag_mapped1[u] = A_conj->element_image_of(cur_flag[u], Elt0, 0);
			}
			if (cur_flag_mapped1[0] != rk0) {
				cout << "cur_flag_mapped1[u] != rk0" << endl;
				exit(1);
			}



			if (!Sorting.lint_vec_search(the_set_sorted,
					set_size, cur_flag_mapped1[1], idx, 0 /*verbose_level*/)) {
				cout << "cannot find cur_flag[1] in the_set_sorted" << endl;
				exit(1);
			}
			pt = position[idx];
			j = Classes->orbit_number(pt);


			if (j == SO[flag]) {
				cout << "flag " << flag << " coset " << h << ", found an automorphism" << endl;

				int orbit_idx;

				Classes->transporter_from_point_to_orbit_rep(pt,
						orbit_idx, Elt1, verbose_level);

				S->A->element_mult(Elt0, Elt1, Elt2, 0);
				S->A->element_print_quick(Elt2, cout);

				for (int u = 0; u < 3; u++) {
					cur_flag_mapped1[u] = A_conj->element_image_of(cur_flag[u], Elt2, 0);
				}
				cout << "which maps as follows:" << endl;
				for (int u = 0; u < 3; u++) {
					cout << cur_flag[u] << " -> " << cur_flag_mapped1[u] << endl;
				}

				S->A->element_move(Elt2, transversal->ith(upstep_transversal_size[nb_iso]), 0);

				upstep_transversal_size[nb_iso]++;
			}
			else {
				if (!Sorting.int_vec_search(SO, nb_flag_orbits, j, idx)) {
					cout << "cannot find j in SO" << endl;
					exit(1);
				}
				cout << "flag " << flag << " coset " << h << ", fusing with flag " << idx << endl;
				f_fused[idx] = TRUE;
			}

		}

		groups::strong_generators *aut;

		aut = NEW_OBJECT(groups::strong_generators);

		if (f_v) {
			cout << "flag " << flag << " stab order = " << Flag_stab[flag]->group_order_as_lint()
					<< " upstep ransversal length = " << upstep_transversal_size[nb_iso] << endl;
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
		S->A->element_print_quick(Elt0, cout);
		S->A->element_print_quick(Elt1, cout);
		S->A->element_print_quick(Elt2, cout);

		Aut[i]->print_generators_tex(cout);

	}


	{
		char str[1000];
		string fname;
		string title, author, extra_praeamble;

		snprintf(str, 1000, "subgroups_of_order_4.tex");
		fname.assign(str);
		snprintf(str, 1000, "Subgroups of order 4");
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::create_subgroups before report" << endl;
			}
#if 1
			int h;
			ost << "There are " << nb_flag_orbits << " flag orbits:\\\\" << endl;
			for (h = 0; h < nb_flag_orbits; h++) {
				ring_theory::longinteger_object go1;

				ost << "Flag orbit " << h << " / " << nb_flag_orbits << ":\\\\" << endl;
				Flag_stab[h]->group_order(go1);
				rk0 = Flags[h * 3 + 0];
				rk1 = Flags[h * 3 + 1];
				rk2 = Flags[h * 3 + 2];
				S->element_unrank_lint(rk0, Elt0);
				S->element_unrank_lint(rk1, Elt1);
				S->element_unrank_lint(rk2, Elt2);
				cout << h << " : " << SO[h] << " : " << SOL[h] << " : (" << rk0 << "," << rk1 << "," << rk2 << ") : " << go1 << endl;
				ost << "The subgroup consists of the following three "
						"non-identity elements:\\\\" << endl;
				ost << "$$" << endl;
				S->A->element_print_latex(Elt0, ost);
				S->A->element_print_latex(Elt1, ost);
				S->A->element_print_latex(Elt2, ost);
				ost << "$$" << endl;
				ost << "The flag stabilizer is the following group:\\\\" << endl;
				Flag_stab[h]->print_generators_tex(ost);

			}

			ost << "\\bigskip" << endl;
#endif

			ost << "We found " << nb_iso << " conjugacy classes of subgroups\\\\" << endl;
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
				cout << i << " : " << flag << " : " <<  " : " << SO[flag] << " l=" << SOL[flag]
						<< " : " << rk0 << "," << rk1 << "," << rk2 << " : "
						<< upstep_transversal_size[i] << " : " << Aut[i]->group_order_as_lint() << endl;

				S->element_unrank_lint(rk0, Elt0);
				S->element_unrank_lint(rk1, Elt1);
				S->element_unrank_lint(rk2, Elt2);

				ost << "The subgroup consists of the following three "
						"non-identity elements:\\\\" << endl;
				ost << "$$" << endl;
				S->A->element_print_latex(Elt0, ost);
				S->A->element_print_latex(Elt1, ost);
				S->A->element_print_latex(Elt2, ost);
				ost << "$$" << endl;
				S->A->element_print_for_make_element(Elt0, ost);
				ost << "\\\\" << endl;
				S->A->element_print_for_make_element(Elt1, ost);
				ost << "\\\\" << endl;
				S->A->element_print_for_make_element(Elt2, ost);
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
		orbiter_kernel_system::file_io Fio;

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

void algebra_global_with_action::compute_orbit_of_set(
		long int *the_set, int set_size,
		actions::action *A1, actions::action *A2,
		data_structures_groups::vector_ge *gens,
		std::string &label_set,
		std::string &label_group,
		long int *&Table,
		int &orbit_length,
		int verbose_level)
// called by any_group::orbits_on_set_from_file
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::compute_orbit_of_set" << endl;
	}
	if (f_v) {
		cout << "algebra_global_with_action::compute_orbit_of_set A1=";
		A1->print_info();
		cout << "algebra_global_with_action::compute_orbit_of_set A2=";
		A2->print_info();
	}

	orbits_schreier::orbit_of_sets *OS;

	OS = NEW_OBJECT(orbits_schreier::orbit_of_sets);

	if (f_v) {
		cout << "algebra_global_with_action::compute_orbit_of_set before OS->init" << endl;
	}
	OS->init(A1, A2, the_set, set_size, gens, verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::compute_orbit_of_set after OS->init" << endl;
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


	string fname;

#if 0
	// write transporter as csv file:


	data_structures_groups::vector_ge *Coset_reps;

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
#endif

	// write as csv file:


	fname.assign(label_set);
	fname.append("_orbit_under_");
	fname.append(label_group);
	fname.append(".csv");

	if (f_v) {
		cout << "Writing orbit to file " << fname << endl;
	}
	orbiter_kernel_system::file_io Fio;

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
	//FREE_OBJECT(Coset_reps);
	if (f_v) {
		cout << "algebra_global_with_action::compute_orbit_of_set done" << endl;
	}
}


void algebra_global_with_action::conjugacy_classes_based_on_normal_forms(
		actions::action *A,
		groups::sims *override_Sims,
		std::string &label,
		std::string &label_tex,
		int verbose_level)
// called from group_theoretic_activity by means of any_group::classes_based_on_normal_form
{
	int f_v = (verbose_level >= 1);
	string prefix;
	string fname_output;
	orbiter_kernel_system::file_io Fio;
	int d;
	field_theory::finite_field *F;


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

	algebra::gl_classes C;
	algebra::gl_class_rep *R;
	int nb_classes;
	int *Mtx;
	int *Elt;
	int i, order;
	long int a;

	char str[1000];

	snprintf(str, sizeof(str), "_classes_based_on_normal_forms_%d_%d.tex", d, F->q);
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

		Int_vec_zero(Mtx, d * d + 1);
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
		Int_matrix_print(Elt, d, d);

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

	data_structures::tally T_order;

	T_order.init(Order, nb_classes, FALSE, 0);


	{
		ofstream ost(fname_output);
		orbiter_kernel_system::latex_interface L;

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

				Int_vec_zero(Mtx, d * d + 1);
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
				Int_matrix_print(Elt, d, d);

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



void algebra_global_with_action::classes_GL(field_theory::finite_field *F, int d,
		int f_no_eigenvalue_one, int verbose_level)
// called from interface_algebra
// creates an object of type action
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::classes_GL" << endl;
	}

	algebra::gl_classes C;
	algebra::gl_class_rep *R;
	int nb_classes;
	int i;


	C.init(d, F, verbose_level);

	C.make_classes(R, nb_classes, f_no_eigenvalue_one, verbose_level);

	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;
	int a;
	int *Mtx;
	int *Elt;



	A = NEW_OBJECT(actions::action);
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
		Int_matrix_print(Elt, d, d);

		C.print_matrix_and_centralizer_order_latex(
				cout, R + i);

		}


	char fname[1000];

	snprintf(fname, sizeof(fname), "Class_reps_GL_%d_%d.tex", d, F->q);
	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		L.head_easy(fp);
		C.report(fp, verbose_level);
		L.foot(fp);
	}

	//make_gl_classes(d, q, f_no_eigenvalue_one, verbose_level);

	FREE_int(Mtx);
	FREE_int(Elt);
	FREE_OBJECTS(R);
	FREE_OBJECT(A);
	if (f_v) {
		cout << "algebra_global_with_action::classes_GL done" << endl;
	}
}

void algebra_global_with_action::do_normal_form(int q, int d,
		int f_no_eigenvalue_one, int *data, int data_sz,
		int verbose_level)
// not called from anywhere at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_normal_form" << endl;
	}

	algebra::gl_classes C;
	algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);

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



	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
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
	Int_vec_print(cout, data, data_sz);
	cout << endl;

	//A->Sims->element_unrank_int(elt_idx, Elt);
	A->make_element(Elt, data, verbose_level);

	cout << "Looking at element:" << endl;
	Int_matrix_print(Elt, d, d);


	algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(algebra::gl_class_rep);

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


void algebra_global_with_action::do_identify_one(int q, int d,
		int f_no_eigenvalue_one, int elt_idx,
		int verbose_level)
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_identify_one" << endl;
	}
	algebra::gl_classes C;
	algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
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
	Int_matrix_print(Elt, d, d);


	algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(algebra::gl_class_rep);

	C.identify_matrix(Elt, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes, R1, 0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

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

void algebra_global_with_action::do_identify_all(int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_identify_all" << endl;
	}
	algebra::gl_classes C;
	algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	actions::action *A;
	ring_theory::longinteger_object Go;
	int *Class_count;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
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
	Int_vec_zero(Class_count, nb_classes);
	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	go = Go.as_int();
	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << ":" << endl;

		A->Sims->element_unrank_lint(i, Elt);
		Int_matrix_print(Elt, d, d);


		algebra::gl_class_rep *R1;

		R1 = NEW_OBJECT(algebra::gl_class_rep);

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

void algebra_global_with_action::do_random(int q, int d, int f_no_eigenvalue_one, int verbose_level)
// not called at all
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_random" << endl;
	}
	//gl_random_matrix(d, q, verbose_level);

	algebra::gl_classes C;
	algebra::gl_class_rep *Reps;
	int nb_classes;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);
	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);

	int *Mtx;
	int *Basis;
	int class_rep;


	Mtx = NEW_int(d * d);
	Basis = NEW_int(d * d);

	C.F->Linear_algebra->random_invertible_matrix(Mtx, d, verbose_level - 2);


	algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(algebra::gl_class_rep);

	C.identify_matrix(Mtx, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes,
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


void algebra_global_with_action::group_table(int q, int d, int f_poly, std::string &poly,
		int f_no_eigenvalue_one, int verbose_level)
// This function does too many things!
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::group_table" << endl;
	}
	algebra::gl_classes C;
	algebra::gl_class_rep *Reps;
	int nb_classes;
	int *Class_rep;
	int *List;
	int list_sz, a, b, j, h;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	if (f_poly) {
		F->init_override_polynomial(q, poly, FALSE /* f_without_tables */, 0);
	}
	else {
		F->finite_field_init(q, FALSE /* f_without_tables */, 0);
	}

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);


	actions::action *A;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;


	A = NEW_OBJECT(actions::action);
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
		Int_matrix_print(Elt, d, d);

		{
			ring_theory::unipoly_domain U(C.F);
			ring_theory::unipoly_object char_poly;



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
		Int_matrix_print(Elt, d, d);


		algebra::gl_class_rep *R1;

		R1 = NEW_OBJECT(algebra::gl_class_rep);

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
			A->element_mult(Elt1, Elt2, Elt3, 0);
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
		orbiter_kernel_system::latex_interface L;

		L.head(fp, FALSE /* f_book */, FALSE /* f_title */,
			title /*const char *title */, author /*const char *author */,
			FALSE /* f_toc */, FALSE /* f_landscape */, FALSE /* f_12pt */,
			FALSE /* f_enlarged_page */, FALSE /* f_pagenumbers */,
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

void algebra_global_with_action::centralizer_brute_force(int q, int d,
		int elt_idx, int verbose_level)
// problem elt_idx does not describe the group element uniquely.
// Reason: the sims chain is not canonical.
// creates a finite_field object and an action object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_brute_force" << endl;
	}
	actions::action *A;
	ring_theory::longinteger_object Go;
	field_theory::finite_field *F;
	data_structures_groups::vector_ge *nice_gens;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);

	A = NEW_OBJECT(actions::action);
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
	Int_matrix_print(Elt, d, d);

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

		A->element_move(Elt1, gens->ith(i), 0);
		}

	groups::sims *Cent;

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
		cout << "algebra_global_with_action::centralizer_brute_force done" << endl;
	}
}


void algebra_global_with_action::centralizer(int q, int d,
		int elt_idx, int verbose_level)
// creates a finite_field, and two actions
// using init_projective_group and init_general_linear_group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer" << endl;
	}
	field_theory::finite_field *F;
	actions::action *A_PGL;
	actions::action *A_GL;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);

	A_PGL = NEW_OBJECT(actions::action);
	A_PGL->init_projective_group(d /* n */, F,
		FALSE /* f_semilinear */,
		TRUE /* f_basis */, TRUE /* f_init_sims */,
		nice_gens,
		0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A_PGL->print_base();
	A_PGL->group_order(Go);

	A_GL = NEW_OBJECT(actions::action);
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
		cout << "algebra_global_with_action::centralizer done" << endl;
	}

}

void algebra_global_with_action::centralizer(int q, int d, int verbose_level)
// creates a finite_field, and an action
// using init_projective_group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer" << endl;
	}
	actions::action *A;
	field_theory::finite_field *F;
	ring_theory::longinteger_object Go;
	data_structures_groups::vector_ge *nice_gens;
	int go, i;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, 0);
	A = NEW_OBJECT(actions::action);
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
		cout << "algebra_global_with_action::centralizer done" << endl;
	}
}


void algebra_global_with_action::compute_regular_representation(
		actions::action *A, groups::sims *S,
		data_structures_groups::vector_ge *SG, int *&perm, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::compute_regular_representation" << endl;
	}
	ring_theory::longinteger_object go;
	int goi, i;
	combinatorics::combinatorics_domain Combi;

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
	if (f_v) {
		cout << "algebra_global_with_action::compute_regular_representation done" << endl;
	}
}

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

	cout << "presentation of length " << l << endl;
	cout << "primes: ";
	Int_vec_print(cout, primes, l);
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
		Int_vec_print(cout, word, 9);
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
		Int_vec_print(cout, word, l);
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
			Int_vec_print(cout, word, l);
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
	if (f_v) {
		cout << "algebra_global_with_action::presentation done" << endl;
	}
}


void algebra_global_with_action::do_eigenstuff(field_theory::finite_field *F,
		int size, int *Data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	discreta_matrix M;
	int i, j, k, a, h;
	//unipoly_domain U;
	//unipoly_object char_poly;
	number_theory::number_theory_domain NT;

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
	Int_vec_print(cout, roots, nb_roots);
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
		Int_vec_print_integer_matrix_width(cout, A,
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
		Int_vec_print_integer_matrix_width(cout, B,
				size, size, size, F->log10_of_q);

		cout << "B transposed:" << endl;
		F->Linear_algebra->transpose_matrix(B, Bt, size, size);
		Int_vec_print_integer_matrix_width(cout, Bt,
				size, size, size, F->log10_of_q);

		int f_special = FALSE;
		int f_complete = TRUE;
		int *base_cols;
		int nb_base_cols;
		int f_P = FALSE;
		int kernel_m, kernel_n, *kernel;

		base_cols = new int[size];
		kernel = new int[size * size];

		nb_base_cols = F->Linear_algebra->Gauss_int(Bt,
			f_special, f_complete, base_cols,
			f_P, NULL, size, size, size,
			verbose_level - 1);
		cout << "rank = " << nb_base_cols << endl;

		F->Linear_algebra->matrix_get_kernel(Bt, size, size, base_cols, nb_base_cols,
			kernel_m, kernel_n, kernel, 0 /* verbose_level */);
		cout << "kernel = left eigenvectors:" << endl;
		Int_vec_print_integer_matrix_width(cout, kernel,
				size, kernel_n, kernel_n, F->log10_of_q);

		int *vec1, *vec2;
		vec1 = new int[size];
		vec2 = new int[size];
		for (i = 0; i < size; i++) {
			vec1[i] = kernel[i * kernel_n + 0];
			}
		Int_vec_print(cout, vec1, size);
		cout << endl;
		F->PG_element_normalize_from_front(vec1, 1, size);
		Int_vec_print(cout, vec1, size);
		cout << endl;
		F->PG_element_rank_modified(vec1, 1, size, a);
		cout << "has rank " << a << endl;


		cout << "computing xA" << endl;

		F->Linear_algebra->mult_vector_from_the_left(vec1, A, vec2, size, size);
		Int_vec_print(cout, vec2, size);
		cout << endl;
		F->PG_element_normalize_from_front(vec2, 1, size);
		Int_vec_print(cout, vec2, size);
		cout << endl;
		F->PG_element_rank_modified(vec2, 1, size, a);
		cout << "has rank " << a << endl;

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
	number_theory::number_theory_domain NT;


	NT.factor_prime_power(q, p, f);
	domain *dom;

	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_ "
				"q=" << q << ", p=" << p << ", f=" << f << endl;
	}
	dom = allocate_finite_field_domain(q, verbose_level);

	A5_in_PSL_2_q(q, A, B, dom, verbose_level);

	{
		with w(dom);
		D.mult(A, B);

		if (f_v) {
			cout << "A5_in_PSL_2_q done" << endl;
			cout << "A=\n" << A << endl;
			cout << "B=\n" << B << endl;
			cout << "AB=\n" << D << endl;
			int AA[4], BB[4], DD[4];
			matrix_convert_to_numerical(A, AA, q);
			matrix_convert_to_numerical(B, BB, q);
			matrix_convert_to_numerical(D, DD, q);
			cout << "A=" << endl;
			Int_vec_print_integer_matrix_width(cout, AA, 2, 2, 2, 7);
			cout << "B=" << endl;
			Int_vec_print_integer_matrix_width(cout, BB, 2, 2, 2, 7);
			cout << "AB=" << endl;
			Int_vec_print_integer_matrix_width(cout, DD, 2, 2, 2, 7);
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
	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_ done" << endl;
	}
}

void algebra_global_with_action::A5_in_PSL_2_q(int q,
		layer2_discreta::discreta_matrix & A,
		layer2_discreta::discreta_matrix & B,
		layer2_discreta::domain *dom_GFq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_2_q" << endl;
	}
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
	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_2_q done" << endl;
	}
}

void algebra_global_with_action::A5_in_PSL_2_q_easy(int q,
		layer2_discreta::discreta_matrix & A,
		layer2_discreta::discreta_matrix & B,
		layer2_discreta::domain *dom_GFq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, r;
	integer zeta5, zeta5v, b, c, d, b2, e;

	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_2_q_easy "
				"verbose_level=" << verbose_level << endl;
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
	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_2_q_easy done" << endl;
	}
}


void algebra_global_with_action::A5_in_PSL_2_q_hard(int q,
		layer2_discreta::discreta_matrix & A,
		layer2_discreta::discreta_matrix & B,
		layer2_discreta::domain *dom_GFq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	with w(dom_GFq);
	unipoly m;
	int i, q2;
	discreta_matrix S, Sv, E, /*Sbart, SSbart,*/ AA, BB;
	integer a, b, m1;
	int norm_alpha, l;

	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_2_q_hard" << endl;
	}
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
		if (f.compare_with(e) == 0) {
			break;
		}
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
		if (f.compare_with(e) == 0) {
			break;
		}
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
	if (f_v) {
		cout << "algebra_global_with_action::A5_in_PSL_2_q_hard done" << endl;
	}
}

int algebra_global_with_action::proj_order(layer2_discreta::discreta_matrix &A)
{
	discreta_matrix B;
	int m, n;
	int ord;

	m = A.s_m();
	n = A.s_n();
	if (m != n) {
		cout << "algebra_global_with_action::proj_order m != n" << endl;
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

void algebra_global_with_action::trace(
		layer2_discreta::discreta_matrix &A,
		layer2_discreta::discreta_base &tr)
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

void algebra_global_with_action::elementwise_power_int(
		layer2_discreta::discreta_matrix &A, int k)
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

int algebra_global_with_action::is_in_center(
		layer2_discreta::discreta_matrix &B)
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


void algebra_global_with_action::matrix_convert_to_numerical(
		layer2_discreta::discreta_matrix &A, int *AA, int q)
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
	combinatorics::combinatorics_domain Combi;


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

		if (f_v) {
			cout << "partition ";
			Int_vec_print(cout, parts, nb_parts);
			cout << endl;
		}


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		Y->young_symmetrizer(parts, nb_parts, Tableau[cnt], elt1, elt2, h_alpha, verbose_level);


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



void algebra_global_with_action::report_tactical_decomposition_by_automorphism_group(
		ostream &ost, geometry::projective_space *P,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::report_tactical_decomposition_by_automorphism_group" << endl;
	}
	int *Mtx;
	int i, j, h;
	geometry::incidence_structure *Inc;
	Inc = NEW_OBJECT(geometry::incidence_structure);

	Mtx = NEW_int(P->N_points * P->N_lines);
	Int_vec_zero(Mtx, P->N_points * P->N_lines);

	for (j = 0; j < P->N_lines; j++) {
		for (h = 0; h < P->k; h++) {
			i = P->Implementation->Lines[j * P->k + h];
			Mtx[i * P->N_lines + j] = 1;
		}
	}

	Inc->init_by_matrix(P->N_points, P->N_lines, Mtx, 0 /* verbose_level*/);


	data_structures::partitionstack S;

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
		groups::schreier *Sch_points;
		groups::schreier *Sch_lines;
		Sch_points = NEW_OBJECT(groups::schreier);
		Sch_points->init(A_on_points, verbose_level - 2);
		Sch_points->initialize_tables();
		Sch_points->init_generators(*gens->gens /* *generators */, verbose_level - 2);
		Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "found " << Sch_points->nb_orbits
					<< " orbits on points" << endl;
		}
		Sch_lines = NEW_OBJECT(groups::schreier);
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
		poset_classification::poset_classification_control *Control,
		groups::linear_group *LG,
		int d, int target_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance" << endl;
	}

	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *PC;


	Control->f_depth = TRUE;
	Control->depth = target_depth;


	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance group set up, "
				"calling gen->init" << endl;
		cout << "LG->A2->A->f_has_strong_generators="
				<< LG->A2->f_has_strong_generators << endl;
	}

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);

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

	PC = NEW_OBJECT(poset_classification::poset_classification);
	PC->initialize_and_allocate_root_node(Control, Poset,
			target_depth, verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::linear_codes_with_bounded_minimum_distance before gen->main" << endl;
	}

	int t0;
	orbiter_kernel_system::os_interface Os;
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
		actions::action *A, groups::sims *S,
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


	Int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}

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

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element "
				"the element has order " << o << endl;
	}



	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element "
				"before centralizer_using_MAGMA" << endl;
	}

	groups::strong_generators *gens;

	A->centralizer_using_MAGMA(prefix,
			S, Elt, gens, verbose_level);


	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element "
				"after centralizer_using_MAGMA" << endl;
	}


	if (f_v) {
		cout << "generators for the centralizer are:" << endl;
		gens->print_generators_tex();
	}



	{
		string fname, title, author, extra_praeamble;
		char str[1000];

		fname.assign(prefix);
		fname.append("_centralizer.tex");
		snprintf(str, 1000, "Centralizer of element %s", label.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::centralizer_of_element "
						"before report" << endl;
			}
			gens->print_generators_tex(ost);

			if (f_v) {
				cout << "algebra_global_with_action::centralizer_of_element "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(data);

	if (f_v) {
		cout << "algebra_global_with_action::centralizer_of_element done" << endl;
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

	prefix.assign(A->label);
	prefix.append("_elt");
	//prefix.append(label);

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	Int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}

	A->make_element(Elt, data, 0 /* verbose_level */);




	{
		string fname, title, author, extra_praeamble;
		char str[1000];

		fname.assign(prefix);
		fname.append("_permutation.tex");
		snprintf(str, 1000, "Permutation representation of element");
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "algebra_global_with_action::permutation_representation_of_element "
						"before report" << endl;
			}

			ost << "$$" << endl;
			A->element_print_latex(Elt, ost);
			ost << "$$" << endl;

			ost << "$$" << endl;
			A->element_print_as_permutation(Elt, ost);
			ost << "$$" << endl;

			if (f_v) {
				cout << "algebra_global_with_action::permutation_representation_of_element "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

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


void algebra_global_with_action::normalizer_of_cyclic_subgroup(
		actions::action *A, groups::sims *S,
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


	Int_vec_scan(element_description, data, data_len);


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

	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
				"the element has order " << o << endl;
	}



	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
				"before normalizer_of_cyclic_group_using_MAGMA" << endl;
	}

	groups::strong_generators *gens;

	A->normalizer_of_cyclic_group_using_MAGMA(prefix,
			S, Elt, gens, verbose_level);



	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
				"after normalizer_of_cyclic_group_using_MAGMA" << endl;
	}



	cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup "
			"generators for the normalizer are:" << endl;
	gens->print_generators_tex();


	{

		string fname, title, author, extra_praeamble;
		char str[1000];

		fname.assign(prefix);
		fname.append(".tex");
		snprintf(str, 1000, "Normalizer of cyclic subgroup %s", label.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);

			ring_theory::longinteger_object go;
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
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}





	FREE_int(data);
	FREE_int(Elt);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "algebra_global_with_action::normalizer_of_cyclic_subgroup done" << endl;
	}
}

void algebra_global_with_action::find_subgroups(
		actions::action *A, groups::sims *S,
		int subgroup_order,
		std::string &label,
		int &nb_subgroups,
		groups::strong_generators *&H_gens,
		groups::strong_generators *&N_gens,
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
	snprintf(str, sizeof(str), "_find_subgroup_of_order_%d", subgroup_order);
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
		actions::action *A, groups::strong_generators *SG,
		data_structures_groups::vector_ge *cosets,
		int *&relative_order_table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	int *Elt2;
	//int *Elt3;
	groups::sims *S;
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

#if 0
void algebra_global_with_action::do_orbits_on_polynomials(
		groups::linear_group *LG,
		int degree_of_poly,
		int f_recognize, std::string &recognize_text,
		int f_draw_tree, int draw_tree_idx,
		graphics::layered_graph_draw_options *Opt,
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


		snprintf(str, sizeof(str), "_orbit_%d_tree", draw_tree_idx);

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
#endif

void algebra_global_with_action::representation_on_polynomials(
		groups::linear_group *LG,
		int degree_of_poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_stabilizer = TRUE;
	//int f_draw_tree = TRUE;


	if (f_v) {
		cout << "algebra_global_with_action::representation_on_polynomials" << endl;
	}


	field_theory::finite_field *F;
	actions::action *A;
	//matrix_group *M;
	int n;
	//int degree;
	ring_theory::longinteger_object go;

	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}

	ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);


	monomial_ordering_type Monomial_ordering_type = t_PART;


	HPD->init(F, n /* nb_var */, degree_of_poly,
			Monomial_ordering_type,
			verbose_level);

	actions::action *A2;

	A2 = NEW_OBJECT(actions::action);
	A2->induced_action_on_homogeneous_polynomials(A,
		HPD,
		FALSE /* f_induce_action */, NULL,
		verbose_level);

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
		LG->nice_gens->matrix_representation(A_on_HPD, M, nb_gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "algebra_global_with_action::representation_on_polynomials "
					"using strong generators" << endl;
		}
		LG->Strong_gens->gens->matrix_representation(A_on_HPD, M, nb_gens, verbose_level);
	}

	for (i = 0; i < nb_gens; i++) {
		cout << "matrix " << i << " / " << nb_gens << ":" << endl;
		Int_matrix_print(M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);
	}

	for (i = 0; i < nb_gens; i++) {
		string fname;
		char str[1000];
		orbiter_kernel_system::file_io Fio;

		fname.assign(LG->label);
		snprintf(str, sizeof(str), "_rep_%d_%d.csv", degree_of_poly, i);
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
		field_theory::finite_field *F, int n, std::string &coeffs_text, int verbose_level)
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

	algebra_global_with_action A;

	A.do_eigenstuff(F, n, Data, verbose_level);

	FREE_int(Data);
	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_with_coefficients done" << endl;
	}
}

void algebra_global_with_action::do_eigenstuff_from_file(
		field_theory::finite_field *F, int n, std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_eigenstuff_from_file" << endl;
	}

	orbiter_kernel_system::file_io Fio;
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







void algebra_global_with_action::orbits_on_points(
		actions::action *A2,
		groups::strong_generators *Strong_gens,
		int f_load_save,
		std::string &prefix,
		groups::orbits_on_something *&Orb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points" << endl;
	}
	//cout << "computing orbits on points:" << endl;


	//orbits_on_something *Orb;

	Orb = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points before Orb->init" << endl;
	}
	Orb->init(
			A2,
			Strong_gens,
			f_load_save,
			prefix,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points after Orb->init" << endl;
	}




	if (f_v) {
		cout << "algebra_global_with_action::orbits_on_points done" << endl;
	}
}

void algebra_global_with_action::find_singer_cycle(any_group *Any_group,
		actions::action *A1, actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::find_singer_cycle" << endl;
	}
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = SG->create_sims(verbose_level);

	if (f_v) {
		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
	}

	int *Elt;
	ring_theory::longinteger_object go;
	int i, d, q, cnt, ord, order;
	number_theory::number_theory_domain NT;

	if (!A1->is_matrix_group()) {
		cout << "group_theoretic_activity::find_singer_cycle needs matrix group" << endl;
		exit(1);
	}
	groups::matrix_group *M;

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
		if (f_v) {
			cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
			A2->element_print(Elt, cout);
			cout << endl;
			A2->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
		cnt++;
	}
	if (f_v) {
		cout << "we found " << cnt << " group elements of order " << order << endl;
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "algebra_global_with_action::find_singer_cycle done" << endl;
	}
}

void algebra_global_with_action::search_element_of_order(any_group *Any_group,
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
	ring_theory::longinteger_object go;
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
		if (f_v) {
			cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
			A2->element_print(Elt, cout);
			cout << endl;
			A2->element_print_as_permutation(Elt, cout);
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

void algebra_global_with_action::find_standard_generators(any_group *Any_group,
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
	ring_theory::longinteger_object go;
	int i, j, cnt, ord;

	Elt_a = NEW_int(A1->elt_size_in_int);
	Elt_b = NEW_int(A1->elt_size_in_int);
	Elt_ab = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt_a);


		ord = A2->element_order(Elt_a);

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


			ord = A2->element_order(Elt_b);

			if (ord != order_b) {
				continue;
			}

			A2->element_mult(Elt_a, Elt_b, Elt_ab, 0);

			ord = A2->element_order(Elt_ab);

			if (ord != order_ab) {
				continue;
			}

			if (f_v) {
				cout << "algebra_global_with_action::find_standard_generators a = " << setw(5) << i << ", b=" << setw(5) << j << " : " << cnt << ":" << endl;
				cout << "a=" << endl;
				A2->element_print(Elt_a, cout);
				cout << endl;
				A2->element_print_as_permutation(Elt_a, cout);
				cout << endl;
				cout << "b=" << endl;
				A2->element_print(Elt_b, cout);
				cout << endl;
				A2->element_print_as_permutation(Elt_b, cout);
				cout << endl;
				cout << "ab=" << endl;
				A2->element_print(Elt_ab, cout);
				cout << endl;
				A2->element_print_as_permutation(Elt_ab, cout);
				cout << endl;
			}
			cnt++;
		}
	}
	if (f_v) {
		cout << "algebra_global_with_action::find_standard_generators "
				"we found " << cnt << " group elements with "
				"ord_a = " << order_a << " ord_b  = " << order_b << " and ord_ab = " << order_ab << endl;
	}

	FREE_int(Elt_a);
	FREE_int(Elt_b);
	FREE_int(Elt_ab);
	if (f_v) {
		cout << "algebra_global_with_action::find_standard_generators done" << endl;
	}
}

void algebra_global_with_action::Nth_roots(field_theory::finite_field *F,
		int n, int verbose_level)
{
	field_theory::nth_roots *Nth;

	Nth = NEW_OBJECT(field_theory::nth_roots);

	Nth->init(F, n, verbose_level);

	orbiter_kernel_system::file_io Fio;
	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "Nth_roots_q%d_n%d.tex", F->q, n);
		fname.assign(str);
		snprintf(str, 1000, "Nth roots");
		title.assign(str);




		{
			ofstream ost(fname);
			number_theory::number_theory_domain NT;



			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			Nth->report(ost, verbose_level);

			L.foot(ost);


		}

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}

}


}}}

