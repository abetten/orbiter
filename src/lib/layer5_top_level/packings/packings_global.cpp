/*
 * packings_global.cpp
 *
 *  Created on: Oct 18, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


packings_global::packings_global()
{
	Record_birth();
}


packings_global::~packings_global()
{
	Record_death();
}


void packings_global::orbits_under_conjugation(
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
		cout << "packings_global::orbits_under_conjugation" << endl;
	}
	actions::action *A_conj;
	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"before create_induced_action_by_conjugation" << endl;
	}
	A_conj = S->A->Induced_action->create_induced_action_by_conjugation(S,
			false /* f_ownership */, false /* f_basis */, S,
			verbose_level);
	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"after create_induced_action_by_conjugation" << endl;
	}

	actions::action *A_conj_restricted;
	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_on_group");
	label_of_set_tex.assign("\\_on\\_group");

	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"before A_conj->restricted_action" << endl;
	}

	A_conj_restricted = A_conj->Induced_action->restricted_action(
			the_set, set_size,
			label_of_set, label_of_set_tex,
			verbose_level);

	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"after A_conj->restricted_action" << endl;
	}



	groups::schreier Classes;
	Classes.init(A_conj_restricted, verbose_level - 2);
	Classes.init_generators(*SG->gens, verbose_level - 2);

	int print_interval = 10000;

	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"before Classes.compute_all_point_orbits" << endl;
	}
	Classes.compute_all_point_orbits(print_interval, 1 /*verbose_level - 1*/);
	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"after Classes.compute_all_point_orbits" << endl;
		cout << "found " << Classes.Forest->nb_orbits << " conjugacy classes" << endl;
	}


	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"before create_subgroups" << endl;
	}
	create_subgroups(
			SG,
			the_set, set_size, S, A_conj,
			&Classes,
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "packings_global::orbits_under_conjugation "
				"after create_subgroups" << endl;
	}

	FREE_OBJECT(A_conj);

	if (f_v) {
		cout << "packings_global::orbits_under_conjugation done" << endl;
	}
}

void packings_global::create_subgroups(
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
		cout << "packings_global::create_subgroups" << endl;
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


	cout << "There are " << Classes->Forest->nb_orbits << " orbits "
			"on the given set of elements by conjugation" << endl;
	for (i = 0; i < Classes->Forest->nb_orbits; i++) {

		f = Classes->Forest->orbit_first[i];
		l = Classes->Forest->orbit_len[i];
		rep = Classes->Forest->orbit[f];
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

	f = Classes->Forest->orbit_first[0];
	l = Classes->Forest->orbit_len[0];
	if (l != 1) {
		cout << "packings_global::create_subgroups l != 1" << endl;
		exit(1);
	}
	rep = Classes->Forest->orbit[f];
	rk0 = the_set[rep];

	S->element_unrank_lint(rk0, Elt0);

	nb_flag_orbits = 0;
	Flags = NEW_lint(Classes->Forest->nb_orbits * 3);
	Flag_stab = new groups::pstrong_generators [Classes->Forest->nb_orbits];
	SO = NEW_int(Classes->Forest->nb_orbits);
	SOL = NEW_int(Classes->Forest->nb_orbits);

	nb_reject = 0;

	for (j = 1; j < Classes->Forest->nb_orbits; j++) {



		f = Classes->Forest->orbit_first[j];
		l = Classes->Forest->orbit_len[j];
		rep = Classes->Forest->orbit[f];
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
				cout << "packings_global::create_subgroups "
						"before Classes->stabilizer_orbit_rep" << endl;
			}
			Flag_stab[nb_flag_orbits] = Classes->stabilizer_orbit_rep(
					S->A,
					go,
					j /* orbit_idx */, 0 /*verbose_level*/);
			if (f_v) {
				cout << "packings_global::create_subgroups "
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
			j = Classes->Forest->orbit_number(pt);


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
				cout << "packings_global::create_subgroups "
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
				cout << "packings_global::create_subgroups after report" << endl;
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
		cout << "packings_global::create_subgroups done" << endl;
	}
}





#if 0
void packings_global::merge_packings(
		std::string *fnames, int nb_files,
		std::string &file_of_spreads,
		combinatorics::canonical_form_classification::classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packings_global::merge_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);


	// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
	long int *Spread_table;
	int nb_spreads;
	int spread_size;

	if (f_v) {
		cout << "packings_global::merge_packings "
				"Reading spread table from file "
				<< file_of_spreads << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}

	int f, g, N, table_length, nb_reject = 0;

	N = 0;

	if (f_v) {
		cout << "packings_global::merge_packings "
				"counting the overall number of input packings" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "packings_global::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);

		table_length = S->nb_rows - 1;
		N += table_length;



		FREE_OBJECT(S);

	}

	if (f_v) {
		cout << "packings_global::merge_packings file "
				<< "we have " << N << " packings in "
				<< nb_files << " files" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "packings_global::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);
		if (false /*f_v3*/) {
			S->print_table(cout, false);
			}

		int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
		int nb_rows_idx, nb_cols_idx, canonical_form_idx;

		ago_idx = S->find_by_column("ago");
		original_file_idx = S->find_by_column("original_file");
		input_idx_idx = S->find_by_column("input_idx");
		input_set_idx = S->find_by_column("input_set");
		nb_rows_idx = S->find_by_column("nb_rows");
		nb_cols_idx = S->find_by_column("nb_cols");
		canonical_form_idx = S->find_by_column("canonical_form");

		table_length = S->nb_rows - 1;

		//rep,ago,original_file,input_idx,input_set,nb_rows,nb_cols,canonical_form


		for (g = 0; g < table_length; g++) {

			int ago;
			char *text;
			long int *the_set_in;
			int set_size_in;
			long int *canonical_labeling;
			int canonical_labeling_sz;
			int nb_rows, nb_cols;
			object_in_projective_space *OiP;


			ago = S->get_int(g + 1, ago_idx);
			nb_rows = S->get_int(g + 1, nb_rows_idx);
			nb_cols = S->get_int(g + 1, nb_cols_idx);

			text = S->get_string(g + 1, input_set_idx);
			Orbiter->Lint_vec.scan(text, the_set_in, set_size_in);


			if (f_v) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << endl;
				//int_vec_print(cout, the_set_in, set_size_in);
				//cout << endl;
				}

			if (false) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (false) {
				cout << "text=" << text << endl;
			}
			Orbitr->Lint_vec.scan(text, canonical_labeling, canonical_labeling_sz);
			if (false) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				Lint_vec_print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "packings_global::merge_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (false) {
				cout << "packings_global::merge_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (false) {
				cout << "packings_global::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = true;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (false) {
				cout << "packings_global::merge_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (false) {
				cout << "packings_global::merge_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "packings_global::merge_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "packings_global::merge_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (false) {
				cout << "packings_global::merge_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (CB->n == 0) {
				if (f_v) {
					cout << "packings_global::merge_packings "
							"before CB->init" << endl;
				}
				CB->init(N, canonical_form_len, verbose_level);
				}
			if (f_v) {
				cout << "packings_global::merge_packings "
						"before CB->add" << endl;
			}
			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				nb_reject++;
			}
			if (f_v) {
				cout << "packings_global::merge_packings "
						"CB->add returns f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject << endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;


			FREE_lint(the_set_in);
			//FREE_int(canonical_labeling);
			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_uchar(canonical_form);

		} // next g



	} // next f

	if (f_v) {
		cout << "packings_global::merge_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings" << endl;
		}


	//FREE_OBJECT(CB);
	FREE_lint(Spread_table);

	if (f_v) {
		cout << "packings_global::merge_packings done" << endl;
	}
}

void packings_global::select_packings(
		std::string &fname,
		std::string &file_of_spreads_original,
		geometry::finite_geometries::spread_tables *Spread_tables,
		int f_self_polar,
		int f_ago, int select_ago,
		combinatorics::canonical_form_classification::classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	int nb_accept = 0;
	file_io Fio;

	if (f_v) {
		cout << "packings_global::select_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);



	long int *Spread_table;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "packings_global::select_packings "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads_original,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "packings_global::select_packings "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "packings_global::select_packings "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	long int *set;
	long int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_lint(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		Lint_vec_copy(Spread_tables->spread_table +
				i * spread_size, set, spread_size);
		Sorting.lint_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, (void *) set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "packings_global::select_packings "
					"cannot find spread " << i << " = ";
			Lint_vec_print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "packings_global::select_packings file "
				<< fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (false /*f_v3*/) {
		S->print_table(cout, false);
		}

	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = true;


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		long int *the_set_in;
		int set_size_in;
		long int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept = false;
		int *set1;
		int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		Orbiter->Lint_vec.scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		if (f_self_polar) {
			set1 = NEW_int(packing_size);
			set2 = NEW_int(packing_size);

			// test if self-polar:
			for (i = 0; i < packing_size; i++) {
				a = the_set_in[i];
				b = s2l[a];
				set1[i] = b;
			}
			Sorting.int_vec_heapsort(set1, packing_size);
			for (i = 0; i < packing_size; i++) {
				a = set1[i];
				b = Spread_tables->dual_spread_idx[a];
				set2[i] = b;
			}
			Sorting.int_vec_heapsort(set2, packing_size);

#if 0
			cout << "set1: ";
			int_vec_print(cout, set1, packing_size);
			cout << endl;
			cout << "set2: ";
			int_vec_print(cout, set2, packing_size);
			cout << endl;
#endif
			if (int_vec_compare(set1, set2, packing_size) == 0) {
				cout << "The packing is self-polar" << endl;
				f_accept = true;
			}
			else {
				f_accept = false;
			}
			FREE_int(set1);
			FREE_int(set2);
		}
		if (f_ago) {
			if (ago == select_ago) {
				f_accept = true;
			}
			else {
				f_accept = false;
			}
		}



		if (f_accept) {

			nb_accept++;


			if (false) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (false) {
				cout << "text=" << text << endl;
			}
			Orbiter->Lint_vec.scan(text, canonical_labeling, canonical_labeling_sz);
			if (false) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				Lint_vec_print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "packings_global::select_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (false) {
				cout << "packings_global::select_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (false) {
				cout << "packings_global::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = true;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (false) {
				cout << "packings_global::select_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (false) {
				cout << "packings_global::select_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "packings_global::select_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "packings_global::select_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (false) {
				cout << "packings_global::select_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (f_first) {
				if (f_v) {
					cout << "packings_global::select_packings "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = false;
			}


			if (f_v) {
				cout << "packings_global::select_packings "
						"before CB->add" << endl;
			}

			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (f_v) {
				cout << "packings_global::select_packings "
						"CB->add returns f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_lint(the_set_in);

	} // next g




	if (f_v) {
		cout << "packings_global::select_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	//FREE_OBJECT(CB);
	FREE_lint(Spread_table);

	if (f_v) {
		cout << "packings_global::select_packings done" << endl;
	}
}



void packings_global::select_packings_self_dual(
		std::string &fname,
		std::string &file_of_spreads_original,
		int f_split, int split_r, int split_m,
		geometry::finite_geometries::spread_tables *Spread_tables,
		combinatorics::canonical_form_classification::classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packings_global::select_packings_self_dual" << endl;
	}


	int nb_accept = 0;
	file_io Fio;

	CB = NEW_OBJECT(classify_bitvectors);



	long int *Spread_table_original;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads_original,
			Spread_table_original, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "packings_global::select_packings_self_dual "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "packings_global::select_packings_self_dual "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	long int *set;
	long int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_lint(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		Lint_vec_copy(Spread_table_original +
				i * spread_size, set, spread_size);
		Sorting.lint_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, (int *) set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "packings_global::select_packings_self_dual "
					"cannot find spread " << i << " = ";
			Lint_vec_print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"file " << fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (false /*f_v3*/) {
		S->print_table(cout, false);
		}

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"read file " << fname << endl;
	}


	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"finding column indices" << endl;
	}

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = true;


	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"first pass, table_length=" << table_length << endl;
	}

	// first pass: build up the database:

	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		long int *the_set_in;
		int set_size_in;
		long int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		Orbiter->Lint_vec.scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		f_accept = true;



		if (f_accept) {

			nb_accept++;


			if (false) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (false) {
				cout << "text=" << text << endl;
			}
			Orbiter->Lint_vec.scan(text, canonical_labeling, canonical_labeling_sz);
			if (false) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				Lint_vec_print(cout,
						canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "packings_global::select_packings_self_dual "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
					Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = true;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "packings_global::select_packings_self_dual "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "packings_global::select_packings_self_dual "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (f_first) {
				if (f_v) {
					cout << "packings_global::select_packings_self_dual "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = false;
			}


			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"before CB->add" << endl;
			}

			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (false) {
				cout << "packings_global::select_packings_self_dual "
						"CB->add f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_lint(the_set_in);

	} // next g




	if (f_v) {
		cout << "packings_global::select_packings_self_dual done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	// second pass:

	int nb_self_dual = 0;
	int g1 = 0;
	int *self_dual_cases;
	int nb_self_dual_cases = 0;


	self_dual_cases = NEW_int(table_length);


	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"second pass, table_length="
				<< table_length << endl;
	}


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		int *the_set_in;
		int set_size_in;
		int *canonical_labeling1;
		int *canonical_labeling2;
		//int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP1;
		object_in_projective_space *OiP2;
		long int *set1;
		long int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		Int_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;


		if (f_split) {
			if ((g % split_m) != split_r) {
				continue;
			}
		}
		g1++;
		if (f_v && (g1 % 100) == 0) {
			cout << "File " << fname
					<< ", case " << g1 << " input set " << g << " / "
					<< table_length
					<< " nb_self_dual=" << nb_self_dual << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		set1 = NEW_lint(packing_size);
		set2 = NEW_lint(packing_size);

		for (i = 0; i < packing_size; i++) {
			a = the_set_in[i];
			b = s2l[a];
			set1[i] = b;
		}
		Sorting.lint_vec_heapsort(set1, packing_size);
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = Spread_tables->dual_spread_idx[a];
			set2[i] = l2s[b];
		}
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = l2s[a];
			set1[i] = b;
		}
		Sorting.lint_vec_heapsort(set1, packing_size);
		Sorting.lint_vec_heapsort(set2, packing_size);

#if 0
		cout << "set1: ";
		int_vec_print(cout, set1, packing_size);
		cout << endl;
		cout << "set2: ";
		int_vec_print(cout, set2, packing_size);
		cout << endl;
#endif




		OiP1 = NEW_OBJECT(object_in_projective_space);
		OiP2 = NEW_OBJECT(object_in_projective_space);

		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"before init_packing_from_spread_table" << endl;
		}
		OiP1->init_packing_from_spread_table(P, set1,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		OiP2->init_packing_from_spread_table(P, set2,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"after init_packing_from_spread_table" << endl;
		}
		OiP1->f_has_known_ago = true;
		OiP1->known_ago = ago;



		uchar *canonical_form1;
		uchar *canonical_form2;
		int canonical_form_len;



		int *Incma_in1;
		int *Incma_out1;
		int *Incma_in2;
		int *Incma_out2;
		int nb_rows1, nb_cols1;
		int *partition;
		//uchar *canonical_form1;
		//uchar *canonical_form2;
		//int canonical_form_len;


		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"before encode_incma" << endl;
		}
		OiP1->encode_incma(Incma_in1, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		OiP2->encode_incma(Incma_in2, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"after encode_incma" << endl;
		}
		if (nb_rows1 != nb_rows) {
			cout << "packings_global::select_packings_self_dual "
					"nb_rows1 != nb_rows" << endl;
			exit(1);
		}
		if (nb_cols1 != nb_cols) {
			cout << "packings_global::select_packings_self_dual "
					"nb_cols1 != nb_cols" << endl;
			exit(1);
		}


		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"before PA->set_stabilizer_of_object" << endl;
			}


		canonical_labeling1 = NEW_int(nb_rows * nb_cols);
		canonical_labeling2 = NEW_int(nb_rows * nb_cols);

		canonical_labeling(
				OiP1,
				canonical_labeling1,
				0 /*verbose_level - 2*/);
		canonical_labeling(
				OiP2,
				canonical_labeling2,
				0 /*verbose_level - 2*/);


		OiP1->input_fname = S->get_string(g + 1, original_file_idx);
		OiP1->input_idx = S->get_int(g + 1, input_idx_idx);
		OiP2->input_fname = S->get_string(g + 1, original_file_idx);
		OiP2->input_idx = S->get_int(g + 1, input_idx_idx);

		text = S->get_string(g + 1, input_set_idx);

		OiP1->set_as_string.assign(text);

		OiP2->set_as_string.assign(text);

		int i, j, ii, jj, a, ret;
		int L = nb_rows * nb_cols;

		Incma_out1 = NEW_int(L);
		Incma_out2 = NEW_int(L);
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling1[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling1[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out1[i * nb_cols + j] = Incma_in1[ii * nb_cols + jj];
				}
			}
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling2[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling2[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out2[i * nb_cols + j] = Incma_in2[ii * nb_cols + jj];
				}
			}
		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"before bitvector_allocate_and_coded_length" << endl;
		}
		canonical_form1 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out1[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form1, a);
					}
				}
			}
		canonical_form2 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out2[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form2, a);
					}
				}
			}


		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"before CB->search" << endl;
		}

		int idx1, idx2;

		ret = CB->search(canonical_form1, idx1, 0 /*verbose_level*/);

		if (ret == false) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form1, idx1, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form1: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form1[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
			cout << endl;
#endif
			exit(1);
		}
		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"CB->search returns idx1=" << idx1 << endl;
		}
		ret = CB->search(canonical_form2, idx2, 0 /*verbose_level*/);

		if (ret == false) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form2, idx2, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form2: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form2[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
#endif
			exit(1);
		}
		if (false) {
			cout << "packings_global::select_packings_self_dual "
					"CB->search returns idx2=" << idx2 << endl;
		}

		FREE_int(Incma_in1);
		FREE_int(Incma_out1);
		FREE_int(Incma_in2);
		FREE_int(Incma_out2);
		FREE_int(partition);
		FREE_int(canonical_labeling1);
		FREE_int(canonical_labeling2);
		FREE_uchar(canonical_form1);
		FREE_uchar(canonical_form2);

		FREE_lint(set1);
		FREE_lint(set2);

		if (idx1 == idx2) {
			cout << "self-dual" << endl;
			nb_self_dual++;
			self_dual_cases[nb_self_dual_cases++] = g;
		}

		FREE_int(the_set_in);

	} // next g

	string fname_base;
	string fname_self_dual;
	char str[1000];
	string_tools String;

	fname_base.assign(fname);
	String.chop_off_extension(fname_base);
	fname_self_dual.assign(fname);
	String.chop_off_extension(fname_self_dual);
	if (f_split) {
		snprintf(str, sizeof(str), "_self_dual_r%d_m%d.csv", split_r, split_m);
	}
	else {
		snprintf(str, sizeof(str), "_self_dual.csv");
	}
	fname_self_dual += str;
	cout << "saving self_dual_cases to file " << fname_self_dual << endl;
	Fio.int_vec_write_csv(self_dual_cases, nb_self_dual_cases,
			fname_self_dual, "self_dual_idx");
	cout << "written file " << fname_self_dual
			<< " of size " << Fio.file_size(fname_self_dual) << endl;



	//FREE_OBJECT(CB);
	FREE_lint(Spread_table_original);

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"done, nb_self_dual = " << nb_self_dual << endl;
	}

}
#endif



}}}


