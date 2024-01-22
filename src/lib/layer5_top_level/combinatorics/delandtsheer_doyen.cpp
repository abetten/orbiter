/*
 * delandtsheer_doyen.cpp
 *
 *  Created on: Nov 5, 2019
 *      Author: anton
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


static void delandtsheer_doyen_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);




delandtsheer_doyen::delandtsheer_doyen()
{

	Descr = NULL;

	//std::string label;

	Xsize = 0; // = D = q1 = # of rows
	Ysize = 0; // = C = q2 = # of cols

	V = 0;
	b = 0;


	line = NULL;        // [K];
	row_sum = NULL;
	col_sum = NULL;

	Pair_search_control = NULL;
	Search_control = NULL;

	M1 = NULL;
	M2 = NULL;
	A1 = NULL;
	A2 = NULL;
	SG = NULL;
	F1 = NULL;
	F2 = NULL;
	A = NULL;
	A0 = NULL;
	P = NULL;

	Poset_pairs = NULL;
	Poset_search = NULL;
	Pairs = NULL;
	Gen = NULL;

	pair_orbit = NULL;
	nb_orbits = 0;
	transporter = NULL;
	tmp_Elt = NULL;
	orbit_length = NULL;
	orbit_covered = NULL;
	orbit_covered2 = NULL;
	orbit_covered_max = NULL;
		// orbit_covered_max[i] = orbit_length[i] / b;
	orbits_covered = NULL;


	// intersection type tests:

	inner_pairs_in_rows = 0;
	inner_pairs_in_cols = 0;

	// row intersection type
	row_type_cur = NULL; 		// [nb_row_types + 1]
	row_type_this_or_bigger = NULL; 	// [nb_row_types + 1]

	// col intersection type
	col_type_cur = NULL; 		// [nb_col_types + 1]
	col_type_this_or_bigger = NULL; 	// [nb_col_types + 1]



	// for testing the mask:
	f_row_used = NULL; // [Xsize];
	f_col_used = NULL; // [Ysize];
	row_idx = NULL; // [Xsize];
	col_idx = NULL; // [Ysize];
	singletons = NULL; // [K];

	// temporary data
	row_col_idx = NULL; // [Xsize];
	col_row_idx = NULL; // [Ysize];

	// a file where we print the solution, it has the extension bblt
	// for "base block line transitive" design
	//fp_sol = NULL;

	live_points = NULL;
	nb_live_points = 0;

}

delandtsheer_doyen::~delandtsheer_doyen()
{
	if (line) {
		FREE_lint(line);
	}
	if (row_sum) {
		FREE_int(row_sum);
	}
	if (col_sum) {
		FREE_int(col_sum);
	}
	if (pair_orbit) {
		FREE_int(pair_orbit);
	}
	if (transporter) {
		FREE_int(transporter);
	}
	if (tmp_Elt) {
		FREE_int(tmp_Elt);
	}
	if (orbit_length) {
		FREE_int(orbit_length);
	}
	if (orbit_covered) {
		FREE_int(orbit_covered);
	}
	if (orbit_covered2) {
		FREE_int(orbit_covered2);
	}
	if (orbit_covered_max) {
		FREE_int(orbit_covered_max);
	}
	if (orbits_covered) {
		FREE_int(orbits_covered);
	}
	if (row_type_cur) {
		FREE_int(row_type_cur);
	}
	if (row_type_this_or_bigger) {
		FREE_int(row_type_this_or_bigger);
	}
	if (col_type_cur) {
		FREE_int(col_type_cur);
	}
	if (col_type_this_or_bigger) {
		FREE_int(col_type_this_or_bigger);
	}
	if (f_row_used) {
		FREE_int(f_row_used);
	}
	if (f_col_used) {
		FREE_int(f_col_used);
	}
	if (row_idx) {
		FREE_int(row_idx);
	}
	if (col_idx) {
		FREE_int(col_idx);
	}
	if (singletons) {
		FREE_int(singletons);
	}
	if (row_col_idx) {
		FREE_int(row_col_idx);
	}
	if (col_row_idx) {
		FREE_int(col_row_idx);
	}
	if (live_points) {
		FREE_lint(live_points);
	}
}

void delandtsheer_doyen::init(
		delandtsheer_doyen_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen::init" << endl;
	}


	delandtsheer_doyen::Descr = Descr;



	if (!Descr->f_K) {
		cout << "please use -K <K> to specify K" << endl;
		exit(1);
	}
#if 0
	if (!Descr->f_depth) {
		cout << "please use -depth <depth> to specify depth" << endl;
		exit(1);
	}
#endif
	if (Descr->f_R) {
		row_type_cur = NEW_int(Descr->nb_row_types + 1);
		Int_vec_zero(row_type_cur, Descr->nb_row_types + 1);
		row_type_this_or_bigger = NEW_int(Descr->nb_row_types + 1);
	}

	if (Descr->f_C) {
		col_type_cur = NEW_int(Descr->nb_col_types + 1);
		Int_vec_zero(col_type_cur, Descr->nb_col_types + 1);
		col_type_this_or_bigger = NEW_int(Descr->nb_col_types + 1);
	}


	label = Descr->problem_label
			+ "_" + Descr->mask_label
			+ "_" + Descr->group_label;



	if (Descr->q1 == 1) {
		Xsize = Descr->d1;
		Ysize = Descr->d2;
	}
	else {
		Xsize = Descr->q1; // = D = q1 = # of rows
		Ysize = Descr->q2; // = C = q2 = # of cols
	}

	V = Xsize * Ysize;


	if (!Descr->f_pair_search_control) {
		cout << "please provide option -pair_search_control" << endl;
		exit(1);
	}
	else {
		Pair_search_control = Get_poset_classification_control(
				Descr->pair_search_control_label);

	}
	if (!Descr->f_search_control) {
		cout << "please provide option -search_control" << endl;
		exit(1);
	}
	else {
		Search_control = Get_poset_classification_control(
				Descr->search_control_label);
	}


	//cout << "depth=" << depth << endl;
	if (f_v) {
		cout << "delandtsheer_doyen::init" << endl;
		cout << "V=" << V << endl;
		cout << "K=" << Descr->K << endl;
		cout << "Xsize=" << Xsize << endl;
		cout << "Ysize=" << Ysize << endl;
		cout << "V=" << V << endl;
	}

	line = NEW_lint(Descr->K);
	row_sum = NEW_int(Xsize);
	col_sum = NEW_int(Ysize);
	live_points = NEW_lint(V);


	if (f_v) {
		cout << "delandtsheer_doyen::init" << endl;
		cout << "DELANDTSHEER_DOYEN_X=" << Descr->DELANDTSHEER_DOYEN_X << endl;
		cout << "DELANDTSHEER_DOYEN_Y=" << Descr->DELANDTSHEER_DOYEN_Y << endl;
	}

	Int_vec_zero(row_sum, Xsize);
	Int_vec_zero(col_sum, Ysize);


	M1 = NEW_OBJECT(algebra::matrix_group);
	M2 = NEW_OBJECT(algebra::matrix_group);

	F1 = NEW_OBJECT(field_theory::finite_field);
	F2 = NEW_OBJECT(field_theory::finite_field);




	if (f_v) {
		cout << "delandtsheer_doyen::init before create_action" << endl;
	}
	create_action(verbose_level);
	if (f_v) {
		cout << "delandtsheer_doyen::init after create_action" << endl;
	}


	A0 = A->subaction;

	P = A0->G.direct_product_group;



	if (Descr->q1 == 1) {

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before create_monomial_group" << endl;
		}
		create_monomial_group(verbose_level);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after create_monomial_group" << endl;
		}

	}

	else {
		if (!A0->f_has_strong_generators) {
			cout << "delandtsheer_doyen::init "
					"action A0 does not "
					"have strong generators" << endl;
			exit(1);
		}

		SG = A0->Strong_gens;
		SG->group_order(go);

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"The group " << A->label << " has order " << go
				<< " and permutation degree " << A->degree << endl;
		}
	}



	if (f_v) {
		show_generators(verbose_level);
	}


	groups::strong_generators *Strong_gens;

	if (Descr->f_subgroup) {


		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before scan_subgroup_generators" << endl;
		}
		Strong_gens = scan_subgroup_generators(verbose_level);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after scan_subgroup_generators" << endl;
		}


		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before compute_orbits_on_pairs" << endl;
		}
		compute_orbits_on_pairs(Strong_gens, verbose_level);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after compute_orbits_on_pairs" << endl;
		}


	}
	else {
		cout << "We don't have -subgroup, "
				"so orbits on pairs "
				"are not computed" << endl;
		//exit(1);
	}


	if (Descr->f_search_wrt_subgroup) {
		SG = Strong_gens;
		cout << "searching wrt subgroup" << endl;
	}



	f_row_used = NEW_int(Xsize);
	f_col_used = NEW_int(Ysize);
	row_idx = NEW_int(Xsize);
	col_idx = NEW_int(Ysize);
	singletons = NEW_int(Descr->K);

	// temporary data
	row_col_idx = NEW_int(Xsize);
	col_row_idx = NEW_int(Ysize);


	if (Descr->f_singletons) {

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before search_singletons" << endl;
		}
		search_singletons(verbose_level);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after search_singletons" << endl;
		}


	}
	if (Descr->f_search_partial_base_lines) {

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before search_partial_base_lines" << endl;
		}
		search_partial_base_lines(verbose_level);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after search_partial_base_lines" << endl;
		}


	}


	if (f_v) {
		cout << "delandtsheer_doyen::init done" << endl;
	}
}



void delandtsheer_doyen::show_generators(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen::show_generators" << endl;
	}

#if 0
	int i, j, a;

	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->Group_element->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		A->Group_element->element_print_as_permutation_with_offset(
				SG->gens->ith(i), cout,
				0 /* offset*/,
				true /* f_do_it_anyway_even_for_big_degree*/,
				true /* f_print_cycles_of_length_one*/,
				0 /* verbose_level*/);
		//A->element_print_as_permutation(SG->gens->ith(i), cout);
		cout << endl;
	}
	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		A->Group_element->element_print_as_permutation(SG->gens->ith(i), cout);
		cout << endl;
	}
	cout << "Generators in GAP format are:" << endl;
	cout << "G := Group([";
	for (i = 0; i < SG->gens->len; i++) {
		A->Group_element->element_print_as_permutation_with_offset(
				SG->gens->ith(i), cout,
				1 /*offset*/,
				true /* f_do_it_anyway_even_for_big_degree */,
				false /* f_print_cycles_of_length_one */,
				0 /* verbose_level*/);
		if (i < SG->gens->len - 1) {
			cout << ", " << endl;
		}
	}
	cout << "]);" << endl;
	cout << "Generators in compact permutation form are:" << endl;
	cout << SG->gens->len << " " << A->degree << endl;
	for (i = 0; i < SG->gens->len; i++) {
		for (j = 0; j < A->degree; j++) {
			a = A->Group_element->element_image_of(j,
					SG->gens->ith(i), 0 /* verbose_level */);
			cout << a << " ";
		}
		cout << endl;
	}
	cout << "-1" << endl;
#endif

	SG->print_generators(cout);
	SG->print_generators_gap(cout);
	SG->print_generators_compact(cout);


	if (f_v) {
		cout << "delandtsheer_doyen::show_generators done" << endl;
	}
}

void delandtsheer_doyen::search_singletons(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen::search_singletons" << endl;
	}

	int target_depth;
	target_depth = Descr->K - Descr->singletons_starter_size;
	cout << "target_depth=" << target_depth << endl;

	orbiter_kernel_system::orbiter_data_file *ODF;
	string fname;
	int level = Descr->singletons_starter_size;

	fname = Search_control->problem_label + "_lvl_" + std::to_string(level);

	if (f_v) {
		cout << "delandtsheer_doyen::search_singletons fname = " << fname << endl;
	}

	ODF = NEW_OBJECT(orbiter_kernel_system::orbiter_data_file);

	if (f_v) {
		cout << "delandtsheer_doyen::search_singletons "
				"before ODF->load" << endl;
	}
	ODF->load(fname, verbose_level);
	if (f_v) {
		cout << "delandtsheer_doyen::search_singletons "
				"after ODF->load" << endl;
	}

	if (f_v) {
		cout << "found " << ODF->nb_cases << " orbits at level " << level << endl;
	}

	int *Nb_sol;
	int *Orbit_idx;
	int nb_orbits_not_ruled_out;
	int orbit_idx;
	int nb_cases = 0;
	int nb_cases_eliminated = 0;
	int f_vv;

	Orbit_idx = NEW_int(ODF->nb_cases);
	Nb_sol = NEW_int(ODF->nb_cases);
	nb_orbits_not_ruled_out = 0;

	for (orbit_idx = 0; orbit_idx < ODF->nb_cases; orbit_idx++) {

	#if 0
		if (f_split) {
			if ((orbit_idx % split_m) == split_r) {
				continue;
			}
		}
	#endif

		if ((orbit_idx % 100)== 0) {
			f_vv = true;
		}
		else {
			f_vv = false;
		}
		if (f_vv) {
			cout << orbit_idx << " / " << ODF->nb_cases << " : ";
			Lint_vec_print(cout, ODF->sets[orbit_idx],
					ODF->set_sizes[orbit_idx]);
			cout << " : " << ODF->Ago_ascii[orbit_idx] << " : "
					<< ODF->Aut_ascii[orbit_idx] << endl;
		}

		long int *line0;

		line0 = ODF->sets[orbit_idx];
		if (ODF->set_sizes[orbit_idx] != level) {
			cout << "ODF->set_sizes[orbit_idx] != level" << endl;
			exit(1);
		}

		compute_live_points(line0, level, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "case " << orbit_idx << " / " << ODF->nb_cases
					<< " we found " << nb_live_points << " live points" << endl;
		}
		if (nb_live_points < target_depth) {
			if (f_vv) {
				cout << "eliminated!" << endl;
			}
			Nb_sol[orbit_idx] = 0;
			nb_cases_eliminated++;
		}
		else {
			Orbit_idx[nb_orbits_not_ruled_out++] = orbit_idx;
			nb_cases++;
		}
		if (f_vv) {
			cout << "nb_cases=" << nb_cases << " vs ";
			cout << "nb_cases_eliminated=" << nb_cases_eliminated << endl;
		}
	} // orbit_idx
	cout << "nb_cases=" << nb_cases << endl;
	cout << "nb_cases_eliminated=" << nb_cases_eliminated << endl;

	int orbit_not_ruled_out;
	int nb_sol_total = 0;

	for (orbit_not_ruled_out = 0;
			orbit_not_ruled_out < nb_orbits_not_ruled_out;
			orbit_not_ruled_out++) {
		orbit_idx = Orbit_idx[orbit_not_ruled_out];


		if ((orbit_not_ruled_out % 100)== 0) {
			f_vv = true;
		}
		else {
			f_vv = false;
		}


		if (f_vv) {
			cout << "orbit_not_ruled_out=" << orbit_not_ruled_out
					<< " / " << nb_orbits_not_ruled_out
					<< " is orbit_idx " << orbit_idx << endl;
		}

		long int *line0;

		line0 = ODF->sets[orbit_idx];
		if (ODF->set_sizes[orbit_idx] != level) {
			cout << "ODF->set_sizes[orbit_idx] != level" << endl;
			exit(1);
		}

		compute_live_points(
				line0, level, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "orbit_not_ruled_out=" << orbit_not_ruled_out << " / "
					<< nb_orbits_not_ruled_out << " is orbit_idx"
					<< orbit_idx << " / " << ODF->nb_cases
					<< " we found " << nb_live_points
					<< " live points" << endl;
		}
		if (nb_live_points == target_depth) {
			Lint_vec_copy(line0, line, level);
			Lint_vec_copy(live_points, line + level, target_depth);
			if (check_orbit_covering(line, Descr->K, 0 /* verbose_level */)) {
				if (f_vv) {
					cout << "found a solution in orbit " << orbit_idx << endl;
				}
				nb_sol_total++;
				Nb_sol[orbit_idx] = 1;
			}



		}
		else {
			if (f_vv) {
				cout << "orbit_not_ruled_out=" << orbit_not_ruled_out << " / "
						<< nb_orbits_not_ruled_out << " is orbit_idx"
						<< orbit_idx << " / " << ODF->nb_cases
						<< " we found " << nb_live_points
						<< " live points, doing a search; " << endl;
			}

#if 0
			int *subset;
			int nCk, l;
			combinatorics::combinatorics_domain Combi;

			subset = NEW_int(target_depth);
			nCk = Combi.int_n_choose_k(
					nb_live_points, target_depth);

			if (f_vv) {
				cout << "nb_live_points = " << nb_live_points
						<< " target_depth = " << target_depth << " nCk = " << nCk << endl;
			}
			for (l = 0; l < nCk; l++) {

				Combi.unrank_k_subset(
						l, subset, nb_live_points, target_depth);

				Lint_vec_copy(
						line0, line, level);

				Int_vec_apply_lint(
						subset, live_points, line + level, target_depth);

				if (check_orbit_covering(
						line, Descr->K, 0 /* verbose_level */)) {
					cout << "found a solution, subset " << l
							<< " / " << nCk << " in orbit "
							<< orbit_idx << endl;
					nb_sol++;
				}
			} // next l

			FREE_int(subset);
#else
			long int nb_sol;

			nb_sol = 0;
			search_case_singletons(
					ODF,
					orbit_idx, nb_sol, verbose_level);

			Nb_sol[orbit_idx] = nb_sol;
			nb_sol_total += nb_sol;
#endif
		} // else
	} // next orbit_not_ruled_out

	cout << "nb_sol_total=" << nb_sol_total << endl;
	cout << "searching singletons done" << endl;



	if (f_v) {
		cout << "delandtsheer_doyen::search_singletons done" << endl;
	}
}

void delandtsheer_doyen::search_case_singletons(
		orbiter_kernel_system::orbiter_data_file *ODF,
		int orbit_idx, long int &nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "delandtsheer_doyen::search_case_singletons " << orbit_idx << " / " << ODF->nb_cases << endl;
	}
	int target_depth;
	target_depth = Descr->K - Descr->singletons_starter_size;
	if (f_v) {
		cout << "target_depth=" << target_depth << endl;
		cout << "nb_live_points=" << nb_live_points << endl;
	}

	data_structures::set_of_sets *Live_points;

	Live_points = NEW_OBJECT(data_structures::set_of_sets);

	Live_points->init_basic_constant_size(
			nb_live_points,
			target_depth + 1, nb_live_points,
			0 /*verbose_level*/);

	Lint_vec_copy(live_points, Live_points->Sets[0], nb_live_points);
	Live_points->Set_size[0] = nb_live_points;



	long int *chosen_set;
	int *index;
	long int nb_nodes = 0;


	chosen_set = NEW_lint(target_depth);
	index = NEW_int(target_depth);


	nb_sol = 0;

	search_case_singletons_recursion(
			ODF, orbit_idx, target_depth, 0 /* level */,
			Live_points,
			chosen_set, index, nb_nodes, nb_sol,
			verbose_level);

	FREE_lint(chosen_set);
	FREE_int(index);
	FREE_OBJECT(Live_points);
	if (f_v) {
		cout << "delandtsheer_doyen::search_case_singletons done" << endl;
	}
}

void delandtsheer_doyen::search_case_singletons_recursion(
		orbiter_kernel_system::orbiter_data_file *ODF,
		int orbit_idx, int target_depth, int level,
		data_structures::set_of_sets *Live_points,
		long int *chosen_set, int *index, long int &nb_nodes, long int &nb_sol,
		int verbose_level)
{
	if (level == target_depth) {
		cout << "solution: ";
		Lint_vec_print(cout, chosen_set, target_depth);
		cout << endl;
		nb_sol++;
		return;
	}

	nb_nodes++;

	if ((nb_nodes % (1 << 20)) == 0) {
		cout << "level " << level << " nb_live = " << Live_points->Set_size[level] << " : ";
		cout << "case " << index[0] << " / " << Live_points->Set_size[0] << endl;
	}

	//cout << "level " << level << " nb_live = " << Live_points->Set_size[level] << endl;

	for (index[level] = 0; index[level] < Live_points->Set_size[level]; index[level]++) {

		//int i, o;
		long int c;

		c = Live_points->Sets[level][index[level]];


		increase_orbit_covering_firm(ODF->sets[orbit_idx], Descr->singletons_starter_size, c);
		increase_orbit_covering_firm(chosen_set, level, c);

#if 0
		for (i = 0; i < Descr->singletons_starter_size; i++) {
			a = ODF->sets[orbit_idx][i];
			o = find_pair_orbit(
					a, c, 0 /*verbose_level - 1*/);
			orbit_covered[o]++;
			if (orbit_covered[o] > orbit_covered_max[o]) {
				cout << "level " << level << " live point is not alive" << endl;
				exit(1);
			}
		}
		for (i = 0; i < level; i++) {
			a = chosen_set[i];
			o = find_pair_orbit(
					a, c, 0 /*verbose_level - 1*/);
			orbit_covered[o]++;
			if (orbit_covered[o] > orbit_covered_max[o]) {
				cout << "level " << level << " live point is not alive" << endl;
				exit(1);
			}
		}
#endif


		chosen_set[level] = c;

		// compute live points for the next level:

		Live_points->Set_size[level + 1] = 0;

		int h, flag;
		long int d;

		for (h = index[level] + 1; h < Live_points->Set_size[level]; h++) {

			Int_vec_copy(orbit_covered, orbit_covered2, nb_orbits);

			d = Live_points->Sets[level][h];

			flag = try_to_increase_orbit_covering_based_on_two_sets(
					ODF->sets[orbit_idx], Descr->singletons_starter_size,
					chosen_set, level + 1,
					d);
			if (flag) {
				Live_points->Sets[level + 1][Live_points->Set_size[level + 1]++] = d;
			}
#if 0
			for (i = 0; i < Descr->singletons_starter_size; i++) {
				a = ODF->sets[orbit_idx][i];
				o = find_pair_orbit(
						a, d, 0 /*verbose_level - 1*/);
				orbit_covered2[o]++;
				if (orbit_covered2[o] > orbit_covered_max[o]) {
					break;
				}
			}
			if (i < Descr->singletons_starter_size) {
				continue;
			}

			for (i = 0; i < level + 1; i++) {
				a = chosen_set[i];
				o = find_pair_orbit(
						a, d, 0 /*verbose_level - 1*/);
				orbit_covered2[o]++;
				if (orbit_covered2[o] > orbit_covered_max[o]) {
					break;
				}
			}
			if (i < level + 1) {
				continue;
			}
			Live_points->Sets[level + 1][Live_points->Set_size[level + 1]++] = d;
#endif

		}

		search_case_singletons_recursion(
				ODF, orbit_idx, target_depth, level + 1,
				Live_points, chosen_set, index, nb_nodes, nb_sol,
				verbose_level);

		c = chosen_set[level]; // redundant

		decrease_orbit_covering(ODF->sets[orbit_idx], Descr->singletons_starter_size, c);
		decrease_orbit_covering(chosen_set, level, c);

#if 0
		for (i = 0; i < Descr->singletons_starter_size; i++) {
			a = ODF->sets[orbit_idx][i];
			o = find_pair_orbit(
					a, c, 0 /*verbose_level - 1*/);
			orbit_covered[o]--;
			if (orbit_covered[o] < 0) {
				cout << "orbit_covered[o] < 0" << endl;
				exit(1);
			}
		}
		for (i = 0; i < level; i++) {
			a = chosen_set[i];
			o = find_pair_orbit(
					a, c, 0 /*verbose_level - 1*/);
			orbit_covered[o]--;
			if (orbit_covered[o] < 0) {
				cout << "orbit_covered[o] < 0" << endl;
				exit(1);
			}
		}
#endif

	}
}

int delandtsheer_doyen::try_to_increase_orbit_covering_based_on_two_sets(
		long int *pts1, int sz1, long int *pts2, int sz2, long int pt0)
{
	int i, o;
	long int a;

	Int_vec_copy(orbit_covered, orbit_covered2, nb_orbits);

	for (i = 0; i < sz1; i++) {
		a = pts1[i];
		o = find_pair_orbit(
				a, pt0, 0 /*verbose_level - 1*/);
		orbit_covered2[o]++;
		if (orbit_covered2[o] > orbit_covered_max[o]) {
			return false;
		}
	}

	for (i = 0; i < sz2; i++) {
		a = pts2[i];
		o = find_pair_orbit(
				a, pt0, 0 /*verbose_level - 1*/);
		orbit_covered2[o]++;
		if (orbit_covered2[o] > orbit_covered_max[o]) {
			return false;
		}
	}
	return true;

}
void delandtsheer_doyen::increase_orbit_covering_firm(
		long int *pts, int sz, long int pt0)
// firm means that an excess in the orbit covering raises an error
{
	int i, o;
	long int a;

	for (i = 0; i < sz; i++) {
		a = pts[i];
		o = find_pair_orbit(
				a, pt0, 0 /*verbose_level - 1*/);
		orbit_covered[o]++;
		if (orbit_covered[o] > orbit_covered_max[o]) {
			cout << "delandtsheer_doyen::increase_orbit_covering_firm: live point is not alive" << endl;
			exit(1);
		}
	}

}

void delandtsheer_doyen::decrease_orbit_covering(
		long int *pts, int sz, long int pt0)
{
	int i, o;
	long int a;

	for (i = 0; i < sz; i++) {
		a = pts[i];
		o = find_pair_orbit(
				a, pt0, 0 /*verbose_level - 1*/);
		orbit_covered[o]--;
		if (orbit_covered[o] < 0) {
			cout << "delandtsheer_doyen::decrease_orbit_covering: orbit_covered[o] < 0" << endl;
			exit(1);
		}
	}

}

void delandtsheer_doyen::search_partial_base_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines" << endl;
	}
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (!Search_control->f_depth) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"!Search_control->f_depth" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"depth = " << Search_control->depth << endl;
	}

	Gen = NEW_OBJECT(poset_classification::poset_classification);


	if (!Descr->f_problem_label) {
		cout << "please use -problem_label <string : problem_label>" << endl;
		exit(1);
	}

	Poset_search = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset_search->init_subset_lattice(
			A0, A, SG,
			verbose_level);

	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset_search->add_testing_without_group(
			delandtsheer_doyen_early_test_func_callback,
				this /* void *data */,
				verbose_level);


	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"before Gen->init" << endl;
	}
	Gen->initialize_and_allocate_root_node(
			Search_control, Poset_search,
			Search_control->depth /* sz */, verbose_level);
	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"after Gen->init" << endl;
	}


	int f_use_invariant_subset_if_available = true;
	int f_debug = false;

	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"before Gen->main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
	}


	//Control->f_max_depth = false;
	//Gen->depth = Descr->depth;
	Gen->main(
			t0,
			Search_control->depth /* schreier_depth */,
			f_use_invariant_subset_if_available,
			f_debug,
			verbose_level - 2);

	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"after Gen->main" << endl;
	}

	int nb_s_orbits;
	int sz;
	int h;
	int s; // starter size
	int *Covered_orbits;
	int *Nb_live_points;
	int s2;
	int nb_cols;
	string *Table;

	s = Search_control->depth;
	s2 = s * (s - 1) >> 1;
	nb_s_orbits = Gen->nb_orbits_at_level(s);
	if (f_v) {
		cout << "target level = s = " << s << endl;
		cout << "s2 = {s choose 2} = " << s2 << endl;
		cout << "number of s-orbits: " << nb_s_orbits << endl;
	}

	nb_cols = 6;
	Table = new string[nb_s_orbits * nb_cols];

	Covered_orbits = NEW_int(nb_s_orbits * s2);
	Nb_live_points = NEW_int(nb_s_orbits);

	for (h = 0; h < nb_s_orbits; h++) {


		if (f_v) {
			cout << "delandtsheer_doyen::search_partial_base_lines "
					"case " << h << " / " << nb_s_orbits << endl;
		}

		Gen->get_set(s, h, line, sz);

		Table[h * nb_cols + 0] = std::to_string(h);
		Table[h * nb_cols + 1] = "\"" + Lint_vec_stringify(line, s) + "\"";

		if (false) {
			cout << h << " : ";
			Lint_vec_print(cout, line, sz);
		}


		// compute orbits covered:
		{
			int i, pi, j, pj, o, cnt;


			cnt = 0;
			for (i = 0; i < s; i++) {
				pi = line[i];
				for (j = i + 1; j < s; j++, cnt++) {
					pj = line[j];
					o = find_pair_orbit(pi, pj, 0 /*verbose_level - 1*/);
					if (pi == pj) {
						cout << "delandtsheer_doyen::check_orbit_covering "
								"pi = " << pi << " == pj = " << pj << endl;
						exit(1);
					}
					Covered_orbits[h * s2 + cnt] = o;
				}
			}

		}

		std::string fname;

		create_graph(h, line, s, s2, Covered_orbits + h * s2,
				nb_live_points,
				fname,
				verbose_level - 2);

		Nb_live_points[h] = nb_live_points;

		Table[h * nb_cols + 2] = "\"" + Int_vec_stringify(Covered_orbits + h * s2, s2) + "\"";
		Table[h * nb_cols + 3] = std::to_string(nb_live_points);
		Table[h * nb_cols + 4] = fname;
		Table[h * nb_cols + 5] = "\"" + Lint_vec_stringify(live_points, nb_live_points) + "\"";

		if (false) {
			cout << " : ";
			Int_vec_print(cout, Covered_orbits + h * s2, s2);
			cout << endl;
		}

	}

	orbiter_kernel_system::file_io Fio;
	string fname;

#if 0
	fname = label + "_pair_covering.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Covered_orbits, nb_s_orbits, s2);

	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;
#endif

	FREE_int(Covered_orbits);
	FREE_int(Nb_live_points);

	fname = label + "_cases.csv";

	std::string headings;

	headings = "Case,Starter,Covering,NbLive,fname,Live";
	Fio.Csv_file_support->write_table_of_strings(
				fname,
				nb_s_orbits, nb_cols, Table,
				headings,
				verbose_level - 2);

	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;


	delete [] Table;
#if 0

	if (f_v) {
		cout << "delandtsheer_doyen::search_partial_base_lines "
				"before Gen->draw_poset" << endl;
	}
	Gen->draw_poset(Gen->get_problem_label_with_path(), Descr->depth,
			0 /* data1 */, true /* f_embedded */, true /* f_sideways */, 100 /* rad */, 0.45 /* scale */,
			verbose_level);
#endif



	if (f_v) {
		cout << "delandtsheer_doyen::search_starter done" << endl;
	}

}

void delandtsheer_doyen::create_graph(
		int case_number, long int *line, int s, int s2, int *Covered_orbits,
		int &nb_live_points,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen::create_graph" << endl;
	}
	int i, j, a, x, y;

	Int_vec_zero(row_sum, Xsize);
	Int_vec_zero(col_sum, Ysize);

	for (i = 0; i < s; i++) {
		a = line[i];
		x = a / Ysize;
		y = a % Ysize;
		//cout << "i=" << i << " / " << len << " a=" << a
		//	<< " x=" << x << " y=" << y << endl;
		row_sum[x]++;
		col_sum[y]++;
	}

	if (!check_orbit_covering(
			line,
			s, 0 /* verbose_level */)) {
		cout << "delandtsheer_doyen::create_graph "
				"line is not good (check_orbit_covering)" << endl;
		//check_orbit_covering(line0, len, 2 /* verbose_level */);
		exit(1);
	}

	// now: orbit_covered[nb_orbits] has been computed

	long int ph;
	int o, h;
	int xi, yi, xj, yj;

	nb_live_points = 0;
	for (x = 0; x < Xsize; x++) {
		if (row_sum[x]) {
			continue;
		}
		for (y = 0; y < Ysize; y++) {
			if (col_sum[y]) {
				continue;
			}
			a = x * Ysize + y;

			Int_vec_copy(orbit_covered, orbit_covered2, nb_orbits);

			//cout << "testing point a=" << a << endl;
			for (h = 0; h < s; h++) {

				ph = line[h];
				o = find_pair_orbit(
						ph, a, 0 /*verbose_level - 1*/);
				orbit_covered2[o]++;
				if (orbit_covered2[o] > orbit_covered_max[o]) {
					break;
				}
			} // next h
			if (h == s) {
				live_points[nb_live_points++] = a;
			}
		} // next y
	} // next x
	if (f_v) {
		cout << "delandtsheer_doyen::create_graph "
				"found " << nb_live_points << " live points" << endl;
	}

	int *Adj;
	int pi, pj, adj;

	Adj = NEW_int(nb_live_points * nb_live_points);
	Int_vec_zero(Adj, nb_live_points * nb_live_points);

	for (i = 0; i < nb_live_points; i++) {
		pi = live_points[i];
		xi = pi / Ysize;
		yi = pi % Ysize;

		for (j = i + 1; j < nb_live_points; j++) {
			pj = live_points[j];
			xj = pj / Ysize;
			yj = pj % Ysize;

			Int_vec_copy(orbit_covered, orbit_covered2, nb_orbits);

			adj = 0;
			if (xi == xj || yi == yj) {
				adj = 0;
			}
			else {
				//cout << "testing point pi=" << pi << endl;
				for (h = 0; h < s; h++) {

					ph = line[h];
					o = find_pair_orbit(
							ph, pi, 0 /*verbose_level - 1*/);
					orbit_covered2[o]++;
				} // next h

				//cout << "testing point pj=" << pj << endl;
				for (h = 0; h < s; h++) {

					ph = line[h];
					o = find_pair_orbit(
							ph, pj, 0 /*verbose_level - 1*/);
					orbit_covered2[o]++;
					if (orbit_covered2[o] > orbit_covered_max[o]) {
						break;
					}
				} // next h

				if (h < s) {
					adj = 0;
				}
				else {
					o = find_pair_orbit(
							pi, pj, 0 /*verbose_level - 1*/);
					orbit_covered2[o]++;
					if (orbit_covered2[o] > orbit_covered_max[o]) {
						adj = 0;
					}
					else {
						adj = 1;
					}
				}
			}

			if (adj) {
				Adj[i * nb_live_points + j] = 1;
				Adj[j * nb_live_points + i] = 1;
			}


		}
	}

	graph_theory::colored_graph *CG;

	CG = NEW_OBJECT(graph_theory::colored_graph);
	if (f_v) {
		cout << "delandtsheer_doyen::create_graph "
				"before CG->init_adjacency_no_colors" << endl;
	}
	CG->init_adjacency_no_colors(nb_live_points, Adj, label, label,
			verbose_level);
	if (f_v) {
		cout << "delandtsheer_doyen::create_graph "
				"after CG->init_adjacency_no_colors" << endl;
	}

	fname = label + "_case_" + std::to_string(case_number) + ".graph";
	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);

	FREE_int(Adj);

	if (f_v) {
		cout << "delandtsheer_doyen::create_graph done" << endl;
	}
}


void delandtsheer_doyen::compute_orbits_on_pairs(
		groups::strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "delandtsheer_doyen::compute_orbits_on_pairs" << endl;
	}
	Pairs = NEW_OBJECT(poset_classification::poset_classification);


	Pair_search_control->f_depth = true;
	Pair_search_control->depth = 2;

	Poset_pairs = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset_pairs->init_subset_lattice(
			A0, A, Strong_gens,
			verbose_level);


	if (f_v) {
		cout << "delandtsheer_doyen::compute_orbits_on_pairs "
				"before Pairs->init" << endl;
	}
	Pairs->initialize_and_allocate_root_node(
			Pair_search_control, Poset_pairs,
			2 /* sz */, verbose_level);
	if (f_v) {
		cout << "direct_product_action::compute_orbits_on_pairs "
				"after Pairs->init" << endl;
	}



	int f_use_invariant_subset_if_available;
	int f_debug;

	f_use_invariant_subset_if_available = true;
	f_debug = false;


	if (f_v) {
		cout << "delandtsheer_doyen::compute_orbits_on_pairs "
				"before Pairs->main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
	}


	//Pairs->f_allowed_to_show_group_elements = true;

	Pair_search_control->f_depth = true;
	Pair_search_control->depth = 2;

	//Pairs->depth = 2;
	Pairs->main(t0,
		2 /* schreier_depth */,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 2);

	if (f_v) {
		cout << "delandtsheer_doyen::compute_orbits_on_pairs "
				"after Pairs->main" << endl;
	}


	nb_orbits = Pairs->nb_orbits_at_level(2);

	if (f_v) {
		cout << "delandtsheer_doyen::compute_orbits_on_pairs "
				"nb_orbits = "
				<< nb_orbits << endl;
	}

	transporter = NEW_int(A0->elt_size_in_int);
	tmp_Elt = NEW_int(A0->elt_size_in_int);

	orbit_length = NEW_int(nb_orbits);
	orbit_covered = NEW_int(nb_orbits);
	orbit_covered2 = NEW_int(nb_orbits);
	orbit_covered_max = NEW_int(nb_orbits);
	orbits_covered = NEW_int(Descr->K * Descr->K);

	Int_vec_zero(orbit_covered, nb_orbits);



	for (i = 0; i < nb_orbits; i++) {
		orbit_length[i] = Pairs->orbit_length_as_int(
				i /* orbit_at_level*/, 2 /* level*/);
		orbit_covered_max[i] = (orbit_length[i] * Descr->nb_orbits_on_blocks) / b;
		if (orbit_covered_max[i] * b != orbit_length[i] * Descr->nb_orbits_on_blocks) {
			cout << "integrality conditions violated (2)" << endl;
			cout << "Descr->nb_orbits_on_blocks = " << Descr->nb_orbits_on_blocks << endl;
			cout << "pair orbit i=" << i << " / " << nb_orbits << endl;
			cout << "orbit_length[i]=" << orbit_length[i] << endl;
			cout << "b=" << b << endl;
			exit(1);
		}
	}
	cout << "i : orbit_length[i] : orbit_covered_max[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		cout << i << " : " << orbit_length[i]
			<< " : " << orbit_covered_max[i] << endl;
	}

	compute_pair_orbit_table(verbose_level);
	//write_pair_orbit_file(verbose_level);
	if (f_v) {
		cout << "delandtsheer_doyen::compute_orbits_on_pairs done" << endl;
	}
}

groups::strong_generators *delandtsheer_doyen::scan_subgroup_generators(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	groups::strong_generators *Strong_gens;

	#if 0

	if (f_v) {
		cout << "delandtsheer_doyen::scan_subgroup_generators" << endl;
	}
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	int *data;
	int sz;
	int nb_gens;
	data_structures_groups::vector_ge *nice_gens;

	Int_vec_scan(Descr->subgroup_gens, data, sz);
	nb_gens = sz / A->make_element_size;
	if (f_v) {
		cout << "before Strong_gens->init_from_data_with_target_go_ascii" << endl;
	}
	cout << "nb_gens=" << nb_gens << endl;
	Strong_gens->init_from_data_with_target_go_ascii(
			A0,
			data,
			nb_gens, A0->make_element_size,
			Descr->subgroup_order,
			nice_gens,
			verbose_level + 2);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "delandtsheer_doyen "
				"after Strong_gens->init_from_data_with_target_go_ascii" << endl;
	}
#endif

	actions::action_global AG;

	Strong_gens = AG.scan_generators(
			A0,
			Descr->subgroup_gens,
			Descr->subgroup_order,
			verbose_level);
	if (f_v) {
		cout << "delandtsheer_doyen::scan_subgroup_generators done" << endl;
	}
	return Strong_gens;
}

void delandtsheer_doyen::create_monomial_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "delandtsheer_doyen::create_monomial_group" << endl;
	}
	groups::strong_generators *SG1;
	groups::strong_generators *SG2;
	groups::strong_generators *SG3;

	SG1 = NEW_OBJECT(groups::strong_generators);
	SG2 = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "before generators_for_the_monomial_group "
				"action" << A1->label << endl;
	}
	SG1->generators_for_the_monomial_group(
			A1,
		M1, verbose_level);
	if (f_v) {
		cout << "after generators_for_the_monomial_group "
				"action" << A1->label << endl;
	}


	if (f_v) {
		cout << "before generators_for_the_monomial_group "
				"action" << A2->label << endl;
	}
	SG2->generators_for_the_monomial_group(
			A2,
		M2, verbose_level);
	if (f_v) {
		cout << "after generators_for_the_monomial_group "
				"action" << A2->label << endl;
	}

	if (f_v) {
		cout << "direct_product_action::init "
				"before lift_generators" << endl;
	}
	P->lift_generators(
			SG1,
			SG2,
			A0, SG3,
			verbose_level);
	if (f_v) {
		cout << "direct_product_action::init "
				"after lift_generators" << endl;
	}

	SG = SG3;
	SG->group_order(go);

	cout << "The group has order " << go << endl;

	actions::action *Ar;
	long int *points;
	int nb_points;
	int h;

	nb_points = Descr->d1 * Descr->d2;
	points = NEW_lint(nb_points);
	h = 0;
	for (i = 0; i < Descr->d1; i++) {
		for (j = 0; j < Descr->d2; j++) {
			a = i * A2->degree + j;
			points[h++] = a;
		}
	} // next i


	std::string label_of_set;


	label_of_set.assign("points");

	Ar = A->Induced_action->restricted_action(
			points, nb_points, label_of_set,
			verbose_level);

	A = Ar;
	if (f_v) {
		cout << "delandtsheer_doyen::create_monomial_group done" << endl;
	}
}


void delandtsheer_doyen::create_action(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen::create_action" << endl;
	}
	A1 = NEW_OBJECT(actions::action);
	A2 = NEW_OBJECT(actions::action);

	if (Descr->q1 == 1) {

		data_structures_groups::vector_ge *nice_gens;

		F1->finite_field_init_small_order(2,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		F2->finite_field_init_small_order(2,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);

		if (f_v) {
			cout << "delandtsheer_doyen::create_action "
					"initializing projective groups:" << endl;
		}

		A1->Known_groups->init_projective_group(
				Descr->d1, F1,
				false /* f_semilinear */, true /* f_basis */, true /* f_init_sims */,
				nice_gens,
				verbose_level - 1);
		M1 = A1->G.matrix_grp;
		FREE_OBJECT(nice_gens);

		A2->Known_groups->init_projective_group(
				Descr->d2, F2,
				false /* f_semilinear */, true /* f_basis */, true /* f_init_sims */,
				nice_gens,
				verbose_level - 1);
		M2 = A1->G.matrix_grp;
		FREE_OBJECT(nice_gens);

		b = 0;

	}
	else {



		b = (V * (V - 1)) / (Descr->K * (Descr->K - 1));

		if (b * (Descr->K * (Descr->K - 1)) != (V * (V - 1))) {
			cout << "delandtsheer_doyen::create_action integrality conditions violated" << endl;
			exit(1);
		}

		cout << "b=" << b << endl;



		F1->finite_field_init_small_order(
				Descr->q1,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		F2->finite_field_init_small_order(
				Descr->q2,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);

		actions::action_global AG;


		if (f_v) {
			cout << "delandtsheer_doyen::create_action "
					"initializing affine groups:" << endl;
		}

		M1->init_affine_group(
				Descr->d1, F1,
				false /* f_semilinear */, /*A1,*/ verbose_level);

		if (f_v) {
			cout << "delandtsheer_doyen::create_action "
					"before AG.init_base" << endl;
		}
		AG.init_base(
				A1, M1, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "delandtsheer_doyen::create_action "
					"after AG.init_base" << endl;
		}


		M2->init_affine_group(
				Descr->d2, F2,
				false /* f_semilinear */, /*A2,*/ verbose_level);

		if (f_v) {
			cout << "delandtsheer_doyen::create_action "
					"before AG.init_base" << endl;
		}
		AG.init_base(
				A2, M2, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "delandtsheer_doyen::create_action "
					"after AG.init_base" << endl;
		}

	}

	if (f_v) {
		cout << "delandtsheer_doyen::create_action before "
				"AG.init_direct_product_group_and_restrict" << endl;
	}

	actions::action_global AG;

	A = AG.init_direct_product_group_and_restrict(
			M1, M2,
			verbose_level);

	if (f_v) {
		cout << "delandtsheer_doyen::create_action after "
				"AG.init_direct_product_group_and_restrict" << endl;
	}

	if (!A->f_has_subaction) {
		cout << "delandtsheer_doyen::create_action action "
				"A does not have a subaction" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "delandtsheer_doyen::create_action done" << endl;
	}
}


void delandtsheer_doyen::compute_live_points(
		long int *line0, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, x, y, h, ph, k, pk, o;

	if (f_v) {
		cout << "delandtsheer_doyen::compute_live_points" << endl;
	}

	Int_vec_zero(row_sum, Xsize);
	Int_vec_zero(col_sum, Ysize);

	for (i = 0; i < len; i++) {
		a = line0[i];
		x = a / Ysize;
		y = a % Ysize;
		//cout << "i=" << i << " / " << len << " a=" << a
		//	<< " x=" << x << " y=" << y << endl;
		row_sum[x]++;
		col_sum[y]++;
	}

	if (!check_orbit_covering(
			line0,
		len, 0 /* verbose_level */)) {
		cout << "delandtsheer_doyen::compute_live_points "
				"line0 is not good (check_orbit_covering)" << endl;
		check_orbit_covering(line0, len, 2 /* verbose_level */);
		exit(1);
	}

	nb_live_points = 0;
	for (x = 0; x < Xsize; x++) {
		if (row_sum[x]) {
			continue;
		}
		for (y = 0; y < Ysize; y++) {
			if (col_sum[y]) {
				continue;
			}
			a = x * Ysize + y;
			//cout << "testing point a=" << a << endl;
			for (h = 0; h < len; h++) {

				ph = line0[h];
				o = find_pair_orbit(
						ph, a, 0 /*verbose_level - 1*/);
				orbit_covered[o]++;
				if (orbit_covered[o] > orbit_covered_max[o]) {
					for (k = h; k >= 0; k--) {
						pk = line0[k];
						o = find_pair_orbit(
								pk, a, 0 /*verbose_level - 1*/);
						orbit_covered[o]--;
					}
					break;
				}
			} // next h
			if (h == len) {
				live_points[nb_live_points++] = a;
				for (h = 0; h < len; h++) {

					ph = line0[h];
					o = find_pair_orbit(
							ph, a, 0 /*verbose_level - 1*/);
					orbit_covered[o]--;
				}
			}
		} // next y
	} // next x
	if (f_v) {
		cout << "found " << nb_live_points << " live points" << endl;
	}

	if (f_v) {
		cout << "delandtsheer_doyen::compute_live_points done" << endl;
	}
}


int delandtsheer_doyen::find_pair_orbit(
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_no;

	if (f_v) {
		cout << "delandtsheer_doyen::find_pair_orbit" << endl;
	}
	if (i == j) {
		cout << "delandtsheer_doyen::find_pair_orbit "
				"i = j = " << j << endl;
		exit(1);
	}
	orbit_no = pair_orbit[i * V + j];
	if (f_v) {
		cout << "delandtsheer_doyen::find_pair_orbit done" << endl;
	}
	return orbit_no;
}

int delandtsheer_doyen::find_pair_orbit_by_tracing(
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_no;
	long int set[2];
	long int canonical_set[2];

	if (f_v) {
		cout << "delandtsheer_doyen::find_pair_orbit_by_tracing" << endl;
	}
	if (i == j) {
		cout << "delandtsheer_doyen::find_pair_orbit_by_tracing "
				"i = j = " << j << endl;
		exit(1);
	}
	set[0] = i;
	set[1] = j;
	orbit_no = Pairs->trace_set(
			set, 2, 2,
		canonical_set, transporter,
		verbose_level - 1);
	if (f_v) {
		cout << "delandtsheer_doyen::find_pair_orbit_by_tracing "
				"done" << endl;
	}
	return orbit_no;
}

void delandtsheer_doyen::compute_pair_orbit_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;


	if (f_v) {
		cout << "delandtsheer_doyen::compute_pair_orbit_table" << endl;
	}
	pair_orbit = NEW_int(V * V);
	Int_vec_zero(pair_orbit, V * V);
	for (i = 0; i < V; i++) {
		for (j = i + 1; j < V; j++) {
			k = find_pair_orbit_by_tracing(i, j, 0 /*verbose_level - 2*/);
			pair_orbit[i * V + j] = k;
			pair_orbit[j * V + i] = k;
		}
		if ((i % 100) == 0) {
			cout << "i=" << i << endl;
		}
	}
	if (f_v) {
		cout << "delandtsheer_doyen::compute_pair_orbit_table done" << endl;
	}
}

void delandtsheer_doyen::write_pair_orbit_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	long int set[1000];
	int i, j, k, n, size, l;


	if (f_v) {
		cout << "delandtsheer_doyen::write_pair_orbit_file" << endl;
	}

	fname = Descr->group_label + ".2orbits";

	cout << "writing pair-orbit file " << fname << endl;
	{
		ofstream f(fname);
		f << nb_orbits << endl;
		for (i = 0; i < nb_orbits; i++) {
			n = Pairs->first_node_at_level(2) + i;
			Pairs->get_set(n, set, size);
			if (size != 2) {
				cout << "delandtsheer_doyen::write_pair_orbit_file "
						"size != 2" << endl;
				exit(1);
			}
			l = Pairs->orbit_length_as_int(i, 2);
			f << set[0] << " " << set[1] << " " << l << endl;
		}
		for (i = 0; i < V; i++) {
			for (j = i + 1; j < V; j++) {
				k = find_pair_orbit(i, j, 0 /*verbose_level - 2*/);
				f << k << " ";
			}
			f << endl;
			if ((i % 100) == 0) {
				cout << "i=" << i << endl;
			}
		}
	}
	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "delandtsheer_doyen::write_pair_orbit_file done" << endl;
	}

}

void delandtsheer_doyen::print_mask_test_i(
		std::ostream &ost, int i)
{
	int who, what;

	ost << "mask test at level " << Descr->mask_test_level[i] << " : ";
	who = Descr->mask_test_who[i];
	what = Descr->mask_test_what[i];
	if (who == 1) {
		ost << "x ";
	}
	else if (who == 2) {
		ost << "y ";
	}
	else if (who == 3) {
		ost << "x+y ";
	}
	else if (who == 4) {
		ost << "s ";
	}
	if (what == 1) {
		ost << "= ";
	}
	else if (what == 2) {
		ost << ">= ";
	}
	else if (what == 3) {
		ost << "<= ";
	}
	ost << Descr->mask_test_value[i];
	ost << endl;
}

void delandtsheer_doyen::early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int j;
	int f_OK;

	if (f_v) {
		cout << "delandtsheer_doyen::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}


	if (len == 0) {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "delandtsheer_doyen::early_test_func before testing" << endl;
		}
		for (j = 0; j < nb_candidates; j++) {

			S[len] = candidates[j];

			f_OK = check_conditions(S, len + 1, verbose_level);
			if (f_vv) {
				cout << "delandtsheer_doyen::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
			}

			if (f_OK) {
				good_candidates[nb_good_candidates++] = candidates[j];
			}
		} // next j
	} // else
}

int delandtsheer_doyen::check_conditions(
		long int *S, int len, int verbose_level)
{
	//verbose_level = 4;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_OK = true;
	int f_bad_orbit = false;
	int f_bad_row = false;
	int f_bad_col = false;
	int f_bad_mask = false;
	int pt, idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "delandtsheer_doyen::check_conditions "
				"checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		//cout << "offset=" << offset << endl;
	}

	pt = S[len - 1];
	if (Sorting.lint_vec_search_linear(S, len - 1, pt, idx)) {
		if (f_v) {
			cout << "delandtsheer_doyen::check_conditions "
					"not OK, "
					"repeat entry" << endl;
		}
		return false;
	}
	if (Descr->f_subgroup) {
		if (!check_orbit_covering(S, len, verbose_level)) {
			f_bad_orbit = true;
			f_OK = false;
		}
	}

	if (f_OK && !check_row_sums(S, len, verbose_level)) {
		f_bad_row = true;
		f_OK = false;
	}
	if (f_OK && !check_col_sums(S, len, verbose_level)) {
		f_bad_col = true;
		f_OK = false;
	}
	if (f_OK && !check_mask(S, len, verbose_level)) {
		f_bad_mask = true;
		f_OK = false;
	}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
		}
		return true;
	}
	else {
		if (f_v) {
			cout << "not OK" << endl;
		}
		if (f_vv) {
			cout << "because of ";
			if (f_bad_orbit)
				cout << "orbit covering";
			else if (f_bad_row)
				cout << "row-test";
			else if (f_bad_col)
				cout << "col-test";
			else if (f_bad_mask)
				cout << "mask";
			cout << endl;
		}
		return false;
	}
}

int delandtsheer_doyen::check_orbit_covering(
		long int *line,
		int len, int verbose_level)
// computes orbit_covered[nb_orbits] based on line[len]
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, pi, j, pj, o, f_OK = true;

	Int_vec_zero(orbit_covered, nb_orbits);

	for (i = 0; i < len; i++) {
		pi = line[i];
		for (j = i + 1; j < len; j++) {
			pj = line[j];
			o = find_pair_orbit(pi, pj, 0 /*verbose_level - 1*/);
			if (pi == pj) {
				cout << "delandtsheer_doyen::check_orbit_covering "
						"pi = " << pi << " == pj = " << pj << endl;
				exit(1);
			}
			orbit_covered[o]++;
			if (orbit_covered[o] > orbit_covered_max[o]) {
				f_OK = false;
				break;
			}
		}
		if (!f_OK) {
			break;
		}
	}
	if (f_v) {
		if (!f_OK) {
			cout << "orbit condition violated" << endl;
#if 0
			if (f_vv) {
				print_orbit_covered(cout);
				print_orbit_covered_max(cout);
				get_orbit_covering_matrix(line, len, verbose_level - 1);
				print_orbit_covering_matrix(len);
			}
#endif
		}
	}
	return f_OK;
}

int delandtsheer_doyen::check_row_sums(
		long int *line,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, p, x, s, f_OK = true;
	int f_DD_problem = false;

	inner_pairs_in_rows = 0;
	Int_vec_zero(row_sum, Xsize);
	if (Descr->f_R) {
		for (i = 1; i <= Descr->nb_row_types; i++) {
			row_type_cur[i] = 0;
		}
	}
	for (i = 0; i < len; i++) {
		p = line[i];
		x = p / Ysize;
		//y = p % Ysize;
		inner_pairs_in_rows += row_sum[x];
		row_sum[x]++;
		if (Descr->DELANDTSHEER_DOYEN_X != -1) {
			if (inner_pairs_in_rows > Descr->DELANDTSHEER_DOYEN_X) {
				f_OK = false;
				f_DD_problem = true;
				break;
			}
		}
		if (Descr->f_R) {
			s = row_sum[x];
			if (s > Descr->nb_row_types) {
				f_OK = false;
				break;
			}
			if (row_type_cur[s] >= row_type_this_or_bigger[s]) {
				f_OK = false;
				break;
			}
			if (s > 1) {
				row_type_cur[s - 1]--;
			}
			row_type_cur[s]++;
		}
	}
	if (f_v) {
		if (!f_OK) {
			cout << "delandtsheer_doyen::check_row_sums "
					"row condition violated" << endl;
			if (f_vv) {
				if (f_DD_problem) {
					cout << "delandtsheer_doyen::check_row_sums "
							"inner_pairs_in_rows = "
						<< inner_pairs_in_rows
						<< " > DELANDTSHEER_DOYEN_X = "
						<< Descr->DELANDTSHEER_DOYEN_X
						<< ", not OK" << endl;
				}
				else {
					cout << "delandtsheer_doyen::check_row_sums"
							"problem with row-type:" << endl;
					for (i = 1; i <= Descr->nb_row_types; i++) {
						cout << row_type_cur[i] << " ";
					}
					cout << endl;
					for (i = 1; i <= Descr->nb_row_types; i++) {
						cout << row_type_this_or_bigger[i] << " ";
					}
					cout << endl;
				}
			}
		}
	}
	return f_OK;
}

int delandtsheer_doyen::check_col_sums(
		long int *line,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, p, y, s, f_OK = true;
	int f_DD_problem = false;

	inner_pairs_in_cols = 0;
	Int_vec_zero(col_sum, Ysize);
	if (Descr->f_C) {
		for (i = 1; i <= Descr->nb_col_types; i++) {
			col_type_cur[i] = 0;
		}
	}
	for (i = 0; i < len; i++) {
		p = line[i];
		//x = p / Ysize;
		y = p % Ysize;
		inner_pairs_in_cols += col_sum[y];
		col_sum[y]++;
		if (Descr->DELANDTSHEER_DOYEN_Y != -1) {
			if (inner_pairs_in_cols > Descr->DELANDTSHEER_DOYEN_Y) {
				f_OK = false;
				f_DD_problem = true;
				break;
			}
		}
		if (Descr->f_C) {
			s = col_sum[y];
			if (s > Descr->nb_col_types) {
				f_OK = false;
				break;
			}
			if (col_type_cur[s] >= col_type_this_or_bigger[s]) {
				f_OK = false;
				break;
			}
			if (s > 1) {
				col_type_cur[s - 1]--;
			}
			col_type_cur[s]++;
		}
	}
	if (f_v) {
		if (!f_OK) {
			cout << "delandtsheer_doyen::check_col_sums "
					"col condition violated" << endl;
			if (f_vv) {
				if (f_DD_problem) {
					cout << "delandtsheer_doyen::check_col_sums "
							"inner_pairs_in_cols = "
						<< inner_pairs_in_cols
						<< " > DELANDTSHEER_DOYEN_Y = "
						<< Descr->DELANDTSHEER_DOYEN_Y
						<< ", not OK" << endl;
				}
				else {
					cout << "delandtsheer_doyen::check_col_sums "
							"problem with col-type:" << endl;
					for (i = 1; i <= Descr->nb_col_types; i++) {
						cout << col_type_cur[i] << " ";
					}
					cout << endl;
					for (i = 1; i <= Descr->nb_col_types; i++) {
						cout << col_type_this_or_bigger[i] << " ";
					}
					cout << endl;
				}
			}
		}
	}
	return f_OK;
}

int delandtsheer_doyen::check_mask(
		long int *line,
		int len, int verbose_level)
{
	//verbose_level = 4;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_OK = true;
	int k, who;
	int nb_rows_used, nb_cols_used;
	int nb_singletons;


	if (f_vv) {
		cout << "delandtsheer_doyen::check_mask" << endl;
	}
	get_mask_core_and_singletons(line, len,
			nb_rows_used, nb_cols_used,
			nb_singletons, verbose_level);

	for (k = 0; k < Descr->nb_mask_tests; k++) {
		if (Descr->mask_test_level[k] != len) {
			continue;
		}
		if (Descr->mask_test_who[k] == 1) {
			who = inner_pairs_in_rows;
		}
		else if (Descr->mask_test_who[k] == 2) {
			who = inner_pairs_in_cols;
		}
		else if (Descr->mask_test_who[k] == 3) {
			who = inner_pairs_in_rows + inner_pairs_in_cols;
		}
		else if (Descr->mask_test_who[k] == 4) {
			who = nb_singletons;
		}
		else {
			cout << "delandtsheer_doyen::check_mask: "
					"unknown mask_test_who value "
					<< Descr->mask_test_who[k] << " in test " << k << endl;
			exit(1);
		}
		if (Descr->mask_test_what[k] == 1) {
			// eq
			if (who != Descr->mask_test_value[k]) {
				f_OK = false;
				break;
			}
		}
		else if (Descr->mask_test_what[k] == 2) {
			// ge
			if (who < Descr->mask_test_value[k]) {
				f_OK = false;
				break;
			}
		}
		else if (Descr->mask_test_what[k] == 3) {
			// le
			if (who > Descr->mask_test_value[k]) {
				f_OK = false;
				break;
			}
		}
		else {
			cout << "delandtsheer_doyen::check_mask: "
					"unknown mask_test_what value "
					<< Descr->mask_test_what[k] << " in test " << k << endl;
			exit(1);
		}
	}
	if (f_v) {
		if (f_OK) {
			cout << "mask" << endl;
			//print_mask(cout, Xsize, Ysize, M);
			cout << "is OK" << endl;
		}
		else {
			if (f_vv) {
				cout << "mask test " << k << " failed:" << endl;
				print_mask_test_i(cout, k);
				//cout << "x=" << inner_pairs_in_rows
					//<< "y=" << inner_pairs_in_cols
					//<< "s=" << nb_singletons << endl;
			}
		}
	}

	return f_OK;
}


void delandtsheer_doyen::get_mask_core_and_singletons(
	long int *line, int len,
	int &nb_rows_used, int &nb_cols_used,
	int &nb_singletons, int verbose_level)
{
	int i, j, h, a;
	int m = Xsize;
	int n = Ysize;

	Int_vec_zero(f_row_used, m);
	Int_vec_zero(f_col_used, n);
	for (h = 0; h < len; h++) {
		a = line[h];
		i = a / Ysize;
		j = a % Ysize;
		f_row_used[i]++;
		row_col_idx[i] = j;
		f_col_used[j]++;
		col_row_idx[j] = i;
	}
	nb_singletons = 0;
	nb_rows_used = 0;
	for (i = 0; i < m; i++) {
		if (f_row_used[i] > 1) {
			row_idx[nb_rows_used] = i;
			nb_rows_used++;
		}
		else if (f_row_used[i] == 1) {
			j = row_col_idx[i];
			if (f_col_used[j] == 1) {
				singletons[nb_singletons++] = i * n + j;
			}
			else {
				row_idx[nb_rows_used] = i;
				nb_rows_used++;
			}
		}
	}
	nb_cols_used = 0;
	for (j = 0; j < n; j++) {
		if (f_col_used[j] > 1) {
			col_idx[nb_cols_used] = j;
			nb_cols_used++;
		}
		else if (f_col_used[j] == 1) {
			i = col_row_idx[j];
			if (f_row_used[i] > 1) {
				col_idx[nb_cols_used] = j;
				nb_cols_used++;
			}
		}
	}
}

// #############################################################################
// global functions:
// #############################################################################


static void delandtsheer_doyen_early_test_func_callback(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	delandtsheer_doyen *DD = (delandtsheer_doyen *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen_early_test_func_callback for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	DD->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "delandtsheer_doyen_early_test_func_callback done" << endl;
	}
}






}}}


