/*
 * delandtsheer_doyen.cpp
 *
 *  Created on: Nov 5, 2019
 *      Author: anton
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

delandtsheer_doyen::delandtsheer_doyen()
{
	argc= 0;
	argv = NULL;
	d1 = 0;
	d2 = 0;
	q1 = 0;
	q2 = 0;

	group_label = NULL;

	DELANDTSHEER_DOYEN_X = -1;
	DELANDTSHEER_DOYEN_Y = -1;
	K = 0;
	V = 0;
	b = 0;

	Xsize = 0; // = D = q1 = # of rows
	Ysize = 0; // = C = q2 = # of cols

	line = NULL;        // [K];
	row_sum = NULL;
	col_sum = NULL;


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

	f_subgroup = FALSE;
	pair_orbit = NULL;
	nb_orbits = 0;
	transporter = NULL;
	tmp_Elt = NULL;
	orbit_length = NULL;
	orbit_covered = NULL;
	orbit_covered_max = NULL;
		// orbit_covered_max[i] = orbit_length[i] / b;
	orbits_covered = NULL;


	// intersection type tests:

	inner_pairs_in_rows = 0;
	inner_pairs_in_cols = 0;

	// row intersection type
	f_R = FALSE;
	nb_row_types = 0;
	row_type = NULL;     		// [nb_row_types + 1]
	row_type_cur = NULL; 		// [nb_row_types + 1]
	row_type_this_or_bigger = NULL; 	// [nb_row_types + 1]

	// col intersection type
	f_C = FALSE;
	nb_col_types = 0;
	col_type = NULL;     		// [nb_col_types + 1]
	col_type_cur = NULL; 		// [nb_col_types + 1]
	col_type_this_or_bigger = NULL; 	// [nb_col_types + 1]


	// mask related test:
	nb_mask_tests = 0;

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
	if (orbit_covered_max) {
		FREE_int(orbit_covered_max);
	}
	if (orbits_covered) {
		FREE_int(orbits_covered);
	}
	if (row_type) {
		FREE_int(row_type);
	}
	if (row_type_cur) {
		FREE_int(row_type_cur);
	}
	if (row_type_this_or_bigger) {
		FREE_int(row_type_this_or_bigger);
	}
	if (col_type) {
		FREE_int(col_type);
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

void delandtsheer_doyen::init(int argc, const char **argv,
		int d1, int q1, int d2, int q2,
		int f_subgroup, const char *subgroup_gens_text,
		const char *subgroup_order_text,
		const char *group_label,
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;
	int f_K = FALSE;
	int f_singletons = FALSE;
	os_interface Os;

	if (f_v) {
		cout << "delandtsheer_doyen::init" << endl;
	}
	delandtsheer_doyen::argc = argc;
	delandtsheer_doyen::argv = argv;
	delandtsheer_doyen::d1 = d1;
	delandtsheer_doyen::d2 = d2;
	delandtsheer_doyen::q1 = q1;
	delandtsheer_doyen::q2 = q2;
	delandtsheer_doyen::group_label = group_label;
	delandtsheer_doyen::f_subgroup = f_subgroup;

	A = NEW_OBJECT(action);
	A1 = NEW_OBJECT(action);
	A2 = NEW_OBJECT(action);

	nb_mask_tests = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-DDx") == 0) {
			DELANDTSHEER_DOYEN_X = atoi(argv[++i]);
			cout << "-DDx " << DELANDTSHEER_DOYEN_X << endl;
		}
		else if (strcmp(argv[i], "-DDy") == 0) {
			DELANDTSHEER_DOYEN_Y = atoi(argv[++i]);
			cout << "-DDy " << DELANDTSHEER_DOYEN_Y << endl;
		}
		else if (strcmp(argv[i], "-K") == 0) {
			f_K = TRUE;
			K = atoi(argv[++i]);
			cout << "-K " << K << endl;
		}
		else if (strcmp(argv[i], "-R") == 0) {
			f_R = TRUE;
			nb_row_types = atoi(argv[++i]);
			row_type = NEW_int(nb_row_types + 1);
			row_type_cur = NEW_int(nb_row_types + 1);
			row_type_this_or_bigger = NEW_int(nb_row_types + 1);
			row_type[0] = 0;
			for (j = 1; j <= nb_row_types; j++) {
				row_type[j] = atoi(argv[++i]);
				row_type_cur[j] = 0;
				}
			cout << "-R ";
			int_vec_print(cout, row_type + 1, nb_row_types);
			cout << endl;
		}
		else if (strcmp(argv[i], "-C") == 0) {
			f_C = TRUE;
			nb_col_types = atoi(argv[++i]);
			col_type = NEW_int(nb_col_types + 1);
			col_type_cur = NEW_int(nb_col_types + 1);
			col_type_this_or_bigger = NEW_int(nb_col_types + 1);
			col_type[0] = 0;
			for (j = 1; j <= nb_col_types; j++) {
				col_type[j] = atoi(argv[++i]);
				col_type_cur[j] = 0;
				}
			cout << "-C ";
			int_vec_print(cout, col_type + 1, nb_col_types);
			cout << endl;
		}
		else if (strcmp(argv[i], "-masktest") == 0) {
			const char *who;
			const char *what;

			mask_test_level[nb_mask_tests] = atoi(argv[++i]);
			who = argv[++i];
			what = argv[++i];
			mask_test_value[nb_mask_tests] = atoi(argv[++i]);

			if (strcmp(who, "x") == 0)
				mask_test_who[nb_mask_tests] = 1;
			else if (strcmp(who, "y") == 0)
				mask_test_who[nb_mask_tests] = 2;
			else if (strcmp(who, "x+y") == 0)
				mask_test_who[nb_mask_tests] = 3;
			else if (strcmp(who, "s") == 0)
				mask_test_who[nb_mask_tests] = 4;
			else {
				cout << "masktest: unknown 'who' option: " << who << endl;
				cout << "must be one of 'x', 'y', 'x+y' or 's'" << endl;
				exit(1);
				}
			if (strcmp(what, "eq") == 0)
				mask_test_what[nb_mask_tests] = 1;
			else if (strcmp(what, "ge") == 0)
				mask_test_what[nb_mask_tests] = 2;
			else if (strcmp(what, "le") == 0)
				mask_test_what[nb_mask_tests] = 3;
			else {
				cout << "masktest: unknown 'what' option: " << who << endl;
				cout << "must be one of 'eq', 'ge' or 'le'" << endl;
				exit(1);
				}
			cout << "-masktest "
				<< mask_test_level[nb_mask_tests] << " "
				<< mask_test_who[nb_mask_tests] << " "
				<< mask_test_what[nb_mask_tests] << " "
				<< mask_test_value[nb_mask_tests] << endl;
			nb_mask_tests++;
			cout << "nb_mask_tests=" << nb_mask_tests << endl;
		}
		else if (strcmp(argv[i], "-singletons") == 0) {
			f_singletons = TRUE;
			cout << "-singletons" << endl;
		}
	}

	int t0 = Os.os_ticks();

	if (!f_K) {
		cout << "please use -K <K> to specify K" << endl;
		exit(1);
	}
	if (q1 == 1) {
		Xsize = d1;
		Ysize = d2;
	}
	else {
		Xsize = q1; // = D = q1 = # of rows
		Ysize = q2; // = C = q2 = # of cols
	}

	V = Xsize * Ysize;

	cout << "depth=" << depth << endl;
	cout << "V=" << V << endl;
	cout << "K=" << K << endl;
	cout << "Xsize=" << Xsize << endl;
	cout << "Ysize=" << Ysize << endl;
	cout << "V=" << V << endl;

	line = NEW_lint(K);
	row_sum = NEW_int(Xsize);
	col_sum = NEW_int(Ysize);
	live_points = NEW_lint(V);


	cout << "DELANDTSHEER_DOYEN_X=" << DELANDTSHEER_DOYEN_X << endl;
	cout << "DELANDTSHEER_DOYEN_Y=" << DELANDTSHEER_DOYEN_Y << endl;

	int_vec_zero(row_sum, Xsize);
	int_vec_zero(col_sum, Ysize);


	M1 = NEW_OBJECT(matrix_group);
	M2 = NEW_OBJECT(matrix_group);

	F1 = NEW_OBJECT(finite_field);
	F2 = NEW_OBJECT(finite_field);



	if (q1 == 1) {

		vector_ge *nice_gens;

		F1->init(2, 0);
		F2->init(2, 0);

		cout << "initializing projective groups:" << endl;

		A1->init_projective_group(d1, F1,
				FALSE /* f_semilinear */, TRUE /* f_basis */, TRUE /* f_init_sims */,
				nice_gens,
				verbose_level - 1);
		M1 = A1->G.matrix_grp;
		FREE_OBJECT(nice_gens);

		A2->init_projective_group(d2, F2,
				FALSE /* f_semilinear */, TRUE /* f_basis */, TRUE /* f_init_sims */,
				nice_gens,
				verbose_level - 1);
		M2 = A1->G.matrix_grp;
		FREE_OBJECT(nice_gens);

		b = 0;

	}
	else {



		b = (V * (V - 1)) / (K * (K - 1));

		if (b * (K * (K - 1)) != (V * (V - 1))) {
			cout << "integrality conditions violated" << endl;
			exit(1);
		}

		cout << "b=" << b << endl;



		F1->init(q1, 0);
		F2->init(q2, 0);



		cout << "initializing affine groups:" << endl;

		M1->init_affine_group(d1, F1,
				FALSE /* f_semilinear */, A1, verbose_level);

		M2->init_affine_group(d2, F2,
				FALSE /* f_semilinear */, A2, verbose_level);
	}

	cout << "direct_product_action::init before "
			"A->init_direct_product_group_and_restrict" << endl;

	A->init_direct_product_group_and_restrict(M1, M2,
			verbose_level);

	cout << "direct_product_action::init after "
			"A->init_direct_product_group_and_restrict" << endl;

	if (!A->f_has_subaction) {
		cout << "direct_product_action::init action "
				"A does not have a subaction" << endl;
		exit(1);
	}
	A0 = A->subaction;

	P = A0->G.direct_product_group;

	//vector_space_dimension = W->dimension_of_tensor_action;



	if (q1 == 1) {

		strong_generators *SG1;
		strong_generators *SG2;
		strong_generators *SG3;

		SG1 = NEW_OBJECT(strong_generators);
		SG2 = NEW_OBJECT(strong_generators);

		if (f_v) {
			cout << "before generators_for_the_monomial_group "
					"action" << A1->label << endl;
		}
		SG1->generators_for_the_monomial_group(A1,
			M1, verbose_level);
		if (f_v) {
			cout << "after generators_for_the_monomial_group "
					"action" << A1->label << endl;
		}


		if (f_v) {
			cout << "before generators_for_the_monomial_group "
					"action" << A2->label << endl;
		}
		SG2->generators_for_the_monomial_group(A2,
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

		action *Ar;
		long int *points;
		int nb_points;
		int h;

		nb_points = d1 * d2;
		points = NEW_lint(nb_points);
		h = 0;
		for (i = 0; i < d1; i++) {
			for (j = 0; j < d2; j++) {
				a = i * A2->degree + j;
				points[h++] = a;
			}
		} // next i


		Ar = A->restricted_action(points, nb_points,
				verbose_level);

		A = Ar;
	} // if (q1 == 1)

	else {
		if (!A0->f_has_strong_generators) {
			cout << "delandtsheer_doyen::init action A0 does not "
					"have strong generators" << endl;
			exit(1);
			}

		SG = A0->Strong_gens;
		SG->group_order(go);

		cout << "The group " << A->label << " has order " << go
				<< " and permutation degree " << A->degree << endl;
	}


	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		A->element_print_as_permutation_with_offset(
				SG->gens->ith(i), cout,
				0 /* offset*/,
				TRUE /* f_do_it_anyway_even_for_big_degree*/,
				TRUE /* f_print_cycles_of_length_one*/,
				0 /* verbose_level*/);
		//A->element_print_as_permutation(SG->gens->ith(i), cout);
		cout << endl;
		}
	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		A->element_print_as_permutation(SG->gens->ith(i), cout);
		cout << endl;
		}
	cout << "Generators in GAP format are:" << endl;
	cout << "G := Group([";
	for (i = 0; i < SG->gens->len; i++) {
		A->element_print_as_permutation_with_offset(
				SG->gens->ith(i), cout,
				1 /*offset*/,
				TRUE /* f_do_it_anyway_even_for_big_degree */,
				FALSE /* f_print_cycles_of_length_one */,
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
			a = A->element_image_of(j,
					SG->gens->ith(i), 0 /* verbose_level */);
			cout << a << " ";
			}
		cout << endl;
		}
	cout << "-1" << endl;





	strong_generators *Strong_gens;

	if (f_subgroup) {
		Strong_gens = NEW_OBJECT(strong_generators);
		int *data;
		int sz;
		int nb_gens;
		vector_ge *nice_gens;

		int_vec_scan(subgroup_gens_text, data, sz);
		nb_gens = sz / A->make_element_size;
		if (f_v) {
			cout << "before Strong_gens->init_from_data_with_target_go_ascii" << endl;
			}
		cout << "nb_gens=" << nb_gens << endl;
		Strong_gens->init_from_data_with_target_go_ascii(A0,
				data,
				nb_gens, A0->make_element_size,
				subgroup_order_text,
				nice_gens,
				verbose_level + 2);
		FREE_OBJECT(nice_gens);
		if (f_v) {
			cout << "delandtsheer_doyen "
					"after Strong_gens->init_from_data_with_target_go_ascii" << endl;
			}
		Pairs = NEW_OBJECT(poset_classification);

		Pairs->read_arguments(argc, argv, 0);

		//Pairs->prefix[0] = 0;
		sprintf(Pairs->fname_base, "pairs_%s_%d_%d",
				group_label, q1, q2);


		Pairs->depth = 2;

		Poset_pairs = NEW_OBJECT(poset);
		Poset_pairs->init_subset_lattice(A0, A, Strong_gens,
				verbose_level);


		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Pairs->init" << endl;
			}
		Pairs->init(Poset_pairs,
				Pairs->depth /* sz */, verbose_level);
		if (f_v) {
			cout << "direct_product_action::init "
					"after Pairs->init" << endl;
			}


		int nb_nodes = 1000;

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Pairs->init_poset_orbit_node" << endl;
			}
		Pairs->init_poset_orbit_node(nb_nodes, verbose_level - 1);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"calling Pairs->init_root_node" << endl;
			}
		Pairs->root[0].init_root_node(Pairs, verbose_level - 1);

		int f_use_invariant_subset_if_available;
		int f_debug;

		f_use_invariant_subset_if_available = TRUE;
		f_debug = FALSE;


		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Pairs->main" << endl;
			cout << "A=";
			A->print_info();
			cout << "A0=";
			A0->print_info();
			}


		//Pairs->f_allowed_to_show_group_elements = TRUE;

		Pairs->f_max_depth = FALSE;
		Pairs->depth = 2;
		Pairs->main(t0,
			Pairs->depth /* schreier_depth */,
			f_use_invariant_subset_if_available,
			f_debug,
			verbose_level - 2);

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after Pairs->main" << endl;
		}


		nb_orbits = Pairs->nb_orbits_at_level(2);

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"nb_orbits = "
					<< nb_orbits << endl;
		}

		transporter = NEW_int(A0->elt_size_in_int);
		tmp_Elt = NEW_int(A0->elt_size_in_int);

		orbit_length = NEW_int(nb_orbits);
		orbit_covered = NEW_int(nb_orbits);
		orbit_covered_max = NEW_int(nb_orbits);
		orbits_covered = NEW_int(K * K);

		int_vec_zero(orbit_covered, nb_orbits);

		for (i = 0; i < nb_orbits; i++) {
			orbit_length[i] = Pairs->orbit_length_as_int(
					i /* orbit_at_level*/, 2 /* level*/);
			orbit_covered_max[i] = orbit_length[i] / b;
			if (orbit_covered_max[i] * b != orbit_length[i]) {
				cout << "integrality conditions violated (2)" << endl;
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

	}
	else {
		cout << "We don't have -subgroup, so orbits on pairs "
				"are not computed" << endl;
		//exit(1);
	}



	f_row_used = NEW_int(Xsize);
	f_col_used = NEW_int(Ysize);
	row_idx = NEW_int(Xsize);
	col_idx = NEW_int(Ysize);
	singletons = NEW_int(K);

	// temporary data
	row_col_idx = NEW_int(Xsize);
	col_row_idx = NEW_int(Ysize);


	if (f_singletons) {
		cout << "searching singletons" << endl;

		int target_depth;
		target_depth = K - depth;
		cout << "target_depth=" << target_depth << endl;

		orbiter_data_file *ODF;
		char fname[1000];
		//int i, j;
		int level = depth;

		sprintf(fname, "design_%s_%d_%d_lvl_%d", group_label, q1, q2, level);

		ODF = NEW_OBJECT(orbiter_data_file);
		ODF->load(fname, verbose_level);
		cout << "found " << ODF->nb_cases
				<< " orbits at level " << level << endl;

		int *Orbit_idx;
		int nb_orbits_not_ruled_out;
		int orbit_idx;
		int nb_cases = 0;
		int nb_cases_eliminated = 0;
		int f_vv;

		Orbit_idx = NEW_int(ODF->nb_cases);
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
				f_vv = TRUE;
			}
			else {
				f_vv = FALSE;
			}
			if (f_vv) {
				cout << orbit_idx << " / " << ODF->nb_cases << " : ";
				lint_vec_print(cout, ODF->sets[orbit_idx],
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

			create_graph(line0, level, 0 /*verbose_level*/);

			if (f_vv) {
				cout << "case " << orbit_idx << " / " << ODF->nb_cases
						<< " we found " << nb_live_points << " live points" << endl;
			}
			if (nb_live_points < target_depth) {
				if (f_vv) {
					cout << "eliminated!" << endl;
				}
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
		int nb_sol = 0;

		for (orbit_not_ruled_out = 0;
				orbit_not_ruled_out < nb_orbits_not_ruled_out;
				orbit_not_ruled_out++) {
			orbit_idx = Orbit_idx[orbit_not_ruled_out];


			if ((orbit_not_ruled_out % 100)== 0) {
				f_vv = TRUE;
			} else {
				f_vv = FALSE;
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

			create_graph(
					line0, level, 0 /*verbose_level*/);

			if (f_vv) {
				cout << "orbit_not_ruled_out=" << orbit_not_ruled_out << " / "
						<< nb_orbits_not_ruled_out << " is orbit_idx"
						<< orbit_idx << " / " << ODF->nb_cases
						<< " we found " << nb_live_points
						<< " live points" << endl;
			}
			if (nb_live_points == target_depth) {
				lint_vec_copy(line0, line, level);
				lint_vec_copy(live_points, line + level, target_depth);
				if (check_orbit_covering(line,
					K, 0 /* verbose_level */)) {
					cout << "found a solution in orbit " << orbit_idx << endl;
					nb_sol++;
				}



			}
			else {
				cout << "orbit_not_ruled_out=" << orbit_not_ruled_out << " / "
						<< nb_orbits_not_ruled_out << " is orbit_idx"
						<< orbit_idx << " / " << ODF->nb_cases
						<< " we found " << nb_live_points
						<< " live points, doing a search" << endl;
				int *subset;
				int nCk, l;
				combinatorics_domain Combi;

				subset = NEW_int(target_depth);
				nCk = Combi.int_n_choose_k(nb_live_points, target_depth);
				for (l = 0; l < nCk; l++) {

					Combi.unrank_k_subset(l, subset, nb_live_points, target_depth);

					lint_vec_copy(line0, line, level);

					int_vec_apply_lint(subset, live_points, line + level, target_depth);

					if (check_orbit_covering(line,
						K, 0 /* verbose_level */)) {
						cout << "found a solution, subset " << l
								<< " / " << nCk << " in orbit "
								<< orbit_idx << endl;
						nb_sol++;
					}
				} // next l

				FREE_int(subset);
			} // else
		} // next orbit_not_ruled_out

		cout << "nb_sol=" << nb_sol << endl;
		cout << "searching singletons done" << endl;
	}
	else {


		Gen = NEW_OBJECT(poset_classification);

		Gen->read_arguments(argc, argv, 0);

		//Gen->prefix[0] = 0;

		if (f_subgroup) {
			sprintf(Gen->fname_base, "design_%s_%d_%d",
					group_label, q1, q2);
		}
		else {
			sprintf(Gen->fname_base, "design_no_group_%d_%d",
					d1, d2);

		}


		Gen->depth = depth;
		Poset_search = NEW_OBJECT(poset);
		Poset_search->init_subset_lattice(A0, A, SG,
				verbose_level);

		if (f_v) {
			cout << "delandtsheer_doyen::init before "
					"Poset->add_testing_without_group" << endl;
			}
		Poset_search->add_testing_without_group(
				delandtsheer_doyen_early_test_func_callback,
					this /* void *data */,
					verbose_level);

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Gen->init" << endl;
			}
		Gen->init(Poset_search,
				Gen->depth /* sz */, verbose_level);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after Gen->init" << endl;
			}

		int nb_nodes = 1000;

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Gen->init_poset_orbit_node" << endl;
			}
		Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"calling Pairs->init_root_node" << endl;
			}
		Gen->root[0].init_root_node(Gen, verbose_level - 1);

		int f_use_invariant_subset_if_available = TRUE;
		int f_debug = FALSE;

		//t0 = os_ticks();

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Gen->main" << endl;
			cout << "A=";
			A->print_info();
			cout << "A0=";
			A0->print_info();
			}


		//Gen->f_allowed_to_show_group_elements = TRUE;

		Gen->f_max_depth = FALSE;
		Gen->depth = depth;
		Gen->main(t0,
			Gen->depth /* schreier_depth */,
			f_use_invariant_subset_if_available,
			f_debug,
			verbose_level - 2);

		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"after Gen->main" << endl;
		}


		if (f_v) {
			cout << "delandtsheer_doyen::init "
					"before Gen->draw_poset" << endl;
		}
		Gen->draw_poset(Gen->fname_base, depth,
				0 /* data1 */, TRUE /* f_embedded */, TRUE /* f_sideways */,
				verbose_level);
	} // else


	if (f_v) {
		cout << "delandtsheer_doyen::init done" << endl;
	}
}

void delandtsheer_doyen::create_graph(
		long int *line0, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, x, y, h, ph, k, pk, o;

	if (f_v) {
		cout << "delandtsheer_doyen::create_graph" << endl;
	}

	int_vec_zero(row_sum, Xsize);
	int_vec_zero(col_sum, Ysize);

	for (i = 0; i < len; i++) {
		a = line0[i];
		x = a / Ysize;
		y = a % Ysize;
		//cout << "i=" << i << " / " << len << " a=" << a
		//	<< " x=" << x << " y=" << y << endl;
		row_sum[x]++;
		col_sum[y]++;
	}

	if (!check_orbit_covering(line0,
		len, 0 /* verbose_level */)) {
		cout << "delandtsheer_doyen::create_graph line0 is not good (check_orbit_covering)" << endl;
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
				o = find_pair_orbit(ph, a, 0 /*verbose_level - 1*/);
				orbit_covered[o]++;
				if (orbit_covered[o] > orbit_covered_max[o]) {
					for (k = h; k >= 0; k--) {
						pk = line0[k];
						o = find_pair_orbit(pk, a, 0 /*verbose_level - 1*/);
						orbit_covered[o]--;
					}
					break;
				}
			} // next h
			if (h == len) {
				live_points[nb_live_points++] = a;
			}
		} // next y
	} // next x
	if (f_v) {
		cout << "found " << nb_live_points << " live points" << endl;
	}

	if (f_v) {
		cout << "delandtsheer_doyen::create_graph done" << endl;
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
	orbit_no = Pairs->trace_set(set, 2, 2,
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
	int_vec_zero(pair_orbit, V * V);
	for (i = 0; i < V; i++) {
		for (j = i + 1; j < V; j++) {
			k = find_pair_orbit_by_tracing(i, j,
					0 /*verbose_level - 2*/);
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
	char fname[1000];
	long int set[1000];
	int i, j, k, n, size, l;


	if (f_v) {
		cout << "delandtsheer_doyen::write_pair_orbit_file" << endl;
	}
	sprintf(fname, "%s.2orbits", group_label);
	cout << "writing pair-orbit file " << fname << endl;
	{
	ofstream f(fname);
	f << nb_orbits << endl;
	for (i = 0; i < nb_orbits; i++) {
		n = Pairs->first_poset_orbit_node_at_level[2] + i;
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
	file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "delandtsheer_doyen::write_pair_orbit_file done" << endl;
	}

}

void delandtsheer_doyen::print_mask_test_i(
		ostream &ost, int i)
{
	int who, what;

	ost << "mask test at level " << mask_test_level[i] << " : ";
	who = mask_test_who[i];
	what = mask_test_what[i];
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
	ost << mask_test_value[i];
	ost << endl;
}

void delandtsheer_doyen::early_test_func(long int *S, int len,
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
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}


	if (len == 0) {
		lint_vec_copy(candidates, good_candidates, nb_candidates);
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
	int f_OK = TRUE;
	int f_bad_orbit = FALSE;
	int f_bad_row = FALSE;
	int f_bad_col = FALSE;
	int f_bad_mask = FALSE;
	int pt, idx;
	sorting Sorting;

	if (f_v) {
		cout << "delandtsheer_doyen::check_conditions "
				"checking set ";
		print_set(cout, len, S);
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
		return FALSE;
	}
	if (f_subgroup) {
		if (!check_orbit_covering(S, len, verbose_level)) {
			f_bad_orbit = TRUE;
			f_OK = FALSE;
		}
	}

	if (f_OK && !check_row_sums(S, len, verbose_level)) {
		f_bad_row = TRUE;
		f_OK = FALSE;
		}
	if (f_OK && !check_col_sums(S, len, verbose_level)) {
		f_bad_col = TRUE;
		f_OK = FALSE;
		}
	if (f_OK && !check_mask(S, len, verbose_level)) {
		f_bad_mask = TRUE;
		f_OK = FALSE;
		}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
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
		return FALSE;
		}
}

int delandtsheer_doyen::check_orbit_covering(
		long int *line,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, pi, j, pj, o, f_OK = TRUE;

	int_vec_zero(orbit_covered, nb_orbits);

	for (i = 0; i < len; i++) {
		pi = line[i];
		for (j = i + 1; j < len; j++) {
			pj = line[j];
			o = find_pair_orbit(pi, pj, 0 /*verbose_level - 1*/);
			//o = orbits_on_pairs[pi * V + pj];
			if (pi == pj) {
				cout << "delandtsheer_doyen::check_orbit_covering "
						"pi = " << pi << " == pj = " << pj << endl;
				exit(1);
				}
			orbit_covered[o]++;
			if (orbit_covered[o] > orbit_covered_max[o]) {
				f_OK = FALSE;
				break;
				}
			}
		if (!f_OK)
			break;
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

int delandtsheer_doyen::check_row_sums(long int *line,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, p, x, s, f_OK = TRUE;
	int f_DD_problem = FALSE;

	inner_pairs_in_rows = 0;
	int_vec_zero(row_sum, Xsize);
	if (f_R) {
		for (i = 1; i <= nb_row_types; i++) {
			row_type_cur[i] = 0;
			}
		}
	for (i = 0; i < len; i++) {
		p = line[i];
		x = p / Ysize;
		//y = p % Ysize;
		inner_pairs_in_rows += row_sum[x];
		row_sum[x]++;
		if (DELANDTSHEER_DOYEN_X != -1) {
			if (inner_pairs_in_rows > DELANDTSHEER_DOYEN_X) {
				f_OK = FALSE;
				f_DD_problem = TRUE;
				break;
				}
			}
		if (f_R) {
			s = row_sum[x];
			if (s > nb_row_types) {
				f_OK = FALSE;
				break;
				}
			if (row_type_cur[s] >= row_type_this_or_bigger[s]) {
				f_OK = FALSE;
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
						<< DELANDTSHEER_DOYEN_X
						<< ", not OK" << endl;
					}
				else {
					cout << "delandtsheer_doyen::check_row_sums"
							"problem with row-type:" << endl;
					for (i = 1; i <= nb_row_types; i++) {
						cout << row_type_cur[i] << " ";
						}
					cout << endl;
					for (i = 1; i <= nb_row_types; i++) {
						cout << row_type_this_or_bigger[i] << " ";
						}
					cout << endl;
					}
				}
			}
		}
	return f_OK;
}

int delandtsheer_doyen::check_col_sums(long int *line,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, p, y, s, f_OK = TRUE;
	int f_DD_problem = FALSE;

	inner_pairs_in_cols = 0;
	int_vec_zero(col_sum, Ysize);
	if (f_C) {
		for (i = 1; i <= nb_col_types; i++) {
			col_type_cur[i] = 0;
			}
		}
	for (i = 0; i < len; i++) {
		p = line[i];
		//x = p / Ysize;
		y = p % Ysize;
		inner_pairs_in_cols += col_sum[y];
		col_sum[y]++;
		if (DELANDTSHEER_DOYEN_Y != -1) {
			if (inner_pairs_in_cols > DELANDTSHEER_DOYEN_Y) {
				f_OK = FALSE;
				f_DD_problem = TRUE;
				break;
				}
			}
		if (f_C) {
			s = col_sum[y];
			if (s > nb_col_types) {
				f_OK = FALSE;
				break;
				}
			if (col_type_cur[s] >= col_type_this_or_bigger[s]) {
				f_OK = FALSE;
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
						<< DELANDTSHEER_DOYEN_Y
						<< ", not OK" << endl;
					}
				else {
					cout << "delandtsheer_doyen::check_col_sums "
							"problem with col-type:" << endl;
					for (i = 1; i <= nb_col_types; i++) {
						cout << col_type_cur[i] << " ";
						}
					cout << endl;
					for (i = 1; i <= nb_col_types; i++) {
						cout << col_type_this_or_bigger[i] << " ";
						}
					cout << endl;
					}
				}
			}
		}
	return f_OK;
}

int delandtsheer_doyen::check_mask(long int *line,
		int len, int verbose_level)
{
	//verbose_level = 4;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_OK = TRUE;
	int k, who;
	int nb_rows_used, nb_cols_used;
	int nb_singletons;


	if (f_vv) {
		cout << "delandtsheer_doyen::check_mask" << endl;
		}
	get_mask_core_and_singletons(line, len,
			nb_rows_used, nb_cols_used,
			nb_singletons, verbose_level);

	for (k = 0; k < nb_mask_tests; k++) {
		if (mask_test_level[k] != len)
			continue;
		if (mask_test_who[k] == 1) {
			who = inner_pairs_in_rows;
			}
		else if (mask_test_who[k] == 2) {
			who = inner_pairs_in_cols;
			}
		else if (mask_test_who[k] == 3) {
			who = inner_pairs_in_rows + inner_pairs_in_cols;
			}
		else if (mask_test_who[k] == 4) {
			who = nb_singletons;
			}
		else {
			cout << "delandtsheer_doyen::check_mask: "
					"unknown mask_test_who value "
					<< mask_test_who[k] << " in test " << k << endl;
			exit(1);
			}
		if (mask_test_what[k] == 1) {
			// eq
			if (who != mask_test_value[k]) {
				f_OK = FALSE;
				break;
				}
			}
		else if (mask_test_what[k] == 2) {
			// ge
			if (who < mask_test_value[k]) {
				f_OK = FALSE;
				break;
				}
			}
		else if (mask_test_what[k] == 3) {
			// le
			if (who > mask_test_value[k]) {
				f_OK = FALSE;
				break;
				}
			}
		else {
			cout << "delandtsheer_doyen::check_mask: "
					"unknown mask_test_what value "
					<< mask_test_what[k] << " in test " << k << endl;
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

	int_vec_zero(f_row_used, m);
	int_vec_zero(f_col_used, n);
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

#if 0
int delandtsheer_doyen_check_conditions(int len, long int *S,
		void *data, int verbose_level)
{
	delandtsheer_doyen *P = (delandtsheer_doyen *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen_check_conditions" << endl;
	}
	return P->check_conditions(S, len, verbose_level);
}
#endif

void delandtsheer_doyen_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	delandtsheer_doyen *P = (delandtsheer_doyen *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "delandtsheer_doyen_early_test_func_callback for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	P->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "delandtsheer_doyen_early_test_func_callback done" << endl;
		}
}






}}


