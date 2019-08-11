/*
 * packing_was.cpp
 *
 *  Created on: Aug 7, 2019
 *      Author: betten
 */




//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




packing_was::packing_was()
{
	f_poly = FALSE;
	poly = NULL;
	f_order = FALSE;
	order = 0;
	f_dim_over_kernel = FALSE;
	dim_over_kernel = 0;
	f_recoordinatize = FALSE;
	f_select_spread = FALSE;
	//select_spread[1000];
	select_spread_nb = 0;
	f_spreads_invariant_under_H = FALSE;

	f_cliques_on_fixpoint_graph = FALSE;
	clique_size = 0;
	f_process_long_orbits = FALSE;
	process_long_orbits_r = 0;
	process_long_orbits_m = 0;
	long_orbit_length = 0;
	long_orbits_clique_size = 0;
	f_expand_cliques_of_long_orbits = FALSE;
	clique_no_r = 0;
	clique_no_m = 0;
	f_type_of_fixed_spreads = FALSE;
	f_label = FALSE;
	label = NULL;
	f_spread_tables_prefix = FALSE;
	spread_tables_prefix = "";
	f_output_path = FALSE;
	output_path = "";

	ECA = new exact_cover_arguments;
	IA = new isomorph_arguments;

	f_H = FALSE;
	H_Descr = NULL;
	H_LG = NULL;

	f_N = FALSE;
	N_Descr = NULL;
	N_LG = NULL;

	p = e = n = k = q = 0;
	F = NULL;
	T = NULL;
	P = NULL;


	H_gens = NULL;
	//longinteger_object H_go;
	H_goi = 0;
	A = NULL;
	f_semilinear = FALSE;
	M = NULL;
	dim = 0;

	Spread_type = NULL;

	//char prefix_line_orbits[1000];
	Line_orbits_under_H = NULL;
	Spread_type = NULL;

	f_report = FALSE;

	prefix_spread_orbits[0] = 0;
	Spread_orbits_under_H = NULL;
	A_on_spread_orbits = NULL;

	fname_good_orbits[0] = 0;
	nb_good_orbits = 0;
	Good_orbit_idx = NULL;
	Good_orbit_len = NULL;
	orb = NULL;

	Spread_tables_reduced = NULL;
	Spread_type_reduced = NULL;
}

packing_was::~packing_was()
{
}

void packing_was::null()
{
}

void packing_was::freeself()
{
	if (P) {
		FREE_OBJECT(P);
	}
	if (T) {
		FREE_OBJECT(T);
	}
	if (F) {
		FREE_OBJECT(F);
	}
	null();
}

void packing_was::init(int argc, const char **argv)
{

	cout << "packing_was::init" << endl;

	long int t0 = 0;
	int i;
	int verbose_level = 0;



	t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
		}
		else if (strcmp(argv[i], "-label") == 0) {
			f_label = TRUE;
			label = argv[++i];
			cout << "-label " << label << endl;
		}
		else if (strcmp(argv[i], "-order") == 0) {
			f_order = TRUE;
			order = atoi(argv[++i]);
			cout << "-order " << order << endl;
		}
		else if (strcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = TRUE;
			dim_over_kernel = atoi(argv[++i]);
			cout << "-dim_over_kernel " << dim_over_kernel << endl;
		}
		else if (strcmp(argv[i], "-recoordinatize") == 0) {
			f_recoordinatize = TRUE;
			cout << "-recoordinatize " << endl;
		}
		else if (strcmp(argv[i], "-select_spread") == 0) {
			int a;

			f_select_spread = TRUE;
			select_spread_nb = 0;
			while (TRUE) {
				a = atoi(argv[++i]);
				if (a == -1) {
					break;
				}
				select_spread[select_spread_nb++] = a;
			}
			cout << "-select_spread ";
			int_vec_print(cout, select_spread, select_spread_nb);
			cout << endl;
		}

		else if (strcmp(argv[i], "-H") == 0) {
			f_H = TRUE;
			H_Descr = NEW_OBJECT(linear_group_description);
			i += H_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-H" << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N_Descr = NEW_OBJECT(linear_group_description);
			i += N_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-N" << endl;
			}

		else if (strcmp(argv[i], "-spreads_invariant_under_H") == 0) {
			f_spreads_invariant_under_H = TRUE;
			cout << "-spreads_invariant_under_H " << endl;
		}
		else if (strcmp(argv[i], "-cliques_on_fixpoint_graph") == 0) {
			f_cliques_on_fixpoint_graph = TRUE;
			clique_size = atoi(argv[++i]);
			cout << "-cliques_on_fixpoint_graph " << clique_size << endl;
		}
		else if (strcmp(argv[i], "-type_of_fixed_spreads") == 0) {
			f_type_of_fixed_spreads = TRUE;
			clique_size = atoi(argv[++i]);
			cout << "-type_of_fixed_spreads " << clique_size << endl;
		}
		else if (strcmp(argv[i], "-process_long_orbits") == 0) {
			f_process_long_orbits = TRUE;
			clique_size = atoi(argv[++i]);
			process_long_orbits_r = atoi(argv[++i]);
			process_long_orbits_m = atoi(argv[++i]);
			long_orbit_length = atoi(argv[++i]);
			long_orbits_clique_size = atoi(argv[++i]);
			cout << "-process_long_orbits "
				<< clique_size << " "
				<< process_long_orbits_r << " "
				<< process_long_orbits_m << " "
				<< long_orbit_length << " "
				<< long_orbits_clique_size
				<< endl;
		}
		else if (strcmp(argv[i], "-expand_cliques_of_long_orbits") == 0) {
			f_expand_cliques_of_long_orbits = TRUE;
			clique_size = atoi(argv[++i]);
			clique_no_r = atoi(argv[++i]);
			clique_no_m = atoi(argv[++i]);
			long_orbit_length = atoi(argv[++i]);
			long_orbits_clique_size = atoi(argv[++i]);
			cout << "-expand_cliques_of_long_orbits "
				<< clique_size << " "
				<< clique_no_r << " "
				<< clique_no_m << " "
				<< long_orbit_length << " "
				<< long_orbits_clique_size
				<< endl;
		}
		else if (strcmp(argv[i], "-spread_tables_prefix") == 0) {
			f_spread_tables_prefix = TRUE;
			spread_tables_prefix = argv[++i];
			cout << "-spread_tables_prefix "
				<< spread_tables_prefix << endl;
		}
		else if (strcmp(argv[i], "-output_path") == 0) {
			f_output_path = TRUE;
			output_path = argv[++i];
			cout << "-output_path " << output_path << endl;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
	}

#if 1
	ECA->read_arguments(argc, argv, verbose_level);
	IA->read_arguments(argc, argv, verbose_level);


	if (!ECA->f_starter_size) {
		cout << "packing_was::init "
				"please use option -starter_size <starter_size>" << endl;
		exit(1);
		}
	if (!ECA->f_has_input_prefix) {
		cout << "packing_was::init "
				"please use option -input_prefix <input_prefix>" << endl;
		exit(1);
		}
#endif

	int f_v = (verbose_level >= 1);

	if (!f_order) {
		cout << "packing_was::init "
				"please use option -order <order>" << endl;
		exit(1);
	}
	if (!f_H) {
		cout << "packing_was::init "
				"please use option -H <group description> -end" << endl;
		exit(1);
	}


	number_theory_domain NT;

	int e1;

	NT.factor_prime_power(order, p, e);
	if (f_v) {
		cout << "packing_was::init order = " << order << " = " << p << "^" << e << endl;
	}

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "packing_was::init "
					"dim_over_kernel does not divide e" << endl;
			exit(1);
		}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = NT.i_power_j(p, e1);
		if (f_v) {
			cout << "packing_was::init "
					"order=" << order << " n=" << n
					<< " k=" << k << " q=" << q << endl;
		}
	}
	else {
		n = 2 * e;
		k = e;
		q = p;
		if (f_v) {
			cout << "packing_was::init "
					"order=" << order << " n=" << n
					<< " k=" << k << " q=" << q << endl;
		}
	}

	if (q != H_Descr->input_q) {
		cout << "packing_was::init "
				"q != H_Descr->input_q" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "packing_was::init q=" << q << endl;
		}
	F = NEW_OBJECT(finite_field);

	F->init_override_polynomial(q, poly, 0 /* verbose_level */);

	H_Descr->F = F;



	// set up the group H:


	H_LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "packing_was::init before H_LG->init, "
				"creating the group" << endl;
		}

	H_LG->init(H_Descr, verbose_level - 1);

	if (f_v) {
		cout << "packing_was::init after H_LG->init" << endl;
		}


	A = H_LG->A2;

	if (f_v) {
		cout << "packing_was::init created group " << H_LG->prefix << endl;
	}

	if (!A->is_matrix_group()) {
		cout << "packing_was::init the group is not a matrix group " << endl;
		exit(1);
	}


	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "packing_was::init f_semilinear=" << f_semilinear << endl;
	}


	M = A->get_matrix_group();
	dim = M->n;

	if (f_v) {
		cout << "packing_was::init dim=" << dim << endl;
	}

	H_gens = H_LG->Strong_gens;
	if (f_v) {
		cout << "packing_was::init H_gens=" << endl;
		H_gens->print_generators_tex(cout);
	}
	H_goi = H_gens->group_order_as_int();
	if (f_v) {
		cout << "packing_was::init H_goi=" << H_goi << endl;
	}

	orb = NEW_int(H_goi);

	// end set up H




	T = NEW_OBJECT(spread_classify);

	T->read_arguments(argc, argv);

	int max_depth = order + 1;

	if (f_v) {
		cout << "packing_was::init before T->init" << endl;
	}
	T->init(order, n, k, max_depth,
		F, f_recoordinatize,
		"TP_STARTER", "TP", order + 1,
		argc, argv,
		MINIMUM(verbose_level - 1, 2));
	if (f_v) {
		cout << "packing_was::init after T->init" << endl;
	}

	if (f_v) {
		cout << "packing_was::init before T->init2" << endl;
	}
	T->init2(verbose_level);
	if (f_v) {
		cout << "packing_was::init after T->init2" << endl;
	}




	P = NEW_OBJECT(packing_classify);


	if (f_v) {
		cout << "packing_was::init before P->init" << endl;
	}
	P->init(T,
		f_select_spread,
		select_spread,
		select_spread_nb,
		ECA->input_prefix, ECA->base_fname,
		ECA->starter_size,
		ECA->f_lex,
		spread_tables_prefix,
		verbose_level);
	if (f_v) {
		cout << "packing_was::init after P->init" << endl;
	}

	if (f_v) {
		cout << "packing_was::init before IA->init" << endl;
	}
	IA->init(T->A, P->A_on_spreads, P->gen,
		P->size_of_packing, P->prefix_with_directory, ECA,
		callback_packing_report,
		NULL /*callback_subset_orbits*/,
		P,
		verbose_level);
	if (f_v) {
		cout << "packing_was::init after IA->init" << endl;
	}



	init_spreads(verbose_level);





	if (f_report) {
		cout << "doing a report" << endl;

		file_io Fio;

		{
		char fname[1000];
		char title[1000];
		char author[1000];
		//int f_with_stabilizers = TRUE;

		sprintf(title, "Packings in PG(3,%d) ", q);
		sprintf(author, "Orbiter");
		sprintf(fname, "Packings_q%d.tex", q);

			{
			ofstream fp(fname);
			latex_interface L;

			//latex_head_easy(fp);
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

			fp << "\\section{The field of order " << q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< q
					<< "}$ :\\\\" << endl;
			F->cheat_sheet(fp, verbose_level);

#if 0
			fp << "\\section{The space PG$(3, " << q << ")$}" << endl;

			fp << "The points in the plane PG$(2, " << q << ")$:\\\\" << endl;

			fp << "\\bigskip" << endl;


			Gen->P->cheat_sheet_points(fp, 0 /*verbose_level*/);

			fp << endl;
			fp << "\\section{Poset Classification}" << endl;
			fp << endl;
#endif
			fp << "\\section{The Group $H$}" << endl;
			H_gens->print_generators_tex(fp);

			report(fp);

			L.foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}



	the_end(t0);
	//the_end_quietly(t0);

	if (f_v) {
		cout << "packing_was::init done" << endl;
	}
}

void packing_was::init_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_spreads" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads before P->read_spread_table" << endl;
		}
	P->read_spread_table(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after P->read_spread_table" << endl;
		}


	if (f_v) {
		cout << "packing_was::init_spreads before P->read_spread_table" << endl;
		}
	compute_H_orbits_on_lines(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after P->read_spread_table" << endl;
		}
	if (f_v) {
		cout << "packing_was::init_spreads before compute_spread_types_wrt_H" << endl;
		}
	compute_spread_types_wrt_H(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after compute_spread_types_wrt_H" << endl;
		}


	if (f_v) {
		cout << "packing_was::init_spreads before P->create_action_on_spreads" << endl;
		}
	P->create_action_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after P->create_action_on_spreads" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before compute_H_orbits_on_spreads" << endl;
		}
	compute_H_orbits_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after compute_H_orbits_on_spreads" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before test_orbits_on_spreads" << endl;
		}
	test_orbits_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after test_orbits_on_spreads" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before reduce_spreads" << endl;
		}
	reduce_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after reduce_spreads" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before compute_reduced_spread_types_wrt_H" << endl;
		}
	compute_reduced_spread_types_wrt_H(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after compute_reduced_spread_types_wrt_H" << endl;
		}



#if 0
	if (f_v) {
		cout << "packing_was::init_spreads "
				"before create_action_on_good_orbits" << endl;
		}
	create_action_on_good_orbits(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after create_action_on_good_orbits" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before classify_orbits_by_length" << endl;
		}
	classify_orbits_by_length(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after classify_orbits_by_length" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before action_on_fixpoints" << endl;
		}
	action_on_fixpoints(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after action_on_fixpoints" << endl;
		}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before create_graph_on_fixpoints" << endl;
		}
	create_graph_on_fixpoints(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after create_graph_on_fixpoints" << endl;
		}
#endif

	if (f_v) {
		cout << "packing_was::init_spreads done" << endl;
		}
}

void packing_was::compute_H_orbits_on_lines(int verbose_level)
// computes the orbits of H on lines (NOT on spreads!)
// and writes to file prefix_line_orbits
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines" << endl;
		}

	if (f_output_path) {
		sprintf(prefix_line_orbits, "%s%s_line_orbits", output_path, H_LG->prefix);
	}
	else {
		sprintf(prefix_line_orbits, "%s_line_orbits", H_LG->prefix);
	}
	Line_orbits_under_H = NEW_OBJECT(orbits_on_something);

	Line_orbits_under_H->init(P->T->A2, H_gens, TRUE /*f_load_save*/,
			prefix_line_orbits,
			verbose_level);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines done" << endl;
		}
}

void packing_was::compute_spread_types_wrt_H(int verbose_level)
// Spread_types[P->nb_spreads * (group_order + 1)]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_was::compute_spread_types_wrt_H" << endl;
	}
	Spread_type = NEW_OBJECT(orbit_type_repository);
	Spread_type->init(
			Line_orbits_under_H,
			P->Spread_tables->nb_spreads,
			P->spread_size,
			P->Spread_tables->spread_table,
			H_goi,
			verbose_level);
	cout << "The spread types are:" << endl;
	Spread_type->report(cout);

	if (f_v) {
		cout << "packing_was::compute_spread_types_wrt_H done" << endl;
	}
}

void packing_was::compute_H_orbits_on_spreads(int verbose_level)
// computes the orbits of H on spreads (NOT on lines!)
// and writes to file fname_orbits
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads" << endl;
		}


	Spread_orbits_under_H = NEW_OBJECT(orbits_on_something);

	if (f_output_path) {
		sprintf(prefix_spread_orbits, "%s%s_spread_orbits", output_path, H_LG->prefix);
	}
	else {
		sprintf(prefix_spread_orbits, "%s_spread_orbits", H_LG->prefix);
	}
	Spread_orbits_under_H->init(P->A_on_spreads,
			H_gens, TRUE /*f_load_save*/,
			prefix_spread_orbits,
			verbose_level);


	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads "
				"creating action A_on_spread_orbits" << endl;
		}


	A_on_spread_orbits = NEW_OBJECT(action);
	A_on_spread_orbits->induced_action_on_orbits(P->A_on_spreads,
			Spread_orbits_under_H->Sch /* H_orbits_on_spreads*/,
			TRUE /*f_play_it_safe*/, 0 /* verbose_level */);

	if (f_v) {
		cout << "prime_at_a_time::compute_H_orbits_on_spreads "
				"created action on orbits of degree "
				<< A_on_spread_orbits->degree << endl;
		}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads "
				"created action A_on_spread_orbits done" << endl;
		}
}

void packing_was::test_orbits_on_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads "
				"We will now test "
				"which of the " << Spread_orbits_under_H->Sch->nb_orbits
				<< " orbits are partial packings:" << endl;
		}

	if (f_output_path) {
		sprintf(fname_good_orbits, "%s%s_good_orbits", output_path, H_LG->prefix);
	}
	else {
		sprintf(fname_good_orbits, "%s_good_orbits", H_LG->prefix);
	}

	if (Fio.file_size(fname_good_orbits) > 0) {

		int *M;
		int m, n, i;

		Fio.int_matrix_read_csv(fname_good_orbits, M, m, n,
				0 /* verbose_level */);

		nb_good_orbits = m;
		Good_orbit_idx = NEW_int(Spread_orbits_under_H->Sch->nb_orbits);
		Good_orbit_len = NEW_int(Spread_orbits_under_H->Sch->nb_orbits);
		for (i = 0; i < m; i++) {
			Good_orbit_idx[i] = M[i * 2 + 0];
			Good_orbit_len[i] = M[i * 2 + 1];
			}

		}
	else {
		int orbit_idx;

		nb_good_orbits = 0;
		Good_orbit_idx = NEW_int(Spread_orbits_under_H->Sch->nb_orbits);
		Good_orbit_len = NEW_int(Spread_orbits_under_H->Sch->nb_orbits);
		for (orbit_idx = 0;
				orbit_idx < Spread_orbits_under_H->Sch->nb_orbits;
				orbit_idx++) {


			if (P->test_if_orbit_is_partial_packing(
					Spread_orbits_under_H->Sch, orbit_idx,
					orb, 0 /* verbose_level*/)) {
				Good_orbit_idx[nb_good_orbits] = orbit_idx;
				Good_orbit_len[nb_good_orbits] =
						Spread_orbits_under_H->Sch->orbit_len[orbit_idx];
				nb_good_orbits++;
				}


			}


		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads "
					"We found "
					<< nb_good_orbits << " orbits which are "
							"partial packings" << endl;
			}

		int *Vec[2];
		const char *Col_labels[2] = {"Orbit_idx", "Orbit_len"};

		Vec[0] = Good_orbit_idx;
		Vec[1] = Good_orbit_len;


		Fio.int_vec_array_write_csv(2 /* nb_vecs */, Vec,
				nb_good_orbits, fname_good_orbits, Col_labels);
		cout << "Written file " << fname_good_orbits
				<< " of size " << Fio.file_size(fname_good_orbits) << endl;
		}


	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads "
				"We found "
				<< nb_good_orbits << " orbits which "
						"are partial packings" << endl;
		}


	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads done" << endl;
		}
}

void packing_was::reduce_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::reduce_spreads " << endl;
		}

	int nb_good_spreads;
	int *good_spreads;
	int i, j, h, f, l, c;


	nb_good_spreads = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		j = Good_orbit_idx[i];
		nb_good_spreads += Spread_orbits_under_H->Sch->orbit_len[j];
	}

	if (f_v) {
		cout << "packing_was::reduce_spreads nb_good_spreads = " << nb_good_spreads << endl;
		}

	good_spreads = NEW_int(nb_good_spreads);

	c = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		j = Good_orbit_idx[i];
		f = Spread_orbits_under_H->Sch->orbit_first[j];
		l = Spread_orbits_under_H->Sch->orbit_len[j];
		for (h = 0; h < l; h++) {
			good_spreads[c++] = Spread_orbits_under_H->Sch->orbit[f + h];
		}
	}
	if (c != nb_good_spreads) {
		cout << "packing_was::reduce_spreads c != nb_good_spreads" << endl;
		exit(1);
	}


	Spread_tables_reduced = NEW_OBJECT(spread_tables);

	Spread_tables_reduced->init_reduced(
			nb_good_spreads, good_spreads,
			P->Spread_tables,
			verbose_level);

	if (f_v) {
		cout << "packing_was::reduce_spreads done" << endl;
		}

}

void packing_was::compute_reduced_spread_types_wrt_H(int verbose_level)
// Spread_types[P->nb_spreads * (group_order + 1)]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_was::compute_reduced_spread_types_wrt_H" << endl;
	}
	Spread_type_reduced = NEW_OBJECT(orbit_type_repository);
	Spread_type_reduced->init(
			Line_orbits_under_H,
			Spread_tables_reduced->nb_spreads,
			P->spread_size,
			Spread_tables_reduced->spread_table,
			H_goi,
			verbose_level);
	cout << "The reduced spread types are:" << endl;
	Spread_type_reduced->report(cout);

	if (f_v) {
		cout << "packing_was::compute_reduced_spread_types_wrt_H done" << endl;
	}
}



void packing_was::report(ostream &ost)
{



	ost << "\\section{Fixed Objects of $H$}" << endl;
	ost << endl;
	H_gens->report_fixed_objects_in_P3(
			ost,
			P->P3,
			0 /* verbose_level */);
	ost << endl;


	ost << "\\section{Line Orbits of $H$}" << endl;
	ost << endl;
	Line_orbits_under_H->report_orbit_lengths(ost);
	ost << endl;

	ost << "\\section{Spread Orbits of $H$}" << endl;
	ost << endl;
	Spread_orbits_under_H->report_orbit_lengths(ost);
	ost << endl;

	ost << "\\section{Spread Types}" << endl;
	Spread_type->report(ost);
	ost << endl;

	ost << "\\section{Reduced Spread Types}" << endl;
	Spread_type_reduced->report(ost);
	ost << endl;

}

}}


