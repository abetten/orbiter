// translation_plane_via_andre_model.cpp
// 
// Anton Betten
// June 2, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {

translation_plane_via_andre_model::translation_plane_via_andre_model()
{
	F = NULL;
	q = k = n = k1 = n1 = 0;
	Andre = NULL;
	N = 0;
	twoN = 0;
	f_semilinear = FALSE;
	Line = NULL;
	Incma = NULL;
	pts_on_line = NULL;
	Line_through_two_points = NULL;
	Line_intersection = NULL;
	An = NULL;
	An1 = NULL;
	OnAndre = NULL;
	strong_gens = NULL;
	Inc = NULL;
	Stack = NULL;
	Control = NULL;
	Poset = NULL;
	arcs = NULL;
	T = NULL;
	//null();
}

translation_plane_via_andre_model::~translation_plane_via_andre_model()
{
	freeself();
}

void translation_plane_via_andre_model::null()
{
}

void translation_plane_via_andre_model::freeself()
{
	if (Andre) {
		FREE_OBJECT(Andre);
	}
	if (Line) {
		FREE_OBJECT(Line);
	}
	if (Incma) {
		FREE_int(Incma);
	}
	if (pts_on_line) {
		FREE_int(pts_on_line);
	}
	if (Line_through_two_points) {
		FREE_int(Line_through_two_points);
	}
	if (Line_intersection) {
		FREE_int(Line_intersection);
	}
#if 0
	if (An) {
		FREE_OBJECT(An);
	}
	if (An1) {
		FREE_OBJECT(An1);
	}
#endif
	if (OnAndre) {
		FREE_OBJECT(OnAndre);
	}
	if (strong_gens) {
		FREE_OBJECT(strong_gens);
	}
	if (Inc) {
		FREE_OBJECT(Inc);
	}
	if (Stack) {
		FREE_OBJECT(Stack);
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (arcs) {
		FREE_OBJECT(arcs);
	}
	null();
}


void translation_plane_via_andre_model::init(
	long int *spread_elements_numeric,
	int k, action *An, action *An1,
	vector_ge *spread_stab_gens, longinteger_object &spread_stab_go, 
	std::string &label,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_v4 = (verbose_level >= 6);
	int f_v10 = (verbose_level >= 10);
	int i, j, h, u, v, i1, i2, j1, j2;
	number_theory_domain NT;

	if (f_v) {
		cout << "translation_plane_via_andre_model::init" << endl;
		cout << "translation_plane_via_andre_model::init "
				"verbose_level=" << verbose_level << endl;
	}

	translation_plane_via_andre_model::label.assign(label);

	//translation_plane_via_andre_model::F = F;
	F = An->matrix_group_finite_field();
	if (An1->matrix_group_finite_field()->q != F->q) {
		cout << "translation_plane_via_andre_model::init "
				"The finite fields must have the same order" << endl;
		exit(1);
	}
	translation_plane_via_andre_model::q = F->q;
	translation_plane_via_andre_model::k = k;
	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"q=" << q << endl;
		cout << "translation_plane_via_andre_model::init "
				"k=" << k << endl;
	}
	n = 2 * k;
	n1 = n + 1;
	k1 = k + 1;
	
	Andre = NEW_OBJECT(andre_construction);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"spread_elements_numeric:" << endl;
		Orbiter->Lint_vec.print(cout, spread_elements_numeric,
				NT.i_power_j(q, k) + 1);
		cout << endl;
	}

	Andre->init(F, k, spread_elements_numeric, verbose_level - 2);
	
	N = Andre->N;
	twoN = 2 * N;
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"N=" << N << endl;
		cout << "translation_plane_via_andre_model::init "
				"Andre->spread_size=" << Andre->spread_size << endl;
	}



	Line = NEW_OBJECT(andre_construction_line_element);
	Incma = NEW_int(N * N);
	pts_on_line = NEW_int(Andre->spread_size);

	for (i = 0; i < N * N; i++) {
		Incma[i] = 0;
	}
	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"computing incidence matrix:" << endl;
	}

	Line->init(Andre, verbose_level);

	for (j = 0; j < N; j++) {
		if (f_v10) {
			cout << "translation_plane_via_andre_model::init "
					"before Line->unrank j=" << j << endl;
		}
		Line->unrank(j, 0 /*verbose_level*/);
		Andre->points_on_line(Line,
				pts_on_line, 0 /* verbose_level */);
		if (f_v10) {
			cout << "translation_plane_via_andre_model::init "
					"Line_" << j << "=";
			Orbiter->Int_vec.print(cout, pts_on_line, Andre->order + 1);
			cout << endl;
		}
		for (h = 0; h < Andre->order + 1; h++) {
			i = pts_on_line[h];
			if (i >= N) {
				cout << "translation_plane_via_andre_model::init "
						"i >= N" << endl;
				exit(1);
			}
			Incma[i * N + j] = 1;
		}
	}



	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"Incidence matrix of the translation plane "
				"has been computed" << endl;
	}
	

	string fname;
	file_io Fio;

	fname.assign(label);
	fname.append("_incma.csv");
	Fio.int_matrix_write_csv(fname, Incma, N, N);
	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	Line_through_two_points = NEW_int(N * N);
	for (i = 0; i < N * N; i++) {
		Line_through_two_points[i] = -1;
	}
	for (j = 0; j < N; j++) {
		Line->unrank(j, 0 /*verbose_level*/);
		Andre->points_on_line(Line,
				pts_on_line, 0 /* verbose_level */);
		for (u = 0; u < Andre->order + 1; u++) {
			i1 = pts_on_line[u];
			for (v = u + 1; v < Andre->order + 1; v++) {
				i2 = pts_on_line[v];
				Line_through_two_points[i1 * N + i2] = j;
				Line_through_two_points[i2 * N + i1] = j;
			}
		}
	}
	Line_intersection = NEW_int(N * N);
	for (i = 0; i < N * N; i++) {
		Line_intersection[i] = -1;
	}
	for (i = 0; i < N; i++) {
		for (j1 = 0; j1 < N; j1++) {
			if (Incma[i * N + j1] == 0) {
				continue;
			}
			for (j2 = j1 + 1; j2 < N; j2++) {
				if (Incma[i * N + j2] == 0) {
					continue;
				}
				Line_intersection[j1 * N + j2] = i;
				Line_intersection[j2 * N + j1] = i;
			}
		}
	}
	

	//int_matrix_print(Incma, N, N);

	//exit(1);

#if 0
	int *Adj;

	Adj = NEW_int(twoN * twoN);
	for (i = 0; i < twoN * twoN; i++) {
		Adj[i] = 0;
	}
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			if (Incma[i * N + j]) {
				Adj[i * twoN + N + j] = 1;
				Adj[(N + j) * twoN + i] = 1;
			}
		}
	}


	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"Adjacency matrix of incidence matrix has "
				"been computed" << endl;
		//int_matrix_print(Adj, twoN, twoN);
	}


	//exit(1);

	
	action *Aut;
	int parts[3];
	int nb_parts;
	int *labeling;
	longinteger_object ago;

	labeling = NEW_int(2 * twoN);

	parts[0] = N;
	parts[1] = 1;
	parts[2] = N - 1;
	nb_parts = 3;
	cout << "translation_plane_via_andre_model::init "
			"computing automorphism group of graph" << endl;
	Aut = create_automorphism_group_of_graph_with_partition_and_labeling(
		twoN, Adj, 
		nb_parts, parts, 
		labeling, 
		0 /*verbose_level*/);

	Aut->group_order(ago);

	cout << "translation_plane_via_andre_model::init "
			"Automorphism group order = " << ago << endl;
#endif

	int f_combined_action = TRUE;
	//int f_write_tda_files = TRUE;
	//int f_include_group_order = TRUE;
	//int f_pic = FALSE;
	//int f_include_tda_scheme = TRUE;
	int nb_rows = N;
	int nb_cols = N;
	

	Inc = NEW_OBJECT(incidence_structure);

	Inc->init_by_matrix(nb_rows, nb_cols,
			Incma, verbose_level - 2);
	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"after Inc->init_by_matrix" << endl;
	}
	
	translation_plane_via_andre_model::An = An;
	translation_plane_via_andre_model::An1 = An1;

	f_semilinear = An->is_semilinear_matrix_group();
	n = An->matrix_group_dimension();
	if (An1->matrix_group_dimension() != n + 1) {
		cout << "dim An1 != dim An + 1" << endl;
		cout << "dim An = " << n << endl;
		cout << "dim An1 = " << An1->matrix_group_dimension() << endl;
		exit(1);
	}


#if 0
	int f_basis = FALSE;
	vector_ge *nice_gens;



	f_semilinear = FALSE;
	if (!NT.is_prime(q)) {
		f_semilinear = TRUE;
	}
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"initializing action An" << endl;
	}
	An = NEW_OBJECT(action);
	An->init_projective_group(n, F, f_semilinear,
			f_basis, TRUE /* f_init_sims */,
			nice_gens,
			0 /* verbose_level */);
	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"initializing action An1" << endl;
	}
	An1 = NEW_OBJECT(action);
	An1->init_projective_group(n1, F, f_semilinear,
			f_basis, TRUE /* f_init_sims */,
			nice_gens,
			0 /*verbose_level */);
	FREE_OBJECT(nice_gens);
#endif

	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"initializing OnAndre" << endl;
	}


	OnAndre = NEW_OBJECT(action);
	OnAndre->induced_action_on_andre(An, An1, Andre, verbose_level);


	strong_gens = NEW_OBJECT(strong_generators);


	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"initializing spread stabilizer" << endl;
	}

	strong_gens->generators_for_translation_plane_in_andre_model(
		An1, An, 
		An1->G.matrix_grp, An->G.matrix_grp, 
		spread_stab_gens, spread_stab_go, 
		verbose_level);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"initializing spread stabilizer" << endl;
	}


	longinteger_object stab_go;

	strong_gens->group_order(stab_go);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init "
				"Stabilizer has order " << stab_go << endl;
		cout << "translation_plane_via_andre_model::init "
				"we will now compute the tactical decomposition "
				"induced by the spread stabilizer" << endl;
	}



	T = NEW_OBJECT(tactical_decomposition);
	T->init(nb_rows, nb_cols,
			Inc,
			f_combined_action,
			OnAndre /* Aut */,
			NULL /* A_on_points */,
			NULL /*A_on_lines*/,
			strong_gens /* Aut->strong_generators*/,
			verbose_level - 1);

#if 0
	int set_size = nb_rows;
	int nb_blocks = nb_cols;
		
	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();



	incidence_structure_compute_TDA_general(*Stack, 
		Inc, 
		f_combined_action, 
		OnAndre /* Aut */,
		NULL /* A_on_points */,
		NULL /*A_on_lines*/,
		strong_gens->gens /* Aut->strong_generators*/, 
		f_write_tda_files, 
		f_include_group_order, 
		f_pic, 
		f_include_tda_scheme, 
		verbose_level - 4);




	if (f_vv) {
		cout << "translation_plane_via_andre_model::init "
				"Row-scheme:" << endl;
		Inc->get_and_print_row_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
		cout << "translation_plane_via_andre_model::init "
				"Col-scheme:" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
	}
#endif
	//FREE_OBJECT(T);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init done" << endl;
	}
}


void translation_plane_via_andre_model::classify_arcs(
		const char *prefix, int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;
	int t0 = Os.os_ticks();
	//char fname_base[1000];

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs" << endl;
	}

	arcs = NEW_OBJECT(poset_classification);

	//gen->read_arguments(argc, argv, 0);

	//arcs->depth = depth;
	
	//sprintf(fname_base, "%sarcs", prefix);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::"
				"classify_arcs "
				"before gen->initialize" << endl;
	}

	Control = NEW_OBJECT(poset_classification_control);

	Control->f_w = TRUE;
	Control->f_depth = TRUE;
	Control->depth = depth;

	Poset = NEW_OBJECT(poset_with_group_action);
	Poset->init_subset_lattice(An1, OnAndre,
			strong_gens,
			verbose_level);
	arcs->initialize_and_allocate_root_node(Control, Poset,
		depth, 
		//prefix, "arcs",
		verbose_level - 1);


#if 0
	// ToDo
	arcs->init_check_func(translation_plane_via_andre_model_check_arc, 
		(void *)this /* candidate_check_data */);
#endif


	
#if 0
	arcs->f_print_function = TRUE;
	arcs->print_function = print_arc;
	arcs->print_function_data = this;
#endif

#if 0
	if (arcs->f_extend) {
		do_extend(verbose_level, arcs->verbose_level_group_theory);
		time_check(cout, t0);
		cout << endl;
		exit(0);
		}
#endif

	int schreier_depth = 1000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	//int f_implicit_fusion = FALSE;


	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs "
				"before generator_main" << endl;
	}

	arcs->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level);


	arcs->print_orbit_numbers(depth);


#if 0
	char prefix_iso[1000];
	char cmd[1000];

	sprintf(prefix_iso, "ISO/");
	sprintf(cmd, "mkdir %s", prefix_iso);
	system(cmd);

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs "
				"before isomorph_build_db" << endl;
	}

	isomorph_build_db(An1, OnAndre, arcs, 
		depth, 
		arcs->fname_base, prefix_iso, 
		depth, verbose_level);

#endif

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs done" << endl;
	}

}

void translation_plane_via_andre_model::classify_subplanes(
		const char *prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;
	int t0 = Os.os_ticks();
	int depth = 7;
	//char fname_base[1000];

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes" << endl;
	}


	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes "
				"before gen->initialize" << endl;
	}

	Control = NEW_OBJECT(poset_classification_control);

	Control->f_w = TRUE;
	Control->f_depth = TRUE;
	Control->depth = depth;

	Poset = NEW_OBJECT(poset_with_group_action);
	Poset->init_subset_lattice(An1, OnAndre,
			strong_gens,
			verbose_level);

	arcs = NEW_OBJECT(poset_classification);

	arcs->initialize_and_allocate_root_node(Control, Poset,
		depth, 
		//prefix, "subplanes",
		verbose_level - 1);


#if 0
	// ToDo
	arcs->init_check_func(
		translation_plane_via_andre_model_check_subplane,
		(void *)this /* candidate_check_data */);
#endif


	
#if 0
	arcs->f_print_function = TRUE;
	arcs->print_function = print_arc;
	arcs->print_function_data = this;
#endif

#if 0
	if (arcs->f_extend) {
		do_extend(verbose_level,
				arcs->verbose_level_group_theory);
		time_check(cout, t0);
		cout << endl;
		exit(0);
	}
#endif

	int schreier_depth = 1000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	//int f_implicit_fusion = FALSE;


	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes "
				"before generator_main" << endl;
	}

	arcs->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		f_debug, 
		verbose_level - 2);


	arcs->print_orbit_numbers(depth);


#if 0
	char prefix_iso[1000];
	char cmd[1000];

	sprintf(prefix_iso, "ISO/");
	sprintf(cmd, "mkdir %s", prefix_iso);
	system(cmd);

	if (f_v) {
		cout << "translation_plane_via_andre_model::"
				"classify_arcs before isomorph_build_db" << endl;
		}

	isomorph_build_db(An1, OnAndre, arcs, 
		depth, 
		arcs->fname_base, prefix_iso, 
		depth, verbose_level);

#endif

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes done" << endl;
	}

}

int translation_plane_via_andre_model::check_arc(
		long int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, j, h, a, b, c, l;

	if (f_v) {
		cout << "translation_plane_via_andre_model::check_arc" << endl;
	}
	if (f_vv) {
		cout << "translation_plane_via_andre_model::"
				"check_arc the set is";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
	}
	for (i = 0; i < len; i++) {
		if (/*S[i] < Andre->spread_size ||*/ S[i] >= N) {
			ret = FALSE;
			goto finish;
		}
	}
	if (len >= 3) {
		for (i = 0; i < len; i++) {
			a = S[i];
			for (j = i + 1; j < len; j++) {
				b = S[j];
				l = Line_through_two_points[a * N + b];
				for (h = 0; h < len; h++) {
					if (h == i) {
						continue;
					}
					if (h == j) {
						continue;
					}
					c = S[h];
					if (Incma[c * N + l]) {
						ret = FALSE;
						goto finish;
					}
				}
			}
		}
	}

finish:
	if (f_v) {
		cout << "translation_plane_via_andre_model::check_arc done ret=" << ret << endl;
	}
	return ret;
}

int translation_plane_via_andre_model::check_subplane(
		long int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int len2;
	int i, j, h, a, b, c, l;
	int *L;

	if (f_v) {
		cout << "translation_plane_via_andre_model::check_subplane" << endl;
	}
	if (f_vv) {
		cout << "translation_plane_via_andre_model::"
				"check_subplane the set is";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
	}


	if (len >= 2) {
		len2 = (len * (len - 1)) >> 1;
	}
	else {
		len2 = 1;
	}
	L = NEW_int(len2); // bad! No memory stuff in a test function


	for (i = 0; i < len; i++) {
		a = S[i];
		for (j = i + 1; j < len; j++) {
			b = S[j];
			if (a == b) {
				ret = FALSE;
				goto finish;
			}
		}
	}

	for (i = 0; i < len; i++) {
		if (/*S[i] < Andre->spread_size ||*/ S[i] >= N) {
			ret = FALSE;
			goto finish;
		}
	}
	if (len >= 3) {

		// compute all secants:

		h = 0;
		for (i = 0; i < len; i++) {
			a = S[i];
			for (j = i + 1; j < len; j++) {
				b = S[j];
				if (a == b) {
					cout << "translation_plane_via_andre_model::"
							"check_subplane a == b" << endl;
					exit(1);
				}
				c = Line_through_two_points[a * N + b];
				L[h] = c;
				if (f_vv) {
					cout << "Line through point " << a
							<< " and point " << b << " is " << c << endl;
				}
				h++;
			}
		}
		if (h != len2) {
			cout << "translation_plane_via_andre_model::"
					"check_subplane h != len2" << endl;
			exit(1);
		}
		tally C;

		C.init(L, len2, FALSE, 0);

		// check if no more than 7 lines:
		if (C.nb_types > 7) {
			if (f_v) {
				cout << "The set determines too many lines, "
						"namely " << C.nb_types << endl;
			}
			ret = FALSE;
			goto finish;
		}
		//check if no more than three points per line:
		for (i = 0; i < C.nb_types; i++) {
			l = C.type_len[i];
			if (l > 3) {
				if (f_v) {
					cout << "The set contains 4 collinear points" << endl;
				}
				ret = FALSE;
				goto finish;
			}
		}
	}

finish:
	FREE_int(L);
	if (f_v) {
		cout << "translation_plane_via_andre_model::check_subplane done ret=" << ret << endl;
	}
	return ret;
}

int translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane(
		long int *S, int *subplane7, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int i, j, h, a, b, l[6], d1, d2, d3, dl;

	if (f_v) {
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane" << endl;
	}
	if (f_vv) {
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane the set is";
		Orbiter->Lint_vec.print(cout, S, 4);
		cout << endl;
	}
	h = 0;
	for (i = 0; i < 4; i++) {
		a = S[i];
		for (j = i + 1; j < 4; j++) {
			b = S[j];
			l[h] = Line_through_two_points[a * N + b];
			h++;
		}
	}
	if (h != 6) {
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane" << endl;
		exit(1);
	}
	d1 = Line_intersection[l[0] * N + l[5]];
	d2 = Line_intersection[l[1] * N + l[4]];
	d3 = Line_intersection[l[2] * N + l[3]];
	dl = Line_through_two_points[d1 * N + d2];
	if (Incma[d3 * N + dl]) {
		ret = TRUE;
		for (i = 0; i < 4; i++) {
			subplane7[i] = S[i];
		}
		subplane7[4] = d1;
		subplane7[5] = d2;
		subplane7[6] = d3;
	}
	else {
		ret = FALSE;
	}

//finish:
	if (f_v) {
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane "
				"done ret=" << ret << endl;
	}
	return ret;
}


void translation_plane_via_andre_model::create_latex_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "translation_plane_via_andre_model::create_latex_report" << endl;
	}

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "%s_report.tex", label.c_str());
		fname.assign(str);
		snprintf(title, 1000, "Translation plane %s", label.c_str());
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
				cout << "translation_plane_via_andre_model::create_latex_report before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "translation_plane_via_andre_model::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "translation_plane_via_andre_model::create_latex_report done" << endl;
	}
}

void translation_plane_via_andre_model::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "translation_plane_via_andre_model::report" << endl;
	}


	Andre->report(ost, verbose_level);

	ost << "Automorphism group:\\\\" << endl;

	ost << "{\\small\\arraycolsep=2pt" << endl;
	strong_gens->print_generators_tex(ost);
	ost << "}" << endl;

	T->report(TRUE /* f_enter_math */, ost);

	if (f_v) {
		cout << "translation_plane_via_andre_model::report done" << endl;
	}
}




//
//
//

int translation_plane_via_andre_model_check_arc(
		int len, long int *S, void *data, int verbose_level)
{
	translation_plane_via_andre_model *TP =
			(translation_plane_via_andre_model *) data;
	int f_OK;
	int f_v = FALSE; //(verbose_level >= 1);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model_check_arc "
				"checking set ";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
	}
	f_OK = TP->check_arc(S, len, 0 /*verbose_level - 1*/);
	if (f_OK) {
		if (f_v) {
			cout << "accepted" << endl;
		}
		return TRUE;
	}
	else {
		if (f_v) {
			cout << "rejected" << endl;
		}
		return FALSE;
	}
}

int translation_plane_via_andre_model_check_subplane(
		int len, long int *S, void *data,
		int verbose_level)
{
	translation_plane_via_andre_model *TP =
			(translation_plane_via_andre_model *) data;
	int f_OK;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model_check_subplane checking set ";
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
	}
	f_OK = TP->check_subplane(S, len, verbose_level - 1);
	if (f_OK) {
		if (f_v) {
			cout << "accepted" << endl;
		}
		return TRUE;
	}
	else {
		if (f_v) {
			cout << "rejected" << endl;
		}
		return FALSE;
	}
}

}}


