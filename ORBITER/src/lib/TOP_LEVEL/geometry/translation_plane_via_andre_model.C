// translation_plane_via_andre_model.C
// 
// Anton Betten
// June 2, 2013
//
//
// 
//
//

#include "orbiter.h"


translation_plane_via_andre_model::translation_plane_via_andre_model()
{
	null();
}

translation_plane_via_andre_model::~translation_plane_via_andre_model()
{
	freeself();
}

void translation_plane_via_andre_model::null()
{
	Andre = NULL;
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
	arcs = NULL;
}

void translation_plane_via_andre_model::freeself()
{
	if (Andre) {
		delete Andre;
		}
	if (Line) {
		delete Line;
		}
	if (Incma) {
		FREE_INT(Incma);
		}
	if (pts_on_line) {
		FREE_INT(pts_on_line);
		}
	if (Line_through_two_points) {
		FREE_INT(Line_through_two_points);
		}
	if (Line_intersection) {
		FREE_INT(Line_intersection);
		}
	if (An) {
		delete An;
		}
	if (An1) {
		delete An1;
		}
	if (OnAndre) {
		delete OnAndre;
		}
	if (strong_gens) {
		delete strong_gens;
		}
	if (Inc) {
		delete Inc;
		}
	if (Stack) {
		delete Stack;
		}
	if (arcs) {
		delete arcs;
		}
	null();
}


void translation_plane_via_andre_model::init(INT *spread_elements_numeric, 
	INT k, finite_field *F, 
	vector_ge *spread_stab_gens, longinteger_object &spread_stab_go, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_v4 = (verbose_level >= 6);
	INT f_v10 = (verbose_level >= 10);
	INT i, j, h, u, v, i1, i2, j1, j2;

	if (f_v) {
		cout << "translation_plane_via_andre_model::init" << endl;
		cout << "translation_plane_via_andre_model::init verbose_level=" << verbose_level << endl;
		}

	translation_plane_via_andre_model::F = F;
	translation_plane_via_andre_model::q = F->q;
	translation_plane_via_andre_model::k = k;
	if (f_v) {
		cout << "translation_plane_via_andre_model::init q=" << q << endl;
		cout << "translation_plane_via_andre_model::init k=" << k << endl;
		}
	n = 2 * k;
	n1 = n + 1;
	k1 = k + 1;
	
	Andre = new andre_construction;

	if (f_v) {
		cout << "translation_plane_via_andre_model::init spread_elements_numeric:" << endl;
		INT_vec_print(cout, spread_elements_numeric, i_power_j(q, k) + 1);
		cout << endl;
		}

	Andre->init(F, k, spread_elements_numeric, verbose_level - 2);
	
	N = Andre->N;
	twoN = 2 * N;
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::init N=" << N << endl;
		cout << "translation_plane_via_andre_model::init Andre->spread_size=" << Andre->spread_size << endl;
		}



	Line = new andre_construction_line_element;
	Incma = NEW_INT(N * N);
	pts_on_line = NEW_INT(Andre->spread_size);

	for (i = 0; i < N * N; i++) {
		Incma[i] = 0;
		}
	if (f_v) {
		cout << "translation_plane_via_andre_model::init computing incidence matrix:" << endl;
		}

	Line->init(Andre, verbose_level);

	for (j = 0; j < N; j++) {
		if (f_v10) {
			cout << "translation_plane_via_andre_model::init before Line->unrank j=" << j << endl;
			}
		Line->unrank(j, 0 /*verbose_level*/);
		Andre->points_on_line(Line, pts_on_line, 0 /* verbose_level */);
		if (f_v10) {
			cout << "translation_plane_via_andre_model::init Line_" << j << "=";
			INT_vec_print(cout, pts_on_line, Andre->order + 1);
			cout << endl;
			}
		for (h = 0; h < Andre->order + 1; h++) {
			i = pts_on_line[h];
			if (i >= N) {
				cout << "translation_plane_via_andre_model::init i >= N" << endl;
				exit(1);
				}
			Incma[i * N + j] = 1;
			}
		}

	if (f_v) {
		cout << "translation_plane_via_andre_model::init Incidence matrix of the translation plane has been computed" << endl;
		}
	
	Line_through_two_points = NEW_INT(N * N);
	for (i = 0; i < N * N; i++) {
		Line_through_two_points[i] = -1;
		}
	for (j = 0; j < N; j++) {
		Line->unrank(j, 0 /*verbose_level*/);
		Andre->points_on_line(Line, pts_on_line, 0 /* verbose_level */);
		for (u = 0; u < Andre->order + 1; u++) {
			i1 = pts_on_line[u];
			for (v = u + 1; v < Andre->order + 1; v++) {
				i2 = pts_on_line[v];
				Line_through_two_points[i1 * N + i2] = j;
				Line_through_two_points[i2 * N + i1] = j;
				}
			}
		}
	Line_intersection = NEW_INT(N * N);
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
	

	//INT_matrix_print(Incma, N, N);

	//exit(1);

#if 0
	INT *Adj;

	Adj = NEW_INT(twoN * twoN);
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
		cout << "translation_plane_via_andre_model::init Adjacency matrix of incidence matrix has been computed" << endl;
		//INT_matrix_print(Adj, twoN, twoN);
		}


	//exit(1);

	
	action *Aut;
	INT parts[3];
	INT nb_parts;
	INT *labeling;
	longinteger_object ago;

	labeling = NEW_INT(2 * twoN);

	parts[0] = N;
	parts[1] = 1;
	parts[2] = N - 1;
	nb_parts = 3;
	cout << "translation_plane_via_andre_model::init computing automorphism group of graph" << endl;
	Aut = create_automorphism_group_of_graph_with_partition_and_labeling(
		twoN, Adj, 
		nb_parts, parts, 
		labeling, 
		0 /*verbose_level*/);

	Aut->group_order(ago);

	cout << "translation_plane_via_andre_model::init Automorphism group order = " << ago << endl;
#endif

	INT f_combined_action = TRUE;
	INT f_write_tda_files = TRUE;
	INT f_include_group_order = TRUE;
	INT f_pic = FALSE;
	INT f_include_tda_scheme = TRUE;
	INT nb_rows = N;
	INT nb_cols = N;
	

	Inc = new incidence_structure;

	Inc->init_by_matrix(nb_rows, nb_cols, Incma, verbose_level - 2);
	if (f_v) {
		cout << "translation_plane_via_andre_model::init after Inc->init_by_matrix" << endl;
		}
	

	INT f_basis = FALSE;



	f_semilinear = FALSE;
	if (!is_prime(q)) {
		f_semilinear = TRUE;
		}
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::init initializing action An" << endl;
		}
	An = new action;
	An->init_projective_group(n, F, f_semilinear, f_basis, 0 /* verbose_level */);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init initializing action An1" << endl;
		}
	An1 = new action;
	An1->init_projective_group(n1, F, f_semilinear, f_basis, 0 /*verbose_level */);


	if (f_v) {
		cout << "translation_plane_via_andre_model::init initializing OnAndre" << endl;
		}


	OnAndre = new action;
	OnAndre->induced_action_on_andre(An, An1, Andre, verbose_level);


	strong_gens = new strong_generators;


	if (f_v) {
		cout << "translation_plane_via_andre_model::init initializing spread stabilizer" << endl;
		}

	strong_gens->generators_for_translation_plane_in_andre_model(
		An1, An, 
		An1->G.matrix_grp, An->G.matrix_grp, 
		spread_stab_gens, spread_stab_go, 
		verbose_level);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init initializing spread stabilizer" << endl;
		}


	longinteger_object stab_go;

	strong_gens->group_order(stab_go);

	if (f_v) {
		cout << "translation_plane_via_andre_model::init Stabilizer has order " << stab_go << endl;
		cout << "translation_plane_via_andre_model::init we will now compute the tactical decomposition induced by the spread stabilizer" << endl;
		}

	INT set_size = nb_rows;
	INT nb_blocks = nb_cols;
		
	Stack = new partitionstack;
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();




	incidence_structure_compute_TDA_general(*Stack, 
		Inc, 
		f_combined_action, 
		OnAndre /* Aut */, NULL /* A_on_points */, NULL /*A_on_lines*/, 
		strong_gens->gens /* Aut->strong_generators*/, 
		f_write_tda_files, 
		f_include_group_order, 
		f_pic, 
		f_include_tda_scheme, 
		verbose_level - 4);
		// TOP_LEVEL/incidence_structure.C




	if (f_vv) {
		cout << "translation_plane_via_andre_model::init Row-scheme:" << endl;
		Inc->get_and_print_row_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
		cout << "translation_plane_via_andre_model::init Col-scheme:" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
		}


	if (f_v) {
		cout << "translation_plane_via_andre_model::init done" << endl;
		}
}

void translation_plane_via_andre_model::classify_arcs(const BYTE *prefix, INT depth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t0 = os_ticks();
	//BYTE fname_base[1000];

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs" << endl;
		}

	arcs = new generator;

	//gen->read_arguments(argc, argv, 0);

	arcs->depth = depth;
	
	//sprintf(fname_base, "%sarcs", prefix);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs before gen->initialize" << endl;
		}

	arcs->initialize(An1, OnAndre, 
		strong_gens,  
		depth, 
		prefix, "arcs", verbose_level - 1);


	arcs->init_check_func(translation_plane_via_andre_model_check_arc, 
		(void *)this /* candidate_check_data */);

	arcs->f_w = TRUE;

	
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

	INT schreier_depth = 1000;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	//INT f_implicit_fusion = FALSE;


	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs before generator_main" << endl;
		}

	arcs->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level);


	arcs->print_orbit_numbers(depth);


#if 0
	BYTE prefix_iso[1000];
	BYTE cmd[1000];

	sprintf(prefix_iso, "ISO/");
	sprintf(cmd, "mkdir %s", prefix_iso);
	system(cmd);

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs before isomorph_build_db" << endl;
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

void translation_plane_via_andre_model::classify_subplanes(const BYTE *prefix, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t0 = os_ticks();
	INT depth = 7;
	//BYTE fname_base[1000];

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes" << endl;
		}

	arcs = new generator;

	//gen->read_arguments(argc, argv, 0);

	arcs->f_max_depth = TRUE;
	arcs->max_depth = depth;
	arcs->depth = depth;
	
	//sprintf(fname_base, "%sarcs", prefix);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes before gen->initialize" << endl;
		}

	arcs->initialize(An1, OnAndre,  
		strong_gens, 
		depth, 
		prefix, "subplanes", verbose_level - 1);


	arcs->init_check_func(translation_plane_via_andre_model_check_subplane, 
		(void *)this /* candidate_check_data */);

	arcs->f_w = TRUE;

	
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

	INT schreier_depth = 1000;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	//INT f_implicit_fusion = FALSE;


	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_subplanes before generator_main" << endl;
		}

	arcs->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		f_debug, 
		verbose_level - 2);


	arcs->print_orbit_numbers(depth);


#if 0
	BYTE prefix_iso[1000];
	BYTE cmd[1000];

	sprintf(prefix_iso, "ISO/");
	sprintf(cmd, "mkdir %s", prefix_iso);
	system(cmd);

	if (f_v) {
		cout << "translation_plane_via_andre_model::classify_arcs before isomorph_build_db" << endl;
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

INT translation_plane_via_andre_model::check_arc(INT *S, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT ret = TRUE;
	INT i, j, h, a, b, c, l;

	if (f_v) {
		cout << "translation_plane_via_andre_model::check_arc" << endl;
		}
	if (f_vv) {
		cout << "translation_plane_via_andre_model::check_arc the set is";
		INT_vec_print(cout, S, len);
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
					if (h == i) 
						continue;
					if (h == j)
						continue;
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

INT translation_plane_via_andre_model::check_subplane(INT *S, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT ret = TRUE;
	INT len2;
	INT i, j, h, a, b, c, l;
	INT *L;

	if (f_v) {
		cout << "translation_plane_via_andre_model::check_subplane" << endl;
		}
	if (f_vv) {
		cout << "translation_plane_via_andre_model::check_subplane the set is";
		INT_vec_print(cout, S, len);
		cout << endl;
		}


	if (len >= 2) {
		len2 = (len * (len - 1)) >> 1;
		}
	else {
		len2 = 1;
		}
	L = NEW_INT(len2);


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
		h = 0;
		for (i = 0; i < len; i++) {
			a = S[i];
			for (j = i + 1; j < len; j++) {
				b = S[j];
				if (a == b) {
					cout << "translation_plane_via_andre_model::check_subplane a == b" << endl;
					exit(1);
					}
				c = Line_through_two_points[a * N + b];
				L[h] = c;
				if (f_vv) {
					cout << "Line through point " << a << " and point " << b << " is " << c << endl;
					}
				h++;
				}
			}
		if (h != len2) {
			cout << "translation_plane_via_andre_model::check_subplane h != len2" << endl;
			exit(1);
			}
		classify C;

		C.init(L, len2, FALSE, 0);

		// check if no more than 7 lines:
		if (C.nb_types > 7) {
			if (f_v) {
				cout << "The set determines too many lines, namely " << C.nb_types << endl;
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
	FREE_INT(L);
	if (f_v) {
		cout << "translation_plane_via_andre_model::check_subplane done ret=" << ret << endl;
		}
	return ret;
}

INT translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane(INT *S, INT *subplane7, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT ret = TRUE;
	INT i, j, h, a, b, l[6], d1, d2, d3, dl;

	if (f_v) {
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane" << endl;
		}
	if (f_vv) {
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane the set is";
		INT_vec_print(cout, S, 4);
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
		cout << "translation_plane_via_andre_model::check_if_quadrangle_defines_a_subplane done ret=" << ret << endl;
		}
	return ret;
}

INT translation_plane_via_andre_model_check_arc(INT len, INT *S, void *data, INT verbose_level)
{
	translation_plane_via_andre_model *TP = (translation_plane_via_andre_model *) data;
	INT f_OK;
	INT f_v = FALSE; //(verbose_level >= 1);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model_check_arc checking set ";
		print_set(cout, len, S);
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

INT translation_plane_via_andre_model_check_subplane(INT len, INT *S, void *data, INT verbose_level)
{
	translation_plane_via_andre_model *TP = (translation_plane_via_andre_model *) data;
	INT f_OK;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "translation_plane_via_andre_model_check_subplane checking set ";
		print_set(cout, len, S);
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



