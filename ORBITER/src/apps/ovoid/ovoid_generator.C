// ovoid_generator.C
// 
// Anton Betten
// May 16, 2011
//
//
// 
// pulled out of ovoid: Jul 30, 2018
//

#include "orbiter.h"

#include "ovoid.h"

ovoid_generator::ovoid_generator()
{
	gen = NULL;
	F = NULL;
	A = NULL;
	O = NULL;
	
	u = NULL;
	v = NULL;
	w = NULL;
	tmp1 = NULL;

	f_max_depth = FALSE;
	f_list = FALSE;
	f_poly = FALSE;
	override_poly = NULL;
	f_draw_poset = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;

	f_read = FALSE;
	read_level = 0;

	K = NULL;
	color_table = NULL;

	Pts = NULL;
	Candidates = NULL;
}

ovoid_generator::~ovoid_generator()
{
	INT f_v = FALSE;

	if (f_v) {
		cout << "ovoid_generator::~ovoid_generator()" << endl;
		}
	if (A) {
		delete A;
		}
	if (F) {
		delete F;
		}
	if (K) {
		delete K;
	}
	if (u) {
		FREE_INT(u);
	}
	if (v) {
		FREE_INT(v);
	}
	if (w) {
		FREE_INT(w);
	}
	if (tmp1) {
		FREE_INT(tmp1);
	}
	if (color_table) {
		FREE_INT(color_table);
	}
	if (Pts) {
		FREE_INT(Pts);
	}
	if (Candidates) {
		FREE_INT(Candidates);
	}
	
	if (f_v) {
		cout << "ovoid_generator::~ovoid_generator() after delete A" << endl;
		cout << "ovoid_generator::~ovoid_generator() finished" << endl;
		}
	
}

void ovoid_generator::init(int argc, const char **argv, INT &verbose_level)
{
	INT f_semilinear;
	INT f_basis = TRUE;

	F = new finite_field;
	A = new action;
	gen = new generator;

	read_arguments(argc, argv, verbose_level);


	gen->read_arguments(argc, argv, 0);

	
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 4);
	
	u = NEW_INT(d);
	v = NEW_INT(d);
	w = NEW_INT(d);
	tmp1 = NEW_INT(d);

	INT p, h;
	is_prime_power(q, p, h);


	if (h > 1) {
		f_semilinear = TRUE;
		}
	else {
		f_semilinear = FALSE;
		}


	//f_semilinear = TRUE;

	
	sprintf(prefix, "ovoid_Q%ld_%ld_%ld", epsilon, d - 1, q);
	sprintf(prefix_with_directory, "%s", prefix);
	
	F->init_override_polynomial(q, override_poly, 0);

	INT f_siegel = TRUE;
	INT f_reflection = TRUE;
	INT f_similarity = TRUE;
	INT f_semisimilarity = TRUE;
	set_orthogonal_group_type(f_siegel, f_reflection, f_similarity, f_semisimilarity);


	cout << "ovoid_generator::init d=" << d << endl;
	cout << "ovoid_generator::init f_siegel=" << f_siegel << endl;
	cout << "ovoid_generator::init f_reflection=" << f_reflection << endl;
	cout << "ovoid_generator::init f_similarity=" << f_similarity << endl;
	cout << "ovoid_generator::init f_semisimilarity=" << f_semisimilarity << endl;
	
	A->init_orthogonal_group(epsilon, d, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level);
	

	action_on_orthogonal *AO;
	
	AO = A->G.AO;
	O = AO->O;

	N = O->nb_points;
	
	if (f_vv) {
		cout << "The finite field is:" << endl;
		O->F->print(TRUE);
		}

	if (f_v) {
		cout << "nb_points=" << O->nb_points << endl;
		cout << "nb_lines=" << O->nb_lines << endl;
		cout << "alpha=" << O->alpha << endl;
		}

	Pts = NEW_INT(N * d);
	Candidates = NEW_INT(N * d);



	//A->Strong_gens->print_generators_even_odd();
	

	if (f_max_depth) {
		depth = max_depth;
		}
	else {
		if (epsilon == 1) {
			depth = i_power_j(q, m - 1) + 1;
			}
		else if (epsilon == -1) {
			depth = i_power_j(q, m + 1) + 1;
			}
		else if (epsilon == 0) {
			depth = i_power_j(q, m) + 1;
			}
		else {
			cout << "epsilon must be 0, 1, or -1" << endl;
			exit(1);
			}
		}
	

	gen->depth = depth;
	if (f_v) {
		cout << "depth = " << depth << endl;
		}
	

	gen->init(A, A,
		A->Strong_gens,
		gen->depth /* sz */,
		verbose_level - 1);

#if 0
	gen->init_check_func(callback_check_conditions,
		(void *)this /* candidate_check_data */);
#endif

	// we have an early test function:

	gen->init_early_test_func(
		ovoid_generator_early_test_func_callback,
		this,
		verbose_level);



	gen->f_print_function = TRUE;
	gen->print_function = callback_print_set;
	gen->print_function_data = (void *) this;


	sprintf(gen->fname_base, "ovoid_Q%ld_%ld_%ld", epsilon, n, q);

	if (f_v) {
		cout << "fname_base = " << gen->fname_base << endl;
		}
	
	
	INT nb_oracle_nodes = ONE_MILLION;
	
	if (f_v) {
		cout << "calling init_oracle with " << nb_oracle_nodes << " nodes" << endl;
		}
	
	gen->init_oracle(nb_oracle_nodes, verbose_level - 1);

	if (f_v) {
		cout << "after calling init_root_node" << endl;
		}
	
	gen->root[0].init_root_node(gen, gen->verbose_level);


	if (epsilon == 1 && d == 6) {
		if (f_v) {
			cout << "allocating Klein correspondence" << endl;
			}
		K = new klein_correspondence;

		if (f_v) {
			cout << "before K->init" << endl;
		}
		INT i, j, c, fxy;
		INT B[8];
		INT pivots[2] = {2,3};

		K->init(F, O, verbose_level);
		color_table = NEW_INT(N);
		nb_colors = nb_AG_elements(2, F->q);
		O->unrank_point(u, 1, 0, 0);
		for (i = 0; i < N; i++) {
			O->unrank_point(v, 1, i, 0);
			fxy = O->evaluate_bilinear_form(u, v, 1);
			if (i && fxy != 0) {
				j = K->Point_on_quadric_to_line[i];
				K->P3->Grass_lines->unrank_INT_here(B, j, 0 /* verbose_level */);
				F->Gauss_INT_with_given_pivots(B,
					FALSE /* f_special */, TRUE /* f_complete */, pivots, 2 /*nb_pivots*/,
					2 /*m*/, 4 /* n*/,
					0 /*verbose_level*/);
				if (B[2] != 1 || B[3] != 0 || B[6] != 0 || B[7] != 1) {
					cout << "The shape of B is wrong" << endl;
					exit(1);
				}
				AG_element_rank(F->q, B, 1, 2, c);
			} else {
				c = -1;
			}
			color_table[i] = c;
		}
		cout << "nb_colors = " << nb_colors << endl;
		cout << "color table:" << endl;
		for (i = 0; i < N; i++) {
			cout << i << " / " << N << " : ";
			INT_vec_print(cout, v, d);

			O->unrank_point(v, 1, i, 0);
			fxy = O->evaluate_bilinear_form(u, v, 1);
			if (i && fxy != 0) {
				j = K->Point_on_quadric_to_line[i];
				K->P3->Grass_lines->unrank_INT_here(B, j, 0 /* verbose_level */);
				F->Gauss_INT_with_given_pivots(B,
					FALSE /* f_special */, TRUE /* f_complete */, pivots, 2 /*nb_pivots*/,
					2 /*m*/, 4 /* n*/,
					0 /*verbose_level*/);
				cout << " : " << endl;
				INT_matrix_print(B, 2, 4);
			}
			cout << " : " << color_table[i] << endl;
		}

	}
	if (f_v) {
		cout << "init() finished" << endl;
		}
}

void ovoid_generator::read_arguments(int argc, const char **argv, INT &verbose_level)
{
	INT i;
	INT f_epsilon = FALSE;
	INT f_n = FALSE;
	INT f_q = FALSE;
	
	if (argc < 1) {
		usage(argc, argv);
		exit(1);
		}
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v" << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list" << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_max_depth = TRUE;
			max_depth = atoi(argv[++i]);
			cout << "-depth " << max_depth << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			override_poly = argv[++i];
			cout << "-poly " << override_poly << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-read") == 0) {
			f_read = TRUE;
			read_level = atoi(argv[++i]);
			cout << "-read " << read_level << endl;
			}
		}
	if (!f_epsilon) {
		cout << "Please use option -epsilon <epsilon>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "Please use option -n <n> to specify the projective dimension" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}
	m = Witt_index(epsilon, n);
	d = n + 1;
	cout << "epsilon=" << epsilon << endl;
	cout << "projective dimension n=" << n << endl;
	cout << "d=" << d << endl;
	cout << "q=" << q << endl;
	cout << "Witt index " << m << endl;
}

INT ovoid_generator::check_conditions(INT len, INT *S, INT verbose_level)
{
	INT f_OK = TRUE;
	INT f_collinearity_test = FALSE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "ovoid_generator::check_conditions checking set ";
		print_set(cout, len, S);
		}
	if (!collinearity_test(S, len, verbose_level - 1)) {
		f_OK = FALSE;
		f_collinearity_test = TRUE;
		}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "not OK because of ";
			if (f_collinearity_test) {
				cout << "collinearity test";
				}
			cout << endl;
			}
		return FALSE;
		}
}

void ovoid_generator::early_test_func(INT *S, INT len,
	INT *candidates, INT nb_candidates,
	INT *good_candidates, INT &nb_good_candidates,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j;
	INT *v1, *v2;
	INT fxy;

	if (f_v) {
		cout << "ovoid_generator::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->unrank_point(u, 1, candidates[i], 0/*verbose_level - 4*/);
				cout << "candidate " << i << "=" << candidates[i] << ": ";
				INT_vec_print(cout, u, d);
				cout << endl;
				}
			}
		}
	for (i = 0; i < len; i++) {
		O->unrank_point(Pts + i * d, 1, S[i], 0/*verbose_level - 4*/);
		}
	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(Candidates + i * d, 1, candidates[i], 0/*verbose_level - 4*/);
		}

	if (len == 0) {
		INT_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
		}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "ovoid_generator::early_test_func before testing" << endl;
			}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "ovoid_generator::early_test_func testing " << j << " / " << nb_candidates << endl;
				}

			v1 = Pts + (len - 1) * d;
			v2 = Candidates + j * d;


			fxy = O->evaluate_bilinear_form(v1, v2, 1);


			if (fxy) {
				good_candidates[nb_good_candidates++] = candidates[j];
				}
			} // next j
		} // else
}

INT ovoid_generator::collinearity_test(INT *S, INT len, INT verbose_level)
{
	INT i, x, y;
	INT f_OK = TRUE;
	INT fxy;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "collinearity test" << endl;
		}
	if (f_vv) {
		for (i = 0; i < len; i++) {
			O->unrank_point(O->v1, 1, S[i], 0);
			INT_vec_print(cout, u, n);
			cout << endl;
			}
		}
	y = S[len - 1];
	O->unrank_point(v, 1, y, 0);
	
	for (i = 0; i < len - 1; i++) {
		x = S[i];
		O->unrank_point(u, 1, x, 0);

		fxy = O->evaluate_bilinear_form(u, v, 1);
		
		if (fxy == 0) {
			f_OK = FALSE;
			if (f_vv) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y << "} are collinear" << endl;
				INT_vec_print(cout, u, n);
				cout << endl;
				INT_vec_print(cout, v, n);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}
	
	if (f_v) {
		if (!f_OK) {
			cout << "collinearity test fails" << endl;
			}
		}
	return f_OK;
}

void ovoid_generator::print(INT *S, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		for (i = 0; i < len; i++) {
			O->unrank_point(u, 1, S[i], 0);
			INT_vec_print(cout, u, n);
			cout << endl;
			}
		}
}

void ovoid_generator::make_graphs(orbiter_data_file *ODF,
	INT f_split, INT split_r, INT split_m,
	INT f_lexorder_test,
	const BYTE *fname_mask,
	INT verbose_level)
{
	INT orbit_idx;
	INT f_v = (verbose_level >= 1);
	INT f_v3 = (verbose_level >= 3);
	BYTE fname_graph[1000];
	INT level;

	if (f_v) {
		cout << "ovoid_generator::make_graphs" << endl;
		}

	level = ODF->set_sizes[0];

	for (orbit_idx = 0; orbit_idx < ODF->nb_cases; orbit_idx++) {

		if (f_split) {
			if ((orbit_idx % split_m) == split_r) {
				continue;
			}
		}
		cout << orbit_idx << " / " << ODF->nb_cases << " : ";
		INT_vec_print(cout, ODF->sets[orbit_idx], ODF->set_sizes[orbit_idx]);
		cout << " : " << ODF->Ago_ascii[orbit_idx] << " : " << ODF->Aut_ascii[orbit_idx] << endl;

		sprintf(fname_graph, fname_mask, orbit_idx);

		INT *candidates;
		INT nb_candidates;

#if 0
		generator_read_candidates_of_orbit(
				candidates_fname, orbit_idx /* orbit_at_level */,
				candidates, nb_candidates, 0 /* verbose_level */);
#endif

		cout << "ovoid_generator::make_graphs before read_candidates_for_one_orbit_from_file prefix=" << prefix << endl;
		read_candidates_for_one_orbit_from_file(prefix,
				level, orbit_idx, level - 1 /* level_of_candidates_file */,
				ODF->sets[orbit_idx],
				ovoid_generator_early_test_func_callback,
				this,
				candidates,
				nb_candidates,
				verbose_level);



		cout << "With " << nb_candidates << " live points: ";
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;



		if (strcmp(ODF->Ago_ascii[orbit_idx], "1") != 0) {


			INT max_starter;


			strong_generators *SG;
			longinteger_object go;

			SG = new strong_generators;
			SG->init(A);
			SG->decode_ascii_coding(ODF->Aut_ascii[orbit_idx], 0 /* verbose_level */);
			SG->group_order(go);

			max_starter = ODF->sets[orbit_idx][ODF->set_sizes[orbit_idx] - 1];

			if (f_v) {
				cout << "max_starter=" << max_starter << endl;
			}



			if (f_lexorder_test) {
				INT nb_candidates2;

				if (f_v) {
					cout << "ovoid_generator::make_graphs Case " << orbit_idx << " / " << ODF->nb_cases << " Before lexorder_test" << endl;
				}
				A->lexorder_test(candidates, nb_candidates, nb_candidates2,
					SG->gens, max_starter, 0 /*verbose_level - 3*/);
				if (f_v) {
					cout << "ovoid_generator::make_graphs After lexorder_test nb_candidates=" << nb_candidates2 << " eliminated " << nb_candidates - nb_candidates2 << " candidates" << endl;
				}
				nb_candidates = nb_candidates2;
			}
		}







		colored_graph *CG;

		create_graph(ODF,
				orbit_idx,
				candidates, nb_candidates,
				CG,
				verbose_level);

		CG->save(fname_graph, 0);

		if (f_v3) {
			CG->print();
			//CG->print_points_and_colors();
		}

		delete CG;
		FREE_INT(candidates);

	}
	if (f_v) {
		cout << "ovoid_generator::make_graphs done" << endl;
		}
}

void ovoid_generator::make_one_graph(orbiter_data_file *ODF,
	INT orbit_idx,
	INT f_lexorder_test,
	colored_graph *&CG,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT level;

	if (f_v) {
		cout << "ovoid_generator::make_one_graph" << endl;
		}

	level = ODF->set_sizes[0];


	INT *candidates;
	INT nb_candidates;


	cout << "ovoid_generator::make_one_graph before read_candidates_for_one_orbit_from_file prefix=" << prefix << endl;
	read_candidates_for_one_orbit_from_file(prefix,
			level, orbit_idx, level - 1 /* level_of_candidates_file */,
			ODF->sets[orbit_idx],
			ovoid_generator_early_test_func_callback,
			this,
			candidates,
			nb_candidates,
			verbose_level);



	cout << "With " << nb_candidates << " live points." << endl;
#if 0
	if (f_v3) {
		INT_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
#endif


	if (strcmp(ODF->Ago_ascii[orbit_idx], "1") != 0) {


		INT max_starter;


		strong_generators *SG;
		longinteger_object go;

		SG = new strong_generators;
		SG->init(A);
		SG->decode_ascii_coding(ODF->Aut_ascii[orbit_idx], 0 /* verbose_level */);
		SG->group_order(go);

		max_starter = ODF->sets[orbit_idx][ODF->set_sizes[orbit_idx] - 1];

		if (f_v) {
			cout << "max_starter=" << max_starter << endl;
		}



		if (f_lexorder_test) {
			INT nb_candidates2;

			if (f_v) {
				cout << "ovoid_generator::make_graphs Case " << orbit_idx << " / " << ODF->nb_cases << " Before lexorder_test" << endl;
			}
			A->lexorder_test(candidates, nb_candidates, nb_candidates2,
				SG->gens, max_starter, 0 /*verbose_level - 3*/);
			if (f_v) {
				cout << "ovoid_generator::make_graphs After lexorder_test nb_candidates=" << nb_candidates2 << " eliminated " << nb_candidates - nb_candidates2 << " candidates" << endl;
			}
			nb_candidates = nb_candidates2;
		}
	}








	create_graph(ODF,
			orbit_idx,
			candidates, nb_candidates,
			CG,
			verbose_level);


#if 0
	if (f_v3) {
		CG->print();
		//CG->print_points_and_colors();
	}
#endif

	FREE_INT(candidates);


	if (f_v) {
		cout << "ovoid_generator::make_one_graph done" << endl;
		}
}

void ovoid_generator::create_graph(orbiter_data_file *ODF,
	INT orbit_idx,
	INT *candidates, INT nb_candidates,
	colored_graph *&CG,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k, fxy;
	INT nb_points = nb_candidates;
	INT nb_colors = 1;
	INT starter_size;
	INT *point_color = NULL;
	INT *Pts;
	INT L, nb_colors_used;

	if (f_v) {
		cout << "ovoid_generator::create_graph for orbit_idx = " << orbit_idx << " nb_points = " << nb_points << endl;
	}

	starter_size = ODF->set_sizes[orbit_idx];

	UBYTE *bitvector_adjacency = NULL;
	INT bitvector_length_in_bits;
	INT bitvector_length;
	Pts = NEW_INT(nb_points * d);
	for (i = 0; i < nb_points; i++) {
		O->unrank_point(Pts + i * d, 1, candidates[i], 0);
	}

	L = (nb_points * (nb_points - 1)) >> 1;

	bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_UBYTE(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
		}

	k = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++, k++) {
			fxy = O->evaluate_bilinear_form(Pts + i * d, Pts + j * d, 1);
			if (fxy != 0) {
				bitvector_m_ii(bitvector_adjacency, k, 1);
			}
		}
	}

	point_color = NEW_INT(nb_points);
	for (i = 0; i < nb_points; i++) {
		point_color[i] = 0;
	}

	if (epsilon == 1 && d == 6) {
		compute_coloring(ODF->sets[orbit_idx], starter_size,
				candidates, nb_points, point_color, nb_colors_used, verbose_level);
		// check if coloring is proper:
		k = 0;
		for (i = 0; i < nb_points; i++) {
			for (j = i + 1; j < nb_points; j++, k++) {
				if (bitvector_s_i(bitvector_adjacency, k)) {
					if (point_color[i] == point_color[j]) {
						cout << "the coloring is not proper" << endl;
						cout << "point " << i << " has color " << point_color[i] << endl;
						cout << "point " << j << " has color " << point_color[j] << endl;
						exit(1);
					}
				}
			}
		}
	} else {
		nb_colors_used = nb_colors;
	}

	CG = new colored_graph;

	CG->init(nb_points, nb_colors_used,
		point_color, bitvector_adjacency, TRUE, verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	INT_vec_copy(candidates, CG->points, nb_candidates);
	CG->init_user_data(ODF->sets[orbit_idx], starter_size, verbose_level - 2);
	sprintf(CG->fname_base, "graph_ovoid_%ld_%ld_%ld", q, starter_size, orbit_idx);

	FREE_INT(Pts);
	FREE_INT(point_color);
	// don't free bitvector_adjacency, it has become part of the graph object
	if (f_v) {
		cout << "ovoid_generator::create_graph done" << endl;
	}
}

void ovoid_generator::compute_coloring(INT *starter, INT starter_size,
		INT *candidates, INT nb_points,
		INT *point_color, INT &nb_colors_used, INT verbose_level)
{
	INT f_v (verbose_level >= 1);
	INT i, j, c, pos;

	if (f_v) {
		cout << "ovoid_generator::compute_coloring" << endl;
	}
	if (starter_size < 1) {
		cout << "starter_size < 1" << endl;
		exit(1);
	}
	if (starter[0] != 0) {
		cout << "starter[0] != 0" << endl;
		exit(1);
	}
	INT *colors;
	INT *color_pos;

	colors = NEW_INT(nb_colors);
	color_pos = NEW_INT(nb_colors);
	cout << "starter:";
	INT_vec_print(cout, starter, starter_size);
	cout << endl;
	for (i = 1; i < starter_size; i++) {
		c = color_table[starter[i]];
		colors[i - 1] = c;
		if (c == -1) {
			cout << "c == -1 for starter[i]" << endl;
			exit(1);
		}
	}
	INT_vec_heapsort(colors, starter_size - 1);
	cout << "colors:";
	INT_vec_print(cout, colors, starter_size - 1);
	cout << endl;
	nb_colors_used = nb_colors - (starter_size - 1);
	INT_vec_complement(colors, nb_colors, starter_size - 1);
	for (i = 0; i < nb_colors; i++) {
		c = colors[i];
		color_pos[c] = i;
	}
	for (i = 0; i < nb_points; i++) {
		j = candidates[i];
		c = color_table[j];
		if (c == -1) {
			cout << "c == -1" << endl;
			exit(1);
		}
		pos = color_pos[c];
		if (pos < starter_size - 1) {
			cout << "pos < starter_size - 1" << endl;
			exit(1);
		}
		point_color[i] = pos - (starter_size - 1);
	}
	FREE_INT(colors);
	FREE_INT(color_pos);
	if (f_v) {
		cout << "ovoid_generator::compute_coloring done" << endl;
	}
}


void ovoid_generator_early_test_func_callback(INT *S, INT len,
	INT *candidates, INT nb_candidates,
	INT *good_candidates, INT &nb_good_candidates,
	void *data, INT verbose_level)
{
	ovoid_generator *Gen = (ovoid_generator *) data;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ovoid_generator_early_test_func_callback for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	Gen->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "ovoid_generator_early_test_func_callback done" << endl;
		}
}

