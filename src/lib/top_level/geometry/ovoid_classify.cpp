// ovoid_classify.cpp
// 
// Anton Betten
// May 16, 2011
//
//
// 
// pulled out of ovoid: Jul 30, 2018
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {





ovoid_classify::ovoid_classify()
{
	Control = NULL;
	Poset = NULL;
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

	f_prefix = FALSE;
	nb_sol = 0;
	epsilon = 0;
	nb_colors = 0;
	N = 0;
	depth = 0;
	max_depth = 0;
	m = 0;
	q = 0;
	d = 0;
	n = 0;
}

ovoid_classify::~ovoid_classify()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "ovoid_classify::~ovoid_generator" << endl;
		}
	if (Poset) {
		FREE_OBJECT(Poset);
		}
	if (A) {
		FREE_OBJECT(A);
		}
	if (F) {
		FREE_OBJECT(F);
		}
	if (K) {
		FREE_OBJECT(K);
	}
	if (u) {
		FREE_int(u);
	}
	if (v) {
		FREE_int(v);
	}
	if (w) {
		FREE_int(w);
	}
	if (tmp1) {
		FREE_int(tmp1);
	}
	if (color_table) {
		FREE_int(color_table);
	}
	if (Pts) {
		FREE_int(Pts);
	}
	if (Candidates) {
		FREE_int(Candidates);
	}
	
	if (f_v) {
		cout << "ovoid_classify::~ovoid_generator "
				"finished" << endl;
	}
	
}

void ovoid_classify::init(int argc, const char **argv,
		int &verbose_level)
{
	int f_semilinear;
	int f_basis = TRUE;
	number_theory_domain NT;
	geometry_global Gg;

	F = NEW_OBJECT(finite_field);
	A = NEW_OBJECT(action);
	Control = NEW_OBJECT(poset_classification_control);
	gen = NEW_OBJECT(poset_classification);

	read_arguments(argc, argv, verbose_level);



	//gen->read_arguments(argc, argv, 0);

	
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 4);
	
	u = NEW_int(d);
	v = NEW_int(d);
	w = NEW_int(d);
	tmp1 = NEW_int(d);

	int p, h;
	NT.is_prime_power(q, p, h);


	if (h > 1) {
		f_semilinear = TRUE;
	}
	else {
		f_semilinear = FALSE;
	}


	//f_semilinear = TRUE;

	
	sprintf(prefix, "ovoid_Q%d_%d_%d", epsilon, d - 1, q);
	sprintf(prefix_with_directory, "%s", prefix);
	
	F->init_override_polynomial(q, override_poly, 0);

	int f_siegel = TRUE;
	int f_reflection = TRUE;
	int f_similarity = TRUE;
	int f_semisimilarity = TRUE;
	action_global AG;
	AG.set_orthogonal_group_type(f_siegel, f_reflection,
			f_similarity, f_semisimilarity);


	cout << "ovoid_classify::init "
			"d=" << d << endl;
	cout << "ovoid_classify::init "
			"f_siegel=" << f_siegel << endl;
	cout << "ovoid_classify::init "
			"f_reflection=" << f_reflection << endl;
	cout << "ovoid_classify::init "
			"f_similarity=" << f_similarity << endl;
	cout << "ovoid_classify::init "
			"f_semisimilarity=" << f_semisimilarity << endl;
	
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
		O->F->print();
		}

	if (f_v) {
		cout << "nb_points=" << O->nb_points << endl;
		cout << "nb_lines=" << O->nb_lines << endl;
		cout << "alpha=" << O->alpha << endl;
		}

	Pts = NEW_int(N * d);
	Candidates = NEW_int(N * d);



	//A->Strong_gens->print_generators_even_odd();
	

	if (f_max_depth) {
		depth = max_depth;
		}
	else {
		if (epsilon == 1) {
			depth = NT.i_power_j(q, m - 1) + 1;
			}
		else if (epsilon == -1) {
			depth = NT.i_power_j(q, m + 1) + 1;
			}
		else if (epsilon == 0) {
			depth = NT.i_power_j(q, m) + 1;
			}
		else {
			cout << "epsilon must be 0, 1, or -1" << endl;
			exit(1);
			}
		}

	Control->f_depth = TRUE;
	Control->depth = depth;


	if (f_v) {
		cout << "depth = " << depth << endl;
		}
	
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);

	Poset->add_testing_without_group(
			ovoid_classify_early_test_func_callback,
				this /* void *data */,
				verbose_level);

	Poset->f_print_function = FALSE;
	Poset->print_function = callback_ovoid_print_set;
	Poset->print_function_data = (void *) this;


	gen->initialize_and_allocate_root_node(Control, Poset,
			depth /* sz */,
			verbose_level - 1);




#if 0
	//sprintf(gen->fname_base, "ovoid_Q%d_%d_%d", epsilon, n, q);

	if (f_v) {
		cout << "fname_base = " << gen->fname_base << endl;
		}
#endif
	


	if (epsilon == 1 && d == 6) {
		if (f_v) {
			cout << "allocating Klein correspondence" << endl;
			}
		K = NEW_OBJECT(klein_correspondence);

		if (f_v) {
			cout << "before K->init" << endl;
		}
		int i, j, c, fxy;
		int B[8];
		int pivots[2] = {2,3};

		K->init(F, O, verbose_level);
		color_table = NEW_int(N);
		nb_colors = Gg.nb_AG_elements(2, F->q);
		O->unrank_point(u, 1, 0, 0);
		for (i = 0; i < N; i++) {
			O->unrank_point(v, 1, i, 0);
			fxy = O->evaluate_bilinear_form(u, v, 1);
			if (i && fxy != 0) {
				j = K->point_on_quadric_to_line(i, 0 /* verbose_level */);
				K->P3->Grass_lines->unrank_lint_here(B, j, 0 /* verbose_level */);
				F->Gauss_int_with_given_pivots(B,
					FALSE /* f_special */,
					TRUE /* f_complete */,
					pivots,
					2 /*nb_pivots*/,
					2 /*m*/, 4 /* n*/,
					0 /*verbose_level*/);
				if (B[2] != 1 || B[3] != 0 || B[6] != 0 || B[7] != 1) {
					cout << "The shape of B is wrong" << endl;
					exit(1);
				}
				c = Gg.AG_element_rank(F->q, B, 1, 2);
			}
			else {
				c = -1;
			}
			color_table[i] = c;
		}
		cout << "nb_colors = " << nb_colors << endl;
		cout << "color table:" << endl;
		for (i = 0; i < N; i++) {
			cout << i << " / " << N << " : ";
			int_vec_print(cout, v, d);

			O->unrank_point(v, 1, i, 0);
			fxy = O->evaluate_bilinear_form(u, v, 1);
			if (i && fxy != 0) {
				j = K->point_on_quadric_to_line(i, 0 /* verbose_level */);
				K->P3->Grass_lines->unrank_lint_here(B, j, 0 /* verbose_level */);
				F->Gauss_int_with_given_pivots(B,
					FALSE /* f_special */,
					TRUE /* f_complete */,
					pivots,
					2 /*nb_pivots*/,
					2 /*m*/, 4 /* n*/,
					0 /*verbose_level*/);
				cout << " : " << endl;
				int_matrix_print(B, 2, 4);
			}
			cout << " : " << color_table[i] << endl;
		}

	}
	if (f_v) {
		cout << "init finished" << endl;
		}
}

void ovoid_classify::read_arguments(
		int argc, const char **argv, int &verbose_level)
{
	int i;
	int f_epsilon = FALSE;
	int f_n = FALSE;
	int f_q = FALSE;
	geometry_global Gg;
	
#if 0
	if (argc < 1) {
		usage(argc, argv);
		exit(1);
		}
#endif
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
		cout << "Please use option -n <n> to specify "
				"the projective dimension" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}
	m = Gg.Witt_index(epsilon, n);
	d = n + 1;
	cout << "epsilon=" << epsilon << endl;
	cout << "projective dimension n=" << n << endl;
	cout << "d=" << d << endl;
	cout << "q=" << q << endl;
	cout << "Witt index " << m << endl;
}

void ovoid_classify::early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	int *v1, *v2;
	int fxy;

	if (f_v) {
		cout << "ovoid_classify::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->unrank_point(u, 1, candidates[i],
						0/*verbose_level - 4*/);
				cout << "candidate " << i << "="
						<< candidates[i] << ": ";
				int_vec_print(cout, u, d);
				cout << endl;
				}
			}
		}
	for (i = 0; i < len; i++) {
		O->unrank_point(Pts + i * d, 1, S[i], 0/*verbose_level - 4*/);
		}
	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(Candidates + i * d, 1, candidates[i],
				0/*verbose_level - 4*/);
		}

	if (len == 0) {
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
		}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "ovoid_classify::early_test_func "
					"before testing" << endl;
			}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "ovoid_generator::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
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

void ovoid_classify::print(ostream &ost, long int *S, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		for (i = 0; i < len; i++) {
			O->unrank_point(u, 1, S[i], 0);
			int_vec_print(ost, u, n);
			ost << endl;
			}
		}
}

void ovoid_classify::make_graphs(orbiter_data_file *ODF,
	int f_split, int split_r, int split_m,
	int f_lexorder_test,
	const char *fname_mask,
	int verbose_level)
{
	int orbit_idx;
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);
	char fname_graph[1000];
	int level;
	file_io Fio;

	if (f_v) {
		cout << "ovoid_classify::make_graphs" << endl;
		}

	level = ODF->set_sizes[0];

	for (orbit_idx = 0; orbit_idx < ODF->nb_cases; orbit_idx++) {

		if (f_split) {
			if ((orbit_idx % split_m) == split_r) {
				continue;
			}
		}
		cout << orbit_idx << " / " << ODF->nb_cases << " : ";
		lint_vec_print(cout, ODF->sets[orbit_idx],
				ODF->set_sizes[orbit_idx]);
		cout << " : " << ODF->Ago_ascii[orbit_idx]
				<< " : " << ODF->Aut_ascii[orbit_idx] << endl;

		sprintf(fname_graph, fname_mask, orbit_idx);

		long int *candidates;
		int nb_candidates;

#if 0
		generator_read_candidates_of_orbit(
				candidates_fname, orbit_idx /* orbit_at_level */,
				candidates, nb_candidates, 0 /* verbose_level */);
#endif

		cout << "ovoid_classify::make_graphs before read_candidates_"
				"for_one_orbit_from_file prefix=" << prefix << endl;
		Fio.read_candidates_for_one_orbit_from_file(prefix,
				level,
				orbit_idx,
				level - 1 /* level_of_candidates_file */,
				ODF->sets[orbit_idx],
				ovoid_classify_early_test_func_callback,
				this,
				candidates,
				nb_candidates,
				verbose_level);



		cout << "With " << nb_candidates << " live points: ";
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;



		if (strcmp(ODF->Ago_ascii[orbit_idx], "1") != 0) {


			int max_starter;


			strong_generators *SG;
			longinteger_object go;

			SG = NEW_OBJECT(strong_generators);
			SG->init(A);
			SG->decode_ascii_coding(
					ODF->Aut_ascii[orbit_idx], 0 /* verbose_level */);
			SG->group_order(go);

			max_starter = ODF->sets[orbit_idx]
						[ODF->set_sizes[orbit_idx] - 1];

			if (f_v) {
				cout << "max_starter=" << max_starter << endl;
			}



			if (f_lexorder_test) {
				int nb_candidates2;

				if (f_v) {
					cout << "ovoid_classify::make_graphs "
							"Case " << orbit_idx << " / "
							<< ODF->nb_cases
							<< " Before lexorder_test" << endl;
				}
				A->lexorder_test(candidates, nb_candidates,
					nb_candidates2,
					SG->gens, max_starter, 0 /*verbose_level - 3*/);
				if (f_v) {
					cout << "ovoid_classify::make_graphs "
							"After lexorder_test nb_candidates="
							<< nb_candidates2 << " eliminated "
							<< nb_candidates - nb_candidates2
							<< " candidates" << endl;
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

		FREE_OBJECT(CG);
		FREE_lint(candidates);

	}
	if (f_v) {
		cout << "ovoid_classify::make_graphs done" << endl;
		}
}

void ovoid_classify::make_one_graph(orbiter_data_file *ODF,
	int orbit_idx,
	int f_lexorder_test,
	colored_graph *&CG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int level;
	file_io Fio;

	if (f_v) {
		cout << "ovoid_classify::make_one_graph" << endl;
		}

	level = ODF->set_sizes[0];


	long int *candidates;
	int nb_candidates;


	cout << "ovoid_classify::make_one_graph before read_candidates_"
			"for_one_orbit_from_file prefix=" << prefix << endl;
	Fio.read_candidates_for_one_orbit_from_file(prefix,
			level, orbit_idx, level - 1 /* level_of_candidates_file */,
			ODF->sets[orbit_idx],
			ovoid_classify_early_test_func_callback,
			this,
			candidates,
			nb_candidates,
			verbose_level);



	cout << "With " << nb_candidates << " live points." << endl;
#if 0
	if (f_v3) {
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
#endif


	if (strcmp(ODF->Ago_ascii[orbit_idx], "1") != 0) {


		int max_starter;


		strong_generators *SG;
		longinteger_object go;

		SG = NEW_OBJECT(strong_generators);
		SG->init(A);
		SG->decode_ascii_coding(ODF->Aut_ascii[orbit_idx],
				0 /* verbose_level */);
		SG->group_order(go);

		max_starter = ODF->sets[orbit_idx][ODF->set_sizes[orbit_idx] - 1];

		if (f_v) {
			cout << "max_starter=" << max_starter << endl;
		}



		if (f_lexorder_test) {
			int nb_candidates2;

			if (f_v) {
				cout << "ovoid_classify::make_graphs Case " << orbit_idx
						<< " / " << ODF->nb_cases
						<< " Before lexorder_test" << endl;
			}
			A->lexorder_test(candidates, nb_candidates, nb_candidates2,
				SG->gens, max_starter, 0 /*verbose_level - 3*/);
			if (f_v) {
				cout << "ovoid_classify::make_graphs After "
						"lexorder_test nb_candidates=" << nb_candidates2
						<< " eliminated " << nb_candidates - nb_candidates2
						<< " candidates" << endl;
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

	FREE_lint(candidates);


	if (f_v) {
		cout << "ovoid_classify::make_one_graph done" << endl;
		}
}

void ovoid_classify::create_graph(orbiter_data_file *ODF,
	int orbit_idx,
	long int *candidates, int nb_candidates,
	colored_graph *&CG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	int fxy;
	int nb_points = nb_candidates;
	int nb_colors = 1;
	int starter_size;
	int *point_color = NULL;
	int *Pts;
	long int L;
	int nb_colors_used;

	if (f_v) {
		cout << "ovoid_classify::create_graph for orbit_idx = "
				<< orbit_idx << " nb_points = " << nb_points << endl;
	}

	starter_size = ODF->set_sizes[orbit_idx];

	uchar *bitvector_adjacency = NULL;
	//long int bitvector_length_in_bits;
	long int bitvector_length;
	Pts = NEW_int(nb_points * d);
	for (i = 0; i < nb_points; i++) {
		O->unrank_point(Pts + i * d, 1, candidates[i], 0);
	}

	L = ((long int) nb_points * ((long int) nb_points - 1)) >> 1;

	//bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_uchar(bitvector_length);
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

	point_color = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		point_color[i] = 0;
	}

	if (epsilon == 1 && d == 6) {
		compute_coloring(ODF->sets[orbit_idx], starter_size,
				candidates, nb_points, point_color,
				nb_colors_used, verbose_level);
		// check if coloring is proper:
		k = 0;
		for (i = 0; i < nb_points; i++) {
			for (j = i + 1; j < nb_points; j++, k++) {
				if (bitvector_s_i(bitvector_adjacency, k)) {
					if (point_color[i] == point_color[j]) {
						cout << "the coloring is not proper" << endl;
						cout << "point " << i << " has color "
								<< point_color[i] << endl;
						cout << "point " << j << " has color "
								<< point_color[j] << endl;
						exit(1);
					}
				}
			}
		}
	}
	else {
		nb_colors_used = nb_colors;
	}

	CG = NEW_OBJECT(colored_graph);

	CG->init(nb_points, nb_colors_used, 1,
		point_color, bitvector_adjacency, TRUE, verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	lint_vec_copy(candidates, CG->points, nb_candidates);
	CG->init_user_data(ODF->sets[orbit_idx],
			starter_size, verbose_level - 2);
	sprintf(CG->fname_base, "graph_ovoid_%d_%d_%d",
			q, starter_size, orbit_idx);

	FREE_int(Pts);
	FREE_int(point_color);
	// don't free bitvector_adjacency,
	// it has become part of the graph object
	if (f_v) {
		cout << "ovoid_classify::create_graph done" << endl;
	}
}

void ovoid_classify::compute_coloring(
		long int *starter, int starter_size,
		long int *candidates, int nb_points,
		int *point_color, int &nb_colors_used,
		int verbose_level)
{
	int f_v (verbose_level >= 1);
	int i, j, c, pos;
	sorting Sorting;

	if (f_v) {
		cout << "ovoid_classify::compute_coloring" << endl;
	}
	if (starter_size < 1) {
		cout << "starter_size < 1" << endl;
		exit(1);
	}
	if (starter[0] != 0) {
		cout << "starter[0] != 0" << endl;
		exit(1);
	}
	int *colors;
	int *color_pos;

	colors = NEW_int(nb_colors);
	color_pos = NEW_int(nb_colors);
	cout << "starter:";
	lint_vec_print(cout, starter, starter_size);
	cout << endl;
	for (i = 1; i < starter_size; i++) {
		c = color_table[starter[i]];
		colors[i - 1] = c;
		if (c == -1) {
			cout << "c == -1 for starter[i]" << endl;
			exit(1);
		}
	}
	Sorting.int_vec_heapsort(colors, starter_size - 1);
	cout << "colors:";
	int_vec_print(cout, colors, starter_size - 1);
	cout << endl;
	nb_colors_used = nb_colors - (starter_size - 1);
	int_vec_complement(colors, nb_colors, starter_size - 1);
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
	FREE_int(colors);
	FREE_int(color_pos);
	if (f_v) {
		cout << "ovoid_classify::compute_coloring done" << endl;
	}
}


void ovoid_classify_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	ovoid_classify *Gen = (ovoid_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ovoid_classify_early_test_func_callback for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	Gen->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "ovoid_classify_early_test_func_callback done" << endl;
		}
}

void callback_ovoid_print_set(ostream &ost, int len, long int *S, void *data)
{
	ovoid_classify *Gen = (ovoid_classify *) data;

	//print_vector(ost, S, len);
	Gen->print(ost, S, len);
}




}}
