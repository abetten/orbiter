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
namespace layer5_applications {
namespace apps_geometry {


static void ovoid_classify_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
static void callback_ovoid_print_set(std::ostream &ost, int len, long int *S, void *data);




ovoid_classify::ovoid_classify()
{
	Descr = NULL;
	Control = NULL;
	LG = NULL;

	Poset = NULL;
	gen = NULL;
	A = NULL;
	O = NULL;
	
	u = NULL;
	v = NULL;
	w = NULL;
	tmp1 = NULL;

	K = NULL;
	color_table = NULL;

	Pts = NULL;
	Candidates = NULL;

	nb_sol = 0;
	nb_colors = 0;
	N = 0;
	m = 0;
}

ovoid_classify::~ovoid_classify()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "ovoid_classify::~ovoid_classify" << endl;
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (A) {
		FREE_OBJECT(A);
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
		cout << "ovoid_classify::~ovoid_classify finished" << endl;
	}
	
}

void ovoid_classify::init(ovoid_classify_description *Descr,
		groups::linear_group *LG,
		int &verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 4);

	//int f_semilinear;
	//int f_basis = TRUE;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;

	ovoid_classify::Descr = Descr;

	Control = Get_object_of_type_poset_classification_control(Descr->control_label);

	ovoid_classify::LG = LG;


	A = LG->A2;
	gen = NEW_OBJECT(poset_classification::poset_classification);

	
	u = NEW_int(Descr->d);
	v = NEW_int(Descr->d);
	w = NEW_int(Descr->d);
	tmp1 = NEW_int(Descr->d);

	int p, h;
	NT.is_prime_power(LG->F->q, p, h);

#if 0
	if (h > 1) {
		f_semilinear = TRUE;
	}
	else {
		f_semilinear = FALSE;
	}
#endif



#if 0
	int f_siegel = TRUE;
	int f_reflection = TRUE;
	int f_similarity = TRUE;
	int f_semisimilarity = TRUE;
	action_global AG;
	AG.set_orthogonal_group_type(f_siegel, f_reflection,
			f_similarity, f_semisimilarity);


	cout << "ovoid_classify::init "
			"d=" << Descr->d << endl;
	cout << "ovoid_classify::init "
			"f_siegel=" << f_siegel << endl;
	cout << "ovoid_classify::init "
			"f_reflection=" << f_reflection << endl;
	cout << "ovoid_classify::init "
			"f_similarity=" << f_similarity << endl;
	cout << "ovoid_classify::init "
			"f_semisimilarity=" << f_semisimilarity << endl;
	

	A->init_orthogonal_group(Descr->epsilon, Descr->d, LG->F,
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level);
	
#endif

	induced_actions::action_on_orthogonal *AO;
	
	AO = A->G.AO;
	O = AO->O;

	N = O->Hyperbolic_pair->nb_points;
	
	if (f_vv) {
		cout << "The finite field is:" << endl;
		O->F->print();
		}

	if (f_v) {
		cout << "nb_points=" << O->Hyperbolic_pair->nb_points << endl;
		cout << "nb_lines=" << O->Hyperbolic_pair->nb_lines << endl;
		cout << "alpha=" << O->Hyperbolic_pair->alpha << endl;
		}

	Pts = NEW_int(N * Descr->d);
	Candidates = NEW_int(N * Descr->d);



	//A->Strong_gens->print_generators_even_odd();
	
#if 0

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

	//Control->f_depth = TRUE;
	//Control->depth = depth;


	if (f_v) {
		cout << "depth = " << depth << endl;
		}
#endif
	
	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
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

#if 0
	gen->initialize_and_allocate_root_node(Control, Poset,
			depth /* sz */,
			verbose_level - 1);
#endif



	


	if (Descr->epsilon == 1 && Descr->d == 6) {
		if (f_v) {
			cout << "allocating Klein correspondence" << endl;
			}
		K = NEW_OBJECT(geometry::klein_correspondence);

		if (f_v) {
			cout << "before K->init" << endl;
		}
		int i, j, c, fxy;
		int B[8];
		int pivots[2] = {2,3};

		K->init(LG->F, O, verbose_level);
		color_table = NEW_int(N);
		nb_colors = Gg.nb_AG_elements(2, LG->F->q);
		O->Hyperbolic_pair->unrank_point(u, 1, 0, 0);
		for (i = 0; i < N; i++) {
			O->Hyperbolic_pair->unrank_point(v, 1, i, 0);
			fxy = O->evaluate_bilinear_form(u, v, 1);
			if (i && fxy != 0) {
				j = K->point_on_quadric_to_line(i, 0 /* verbose_level */);
				K->P3->Grass_lines->unrank_lint_here(B, j, 0 /* verbose_level */);
				LG->F->Linear_algebra->Gauss_int_with_given_pivots(B,
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
				c = Gg.AG_element_rank(LG->F->q, B, 1, 2);
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
			Int_vec_print(cout, v, Descr->d);

			O->Hyperbolic_pair->unrank_point(v, 1, i, 0);
			fxy = O->evaluate_bilinear_form(u, v, 1);
			if (i && fxy != 0) {
				j = K->point_on_quadric_to_line(i, 0 /* verbose_level */);
				K->P3->Grass_lines->unrank_lint_here(B, j, 0 /* verbose_level */);
				LG->F->Linear_algebra->Gauss_int_with_given_pivots(B,
					FALSE /* f_special */,
					TRUE /* f_complete */,
					pivots,
					2 /*nb_pivots*/,
					2 /*m*/, 4 /* n*/,
					0 /*verbose_level*/);
				cout << " : " << endl;
				Int_matrix_print(B, 2, 4);
			}
			cout << " : " << color_table[i] << endl;
		}

	}
	if (f_v) {
		cout << "init finished" << endl;
	}
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
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->Hyperbolic_pair->unrank_point(u, 1, candidates[i],
						0/*verbose_level - 4*/);
				cout << "candidate " << i << "="
						<< candidates[i] << ": ";
				Int_vec_print(cout, u, Descr->d);
				cout << endl;
				}
			}
		}
	for (i = 0; i < len; i++) {
		O->Hyperbolic_pair->unrank_point(Pts + i * Descr->d, 1, S[i], 0/*verbose_level - 4*/);
		}
	for (i = 0; i < nb_candidates; i++) {
		O->Hyperbolic_pair->unrank_point(Candidates + i * Descr->d, 1, candidates[i],
				0/*verbose_level - 4*/);
		}

	if (len == 0) {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
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

			v1 = Pts + (len - 1) * Descr->d;
			v2 = Candidates + j * Descr->d;


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
			O->Hyperbolic_pair->unrank_point(u, 1, S[i], 0);
			Int_vec_print(ost, u, Descr->d - 1);
			ost << endl;
			}
		}
}

void ovoid_classify::make_graphs(orbiter_kernel_system::orbiter_data_file *ODF,
		std::string &prefix,
		int f_split, int split_r, int split_m,
		int f_lexorder_test,
		const char *fname_mask,
		int verbose_level)
{
	int orbit_idx;
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);
	string fname_graph;
	char str[1000];
	int level;
	orbiter_kernel_system::file_io Fio;

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
		Lint_vec_print(cout, ODF->sets[orbit_idx],
				ODF->set_sizes[orbit_idx]);
		cout << " : " << ODF->Ago_ascii[orbit_idx]
				<< " : " << ODF->Aut_ascii[orbit_idx] << endl;

		snprintf(str, sizeof(str), fname_mask, orbit_idx);
		fname_graph.assign(str);

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
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;



		if (strcmp(ODF->Ago_ascii[orbit_idx], "1") != 0) {


			int max_starter;


			groups::strong_generators *SG;
			ring_theory::longinteger_object go;

			SG = NEW_OBJECT(groups::strong_generators);
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







		graph_theory::colored_graph *CG;

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

void ovoid_classify::make_one_graph(orbiter_kernel_system::orbiter_data_file *ODF,
		std::string &prefix,
	int orbit_idx,
	int f_lexorder_test,
	graph_theory::colored_graph *&CG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int level;
	orbiter_kernel_system::file_io Fio;

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


		groups::strong_generators *SG;
		ring_theory::longinteger_object go;

		SG = NEW_OBJECT(groups::strong_generators);
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

void ovoid_classify::create_graph(orbiter_kernel_system::orbiter_data_file *ODF,
	int orbit_idx,
	long int *candidates, int nb_candidates,
	graph_theory::colored_graph *&CG,
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

	data_structures::bitvector *Bitvec;

	Pts = NEW_int(nb_points * Descr->d);
	for (i = 0; i < nb_points; i++) {
		O->Hyperbolic_pair->unrank_point(Pts + i * Descr->d, 1, candidates[i], 0);
	}

	L = ((long int) nb_points * ((long int) nb_points - 1)) >> 1;

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);

	k = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++, k++) {
			fxy = O->evaluate_bilinear_form(Pts + i * Descr->d, Pts + j * Descr->d, 1);
			if (fxy != 0) {
				Bitvec->m_i(k, 1);
			}
		}
	}

	point_color = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		point_color[i] = 0;
	}

	if (Descr->epsilon == 1 && Descr->d == 6) {
		compute_coloring(ODF->sets[orbit_idx], starter_size,
				candidates, nb_points, point_color,
				nb_colors_used, verbose_level);
		// check if coloring is proper:
		k = 0;
		for (i = 0; i < nb_points; i++) {
			for (j = i + 1; j < nb_points; j++, k++) {
				if (Bitvec->s_i(k)) {
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

	char str[1000];

	snprintf(str, sizeof(str), "graph_ovoid_%d_%d_%d",
			LG->F->q, starter_size, orbit_idx);

	string label, label_tex;
	label.assign(str);
	label_tex.assign(str);

	CG = NEW_OBJECT(graph_theory::colored_graph);

	CG->init(nb_points, nb_colors_used, 1,
		point_color,
		Bitvec, TRUE /* f_ownership_of_bitvec */,
		label, label_tex,
		verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	Lint_vec_copy(candidates, CG->points, nb_candidates);
	CG->init_user_data(ODF->sets[orbit_idx],
			starter_size, verbose_level - 2);

	CG->fname_base.assign(label);

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
	data_structures::sorting Sorting;

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
	Lint_vec_print(cout, starter, starter_size);
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
	Int_vec_print(cout, colors, starter_size - 1);
	cout << endl;
	nb_colors_used = nb_colors - (starter_size - 1);
	Int_vec_complement(colors, nb_colors, starter_size - 1);
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


static void ovoid_classify_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	ovoid_classify *Gen = (ovoid_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ovoid_classify_early_test_func_callback for set ";
		Lint_vec_print(cout, S, len);
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

static void callback_ovoid_print_set(ostream &ost, int len, long int *S, void *data)
{
	ovoid_classify *Gen = (ovoid_classify *) data;

	//print_vector(ost, S, len);
	Gen->print(ost, S, len);
}




}}}

