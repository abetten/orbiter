// graph_generator.C
// 
// Anton Betten
// Nov 15 2007
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;


#include "graph.h"


void graph_generator_test_function(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
void graph_generator_print_set(ostream &ost,
		int len, int *S, void *data);

graph_generator::graph_generator()
{
	Poset = NULL;
	gen = NULL;
	A_base = NULL;
	A_on_edges = NULL;
	
	adjacency = NULL;
	degree_sequence = NULL;
	neighbor = NULL;
	neighbor_idx = NULL;
	distance = NULL;
	
	f_n = FALSE;
	f_regular = FALSE;
	f_girth = FALSE;
	f_tournament = FALSE;
	f_no_superking = FALSE;
	
	f_list = FALSE;
	f_list_all = FALSE;
	f_draw_level_graph = FALSE;
	f_draw_graphs = FALSE;
	f_draw_graphs_at_level = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;
	f_x_stretch = FALSE;
	x_stretch = 0.4;

	scale = 0.2;

	f_depth = FALSE;
	S1 = NULL;
	
	f_test_multi_edge = FALSE;
	f_draw_poset = FALSE;
	f_draw_full_poset = FALSE;
	f_plesken = FALSE;
	f_identify = FALSE;

	regularity = 0;
	girth = 0;
	n = 0;
	depth = 0;
	level_graph_level = 0;
	n2 = 0;
	level = 0;
	identify_data_sz = 0;
}

graph_generator::~graph_generator()
{
	if (A_base) {
		FREE_OBJECT(A_base);
	}
	if (A_on_edges) {
		FREE_OBJECT(A_on_edges);
	}
	if (gen) {
		FREE_OBJECT(gen);
	}
	if (adjacency) {
		FREE_int(adjacency);
	}
	if (degree_sequence) {
		FREE_int(degree_sequence);
		}
	if (neighbor) {
		FREE_int(neighbor);
		}
	if (neighbor_idx) {
		FREE_int(neighbor_idx);
		}
	if (distance) {
		FREE_int(distance);
		}
	if (S1) {
		FREE_int(S1);
		}
	
}

void graph_generator::read_arguments(int argc, const char **argv)
{
	int i;
	
	if (argc < 1) {
		usage(argc, argv);
		exit(1);
		}
	//for (i = 1; i < argc; i++) {
		//printf("%s\n", argv[i]);
		//}

	gen->read_arguments(argc, argv, 0);

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-regular") == 0) {
			f_regular = TRUE;
			sscanf(argv[++i], "%d", &regularity);
			cout << "-regular " << regularity << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			sscanf(argv[++i], "%d", &n);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-girth") == 0) {
			f_girth = TRUE;
			sscanf(argv[++i], "%d", &girth);
			cout << "-girth " << girth << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list " << endl;
			}
		else if (strcmp(argv[i], "-list_all") == 0) {
			f_list_all = TRUE;
			cout << "-list_all " << endl;
			}
		else if (strcmp(argv[i], "-draw_graphs") == 0) {
			f_draw_graphs = TRUE;
			cout << "-draw_graphs " << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-draw_graphs_at_level") == 0) {
			f_draw_graphs_at_level = TRUE;
			level = atoi(argv[++i]);
			cout << "-draw_graphs_at_level " << level << endl;
			}
		else if (strcmp(argv[i], "-scale") == 0) {
			sscanf(argv[++i], "%lf", &scale);
			cout << "-scale " << scale << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-tournament") == 0) {
			f_tournament = TRUE;
			cout << "-tournament " << endl;
			}
		else if (strcmp(argv[i], "-no_superking") == 0) {
			f_no_superking = TRUE;
			cout << "-no_superking " << endl;
			}
		else if (strcmp(argv[i], "-test_multi_edge") == 0) {
			f_test_multi_edge = TRUE;
			cout << "-test_multi_edge " << endl;
			}
		else if (strcmp(argv[i], "-draw_level_graph") == 0) {
			f_draw_level_graph = TRUE;
			sscanf(argv[++i], "%d", &level_graph_level);
			cout << "-draw_level_graph " << level_graph_level << endl;
			}
		else if (strcmp(argv[i], "-plesken") == 0) {
			f_plesken = TRUE;
			cout << "-plesken" << endl;
			}
		else if (strcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = TRUE;
			cout << "-draw_full_poset" << endl;
			}
		else if (strcmp(argv[i], "-identify") == 0) {
			int a, j;
			
			f_identify = TRUE;
			j = 0;
			while (TRUE) {
				a = atoi(argv[++i]);
				if (a == -1) {
					break;
					}
				identify_data[j++] = a;
				}
			identify_data_sz = j;
			cout << "-identify ";
			int_vec_print(cout, identify_data, identify_data_sz);
			cout << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			sscanf(argv[++i], "%d", &depth);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-x_stretch") == 0) {
			f_x_stretch = TRUE;
			sscanf(argv[++i], "%lf", &x_stretch);
			cout << "-x_stretch " << endl;
			}
		}
	if (!f_n) {
		cout << "please use option -n <n> "
				"to specify the number of vertices" << endl;
		exit(1);
		}
}

void graph_generator::init(int argc, const char **argv)
{
	int N;
	int target_depth;
	char prefix[1000];
	combinatorics_domain Combi;
	
	A_base = NEW_OBJECT(action);
	A_on_edges = NEW_OBJECT(action);
	gen = NEW_OBJECT(poset_classification);
	
	read_arguments(argc, argv);
	
	int verbose_level = gen->verbose_level;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_generator::init" << endl;
		}

	if (f_tournament) {
		if (f_v) {
			cout << "graph_generator::init tournaments "
					"on " << n << " vertices" << endl;
			}
		sprintf(prefix, "tournament_%d", n);
		if (f_no_superking) {
			sprintf(prefix + strlen(prefix), "_no_superking");
			}
		}
	else {
		if (f_v) {
			cout << "graph_generator::init graphs "
					"on " << n << " vertices" << endl;
			}
		sprintf(prefix, "graph_%d", n);
		}
	

	
	if (f_regular) {
		sprintf(prefix + strlen(prefix), "_r%d", regularity);
		}
	
	if (f_girth) {
		sprintf(prefix + strlen(prefix), "_g%d", girth);
		}

	if (f_v) {
		cout << "prefix=" << prefix << endl;
		}
	
	
	n2 = Combi.int_n_choose_k(n, 2);
	if (f_v) {
		cout << "n2=" << n2 << endl;
		}

	S1 = NEW_int(n2);

	A_base->init_symmetric_group(n, verbose_level - 3);
	if (f_v) {
		cout << "A_base->init_symmetric_group done" << endl;
		}
	
	if (!A_base->f_has_sims) {
		cout << "!A_base->f_has_sims" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "generators for the symmetric group are:" << endl;
		A_base->Sims->print_generators();
		}

	if (f_tournament) {
		A_on_edges->induced_action_on_ordered_pairs(
				*A_base, A_base->Sims, verbose_level - 3);
		if (f_v) {
			cout << "A_on_edges->induced_action_on_ordered_pairs "
					"done, created the following action:" << endl;
			A_on_edges->print_info();
			cout << "generators for the symmetric group in the "
					"action on ordered_pairs are:" << endl;
			A_on_edges->Sims->print_generators();
			}
		}
	else {
		A_on_edges->induced_action_on_pairs(
				*A_base, A_base->Sims, verbose_level - 3);
		if (f_v) {
			cout << "A_on_edges->induced_action_on_pairs done, "
					"created the following action:" << endl;
			A_on_edges->print_info();
			cout << "generators for the symmetric group in the action "
					"on pairs are:" << endl;
			A_on_edges->Sims->print_generators();
			}
		}
	A_on_edges->lex_least_base_in_place(verbose_level - 3);
	if (f_v) {
		cout << "After lex_least_base, we have the following "
				"action:" << endl;
		A_on_edges->print_info();
		cout << "generators for the symmetric group in the "
				"induced action are:" << endl;
		A_on_edges->Sims->print_generators();
		}

	
	adjacency = NEW_int(n * n);

	if (f_tournament) {
		target_depth = n2;
		}
	if (f_regular) {
		degree_sequence = NEW_int(n);
		N = n * regularity;
		if (ODD(N)) {
			cout << "n * regularity must be even" << endl;
			exit(1);
			}
		N >>= 1;
		target_depth = N;
		}
	else {
		degree_sequence = NULL;
		target_depth = n2;
		}
	if (f_depth) {
		target_depth = depth;
		}
	if (f_girth) {
		neighbor = NEW_int(n);
		neighbor_idx = NEW_int(n);
		distance = NEW_int(n);
		}
	else {
		neighbor = NULL;
		neighbor_idx = NULL;
		distance = NULL;
		}
	
	
	if (f_v) {
		cout << "graph_generator::init target_depth = "
				<< target_depth << endl;
		}


	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A_base, A_on_edges,
			A_base->Strong_gens,
			verbose_level);

	gen->initialize(Poset,
		target_depth, 
		"", prefix, verbose_level - 1);
	
	Poset->add_testing_without_group(
			graph_generator_test_function,
			this,
			verbose_level);

	
	gen->f_print_function = TRUE;
	gen->print_function = graph_generator_print_set;
	gen->print_function_data = (void *) this;

	if (f_v) {
		cout << "graph_generator::init done" << endl;
		}


}

int graph_generator::check_conditions(int len,
		int *S, int verbose_level)
{
	//verbose_level = 2;

	int f_OK = TRUE;
	int f_not_regular = FALSE;
	int f_bad_girth = FALSE;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "graph_generator::check_conditions checking set ";
		print_set(cout, len, S);
		}
	if (f_regular && !check_regularity(S, len, verbose_level - 1)) {
		f_not_regular = TRUE;
		f_OK = FALSE;
		}
	if (f_OK) {
		if (f_girth && !girth_check(S, len, verbose_level - 1)) {
			f_bad_girth = TRUE;
			f_OK = FALSE;
			}
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
			if (f_not_regular) {
				cout << "regularity test";
				}
			if (f_bad_girth) {
				cout << "girth test";
				}
			cout << endl;
			}
		return FALSE;
		}
}

int graph_generator::check_conditions_tournament(
		int len, int *S,
		int verbose_level)
{
	//verbose_level = 2;


	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, a2, swap, swap2, b2, b, i, idx;
	int *S_sorted;
	combinatorics_domain Combi;
	sorting Sorting;
	
	if (f_v) {
		cout << "graph_generator::check_conditions_tournament "
				"checking set ";
		print_set(cout, len, S);
		}

	S_sorted = NEW_int(len);
	int_vec_copy(S, S_sorted, len);
	Sorting.int_vec_heapsort(S_sorted, len);

	for (i = 0; i < len; i++) {
		a = S_sorted[i];
		swap = a % 2;
		a2 = a / 2;
		swap2 = 1 - swap;
		b2 = a2;
		b = 2 * b2 + swap2;
		if (Sorting.int_vec_search(S_sorted, len, b, idx)) {
			if (f_vv) {
				cout << "graph_generator::check_conditions_tournament "
						"elements " << a << " and " << b
						<< " cannot both exist" << endl;
				}
			f_OK = FALSE;
			break;
			}
		}


	if (f_OK && f_no_superking) {
		int *score;
		int u, v;

		score = NEW_int(n);
		int_vec_zero(score, n);
		for (i = 0; i < len && f_OK; i++) {
			a = S_sorted[i];
			swap = a % 2;
			a2 = a / 2;
			Combi.k2ij(a2, u, v, n);
			if (swap) {
				score[v]++;
				if (score[v] == n - 1) {
					f_OK = FALSE;
					}
				}
			else {
				score[u]++;
				if (score[u] == n - 1) {
					f_OK = FALSE;
					}
				}
			}

		FREE_int(score);
		}
	FREE_int(S_sorted);

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
		return FALSE;
		}
}


int graph_generator::check_regularity(
		int *S, int len,
		int verbose_level)
{
	int f_OK;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "check_regularity for ";
		int_vec_print(cout, S, len);
		cout << endl;
		}
	f_OK = compute_degree_sequence(S, len);
	if (f_v) {
		if (!f_OK) {
			cout << "regularity test violated" << endl;
			}
		else {
			cout << "regularity test OK" << endl;
			}
		}
	return f_OK;
}


int graph_generator::compute_degree_sequence(int *S, int len)
{
	int h, a, i, j;
	combinatorics_domain Combi;
	
	if (f_tournament) {
		cout << "graph_generator::compute_degree_sequence "
				"tournament is TRUE" << endl;
		exit(1);
		}
	int_vec_zero(degree_sequence, n);
	for (h = 0; h < len; h++) {
		a = S[h];
		Combi.k2ij(a, i, j, n);
		degree_sequence[i]++;
		if (degree_sequence[i] > regularity) {
			return FALSE;
			}
		degree_sequence[j]++;
		if (degree_sequence[j] > regularity) {
			return FALSE;
			}
		}
	return TRUE;
}

int graph_generator::girth_check(int *line, int len,
		int verbose_level)
{
	int f_OK = TRUE, i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "girth check for ";
		int_vec_print(cout, line, len);
		cout << endl;
		}
	for (i = 0; i < n; i++) {
		if (!girth_test_vertex(line, len, i,
				girth, verbose_level - 2)) {
			f_OK = FALSE;
			if (f_vv) {
				cout << "girth check fails for vertex " << i << endl;
				}
			break;
			}
		}
	if (f_v) {
		if (!f_OK) {
			cout << "girth check fails" << endl;
			}
		else {
			cout << "girth check OK" << endl;
			}
		}
	return f_OK;
}

int graph_generator::girth_test_vertex(int *S, int len,
		int vertex, int girth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l, i, cur = 0, a, b, da, db, g;
	
	get_adjacency(S, len, verbose_level - 1);
	for (i = 0; i < n; i++) {
		neighbor_idx[i] = -1;
		}
	neighbor[0] = vertex;
	distance[vertex] = 0;
	neighbor_idx[vertex] = 0;
	l = 1;
	while (cur < l) {
		a = neighbor[cur];
		da = distance[a];
		for (b = 0; b < n; b++) {
			if (adjacency[a * n + b]) {
				if (neighbor_idx[b] >= 0) {
					db = distance[b];
					g = da + 1 + db;
					if (g < girth) {
						if (f_v) {
							cout << "found a cycle of length "
									<< g << " < " << girth << endl;
							cout << vertex << " - " << a
									<< " - " << b << endl;
							cout << da << " + " << 1
									<< " + " << db << endl;
							}
						return FALSE;
						}
					else {
						if (da + 1 < db) {
							cout << "da + 1 < db, this "
									"should not happen" << endl;
							cout << "vertex=" << vertex << endl;
							cout << "a=" << a << endl;
							cout << "b=" << b << endl;
							cout << "da=" << da << endl;
							cout << "db=" << db << endl;
							exit(1);
							}
						}
					}
				else {
					neighbor[l] = b;
					distance[b] = da + 1;
					neighbor_idx[b] = l;
					l++;
					}
				}
			adjacency[a * n + b] = 0;
			adjacency[b * n + a] = 0;
			}
		cur++;
		}
	return TRUE;
}

void graph_generator::get_adjacency(int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, a;
	combinatorics_domain Combi;
	
	int_vec_zero(adjacency, n * n);

	if (f_tournament) {
		int swap, a2;
		
		for (h = 0; h < len; h++) {
			a = S[h];
			swap = a % 2;
			a2 = a / 2;
			Combi.k2ij(a2, i, j, n);
			if (!swap) {
				adjacency[i * n + j] = 1;
				adjacency[j * n + i] = 0;
				}
			else {
				adjacency[i * n + j] = 0;
				adjacency[j * n + i] = 1;
				}
			}
		}
	else {
		for (h = 0; h < len; h++) {
			a = S[h];
			Combi.k2ij(a, i, j, n);
			adjacency[i * n + j] = 1;
			adjacency[j * n + i] = 1;
			}
		}
	if (f_v) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				cout << adjacency[i * n + j];
				}
			cout << endl;
			}
		}
}

void graph_generator::print(ostream &ost, int *S, int len)
{
	int i, j;
	
	ost << "graph_generator::print" << endl;
	
	for (i = 0; i < len; i++) {
		ost << S[i] << " ";
		}
	ost << endl;
	get_adjacency(S, len, 0);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			ost << setw(2) << adjacency[i * n + j];
			}
		ost << endl;
		}
	
}

void graph_generator::print_score_sequences(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, nb_orbits;
	int *set;
	int *score;

	if (f_v) {
		cout << "graph_generator::print_score_sequences "
				"level = " << level << endl;
		}

	set = NEW_int(level);
	score = NEW_int(n);
	nb_orbits = gen->nb_orbits_at_level(level);
	for (h = 0; h < nb_orbits; h++) {
		strong_generators *Strong_gens;
		longinteger_object go;

		gen->get_set_by_level(level, h, set);
		gen->get_stabilizer_generators(Strong_gens,  
			level, h, 0 /* verbose_level*/);

		Strong_gens->group_order(go);


		cout << h << " : ";
		int_vec_print(cout, set, level);
		cout << " : " << go << " : ";
		
		score_sequence(n, set, level, score, verbose_level - 1);

		int_vec_print(cout, score, n);
		cout << endl;

		delete Strong_gens;
		}

	FREE_int(set);
	FREE_int(score);

}

void graph_generator::score_sequence(int n,
		int *set, int sz, int *score, int verbose_level)
{
	int i, a, swap, a2, u, v;
	combinatorics_domain Combi;

	int_vec_zero(score, n);
	for (i = 0; i < sz; i++) {
		a = set[i];



		swap = a % 2;
		a2 = a / 2;
		Combi.k2ij(a2, u, v, n);

		if (swap) {
			// edge from v to u
			score[v]++;
			}
		else {
			// edge from u to v
			score[u]++;
			}
		}

}


void graph_generator::draw_graphs(int level,
	double scale, int xmax_in, int ymax_in,
	int xmax, int ymax, int f_embedded, int f_sideways,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, nb_orbits;
	int *set;
	int *v;

	if (f_v) {
		cout << "graph_generator::draw_graphs "
				"level = " << level << endl;
		}

	set = NEW_int(level);
	v = NEW_int(n2);
	nb_orbits = gen->nb_orbits_at_level(level);
	for (h = 0; h < nb_orbits; h++) {
		strong_generators *Strong_gens;
		longinteger_object go;

		gen->get_set_by_level(level, h, set);
		gen->get_stabilizer_generators(Strong_gens,  
			level, h, 0 /* verbose_level*/);

		Strong_gens->group_order(go);
		
		int_vec_zero(v, n2);
		for (i = 0; i < level; i++) {
			v[set[i]] = 1;
			}

		cout << h << " : ";
		int_vec_print(cout, set, level);
		cout << " : ";
		for (i = 0; i < n2; i++) {
			cout << v[i];
			}
		cout << " : " << go << endl;


		char fname_full[1000];

		sprintf(fname_full, "%s_rep_%d_%d.mp",
				gen->fname_base, level, h);
		int x_min = 0, x_max = xmax_in;
		int y_min = 0, y_max = ymax_in;
		int x, y, dx, dy;

		x = (x_max - x_min) >> 1;
		y = (y_max - y_min) >> 1;
		dx = x;
		dy = y;
		{
		mp_graphics G(fname_full,
				x_min, y_min, x_max, y_max, f_embedded, f_sideways);
		G.out_xmin() = 0;
		G.out_ymin() = 0;
		G.out_xmax() = xmax;
		G.out_ymax() = ymax;
		//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
		G.set_scale(scale);
		G.header();
		G.begin_figure(1000 /*factor_1000*/);

		G.sl_thickness(10); // 100 is normal
		//G.frame(0.05);


		if (f_tournament) {
			G.draw_tournament(x, y, dx, dy, n, set, level, 0);
			}
		else {
			G.draw_graph(x, y, dx, dy, n, set, level);
			}
		
		G.end_figure();
		G.footer();
		}
		cout << "written file " << fname_full
				<< " of size " << file_size(fname_full) << endl;

		delete Strong_gens;
		}

	FREE_int(set);
}


// #############################################################################
// global functions
// #############################################################################


void graph_generator_test_function(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level)
{
	graph_generator *Gen = (graph_generator *) data;
	int i, f_OK;

	int_vec_copy(S, Gen->S1, len);
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		Gen->S1[len] = candidates[i];
		if (Gen->f_tournament) {
			f_OK = Gen->check_conditions_tournament(
					len + 1, Gen->S1, verbose_level);
		}
		else {
			f_OK = Gen->check_conditions(len + 1, Gen->S1, verbose_level);
		}
		if (f_OK) {
			good_candidates[nb_good_candidates++] = candidates[i];
		}
	}
}

void graph_generator_print_set(ostream &ost,
		int len, int *S, void *data)
{
	graph_generator *Gen = (graph_generator *) data;
	
	//print_vector(ost, S, len);
	Gen->print(ost, S, len);
}



