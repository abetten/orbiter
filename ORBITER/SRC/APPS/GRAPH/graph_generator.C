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
#include "graph.h"

graph_generator::graph_generator()
{
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
	
	f_test_multi_edge = FALSE;
	f_draw_poset = FALSE;
	f_draw_full_poset = FALSE;
	f_plesken = FALSE;
	f_identify = FALSE;
}

graph_generator::~graph_generator()
{
	if (A_base)
		delete A_base;
	if (A_on_edges)
		delete A_on_edges;
	if (gen)
		delete gen;
	if (adjacency)
		FREE_INT(adjacency);
	if (degree_sequence) {
		FREE_INT(degree_sequence);
		}
	if (neighbor) {
		FREE_INT(neighbor);
		}
	if (neighbor_idx) {
		FREE_INT(neighbor_idx);
		}
	if (distance) {
		FREE_INT(distance);
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
			sscanf(argv[++i], "%ld", &regularity);
			cout << "-regular " << regularity << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			sscanf(argv[++i], "%ld", &n);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-girth") == 0) {
			f_girth = TRUE;
			sscanf(argv[++i], "%ld", &girth);
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
			sscanf(argv[++i], "%ld", &level_graph_level);
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
			INT a, j;
			
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
			INT_vec_print(cout, identify_data, identify_data_sz);
			cout << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			sscanf(argv[++i], "%ld", &depth);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-x_stretch") == 0) {
			f_x_stretch = TRUE;
			sscanf(argv[++i], "%lf", &x_stretch);
			cout << "-x_stretch " << endl;
			}
		}
	if (!f_n) {
		cout << "please use option -n <n> to specify the number of vertices" << endl;
		exit(1);
		}
}

void graph_generator::init(int argc, const char **argv)
{
	INT N;
	INT target_depth;
	BYTE prefix[1000];
	
	A_base = new action;
	A_on_edges = new action;
	gen = new generator;
	
	read_arguments(argc, argv);
	
	INT verbose_level = gen->verbose_level;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_generator::init" << endl;
		}

	if (f_tournament) {
		if (f_v) {
			cout << "graph_generator::init tournaments on " << n << " vertices" << endl;
			}
		sprintf(prefix, "tournament_%ld", n);
		if (f_no_superking) {
			sprintf(prefix + strlen(prefix), "_no_superking");
			}
		}
	else {
		if (f_v) {
			cout << "graph_generator::init graphs on " << n << " vertices" << endl;
			}
		sprintf(prefix, "graph_%ld", n);
		}
	

	
	if (f_regular) {
		sprintf(prefix + strlen(prefix), "_r%ld", regularity);
		}
	
	if (f_girth) {
		sprintf(prefix + strlen(prefix), "_g%ld", girth);
		}

	if (f_v) {
		cout << "prefix=" << prefix << endl;
		}
	
	
	n2 = INT_n_choose_k(n, 2);	
	if (f_v) {
		cout << "n2=" << n2 << endl;
		}

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
		A_on_edges->induced_action_on_ordered_pairs(*A_base, A_base->Sims, verbose_level - 3);	
		if (f_v) {
			cout << "A_on_edges->induced_action_on_ordered_pairs done, created the following action:" << endl;
			A_on_edges->print_info();
			cout << "generators for the symmetric group in the action on ordered_pairs are:" << endl;
			A_on_edges->Sims->print_generators();
			}
		}
	else {
		A_on_edges->induced_action_on_pairs(*A_base, A_base->Sims, verbose_level - 3);	
		if (f_v) {
			cout << "A_on_edges->induced_action_on_pairs done, created the following action:" << endl;
			A_on_edges->print_info();
			cout << "generators for the symmetric group in the action on pairs are:" << endl;
			A_on_edges->Sims->print_generators();
			}
		}
	A_on_edges->lex_least_base_in_place(verbose_level - 3);
	if (f_v) {
		cout << "After lex_least_base, we have the following action:" << endl;
		A_on_edges->print_info();
		cout << "generators for the symmetric group in the induced action are:" << endl;
		A_on_edges->Sims->print_generators();
		}

	
	adjacency = NEW_INT(n * n);

	if (f_tournament) {
		target_depth = n2;
		}
	if (f_regular) {
		degree_sequence = NEW_INT(n);
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
		neighbor = NEW_INT(n);
		neighbor_idx = NEW_INT(n);
		distance = NEW_INT(n);
		}
	else {
		neighbor = NULL;
		neighbor_idx = NULL;
		distance = NULL;
		}
	
	
	if (f_v) {
		cout << "graph_generator::init target_depth = " << target_depth << endl;
		}

	gen->initialize(A_base, A_on_edges,  
		A_base->Strong_gens, 
		target_depth, 
		"", prefix, verbose_level - 1);
	

	gen->init_check_func(::check_conditions, 
		(void *)this /* candidate_check_data */);

	
	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = (void *) this;

	if (f_v) {
		cout << "graph_generator::init done" << endl;
		}


}

INT graph_generator::check_conditions(INT len, INT *S, INT verbose_level)
{
	//verbose_level = 2;

	INT f_OK = TRUE;
	INT f_not_regular = FALSE;
	INT f_bad_girth = FALSE;
	INT f_v = (verbose_level >= 1);
	
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

INT graph_generator::check_conditions_tournament(INT len, INT *S, INT verbose_level)
{
	//verbose_level = 2;


	INT f_OK = TRUE;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT a, a2, swap, swap2, b2, b, i, idx;
	INT *S_sorted;
	
	if (f_v) {
		cout << "graph_generator::check_conditions_tournament checking set ";
		print_set(cout, len, S);
		}

	S_sorted = NEW_INT(len);
	INT_vec_copy(S, S_sorted, len);
	INT_vec_heapsort(S_sorted, len);

	for (i = 0; i < len; i++) {
		a = S_sorted[i];
		swap = a % 2;
		a2 = a / 2;
		swap2 = 1 - swap;
		b2 = a2;
		b = 2 * b2 + swap2;
		if (INT_vec_search(S_sorted, len, b, idx)) {
			if (f_vv) {
				cout << "graph_generator::check_conditions_tournament elements " << a << " and " << b << " cannot both exist" << endl;
				}
			f_OK = FALSE;
			break;
			}
		}


	if (f_OK && f_no_superking) {
		INT *score;
		INT u, v;

		score = NEW_INT(n);
		INT_vec_zero(score, n);
		for (i = 0; i < len && f_OK; i++) {
			a = S_sorted[i];
			swap = a % 2;
			a2 = a / 2;
			k2ij(a2, u, v, n);
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

		FREE_INT(score);
		}
	FREE_INT(S_sorted);

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


INT graph_generator::check_regularity(INT *S, INT len, INT verbose_level)
{
	INT f_OK;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "check_regularity for ";
		INT_vec_print(cout, S, len);
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


INT graph_generator::compute_degree_sequence(INT *S, INT len)
{
	INT h, a, i, j;
	
	if (f_tournament) {
		cout << "graph_generator::compute_degree_sequence tournament is TRUE" << endl;
		exit(1);
		}
	INT_vec_zero(degree_sequence, n);
	for (h = 0; h < len; h++) {
		a = S[h];
		k2ij(a, i, j, n);
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

INT graph_generator::girth_check(INT *line, INT len, INT verbose_level)
{
	INT f_OK = TRUE, i;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "girth check for ";
		INT_vec_print(cout, line, len);
		cout << endl;
		}
	for (i = 0; i < n; i++) {
		if (!girth_test_vertex(line, len, i, girth, verbose_level - 2)) {
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

INT graph_generator::girth_test_vertex(INT *S, INT len, INT vertex, INT girth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT l, i, cur = 0, a, b, da, db, g;
	
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
							cout << "found a cycle of length " << g << " < " << girth << endl;
							cout << vertex << " - " << a << " - " << b << endl;
							cout << da << " + " << 1 << " + " << db << endl;
							}
						return FALSE;
						}
					else {
						if (da + 1 < db) {
							cout << "da + 1 < db, this should not happen" << endl;
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

void graph_generator::get_adjacency(INT *S, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h, i, j, a;
	
	INT_vec_zero(adjacency, n * n);

	if (f_tournament) {
		INT swap, a2;
		
		for (h = 0; h < len; h++) {
			a = S[h];
			swap = a % 2;
			a2 = a / 2;
			k2ij(a2, i, j, n);
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
			k2ij(a, i, j, n);
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

void graph_generator::print(INT *S, INT len)
{
	INT i, j;
	
	cout << "graph_generator::print" << endl;
	
	for (i = 0; i < len; i++) {
		cout << S[i] << " ";
		}
	cout << endl;
	get_adjacency(S, len, 0);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			cout << setw(2) << adjacency[i * n + j];
			}
		cout << endl;
		}
	
}

void graph_generator::print_score_sequences(INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h, nb_orbits;
	INT *set;
	INT *score;

	if (f_v) {
		cout << "graph_generator::print_score_sequences level = " << level << endl;
		}

	set = NEW_INT(level);
	score = NEW_INT(n);
	nb_orbits = gen->nb_orbits_at_level(level);
	for (h = 0; h < nb_orbits; h++) {
		strong_generators *Strong_gens;
		longinteger_object go;

		gen->get_set_by_level(level, h, set);
		gen->get_stabilizer_generators(Strong_gens,  
			level, h, 0 /* verbose_level*/);

		Strong_gens->group_order(go);


		cout << h << " : ";
		INT_vec_print(cout, set, level);
		cout << " : " << go << " : ";
		
		score_sequence(n, set, level, score, verbose_level - 1);

		INT_vec_print(cout, score, n);
		cout << endl;

		delete Strong_gens;
		}

	FREE_INT(set);
	FREE_INT(score);

}

void graph_generator::score_sequence(INT n, INT *set, INT sz, INT *score, INT verbose_level)
{
	INT i, a, swap, a2, u, v;

	INT_vec_zero(score, n);
	for (i = 0; i < sz; i++) {
		a = set[i];



		swap = a % 2;
		a2 = a / 2;
		k2ij(a2, u, v, n);

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


void graph_generator::draw_graphs(INT level, double scale, INT xmax_in, INT ymax_in, 
	INT xmax, INT ymax, INT f_embedded, INT f_sideways, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h, i, nb_orbits;
	INT *set;
	INT *v;

	if (f_v) {
		cout << "graph_generator::draw_graphs level = " << level << endl;
		}

	set = NEW_INT(level);
	v = NEW_INT(n2);
	nb_orbits = gen->nb_orbits_at_level(level);
	for (h = 0; h < nb_orbits; h++) {
		strong_generators *Strong_gens;
		longinteger_object go;

		gen->get_set_by_level(level, h, set);
		gen->get_stabilizer_generators(Strong_gens,  
			level, h, 0 /* verbose_level*/);

		Strong_gens->group_order(go);
		
		INT_vec_zero(v, n2);
		for (i = 0; i < level; i++) {
			v[set[i]] = 1;
			}

		cout << h << " : ";
		INT_vec_print(cout, set, level);
		cout << " : ";
		for (i = 0; i < n2; i++) {
			cout << v[i];
			}
		cout << " : " << go << endl;


		BYTE fname_full[1000];

		sprintf(fname_full, "%s_rep_%ld_%ld.mp", gen->fname_base, level, h);
		INT x_min = 0, x_max = xmax_in;
		INT y_min = 0, y_max = ymax_in;
		INT x, y, dx, dy;

		x = (x_max - x_min) >> 1;
		y = (y_max - y_min) >> 1;
		dx = x;
		dy = y;
		{
		mp_graphics G(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
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
			draw_tournament(&G, x, y, dx, dy, n, set, level, 0);
			}
		else {
			draw_graph(&G, x, y, dx, dy, n, set, level);
			}
		
		G.end_figure();
		G.footer();
		}
		cout << "written file " << fname_full << " of size " << file_size(fname_full) << endl;

		delete Strong_gens;
		}

	FREE_INT(set);
}


// ##################################################################################################
// global functions
// ##################################################################################################


INT check_conditions(INT len, INT *S, void *data, INT verbose_level)
{
	graph_generator *Gen = (graph_generator *) data;

	if (Gen->f_tournament) {
		return Gen->check_conditions_tournament(len, S, verbose_level);
		}
	else {
		return Gen->check_conditions(len, S, verbose_level);
		}
}

void print_set(INT len, INT *S, void *data)
{
	graph_generator *Gen = (graph_generator *) data;
	
	//print_vector(ost, S, len);
	Gen->print(S, len);
}



