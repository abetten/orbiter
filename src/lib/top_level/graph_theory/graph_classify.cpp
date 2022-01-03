// graph_classify.cpp
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

namespace orbiter {
namespace top_level {



graph_classify::graph_classify()
{
	Descr = NULL;

	Poset = NULL;
	gen = NULL;
	A_base = NULL;
	A_on_edges = NULL;
	
	adjacency = NULL;
	degree_sequence = NULL;
	neighbor = NULL;
	neighbor_idx = NULL;
	distance = NULL;
	
	S1 = NULL;
	
	n2 = 0;
}

graph_classify::~graph_classify()
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
		FREE_lint(S1);
		}
	
}


void graph_classify::init(graph_classify_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_classify::init" << endl;
		}

	int N;
	int target_depth;
	char prefix[1000];
	combinatorics_domain Combi;
	

	graph_classify::Descr = Descr;

	A_base = NEW_OBJECT(action);
	A_on_edges = NEW_OBJECT(action);
	gen = NEW_OBJECT(poset_classification);


	if (Descr->f_tournament) {
		if (f_v) {
			cout << "graph_classify::init tournaments "
					"on " << Descr->n << " vertices" << endl;
			}
		sprintf(prefix, "tournament_%d", Descr->n);
		if (Descr->f_no_superking) {
			sprintf(prefix + strlen(prefix), "_no_superking");
			}
		}
	else {
		if (f_v) {
			cout << "graph_classify::init graphs "
					"on " << Descr->n << " vertices" << endl;
			}
		sprintf(prefix, "graph_%d", Descr->n);
		}
	

	
	if (Descr->f_regular) {
		sprintf(prefix + strlen(prefix), "_r%d", Descr->regularity);
		}
	
	if (Descr->f_girth) {
		sprintf(prefix + strlen(prefix), "_g%d", Descr->girth);
		}

	if (f_v) {
		cout << "prefix=" << prefix << endl;
		}
	
	
	n2 = Combi.int_n_choose_k(Descr->n, 2);
	if (f_v) {
		cout << "n2=" << n2 << endl;
		}

	S1 = NEW_lint(n2);
	int f_no_base = FALSE;

	A_base->init_symmetric_group(Descr->n, f_no_base, verbose_level - 3);
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

	if (Descr->f_tournament) {
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

	
	adjacency = NEW_int(Descr->n * Descr->n);

	if (Descr->f_tournament) {
		target_depth = n2;
		}
	if (Descr->f_regular) {
		degree_sequence = NEW_int(Descr->n);
		N = Descr->n * Descr->regularity;
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
	if (Descr->f_depth) {
		target_depth = Descr->depth;
		}
	if (Descr->f_girth) {
		neighbor = NEW_int(Descr->n);
		neighbor_idx = NEW_int(Descr->n);
		distance = NEW_int(Descr->n);
		}
	else {
		neighbor = NULL;
		neighbor_idx = NULL;
		distance = NULL;
		}
	
	
	if (f_v) {
		cout << "graph_classify::init target_depth = "
				<< target_depth << endl;
	}


	if (!Descr->f_control) {
		cout << "please use -poset_classification_control ... -end" << endl;
		exit(1);
	}
	Poset = NEW_OBJECT(poset_with_group_action);
	Poset->init_subset_lattice(A_base, A_on_edges,
			A_base->Strong_gens,
			verbose_level);

	Poset->add_testing_without_group(
			graph_classify_test_function,
			this,
			verbose_level);

	
	Poset->f_print_function = TRUE;
	Poset->print_function = graph_classify_print_set;
	Poset->print_function_data = (void *) this;

	gen->initialize_and_allocate_root_node(Descr->Control, Poset,
		target_depth,
		verbose_level - 1);

	long int t0;
	os_interface Os;
	int depth;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "graph_classify::init before gen->main" << endl;
	}
	depth = gen->main(t0,
			target_depth /*schreier_depth*/,
		TRUE /*f_use_invariant_subset_if_available*/,
		FALSE /*f_debug*/,
		verbose_level);
	if (f_v) {
		cout << "graph_classify::init after gen->main" << endl;
		cout << "gen->main returns depth=" << depth << endl;
	}

	if (f_v) {
		cout << "graph_classify::init done" << endl;
	}


}

int graph_classify::check_conditions(int len,
		long int *S, int verbose_level)
{
	//verbose_level = 2;

	int f_OK = TRUE;
	int f_not_regular = FALSE;
	int f_bad_girth = FALSE;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "graph_classify::check_conditions checking set ";
		print_set(cout, len, S);
		}
	if (Descr->f_regular && !check_regularity(S, len, verbose_level - 1)) {
		f_not_regular = TRUE;
		f_OK = FALSE;
		}
	if (f_OK) {
		if (Descr->f_girth && !girth_check(S, len, verbose_level - 1)) {
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

int graph_classify::check_conditions_tournament(
		int len, long int *S,
		int verbose_level)
{
	//verbose_level = 2;


	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, a2, swap, swap2, b2, b, i, idx;
	long int *S_sorted;
	combinatorics_domain Combi;
	sorting Sorting;
	
	if (f_v) {
		cout << "graph_classify::check_conditions_tournament "
				"checking set ";
		Orbiter->Lint_vec.print(cout, S, len);
		}

	S_sorted = NEW_lint(len);
	Orbiter->Lint_vec.copy(S, S_sorted, len);
	Sorting.lint_vec_heapsort(S_sorted, len);

	for (i = 0; i < len; i++) {
		a = S_sorted[i];
		swap = a % 2;
		a2 = a / 2;
		swap2 = 1 - swap;
		b2 = a2;
		b = 2 * b2 + swap2;
		if (Sorting.lint_vec_search(S_sorted, len, b, idx, 0)) {
			if (f_vv) {
				cout << "graph_classify::check_conditions_tournament "
						"elements " << a << " and " << b
						<< " cannot both exist" << endl;
				}
			f_OK = FALSE;
			break;
			}
		}


	if (f_OK && Descr->f_no_superking) {
		int *score;
		int u, v;

		score = NEW_int(Descr->n);
		Orbiter->Int_vec.zero(score, Descr->n);
		for (i = 0; i < len && f_OK; i++) {
			a = S_sorted[i];
			swap = a % 2;
			a2 = a / 2;
			Combi.k2ij(a2, u, v, Descr->n);
			if (swap) {
				score[v]++;
				if (score[v] == Descr->n - 1) {
					f_OK = FALSE;
					}
				}
			else {
				score[u]++;
				if (score[u] == Descr->n - 1) {
					f_OK = FALSE;
					}
				}
			}

		FREE_int(score);
		}
	FREE_lint(S_sorted);

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


int graph_classify::check_regularity(
		long int *S, int len,
		int verbose_level)
{
	int f_OK;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "check_regularity for ";
		Orbiter->Lint_vec.print(cout, S, len);
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


int graph_classify::compute_degree_sequence(long int *S, int len)
{
	long int h, a, i, j;
	combinatorics_domain Combi;
	
	if (Descr->f_tournament) {
		cout << "graph_classify::compute_degree_sequence "
				"tournament is TRUE" << endl;
		exit(1);
		}
	Orbiter->Int_vec.zero(degree_sequence, Descr->n);
	for (h = 0; h < len; h++) {
		a = S[h];
		Combi.k2ij_lint(a, i, j, Descr->n);
		degree_sequence[i]++;
		if (degree_sequence[i] > Descr->regularity) {
			return FALSE;
			}
		degree_sequence[j]++;
		if (degree_sequence[j] > Descr->regularity) {
			return FALSE;
			}
		}
	return TRUE;
}

int graph_classify::girth_check(long int *line, int len,
		int verbose_level)
{
	int f_OK = TRUE, i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "girth check for ";
		Orbiter->Lint_vec.print(cout, line, len);
		cout << endl;
		}
	for (i = 0; i < Descr->n; i++) {
		if (!girth_test_vertex(line, len, i,
				Descr->girth, verbose_level - 2)) {
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

int graph_classify::girth_test_vertex(long int *S, int len,
		int vertex, int girth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l, i, cur = 0, a, b, da, db, g;
	
	get_adjacency(S, len, verbose_level - 1);
	for (i = 0; i < Descr->n; i++) {
		neighbor_idx[i] = -1;
		}
	neighbor[0] = vertex;
	distance[vertex] = 0;
	neighbor_idx[vertex] = 0;
	l = 1;
	while (cur < l) {
		a = neighbor[cur];
		da = distance[a];
		for (b = 0; b < Descr->n; b++) {
			if (adjacency[a * Descr->n + b]) {
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
			adjacency[a * Descr->n + b] = 0;
			adjacency[b * Descr->n + a] = 0;
			}
		cur++;
		}
	return TRUE;
}

void graph_classify::get_adjacency(long int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int h, i, j, a;
	combinatorics_domain Combi;
	
	Orbiter->Int_vec.zero(adjacency, Descr->n * Descr->n);

	if (Descr->f_tournament) {
		int swap, a2;
		
		for (h = 0; h < len; h++) {
			a = S[h];
			swap = a % 2;
			a2 = a / 2;
			Combi.k2ij_lint(a2, i, j, Descr->n);
			if (!swap) {
				adjacency[i * Descr->n + j] = 1;
				adjacency[j * Descr->n + i] = 0;
				}
			else {
				adjacency[i * Descr->n + j] = 0;
				adjacency[j * Descr->n + i] = 1;
				}
			}
		}
	else {
		for (h = 0; h < len; h++) {
			a = S[h];
			Combi.k2ij_lint(a, i, j, Descr->n);
			adjacency[i * Descr->n + j] = 1;
			adjacency[j * Descr->n + i] = 1;
			}
		}
	if (f_v) {
		for (i = 0; i < Descr->n; i++) {
			for (j = 0; j < Descr->n; j++) {
				cout << adjacency[i * Descr->n + j];
				}
			cout << endl;
			}
		}
}

void graph_classify::print(ostream &ost, long int *S, int len)
{
	int i, j;
	
	ost << "graph_classify::print" << endl;
	
	for (i = 0; i < len; i++) {
		ost << S[i] << " ";
		}
	ost << endl;
	get_adjacency(S, len, 0);
	for (i = 0; i < Descr->n; i++) {
		for (j = 0; j < Descr->n; j++) {
			ost << setw(2) << adjacency[i * Descr->n + j];
			}
		ost << endl;
		}
	
}

void graph_classify::print_score_sequences(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, nb_orbits;
	long int *set;
	long int *score;

	if (f_v) {
		cout << "graph_classify::print_score_sequences "
				"level = " << level << endl;
		}

	set = NEW_lint(level);
	score = NEW_lint(Descr->n);
	nb_orbits = gen->nb_orbits_at_level(level);
	for (h = 0; h < nb_orbits; h++) {
		strong_generators *Strong_gens;
		longinteger_object go;

		gen->get_set_by_level(level, h, set);
		gen->get_stabilizer_generators(Strong_gens,  
			level, h, 0 /* verbose_level*/);

		Strong_gens->group_order(go);


		cout << h << " : ";
		Orbiter->Lint_vec.print(cout, set, level);
		cout << " : " << go << " : ";
		
		score_sequence(Descr->n, set, level, score, verbose_level - 1);

		Orbiter->Lint_vec.print(cout, score, Descr->n);
		cout << endl;

		delete Strong_gens;
		}

	FREE_lint(set);
	FREE_lint(score);

}

void graph_classify::score_sequence(int n,
		long int *set, int sz, long int *score, int verbose_level)
{
	int i, a, swap, a2, u, v;
	combinatorics_domain Combi;

	Orbiter->Lint_vec.zero(score, n);
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


void graph_classify::draw_graphs(int level,
	layered_graph_draw_options *draw_options,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, nb_orbits;
	long int *set;
	int *v;

	if (f_v) {
		cout << "graph_classify::draw_graphs level = " << level << endl;
	}

	set = NEW_lint(level);
	v = NEW_int(n2);
	nb_orbits = gen->nb_orbits_at_level(level);

	if (f_v) {
		cout << "graph_classify::draw_graphs nb_orbits = " << nb_orbits << endl;
	}

	for (h = 0; h < nb_orbits; h++) {
		strong_generators *Strong_gens;
		longinteger_object go;

		gen->get_set_by_level(level, h, set);
		gen->get_stabilizer_generators(Strong_gens,  
			level, h, 0 /* verbose_level*/);

		Strong_gens->group_order(go);
		
		Orbiter->Int_vec.zero(v, n2);
		for (i = 0; i < level; i++) {
			v[set[i]] = 1;
			}

		cout << h << " : ";
		Orbiter->Lint_vec.print(cout, set, level);
		cout << " : ";
		for (i = 0; i < n2; i++) {
			cout << v[i];
			}
		cout << " : " << go << endl;


		string fname_full;
		char str[1000];

		fname_full.assign(gen->get_problem_label_with_path());

		sprintf(str, "_rep_%d_%d.mp", level, h);

		fname_full.append(str);

#if 1
		int x_min = 0, x_max = draw_options->xin;
		int y_min = 0, y_max = draw_options->yin;
		int x, y, dx, dy;

		x = (x_max - x_min) >> 1;
		y = (y_max - y_min) >> 1;
		dx = x;
		dy = y;
#endif

		{

			mp_graphics G;

			G.init(fname_full, draw_options, verbose_level - 1);

#if 0
			mp_graphics G(fname_full, draw_options, verbose_level - 1);
#endif

			G.header();
			G.begin_figure(1000 /*factor_1000*/);

			//G.sl_thickness(50); // 100 is normal
			//G.frame(0.05);


			if (Descr->f_tournament) {
				cout << "graph_classify::draw_graphs before G.draw_tournament" << endl;
				G.draw_tournament(x, y, dx, dy, Descr->n, set, level, draw_options->rad,
						verbose_level - 1);
				cout << "graph_classify::draw_graphs after G.draw_tournament" << endl;
				}
			else {
				cout << "graph_classify::draw_graphs before G.draw_graph" << endl;
				G.draw_graph(x, y, dx, dy, Descr->n, set, level, draw_options->rad,
						verbose_level - 1);
				cout << "graph_classify::draw_graphs after G.draw_graph" << endl;
				}

			G.end_figure();
			G.footer();
		}
		file_io Fio;

		cout << "written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;

		cout << "before FREE_OBJECT(Strong_gens)" << endl;
		//FREE_OBJECT(Strong_gens);
		cout << "after FREE_OBJECT(Strong_gens)" << endl;
		}

	FREE_lint(set);
	if (f_v) {
		cout << "graph_classify::draw_graphs level = " << level << " done" << endl;
	}
}


// #############################################################################
// global functions
// #############################################################################


void graph_classify_test_function(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level)
{
	graph_classify *Gen = (graph_classify *) data;
	int i, f_OK;

	Orbiter->Lint_vec.copy(S, Gen->S1, len);
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		Gen->S1[len] = candidates[i];
		if (Gen->Descr->f_tournament) {
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

void graph_classify_print_set(ostream &ost,
		int len, long int *S, void *data)
{
	graph_classify *Gen = (graph_classify *) data;
	
	//print_vector(ost, S, len);
	Gen->print(ost, S, len);
}


}}

