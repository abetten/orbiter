// spread_lifting.cpp
// 
// Anton Betten
// April 1, 2018
//
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_lifting::spread_lifting()
{
	S = NULL;
	R = NULL;
	//std::string output_prefix;

	//starter = NULL;
	//starter_size = 0;
	//starter_case_number = 0;
	//starter_number_of_cases = 0;

	//candidates = NULL;
	//nb_candidates = 0;
	//Strong_gens = NULL;

	f_lex = FALSE;

	points_covered_by_starter = NULL;
	nb_points_covered_by_starter = 0;

	nb_free_points = 0;
	free_point_list = NULL;
	point_idx = NULL;

	nb_colors = 0;
	colors = NULL;

	nb_needed = 0;

	reduced_candidates = NULL;
	nb_reduced_candidates = 0;

	nb_cols = 0;
	col_color = NULL;
	col_labels = NULL;

}




spread_lifting::~spread_lifting()
{
	if (points_covered_by_starter) {
		FREE_lint(points_covered_by_starter);
	}
	if (free_point_list) {
		FREE_lint(free_point_list);
	}
	if (point_idx) {
		FREE_lint(point_idx);
	}
	if (colors) {
		FREE_int(colors);
	}
	if (reduced_candidates) {
		FREE_lint(reduced_candidates);
	}
	if (col_color) {
		FREE_int(col_color);
	}
	if (col_labels) {
		FREE_lint(col_labels);
	}
}

void spread_lifting::init(
		spread_classify *S,
		data_structures_groups::orbit_rep *R,
	std::string &output_prefix,
	int f_lex, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go;
	
	
	if (f_v) {
		cout << "spread_lifting::init" << endl;
	}
	spread_lifting::S = S;
	spread_lifting::R = R;
	spread_lifting::output_prefix.assign(output_prefix);

	//spread_lifting::starter = R->rep; // starter;
	//spread_lifting::starter_size = R->level; // starter_size;
	//spread_lifting::starter_case_number = R->orbit_at_level; //starter_case_number;
	//spread_lifting::starter_number_of_cases = R->nb_cases; // starter_number_of_cases;
	//spread_lifting::candidates = R->candidates; // candidates;
	//spread_lifting::nb_candidates = R->nb_candidates; // nb_candidates;
	//spread_lifting::Strong_gens = R->Strong_gens; // Strong_gens;

	spread_lifting::f_lex = f_lex;
	
	if (f_v) {
		cout << "spread_lifting::init starter_size=" << R->level << endl;
	}
	if (f_v) {
		cout << "spread_lifting::init "
				"before compute_points_covered_by_starter" << endl;
	}
	compute_points_covered_by_starter(verbose_level - 2);
	if (f_v) {
		cout << "spread_lifting::init "
				"after compute_points_covered_by_starter" << endl;
	}
	
	if (f_v) {
		cout << "spread_lifting::init "
				"before prepare_free_points" << endl;
	}
	prepare_free_points(verbose_level - 2);
	if (f_v) {
		cout << "spread_lifting::init "
				"after prepare_free_points" << endl;
	}

	nb_needed = S->SD->spread_size - R->level;
	if (f_v) {
		cout << "spread_lifting::init "
				"nb_needed=" << nb_needed << endl;
		cout << "spread_lifting::init "
				"nb_candidates=" << R->nb_candidates << endl;
	}




#if 0
	if (f_lex) {
		int nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "spread_lifting::init after lexorder test "
					"nb_candidates before: " << nb_cols_before
					<< " reduced to  " << nb_cols << " (deleted "
					<< nb_cols_before - nb_cols << ")" << endl;
		}
	}
#endif

	if (f_v) {
		cout << "spread_lifting::init after lexorder test" << endl;
		cout << "spread_lifting::init nb_cols=" << nb_cols << endl;
	}

	if (f_v) {
		cout << "spread_lifting::init done" << endl;
	}
}

void spread_lifting::compute_points_covered_by_starter(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int i, a;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "spread_lifting::compute_points_covered_by_starter" << endl;
	}
	nb_points_covered_by_starter = R->level * S->block_size;
	points_covered_by_starter = NEW_lint(nb_points_covered_by_starter);

	for (i = 0; i < R->level; i++) {
		long int *point_list;
		int nb_points;

		a = R->rep[i];
		S->SD->Grass->unrank_lint(a, 0/*verbose_level - 4*/);
		S->SD->F->Projective_space_basic->all_PG_elements_in_subspace(
			S->SD->Grass->M, S->SD->k, S->SD->n, point_list,
			nb_points, 0 /*verbose_level - 2*/);
		
		if (nb_points != S->block_size) {
			cout << "spread_lifting::compute_points_covered_by_starter "
					"nb_points != S->block_size" << endl;
			exit(1);
		}

		Lint_vec_copy(point_list,
				points_covered_by_starter + i * S->block_size,
				S->block_size);

		if (f_v3) {
			cout << "starter element " << i << " / "
					<< R->level << " is " << a << ":" << endl;
			Int_matrix_print(S->SD->Grass->M, S->SD->k, S->SD->n);
			//cout << endl;
			cout << "points_covered_by_starter: " << endl;
			Lint_vec_print(cout,
					points_covered_by_starter +
						i * S->block_size, S->block_size);
			cout << endl;
		}

		FREE_lint(point_list);
	}
	Sorting.lint_vec_heapsort(points_covered_by_starter,
			nb_points_covered_by_starter);
	if (f_vv) {
		cout << "spread_lifting::compute_points_covered_by_starter "
				"covered points computed:" << endl;
		cout << "spread_lifting::compute_points_covered_by_starter "
				"nb_points_covered_by_starter="
				<< nb_points_covered_by_starter << endl;
		Lint_vec_print(cout, points_covered_by_starter,
				nb_points_covered_by_starter);
		cout << endl;
	}

	if (f_v) {
		cout << "spread_lifting::compute_points_covered_by_starter done" << endl;
	}
}

void spread_lifting::prepare_free_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	int i, j, idx;
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "spread_lifting::prepare_free_points" << endl;
	}

	nb_free_points = S->SD->nb_points_total - nb_points_covered_by_starter;
	if (f_vv) {
		cout << "spread_lifting::prepare_free_points "
				"nb_free_points=" << nb_free_points << endl;
	}
	free_point_list = NEW_lint(nb_free_points);
	point_idx = NEW_lint(S->SD->nb_points_total);
	j = 0;
	for (i = 0; i < S->SD->nb_points_total; i++) {
		if (Sorting.lint_vec_search(points_covered_by_starter,
				nb_points_covered_by_starter, i, idx, 0)) {
			point_idx[i] = -1;
		}
		else {
			free_point_list[j] = i;
			point_idx[i] = j;
			j++;
		}
	}
	if (j != nb_free_points) {
		cout << "spread_lifting::prepare_free_points "
				"j != nb_free_points" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "spread_lifting::prepare_free_points "
				"computed free points" << endl;
	}

	if (f_v) {
		print_free_points();
	}
	if (f_v) {
		cout << "spread_lifting::prepare_free_points done" << endl;
	}
}

void spread_lifting::print_free_points()
{
	cout << "spread_lifting::print_free_points "
			"The " << nb_free_points << " free points are:" << endl;
	Lint_vec_print(cout,
			free_point_list, nb_free_points);
	cout << endl;
	S->SD->print_points(free_point_list, nb_free_points);

}

void spread_lifting::compute_colors(int &f_ruled_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	int *v;
	geometry::geometry_global Geo;
	
	if (f_v) {
		cout << "spread_lifting::compute_colors" << endl;
		cout << "spread_lifting::compute_colors, "
				"nb_free_points = " << nb_free_points << endl;
	}

	f_ruled_out = FALSE;

	v = NEW_int(S->SD->n);
	colors = NEW_int(nb_free_points);

	nb_colors = 0;

	for (i = 0; i < nb_free_points; i++) {
		a = free_point_list[i];
		S->SD->unrank_point(v, a);
		S->SD->F->Projective_space_basic->PG_element_normalize_from_front(
			v, 1, S->SD->n);


		if (is_e1_vector(v) || (is_zero_vector(v) && is_e1_vector(v + S->SD->k))) {


#if 0
			if (is_zero_vector(v)) {
				c = 0;
			}
			else {
				c = Geo.AG_element_rank(S->SD->q, v + S->SD->k, 1, S->SD->k) + 1;
			}
#endif

			if (f_v) {
				cout << "spread_lifting::compute_colors found color " << nb_colors
						<< " : " << i << " = " << a << " = ";
				Int_vec_print(cout, v, S->SD->n);
				cout << endl;
			}

			colors[nb_colors++] = i;

		}

	}

	if (f_v) {
		cout << "spread_lifting::compute_colors "
				"we found " << nb_colors << " colors" << endl;
	}

	if (nb_colors != nb_needed) {
		cout << "spread_lifting::compute_colors "
				"nb_colors != nb_needed" << endl;
		f_ruled_out = TRUE;
		return;
	}
	

#if 0
	col_color = NEW_int(nb_cols);
	for (j = 0; j < nb_cols; j++) {
		for (c = 0; c < nb_colors; c++) {
			i = colors[c];
			if (Dio->Aij(i, j)) {
				col_color[j] = c;
				break;
			}
		}
		if (c == nb_colors) {
			cout << "spread_lifting::compute_colors "
					"candidate " << j << " does not have a color" << endl;
			exit(1);
		}
	}

	data_structures::tally T;


	T.init(col_color, nb_cols, FALSE, verbose_level);
	if (f_v) {
		cout << "spread_lifting::compute_colors color frequencies amongst columns:" << endl;
	}
	T.print(FALSE /* f_backwards */);
#endif


	//FREE_int(colors);
	FREE_int(v);
	if (f_v) {
		cout << "spread_lifting::compute_colors done" << endl;
	}
}


void spread_lifting::reduce_candidates(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//geometry::geometry_global Geo;

	if (f_v) {
		cout << "spread_lifting::reduce_candidates" << endl;
	}

	long int *Part;
	int s;

	s = S->SD->Grass->nb_points_covered(verbose_level);

	if (f_v) {
		cout << "spread_lifting::reduce_candidates "
				"before S->SD->Grass->make_partition" << endl;
	}
	S->SD->Grass->make_partition(R->candidates, R->nb_candidates, Part, s, 0 /* verbose_level */);
	if (f_v) {
		cout << "spread_lifting::reduce_candidates "
				"after S->SD->Grass->make_partition" << endl;
	}

	if (f_v) {
		cout << "spread_lifting::reduce_candidates Partition:" << endl;
		Lint_matrix_print(Part, R->nb_candidates, s);
	}

	int *Adj;
	int u, j, pt, i, c;

	Adj = NEW_int(nb_free_points * R->nb_candidates);
	Int_vec_zero(Adj, nb_free_points * R->nb_candidates);

	for (j = 0; j < R->nb_candidates; j++) {

		// fill column j of Adj[]:

		for (u = 0; u < s; u++) {
			pt = Part[j * s + u];
			i = point_idx[pt];
			if (i == -1) {
				cout << "spread_lifting::reduce_candidates "
						"candidate block contains point that "
						"is already covered" << endl;
				exit(1);
			}
			if (i < 0 || i >= nb_free_points) {
				cout << "spread_lifting::reduce_candidates "
						"i < 0 || i >= nb_free_points" << endl;
				exit(1);
			}
			Adj[i * R->nb_candidates + j] = 1;
		}
	}

	if (FALSE) {
		cout << "spread_lifting::reduce_candidates Adj:" << endl;
		Int_matrix_print(Adj, nb_free_points, R->nb_candidates);
	}

	reduced_candidates = NEW_lint(R->nb_candidates);
	col_color = NEW_int(R->nb_candidates);

	nb_reduced_candidates = 0;
	for (j = 0; j < R->nb_candidates; j++) {
		if (FALSE) {
			cout << "spread_lifting::reduce_candidates "
					"j=" << j << endl;
		}
		for (c = 0; c < nb_colors; c++) {
			if (FALSE) {
				cout << "spread_lifting::reduce_candidates "
						"j=" << j << " c=" << c << endl;
			}
			i = colors[c];
			if (FALSE) {
				cout << "spread_lifting::reduce_candidates "
						"j=" << j << " c=" << c << " i=" << i << endl;
			}
			if (Adj[i * R->nb_candidates + j]) {
				break;
			}
		}
		if (FALSE) {
			cout << "spread_lifting::reduce_candidates "
					"j=" << j << " c=" << c << endl;
		}
		if (c < nb_colors) {
			reduced_candidates[nb_reduced_candidates] = R->candidates[j];
			col_color[nb_reduced_candidates] = c;
			nb_reduced_candidates++;
		}
	}

	if (f_v) {
		cout << "spread_lifting::reduce_candidates "
				"nb_reduced_candidates=" << nb_reduced_candidates << endl;
	}

	col_labels = NEW_lint(nb_reduced_candidates);
	Lint_vec_copy(reduced_candidates, col_labels, nb_reduced_candidates);
	nb_cols = nb_reduced_candidates;

	if (f_v) {
		cout << "spread_lifting::reduce_candidates "
				"nb_candidates = " << R->nb_candidates << endl;
		cout << "spread_lifting::reduce_candidates "
				"nb_reduced_candidates = " << nb_reduced_candidates << endl;
	}

	data_structures::tally T;


	T.init(col_color, nb_cols, FALSE, verbose_level);
	if (f_v) {
		cout << "spread_lifting::reduce_candidates "
				"color frequencies amongst columns:" << endl;
	}
	T.print(FALSE /* f_backwards */);


	FREE_lint(Part);
	FREE_int(Adj);

	if (f_v) {
		cout << "spread_lifting::reduce_candidates done" << endl;
	}


}

solvers::diophant *spread_lifting::create_system(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v5 = (verbose_level >= 5);
	solvers::diophant *Dio;
	int nb_rows = nb_free_points;
	int i, j, a, b, h;

	if (f_v) {
		cout << "spread_lifting::create_system" << endl;
	}

	Dio = NEW_OBJECT(solvers::diophant);
	Dio->open(nb_rows, nb_cols, verbose_level - 1);
	Dio->f_has_sum = TRUE;
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
	}

	Dio->fill_coefficient_matrix_with(0);
	if (f_vv) {
		cout << "spread_lifting::create_system "
				"nb_rows = " << nb_rows << endl;
		cout << "spread_lifting::create_system "
				"nb_cols = " << nb_cols << endl;
	}
	for (j = 0; j < nb_cols; j++) {

		long int *point_list;
		int nb_points;

		a = col_labels[j];
		S->SD->Grass->unrank_lint(a, 0/*verbose_level - 4*/);
		if (f_vv) {
			cout << "candidate " << j << " / "
					<< nb_cols << " is " << a << endl;
		}

		if (f_v5) {
			cout << "Which is " << endl;
			Int_matrix_print(S->SD->Grass->M, S->SD->k, S->SD->n);
		}
		S->SD->F->Projective_space_basic->all_PG_elements_in_subspace(
				S->SD->Grass->M, S->SD->k, S->SD->n,
				point_list, nb_points,
				0 /*verbose_level*/);
		if (nb_points != S->block_size) {
			cout << "spread_lifting::create_system "
					"nb_points != S->block_size" << endl;
			exit(1);
		}
		if (FALSE /*f_vv*/) {
			cout << "List of points: ";
			Lint_vec_print(cout, point_list, nb_points);
			cout << endl;
		}




		for (h = 0; h < S->block_size; h++) {
			b = point_list[h];
			i = point_idx[b];
			if (i == -1) {
				cout << "spread_lifting::create_system "
						"candidate block contains point that "
						"is already covered" << endl;
				exit(1);
			}
			if (i < 0 || i >= nb_free_points) {
				cout << "spread_lifting::create_system "
						"i < 0 || i >= nb_free_points" << endl;
				exit(1);
			}
			Dio->Aij(i, j) = 1;
		}
		FREE_lint(point_list);
	}

	if (FALSE) {
		cout << "spread_lifting::create_system "
				"coefficient matrix" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << Dio->Aij(i, j);
			}
			cout << endl;
		}
	}
	if (f_v) {
		cout << "spread_lifting::create_system done" << endl;
	}
	return Dio;
}


int spread_lifting::is_e1_vector(int *v)
{
	int j;

	if (v[0] != 1) {
		return FALSE;
	}
	for (j = 1; j < S->SD->k; j++) {
		if (v[j] != 0) {
			return FALSE;
		}
	}
	return TRUE;
}

int spread_lifting::is_zero_vector(int *v)
{
	int j;

	for (j = 0; j < S->SD->k; j++) {
		if (v[j] != 0) {
			return FALSE;
		}
	}
	return TRUE;
}

void spread_lifting::create_graph(
		data_structures::bitvector *Adj,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_v3 = (verbose_level >= 3);

	if (f_v) {
		cout << "spread_lifting::create_graph" << endl;
	}


	graph_theory::colored_graph *CG;

	CG = NEW_OBJECT(graph_theory::colored_graph);

	char str[1000];
	string label, label_tex;
	snprintf(str, sizeof(str), "_graph_%d", R->orbit_at_level);
	label.assign(S->prefix);
	label.append(str);
	label_tex.assign(str);

	if (f_v) {
		cout << "spread_lifting::create_graph "
				"before CG->init_with_point_labels" << endl;
	}
	CG->init_with_point_labels(nb_cols, nb_colors, 1,
		col_color,
		Adj, FALSE /* f_ownership_of_bitvec */,
		col_labels /* point_labels */,
		label, label_tex,
		verbose_level);
	if (f_v) {
		cout << "spread_lifting::create_graph "
				"after CG->init_with_point_labels" << endl;
	}


	if (f_v) {
		cout << "spread_lifting::create_graph "
				"before CG->init_user_data" << endl;
	}
	CG->init_user_data(R->rep, R->level, verbose_level);
	if (f_v) {
		cout << "spread_lifting::create_graph "
				"after CG->init_user_data" << endl;
	}

	string fname_clique_graph;
	orbiter_kernel_system::file_io Fio;

	fname_clique_graph.assign(output_prefix);
	fname_clique_graph.append(label);
	fname_clique_graph.append(".bin");

	CG->save(fname_clique_graph, verbose_level - 1);
	if (f_v) {
		cout << "Written file " << fname_clique_graph
				<< " of size " << Fio.file_size(fname_clique_graph) << endl;
	}

	if (f_v) {
		cout << "spread_lifting::create_graph "
				"before FREE_OBJECT(CG)" << endl;
	}
	FREE_OBJECT(CG);

	if (f_v) {
		cout << "spread_lifting::create_graph done" << endl;
	}
}


void spread_lifting::create_dummy_graph(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_v3 = (verbose_level >= 3);

	if (f_v) {
		cout << "spread_lifting::create_dummy_graph" << endl;
	}

	data_structures::bitvector *Adj;

	Adj = NEW_OBJECT(data_structures::bitvector);

	long int L;
	int n = 2;
	int i;

	L = (n * (n - 1)) >> 1;
	if (f_v) {
		cout << "spread_lifting::create_dummy_graph L=" << L << endl;
	}


	if (f_v) {
		cout << "spread_lifting::create_dummy_graph before Adj->allocate" << endl;
	}
	Adj->allocate(L);
	if (f_v) {
		cout << "spread_lifting::create_dummy_graph after Adj->allocate" << endl;
	}
	if (f_v) {
		cout << "spread_lifting::create_dummy_graph setting array to one" << endl;
	}
	for (i = 0; i < L; i++) {
		Adj->m_i(i, 0);
	}


	graph_theory::colored_graph *CG;

	CG = NEW_OBJECT(graph_theory::colored_graph);

	char str[1000];
	string label, label_tex;
	snprintf(str, sizeof(str), "_graph_%d", R->orbit_at_level);
	label.assign(S->prefix);
	label.append(str);
	label_tex.assign(str);

	if (f_v) {
		cout << "spread_lifting::create_dummy_graph "
				"before CG->init_with_point_labels" << endl;
	}

	int col_color[2] = {0, 1};
	long int col_labels[2] = {-1, -1};

	CG->init_with_point_labels(n /* nb_vertices */, 2 /* nb_colors */, 1,
		col_color,
		Adj, FALSE /* f_ownership_of_bitvec */,
		col_labels /* point_labels */,
		label, label_tex,
		verbose_level);
	if (f_v) {
		cout << "spread_lifting::create_dummy_graph "
				"after CG->init_with_point_labels" << endl;
	}

	string fname_clique_graph;
	orbiter_kernel_system::file_io Fio;

	fname_clique_graph.assign(output_prefix);
	fname_clique_graph.append(label);
	fname_clique_graph.append(".bin");

	CG->save(fname_clique_graph, verbose_level - 1);
	if (f_v) {
		cout << "Written file " << fname_clique_graph
				<< " of size " << Fio.file_size(fname_clique_graph) << endl;
	}

	if (f_v) {
		cout << "spread_lifting::create_dummy_graph "
				"before FREE_OBJECT(CG)" << endl;
	}
	FREE_OBJECT(CG);

	if (f_v) {
		cout << "spread_lifting::create_dummy_graph done" << endl;
	}
}



}}}

