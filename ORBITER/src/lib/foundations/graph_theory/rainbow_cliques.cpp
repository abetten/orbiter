// rainbow_cliques.C
//
// Anton Betten
//
// started:  October 28, 2012




#include "foundations.h"

rainbow_cliques::rainbow_cliques()
{
	null();
}

rainbow_cliques::~rainbow_cliques()
{
}

void rainbow_cliques::null()
{
	f_has_additional_test_function = FALSE;
}

void rainbow_cliques::freeself()
{
	null();
}

void rainbow_cliques::search(colored_graph *graph,
	ofstream *fp_sol, INT f_output_solution_raw,
	INT f_maxdepth, INT maxdepth, 
	INT f_restrictions, INT *restrictions, 
	INT f_tree, INT f_decision_nodes_only, const BYTE *fname_tree,  
	INT print_interval, 
	INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT i;
	
	if (f_v) {
		cout << "rainbow_cliques::search" << endl;
		}

	search_with_additional_test_function(graph,
		fp_sol, f_output_solution_raw,
		f_maxdepth, maxdepth,
		f_restrictions, restrictions,
		f_tree, f_decision_nodes_only, fname_tree,  
		print_interval, 
		FALSE /* f_has_additional_test_function */,
		NULL, 
		FALSE /* f_has_print_current_choice_function */, 
		NULL, 
		NULL /* user_data */,
		search_steps, decision_steps, nb_sol, dt, 
		verbose_level);
	
	if (f_v) {
		cout << "rainbow_cliques::search done" << endl;
		}
}

void rainbow_cliques::search_with_additional_test_function(
	colored_graph *graph,
	ofstream *fp_sol, INT f_output_solution_raw,
	INT f_maxdepth, INT maxdepth, 
	INT f_restrictions, INT *restrictions,
	INT f_tree, INT f_decision_nodes_only, const BYTE *fname_tree,  
	INT print_interval, 
	INT f_has_additional_test_function,
	void (*call_back_additional_test_function)(
		rainbow_cliques *R, void *user_data,
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT verbose_level), 
	INT f_has_print_current_choice_function,
	void (*call_back_print_current_choice)(clique_finder *CF, 
		INT depth, void *user_data, INT verbose_level), 
	void *user_data, 
	INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "rainbow_cliques::search_with_additional_"
				"test_function" << endl;
		}

	rainbow_cliques::f_output_solution_raw = f_output_solution_raw;

	if (f_has_additional_test_function) {
		rainbow_cliques::f_has_additional_test_function = TRUE;
		rainbow_cliques::call_back_additional_test_function =
				call_back_additional_test_function;
		rainbow_cliques::user_data = user_data;
		}
	else {
		rainbow_cliques::f_has_additional_test_function = FALSE;
		}
	rainbow_cliques::graph = graph;
	rainbow_cliques::fp_sol = fp_sol;
	f_color_satisfied = NEW_INT(graph->nb_colors);
	color_chosen_at_depth = NEW_INT(graph->nb_colors);
	color_frequency = NEW_INT(graph->nb_colors);
	
	for (i = 0; i < graph->nb_colors; i++) {
		f_color_satisfied[i] = FALSE;
		}

	CF = NEW_OBJECT(clique_finder);

	target_depth = graph->nb_colors;
	
	CF->init(graph->fname_base, graph->nb_points, 
		target_depth, 
		FALSE, NULL, 
		TRUE, graph->bitvector_adjacency, 
		print_interval, 
		f_maxdepth, maxdepth, 
		FALSE /* f_store_solutions */, 
		verbose_level - 2);

	CF->call_back_clique_found = call_back_colored_graph_clique_found;
	CF->call_back_add_point = call_back_colored_graph_add_point;
	CF->call_back_delete_point = call_back_colored_graph_delete_point;
	CF->call_back_find_candidates = call_back_colored_graph_find_candidates;
	CF->call_back_is_adjacent = NULL;

	//CF->call_back_after_reduction = call_back_after_reduction;
	CF->call_back_after_reduction = NULL;

	if (f_has_print_current_choice_function) {
		CF->f_has_print_current_choice_function = TRUE;
		CF->call_back_print_current_choice = call_back_print_current_choice;
		CF->print_current_choice_data = user_data;
		}
	
	CF->call_back_clique_found_data1 = this;
	
	
	if (f_restrictions) {
		if (f_v) {
			cout << "rainbow_cliques::search_with_additional_"
					"test_function before init_restrictions" << endl;
			}
		CF->init_restrictions(restrictions, verbose_level - 2);
		}

	if (f_tree) {
		CF->open_tree_file(fname_tree, f_decision_nodes_only);
		}
	
	INT t0, t1;

	t0 = os_ticks();

	if (f_vv) {
		cout << "rainbow_cliques::search now we start the "
				"rainbow clique finder process" << endl;
		}

#if 1

	CF->backtrack_search(0, 0 /*verbose_level*/);

#else
	if (f_vv) {
		cout << "rainbow_cliques::search before "
				"CF->backtrack_search_not_recursive" << endl;
		}
	CF->backtrack_search_not_recursive(verbose_level - 2);
	if (f_vv) {
		cout << "rainbow_cliques::search after "
				"CF->backtrack_search_not_recursive" << endl;
		}
#endif

	if (f_vv) {
		cout << "rainbow_cliques::search done with "
				"finding all rainbow cliques" << endl;
		}

	if (f_v) {
		cout << "depth : level_counter" << endl;
		for (i = 0; i < CF->target_depth; i++) {
			cout << setw(3) << i << " : " << setw(6)
					<< CF->level_counter[i] << endl;
			}
		}

	if (f_tree) {
		CF->close_tree_file();
		}

	search_steps = CF->counter;
	decision_steps = CF->decision_step_counter;
	nb_sol = CF->nb_sol;
	
	t1 = os_ticks();

	
	dt = t1 - t0;


	FREE_OBJECT(CF);
	FREE_INT(f_color_satisfied);
	FREE_INT(color_chosen_at_depth);
	FREE_INT(color_frequency);

	CF = NULL;
	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;

	if (f_v) {
		cout << "rainbow_cliques::search done" << endl;
		}
	
}

INT rainbow_cliques::find_candidates(
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c, i, j, c0, c0_freq, pt;
	
	if (f_v) {
		cout << "rainbow_cliques::find_candidates "
				"nb_pts = " << nb_pts << endl;
		}
	reduced_nb_pts = nb_pts;

	// determine the array color_frequency[].
	// color_frequency[i] is the frequency of points with color i 
	// in the list pt_list[]:

	INT_vec_zero(color_frequency, graph->nb_colors);
	for (i = 0; i < nb_pts; i++) {
		pt = pt_list[i];
		if (pt >= graph->nb_points) {
			cout << "rainbow_cliques::find_candidates "
					"pt >= nb_points" << endl;
			exit(1);
			}
		c = graph->point_color[pt];
		if (c >= graph->nb_colors) {
			cout << "rainbow_cliques::find_candidates "
					"c >= nb_colors" << endl;
			exit(1);
			}
		color_frequency[c]++;
		}
	if (f_v) {
		cout << "rainbow_cliques::find_candidates color_frequency: ";
		INT_vec_print(cout, color_frequency, graph->nb_colors);
		cout << endl;
		}

	// Determine the color c0 with the minimal frequency:
	c0 = -1;
	c0_freq = 0;
	for (c = 0; c < graph->nb_colors; c++) {
		if (f_color_satisfied[c]) {
			if (color_frequency[c]) {
				cout << "rainbow_cliques::find_candidates "
						"satisfied color appears with positive "
						"frequency" << endl;
				cout << "current clique:";
				INT_vec_print(cout, current_clique, current_clique_size);
				cout << endl;
				exit(1);
				}
			}
		else {
			if (color_frequency[c] == 0)
				return 0;
			if (c0 == -1) {
				c0 = c;
				c0_freq = color_frequency[c];
				}
			else {
				if (color_frequency[c] < c0_freq) {
					c0 = c;
					c0_freq = color_frequency[c];
					}
				}
			}
		}
	if (f_v) {
		cout << "rainbow_cliques::find_candidates "
				"minimal color is " << c0 << " with frequency "
				<< c0_freq << endl;
		}

	// And now we collect the points with color c0
	// in the array candidates:
	j = 0;
	for (i = 0; i < nb_pts; i++) {
		c = graph->point_color[pt_list[i]];
		if (c == c0) {
			candidates[j++] = pt_list[i];
			}
		}
	if (j != c0_freq) {
		cout << "rainbow_cliques::find_candidates "
				"j != c0_freq" << endl;
		exit(1);
		}

	// Mark color c0 as chosen:
	color_chosen_at_depth[current_clique_size] = c0;

	// we return the size of the candidate set:
	return c0_freq;
}

void rainbow_cliques::clique_found(
		INT *current_clique, INT verbose_level)
{
	INT i;
	
	for (i = 0; i < target_depth; i++) {
		*fp_sol << current_clique[i] << " ";
		}
	*fp_sol << endl;
}

void rainbow_cliques::clique_found_record_in_original_labels(
		INT *current_clique, INT verbose_level)
{
	INT i;
	
	*fp_sol << graph->user_data_size + target_depth << " ";
	for (i = 0; i < graph->user_data_size; i++) {
		*fp_sol << graph->user_data[i] << " ";
		}
	for (i = 0; i < target_depth; i++) {
		*fp_sol << graph->points[current_clique[i]] << " ";
		}
	*fp_sol << endl;
}


void call_back_colored_graph_clique_found(
		clique_finder *CF, INT verbose_level)
{
	INT f_v = FALSE; //(verbose_level >= 1);

	//cout << "call_back_colored_graph_clique_found" << endl;
	
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;

	if (f_v) {
		INT i, pt, c;
		
		cout << "call_back_colored_graph_clique_found clique";
		INT_set_print(cout, CF->current_clique, CF->target_depth);
		cout << endl;
		for (i = 0; i < CF->target_depth; i++) {
			pt = CF->current_clique[i];
			c = R->graph->point_color[pt];
			cout << i << " : " << pt << " : " << c << " : "
					<< R->f_color_satisfied[c] << endl;
			}
		}
	if (R->f_output_solution_raw) {
		R->clique_found(CF->current_clique, verbose_level);
		}
	else {
		R->clique_found_record_in_original_labels(
				CF->current_clique, verbose_level);
		}
}

void call_back_colored_graph_add_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;
	INT c;
	
	c = R->graph->point_color[pt];
	if (R->f_color_satisfied[c]) {
		cout << "call_back_colored_graph_add_point "
				"color already satisfied" << endl;
		exit(1);
		}
	if (c != R->color_chosen_at_depth[current_clique_size]) {
		cout << "call_back_colored_graph_add_point "
				"c != color_chosen_at_depth[current_clique_size]" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "call_back_colored_graph_add_point "
				"add_point " << pt << " at depth "
				<< current_clique_size
				<< " color=" << c << endl;
		}
	R->f_color_satisfied[c] = TRUE;
}

void call_back_colored_graph_delete_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;
	INT c;
	
	c = R->graph->point_color[pt];
	if (!R->f_color_satisfied[c]) {
		cout << "call_back_colored_graph_delete_point "
				"color not satisfied" << endl;
		exit(1);
		}
	R->f_color_satisfied[c] = FALSE;
	if (f_v) {
		cout << "call_back_colored_graph_delete_point "
				"delete_point " << pt << " at depth "
				<< current_clique_size << " color=" << c << endl;
		}
}

INT call_back_colored_graph_find_candidates(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;
	INT ret;

	if (R->f_has_additional_test_function) {

		INT tmp_nb_points;

		if (f_v) {
			cout << "call_back_colored_graph_find_candidates "
					"before call_back_additional_test_function" << endl;
			}
		(*R->call_back_additional_test_function)(R, R->user_data, 
			current_clique_size, current_clique, 
			nb_pts, tmp_nb_points, 
			pt_list, pt_list_inv, 
			verbose_level);

		nb_pts = tmp_nb_points;

		if (f_v) {
			cout << "call_back_colored_graph_find_candidates "
					"after call_back_additional_test_function "
					"nb_pts = " << nb_pts << endl;
			}

		}
	
	if (f_v) {
		cout << "call_back_colored_graph_find_candidates "
				"before R->find_candidates" << endl;
		}
	ret = R->find_candidates(current_clique_size, current_clique, 
			nb_pts, reduced_nb_pts, 
			pt_list, pt_list_inv, 
			candidates, verbose_level);
	if (f_v) {
		cout << "call_back_colored_graph_find_candidates "
				"after R->find_candidates" << endl;
		}
	
	return ret;
}



