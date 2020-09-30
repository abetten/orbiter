// rainbow_cliques.cpp
//
// Anton Betten
//
// started:  October 28, 2012




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


rainbow_cliques::rainbow_cliques()
{
	fp_sol = NULL;
	f_output_solution_raw = FALSE;

	graph = NULL;
	CF = NULL;
	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;
	target_depth = 0;

	// added November 5, 2014:
	f_has_additional_test_function = FALSE;
	call_back_additional_test_function = NULL;
	user_data = NULL;
	null();
}

rainbow_cliques::~rainbow_cliques()
{
}

void rainbow_cliques::null()
{
}

void rainbow_cliques::freeself()
{
	null();
}

void rainbow_cliques::search(colored_graph *graph,
	ofstream *fp_sol, int f_output_solution_raw,
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, std::string &fname_tree,
	int print_interval, 
	unsigned long int &search_steps, unsigned long int &decision_steps,
	int &nb_sol, int &dt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int i;
	
	if (f_v) {
		cout << "rainbow_cliques::search" << endl;
	}

	if (f_v) {
		cout << "rainbow_cliques::search before search_with_additional_test_function" << endl;
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
		cout << "rainbow_cliques::search after search_with_additional_test_function" << endl;
	}
	
	if (f_v) {
		cout << "rainbow_cliques::search done" << endl;
	}
}

void rainbow_cliques::search_with_additional_test_function(
	colored_graph *graph,
	ofstream *fp_sol, int f_output_solution_raw,
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions,
	int f_tree, int f_decision_nodes_only, std::string &fname_tree,
	int print_interval, 
	int f_has_additional_test_function,
	void (*call_back_additional_test_function)(
		rainbow_cliques *R, void *user_data,
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int verbose_level), 
	int f_has_print_current_choice_function,
	void (*call_back_print_current_choice)(clique_finder *CF, 
		int depth, void *user_data, int verbose_level), 
	void *user_data, 
	unsigned long int &search_steps, unsigned long int &decision_steps,
	int &nb_sol, int &dt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	os_interface Os;
	
	if (f_v) {
		cout << "rainbow_cliques::search_with_additional_test_function" << endl;
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
	f_color_satisfied = NEW_int(graph->nb_colors);
	color_chosen_at_depth = NEW_int(graph->nb_colors);
	color_frequency = NEW_int(graph->nb_colors);
	
	for (i = 0; i < graph->nb_colors; i++) {
		f_color_satisfied[i] = FALSE;
		}

	CF = NEW_OBJECT(clique_finder);

	target_depth = graph->nb_colors / graph->nb_colors_per_vertex;
	if (f_v) {
		cout << "rainbow_cliques::search_with_additional_test_function target_depth = " << target_depth << endl;
	}
	
	CF->init(graph->fname_base, graph->nb_points,
		target_depth, 
		FALSE, NULL, 
		TRUE, graph->Bitvec,
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
			cout << "rainbow_cliques::search_with_additional_test_function "
					"before init_restrictions" << endl;
		}
		CF->init_restrictions(restrictions, verbose_level - 2);
	}

	if (f_tree) {
		CF->open_tree_file(fname_tree, f_decision_nodes_only);
	}
	
	int t0, t1;

	t0 = Os.os_ticks();

	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function before backtrack_search" << endl;
	}

#if 1

	CF->backtrack_search(0, 0 /*verbose_level*/);

#else
	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function before "
				"CF->backtrack_search_not_recursive" << endl;
		}
	CF->backtrack_search_not_recursive(verbose_level - 2);
	if (f_vv) {
		cout << "rainbow_cliques::search after "
				"CF->backtrack_search_not_recursive" << endl;
		}
#endif

	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function after backtrack_search" << endl;
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
	
	t1 = Os.os_ticks();

	
	dt = t1 - t0;


	FREE_OBJECT(CF);
	FREE_int(f_color_satisfied);
	FREE_int(color_chosen_at_depth);
	FREE_int(color_frequency);

	CF = NULL;
	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;

	if (f_v) {
		cout << "rainbow_cliques::search_with_additional_test_function done" << endl;
	}
}

int rainbow_cliques::find_candidates(
	int current_clique_size, int *current_clique, 
	int nb_pts, int &reduced_nb_pts, 
	int *pt_list, int *pt_list_inv, 
	int *candidates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c, i, j, h, c0, c0_freq, pt;
	
	if (f_v) {
		cout << "rainbow_cliques::find_candidates "
				"nb_pts = " << nb_pts << endl;
		}
	reduced_nb_pts = nb_pts;

	// determine the array color_frequency[].
	// color_frequency[i] is the frequency of points with color i 
	// in the list pt_list[]:

	int_vec_zero(color_frequency, graph->nb_colors);
	for (i = 0; i < nb_pts; i++) {
		pt = pt_list[i];
		if (pt >= graph->nb_points) {
			cout << "rainbow_cliques::find_candidates "
					"pt >= nb_points" << endl;
			exit(1);
			}
		for (j = 0; j < graph->nb_colors_per_vertex; j++) {
			c = graph->point_color[pt * graph->nb_colors_per_vertex + j];
			if (c >= graph->nb_colors) {
				cout << "rainbow_cliques::find_candidates "
						"c >= nb_colors" << endl;
				exit(1);
				}
			color_frequency[c]++;
		}
	}
	if (f_v) {
		cout << "rainbow_cliques::find_candidates color_frequency: ";
		int_vec_print(cout, color_frequency, graph->nb_colors);
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
				int_vec_print(cout, current_clique, current_clique_size);
				cout << endl;
				exit(1);
			}
		}
		else {
			if (color_frequency[c] == 0) {
				return 0;
			}
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
	h = 0;
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < graph->nb_colors_per_vertex; j++) {
			c = graph->point_color[pt_list[i] * graph->nb_colors_per_vertex + j];
			if (c == c0) {
				break;
			}
		}
		if (j < graph->nb_colors_per_vertex) {
			candidates[h++] = pt_list[i];
		}
	}
	if (h != c0_freq) {
		cout << "rainbow_cliques::find_candidates "
				"h != c0_freq" << endl;
		exit(1);
	}

	// Mark color c0 as chosen:
	color_chosen_at_depth[current_clique_size] = c0;

	// we return the size of the candidate set:
	return c0_freq;
}

void rainbow_cliques::clique_found(
		int *current_clique, int verbose_level)
{
	int i;
	
	for (i = 0; i < target_depth; i++) {
		*fp_sol << current_clique[i] << " ";
		}
	*fp_sol << endl;
}

void rainbow_cliques::clique_found_record_in_original_labels(
		int *current_clique, int verbose_level)
{
	int i;
	
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
		clique_finder *CF, int verbose_level)
{
	int f_v = FALSE; //(verbose_level >= 1);

	//cout << "call_back_colored_graph_clique_found" << endl;
	
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;

	if (f_v) {
		int i, j, pt, c;
		
		cout << "call_back_colored_graph_clique_found clique";
		int_set_print(cout, CF->current_clique, CF->target_depth);
		cout << endl;
		for (i = 0; i < CF->target_depth; i++) {
			pt = CF->current_clique[i];
			cout << i << " : " << pt << " : ";
			for (j = 0; j < R->graph->nb_colors_per_vertex; j++) {
				c = R->graph->point_color[pt * R->graph->nb_colors_per_vertex + j];
				cout << c;
				if (j < R->graph->nb_colors_per_vertex) {
					cout << ", ";
				}
			}
		cout << endl;
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
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;
	int c, j;
	
	for (j = 0; j < R->graph->nb_colors_per_vertex; j++) {
		c = R->graph->point_color[pt * R->graph->nb_colors_per_vertex + j];
		if (R->f_color_satisfied[c]) {
			cout << "call_back_colored_graph_add_point "
					"color already satisfied" << endl;
			exit(1);
			}
		R->f_color_satisfied[c] = TRUE;
	}
	if (f_v) {
		cout << "call_back_colored_graph_add_point "
				"add_point " << pt << " at depth "
				<< current_clique_size << endl;
		}
}

void call_back_colored_graph_delete_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;
	int c, j;
	
	for (j = 0; j < R->graph->nb_colors_per_vertex; j++) {
		c = R->graph->point_color[pt * R->graph->nb_colors_per_vertex + j];
		if (!R->f_color_satisfied[c]) {
			cout << "call_back_colored_graph_delete_point "
					"color not satisfied" << endl;
			exit(1);
			}
		R->f_color_satisfied[c] = FALSE;
	}
	if (f_v) {
		cout << "call_back_colored_graph_delete_point "
				"delete_point " << pt << " at depth "
				<< current_clique_size << endl;
		}
}

int call_back_colored_graph_find_candidates(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int nb_pts, int &reduced_nb_pts, 
	int *pt_list, int *pt_list_inv, 
	int *candidates, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data1;
	int ret;

	if (R->f_has_additional_test_function) {

		int tmp_nb_points;

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


}
}

