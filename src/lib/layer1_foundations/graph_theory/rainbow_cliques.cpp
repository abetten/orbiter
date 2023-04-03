// rainbow_cliques.cpp
//
// Anton Betten
//
// started:  October 28, 2012




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graph_theory {


rainbow_cliques::rainbow_cliques()
{
	Control = NULL;

	ost_sol = NULL;

	graph = NULL;
	CF = NULL;
	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;

}

rainbow_cliques::~rainbow_cliques()
{
}

void rainbow_cliques::search(
	clique_finder_control *Control,
	colored_graph *graph,
	std::ostream &ost_sol,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	orbiter_kernel_system::os_interface Os;
	
	if (f_v) {
		cout << "rainbow_cliques::search" << endl;
	}

	rainbow_cliques::Control = Control;
	//rainbow_cliques::f_output_solution_raw = f_output_solution_raw;

	rainbow_cliques::graph = graph;
	rainbow_cliques::ost_sol = &ost_sol;
	f_color_satisfied = NEW_int(graph->nb_colors);
	color_chosen_at_depth = NEW_int(graph->nb_colors);
	color_frequency = NEW_int(graph->nb_colors);
	
	for (i = 0; i < graph->nb_colors; i++) {
		f_color_satisfied[i] = false;
	}


	CF = NEW_OBJECT(clique_finder);

	Control->target_size = graph->nb_colors / graph->nb_colors_per_vertex;
	if (f_v) {
		cout << "rainbow_cliques::search target_depth = " << Control->target_size << endl;
	}
	
	CF->init(Control,
			graph->fname_base, graph->nb_points,
		false, NULL, 
		true, graph->Bitvec,
		verbose_level - 2);

	CF->call_back_clique_found = call_back_colored_graph_clique_found;
	CF->call_back_add_point = call_back_colored_graph_add_point;
	CF->call_back_delete_point = call_back_colored_graph_delete_point;
	CF->call_back_find_candidates = call_back_colored_graph_find_candidates;
	CF->call_back_is_adjacent = NULL;

	//CF->call_back_after_reduction = call_back_after_reduction;
	CF->call_back_after_reduction = NULL;

	CF->call_back_clique_found_data1 = this;
	
	
	if (Control->f_restrictions) {
		if (f_v) {
			cout << "rainbow_cliques::search "
					"before init_restrictions" << endl;
		}
		CF->init_restrictions(Control->restrictions, verbose_level - 2);
	}

	if (Control->f_tree) {
		CF->open_tree_file(Control->fname_tree);
	}
	
	int t0, t1;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "rainbow_cliques::search before backtrack_search" << endl;
	}


	CF->backtrack_search(0, 0 /*verbose_level*/);


	if (f_v) {
		cout << "rainbow_cliques::search after backtrack_search" << endl;
	}

	if (f_v) {
		cout << "depth : level_counter" << endl;
		for (i = 0; i < Control->target_size; i++) {
			cout << setw(3) << i << " : " << setw(6)
					<< CF->level_counter[i] << endl;
		}
	}

	if (Control->f_tree) {
		CF->close_tree_file();
	}

	Control->nb_search_steps = CF->counter;
	Control->nb_decision_steps = CF->decision_step_counter;


	if (Control->f_store_solutions) {
		Control->nb_sol = CF->solutions.size();


		long int nb_sol;

		if (f_v) {
			cout << "rainbow_cliques::search before CF->get_solutions" << endl;
		}

		CF->get_solutions(Control->Sol,
				nb_sol, Control->target_size, verbose_level);

		if (f_v) {
			cout << "rainbow_cliques::search after CF->get_solutions" << endl;
		}
	}


	
	t1 = Os.os_ticks();

	
	Control->dt = t1 - t0;


	FREE_OBJECT(CF);
	FREE_int(f_color_satisfied);
	FREE_int(color_chosen_at_depth);
	FREE_int(color_frequency);

	CF = NULL;
	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;

	if (f_v) {
		cout << "rainbow_cliques::search done" << endl;
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

	Int_vec_zero(color_frequency, graph->nb_colors);
	for (i = 0; i < nb_pts; i++) {
		pt = pt_list[i];
		if (pt >= graph->nb_points) {
			cout << "rainbow_cliques::find_candidates pt >= nb_points" << endl;
			exit(1);
		}
		for (j = 0; j < graph->nb_colors_per_vertex; j++) {
			c = graph->point_color[pt * graph->nb_colors_per_vertex + j];
			if (c >= graph->nb_colors) {
				cout << "rainbow_cliques::find_candidates c >= nb_colors" << endl;
				exit(1);
			}
			color_frequency[c]++;
		}
	}
	if (f_v) {
		cout << "rainbow_cliques::find_candidates color_frequency: ";
		Int_vec_print(cout, color_frequency, graph->nb_colors);
		cout << endl;
	}

	// Determine the color c0 with the least positive frequency
	// A frequency of zero means that we cannot complete the partial rainbow clique:

	c0 = -1;
	c0_freq = 0;
	for (c = 0; c < graph->nb_colors; c++) {
		if (f_color_satisfied[c]) {
			if (color_frequency[c]) {
				cout << "rainbow_cliques::find_candidates "
						"satisfied color appears with positive "
						"frequency" << endl;
				cout << "current clique:";
				Int_vec_print(cout, current_clique, current_clique_size);
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
		// this should not happen.
		// I may happen if the coloring is not correct.
		// For instance, each color can appear at most once for each vertex (i.e., no repeats allowed)

		cout << "rainbow_cliques::find_candidates h != c0_freq" << endl;
		cout << "h=" << h << endl;
		cout << "c0_freq=" << c0_freq << endl;
		cout << "c0=" << c0 << endl;
		cout << "nb_pts=" << nb_pts << endl;
		cout << "nb_colors=" << graph->nb_colors << endl;
		cout << "nb_points=" << graph->nb_points << endl;
		cout << "nb_colors_per_vertex=" << graph->nb_colors_per_vertex << endl;
		cout << "current_clique_size=" << current_clique_size << endl;
		cout << "color_frequency=";
		Int_vec_print(cout, color_frequency, graph->nb_colors);
		cout << endl;
		cout << "f_color_satisfied=";
		Int_vec_print(cout, f_color_satisfied, graph->nb_colors);
		cout << endl;
		cout << "current clique:";
		Int_vec_print(cout, current_clique, current_clique_size);
		cout << endl;
		cout << "c : f_color_satisfied[c] : color_frequency[c]" << endl;
		for (c = 0; c < graph->nb_colors; c++) {
			cout << c << " : " << f_color_satisfied[c] << " : " << color_frequency[c] << endl;
		}
		exit(1);
	}

	// mark color c0 as chosen:
	color_chosen_at_depth[current_clique_size] = c0;

	// we return the size of the candidate set:
	return c0_freq;
}

void rainbow_cliques::clique_found(
		int *current_clique,
		int verbose_level)
{
	int i;
	
	for (i = 0; i < Control->target_size; i++) {
		*ost_sol << current_clique[i] << " ";
		}
	*ost_sol << endl;
}

void rainbow_cliques::clique_found_record_in_original_labels(
		int *current_clique, int verbose_level)
{
	int i;
	
	*ost_sol << graph->user_data_size + Control->target_size << " ";
	for (i = 0; i < graph->user_data_size; i++) {
		*ost_sol << graph->user_data[i] << " ";
	}
	for (i = 0; i < Control->target_size; i++) {
		*ost_sol << graph->points[current_clique[i]] << " ";
	}
	*ost_sol << endl;
}


void call_back_colored_graph_clique_found(
		clique_finder *CF, int verbose_level)
{
	int f_v = false; //(verbose_level >= 1);

	//cout << "call_back_colored_graph_clique_found" << endl;
	
	rainbow_cliques *R = (rainbow_cliques *)  CF->call_back_clique_found_data1;

	if (f_v) {
		int i, j, pt, c;
		
		cout << "call_back_colored_graph_clique_found clique";
		orbiter_kernel_system::Orbiter->Int_vec->set_print(cout, CF->current_clique, CF->Control->target_size);
		cout << endl;
		for (i = 0; i < CF->Control->target_size; i++) {
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
	if (R->Control->f_output_solution_raw) {
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
		R->f_color_satisfied[c] = true;
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
		R->f_color_satisfied[c] = false;
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
	rainbow_cliques *R = (rainbow_cliques *) CF->call_back_clique_found_data1;
	int ret;

	if (R->Control->f_has_additional_test_function) {

		int tmp_nb_points;

		if (f_v) {
			cout << "call_back_colored_graph_find_candidates "
					"before call_back_additional_test_function" << endl;
		}
		(*R->Control->call_back_additional_test_function)(R, R->Control->additional_test_function_data,
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


}}}



