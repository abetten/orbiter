/*
 * colored_graph_cliques.cpp
 *
 *  Created on: May 29, 2024
 *      Author: betten
 */




#include "../combinatorics/graph_theory/Clique/KClique.h"
#include "../combinatorics/graph_theory/Clique/RainbowClique.h"
#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {

#if 0
static void call_back_clique_found_using_file_output(
	clique_finder *CF, int verbose_level);
#endif

colored_graph_cliques::colored_graph_cliques()
{
	Record_birth();
	CG = NULL;
}


colored_graph_cliques::~colored_graph_cliques()
{
	Record_death();
}

void colored_graph_cliques::init(
		colored_graph *CG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph_cliques::init" << endl;
	}
	colored_graph_cliques::CG = CG;
}


void colored_graph_cliques::early_test_func_for_clique_search(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int j, a, pt;

	if (f_v) {
		cout << "colored_graph_cliques::early_test_func_for_clique_search "
				"checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
	}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];

		if (CG->is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
		}
	}

}

void colored_graph_cliques::early_test_func_for_coclique_search(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int j, a, pt;

	if (f_v) {
		cout << "colored_graph_cliques::early_test_func_for_"
				"coclique_search checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
	}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];

		if (!CG->is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
		}
	} // next j

}



void colored_graph_cliques::all_cliques(
		clique_finder_control *Control,
		std::string &graph_label,
		std::vector<std::string> &feedback,
		clique_finder *&CF,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "colored_graph_cliques::all_cliques" << endl;
	}

#if 0
	string fname_sol;
	data_structures::string_tools ST;

	if (Control->f_output_file) {
		fname_sol.assign(Control->output_file);
	}
	else {

		fname_sol = graph_label;
		ST.chop_off_extension(fname_sol);
		fname_sol += "_sol";
		//ST.replace_extension_with(fname_sol, "_sol.txt");
	}
	if (f_v) {
		cout << "colored_graph::all_cliques "
				"graph_label=" << graph_label << endl;
		cout << "colored_graph::all_cliques "
				"fname_sol=" << fname_sol << endl;
	}

#endif


	{

		if (Control->f_rainbow) {

			if (f_v) {
				cout << "colored_graph_cliques::all_cliques f_rainbow" << endl;
			}
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"before all_cliques_rainbow" << endl;
			}
			all_cliques_rainbow(
					Control,
					verbose_level);
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"after all_cliques_rainbow" << endl;
			}
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"f_rainbow done" << endl;
			}

		}
		else if (Control->f_weighted) {

			if (f_v) {
				cout << "colored_graph_cliques::all_cliques weighted cliques" << endl;
			}



			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"before all_cliques_weighted_with_two_colors" << endl;
			}
			all_cliques_weighted_with_two_colors(Control, verbose_level);
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"after all_cliques_weighted_with_two_colors" << endl;
			}



			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"weighted cliques done" << endl;
			}

		}
		else {

			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"before all_cliques_black_and_white" << endl;
			}
			all_cliques_black_and_white(
					Control,
					CF,
					verbose_level);
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques "
						"after all_cliques_black_and_white" << endl;
			}

		}
#if 0
		fp << -1 << " " << Control->nb_sol << " " << Control->nb_search_steps
			<< " " << Control->nb_decision_steps << " " << Control->dt << endl;
		fp_csv << "END" << endl;
#endif
	}

	string s;

	s = std::to_string(Control->nb_sol);

	feedback.push_back(s);

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques done" << endl;
	}
}




void colored_graph_cliques::all_cliques_rainbow(
		clique_finder_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_rainbow" << endl;
		cout << "nb_points = " << CG->nb_points << endl;
	}


	//std::vector<std::vector<long int> > solutions;
	std::vector<std::vector<unsigned int> > solutions;



	if (Control->f_Sajeeb) {
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_rainbow "
					"before do_Sajeeb" << endl;
		}
		//std::vector<std::vector<unsigned int> > solutions;

		do_Sajeeb(Control, solutions, verbose_level);
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_rainbow "
					"after do_Sajeeb" << endl;
		}
		if (Control->f_store_solutions) {




			int sz;

			sz = CG->nb_colors;

#if 0
			ost_csv << "ROW";
			for (int j = 0; j < sz; ++j) {
				ost_csv << ",C" << j;
			}
			ost_csv << endl;

			for (int i = 0; i < solutions.size(); ++i) {
				ost_csv << i << ",";

				if (sz != solutions[i].size()) {
					cout << "colored_graph_cliques::all_cliques_rainbow "
							"sz != solutions[i].size()" << endl;
					cout << "sz = " << sz << endl;
					cout << "solutions[i].size() = " << solutions[i].size() << endl;
							exit(1);
				}
				for (int j = 0; j < sz; ++j) {

					if (points) {
						ost_csv << points[solutions[i][j]];
					}
					else {
						ost_csv << solutions[i][j];
					}
					//fp_csv << Control->Sol[i * Control->target_size + j];
					if (j < sz - 1) {
						ost_csv << ",";
					}
				}
				ost_csv << endl;
			}
#endif

			Control->Sol = NEW_int(solutions.size() * sz);
			Control->nb_sol = solutions.size();
			for (int i = 0; i < solutions.size(); ++i) {
				for (int j = 0; j < sz; ++j) {
					long int a;
					if (CG->points) {
						a = CG->points[solutions[i][j]];
					}
					else {
						a = solutions[i][j];
					}
					Control->Sol[i * Control->target_size + j] = a;
				}
			}

		}
	}
	else {
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_rainbow "
					"before CG.all_rainbow_cliques" << endl;
		}
		all_rainbow_cliques(
				Control,
				verbose_level - 1);
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_rainbow "
					"after CG.all_rainbow_cliques" << endl;
		}

#if 0
		if (Control->f_store_solutions) {
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques_rainbow "
						"before write_solutions_to_csv_file" << endl;
			}
			write_solutions_to_csv_file(Control, ost_csv, verbose_level);
			if (f_v) {
				cout << "colored_graph_cliques::all_cliques_rainbow "
						"after write_solutions_to_csv_file" << endl;
			}
		}
#endif

	}


	if (Control->f_output_file) {
		string fname_sol;

		fname_sol = Control->output_file;



		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_rainbow "
					"before CG->write_solutions_to_csv_file" << endl;
		}
		CG->write_solutions_to_csv_file(
							fname_sol,
							solutions,
							Control,
							verbose_level);
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_rainbow "
					"after CG->write_solutions_to_csv_file" << endl;
		}

	}



	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_rainbow done" << endl;
	}
}

void colored_graph_cliques::all_cliques_black_and_white(
		clique_finder_control *Control,
		clique_finder *&CF,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_black_and_white" << endl;
	}

	if (!Control->f_target_size) {
		cout << "colored_graph_cliques::all_cliques_black_and_white "
				"please use -target_size <int : target_size>" << endl;
		exit(1);
	}


	std::vector<std::vector<unsigned int> > solutions;

	if (Control->f_Sajeeb) {
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"before do_Sajeeb_black_and_white" << endl;
		}

		do_Sajeeb_black_and_white(Control, solutions, verbose_level);

		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"after do_Sajeeb_black_and_white "
					"Found " << solutions.size() << " solution(s)." << endl;
		}



	}
	else {


		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"before CG->all_cliques_of_size_k_ignore_colors" << endl;
		}
		all_cliques_of_size_k_ignore_colors(
				Control,
				CF,
				verbose_level - 2);
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"after CG->all_cliques_of_size_k_ignore_colors, "
					"nb_cliques = " << Control->nb_sol << endl;
		}



		for (int i = 0; i < Control->nb_sol; ++i) {
			std::vector<unsigned int> sol;

			for (int j = 0; j < Control->target_size; ++j) {
				long int a;

				a = Control->Sol[i * Control->target_size + j];
				sol.push_back(a);
			}

			solutions.push_back(sol);

		}

	}


	if (Control->f_output_file) {
		string fname_sol;

		fname_sol = Control->output_file;



		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"before CG->write_solutions_to_csv_file" << endl;
		}
		CG->write_solutions_to_csv_file(
							fname_sol,
							solutions,
							Control,
							verbose_level);
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"after CG->write_solutions_to_csv_file" << endl;
		}

	}





	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_black_and_white done" << endl;
	}
}


void colored_graph_cliques::do_Sajeeb(
		clique_finder_control *Control,
		std::vector<std::vector<unsigned int> > &solutions,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb" << endl;
	}

#if 1


	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"creating Graph object" << endl;
	}

	Graph<> G (CG->nb_points, CG->nb_colors, CG->nb_colors_per_vertex);

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before setting vertex labels" << endl;
	}
	for (size_t i=0; i<G.nb_vertices; ++i) {
		G.vertex_label[i] = CG->points[i];
	}
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after setting vertex labels" << endl;
	}

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before setting vertex colors" << endl;
	}
	if (CG->nb_colors * CG->nb_colors_per_vertex) {
		for (size_t i=0; i<G.nb_vertices * G.nb_colors_per_vertex; ++i) {
			G.vertex_color[i] = CG->point_color[i]; // Anton: error corrected, was nb_colors should be nb_vertices
		}
	}
	else {
		for (size_t i=0; i<G.nb_vertices * G.nb_colors_per_vertex; ++i) {
			G.vertex_color[i] = 0;
		}
	}
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after setting vertex colors" << endl;
	}

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before G.set_edge_from_bitvector_adjacency" << endl;
	}
	G.set_edge_from_bitvector_adjacency(CG->Bitvec);
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after G.set_edge_from_bitvector_adjacency" << endl;
	}

	// Create the solution storage. The base type of the solution
	// storage must be the same as data type of the vertex label
	// in the graph
	//std::vector<std::vector<unsigned int> > solutions;
	if (f_v) {
		cout << __FILE__ << ":" << __LINE__ << endl;
	}

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before RainbowClique::find_cliques" << endl;
	}
    // Call the Rainbow Clique finding algorithm
	RainbowClique::find_cliques(G, solutions, 0 /* nb_threads */);
		// nb_threads = 0 automatically detects the number of threads
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after RainbowClique::find_cliques" << endl;
	}
	if (f_v) {
		cout << __FILE__ << ":" << __LINE__ << endl;
	}

	// Print the solutions
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"Found " << solutions.size()
				<< " solution(s)." << endl;
		for (size_t i=0; i<solutions.size(); ++i) {
			for (size_t j=0; j<solutions[i].size(); ++j) {
				cout << solutions[i][j] << " ";
			}
			cout << endl;
		}
	}

	//this->nb_sol = solutions.size();
#endif



	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb done" << endl;
	}
}

void colored_graph_cliques::do_Sajeeb_black_and_white(
		clique_finder_control *Control,
		std::vector<std::vector<unsigned int> >& solutions,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb" << endl;
	}

#if 1
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before opening Graph object" << endl;
	}
	Graph<unsigned int, int> G (CG->nb_points, CG->nb_colors, CG->nb_colors_per_vertex);

#if 0
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before setting vertex labels" << endl;
	}
	G.set_vertex_labels(CG->points);
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after setting vertex labels" << endl;
	}
#endif

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"CG->nb_colors_per_vertex = " << CG->nb_colors_per_vertex << endl;
	}

	if (CG->nb_colors_per_vertex) {
		if (f_v) {
			cout << "colored_graph_cliques::do_Sajeeb "
					"before setting vertex colors" << endl;
		}
		G.set_vertex_colors(CG->point_color);
		if (f_v) {
			cout << "colored_graph_cliques::do_Sajeeb "
					"after setting vertex colors" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "colored_graph_cliques::do_Sajeeb "
					"no vertex colors" << endl;
		}
	}

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before set_edge_from_bitvector_adjacency" << endl;
	}
	G.set_edge_from_bitvector_adjacency(CG->Bitvec);
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after set_edge_from_bitvector_adjacency" << endl;
	}

	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"before KClique::find_cliques" << endl;
	}
	// Call the Rainbow Clique finding algorithm
	KClique::find_cliques(G, solutions, Control->target_size);
	//RainbowClique::find_cliques(G, solutions, 0 /* nb_threads */);
	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb "
				"after KClique::find_cliques" << endl;
	}

	//this->nb_sol = solutions.size();
#endif



	if (f_v) {
		cout << "colored_graph_cliques::do_Sajeeb done" << endl;
	}
}


void colored_graph_cliques::all_cliques_weighted_with_two_colors(
		clique_finder_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors" << endl;
	}

	int *weights;
	int nb_weights;
	int *bounds;
	int nb_bounds;
	int target_value;
	int i;


	Int_vec_scan(Control->weights_string, weights, nb_weights);
	Int_vec_scan(Control->weights_bounds, bounds, nb_bounds);

	if (nb_bounds != nb_weights) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"nb_bounds != nb_weights" << endl;
		exit(1);
	}

	if (nb_weights != 2) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"nb_weights != 2" << endl;
		exit(1);
	}
	if (CG->nb_colors < nb_weights + Control->weights_offset) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"CG->nb_colors < nb_weights + weights_offset" << endl;
		exit(1);
	}

	target_value = Control->weights_total;

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"target_value = " << target_value << endl;
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"the weights are ";
		Int_vec_print(cout, weights, nb_weights);
		cout << endl;
	}


	solvers::diophant D;
	long int nb_backtrack_nodes;
	int nb_sol;
	int *Sol_weights;
	int j;
	vector<int> res;
	string label;

	label = "weights";

	D.init_partition_problem_with_bounds(
			weights, bounds, nb_weights, target_value,
			verbose_level);


	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"before D.solve_mckay" << endl;
	}
	D.solve_mckay(
			label, INT_MAX /* maxresults */,
			nb_backtrack_nodes, nb_sol,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"after D.solve_mckay" << endl;
	}
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"we found " << nb_sol << " solutions for the "
			"weight distribution" << endl;
	}

	Sol_weights = NEW_int(nb_sol * nb_weights);

	for (i = 0; i < D._resultanz; i++) {
		res = D._results[i]; //.front();
		for (j = 0; j < nb_weights; j++) {
			Sol_weights[i * nb_weights + j] = res[j];
		}
		//D._results.pop_front();
	}

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors "
				"The solutions are:" << endl;
		for (i = 0; i < nb_sol; i++) {
			cout << i << " : ";
			Int_vec_print(cout, Sol_weights + i * nb_weights, nb_weights);
			cout << endl;
		}
	}

	int c1 = Control->weights_offset + 0;
	int c2 = Control->weights_offset + 1;

	if (f_v) {
		cout << "creating subgraph of color " << c1 << ":" << endl;
	}

	colored_graph *subgraph;

	subgraph = CG->subgraph_by_color_classes(c1, verbose_level);

	if (f_v) {
		cout << "The subgraph of color " << c1 << " has size "
				<< subgraph->nb_points << endl;
	}

	int target_depth1;
	int target_depth2;
	int nb_solutions_total;

	//clique_finder_control *Control1;

	//Control1 = NEW_OBJECT(clique_finder_control);

	nb_solutions_total = 0;

	for (i = 0; i < nb_sol; i++) {

		target_depth1 = Sol_weights[i * nb_weights + c1];
		target_depth2 = Sol_weights[i * nb_weights + c2];


		clique_finder_control *Control1;

		Control1 = NEW_OBJECT(clique_finder_control);
		Control1->target_size = target_depth1;

		clique_finder *CF;

		subgraph->Colored_graph_cliques->all_cliques_of_size_k_ignore_colors(Control1,
				CF,
				verbose_level);

		if (f_v) {
			cout << "solution " << i << " / " << nb_sol
					<< " with target_depth = " << target_depth1
					<< " Control1->nb_sol=" << Control1->nb_sol << endl;
		}

		for (j = 0; j < Control1->nb_sol; j++) {




			colored_graph *subgraph2;
			clique_finder_control *Control2;

			Control2 = NEW_OBJECT(clique_finder_control);
			Control2->target_size = target_depth2;

			if (f_v) {
				cout <<  "solution " << i << " / " << nb_sol
						<< ", clique1 " << j << " / " << Control1->nb_sol << ":" << endl;
			}

			subgraph2 = CG->subgraph_by_color_classes_with_condition(
					Control1->Sol + j * target_depth1, target_depth1,
					c2, verbose_level);

			if (f_v) {
				cout << "solution " << i << " / " << nb_sol
						<< ", clique1 " << j << " / " << Control1->nb_sol
						<< ", subgraph2 has " << subgraph2->nb_points << " vertices" << endl;
			}

			clique_finder *CF2;

			subgraph2->Colored_graph_cliques->all_cliques_of_size_k_ignore_colors(
					Control2,
					CF2,
					verbose_level);

			nb_solutions_total += Control2->nb_sol;

			FREE_OBJECT(CF2);

			if (f_v) {
				cout << "solution " << i << " / " << nb_sol << ", "
						"clique1 " << j << " / " << Control1->nb_sol
						<< ", Control2->nb_sol=" << Control2->nb_sol
					<< " nb_solutions_total=" << nb_solutions_total << endl;
			}

			FREE_OBJECT(subgraph2);
			FREE_OBJECT(Control2);
		}
		FREE_OBJECT(CF);
		FREE_OBJECT(Control1);
	}

	if (f_v) {
		cout << "nb_solutions_total=" << nb_solutions_total << endl;
	}

	FREE_int(Sol_weights);

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_weighted_with_two_colors done" << endl;
	}
}


void colored_graph_cliques::all_cliques_of_size_k_ignore_colors(
	clique_finder_control *Control,
	clique_finder *&CF,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//clique_finder *CF;

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"target_size = " << Control->target_size << endl;
	}
	CF = NEW_OBJECT(clique_finder);

	string dummy;

	dummy.assign("");
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"before CF->init" << endl;
	}
	CF->init(
			Control,
			dummy, CG->nb_points,
			false /* f_has_adj_list */, NULL /* int *adj_list_coded */,
			true /* f_has_bitvector */, CG->Bitvec,
			verbose_level - 1);
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"after CF->init" << endl;
	}

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"before CF->clique_finder_backtrack_search" << endl;
	}
	CF->clique_finder_backtrack_search(0 /* depth */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"after CF->clique_finder_backtrack_search, "
				"nb_sol = " << CF->solutions.size() << endl;
	}

	Control->nb_sol = CF->solutions.size();
	Control->nb_decision_steps = CF->decision_step_counter;

	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"before CF->get_solutions" << endl;
	}
	long int nb_sol;
	CF->get_solutions(Control->Sol,
			nb_sol, Control->target_size, verbose_level);
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"after CF->get_solutions" << endl;
	}
	if (nb_sol != Control->nb_sol) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors "
				"nb_sol != Control->nb_sol" << endl;
		exit(1);
	}

	//FREE_OBJECT(CF);
	if (f_v) {
		cout << "colored_graph_cliques::all_cliques_of_size_k_ignore_colors done" << endl;
	}
}


void colored_graph_cliques::all_rainbow_cliques(
		clique_finder_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R;

	if (f_v) {
		cout << "colored_graph_cliques::all_rainbow_cliques" << endl;
	}
	R = NEW_OBJECT(rainbow_cliques);
	if (f_v) {
		cout << "colored_graph_cliques::all_rainbow_cliques "
				"before R->search" << endl;
	}
	R->search(Control, CG, /*ost,*/ verbose_level - 1);
	if (f_v) {
		cout << "colored_graph_cliques::all_rainbow_cliques "
				"after R->search" << endl;
	}
	FREE_OBJECT(R);
	if (f_v) {
		cout << "colored_graph_cliques::all_rainbow_cliques "
				"done" << endl;
	}
}


int colored_graph_cliques::test_Neumaier_property(
		int &regularity,
		int &lambda_value,
		int clique_size,
		int &nexus,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	if (f_v) {
		cout << "colored_graph_cliques::test_Neumaier_property " << endl;
	}
	if (f_v) {
		cout << "colored_graph_cliques::test_Neumaier_property graph = " << endl;
		CG->print();
	}
	int ret = true;


	if (!CG->test_if_regular(
			regularity,
			verbose_level)) {
		ret = false;
	}

	lambda_value = 0;
	nexus = -1;

	if (ret) {

		if (CG->test_lambda_property(
				lambda_value,
				verbose_level)) {
			ret = true;
			if (f_v) {
				cout << "colored_graph_cliques::test_Neumaier_property "
						"the graph has the lambda property "
						"with lambda = " << lambda_value << endl;
			}
		}
		else {
			if (f_v) {
				cout << "colored_graph_cliques::test_Neumaier_property "
						"the graph is not Neumaier" << endl;
			}
			ret = false;
		}
	}

	if (ret) {

		clique_finder_control Control;
		clique_finder *CF;


		Control.f_target_size = true;
		Control.target_size = clique_size;
		Control.f_Sajeeb = false;

		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"searching for all cliques of size " << clique_size << endl;
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"before all_cliques_of_size_k_ignore_colors" << endl;
		}
		all_cliques_of_size_k_ignore_colors(
				&Control,
				CF,
				verbose_level - 2);
		if (f_v) {
			cout << "colored_graph_cliques::all_cliques_black_and_white "
					"after all_cliques_of_size_k_ignore_colors, "
					"nb_cliques = " << Control.nb_sol << endl;
		}

		int *Sol;
		long int nb_sol;

		int sz;
		if (f_v) {
			cout << "colored_graph_cliques::write_solutions "
					"before CF->get_solutions" << endl;
		}
		CF->get_solutions(Sol, nb_sol, sz, verbose_level);
		if (f_v) {
			cout << "colored_graph_cliques::write_solutions "
					"after CF->get_solutions" << endl;
		}
		if (f_v) {
			cout << "colored_graph_cliques::test_Neumaier_property "
					"number of cliques of size "
					<< clique_size << " = " << nb_sol << endl;
		}

		if (nb_sol == 0) {
			cout << "colored_graph_cliques::test_Neumaier_property "
					"no cliques of size "
					<< clique_size << " found" << endl;
			exit(1);
		}

		int c;
		int nx;

		nexus = -1;

		for (c = 0; c < nb_sol; c++) {

			if (f_v) {
				cout << "colored_graph_cliques::test_Neumaier_property "
						"testing clique "
						<< c << " / " << nb_sol << endl;
				Int_vec_print(cout, Sol + c * sz, sz);
				cout << endl;
			}
			if (!test_if_clique_is_regular(
					Sol + c * sz, sz, nx, verbose_level)) {
				ret = false;
				break;
			}
			else {
				if (nexus == -1) {
					nexus = nx;
				}
				else {
					if (nexus != nx) {
						ret = false;
						break;
					}
				}
			}

		}

		FREE_OBJECT(CF);
		if (ret) {
			if (f_v) {
				cout << "colored_graph_cliques::test_Neumaier_property "
						"all cliques are regular with nexus " << nexus << endl;
			}
		}



	}
	if (f_v) {
		cout << "colored_graph_cliques::test_Neumaier_property ret=" << ret << endl;
	}
	return ret;
}

int colored_graph_cliques::test_if_clique_is_regular(
		int *clique, int sz, int &nexus, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "colored_graph_cliques::test_if_clique_is_regular" << endl;
	}
	int ret;

	ret = true;

	other::data_structures::fancy_set *F;

	F = NEW_OBJECT(other::data_structures::fancy_set);

	F->init_with_set(
			CG->nb_points, sz, clique, 0 /* verbose_level */);

	int nb_outside;
	int i, j, nb;
	int pt1, pt2;

	nb_outside = CG->nb_points - sz;
	for (i = 0; i < nb_outside; i++) {
		nb = 0;
		pt1 = F->set[sz + i];
		for (j = 0; j < sz; j++) {
			pt2 = F->set[j];
			if (CG->is_adjacent(
					pt1, pt2)) {
				nb++;
			}
		}
		if (false) {
			cout << "colored_graph_cliques::test_if_clique_is_regular "
					"i=" << i << " pt1=" << pt1 << " nb=" << nb << endl;
		}
		if (i == 0) {
			nexus = nb;
		}
		else {
			if (nb != nexus) {
				ret = false;
				break;
			}
		}
	}



	FREE_OBJECT(F);

	return ret;
}




// #############################################################################
// static functions:
// #############################################################################

#if 0
static void call_back_clique_found_using_file_output(
	clique_finder *CF, int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	//cout << "call_back_clique_found_using_file_output" << endl;

	orbiter_kernel_system::file_output *FO =
			(orbiter_kernel_system::file_output *)
			CF->call_back_clique_found_data1;
	colored_graph *CG = (colored_graph *) CF->call_back_clique_found_data2;
	//clique_finder *CF = (clique_finder *) FO->user_data;

	if (CG->user_data_size && CG->points) {
		int i, a;
		*FO->fp << CG->user_data_size + CF->Control->target_size;
		for (i = 0; i < CG->user_data_size; i++) {
			*FO->fp << " " << CG->user_data[i];
		}
		for (i = 0; i < CF->Control->target_size; i++) {
			a = CF->current_clique[i];
			*FO->fp << " " << CG->points[a];
		}
		*FO->fp << endl;
	}
	else {
		FO->write_line(CF->Control->target_size,
				CF->current_clique, verbose_level);
	}
}
#endif



}}}}


