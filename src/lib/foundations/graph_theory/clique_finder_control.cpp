// clique_finder_control.cpp
//
// Anton Betten
//
// started:  October 11, 2018




#include "foundations.h"
#include "Clique/RainbowClique.h"
#include "KClique.h"

using namespace std;


namespace orbiter {
namespace foundations {


clique_finder_control::clique_finder_control()
{
	f_rainbow = FALSE;
	f_target_size = FALSE;
	target_size = 0;
	f_weighted = FALSE;
	weights_total = FALSE;
	weights_offset = 0;
	//weights_string;
	//weights_bounds
	f_Sajeeb = FALSE;
	//f_file = FALSE;
	//fname_graph = NULL;
	f_nonrecursive = FALSE;
	f_output_solution_raw = FALSE;
	f_output_file = FALSE;
	//output_file;
	f_maxdepth = FALSE;
	maxdepth = 0;
	f_restrictions = FALSE;
	nb_restrictions = 0;
	f_tree = FALSE;
	f_decision_nodes_only = FALSE;
	//fname_tree = NULL;
	print_interval = 1;
	nb_search_steps = 0;
	nb_decision_steps = 0;
	nb_sol = 0;
	dt = 0;
}

clique_finder_control::~clique_finder_control()
{
}

int clique_finder_control::parse_arguments(
		int argc, std::string *argv)
{
	int i;

	cout << "clique_finder_control::parse_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-rainbow") == 0) {
			f_rainbow = TRUE;
			cout << "-rainbow " << endl;
		}
		else if (stringcmp(argv[i], "-target_size") == 0) {
			f_target_size = TRUE;
			target_size = strtoi(argv[++i]);
			cout << "-target_size " << target_size << endl;
		}
		else if (stringcmp(argv[i], "-weighted") == 0) {
			f_weighted = TRUE;
			weights_total = strtoi(argv[++i]);
			weights_offset = strtoi(argv[++i]);
			weights_string.assign(argv[++i]);
			weights_bounds.assign(argv[++i]);
			cout << "-weighted " << weights_total << " " << weights_offset << " " << weights_string << " " << weights_bounds << endl;
		}
		else if (stringcmp(argv[i], "-Sajeeb") == 0) {
			f_Sajeeb = TRUE;
			cout << "-Sajeeb " << endl;
		}
		else if (stringcmp(argv[i], "-nonrecursive") == 0) {
			f_nonrecursive = TRUE;
			cout << "-nonrecursive " << endl;
		}
		else if (stringcmp(argv[i], "-tree") == 0) {
			f_tree = TRUE;
			f_decision_nodes_only = FALSE;
			fname_tree.assign(argv[++i]);
			cout << "-tree " << fname_tree << endl;
		}
		else if (stringcmp(argv[i], "-tree_decision_nodes_only") == 0) {
			f_tree = TRUE;
			f_decision_nodes_only = TRUE;
			fname_tree.assign(argv[++i]);
			cout << "-tree_decision_nodes_only " << fname_tree << endl;
		}
		else if (stringcmp(argv[i], "-output_file") == 0) {
			f_output_file = TRUE;
			output_file.assign(argv[++i]);
			cout << "-output_file " << output_file << endl;
		}
		else if (stringcmp(argv[i], "-output_solution_raw") == 0) {
			f_output_solution_raw = TRUE;
			cout << "-output_solution_raw " << endl;
		}
		else if (stringcmp(argv[i], "-restrictions") == 0) {
			f_restrictions = TRUE;
			int j;
			for (j = 0; TRUE; j++) {
				restrictions[j] = strtoi(argv[++i]);
				if (restrictions[j] == -1) {
					nb_restrictions = j / 3;
					break;
					}
				if (nb_restrictions >= CLIQUE_FINDER_CONTROL_MAX_RESTRICTIONS) {
					cout << "clique_finder_control::parse_arguments "
							"restrictions must end in -1" << endl;
					exit(1);
				}
			}
			cout << "-restrictions ";
			Orbiter->Int_vec.print(cout, restrictions, 3 * nb_restrictions);
			cout << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			return i;
		}
		else {
			cout << "clique_finder_control::parse_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	}
	cout << "clique_finder_control::parse_arguments "
			"did not see -end option" << endl;
	exit(1);
}


void clique_finder_control::all_cliques(colored_graph *CG,
	std::string &fname_graph,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	string fname_sol;
	string_tools ST;

	if (f_v) {
		cout << "clique_finder_control::all_cliques" << endl;
	}
	if (f_output_file) {
		fname_sol.assign(output_file);
	}
	else {
		fname_sol.assign(fname_graph);
		ST.replace_extension_with(fname_sol, "_sol.txt");
	}

	//CG.print();

	{
		string fname_sol_csv;
		string_tools ST;


		fname_sol_csv.assign(fname_sol);
		ST.replace_extension_with(fname_sol_csv, ".csv");
		ofstream fp(fname_sol);
		ofstream fp_csv(fname_sol_csv);


		if (f_rainbow) {
			if (f_weighted) {

				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"weighted cliques" << endl;
				}

				graph_theory_domain GT;


				GT.all_cliques_weighted_with_two_colors(this, CG, verbose_level);




			}
			else if (f_Sajeeb) {
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before do_Sajeeb" << endl;
				}
				do_Sajeeb(CG, verbose_level);
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"after do_Sajeeb" << endl;
				}
			}
			else {
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before CG.all_rainbow_cliques" << endl;
					}
				CG->all_rainbow_cliques(&fp,
					f_output_solution_raw,
					f_maxdepth, maxdepth,
					f_restrictions, restrictions,
					f_tree, f_decision_nodes_only, fname_tree,
					print_interval,
					nb_search_steps, nb_decision_steps, nb_sol, dt,
					verbose_level - 1);
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"after CG.all_rainbow_cliques" << endl;
				}
			}
		}
		else {
			cout << "clique_finder_control::all_cliques not rainbow" << endl;
			if (!f_target_size) {
				cout << "clique_finder_control::all_cliques please use -target_size <int : target_size>" << endl;
				exit(1);
			}

			if (f_Sajeeb) {
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before do_Sajeeb_black_and_white" << endl;
				}
				std::vector<std::vector<long int> > solutions;

				do_Sajeeb_black_and_white(CG, target_size, solutions, verbose_level - 2);

				// Print the solutions
				if (f_v) {
					cout << "clique_finder_control::all_cliques after do_Sajeeb_black_and_white Found " << solutions.size() << " solution(s)." << endl;
				}


				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before writing solutions to file" << endl;
				}

				#if 1
				for (size_t i = 0; i < solutions.size(); ++i) {
					fp << solutions[i].size() << " ";
					for (size_t j = 0; j < solutions[i].size(); ++j) {
						fp << CG->points[solutions[i][j]] << " ";
					}
					fp << endl;
				}

				fp_csv << "ROW";
				for (int j = 0; j < target_size; ++j) {
					fp_csv << ",C" << j;
				}
				fp_csv << endl;

				for (size_t i = 0; i < solutions.size(); ++i) {
					for (size_t j = 0; j < solutions[i].size(); ++j) {
						fp_csv << CG->points[solutions[i][j]];
						if (j < solutions[i].size() - 1) {
							fp_csv << " ";
						}
					}
					fp_csv << endl;
				}
				#endif
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"after writing solutions to file" << endl;
				}
			}
			else {

				int *Sol = NULL;
				unsigned long int decision_step_counter = 0;

				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before CG->all_cliques_of_size_k_ignore_colors" << endl;
				}
				CG->all_cliques_of_size_k_ignore_colors(
						target_size,
						Sol, nb_sol,
						decision_step_counter,
						verbose_level - 2);
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before CG->all_cliques_of_size_k_ignore_colors" << endl;
				}

				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"before writing solutions to file" << endl;
				}
				for (int i = 0; i < nb_sol; ++i) {
					fp << target_size << " ";
					for (int j = 0; j < target_size; ++j) {
						fp << CG->points[Sol[i * target_size + j]];
						if (j < target_size - 1) {
							fp << " ";
						}
					}
					fp << endl;
				}


				fp_csv << "ROW";
				for (int j = 0; j < target_size; ++j) {
					fp_csv << ",C" << j;
				}
				fp_csv << endl;

				for (int i = 0; i < nb_sol; ++i) {
					fp_csv << i << ",";
					for (int j = 0; j < target_size; ++j) {
						fp_csv << CG->points[Sol[i * target_size + j]];
						if (j < target_size - 1) {
							fp_csv << ",";
						}
					}
					fp_csv << endl;
				}
				if (f_v) {
					cout << "clique_finder_control::all_cliques "
							"after writing solutions to file" << endl;
				}

				FREE_int(Sol);
			}
		}
		fp << -1 << " " << nb_sol << " " << nb_search_steps
			<< " " << nb_decision_steps << " " << dt << endl;
		fp_csv << "END" << endl;
	}

	if (f_v) {
		cout << "clique_finder_control::all_cliques done" << endl;
	}
}

void clique_finder_control::do_Sajeeb(colored_graph *CG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "clique_finder_control::do_Sajeeb" << endl;
	}

#if 1
	Graph<> G (CG->nb_points, CG->nb_colors, CG->nb_colors_per_vertex);

	for (size_t i=0; i<G.nb_vertices; ++i) G.vertex_label[i] = CG->points[i];
	for (size_t i=0; i<G.nb_colors; ++i) G.vertex_color[i] = CG->point_color[i];

	G.set_edge_from_bitvector_adjacency(CG->Bitvec);

	// Create the solution storage. The base type of the solution
	// storage must be the same as data type of the vertex label
	// in the graph
	std::vector<std::vector<unsigned int> > solutions;
	cout << __FILE__ << ":" << __LINE__ << endl;

    // Call the Rainbow Clique finding algorithm
	RainbowClique::find_cliques(G, solutions, 0 /* nb_threads */);
		// nb_threads = 0 automatically detects the number of threads
	cout << __FILE__ << ":" << __LINE__ << endl;

	// Print the solutions
	cout << "clique_finder_control::do_Sajeeb Found " << solutions.size() << " solution(s)." << endl;
//	for (size_t i=0; i<solutions.size(); ++i) {
//		for (size_t j=0; j<solutions[i].size(); ++j) {
//			cout << solutions[i][j] << " ";
//		} cout << endl;
//	}

	this->nb_sol = solutions.size();
#endif



	if (f_v) {
		cout << "clique_finder_control::do_Sajeeb done" << endl;
	}
}

void clique_finder_control::do_Sajeeb_black_and_white(colored_graph *CG,
		int clique_size, std::vector<std::vector<long int> >& solutions,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "clique_finder_control::do_Sajeeb" << endl;
	}

#if 1
	Graph<long int, int> G (CG->nb_points, CG->nb_colors, CG->nb_colors_per_vertex);
	G.set_vertex_labels(CG->points);
	G.set_vertex_colors(CG->point_color);
	G.set_edge_from_bitvector_adjacency(CG->Bitvec);

    // Call the Rainbow Clique finding algorithm
	KClique::find_cliques(G, solutions, clique_size);
	//RainbowClique::find_cliques(G, solutions, 0 /* nb_threads */);

	this->nb_sol = solutions.size();
#endif



	if (f_v) {
		cout << "clique_finder_control::do_Sajeeb done" << endl;
	}
}


}}
