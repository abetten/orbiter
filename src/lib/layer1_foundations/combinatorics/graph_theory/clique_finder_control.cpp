// clique_finder_control.cpp
//
// Anton Betten
//
// started:  October 11, 2018




#include "../combinatorics/graph_theory/Clique/KClique.h"
#include "../combinatorics/graph_theory/Clique/RainbowClique.h"
#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {


clique_finder_control::clique_finder_control()
{
	Record_birth();
	f_rainbow = false;
	f_target_size = false;
	target_size = 0;
	f_weighted = false;
	weights_total = false;
	weights_offset = 0;
	//weights_string;
	//weights_bounds
	f_Sajeeb = false;
	//f_file = false;
	//fname_graph = NULL;
	f_nonrecursive = false;
	f_output_solution_raw = false;
	f_store_solutions = true;
	f_output_file = false;
	//output_file;
	f_maxdepth = false;
	maxdepth = 0;
	f_restrictions = false;
	nb_restrictions = 0;
	f_tree = false;
	f_decision_nodes_only = false;
	//fname_tree = NULL;
	print_interval = 1;

	f_has_additional_test_function = false;
	call_back_additional_test_function = NULL;
	additional_test_function_data = NULL;

	f_has_print_current_choice_function = false;
	call_back_print_current_choice = NULL;
	print_current_choice_data = NULL;

	// output variables:
	nb_search_steps = 0;
	nb_decision_steps = 0;
	dt = 0;
	Sol = NULL;
	nb_sol = 0;
}

clique_finder_control::~clique_finder_control()
{
	Record_death();
	if (Sol) {
		FREE_int(Sol);
		Sol = NULL;
	}
}

int clique_finder_control::parse_arguments(
		int argc, std::string *argv)
{
	int i;
	other::data_structures::string_tools ST;

	cout << "clique_finder_control::parse_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-rainbow") == 0) {
			f_rainbow = true;
			cout << "-rainbow " << endl;
		}
		else if (ST.stringcmp(argv[i], "-target_size") == 0) {
			f_target_size = true;
			target_size = ST.strtoi(argv[++i]);
			cout << "-target_size " << target_size << endl;
		}
		else if (ST.stringcmp(argv[i], "-weighted") == 0) {
			f_weighted = true;
			weights_total = ST.strtoi(argv[++i]);
			weights_offset = ST.strtoi(argv[++i]);
			weights_string.assign(argv[++i]);
			weights_bounds.assign(argv[++i]);
			cout << "-weighted " << weights_total
					<< " " << weights_offset
					<< " " << weights_string
					<< " " << weights_bounds
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-Sajeeb") == 0) {
			f_Sajeeb = true;
			cout << "-Sajeeb " << endl;
		}
		else if (ST.stringcmp(argv[i], "-nonrecursive") == 0) {
			f_nonrecursive = true;
			cout << "-nonrecursive " << endl;
		}
		else if (ST.stringcmp(argv[i], "-tree") == 0) {
			f_tree = true;
			f_decision_nodes_only = false;
			fname_tree.assign(argv[++i]);
			cout << "-tree " << fname_tree << endl;
		}
		else if (ST.stringcmp(argv[i], "-tree_decision_nodes_only") == 0) {
			f_tree = true;
			f_decision_nodes_only = true;
			fname_tree.assign(argv[++i]);
			cout << "-tree_decision_nodes_only " << fname_tree << endl;
		}
		else if (ST.stringcmp(argv[i], "-output_file") == 0) {
			f_output_file = true;
			output_file.assign(argv[++i]);
			cout << "-output_file " << output_file << endl;
		}
		else if (ST.stringcmp(argv[i], "-output_solution_raw") == 0) {
			f_output_solution_raw = true;
			cout << "-output_solution_raw " << endl;
		}
		else if (ST.stringcmp(argv[i], "-count_solutions_only") == 0) {
			f_store_solutions = false;
			cout << "-count_solutions_only " << endl;
		}
		else if (ST.stringcmp(argv[i], "-restrictions") == 0) {
			f_restrictions = true;
			int j;
			for (j = 0; true; j++) {
				restrictions[j] = ST.strtoi(argv[++i]);
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
			Int_vec_print(cout, restrictions, 3 * nb_restrictions);
			cout << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
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


void clique_finder_control::print()
{
	if (f_rainbow) {
		cout << "-rainbow " << endl;
	}
	else if (f_target_size) {
		cout << "-target_size " << target_size << endl;
	}
	else if (f_weighted) {
		cout << "-weighted " << weights_total
				<< " " << weights_offset
				<< " " << weights_string
				<< " " << weights_bounds
				<< endl;
	}
	else if (f_Sajeeb) {
		cout << "-Sajeeb " << endl;
	}
	else if (f_nonrecursive) {
		cout << "-nonrecursive " << endl;
	}
	else if (f_tree && !f_decision_nodes_only) {
		cout << "-tree " << fname_tree << endl;
	}
	else if (f_tree && f_decision_nodes_only) {
		cout << "-tree_decision_nodes_only " << fname_tree << endl;
	}
	else if (f_output_file) {
		cout << "-output_file " << output_file << endl;
	}
	else if (f_output_solution_raw) {
		cout << "-output_solution_raw " << endl;
	}
	else if (f_store_solutions) {
		cout << "-count_solutions_only " << endl;
	}
	else if (f_restrictions) {
		cout << "-restrictions ";
		Int_vec_print(cout, restrictions, 3 * nb_restrictions);
		cout << endl;
	}
}



}}}}


