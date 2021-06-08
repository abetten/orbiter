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
	f_store_solutions = TRUE;
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

	// output variables:
	nb_search_steps = 0;
	nb_decision_steps = 0;
	dt = 0;
	Sol = NULL;
	nb_sol = 0;
}

clique_finder_control::~clique_finder_control()
{
	if (Sol) {
		FREE_int(Sol);
		Sol = NULL;
	}
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
			cout << "-weighted " << weights_total
					<< " " << weights_offset
					<< " " << weights_string
					<< " " << weights_bounds
					<< endl;
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
		else if (stringcmp(argv[i], "-count_solutions_only") == 0) {
			f_store_solutions = FALSE;
			cout << "-count_solutions_only " << endl;
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



}}
