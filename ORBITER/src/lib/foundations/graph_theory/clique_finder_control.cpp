// clique_finder_control.C
//
// Anton Betten
//
// started:  October 11, 2018




#include "foundations.h"

clique_finder_control::clique_finder_control()
{
	f_rainbow = FALSE;
	f_file = FALSE;
	fname_graph = NULL;
	f_nonrecursive = FALSE;
	f_output_solution_raw = FALSE;
	f_output_file = FALSE;
	output_file = NULL;
	f_maxdepth = FALSE;
	maxdepth = 0;
	f_restrictions = FALSE;
	nb_restrictions = 0;
	f_tree = FALSE;
	f_decision_nodes_only = FALSE;
	fname_tree = NULL;
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
		int argc, const char **argv)
{
	int i;

	cout << "clique_finder_control::parse_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname_graph = argv[++i];
			cout << "-file " << fname_graph << endl;
		}
		else if (strcmp(argv[i], "-rainbow") == 0) {
			f_rainbow = TRUE;
			cout << "-rainbow " << endl;
		}
		else if (strcmp(argv[i], "-nonrecursive") == 0) {
			f_nonrecursive = TRUE;
			cout << "-nonrecursive " << endl;
		}
		else if (strcmp(argv[i], "-tree") == 0) {
			f_tree = TRUE;
			f_decision_nodes_only = FALSE;
			fname_tree = argv[++i];
			cout << "-tree " << fname_tree << endl;
		}
		else if (strcmp(argv[i], "-tree_decision_nodes_only") == 0) {
			f_tree = TRUE;
			f_decision_nodes_only = TRUE;
			fname_tree = argv[++i];
			cout << "-tree_decision_nodes_only " << fname_tree << endl;
		}
		else if (strcmp(argv[i], "-output_file") == 0) {
			f_output_file = TRUE;
			output_file = argv[++i];
			cout << "-output_file " << output_file << endl;
		}
		else if (strcmp(argv[i], "-output_solution_raw") == 0) {
			f_output_solution_raw = TRUE;
			cout << "-output_solution_raw " << endl;
		}
		else if (strcmp(argv[i], "-restrictions") == 0) {
			f_restrictions = TRUE;
			int j;
			for (j = 0; TRUE; j++) {
				restrictions[j] = atoi(argv[++i]);
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
			int_vec_print(cout, restrictions, 3 * nb_restrictions);
			cout << endl;
			}
		else if (strcmp(argv[i], "-end") == 0) {
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


void clique_finder_control::all_cliques(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	colored_graph CG;
	char fname_sol[1000];

	if (f_v) {
		cout << "clique_finder_control::all_cliques" << endl;
		}
	CG.load(fname_graph, verbose_level - 1);
	if (f_output_file) {
		sprintf(fname_sol, "%s", output_file);
		}
	else {
		sprintf(fname_sol, "%s_sol.txt", CG.fname_base);
		}

	//CG.print();

	{
	ofstream fp(fname_sol);


	if (f_rainbow) {
		if (f_v) {
			cout << "clique_finder_control::all_cliques "
					"before CG.all_rainbow_cliques" << endl;
			}
		CG.all_rainbow_cliques(&fp,
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
	} else {
		cout << "clique_finder_control::all_cliques !f_rainbow" << endl;
		exit(1);
	}
	fp << -1 << " " << nb_sol << " " << nb_search_steps
		<< " " << nb_decision_steps << " " << dt << endl;
	}
	if (f_v) {
		cout << "clique_finder_control::all_cliques done" << endl;
		}
}


