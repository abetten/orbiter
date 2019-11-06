// all_rainbow_cliques.cpp
// 
// Anton Betten
// October 28, 2012
//
// 
// previously called all_cliques.cpp
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started


int main(int argc, char **argv)
{
	int i, j;
	os_interface Os;
	t0 = Os.os_ticks();
	int verbose_level = 0;
	int f_file = FALSE;	
	const char *fname = NULL;
	int f_maxdepth = FALSE;
	int maxdepth = 0;
	int print_interval = 1000;
	int f_list_of_cases = FALSE;
	const char *fname_list_of_cases = NULL;
	const char *fname_template = NULL;
	int f_prefix = FALSE;
	const char *prefix = "";
	int f_output_file = FALSE;
	const char *output_file = NULL;
	int f_tree = FALSE;
	int f_decision_nodes_only = FALSE;
	const char *fname_tree = NULL;
	int f_output_solution_raw = FALSE;
	int f_no_colors = FALSE;
	int clique_size;
	int f_solution_file = FALSE;
	int f_restrictions = FALSE;
	int restrictions[1000];
	int nb_restrictions = 0;
	int f_export_magma = FALSE;
	int f_split = FALSE;
	int split_r = 0;
	int split_m = 0;
	int f_input_list_of_files = FALSE;
	int input_list_of_files_nb = 0;
	int input_list_of_files_case[100000];
	const char *input_list_of_files_fname[100000];
	int f_success_file = FALSE;
	const char *success_fname = NULL;
	//int f_nonrecursive = FALSE;


	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
#if 0
		else if (strcmp(argv[i], "-nonrecursive") == 0) {
			f_nonrecursive = TRUE;
			cout << "-nonrecursive " << endl;
			}
#endif
		else if (strcmp(argv[i], "-success_file") == 0) {
			f_success_file = TRUE;
			success_fname = argv[++i];
			cout << "-success_file " << success_fname << endl;
			}
		else if (strcmp(argv[i], "-input_list_of_files") == 0) {
			f_input_list_of_files = TRUE;
			cout << "-input_list_of_files " << endl;
			while (TRUE) {
				input_list_of_files_case[input_list_of_files_nb] = atoi(argv[++i]);
				input_list_of_files_fname[input_list_of_files_nb] = argv[++i];
				if (input_list_of_files_case[input_list_of_files_nb] == -1) {
					i--;
					break;
					}
				cout << input_list_of_files_case[input_list_of_files_nb]
					<< " "
					<< input_list_of_files_fname[input_list_of_files_nb]
					<< endl;
				input_list_of_files_nb++;
				}
			cout << "-1" << endl;
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
		else if (strcmp(argv[i], "-list_of_cases") == 0) {
			f_list_of_cases = TRUE;
			fname_list_of_cases = argv[++i];
			fname_template = argv[++i];
			cout << "-list_of_cases " << fname_list_of_cases
					<< " " << fname_template << endl;
			}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
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
		else if (strcmp(argv[i], "-no_colors") == 0) {
			f_no_colors = TRUE;
			clique_size = atoi(argv[++i]);
			cout << "-no_colors " << clique_size << endl;
			}
		else if (strcmp(argv[i], "-solution_file") == 0) {
			f_solution_file = TRUE;
			cout << "-solution_file " << endl;
			}
		else if (strcmp(argv[i], "-restrictions") == 0) {
			f_restrictions = TRUE;
			for (j = 0; TRUE; j++) {
				restrictions[j] = atoi(argv[++i]);
				if (restrictions[j] == -1) {
					nb_restrictions = j / 3;
					break;
					}
				}
			cout << "-restrictions ";
			int_vec_print(cout, restrictions, 3 * nb_restrictions);
			cout << endl;
			}
		else if (strcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			cout << "-export_magma " << endl;
			}
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_r = atoi(argv[++i]);
			split_m = atoi(argv[++i]);
			cout << "-split " << split_r << " " << split_m << endl;
			}
		}

	graph_theory_domain Graph;
	file_io Fio;



	if (f_file) {

		if (f_export_magma) {

			cout << "export_magma" << endl;


			colored_graph CG;
			char fname_magma[1000];
			char fname_text[1000];

			strcpy(fname_magma, fname);

			strcpy(fname_text, fname);

		
			replace_extension_with(fname_magma, ".magma");
			replace_extension_with(fname_text, ".txt");

			cout << "exporting to magma as " << fname_magma << endl;

			cout << "loading graph from file " << fname << endl;
			CG.load(fname, verbose_level - 1);
			cout << "found a graph with " << CG.nb_points
					<< " points"  << endl;


			CG.export_to_magma(fname_magma, verbose_level);

			CG.export_to_text(fname_text, verbose_level);
			
			cout << "export_magma done" << endl;
			}
		else {
			cout << "no mgama export" << endl;
			}




		if (f_no_colors) {
			colored_graph CG;
			//char fname_sol[1000];
			//char fname_draw[1000];
			int nb_sol;
			unsigned long int decision_step_counter;



			cout << "finding cliques, ignoring colors" << endl;
			cout << "loading graph from file " << fname << endl;
			CG.load(fname, verbose_level - 1);
			cout << "found a graph with " << CG.nb_points << " points"  << endl;
			cout << "before CG.all_cliques_of_size_k_ignore_colors"  << endl;
			cout << "clique_size = " << clique_size << endl;


			char fname_solution[1000];
			char fname_success[1000];

			strcpy(fname_solution, fname);

			
			replace_extension_with(fname_solution, "");

			if (f_restrictions) {
				for (i = 0; i < nb_restrictions; i++) {
					sprintf(fname_solution + strlen(fname_solution),
						"_case%d_%d_%d",
						restrictions[3 * i + 0],
						restrictions[3 * i + 1],
						restrictions[3 * i + 2]);
					}
				}
			
			strcpy(fname_success, fname_solution);
			strcat(fname_solution, ".solutions");
			strcat(fname_success, ".success");
			//replace_extension_with(fname_solution, ".solutions");



			if (f_solution_file) {

				cout << "before CG.all_cliques_of_size_"
						"k_ignore_colors_"
						"and_write_solutions_to_file" << endl;
				CG.all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(
					clique_size /* target_depth */, 
					fname_solution, 
					f_restrictions, restrictions, 
					nb_sol, decision_step_counter, verbose_level);
				cout << "after CG.all_cliques_of_size_"
						"k_ignore_colors_"
						"and_write_solutions_to_file" << endl;
				}
			else {
				int *Sol;

				cout << "before CG.all_cliques_of_size_"
						"k_ignore_colors" << endl;
				CG.all_cliques_of_size_k_ignore_colors(
					clique_size /* target_depth */,
					Sol, nb_sol, decision_step_counter, verbose_level);
				cout << "after CG.all_cliques_of_size_"
						"k_ignore_colors" << endl;
				FREE_int(Sol);
				}


			{
			ofstream fp(fname_success);
			fp << "success" << endl;
			}
			cout << "nb_sol = " << nb_sol << endl;
			cout << "decision_step_counter = "
				<< decision_step_counter << endl;
			}
		else {


#if 0
			if (f_draw) {
				cout << "before colored_graph_draw" << endl;
				colored_graph_draw(fname, 
					xmax_in, ymax_in, xmax_out, ymax_out, 
					scale, line_width, 
					verbose_level - 1);
				cout << "after colored_graph_draw" << endl;
				}
#endif


#if 0
			if (f_nonrecursive) {
				int nb_sol, nb_backtrack_nodes;

				cout << "finding rainbow cliques, calling "
						"colored_graph_all_rainbow_cliques_nonrecursive"
						<< endl;

				nb_sol = Graph.colored_graph_all_rainbow_cliques_nonrecursive(
					fname, nb_backtrack_nodes, verbose_level);

				cout << "nb_sol = " << nb_sol << endl;
				cout << "nb_backtrack_nodes = " << nb_backtrack_nodes
						<< endl;
			
				cout << "finding rainbow cliques, after "
					"colored_graph_all_rainbow_cliques_nonrecursive"
						<< endl;
				}
			else {
#endif
			cout << "finding rainbow cliques, "
				"calling colored_graph_all_cliques" << endl;

			unsigned long int search_steps, decision_steps;
			int nb_sol, dt;


			Graph.colored_graph_all_cliques(fname, f_output_solution_raw,
				f_output_file, output_file,
				f_maxdepth, maxdepth,
				f_restrictions, restrictions,
				f_tree, f_decision_nodes_only, fname_tree,
				print_interval,
				search_steps, decision_steps, nb_sol, dt,
				verbose_level);
				// in GALOIS/colored_graph.cpp

			cout << "nb_sol = " << nb_sol << endl;

			cout << "after colored_graph_all_cliques" << endl;
			}

			//}
		}
	else if (f_list_of_cases) {
		long int *list_of_cases;
		int nb_cases;
		char fname_sol[1000];
		char fname_stats[1000];
		char split[1000];
		

		if (f_split) {
			sprintf(split, "_split_%d_%d.txt", split_r, split_m);
			}


		if (f_output_file) {
			sprintf(fname_sol, "%s", output_file);
			sprintf(fname_stats, "%s", output_file);
			}
		else {
			sprintf(fname_sol, "solutions_%s", fname_list_of_cases);
			sprintf(fname_stats, "statistics_%s", fname_list_of_cases);
			}


		if (f_split) {
			replace_extension_with(fname_sol, split);
			replace_extension_with(fname_stats, split);
			}
		replace_extension_with(fname_stats, "_stats.csv");


		Fio.read_set_from_file(fname_list_of_cases,
				list_of_cases, nb_cases, verbose_level);
		cout << "nb_cases=" << nb_cases << endl;

		Graph.colored_graph_all_cliques_list_of_cases(
			list_of_cases, nb_cases, f_output_solution_raw,
			fname_template, 
			fname_sol, fname_stats, 
			f_split, split_r, split_m, 
			f_maxdepth, maxdepth, 
			f_prefix, prefix, 
			print_interval, 
			verbose_level);
		
		FREE_lint(list_of_cases);
		cout << "all_rainbow_cliques.out written file " << fname_sol
				<< " of size " << Fio.file_size(fname_sol) << endl;
		cout << "all_rainbow_cliques.out written file " << fname_stats
				<< " of size " << Fio.file_size(fname_stats) << endl;
		}
	else if (f_input_list_of_files) {
		
		cout << "input_list_of_files with " << input_list_of_files_nb
				<< " input files" << endl;

		char fname_sol[1000];
		char fname_stats[1000];

		if (f_output_file) {
			sprintf(fname_sol, "%s", output_file);
			sprintf(fname_stats, "%s", output_file);
			}
		else {
			sprintf(fname_sol, "solutions_%s", fname_list_of_cases);
			sprintf(fname_stats, "statistics_%s", fname_list_of_cases);
			}


		replace_extension_with(fname_stats, "_stats.csv");


		Graph.colored_graph_all_cliques_list_of_files(
			input_list_of_files_nb /* nb_cases */,
			input_list_of_files_case /* Case_number */,
			input_list_of_files_fname /* Case_fname */,
			f_output_solution_raw, 
			fname_sol, fname_stats, 
			f_maxdepth, maxdepth, 
			f_prefix, prefix, 
			print_interval, 
			verbose_level);

		cout << "all_rainbow_cliques.out written file " << fname_sol
			<< " of size " << Fio.file_size(fname_sol) << endl;
		cout << "all_rainbow_cliques.out written file " << fname_stats
			<< " of size " << Fio.file_size(fname_stats) << endl;



		}
	else {
		cout << "Please use options -file or -list_of_cases or "
				"-input_list_of_files" << endl;
		exit(1);
		}


	if (f_success_file) {
		{
		ofstream fp(success_fname);
		fp << "Success" << endl;
		}
		cout << "Written file " << success_fname << " of size "
				<< Fio.file_size(success_fname) << endl;
		}

	cout << "all_rainbow_cliques.out is done" << endl;
	the_end(t0);
	//the_end_quietly(t0);

}


