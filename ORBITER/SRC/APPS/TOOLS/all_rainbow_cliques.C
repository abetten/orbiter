// all_rainbow_cliques.C
// 
// Anton Betten
// October 28, 2012
//
// 
// previously called all_cliques.C
//

#include "orbiter.h"
#include "discreta.h"


// global data:

INT t0; // the system time when the program started



int main(int argc, char **argv)
{
	INT i, j;
	t0 = os_ticks();
	INT verbose_level = 0;
	INT f_file = FALSE;	
	const BYTE *fname = NULL;
	INT f_maxdepth = FALSE;
	INT maxdepth = 0;
	INT print_interval = 1000;
	INT f_list_of_cases = FALSE;
	const BYTE *fname_list_of_cases = NULL;
	const BYTE *fname_template = NULL;
	INT f_prefix = FALSE;
	const BYTE *prefix = NULL;
	INT f_output_file = FALSE;
	const BYTE *output_file = NULL;
	INT f_draw = FALSE;
	INT f_tree = FALSE;
	INT f_decision_nodes_only = FALSE;
	const BYTE *fname_tree = NULL;
	INT f_coordinates = FALSE;
	INT xmax_in = ONE_MILLION;
	INT ymax_in = ONE_MILLION;
	INT xmax_out = ONE_MILLION;
	INT ymax_out = ONE_MILLION;
	INT f_output_solution_raw = FALSE;
	INT f_no_colors = FALSE;
	INT clique_size;
	INT f_solution_file = FALSE;
	INT f_scale = FALSE;
	double scale = .45;
	INT f_line_width = FALSE;
	double line_width = 1.5;
	INT f_restrictions = FALSE;
	INT restrictions[1000];
	INT nb_restrictions = 0;


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
			cout << "-list_of_cases " << fname_list_of_cases << " " << fname_template << endl;
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
		else if (strcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
			cout << "-draw " << endl;
			}
		else if (strcmp(argv[i], "-output_solution_raw") == 0) {
			f_output_solution_raw = TRUE;
			cout << "-output_solution_raw " << endl;
			}
		else if (strcmp(argv[i], "-coordinates") == 0) {
			f_coordinates = TRUE;
			xmax_in = atoi(argv[++i]);
			ymax_in = atoi(argv[++i]);
			xmax_out = atoi(argv[++i]);
			ymax_out = atoi(argv[++i]);
			cout << "-coordinates " << xmax_in << " " << ymax_in << " " << xmax_out << " " << ymax_out << endl;
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
		else if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			sscanf(argv[++i], "%lf", &scale);
			cout << "-scale " << scale << endl;
			}
		else if (strcmp(argv[i], "-line_width") == 0) {
			f_line_width = TRUE;
			sscanf(argv[++i], "%lf", &line_width);
			cout << "-line_width " << line_width << endl;
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
			INT_vec_print(cout, restrictions, 3 * nb_restrictions);
			cout << endl;
			}
		}

	if (f_file) {

		if (f_no_colors) {
			colored_graph CG;
			//BYTE fname_sol[1000];
			//BYTE fname_draw[1000];
			INT nb_sol;
			INT decision_step_counter;



			cout << "finding cliques, ignoring colors" << endl;
			cout << "loading graph from file " << fname << endl;
			CG.load(fname, verbose_level - 1);
			cout << "found a graph with " << CG.nb_points << " points"  << endl;
			cout << "before CG.all_cliques_of_size_k_ignore_colors"  << endl;
			cout << "clique_size = " << clique_size << endl;


			BYTE fname_solution[1000];
			BYTE fname_success[1000];

			strcpy(fname_solution, fname);

			
			replace_extension_with(fname_solution, "");

			if (f_restrictions) {
				for (i = 0; i < nb_restrictions; i++) {
					sprintf(fname_solution + strlen(fname_solution), "_case%ld_%ld_%ld", 
						restrictions[3 * i + 0], restrictions[3 * i + 1], restrictions[3 * i + 2]);
					}
				}
			
			strcpy(fname_success, fname_solution);
			strcat(fname_solution, ".solutions");
			strcat(fname_success, ".success");
			//replace_extension_with(fname_solution, ".solutions");



			if (f_solution_file) {

				cout << "before CG.all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file" << endl;
				CG.all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(
					clique_size /* target_depth */, 
					fname_solution, 
					f_restrictions, restrictions, 
					nb_sol, decision_step_counter, verbose_level);
				cout << "after CG.all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file" << endl;
				}
			else {
				cout << "before CG.all_cliques_of_size_k_ignore_colors" << endl;
				CG.all_cliques_of_size_k_ignore_colors(clique_size /* target_depth */, 
					nb_sol, decision_step_counter, verbose_level);
				cout << "after CG.all_cliques_of_size_k_ignore_colors" << endl;
				}


			{
			ofstream fp(fname_success);
			fp << "success" << endl;
			}
			cout << "nb_sol = " << nb_sol << endl;
			cout << "decision_step_counter = " << decision_step_counter << endl;
			}
		else {



			if (f_draw) {
				cout << "before colored_graph_draw" << endl;
				colored_graph_draw(fname, 
					xmax_in, ymax_in, xmax_out, ymax_out, 
					scale, line_width, 
					verbose_level - 1);
				cout << "after colored_graph_draw" << endl;
				}


			cout << "finding rainbow cliques, calling colored_graph_all_cliques" << endl;

			INT search_steps, decision_steps, nb_sol, dt;


			colored_graph_all_cliques(fname, f_output_solution_raw, 
				//f_draw, xmax_in, ymax_in, xmax_out, ymax_out, 
				f_output_file, output_file, 
				f_maxdepth, maxdepth, 
				f_restrictions, restrictions, 
				f_tree, f_decision_nodes_only, fname_tree,  
				print_interval, 
				search_steps, decision_steps, nb_sol, dt, 
				//scale, line_width, 
				verbose_level);
				// in GALOIS/colored_graph.C


			cout << "after colored_graph_all_cliques" << endl;

			}
		}
	else if (f_list_of_cases) {
		INT *list_of_cases;
		INT nb_cases;
		BYTE fname_sol[1000];
		BYTE fname_stats[1000];
		
		if (f_output_file) {
			sprintf(fname_sol, "%s", output_file);
			sprintf(fname_stats, "%s", output_file);
			replace_extension_with(fname_stats, "_stats.csv");
			}
		else {
			sprintf(fname_sol, "solutions_%s", fname_list_of_cases);
			sprintf(fname_stats, "statistics_%s", fname_list_of_cases);
			replace_extension_with(fname_stats, ".csv");
			}
		read_set_from_file(fname_list_of_cases, list_of_cases, nb_cases, verbose_level);
		cout << "nb_cases=" << nb_cases << endl;

		colored_graph_all_cliques_list_of_cases(list_of_cases, nb_cases, f_output_solution_raw, 
			f_draw, xmax_in, ymax_in, xmax_out, ymax_out, 
			fname_template, 
			fname_sol, fname_stats, 
			f_maxdepth, maxdepth, 
			f_prefix, prefix, 
			print_interval, 
			scale, line_width, 
			verbose_level);
		
		FREE_INT(list_of_cases);
		cout << "all_rainbow_cliques.out written file " << fname_sol << " of size " << file_size(fname_sol) << endl;
		cout << "all_rainbow_cliques.out written file " << fname_stats << " of size " << file_size(fname_stats) << endl;
		}
	else {
		cout << "Please use options -file or -list_of_cases" << endl;
		exit(1);
		}

	cout << "all_rainbow_cliques.out is done" << endl;
	the_end(t0);
	//the_end_quietly(t0);

}


