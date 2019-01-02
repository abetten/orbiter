// solve_diophant.C
// 
// Anton Betten
// May 17, 2015
//
// 

#include "orbiter.h"


// global data:

int t0; // the system time when the program started


int main(int argc, char **argv)
{
	int i;
	t0 = os_ticks();
	int verbose_level = 0;
	int f_file = FALSE;
	const char *fname = NULL;
	int f_general_format = FALSE;
	//int f_maxdepth = FALSE;
	//int maxdepth = 0;
	//int print_interval = 1000;
	int f_list_of_cases = FALSE;
	const char *fname_list_of_cases = NULL;
	const char *fname_template = NULL;
	int f_prefix = FALSE;
	const char *prefix = NULL;
	int f_output_file = FALSE;
	const char *output_file = NULL;
	int f_print = FALSE;
	int f_print_tex = FALSE;
	int f_draw = FALSE;
	int f_draw_solutions = FALSE;
	int f_analyze = FALSE;
	int f_tree = FALSE;
	int f_decision_nodes_only = FALSE;
	const char *fname_tree = NULL;
	int f_coordinates = FALSE;
	int xmax_in = ONE_MILLION;
	int ymax_in = ONE_MILLION;
	int xmax_out = ONE_MILLION;
	int ymax_out = ONE_MILLION;
	int f_output_solution_raw = FALSE;
	int f_test = FALSE;
	const char *solution_file = NULL;
	int f_make_clique_graph = FALSE;
	int f_RHS = FALSE;
	int RHS_value = 0;
	//int f_sum = FALSE;
	//int sum_value = 0;
	int RHS_row_nb = 0;
	int RHS_row[1000];
	int RHS_row_value[1000];
	int f_betten = FALSE;
	int f_McKay = FALSE;
	int nb_xmax = 0;
	int xmax_value[1000];
	int xmax_variable[1000];
	int f_scale = FALSE;
	double scale = .45;
	int f_line_width = FALSE;
	double line_width = 1.5;

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
		else if (strcmp(argv[i], "-general_format") == 0) {
			f_general_format = TRUE;
			cout << "-general_format" << endl;
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
		else if (strcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print " << endl;
			}
		else if (strcmp(argv[i], "-print_tex") == 0) {
			f_print_tex = TRUE;
			cout << "-print_tex " << endl;
			}
		else if (strcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
			cout << "-draw " << endl;
			}
		else if (strcmp(argv[i], "-draw_solutions") == 0) {
			f_draw_solutions = TRUE;
			cout << "-draw_solutions " << endl;
			}
		else if (strcmp(argv[i], "-analyze") == 0) {
			f_analyze = TRUE;
			cout << "-analyze " << endl;
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
			cout << "-coordinates " << xmax_in << " " << ymax_in
				<< " " << xmax_out << " " << ymax_out << endl;
			}
		else if (strcmp(argv[i], "-test") == 0) {
			f_test = TRUE;
			solution_file = argv[++i];
			cout << "-test " << solution_file << endl;
			}
		else if (strcmp(argv[i], "-make_clique_graph") == 0) {
			f_make_clique_graph = TRUE;
			cout << "-make_clique_graph " << endl;
			}
		else if (strcmp(argv[i], "-RHS") == 0) {
			f_RHS = TRUE;
			RHS_value = atoi(argv[++i]);
			cout << "-RHS " << RHS_value << endl;
			}
		else if (strcmp(argv[i], "-RHS_row") == 0) {
			RHS_row[RHS_row_nb] = atoi(argv[++i]);
			RHS_row_value[RHS_row_nb] = atoi(argv[++i]);
			cout << "-RHS_row " << RHS_row[RHS_row_nb]
					<< " " << RHS_row_value[RHS_row_nb] << endl;
			RHS_row_nb++;
			}
#if 0
		else if (strcmp(argv[i], "-sum") == 0) {
			f_sum = TRUE;
			sum_value = atoi(argv[++i]);
			cout << "-sum " << sum_value << endl;
			}
#endif
		else if (strcmp(argv[i], "-betten") == 0) {
			f_betten = TRUE;
			cout << "-betten " << endl;
			}
		else if (strcmp(argv[i], "-McKay") == 0) {
			f_McKay = TRUE;
			cout << "-McKay " << endl;
			}
		else if (strcmp(argv[i], "-xmax") == 0) {
			xmax_variable[nb_xmax] = atoi(argv[++i]);
			xmax_value[nb_xmax] = atoi(argv[++i]);
			nb_xmax++;
			cout << "-xmax " << xmax_variable[nb_xmax - 1]
				<< " " << xmax_value[nb_xmax - 1] << endl;
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
		
		}


	if (!f_file) {
		cout << "Please use options -file" << endl;
		exit(1);
		}

	int f_v = (verbose_level >= 1);

	//int search_steps, decision_steps, nb_sol, dt;

	diophant *Dio;

	Dio = NEW_OBJECT(diophant);

	if (f_v) {
		cout << "reading file " << fname << endl;
		}

	if (f_general_format) {
		Dio->read_general_format(fname, verbose_level);
		}
	else {
		if (is_xml_file(fname)) {
			{
			char label[1000];
			ifstream f(fname);
			Dio->read_xml(f, label);
			}
			}
		else {
			Dio->read_compact_format(fname, verbose_level);
			}
		}

	Dio->eliminate_zero_rows_quick(verbose_level);

#if 0
	if (f_sum) {
		Dio->f_has_sum = TRUE;
		Dio->sum = sum_value;
		}
#endif

	if (f_v) {
		cout << "found a system with " << Dio->m << " rows and "
				<< Dio->n << " columns, the sum is " << Dio->sum << endl;
		}


	if (f_RHS) {
		Dio->init_RHS(RHS_value, verbose_level);
		}

	int r, val;
	
	for (i = 0; i < RHS_row_nb; i++) {
		r = RHS_row[i];
		val = RHS_row_value[i];
		if (r < 0) {
			Dio->RHSi(Dio->m + r) = val;
			}
		else {
			Dio->RHSi(r) = val;
			}
		}


	if (f_draw) {
		char fname_base[1000];

		sprintf(fname_base, "%s", fname);
		replace_extension_with(fname_base, "_drawing");		
		//Dio->draw_it(fname_base, xmax_in, ymax_in, xmax_out, ymax_out);
		Dio->draw_partitioned(fname_base,
			xmax_in, ymax_in, xmax_out, ymax_out,
			FALSE, 0, 0,
			verbose_level);
		}

	if (f_make_clique_graph) {

		char fname_base[1000];


		sprintf(fname_base, "%s", fname);
		replace_extension_with(fname_base, "_clique_graph.bin");

		cout << "making clique_graph" << endl;
		
		colored_graph *CG;
		
		Dio->make_clique_graph(CG, verbose_level);



		cout << "saving clique_graph to file " << fname_base << endl;

		CG->save(fname_base, verbose_level + 10);

		cout << "after CG->save" << endl;

		replace_extension_with(fname_base, "_drawing");
		
		CG->draw_it(fname_base,
				xmax_in, ymax_in, xmax_out, ymax_out, scale, line_width);

		
		FREE_OBJECT(CG);
		}

#if 0
	if (f_make_clique_graph) {
		Dio->make_clique_graph(clique_graph_file, verbose_level);
		}
#endif

	Dio->append_equation();

	int j;

	if (f_v) {
		cout << "appending one equation for the sum" << endl;
		}

	i = Dio->m - 1;
	for (j = 0; j < Dio->n; j++) {
		Dio->Aij(i, j) = 1;
		}
	Dio->type[i] = t_EQ;
	Dio->RHS[i] = Dio->sum;

	Dio->f_x_max = TRUE;
	for (j = 0; j < Dio->n; j++) {
		Dio->x_max[j] = 1;
		}

	for (i = 0; i < nb_xmax; i++) {
		Dio->x_max[xmax_variable[i]] = xmax_value[i];
		}

	if (f_print) {
		Dio->print();
		Dio->print_tight();
		}

	if (f_print_tex) {
		char fname_base[1000];

		sprintf(fname_base, "%s", fname);
		replace_extension_with(fname_base, "_print.tex");
		{
			ofstream fp(fname_base);
			Dio->latex_it(fp);
		}
		cout << "Written file " << fname_base << " of size "
				<< file_size(fname_base) << endl;
		}
	
	if (f_analyze) {
		Dio->analyze(verbose_level);
		}

	if (f_test) {
		cout << "testing solutions from file " << solution_file << endl;
		Dio->test_solution_file(solution_file, verbose_level);
		}

	else {

		if (f_betten) {
			cout << "solving with betten" << endl;
			Dio->solve_all_betten(verbose_level - 2);
			}
		else if (f_McKay) {
			int nb_backtrack_nodes;
			
			cout << "solving with mckay" << endl;
			Dio->solve_all_mckay(nb_backtrack_nodes, verbose_level - 2);
			Dio->nb_steps_betten = nb_backtrack_nodes;
			}
		else {
			cout << "solving with DLX" << endl;
			Dio->solve_all_DLX_with_RHS(f_tree, fname_tree, verbose_level - 2);
			}
		cout << "Found " << Dio->_resultanz << " solutions with "
				<< Dio->nb_steps_betten << " backtrack steps" << endl;

		if (f_output_file) {
			if (f_draw_solutions) {
				Dio->m--; // delete the equation for the sum
				Dio->draw_solutions(output_file, verbose_level);
				}
			else {
				Dio->write_solutions(output_file, verbose_level);
				}
			}
		}

	
	FREE_OBJECT(Dio);


	cout << "solve_diophant.out is done" << endl;
	the_end(t0);
	//the_end_quietly(t0);

	return 0;

}

