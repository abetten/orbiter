// solve_diophant.C
// 
// Anton Betten
// May 17, 2015
//
// 

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started


int main(int argc, char **argv)
{
	INT i;
	t0 = os_ticks();
	INT verbose_level = 0;
	INT f_file = FALSE;
	const BYTE *fname = NULL;
	INT f_general_format = FALSE;
	//INT f_maxdepth = FALSE;
	//INT maxdepth = 0;
	//INT print_interval = 1000;
	INT f_list_of_cases = FALSE;
	const BYTE *fname_list_of_cases = NULL;
	const BYTE *fname_template = NULL;
	INT f_prefix = FALSE;
	const BYTE *prefix = NULL;
	INT f_output_file = FALSE;
	const BYTE *output_file = NULL;
	INT f_print = FALSE;
	INT f_print_tex = FALSE;
	INT f_draw = FALSE;
	INT f_draw_solutions = FALSE;
	INT f_analyze = FALSE;
	INT f_tree = FALSE;
	INT f_decision_nodes_only = FALSE;
	const BYTE *fname_tree = NULL;
	INT f_coordinates = FALSE;
	INT xmax_in = ONE_MILLION;
	INT ymax_in = ONE_MILLION;
	INT xmax_out = ONE_MILLION;
	INT ymax_out = ONE_MILLION;
	INT f_output_solution_raw = FALSE;
	INT f_test = FALSE;
	const BYTE *solution_file = NULL;
	INT f_make_clique_graph = FALSE;
	INT f_RHS = FALSE;
	INT RHS_value = 0;
	INT f_sum = FALSE;
	INT sum_value = 0;
	INT RHS_row_nb = 0;
	INT RHS_row[1000];
	INT RHS_row_value[1000];
	INT f_betten = FALSE;
	INT f_mckay = FALSE;
	INT nb_xmax = 0;
	INT xmax_value[1000];
	INT xmax_variable[1000];
	INT f_scale = FALSE;
	double scale = .45;
	INT f_line_width = FALSE;
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
			cout << "-coordinates " << xmax_in << " " << ymax_in << " " << xmax_out << " " << ymax_out << endl;
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
			cout << "-RHS_row " << RHS_row[RHS_row_nb] << " " << RHS_row_value[RHS_row_nb] << endl;
			RHS_row_nb++;
			}
		else if (strcmp(argv[i], "-sum") == 0) {
			f_sum = TRUE;
			sum_value = atoi(argv[++i]);
			cout << "-sum " << sum_value << endl;
			}
		else if (strcmp(argv[i], "-betten") == 0) {
			f_betten = TRUE;
			cout << "-betten " << endl;
			}
		else if (strcmp(argv[i], "-mckay") == 0) {
			f_mckay = TRUE;
			cout << "-mckay " << endl;
			}
		else if (strcmp(argv[i], "-xmax") == 0) {
			xmax_variable[nb_xmax] = atoi(argv[++i]);
			xmax_value[nb_xmax] = atoi(argv[++i]);
			nb_xmax++;
			cout << "-xmax " << xmax_variable[nb_xmax - 1] << " " << xmax_value[nb_xmax - 1] << endl;
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

	INT f_v = (verbose_level >= 1);

	//INT search_steps, decision_steps, nb_sol, dt;

	diophant *Dio;

	Dio = new diophant;

	if (f_v) {
		cout << "reading file " << fname << endl;
		}

	if (f_general_format) {
		Dio->read_general_format(fname, verbose_level);
		}
	else {
		if (is_xml_file(fname)) {
			{
			BYTE label[1000];
			ifstream f(fname);
			Dio->read_xml(f, label);
			}
			}
		else {
			Dio->read_compact_format(fname, verbose_level);
			}
		}

	Dio->eliminate_zero_rows_quick(verbose_level);

	if (f_sum) {
		Dio->sum = sum_value;
		}

	if (f_v) {
		cout << "found a system with " << Dio->m << " rows and " << Dio->n << " columns, the sum is " << Dio->sum << endl;
		}


	if (f_RHS) {
		Dio->init_RHS(RHS_value, verbose_level);
		}

	INT r, val;
	
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
		BYTE fname_base[1000];

		sprintf(fname_base, "%s", fname);
		replace_extension_with(fname_base, "_drawing");		
		//Dio->draw_it(fname_base, xmax_in, ymax_in, xmax_out, ymax_out);
		Dio->draw_partitioned(fname_base, xmax_in, ymax_in, xmax_out, ymax_out, 
			FALSE, 0, NULL, 
			verbose_level);
		}

	if (f_make_clique_graph) {

		BYTE fname_base[1000];


		sprintf(fname_base, "%s", fname);
		replace_extension_with(fname_base, "_clique_graph.bin");

		cout << "making clique_graph" << endl;
		
		colored_graph *CG;
		
		Dio->make_clique_graph(CG, verbose_level);



		cout << "saving clique_graph to file " << fname_base << endl;

		CG->save(fname_base, verbose_level + 10);

		cout << "after CG->save" << endl;

		replace_extension_with(fname_base, "_drawing");
		
		CG->draw_it(fname_base, xmax_in, ymax_in, xmax_out, ymax_out, scale, line_width);

		
		delete CG;
		}

#if 0
	if (f_make_clique_graph) {
		Dio->make_clique_graph(clique_graph_file, verbose_level);
		}
#endif

	Dio->append_equation();

	INT j;

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
		BYTE fname_base[1000];

		sprintf(fname_base, "%s", fname);
		replace_extension_with(fname_base, "_print.tex");
		{
			ofstream fp(fname_base);
			Dio->latex_it(fp);
		}
		cout << "Written file " << fname_base << " of size " << file_size(fname_base) << endl;
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
		else if (f_mckay) {
			INT nb_backtrack_nodes;
			
			cout << "solving with mckay" << endl;
			Dio->solve_all_mckay(nb_backtrack_nodes, verbose_level - 2);
			Dio->nb_steps_betten = nb_backtrack_nodes;
			}
		else {
			cout << "solving with DLX" << endl;
			Dio->solve_all_DLX_with_RHS(f_tree, fname_tree, verbose_level - 2);
			}
		cout << "Found " << Dio->_resultanz << " solutions with " << Dio->nb_steps_betten << " backtrack steps" << endl;

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

	
	delete Dio;


	cout << "solve_diophant.out is done" << endl;
	the_end(t0);
	//the_end_quietly(t0);

	return 0;

}

