// tdo_refine.C
// Anton Betten
//
// started:  Dec 26 2006

#include "orbiter.h"


#define MY_BUFSIZE 1000000

const char *version = "tdo_refine 7/31 2008";


void print_distribution(ostream &ost, 
	int *types, int nb_types, int type_len,  
	int *distributions, int nb_distributions);
int compare_func_int_vec(void *a, void *b, void *data);
int compare_func_int_vec_inverse(void *a, void *b, void *data);
void distribution_reverse_sorting(int f_increasing, 
	int *types, int nb_types, int type_len,  
	int *distributions, int nb_distributions);

void print_usage()
{
	cout << "usage: tdo_refine.out [options] <tdo-file>" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <verbose_level> " << endl;
	cout << "   Specify the amount of text output" << endl;
	cout << "   the higher verbose_level is, the more output" << endl;
	cout << "   verbose_level = 0 means no output" << endl;
	cout << "-lambda3 <lambda3> <k>" << endl;
	cout << "   do a 3-design parameter refinement" << endl;
	cout << "   lambda3 is the number of blocks on 3 points" << endl;
	cout << "   k is the (constant) block size" << endl;
	cout << "-scale <n>" << endl;
	cout << "   when doing the refinement," << endl;
	cout << "   consider only refinements where each class in" << endl;
	cout << "   the refinement has size a multiple of n" << endl;
	cout << "-select <label>" << endl;
	cout << "   select the TDO whose label is <label>" << endl;
	cout << "-range <f> <l>" << endl;
	cout << "   select the TDO in interval [f..f+l-1]" << endl;
	cout << "   where counting starts from 1 (!!!)" << endl;
	cout << "-solution <n> <file>" << endl;
	cout << "   Read the solutions to system <n> from <file>" << endl;
	cout << "   rather than trying to compute them" << endl;
	cout << "   This option can appear repeatedly" << endl;
	cout << "-o1 <n>" << endl;
	cout << "   omit the last n blocks from refinement (1st system - types)" << endl;
	cout << "-o2 <n>" << endl;
	cout << "   omit the last n blocks from refinement (2nd system - distribution)" << endl;
	cout << "-D1_upper_bound_x0 <n>" << endl;
	cout << "   upper bound <n> for x[0] in the first system (column refinement only!)" << endl;
	cout << "-reverse" << endl;
	cout << "   reverseordering of refinements" << endl;
	cout << "-reverse_inverse" << endl;
	cout << "   reverse ordering of refinements increasing" << endl;
	cout << "-nopacking" << endl;
	cout << "   Do not use inequalities based on packing numbers" << endl;
	cout << "-dual_is_linear_space" << endl;
	cout << "   Dual is a linear space, too (affect refine rows)" << endl;
	cout << "-once" << endl;
	cout << "   When refining, only find the first refinement" << endl;
	cout << "-mckay" << endl;
	cout << "   Use McKay solver for solving the second system" << endl;
}

typedef class tdo_parameter_calculation tdo_parameter_calculation;

//! main class for tdo_refine to refine the parameters of a linear space

class tdo_parameter_calculation {
	public:
	
int t0;
	int cnt;
	char *p_buf;
	char str[1000];
	char ext[1000];
	char *fname_in;
	//char fname_base[1000];
	char fname_out[1000];
	int verbose_level;
	int f_lambda3;
	int lambda3, block_size;
	int f_scale;
	int scaling;
	int f_range;
	int range_first, range_len;
	int f_select;
	char *select_label;
	int f_omit1;
	int omit1;
	int f_omit2;
	int omit2;
	int f_D1_upper_bound_x0;
	int D1_upper_bound_x0;
	int f_reverse;
	int f_reverse_inverse;
	int f_use_packing_numbers;
	int f_dual_is_linear_space;
	int f_do_the_geometric_test;
	int f_once;
	int f_use_mckay_solver;

	
	geo_parameter GP;
	
	geo_parameter GP2;

	//int new_part[10000];
	//int new_entries[1000000];


	int f_doit;
	int nb_written, nb_written_tactical, nb_tactical;
	int cnt_second_system;
	solution_file_data *Sol;

	void read_arguments(int argc, char **argv);
	void main_loop();
	void do_it(ofstream &g, int verbose_level);
	void do_row_refinement(ofstream &g, tdo_scheme &G, partitionstack &P, int verbose_level);
	void do_col_refinement(ofstream &g, tdo_scheme &G, partitionstack &P, int verbose_level);
	void do_all_row_refinements(char *label_in, ofstream &g, tdo_scheme &G, 
		int *point_types, int nb_point_types, int point_type_len, 
		int *distributions, int nb_distributions, int &nb_tactical, int verbose_level);
	void do_all_column_refinements(char *label_in, ofstream &g, tdo_scheme &G, 
		int *line_types, int nb_line_types, int line_type_len, 
		int *distributions, int nb_distributions, int &nb_tactical, int verbose_level);
	int do_row_refinement(int t, char *label_in, ofstream &g, tdo_scheme &G, 
		int *point_types, int nb_point_types, int point_type_len, 
		int *distributions, int nb_distributions, int verbose_level);
		// returns TRUE or FALSE depending on whether the 
		// refinement gave a tactical decomposition
	int do_column_refinement(int t, char *label_in, ofstream &g, tdo_scheme &G, 
		int *line_types, int nb_line_types, int line_type_len, 
		int *distributions, int nb_distributions, int verbose_level);
		// returns TRUE or FALSE depending on whether the 
		// refinement gave a tactical decomposition
};

int main(int argc, char **argv)
{
	tdo_parameter_calculation *G;
	
	cout << version << endl;
	
	G = new tdo_parameter_calculation;
	
	G->t0 = os_ticks();
	
	G->verbose_level = 0;
	G->f_lambda3 = FALSE;
	G->f_scale = FALSE;
	G->f_range = FALSE;
	G->f_select = FALSE;
	G->f_omit1 = FALSE;
	G->f_omit2 = FALSE;
	G->f_D1_upper_bound_x0 = FALSE;

	G->nb_written = 0;
	G->nb_written_tactical = 0;
	G->cnt_second_system = 0;
	
	G->Sol = new solution_file_data;
	G->Sol->nb_solution_files = 0;

	G->GP2.part_nb_alloc = 10000;
	G->GP2.entries_nb_alloc = 1000000;
	G->GP2.part = new int[G->GP2.part_nb_alloc];
	G->GP2.entries = new int[G->GP2.entries_nb_alloc];

	
	G->f_reverse = FALSE;
	G->f_reverse_inverse = FALSE;
	G->f_use_packing_numbers = TRUE;
	G->f_dual_is_linear_space = FALSE;
	G->f_do_the_geometric_test = FALSE;
	G->f_once = FALSE;
	G->f_use_mckay_solver = FALSE;

	G->read_arguments(argc, argv);
	G->main_loop();
	
	cout << "time: ";
	time_check(cout, G->t0);
	cout << endl;
}

void tdo_parameter_calculation::read_arguments(int argc, char **argv)
{
	int i;
	
	if (argc <= 1) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		if (strcmp(argv[i], "-lambda3") == 0) {
			f_lambda3 = TRUE;
			lambda3 = atoi(argv[++i]);
			block_size = atoi(argv[++i]);
			cout << "-lambda3 " << lambda3 << " " << block_size << endl;
			}
		if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			scaling = atoi(argv[++i]);
			cout << "-scale " << scaling << endl;
			}
		if (strcmp(argv[i], "-solution") == 0) {
			//f_solution = TRUE;
			Sol->system_no[Sol->nb_solution_files] = atoi(argv[++i]);
			Sol->solution_file[Sol->nb_solution_files] = argv[++i];
			cout << "-solution " << Sol->system_no[Sol->nb_solution_files] 
				<< " " << Sol->solution_file[Sol->nb_solution_files] << endl;
			Sol->nb_solution_files++;
			}
		else if (strcmp(argv[i], "-range") == 0) {
			f_range = TRUE;
			range_first = atoi(argv[++i]);
			range_len = atoi(argv[++i]);
			cout << "-range " << range_first << " " << range_len << endl;
			}
		else if (strcmp(argv[i], "-select") == 0) {
			f_select = TRUE;
			select_label = argv[++i];
			cout << "-select " << select_label << endl;
			}
		else if (strcmp(argv[i], "-o1") == 0) {
			f_omit1 = TRUE;
			omit1 = atoi(argv[++i]);
			cout << "-o1 " << omit1 << endl;
			}
		else if (strcmp(argv[i], "-o2") == 0) {
			f_omit2 = TRUE;
			omit2 = atoi(argv[++i]);
			cout << "-o2 " << omit2 << endl;
			}
		if (strcmp(argv[i], "-D1_upper_bound_x0") == 0) {
			f_D1_upper_bound_x0 = TRUE;
			D1_upper_bound_x0 = atoi(argv[++i]);
			cout << "-D1_upper_bound_x0 " << D1_upper_bound_x0 << endl;
			}
		else if (strcmp(argv[i], "-reverse") == 0) {
			f_reverse = TRUE;
			cout << "-reverse" << endl;
			}
		else if (strcmp(argv[i], "-reverse_inverse") == 0) {
			f_reverse_inverse = TRUE;
			cout << "-reverse_inverse" << endl;
			}
		else if (strcmp(argv[i], "-nopacking") == 0) {
			f_use_packing_numbers = FALSE;
			cout << "-nopacking" << endl;
			}
		else if (strcmp(argv[i], "-dual_is_linear_space") == 0) {
			f_dual_is_linear_space = TRUE;
			cout << "-dual_is_linear_space" << endl;
			}
		else if (strcmp(argv[i], "-geometric_test") == 0) {
			f_do_the_geometric_test = TRUE;
			cout << "-geometric_test" << endl;
			}
		else if (strcmp(argv[i], "-once") == 0) {
			f_once = TRUE;
			cout << "-once" << endl;
			}
		else if (strcmp(argv[i], "-mckay") == 0) {
			f_use_mckay_solver = TRUE;
			cout << "-mckay" << endl;
			}
#if 0
		if (strcmp(argv[i], "-max") == 0) {
			f_upper_bound_distribution = TRUE;
			upper_bound_distribution = atoi(argv[++i]);
			cout << "-max " << upper_bound_distribution << endl;
			}
#endif
		}
	fname_in = argv[argc - 1];
}

void tdo_parameter_calculation::main_loop()
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	cout << "tdo_parameter_calculation::main_loop "
			"opening file " << fname_in << " for reading" << endl;
	ifstream f(fname_in);
	
	strcpy(str, fname_in);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);


	sprintf(fname_out, "%s", str);
	if (f_range) {
		sprintf(fname_out + strlen(fname_out), "_r%d_%d", range_first, range_len);
		}
	if (f_select) {
		sprintf(fname_out + strlen(fname_out), "_S%s", select_label);
		}
	sprintf(fname_out + strlen(fname_out), "r.tdo");
	{
	cout << "tdo_parameter_calculation::main_loop "
			"opening file " << fname_out << " for writing" << endl;
	ofstream g(fname_out);
	
	for (cnt = 0; ; cnt++) {

		cout << "tdo_parameter_calculation::main_loop "
				"cnt=" << cnt << endl;

		if (f.eof()) {
			cout << "eof reached" << endl;
			break;
			}

#if 0
		if (cnt && (cnt % 1000) == 0) {
			cout << cnt << endl;
			registry_dump();
			}
#endif

		if (!GP.input_mode_stack(f, 0 /*verbose_level - 1*/)) {
			//cout << "GP.input_mode_stack returns FALSE" << endl;
			break;
			}
	
		cout << "tdo_parameter_calculation::main_loop "
				"cnt=" << cnt << " read input TDO" << endl;

		f_doit = TRUE;
		if (f_range) {
			if (cnt + 1 < range_first || cnt + 1 >= range_first + range_len)
				f_doit = FALSE;
			}
		if (f_select) {
			if (strcmp(GP.label, select_label))
				continue;
			}
		if (f_doit) {
			if (f_v) {
				cout << "tdo_parameter_calculation::main_loop "
						"read decomposition " << cnt << endl;
				}
			if (f_vv) {
				GP.print_schemes();
				}
			if (FALSE) {
				cout << "after print_schemes" << endl;
				}
			do_it(g, verbose_level - 1);
			}


		} // next cnt


	
	g << -1 << " " << nb_written << " TDOs, with " << nb_written_tactical << " being tactical" << endl;
	cout << "tdo_parameter_calculation::main_loop " << nb_written
			<< " TDOs, with " << nb_written_tactical << " being tactical" << endl;
	}
}

void tdo_parameter_calculation::do_it(ofstream &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	tdo_scheme G;
	partitionstack P;
	
	if (f_v) {
		cout << "tdo_parameter_calculation::do_it "
				"read TDO " << cnt << " " << GP.label << endl;
		}
	
	GP.init_tdo_scheme(G, verbose_level - 1);
	if (f_vv) {
		cout << "tdo_parameter_calculation::do_it "
				"after init_tdo_scheme" << endl;
		GP.print_schemes(G);
		}


	if (f_vvv) {
		cout << "tdo_parameter_calculation::do_it "
				"calling init_partition_stack" << endl;
		}
	G.init_partition_stack(verbose_level - 4);
	if (f_vvv) {
		cout << "tdo_parameter_calculation::do_it "
				"row_level=" << GP.row_level << endl;
		cout << "tdo_parameter_calculation::do_it "
				"col_level=" << GP.col_level << endl;
		}
			
	if (GP.col_level > GP.row_level) {
		if (f_vvv) {
			cout << "tdo_parameter_calculation::do_it "
					"calling do_row_refinement" << endl;
			}
		do_row_refinement(g, G, P, verbose_level);
		if (f_vvv) {
			cout << "tdo_parameter_calculation::do_it "
					"after do_row_refinement" << endl;
			}
		}
	else if (GP.col_level < GP.row_level) {
		if (f_vvv) {
			cout << "tdo_parameter_calculation::do_it "
					"calling do_col_refinement" << endl;
			}
		do_col_refinement(g, G, P, verbose_level);
		if (f_vvv) {
			cout << "tdo_parameter_calculation::do_it "
					"after do_col_refinement" << endl;
			}
		}
	else {
		GP.write_mode_stack(g, GP.label);
#if 0
		tdo_write(g, GP.label, GP.part, GP.nb_parts, GP.entries, GP.nb_entries, 
			GP.row_level, GP.col_level, GP.lambda_level, 
			GP.extra_row_level, GP.extra_col_level);
#endif
		if (f_vv) {
			cout << "tdo_parameter_calculation::do_it "
					<< GP.label << " written" << endl;
			}
		nb_written++;
		nb_written_tactical++;
		}



}

void tdo_parameter_calculation::do_row_refinement(
	ofstream &g, tdo_scheme &G,
	partitionstack &P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_parameter_calculation::do_row_refinement "
				"col_level > row_level" << endl;
		}
	int *point_types, nb_point_types, point_type_len;
	int *distributions, nb_distributions;
	int f_success;
				
	if (f_lambda3) {
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"before G.td3_refine_rows" << endl;
			}
		f_success = G.td3_refine_rows(verbose_level - 1, f_once, 
			lambda3, block_size, 
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions);
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"after G.td3_refine_rows" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"before G.refine_rows" << endl;
			}
		f_success = G.refine_rows(verbose_level - 1,
			f_use_mckay_solver, f_once, P,
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions, 
			cnt_second_system, Sol,
			f_omit1, omit1, f_omit2, omit2, 
			f_use_packing_numbers,
			f_dual_is_linear_space,
			f_do_the_geometric_test);
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"after G.refine_rows" << endl;
			}
		}
	
	if (f_success) {
		if (f_reverse || f_reverse_inverse) {
			distribution_reverse_sorting(f_reverse_inverse, 
				point_types, nb_point_types, point_type_len,  
				distributions, nb_distributions);
			}
		if (verbose_level >= 5) {
			print_distribution(cout, 
				point_types, nb_point_types, point_type_len,  
				distributions, nb_distributions);
			}
	
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"before do_all_row_refinements" << endl;
			}
		do_all_row_refinements(GP.label, g, G, 
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions, nb_tactical,
			verbose_level - 2);
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"after do_all_row_refinements" << endl;
			}
				
		nb_written += nb_distributions;
		nb_written_tactical += nb_tactical;
		FREE_int(point_types);
		FREE_int(distributions);
		}
	else {
		if (f_v) {
			cout << "tdo_parameter_calculation::do_row_refinement "
					"Case " << GP.label << ", found " << 0
				<< " row refinements, out of which " 
				<< 0 << " are tactical" << endl;
			}
		}
}

void tdo_parameter_calculation::do_col_refinement(
		ofstream &g, tdo_scheme &G,
	partitionstack &P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	int *line_types, nb_line_types, line_type_len;
	int *distributions, nb_distributions;
	int f_success;
				
	if (f_v) {
		cout << "tdo_parameter_calculation::do_col_refinement "
				"col_level < row_level" << endl;
		}
	if (f_lambda3) {
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
					"before G.td3_refine_columns" << endl;
			}
		f_success = G.td3_refine_columns(verbose_level - 1, f_once, 
			lambda3, block_size, f_scale, scaling, 
			line_types, nb_line_types, line_type_len, 
			distributions, nb_distributions);
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
					"after G.td3_refine_columns" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
					"before G.refine_columns" << endl;
			}
		f_success = G.refine_columns(verbose_level - 1, f_once, P, 
			line_types, nb_line_types, line_type_len, 
			distributions, nb_distributions, 
			cnt_second_system, Sol, 
			f_omit1, omit1, f_omit2, omit2, 
			f_D1_upper_bound_x0, D1_upper_bound_x0, 
			f_use_mckay_solver, 
			f_use_packing_numbers);
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
					"after G.refine_columns" << endl;
			}
		}
	if (f_success) {
		if (f_reverse || f_reverse_inverse) {
			if (f_v) {
				cout << "tdo_parameter_calculation::do_col_refinement "
						"before G.distribution_reverse_sorting" << endl;
				}
			distribution_reverse_sorting(f_reverse_inverse, 
				line_types, nb_line_types, line_type_len,  
				distributions, nb_distributions);
			if (f_v) {
				cout << "tdo_parameter_calculation::do_col_refinement "
						"after G.distribution_reverse_sorting" << endl;
				}
			}
		if (verbose_level >= 5) {
			print_distribution(cout, 
				line_types, nb_line_types, line_type_len,  
				distributions, nb_distributions);
			}
				
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
					"before do_all_column_refinements" << endl;
			}
		do_all_column_refinements(GP.label, g, G, 
			line_types, nb_line_types, line_type_len, 
			distributions, nb_distributions, nb_tactical, 
			verbose_level - 1);
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
				"after do_all_column_refinements" << endl;
		}
		nb_written += nb_distributions;
		nb_written_tactical += nb_tactical;
		FREE_int(line_types);
		FREE_int(distributions);
		}
	else {
		if (f_v) {
			cout << "tdo_parameter_calculation::do_col_refinement "
				"Case " << GP.label << ", found " << 0
				<< " col refinements, out of which " 
				<< 0 << " are tactical" << endl;
			}
		}
}

void tdo_parameter_calculation::do_all_row_refinements(
	char *label_in, ofstream &g, tdo_scheme &G,
	int *point_types, int nb_point_types, int point_type_len, 
	int *distributions, int nb_distributions, int &nb_tactical, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, t;
			
	nb_tactical = 0;
	for (i = 0; i < GP.nb_parts; i++) 
		GP2.part[i] = GP.part[i];
	for (i = 0; i < 4 * GP.nb_entries; i++) 
		GP2.entries[i] = GP.entries[i];
			
	for (t = 0; t < nb_distributions; t++) {
		
		if (do_row_refinement(t, label_in, g, G, point_types, nb_point_types, 
			point_type_len, distributions, nb_distributions, 
			verbose_level - 5))
			nb_tactical++;
				
		}
	if (f_v) {
		cout << "Case " << label_in << ", found " << nb_distributions 
			<< " row refinements, out of which " 
			<< nb_tactical << " are tactical" << endl;
		}
			
}

void tdo_parameter_calculation::do_all_column_refinements(
	char *label_in, ofstream &g, tdo_scheme &G,
	int *line_types, int nb_line_types, int line_type_len, 
	int *distributions, int nb_distributions, int &nb_tactical, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, t;
			

	nb_tactical = 0;
	for (i = 0; i < GP.nb_parts; i++) 
		GP2.part[i] = GP.part[i];
	for (i = 0; i < 4 * GP.nb_entries; i++) 
		GP2.entries[i] = GP.entries[i];
			
	for (t = 0; t < nb_distributions; t++) {
		
		//cout << "tdo_parameter_calculation::do_all_column_refinements t=" << t << endl;
		if (do_column_refinement(t, label_in, g, G, line_types, nb_line_types, 
			line_type_len, distributions, nb_distributions, 
			verbose_level - 5))
			nb_tactical++;
		
		}
	if (f_v) {
		cout << "Case " << label_in << ", found " << nb_distributions 
			<< " column refinements, out of which " 
			<< nb_tactical << " are tactical" << endl;
		}
}


int tdo_parameter_calculation::do_row_refinement(
	int t, char *label_in, ofstream &g, tdo_scheme &G,
	int *point_types, int nb_point_types, int point_type_len, 
	int *distributions, int nb_distributions, int verbose_level)
// returns TRUE or FALSE depending on whether the 
// refinement gave a tactical decomposition
{
	int r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	int *type_index;
	//char label_out[1000];
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 6);
	int f_vvv = (verbose_level >= 7);
	int f_tactical;

	if (f_v) {
		cout << "do_row_refinement t=" << t << endl;
		}
	
	type_index = NEW_int(nb_point_types);
	for (i = 0; i < nb_point_types; i++) 
		type_index[i] = -1;
	
	new_nb_parts = GP.nb_parts;
	if (G.row_level >= 2) {
		R = G.nb_row_classes[ROW];
		}
	else {
		R = 1;
		}
	i = 0;
	h = 0;
	S = 0;
	for (r = 0; r < R; r++) {
		if (G.row_level >= 2) {
			l = G.row_classes_len[ROW][r];
			}
		else {
			//partitionstack &P = G.PB.P;
			l = G.P->startCell[1];
			}
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
			}
		while (i < nb_point_types) {
			a = distributions[t * nb_point_types + i];
			if (a == 0) {
				i++;
				continue;
				}
			if (f_vv) {
				cout << "h=" << h << " i=" << i << " a=" << a << " s=" << s << " S=" << S << endl;
				}
			type_index[h++] = i;
			if (s == 0) {
				}
			else {
				GP2.part[new_nb_parts++] = S + s;
				}
			s += a;
			i++;
			if (s == l)
				break;
			if (s > l) {
				cout << "do_row_refinement: s > l" << endl;
				exit(1);
				}
			}
		S += l;
		}
	if (S != G.m) {
		cout << "do_row_refinement: S != G.m" << endl;
		exit(1);
		}
	
	new_nb_entries = GP.nb_entries;
	GP2.part[new_nb_parts] = -1;
	GP2.entries[new_nb_entries * 4 + 0] = -1;
	if (f_vv) {
		cout << "new_part:" << endl;
		for (i = 0; i < new_nb_parts; i++) 
			cout << GP2.part[i] << " ";
		cout << endl;
		cout << "type_index:" << endl;
		for (i = 0; i < h; i++) 
			cout << type_index[i] << " ";
		cout << endl;
		}
		

		
	{
		tdo_scheme G2;
		
		G2.init_part_and_entries_int(GP2.part, GP2.entries, verbose_level - 2);

		G2.row_level = new_nb_parts;
		G2.col_level = G.col_level;
		G2.extra_row_level = G.row_level; // GP.extra_row_level;
		G2.extra_col_level = GP.extra_col_level;
		G2.lambda_level = G.lambda_level;
		G2.level[ROW] = new_nb_parts;
		G2.level[COL] = G.col_level;
		G2.level[EXTRA_ROW] = G.row_level; // G.extra_row_level;
		G2.level[EXTRA_COL] = G.extra_col_level;
		G2.level[LAMBDA] = G.lambda_level;

		G2.init_partition_stack(verbose_level - 2);
		
		if (f_v) {
			cout << "found a " << G2.nb_row_classes[ROW] << " x " << G2.nb_col_classes[ROW] << " scheme" << endl;
			}
		for (i = 0; i < G2.nb_row_classes[ROW]; i++) {
			c1 = G2.row_classes[ROW][i];
			for (j = 0; j < point_type_len /*G2.nb_col_classes[ROW]*/; j++) {
				c2 = G2.col_classes[ROW][j];
				idx = type_index[i];
				if (idx == -1)
					continue;
				a = point_types[idx * point_type_len + j];
				if (f_vv) {
					cout << "i=" << i << " j=" << j << " idx=" << idx << " a=" << a << endl;
					}
				GP2.entries[new_nb_entries * 4 + 0] = new_nb_parts;
				GP2.entries[new_nb_entries * 4 + 1] = c1;
				GP2.entries[new_nb_entries * 4 + 2] = c2;
				GP2.entries[new_nb_entries * 4 + 3] = a;
				new_nb_entries++;
				}
			}
		
		if (f_vvv) {
			for (i = 0; i < new_nb_entries; i++) {
				for (j = 0; j < 4; j++) {
					cout << setw(2) << GP2.entries[i * 4 + j] << " ";
					}
				cout << endl;
				}
			}
			
		sprintf(GP2.label, "%s.%d", label_in, t + 1);
			
		GP2.nb_parts = new_nb_parts;
		GP2.nb_entries = new_nb_entries;
		GP2.row_level = new_nb_parts;
		GP2.col_level = G.col_level;
		GP2.lambda_level = G.lambda_level;
		GP2.extra_row_level = G.row_level;
		GP2.extra_col_level = G.extra_col_level;
		
		GP2.write_mode_stack(g, GP2.label);

#if 0
		tdo_write(g, GP2.label, 
			GP2.part, new_nb_parts, GP2.entries, new_nb_entries, 
			new_nb_parts, G.col_level, G.lambda_level, 
			G.row_level/*G.extra_row_level*/, G.extra_col_level);
#endif

		if (f_vv) {
			cout << GP2.label << " written" << endl;
			}
		if (new_nb_parts == G.col_level)
			f_tactical = TRUE;
		else
			f_tactical = FALSE;
	}

	FREE_int(type_index);
	return f_tactical;
}

int tdo_parameter_calculation::do_column_refinement(
	int t, char *label_in, ofstream &g, tdo_scheme &G,
	int *line_types, int nb_line_types, int line_type_len, 
	int *distributions, int nb_distributions, int verbose_level)
// returns TRUE or FALSE depending on whether the 
// refinement gave a tactical decomposition
{
	int r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	int *type_index;
	//char label_out[1000];
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 6);
	int f_vvv = (verbose_level >= 7);
	int f_tactical;

	if (f_v) {
		cout << "do_column_refinement t=" << t << endl;
		}
	
	type_index = NEW_int(nb_line_types);
	
	for (i = 0; i < nb_line_types; i++) 
		type_index[i] = -1;
	new_nb_parts = GP.nb_parts;
	R = G.nb_col_classes[COL];
	i = 0;
	h = 0;
	S = G.m;
	for (r = 0; r < R; r++) {
		l = G.col_classes_len[COL][r];
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
			}
		while (i < nb_line_types) {
			a = distributions[t * nb_line_types + i];
			if (a == 0) {
				i++;
				continue;
				}
			if (f_vv) {
				cout << "h=" << h << " i=" << i << " a=" << a << " s=" << s << " S=" << S << endl;
				}
			type_index[h++] = i;
			if (s == 0) {
				}
			else {
				GP2.part[new_nb_parts++] = S + s;
				}
			s += a;
			i++;
			if (s == l)
				break;
			if (s > l) {
				cout << "do_column_refinement: s > l" << endl;
				cout << "r=" << r << endl;
				cout << "s=" << s << endl;
				cout << "l=" << l << endl;
				cout << "a=" << a << endl;
				int_vec_print(cout, distributions + t * nb_line_types, nb_line_types);
				cout << endl;
				exit(1);
				}
			}
		S += l;
		}
	if (S != G.m + G.n) {
		cout << "do_column_refinement: S != G.m + G.n" << endl;
		exit(1);
		}
	
	new_nb_entries = G.nb_entries;
	GP2.part[new_nb_parts] = -1;
	GP2.entries[new_nb_entries * 4 + 0] = -1;
	if (f_vv) {
		cout << "new_part:" << endl;
		for (i = 0; i < new_nb_parts; i++) 
			cout << GP2.part[i] << " ";
		cout << endl;
		cout << "type_index:" << endl;
		for (i = 0; i < h; i++) 
			cout << type_index[i] << " ";
		cout << endl;
		}
		
	{
		tdo_scheme *G2;
		
		G2 = new tdo_scheme;
		
		G2->init_part_and_entries_int(GP2.part, GP2.entries, verbose_level - 2);

		G2->row_level = GP.row_level;
		G2->col_level = new_nb_parts;
		G2->extra_row_level = GP.extra_row_level;
		G2->extra_col_level = GP.col_level; // GP.extra_col_level;
		G2->lambda_level = G.lambda_level;
		G2->level[ROW] = G.row_level;
		G2->level[COL] = new_nb_parts;
		G2->level[EXTRA_ROW] = G.extra_row_level;
		G2->level[EXTRA_COL] = GP.col_level; // G.extra_col_level;
		G2->level[LAMBDA] = G.lambda_level;

		G2->init_partition_stack(verbose_level - 2);
		
		if (f_v) {
			cout << "found a " << G2->nb_row_classes[COL] << " x " << G2->nb_col_classes[COL] << " scheme" << endl;
			}
		for (i = 0; i < G2->nb_row_classes[COL]; i++) {
			c1 = G2->row_classes[COL][i];
			for (j = 0; j < G2->nb_col_classes[COL]; j++) {
				c2 = G2->col_classes[COL][j];
				idx = type_index[j];
				if (idx == -1)
					continue;
				a = line_types[idx * line_type_len + i];
				if (f_vv) {
					cout << "i=" << i << " j=" << j << " idx=" << idx << " a=" << a << endl;
					}
				GP2.entries[new_nb_entries * 4 + 0] = new_nb_parts;
				GP2.entries[new_nb_entries * 4 + 1] = c2;
				GP2.entries[new_nb_entries * 4 + 2] = c1;
				GP2.entries[new_nb_entries * 4 + 3] = a;
				new_nb_entries++;
				}
			}
		
		if (f_vvv) {
			for (i = 0; i < new_nb_entries; i++) {
				for (j = 0; j < 4; j++) {
					cout << setw(2) << GP2.entries[i * 4 + j] << " ";
					}
				cout << endl;
				}
			}
		
		sprintf(GP2.label, "%s.%d", label_in, t + 1);
			
		GP2.nb_parts = new_nb_parts;
		GP2.nb_entries = new_nb_entries;
		GP2.row_level = G.row_level;
		GP2.col_level = new_nb_parts;
		GP2.lambda_level = G.lambda_level;
		GP2.extra_row_level = G.extra_row_level;
		GP2.extra_col_level = G.col_level;
		
		GP2.write_mode_stack(g, GP2.label);

#if 0

		tdo_write(g, GP2.label, GP2.part, new_nb_parts, GP2.entries, new_nb_entries, 
			G.row_level, new_nb_parts, G.lambda_level, 
			G.extra_row_level, GP.col_level /*G.extra_col_level*/);
#endif

		if (f_vv) {
			cout << GP2.label << " written" << endl;
			}
		if (new_nb_parts == G.row_level)
			f_tactical = TRUE;
		else
			f_tactical = FALSE;
		delete G2;
		}

	FREE_int(type_index);
	return f_tactical;
}

void print_distribution(ostream &ost, 
	int *types, int nb_types, int type_len,  
	int *distributions, int nb_distributions)
{
	int i, j;
	
	
	ost << "types:" << endl;
	for (i = 0; i < nb_types; i++) {
		ost << setw(3) << i + 1 << " : ";
		for (j = 0; j < type_len; j++) {
			ost << setw(3) << types[i * type_len + j];
			}
		ost << endl;
		}
	ost << endl;


	for (j = 0; j < type_len; j++) {
		ost << setw(3) << j + 1 << " & ";
		for (i = 0; i < nb_types; i++) {
			ost << setw(2) << types[i * type_len + j];
			if (i < nb_types - 1)
				ost << " & ";
			}
		ost << "\\\\" << endl;
		}
	ost << endl;
	
	ost << "distributions:" << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " : ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(3) << distributions[i * nb_types + j];
			}
		ost << endl;
		}
	ost << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " & ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(2) << distributions[i * nb_types + j];
			if (j < nb_types - 1)
				ost << " & ";
			}
		ost << "\\\\" << endl;
		}
	ost << endl;

	ost << "distributions (in compact format):" << endl;
	int f_first, a;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " & ";
		f_first = TRUE;
		for (j = 0; j < nb_types; j++) {
			a = distributions[i * nb_types + j];
			if (a == 0)
				continue;
			if (!f_first) {
				ost << ",";
				}
			ost << nb_types - 1 - j << "^{" << a << "}";
			f_first = FALSE;
			}
		ost << "\\\\" << endl;
		}
	ost << endl;
}

int compare_func_int_vec(void *a, void *b, void *data)
{
	int *p = (int *)a;
	int *q = (int *)b;
	int *d = (int *) data;
	int size = d[0];
	int i;
	
	for (i = 0; i < size; i++) {
		if (p[i] > q[i])
			return -1;
		if (p[i] < q[i])
			return 1;
		}
	return 0;
}

int compare_func_int_vec_inverse(void *a, void *b, void *data)
{
	int *p = (int *)a;
	int *q = (int *)b;
	int *d = (int *) data;
	int size = d[0];
	int i;
	
	for (i = 0; i < size; i++) {
		if (p[i] > q[i])
			return 1;
		if (p[i] < q[i])
			return -1;
		}
	return 0;
}

void distribution_reverse_sorting(int f_increasing, 
	int *types, int nb_types, int type_len,  
	int *distributions, int nb_distributions)
{
	int i, j;
	int *D;
	int **P;

	D = new int[nb_distributions * nb_types];
	P = new pint[nb_distributions];

	for (i = 0; i < nb_distributions; i++) {
		P[i] = D + i * nb_types;
		for (j = 0; j < nb_types; j++) {
			D[i * nb_types + nb_types - 1 - j] = distributions[i * nb_types + j];
			}
		}

	int p[1];

	p[0] = nb_types;

	if (f_increasing) {
		quicksort_array(nb_distributions, (void **) P,
				compare_func_int_vec_inverse, (void *)p);
		}
	else {
		quicksort_array(nb_distributions, (void **) P,
				compare_func_int_vec, (void *)p);
		}

	for (i = 0; i < nb_distributions; i++) {
		for (j = 0; j < nb_types; j++) {
			distributions[i * nb_types + j] = P[i][nb_types - 1 - j];
			}
		}
	delete [] D;
	delete [] P;
}


