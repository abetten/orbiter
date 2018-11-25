// subspace_orbits_main.C
// 
// Anton Betten
// Jan 25, 2010
//
// moved here from PROJECTIVE_SPACE: July 4, 2014
// 
//
//

#include "orbiter.h"



// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);
int extra_test_func(subspace_orbits *SubOrb, 
	int len, int *S, void *data, int verbose_level);
int test_dim_C_cap_Cperp_property(int len, int *S, void *data);
int compute_minimum_distance(int len, int *S, void *data);
void print_subspace(ostream &ost, int len, int *S, void *data);





	int f_mindist = FALSE;
	int the_mindist = 0;
	int f_self_orthogonal = FALSE;
	int f_doubly_even = FALSE;

int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i, j;
	int f_override_poly = FALSE;
	const char *override_poly = NULL;
	int f_depth = FALSE;
	int depth = 0;
	int f_r = FALSE;
	int depth_completed = -1;
	const char *data_file_name = NULL;
	int f_group_generators = FALSE;
	int group_generators_data[1000];
	int group_generators_data_size = 0;
	int f_group_order_target = FALSE;
	const char *group_order_target;
	int f_KM = FALSE;
	int KM_t = 0;
	int KM_k = 0;
	int f_read_solutions = FALSE;
	const char *solution_fname = NULL;
	int f_print_generators = FALSE;
	int f_exportmagma = FALSE;
	int f_draw_poset = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	int f_table_of_nodes = FALSE;
	int f_list = FALSE;
	int f_list_all_levels = FALSE;
	int f_list_LCD = FALSE;
	int f_print_matrix = FALSE;
	int f_run_log_fname = FALSE;
	const char *run_log_fname = NULL;
	
	linear_group_description *Descr;
	
 	t0 = os_ticks();


	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_override_poly = TRUE;
			override_poly = argv[++i];
			cout << "-poly " << override_poly << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-r") == 0) {
			f_r = TRUE;
			depth_completed = atoi(argv[++i]);
			data_file_name = argv[++i];
			cout << "-r " << depth_completed << " " << data_file_name << endl;
			}
		else if (strcmp(argv[i], "-exportmagma") == 0) {
			f_exportmagma = TRUE;
			cout << "-exportmagma" << endl;
			}
		else if (strcmp(argv[i], "-KM") == 0) {
			f_KM = TRUE;
			KM_t = atoi(argv[++i]);
			KM_k = atoi(argv[++i]);
			cout << "-KM " << KM_t << " " << KM_k << endl;
			}
		else if (strcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = TRUE;
			solution_fname = argv[++i];
			cout << "-read_solutions " << solution_fname << endl;
			}
		else if (strcmp(argv[i], "-G") == 0) {
			f_group_generators = TRUE;
			for (j = 0; ; j++) {
				group_generators_data[j] = atoi(argv[++i]);
				if (group_generators_data[j] == -1)
					break;
				}
			group_generators_data_size = j;
			cout << "-G ";
			int_vec_print(cout, group_generators_data, group_generators_data_size);
			cout << endl;
			}
		else if (strcmp(argv[i], "-GO") == 0) {
			f_group_order_target = TRUE;
			group_order_target = argv[++i];
			cout << "-GO " << group_order_target << endl;
			}
		else if (strcmp(argv[i], "-print_generators") == 0) {
			f_print_generators = TRUE;
			cout << "-print_generators " << endl;
			}
		else if (strcmp(argv[i], "-mindist") == 0) {
			f_mindist = TRUE;
			the_mindist = atoi(argv[++i]);
			cout << "-mindist " << the_mindist << endl;
			}
		else if (strcmp(argv[i], "-self_orthogonal") == 0) {
			f_self_orthogonal = TRUE;
			cout << "-self_orthogonal " << endl;
			}
		else if (strcmp(argv[i], "-doubly_even") == 0) {
			f_doubly_even = TRUE;
			cout << "-doubly_even " << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-table_of_nodes") == 0) {
			f_table_of_nodes = TRUE;
			cout << "-table_of_nodes " << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list " << endl;
			}
		else if (strcmp(argv[i], "-list_all_levels") == 0) {
			f_list_all_levels = TRUE;
			cout << "-list_all_levels " << endl;
			}
		else if (strcmp(argv[i], "-list_LCD") == 0) {
			f_list_LCD = TRUE;
			cout << "-list_LCD " << endl;
			}
		else if (strcmp(argv[i], "-print_matrix") == 0) {
			f_print_matrix = TRUE;
			cout << "-print_matrix " << endl;
			}
		else if (strcmp(argv[i], "-run_log_fname") == 0) {
			f_run_log_fname = TRUE;
			run_log_fname = argv[++i];
			cout << "-run_log_fname " << run_log_fname << endl;
			}
		
		}

	Descr = NEW_OBJECT(linear_group_description);
	Descr->read_arguments(argc, argv, verbose_level);

	if (!f_depth) {
		cout << "please use -depth option" << endl;
		exit(1);
		}

	int f_v = (verbose_level >= 1);
	
	finite_field *F;
	linear_group *LG;
	subspace_orbits *SubOrb;

	F = NEW_OBJECT(finite_field);

	F->init_override_polynomial(Descr->input_q, override_poly, 0);
	Descr->F = F;


	LG = NEW_OBJECT(linear_group);

	cout << "before LG->init, creating the group" << endl;

	LG->init(Descr, verbose_level);
	
	cout << "after LG->init, strong generators for the group have been created" << endl;



	SubOrb = NEW_OBJECT(subspace_orbits);

	SubOrb->init(argc, argv, 
		LG, depth, 
		verbose_level);
	
	SubOrb->f_print_generators = f_print_generators;


	if (f_mindist || f_self_orthogonal) {
		SubOrb->f_has_extra_test_func = TRUE;
		SubOrb->extra_test_func = extra_test_func;
		SubOrb->extra_test_func_data = NULL;
		}


#if 0
	if (f_r) {
		//SubOrb->init2(verbose_level);
		SubOrb->read_data_file(depth_completed, data_file_name, f_exportmagma, verbose_level);
		}
	else {
		if (f_group_generators) {
			SubOrb->init_group(group_generators_data, group_generators_data_size, 
				f_group_order_target, group_order_target, verbose_level);
			}
		SubOrb->init2(verbose_level);
		}
#endif


	SubOrb->compute_orbits(verbose_level);
	if (f_KM) {
		SubOrb->Kramer_Mesner_matrix(KM_t, KM_k, f_print_matrix, f_read_solutions, solution_fname, verbose_level);
		}



	if (f_list) {
		int f_show_orbit_decomposition = FALSE, f_show_stab = TRUE, f_save_stab = FALSE, f_show_whole_orbit = FALSE;
		
		SubOrb->Gen->list_all_orbits_at_level(depth, 
			TRUE, 
			print_subspace, 
			SubOrb, 
			f_show_orbit_decomposition, f_show_stab, f_save_stab, f_show_whole_orbit);
		}

	if (f_list_all_levels) {
		int f_show_orbit_decomposition = FALSE, f_show_stab = TRUE, f_save_stab = FALSE, f_show_whole_orbit = FALSE;
		int l;

		for (l = 0; l <= depth; l++) {
			cout << "##### orbits at level " << l << ":" << endl;
			SubOrb->Gen->list_all_orbits_at_level(l, 
				TRUE, 
				print_subspace, 
				SubOrb, 
				f_show_orbit_decomposition, f_show_stab, f_save_stab, f_show_whole_orbit);
			}
		}

	if (f_list_LCD) {

		int *Orbits;
		int nb_orbits;
		int *Data;

		SubOrb->test_dim = 0;
		SubOrb->Gen->test_property(depth, 
			test_dim_C_cap_Cperp_property, 
			SubOrb, 
			nb_orbits, Orbits);

		cout << "We found " << nb_orbits << " LCD codes" << endl;
		cout << "They are: ";
		int_vec_print(cout, Orbits, nb_orbits);
		cout << endl;

		int f_show_orbit_decomposition = FALSE, f_show_stab = FALSE, f_save_stab = FALSE, f_show_whole_orbit = FALSE;
		
		SubOrb->Gen->list_selected_set_of_orbits_at_level(depth, 
			nb_orbits, Orbits, 
			TRUE, 
			print_subspace, 
			SubOrb, 
			f_show_orbit_decomposition, f_show_stab, f_save_stab, f_show_whole_orbit);

		SubOrb->Gen->compute_integer_property_of_selected_list_of_orbits(depth, 
			nb_orbits, Orbits, 
			compute_minimum_distance, 
			SubOrb, 
			Data);

		classify C;

		C.init(Data, nb_orbits, FALSE, 0);
		cout << "The distribution of the minimum distance of the " << nb_orbits << " LCD codes is: ";
		C.print_naked(TRUE);
		cout << endl;
		FREE_int(Data);
		}

	if (f_draw_poset) {
		if (f_v) {
			cout << "before gen->draw_poset" << endl;
			}
		SubOrb->Gen->draw_poset(SubOrb->Gen->fname_base, depth, 
			0 /* data1 */, f_embedded, f_sideways, 0 /* gen->verbose_level */);
		}


	if (f_table_of_nodes) {
		int *Table;
		int nb_rows, nb_cols;
		char fname[1000];

		if (f_v) {
			cout << "before SubOrb.Gen->get_table_of_nodes" << endl;
			}
		SubOrb->Gen->get_table_of_nodes(Table, nb_rows, nb_cols, 0 /*verbose_level*/);
	
		if (f_v) {
			cout << "before int_matrix_write_csv nb_rows=" << nb_rows << " nb_cols=" << nb_cols << endl;
			}

		sprintf(fname, "%s_table_of_nodes.csv", SubOrb->Gen->fname_base);
		if (f_v) {
			cout << "writing to file " << fname << endl;
			}

		int_matrix_write_csv(fname, Table, nb_rows, nb_cols);


		FREE_int(Table);
		}



	cout << "Memory usage = " << os_memory_usage() <<  " Time = " << delta_time(t0) << " tps = " << os_ticks_per_second() << endl;
	char exec_log_fname[1000];
	int M[3];
	const char *column_labels[] = {
		"memory", "time", "tps"
		};

	M[0] = os_memory_usage();
	M[1] = delta_time(t0);
	M[2] = os_ticks_per_second();

	if (f_run_log_fname) {
		strcpy(exec_log_fname, run_log_fname);
		}
	else {
		sprintf(exec_log_fname, "subspace_orbits_run.csv");
		}
	int_matrix_write_csv_with_labels(exec_log_fname, M, 1, 3, column_labels);
	cout << "Written file " << exec_log_fname << " of size " << file_size(exec_log_fname) << endl;

	FREE_OBJECT(LG);
	FREE_OBJECT(Descr);
	FREE_OBJECT(SubOrb);
	FREE_OBJECT(F);
	
	
	the_end(t0);
}

// ##################################################################################################
// callback functions
// ##################################################################################################


int extra_test_func(subspace_orbits *SubOrb, 
	int len, int *S, void *data, int verbose_level)
{
	int f_v = FALSE;//(verbose_level >= 1);
	//int *p_mindist = (int *) data;
	//int mindist = *p_mindist;
	int ret = TRUE;

	if (f_mindist) {
		ret = SubOrb->test_minimum_distance(len, S, the_mindist, 0 /* verbose_level */);
		if (f_v) {
			if (ret) {
				cout << "extra_test_func the minimum distance is OK" << endl;
				}
			else {
				cout << "extra_test_func the minimum distance is not OK" << endl;
				}
			}
		}
	if (ret && f_self_orthogonal) {
		ret = SubOrb->test_if_self_orthogonal(len, S, f_doubly_even, 0 /* verbose_level */);
		if (f_v) {
			if (ret) {
				cout << "extra_test_func the self-orthogonality test is OK" << endl;
				}
			else {
				cout << "extra_test_func the self-orthogonality test is not OK" << endl;
				}
			}
		}
	if (f_v) {
		if (ret) {
			cout << "extra_test_func the set is OK" << endl;
			}
		else {
			cout << "extra_test_func the set is not OK" << endl;
			}
		}
	return ret;
}

#if 0
int mindist_test_func(subspace_orbits *SubOrb, 
	int len, int *S, void *data, int verbose_level)
{
	int f_v = FALSE;//(verbose_level >= 1);
	int *p_mindist = (int *) data;
	int mindist = *p_mindist;
	int ret;

	ret = SubOrb->test_minimum_distance(len, S, mindist, 0 /* verbose_level */);
	if (f_v) {
		if (ret) {
			cout << "mindist_test_func the set is OK" << endl;
			}
		else {
			cout << "mindist_test_func the set is not OK" << endl;
			}
		}
	return ret;
}

int is_self_orthogonal_test_func(subspace_orbits *SubOrb, 
	int len, int *S, void *data, int verbose_level)
{
	int f_v = FALSE;//(verbose_level >= 1);
	int ret;

	ret = SubOrb->test_if_self_orthogonal(len, S, 0 /* verbose_level */);
	if (f_v) {
		if (ret) {
			cout << "is_self_orthogonal_test_func the set is OK" << endl;
			}
		else {
			cout << "is_self_orthogonal_test_func the set is not OK" << endl;
			}
		}
	return ret;
}
#endif

int test_dim_C_cap_Cperp_property(int len, int *S, void *data)
{
	subspace_orbits *so = (subspace_orbits *) data;
	int dim, ret;

	dim = so->test_dim;
	ret = so->test_dim_C_cap_Cperp_property(len, S, dim);
	return ret;
}

int compute_minimum_distance(int len, int *S, void *data)
{
	subspace_orbits *so = (subspace_orbits *) data;
	int d;

	d = so->compute_minimum_distance(len, S);
	return d;
}

void print_subspace(ostream &ost, int len, int *S, void *data)
{
	subspace_orbits *so = (subspace_orbits *) data;
	
	so->print_set(ost, len, S);
}




