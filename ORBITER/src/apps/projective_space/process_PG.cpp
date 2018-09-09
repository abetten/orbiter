// process_PG.C
// 
// Anton Betten
// December 13, 2011
//
// 
//
//

#include "orbiter.h"

#define MY_BUFSIZE ONE_MILLION



// global data:

int t0; // the system time when the program started


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	int f_n = FALSE;
	int n;
	int f_k = FALSE;
	int k;
	int f_poly = FALSE;
	char *poly = NULL;
	int f_conic_type = FALSE;
	int f_randomized = FALSE;
	int nb_times = 0;
	int f_file = FALSE;
	const char *fname = NULL;
	int f_add_nucleus = FALSE;
	int f_TDO = FALSE;
	int f_maxdepth_for_TDO = FALSE;
	int maxdepth_for_TDO = INT_MAX;
	int f_stabilizer = FALSE;
	int stabilizer_starter_size = 0;



	t0 = os_ticks();


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			cout << "-conic_type" << endl;
			}
		else if (strcmp(argv[i], "-randomized") == 0) {
			f_randomized = TRUE;
			nb_times = atoi(argv[++i]);
			cout << "-randomized " << nb_times << endl;
			}
		else if (strcmp(argv[i], "-add_nucleus") == 0) {
			f_add_nucleus = TRUE;
			cout << "-add_nucleus" << endl;
			}
		else if (strcmp(argv[i], "-TDO") == 0) {
			f_TDO = TRUE;
			cout << "-TDO" << endl;
			}
		else if (strcmp(argv[i], "-maxdepth_for_TDO") == 0) {
			f_maxdepth_for_TDO = TRUE;
			maxdepth_for_TDO = atoi(argv[++i]);
			cout << "-maxdepth_for_TDO " << maxdepth_for_TDO << endl;
			}
		else if (strcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			stabilizer_starter_size = atoi(argv[++i]);
			cout << "-stabilizer " << stabilizer_starter_size << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		}

	if (!f_file) {
		cout << "please use option -file <fname> to specify input file" << endl;
		exit(1);
		}	


	finite_field *F;
	projective_space *P;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
		
	if (f_v) {
		cout << "process_PG" << endl;
		cout << "We will now compute the plane type" << endl;
		}


	F = NEW_OBJECT(finite_field);

	F->init_override_polynomial(q, poly, 0);

	P = NEW_OBJECT(projective_space);
	
	if (f_v) {
		cout << "process_PG before P->init" << endl;
		}


	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "process_PG after P->init" << endl;
		}

	int *set;
	int set_size;
	int nb_input;


	if (file_size(fname) <= 0) {
		cout << "The file " << fname << " does not exist or is empty" << endl;
		exit(1);
		}


	nb_input = count_number_of_orbits_in_file(fname, 0 /*verbose_level */);

	cout << "The file " << fname << " contains " << nb_input << " solutions" << endl;


	set = NEW_int(ONE_MILLION);
	{
	char buf[MY_BUFSIZE];
	char *p_buf;
	int cnt;
	
	ifstream fp(fname);

	if (f_stabilizer) {
		}

	cnt = 0;
	while (!fp.eof()) {
		fp.getline(buf, MY_BUFSIZE, '\n');
		if (FALSE /* f_vv */) {
			cout << "Line read :'" << buf << "'" << endl;
			}
		p_buf = buf;
		while (TRUE) {
			int tmp;
			if (!s_scan_int(&p_buf, &tmp)) {
				break;
				}
			if (tmp == -1) {
				break;
				}
			set_size = tmp;
			cout << "reading solution " << cnt << " / " << nb_input << ", a set of size " << set_size << endl;
			for (i = 0; i < set_size; i++) {
				s_scan_int(&p_buf, &tmp);
				set[i] = tmp;
				}
			if (FALSE) {
				cout << "solution " << cnt << " / " << nb_input << " is ";
				int_vec_print(cout, set, set_size);
				cout << endl;
				}




			if (f_conic_type) {
		
				if (f_v) {
					cout << "solution " << cnt << " / " << nb_input << ", computing the conic intersection type" << endl;
					}

				int *intersection_type;
				int highest_intersection_number;
				int f_save_largest_sets = TRUE;
				set_of_sets *largest_sets = NULL;
		
				P->conic_intersection_type(f_randomized, nb_times, 
					set, set_size, 
					intersection_type, highest_intersection_number, 
					f_save_largest_sets, largest_sets, 
					verbose_level - 2);

				if (f_v) {
					cout << "solution " << cnt << " / " << nb_input << ", the conic intersection type is:" << endl;
					for (i = 0; i <= highest_intersection_number; i++) {
						if (intersection_type[i]) {
							cout << i << "^" << intersection_type[i] << " ";
							}
						}
					cout << endl;
					}

				int *Inc;
				int m, n;
				
				largest_sets->compute_incidence_matrix(Inc, m, n,  verbose_level);
				int_matrix_print(Inc, m, n);

				decomposition *D;

				largest_sets->init_decomposition(D, verbose_level);

				int max_depth = 2;
				
				D->setup_default_partition(verbose_level);

				D->compute_TDO(max_depth, verbose_level);

				int f_enter_math = TRUE;
				int f_print_subscripts = TRUE;

				if (ODD(max_depth)) {
					D->get_col_scheme(verbose_level);
					D->print_column_decomposition_tex(cout, 
						f_enter_math, f_print_subscripts, verbose_level);
					}
				else {
					D->get_row_scheme(verbose_level);
					D->print_row_decomposition_tex(cout, 
						f_enter_math, f_print_subscripts, verbose_level);
					}
				FREE_int(Inc);
				FREE_int(intersection_type);
				delete D;
				}

			else if (f_add_nucleus) {
				int nucleus;
				int *set2;
				int sz2;
				char fname_base[1000];
				char fname2[1000];
				
				P->find_nucleus(set, set_size, nucleus, verbose_level);

				sz2 = set_size + 1;
				set2 = NEW_int(sz2);
				int_vec_copy(set, set2, set_size);
				set2[set_size] = nucleus;
				
				get_fname_base(fname, fname_base);
				sprintf(fname2, "%s_hyperoval_%d.txt", fname_base, cnt);
				write_set_to_file(fname2, set2, sz2, verbose_level);
				cout << "Written file " << fname2 << " of size " << file_size(fname2) << endl;
				}

			else if (f_TDO) {

				int max_depth;

				if (f_maxdepth_for_TDO) {
					max_depth = maxdepth_for_TDO;
					}
				else {
					max_depth = INT_MAX;
					}


				cout << "before P->init_incidence_structure" << endl;
				P->init_incidence_structure(verbose_level);

				
				incidence_structure *Inc;
				partitionstack *Stack;

				cout << "before P->make_incidence_structure_and_partition" << endl;
				P->make_incidence_structure_and_partition(Inc, Stack, verbose_level);
				cout << "after P->make_incidence_structure_and_partition" << endl;


				cout << "before Stack->split_cell splitting a set of size " << set_size << endl;
				Stack->split_cell(set, set_size, verbose_level);
				Stack->sort_cells();

				//Stack->print_raw();

				decomposition *D;

				D = NEW_OBJECT(decomposition);

				cout << "before D->init_inc_and_stack" << endl;
				D->init_inc_and_stack(Inc, Stack, verbose_level);

				int m, n;
				int *M;
				
				P->make_incidence_matrix(m, n, M, verbose_level);
				D->init_incidence_matrix(m, n, M, verbose_level);
				FREE_int(M);
				
				cout << "before D->compute_TDO, max_depth=" << max_depth << endl;
				D->compute_TDO(max_depth, verbose_level);
				cout << "after D->compute_TDO" << endl;

				int f_enter_math = TRUE;
				int f_print_subscripts = TRUE;

				if (ODD(max_depth) || max_depth == INT_MAX) {
					D->get_col_scheme(verbose_level);
					D->print_column_decomposition_tex(cout, 
						f_enter_math, f_print_subscripts, verbose_level);
					}
				if (EVEN(max_depth) || max_depth == INT_MAX) {
					D->get_row_scheme(verbose_level);
					D->print_row_decomposition_tex(cout, 
						f_enter_math, f_print_subscripts, verbose_level);
					}
				

				//delete Stack;
				//delete Inc;
				FREE_OBJECT(D);
				}

			else if (f_stabilizer) {
				}

			else {
				cout << "nothing to do" << endl;
				}


			cnt++;
			}
		}


	} // fp

	FREE_int(set);
	FREE_OBJECT(P);
	FREE_OBJECT(F);


	the_end(t0);
}


