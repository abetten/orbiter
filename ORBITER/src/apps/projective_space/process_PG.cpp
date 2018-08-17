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

INT t0; // the system time when the program started


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_q = FALSE;
	INT q;
	INT f_n = FALSE;
	INT n;
	INT f_k = FALSE;
	INT k;
	INT f_poly = FALSE;
	BYTE *poly = NULL;
	INT f_conic_type = FALSE;
	INT f_randomized = FALSE;
	INT nb_times = 0;
	INT f_file = FALSE;
	const BYTE *fname = NULL;
	INT f_add_nucleus = FALSE;
	INT f_TDO = FALSE;
	INT f_maxdepth_for_TDO = FALSE;
	INT maxdepth_for_TDO = INT_MAX;
	INT f_stabilizer = FALSE;
	INT stabilizer_starter_size = 0;



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
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
		
	if (f_v) {
		cout << "process_PG" << endl;
		cout << "We will now compute the plane type" << endl;
		}


	F = new finite_field;

	F->init_override_polynomial(q, poly, 0);

	P = new projective_space;
	
	if (f_v) {
		cout << "process_PG before P->init" << endl;
		}


	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "process_PG after P->init" << endl;
		}

	INT *set;
	INT set_size;
	INT nb_input;


	if (file_size(fname) <= 0) {
		cout << "The file " << fname << " does not exist or is empty" << endl;
		exit(1);
		}


	nb_input = count_number_of_orbits_in_file(fname, 0 /*verbose_level */);

	cout << "The file " << fname << " contains " << nb_input << " solutions" << endl;


	set = NEW_INT(ONE_MILLION);
	{
	BYTE buf[MY_BUFSIZE];
	BYTE *p_buf;
	INT cnt;
	
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
			INT tmp;
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
				INT_vec_print(cout, set, set_size);
				cout << endl;
				}




			if (f_conic_type) {
		
				if (f_v) {
					cout << "solution " << cnt << " / " << nb_input << ", computing the conic intersection type" << endl;
					}

				INT *intersection_type;
				INT highest_intersection_number;
				INT f_save_largest_sets = TRUE;
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

				INT *Inc;
				INT m, n;
				
				largest_sets->compute_incidence_matrix(Inc, m, n,  verbose_level);
				INT_matrix_print(Inc, m, n);

				decomposition *D;

				largest_sets->init_decomposition(D, verbose_level);

				INT max_depth = 2;
				
				D->setup_default_partition(verbose_level);

				D->compute_TDO(max_depth, verbose_level);

				INT f_enter_math = TRUE;
				INT f_print_subscripts = TRUE;

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
				FREE_INT(Inc);
				FREE_INT(intersection_type);
				delete D;
				}

			else if (f_add_nucleus) {
				INT nucleus;
				INT *set2;
				INT sz2;
				BYTE fname_base[1000];
				BYTE fname2[1000];
				
				P->find_nucleus(set, set_size, nucleus, verbose_level);

				sz2 = set_size + 1;
				set2 = NEW_INT(sz2);
				INT_vec_copy(set, set2, set_size);
				set2[set_size] = nucleus;
				
				get_fname_base(fname, fname_base);
				sprintf(fname2, "%s_hyperoval_%ld.txt", fname_base, cnt);
				write_set_to_file(fname2, set2, sz2, verbose_level);
				cout << "Written file " << fname2 << " of size " << file_size(fname2) << endl;
				}

			else if (f_TDO) {

				INT max_depth;

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

				D = new decomposition;

				cout << "before D->init_inc_and_stack" << endl;
				D->init_inc_and_stack(Inc, Stack, verbose_level);

				INT m, n;
				INT *M;
				
				P->make_incidence_matrix(m, n, M, verbose_level);
				D->init_incidence_matrix(m, n, M, verbose_level);
				FREE_INT(M);
				
				cout << "before D->compute_TDO, max_depth=" << max_depth << endl;
				D->compute_TDO(max_depth, verbose_level);
				cout << "after D->compute_TDO" << endl;

				INT f_enter_math = TRUE;
				INT f_print_subscripts = TRUE;

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
				delete D;
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

	FREE_INT(set);
	delete P;
	delete F;


	the_end(t0);
}


