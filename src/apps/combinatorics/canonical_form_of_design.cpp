/*
 * canonical_form_of_design.cpp
 *
 *  Created on: Jan 25, 2020
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started


void classify_objects_using_nauty(
	data_input_stream *Data,
	classify_bitvectors *CB,
	int f_save_incma_in_and_out, const char *prefix,
	const char **test_perm, int nb_test_perm,
	int verbose_level);
int process_object(
	classify_bitvectors *CB,
	int *Incma, int nb_rows, int nb_cols,
	int *partition,
	int f_save_incma_in_and_out, const char *prefix,
	int nb_objects_to_test,
	action *&A_perm,
	long int *canonical_labeling,
	int verbose_level);
action *set_stabilizer_of_incma_object(
		int *Incma, int nb_rows, int nb_cols,
		int *partition,
		int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
		int f_compute_canonical_form,
		uchar *&canonical_form,
		int &canonical_form_len,
		long int *canonical_labeling,
		int verbose_level);



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;

	int f_input = FALSE;
	data_input_stream *Data_input_stream = NULL;


	int f_save_incma_in_and_out = FALSE;

	int f_prefix = FALSE;
	const char *prefix = "";

	int f_save = FALSE;
	const char *output_prefix = "";

	int f_classify_nauty = FALSE;

	int f_report = FALSE;

	int f_max_TDO_depth = FALSE;
	int max_TDO_depth = INT_MAX;

	const char *test_perm[1000];
	int nb_test_perm = 0;

	os_interface Os;

	t0 = Os.os_ticks();


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data_input_stream = NEW_OBJECT(data_input_stream);
			i += Data_input_stream->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-input" << endl;
		}


		else if (strcmp(argv[i], "-save_incma_in_and_out") == 0) {
			f_save_incma_in_and_out = TRUE;
			cout << "-save_incma_in_and_out" << endl;
		}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
		}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			output_prefix = argv[++i];
			cout << "-save " << output_prefix << endl;
		}
		else if (strcmp(argv[i], "-classify_nauty") == 0) {
			f_classify_nauty = TRUE;
			cout << "-classify_nauty " << endl;
		}
		else if (strcmp(argv[i], "-test_perm") == 0) {
			test_perm[nb_test_perm++] = argv[++i];
			cout << "-test_perm " << test_perm[nb_test_perm - 1] << endl;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (strcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = atoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}
	}


	//int f_v = (verbose_level >= 1);

	if (!f_input) {
		cout << "please use option -input ... -end" << endl;
		exit(1);
	}


	if (f_classify_nauty) {
		// classify:

		classify_bitvectors *CB;

		CB = NEW_OBJECT(classify_bitvectors);

		classify_objects_using_nauty(
			Data_input_stream,
			CB,
			f_save_incma_in_and_out, prefix,
			test_perm, nb_test_perm,
			verbose_level);


	}




	the_end(t0);
}


void classify_objects_using_nauty(
	data_input_stream *Data,
	classify_bitvectors *CB,
	int f_save_incma_in_and_out, const char *prefix,
	const char **test_perm, int nb_test_perm,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int input_idx;
	int t0, t1, dt;
	file_io Fio;
	os_interface Os;

	if (f_v) {
		cout << "classify_objects_using_nauty" << endl;
	}

	int nb_objects_to_test;

	if (f_v) {
		cout << "classify_objects_using_nauty "
				"before count_number_of_objects_to_test" << endl;
	}
	nb_objects_to_test = Data->count_number_of_objects_to_test(
		verbose_level - 1);

	t0 = Os.os_ticks();

	for (input_idx = 0; input_idx < Data->nb_inputs; input_idx++) {
		if (f_v) {
			cout << "classify_objects_using_nauty input "
					<< input_idx << " / " << Data->nb_inputs
					<< " is:" << endl;
		}

		if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_DESIGNS) {
			cout << "classify_objects_using_nauty "
				"input from file " << Data->input_string[input_idx]
				<< ":" << endl;



			int nck;
			int N_points = Data->input_data1[input_idx];
			int b = Data->input_data2[input_idx];
			int k = Data->input_data3[input_idx];
			int class_size = Data->input_data4[input_idx];
			int nb_classes = b / class_size;
			int *block;

			combinatorics_domain Combi;

			nck = Combi.int_n_choose_k(N_points, k);

			block = NEW_int(k);
			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			cout << "N_points=" << N_points << endl;
			cout << "b=" << b << endl;
			cout << "k=" << k << endl;
			cout << "class_size=" << class_size << endl;
			cout << "nb_classes=" << nb_classes << endl;
			cout << "classify_objects_using_nauty Reading the file " << Data->input_string[input_idx] << endl;
			SoS->init_from_file(
					nck /* underlying_set_size */,
					Data->input_string[input_idx], verbose_level);
			cout << "Read the file " << Data->input_string[input_idx] << endl;

			int h;


			cout << "classify_objects_using_nauty processing " << SoS->nb_sets << " objects" << endl;

			for (h = 0; h < SoS->nb_sets; h++) {


				long int *the_set_in;
				int set_size_in;
				//object_in_projective_space *OiP;


				set_size_in = SoS->Set_size[h];
				the_set_in = SoS->Sets[h];

				if (set_size_in != b) {
					cout << "classify_objects_using_nauty set_size_in != b" << endl;
					exit(1);
				}

				if (f_vv || ((h % 1024) == 0)) {
					cout << "classify_objects_using_nauty The input set " << h << " / " << SoS->nb_sets
						<< " has size " << set_size_in << ":" << endl;
				}

				if (f_vvv) {
					cout << "classify_objects_using_nauty The input set is:" << endl;
					lint_vec_print(cout, the_set_in, set_size_in);
					cout << endl;
				}

#if 1

				action *A_perm;
				combinatorics_domain Combi;

				int *Incma;
				int *partition;
				int nb_rows = N_points + nb_classes;
				int nb_cols = b + 1;
				int N = nb_rows + nb_cols;
				long int *canonical_labeling;
				int u, i, j, t, a;


				Incma = NEW_int(nb_rows * nb_cols);
				int_vec_zero(Incma, nb_rows * nb_cols);

				for (u = 0; u < nb_classes; u++) {
					for (j = 0; j < class_size; j++) {
						a = the_set_in[u * class_size + j];
						Combi.unrank_k_subset(a, block, N_points, k);
						for (t = 0; t < k; t++) {
							i = block[t];
							Incma[i * nb_cols + u * class_size + j] = 1;
						}
						Incma[(N_points + u) * nb_cols + u * class_size + j] = 1;
					}
					Incma[(N_points + u) * nb_cols + nb_cols - 1] = 1;
				}

				partition = NEW_int(N);
				for (i = 0; i < N; i++) {
					partition[i] = 1;
					}
				partition[N_points - 1] = 0;
				partition[nb_rows - 1] = 0;
				partition[nb_rows + b - 1] = 0;
				partition[nb_rows + nb_cols - 1] = 0;
				canonical_labeling = NEW_lint(nb_rows + nb_cols);



				if (!process_object(
						CB,
						Incma, nb_rows, nb_cols,
						partition,
						f_save_incma_in_and_out, prefix,
						nb_objects_to_test,
						A_perm,
						canonical_labeling,
						verbose_level - 3)) {

					cout << "skip" << endl;

				}
				else {
					t1 = Os.os_ticks();
					//cout << "poset_classification::print_level_info t0=" << t0 << endl;
					//cout << "poset_classification::print_level_info t1=" << t1 << endl;
					dt = t1 - t0;
					//cout << "poset_classification::print_level_info dt=" << dt << endl;

					cout << "Time ";
					Os.time_check_delta(cout, dt);

					longinteger_object go;

					A_perm->group_order(go);

					cout << " --- New isomorphism type! input set " << h
							<< " / " << SoS->nb_sets << " The n e w number of "
							"isomorphism types is " << CB->nb_types << " go=" << go << endl;

					//int idx;

				}
#endif

				FREE_int(Incma);
				FREE_int(partition);
				FREE_lint(canonical_labeling);

				if (f_vv) {
					cout << "classify_objects_using_nauty after input set " << h << " / "
							<< SoS->nb_sets
							<< ", we have " << CB->nb_types
							<< " isomorphism types of objects" << endl;
				}

			} // next h
			FREE_OBJECT(SoS);
			FREE_int(block);
		} // if INPUT_TYPE_FILE_OF_DESIGNS
		else {
			cout << "classify_objects_using_nauty unknown input type" << endl;
			exit(1);
		}
	} // next input_idx


	int t;

	for (t = 0; t < nb_test_perm; t++) {
		int *perm;
		int degree;
		int i;

		cout << "testing permutation " << test_perm[t] << endl;
		scan_permutation_from_string(test_perm[t],
			perm, degree, verbose_level);
		cout << "the permutation is:" << endl;
		for (i = 0; i < degree; i++) {
			cout << i << " -> " << perm[i] << endl;
		}
	}
	if (f_v) {
		cout << "classify_objects_using_nauty before compute_and_print_ago_distribution" << endl;
	}

	compute_and_print_ago_distribution(cout, CB, verbose_level);

	if (f_v) {
		cout << "classify_objects_using_nauty after compute_and_print_ago_distribution" << endl;
	}

	if (f_v) {
		cout << "classify_objects_using_nauty before CB->finalize" << endl;
	}

	//CB->finalize(verbose_level); // computes C_type_of and perm

	if (f_v) {
		cout << "classify_objects_using_nauty done" << endl;
	}
}


int process_object(
	classify_bitvectors *CB,
	int *Incma, int nb_rows, int nb_cols,
	int *partition,
	int f_save_incma_in_and_out, const char *prefix,
	int nb_objects_to_test,
	action *&A_perm,
	long int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "process_object n=" << CB->n << endl;
	}

	longinteger_object go;
	//int *Extra_data;
	char save_incma_in_and_out_prefix[1000];

	if (f_save_incma_in_and_out) {
		sprintf(save_incma_in_and_out_prefix, "%s_%d_", prefix, CB->n);
	}


	uchar *canonical_form;
	int canonical_form_len;


	if (f_v) {
		cout << "process_object "
				"before set_stabilizer_of_incma_object" << endl;
	}

	A_perm = set_stabilizer_of_incma_object(
			Incma, nb_rows, nb_cols,
			partition,
			f_save_incma_in_and_out, save_incma_in_and_out_prefix,
			TRUE /* f_compute_canonical_form */,
			canonical_form, canonical_form_len,
			canonical_labeling,
			verbose_level - 2);


	if (f_v) {
		cout << "process_object "
				"after set_stabilizer_of_incma_object" << endl;
	}


	A_perm->group_order(go);

	cout << "generators for the automorphism group are:" << endl;
	A_perm->Strong_gens->print_generators_tex(cout);

	//cout << "object:" << endl;
	//OiP->print(cout);
	//cout << "go=" << go << endl;
#if 0
	cout << "process_object canonical form: ";
	for (i = 0; i < canonical_form_len; i++) {
		cout << (int)canonical_form[i];
		if (i < canonical_form_len - 1) {
			cout << ", ";
		}
	}
#endif
	//cout << endl;

#if 0
	Extra_data = NEW_int(OiP->sz);
	int_vec_copy(OiP->set, Extra_data, OiP->sz);

	if (CB->n == 0) {
		CB->init(nb_objects_to_test, canonical_form_len, verbose_level);
		sz = OiP->sz;
	}
	else {
		if (OiP->sz != sz) {
			cout << "process_object "
					"OiP->sz != sz" << endl;
			exit(1);
		}
	}
	if (!CB->add(canonical_form, Extra_data, verbose_level)) {
		FREE_int(Extra_data);
	}
#endif
	if (CB->n == 0) {
		CB->init(nb_objects_to_test, canonical_form_len, verbose_level);
	}
	ret = CB->add(canonical_form, NULL /*OiP*/, verbose_level);


	//delete SG;

	if (f_v) {
		cout << "process_object done" << endl;
	}
	return ret;
}


action *set_stabilizer_of_incma_object(
		int *Incma, int nb_rows, int nb_cols,
		int *partition,
		int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
		int f_compute_canonical_form,
		uchar *&canonical_form,
		int &canonical_form_len,
		long int *canonical_labeling,
		int verbose_level)
// canonical_labeling[nb_rows + nb_cols]
{

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	//int *labeling;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	long int *Base_lint;
	int *Transversal_length, Ago;
	int N, i, j, a, L;
	combinatorics_domain Combi;
	file_io Fio;


	if (f_v) {
		cout << "set_stabilizer_of_incma_object" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}


	if (verbose_level > 5) {
		cout << "set_stabilizer_of_incma_object Incma:" << endl;
		int_matrix_print_tight(Incma, nb_rows, nb_cols);
	}

	//canonical_labeling = NEW_int(nb_rows + nb_cols);
	for (i = 0; i < nb_rows + nb_cols; i++) {
		canonical_labeling[i] = i;
		}


	if (f_save_incma_in_and_out) {
		cout << "set_stabilizer_of_incma_object Incma:" << endl;
		if (nb_rows < 10) {
			print_integer_matrix_width(cout,
					Incma, nb_rows, nb_cols, nb_cols, 1);
			}
		else {
			cout << "too large to print" << endl;
			}

		char fname_csv[1000];
		char fname_bin[1000];

		sprintf(fname_csv, "%sIncma_in_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_in_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		Fio.int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
				Incma, nb_rows, nb_cols,
				TRUE, canonical_labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_int(Incma);
		FREE_OBJECT(CG);
		}



	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;


	if (f_vv) {
		cout << "set_stabilizer_of_incma_object initializing Aut, Base, "
				"Transversal_length" << endl;
		}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);

	if (f_v) {
		cout << "set_stabilizer_of_incma_object calling nauty_interface_matrix_int" << endl;
		}
	int t0, t1, dt, tps;
	double delta_t_in_sec;
	os_interface Os;

	tps = Os.os_ticks_per_second();
	t0 = Os.os_ticks();

	int *can_labeling;

	can_labeling = NEW_int(nb_rows + nb_cols);

	nauty_interface_matrix_int(
		Incma, nb_rows, nb_cols,
		can_labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, Ago, verbose_level - 3);

	for (i = 0; i < nb_rows + nb_cols; i++) {
		canonical_labeling[i] = can_labeling[i];
		}
	FREE_int(can_labeling);

	int_vec_copy_to_lint(Base, Base_lint, Base_length);

	t1 = Os.os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) dt / (double) tps;

	if (f_v) {
		cout << "set_stabilizer_of_incma_object done with nauty_interface_matrix_int, "
				"Ago=" << Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
		}
	if (verbose_level > 5) {
		int h;
		int degree = nb_rows +  nb_cols;

		for (h = 0; h < Aut_counter; h++) {
			cout << "aut generator " << h << " / "
					<< Aut_counter << " : " << endl;
			Combi.perm_print(cout, Aut + h * degree, degree);
			cout << endl;
		}
	}

	int *Incma_out;
	int ii, jj;
	if (f_vvv) {
		cout << "set_stabilizer_of_incma_object labeling:" << endl;
		lint_vec_print(cout, canonical_labeling, nb_rows + nb_cols);
		cout << endl;
		}

	Incma_out = NEW_int(L);
	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma[ii * nb_cols + jj];
			}
		}
	if (f_vvv) {
		cout << "set_stabilizer_of_incma_object Incma Out:" << endl;
		if (nb_rows < 20) {
			print_integer_matrix_width(cout,
					Incma_out, nb_rows, nb_cols, nb_cols, 1);
			}
		else {
			cout << "set_stabilizer_of_incma_object too large to print" << endl;
			}
		}



	if (f_compute_canonical_form) {


		canonical_form = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form, a);
					}
				}
			}

		}



	FREE_int(Incma_out);

	action *A_perm;
	longinteger_object ago;


	A_perm = NEW_OBJECT(action);

	if (f_v) {
		cout << "set_stabilizer_of_incma_object before init_permutation_group_from_generators" << endl;
		}
	ago.create(Ago, __FILE__, __LINE__);
	A_perm->init_permutation_group_from_generators(N,
		TRUE, ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		verbose_level);

	if (f_vv) {
		cout << "set_stabilizer_of_incma_object created action ";
		A_perm->print_info();
		cout << endl;
		}


	if (f_v) {
		cout << "set_stabilizer_of_incma_object done" << endl;
		}
	return A_perm;
}
