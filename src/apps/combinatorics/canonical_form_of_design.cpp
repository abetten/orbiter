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
	int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
	const char **test_perm, int nb_test_perm,
	int verbose_level);
void handle_input_file(classify_bitvectors *CB,
		int nb_objects_to_test, int t0,
		const char *fname,
		int N_points, int design_b, int design_k, int partition_class_size,
		int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
		const char **test_perm, int nb_test_perm,
		int verbose_level);
void process_object(
	classify_bitvectors *CB,
	incidence_structure_with_group *IG,
	int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
	int nb_objects_to_test,
	int &f_found, int &idx,
	int verbose_level);



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;

	int f_input = FALSE;
	data_input_stream *Data_input_stream = NULL;


	int f_save_incma_in_and_out = FALSE;
	const char *save_incma_in_and_out_prefix = "";

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
			save_incma_in_and_out_prefix = argv[++i];
			cout << "-save_incma_in_and_out" << endl;
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
			f_save_incma_in_and_out, save_incma_in_and_out_prefix,
			test_perm, nb_test_perm,
			verbose_level);


	}




	the_end(t0);
}


void classify_objects_using_nauty(
	data_input_stream *Data,
	classify_bitvectors *CB,
	int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
	const char **test_perm, int nb_test_perm,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx;
	int t0;
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
			if (f_v) {
				cout << "classify_objects_using_nauty "
					"input " << input_idx << " / " << Data->nb_inputs
					<< " from file " << Data->input_string[input_idx]
					<< ":" << endl;
			}

			int N_points = Data->input_data1[input_idx];
			int design_b = Data->input_data2[input_idx];
			int design_k = Data->input_data3[input_idx];
			int partition_class_size = Data->input_data4[input_idx];

			handle_input_file(CB, nb_objects_to_test, t0,
					Data->input_string[input_idx],
					N_points, design_b, design_k, partition_class_size,
					f_save_incma_in_and_out, save_incma_in_and_out_prefix,
					test_perm, nb_test_perm,
					verbose_level);

			if (f_v) {
				cout << "classify_objects_using_nauty "
					"input " << input_idx << " / " << Data->nb_inputs
					<< " from file " << Data->input_string[input_idx]
					<< " finished" << endl;
			}
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


#if 0
	if (f_v) {
		cout << "classify_objects_using_nauty "
				"before compute_and_print_ago_distribution" << endl;
	}

	compute_and_print_ago_distribution(cout, CB, verbose_level);

	if (f_v) {
		cout << "classify_objects_using_nauty "
				"after compute_and_print_ago_distribution" << endl;
	}

	if (f_v) {
		cout << "classify_objects_using_nauty before CB->finalize" << endl;
	}

	//CB->finalize(verbose_level); // computes C_type_of and perm
#endif

	if (f_v) {
		cout << "classify_objects_using_nauty done" << endl;
	}
}





void handle_input_file(classify_bitvectors *CB,
		int nb_objects_to_test, int t0,
		const char *fname,
		int N_points, int design_b, int design_k, int partition_class_size,
		int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
		const char **test_perm, int nb_test_perm,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	os_interface Os;

	int nck;
	int nb_classes = design_b / partition_class_size;
	int t1, dt;

	if (f_v) {
		cout << "handle_input_file fname=" << fname << endl;
	}

	combinatorics_domain Combi;

	nck = Combi.int_n_choose_k(N_points, design_k);

	set_of_sets *SoS;

	SoS = NEW_OBJECT(set_of_sets);

	if (f_v) {
		cout << "N_points=" << N_points << endl;
		cout << "design_b=" << design_b << endl;
		cout << "design_k=" << design_k << endl;
		cout << "partition_class_size=" << partition_class_size << endl;
		cout << "nb_classes=" << nb_classes << endl;
		cout << "handle_input_file Reading the file " << fname << endl;
	}

	SoS->init_from_file(
			nck /* underlying_set_size */,
			fname, verbose_level);

	if (f_v) {
		cout << "Read the file " << fname << endl;
	}

	int h;
	long int *Ago;


	if (f_v) {
		cout << "classify_objects_using_nauty processing "
			<< SoS->nb_sets << " objects" << endl;
	}

	Ago = NEW_lint(SoS->nb_sets);

	int nb_types0;

	nb_types0 = CB->nb_types;

	for (h = 0; h < SoS->nb_sets; h++) {

		if (f_v) {
			cout << "Input set " << h << " / " << SoS->nb_sets << ":" << endl;
		}

		long int *the_set_in;
		int set_size_in;


		set_size_in = SoS->Set_size[h];
		the_set_in = SoS->Sets[h];



		if (set_size_in != design_b) {
			cout << "classify_objects_using_nauty "
					"set_size_in != design_b" << endl;
			exit(1);
		}

		if (f_v) {
			lint_matrix_print(the_set_in, nb_classes, partition_class_size);
		}
		if (f_vv || ((h % 1024) == 0)) {
			cout << "classify_objects_using_nauty "
					"The input set " << h << " / " << SoS->nb_sets
				<< " has size " << set_size_in << ":" << endl;
		}

		if (f_vvv) {
			cout << "classify_objects_using_nauty "
					"The input set is:" << endl;
			lint_vec_print(cout, the_set_in, set_size_in);
			cout << endl;
		}




		incidence_structure *Inc;
		int *partition;

		Inc = NEW_OBJECT(incidence_structure);
		Inc->init_large_set(
				the_set_in /* blocks */,
				N_points, design_b, design_k, partition_class_size,
				partition, verbose_level);

		incidence_structure_with_group *IG;

		IG = NEW_OBJECT(incidence_structure_with_group);
		IG->init(Inc,
				partition,
				verbose_level - 2);

		int f_found;
		int idx;

		process_object(
					CB,
					IG,
					f_save_incma_in_and_out, save_incma_in_and_out_prefix,
					nb_objects_to_test,
					f_found, idx,
					verbose_level - 2);

		if (f_found) {

			if (f_v) {
				cout << "input set " << h << " found at position " << idx
					<< ", corresponding to input object " << CB->Type_rep[idx]
					<< " and hence is skipped" << endl;
			}
			FREE_OBJECT(IG);
			FREE_OBJECT(Inc);
			FREE_int(partition);

		}
		else {
			t1 = Os.os_ticks();
			//cout << "poset_classification::print_level_info t0=" << t0 << endl;
			//cout << "poset_classification::print_level_info t1=" << t1 << endl;
			dt = t1 - t0;
			//cout << "poset_classification::print_level_info dt=" << dt << endl;

			if (f_v) {
				cout << "Time ";
				Os.time_check_delta(cout, dt);
			}

			longinteger_object go;

			IG->A_perm->group_order(go);

			if (f_v) {
				cout << " --- New isomorphism type! input set " << h
					<< " / " << SoS->nb_sets << " The n e w number of "
					"isomorphism types is " << CB->nb_types << " go=" << go << endl;
			}

			Ago[CB->nb_types - 1 - nb_types0] = go.as_lint();

		}

		if (f_vv) {
			cout << "classify_objects_using_nauty after input set " << h << " / "
					<< SoS->nb_sets
					<< ", we have " << CB->nb_types
					<< " isomorphism types of objects" << endl;
		}

	} // next h
	FREE_OBJECT(SoS);

	if (f_v) {
		cout << "distribution of automorphism group orders of new isomorphism types in this file:" << endl;
		classify C;

		C.init_lint(Ago, CB->nb_types - nb_types0, FALSE, 0);
		C.print_naked(TRUE);
		cout << endl;
	}

	if (f_v) {
		cout << "handle_input_file fname=" << fname << " done" << endl;
	}

}


void process_object(
	classify_bitvectors *CB,
	incidence_structure_with_group *IG,
	int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
	int nb_objects_to_test,
	int &f_found, int &idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "process_object n=" << CB->n << endl;
	}

	longinteger_object go;




	if (f_v) {
		cout << "process_object "
				"before IG->set_stabilizer_and_canonical_form" << endl;
	}

	IG->set_stabilizer_and_canonical_form(
			f_save_incma_in_and_out, save_incma_in_and_out_prefix,
			TRUE /* f_compute_canonical_form */,
			verbose_level);


	if (f_v) {
		cout << "process_object "
				"after IG->set_stabilizer_and_canonical_form" << endl;
	}


	IG->A_perm->group_order(go);

	if (FALSE) {
		cout << "generators for the automorphism group are:" << endl;
		IG->A_perm->Strong_gens->print_generators_tex(cout);
	}


	if (CB->n == 0) {
		CB->init(nb_objects_to_test, IG->canonical_form_len, verbose_level);
	}
	CB->search_and_add_if_new(IG->canonical_form, IG, f_found, idx, verbose_level);


	if (f_v) {
		cout << "process_object done" << endl;
	}
}

