/*
 * design_apply.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started

void apply(
		data_input_stream *Data,
		int *perm, int degree,
		int verbose_level);
void handle_input_file(int nb_objects_to_test, int t0,
		const char *fname_in,
		int N_points, int design_b, int design_k, int partition_class_size,
		int *perm, int degree,
		int verbose_level);


int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;

	int f_input = FALSE;
	data_input_stream *Data_input_stream = NULL;


	int f_save = FALSE;
	const char *output_prefix = "";

	int f_apply_perm = FALSE;
	const char *apply_perm;

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
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			output_prefix = argv[++i];
			cout << "-save " << output_prefix << endl;
		}
		else if (strcmp(argv[i], "-apply_perm") == 0) {
			f_apply_perm = TRUE;
			apply_perm = argv[++i];
			cout << "-apply_perm " << apply_perm << endl;
		}
	}


	//int f_v = (verbose_level >= 1);

	if (!f_input) {
		cout << "please use option -input ... -end" << endl;
		exit(1);
	}
	if (f_apply_perm) {

		int *perm;
		int degree;
		int i;

		cout << "scanning permutation " << apply_perm << endl;
		scan_permutation_from_string(apply_perm,
			perm, degree, verbose_level + 2);
		cout << "the permutation is:" << endl;
		for (i = 0; i < degree; i++) {
			cout << i << " -> " << perm[i] << endl;
		}

		apply(Data_input_stream, perm, degree, verbose_level);
	}






	the_end(t0);
}


void apply(
		data_input_stream *Data,
		int *perm, int degree,
		int verbose_level)

{
	int f_v = (verbose_level >= 1);
	int nb_objects_to_test;
	int input_idx;
	os_interface Os;


	if (f_v) {
		cout << "apply "
				"before count_number_of_objects_to_test" << endl;
	}
	nb_objects_to_test = Data->count_number_of_objects_to_test(
		verbose_level - 1);

	t0 = Os.os_ticks();

	for (input_idx = 0; input_idx < Data->nb_inputs; input_idx++) {
		if (f_v) {
			cout << "apply input "
					<< input_idx << " / " << Data->nb_inputs
					<< " is:" << endl;
		}

		if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_DESIGNS) {
			if (f_v) {
				cout << "apply "
					"input " << input_idx << " / " << Data->nb_inputs
					<< " from file " << Data->input_string[input_idx]
					<< ":" << endl;
			}

			int N_points = Data->input_data1[input_idx];
			int design_b = Data->input_data2[input_idx];
			int design_k = Data->input_data3[input_idx];
			int partition_class_size = Data->input_data4[input_idx];

			handle_input_file(nb_objects_to_test, t0,
					Data->input_string[input_idx],
					N_points, design_b, design_k, partition_class_size,
					perm, degree,
					verbose_level);

			if (f_v) {
				cout << "apply "
					"input " << input_idx << " / " << Data->nb_inputs
					<< " from file " << Data->input_string[input_idx]
					<< " finished" << endl;
			}
		} // if INPUT_TYPE_FILE_OF_DESIGNS
		else {
			cout << "apply unknown input type" << endl;
			exit(1);
		}
	} // next input_idx

}


void handle_input_file(int nb_objects_to_test, int t0,
		const char *fname_in,
		int N_points, int design_b, int design_k, int partition_class_size,
		int *perm, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "handle_input_file fname_in=" << fname_in << endl;
	}


	//os_interface Os;

	int nck;
	int nb_classes = design_b / partition_class_size;
	//int t1, dt;
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
		cout << "handle_input_file Reading the file " << fname_in << endl;
	}

	if (N_points != degree) {
		cout << "handle_input_file "
				"N_points != degree" << endl;
		exit(1);
	}
	SoS->init_from_file(
			nck /* underlying_set_size */,
			fname_in, verbose_level);

	if (f_v) {
		cout << "Read the file " << fname_in << endl;
	}

	int h, j, a, u, i1, i2;
	int *block1;
	int *block2;
	sorting Sorting;

	block1 = NEW_int(design_k);
	block2 = NEW_int(design_k);

	if (f_v) {
		cout << "handle_input_file processing "
			<< SoS->nb_sets << " objects" << endl;
	}

	for (h = 0; h < SoS->nb_sets; h++) {

		if (f_v) {
			cout << "Input set " << h << " / " << SoS->nb_sets << ":" << endl;
		}

		long int *the_set_in;
		int set_size_in;


		set_size_in = SoS->Set_size[h];
		the_set_in = SoS->Sets[h];

		if (set_size_in != design_b) {
			cout << "handle_input_file "
					"set_size_in != design_b" << endl;
			exit(1);
		}

		for (j = 0; j < design_b; j++) {
			a = the_set_in[j];
			Combi.unrank_k_subset(a, block1, N_points, design_k);
			for (u = 0; u < design_k; u++) {
				i1 = block1[u];
				i2 = perm[i1];
				block2[u] = i2;
			}
			Sorting.int_vec_heapsort(block2, design_k);
			a = Combi.rank_k_subset(block2, N_points, design_k);
			the_set_in[j] = a;
		}



	} // next h

	char fname_out[1000];
	sprintf(fname_out, "%s", fname_in);

	replace_extension_with(fname_out, "_a.csv");

	SoS->save_csv(fname_out,
			TRUE /* f_make_heading */, verbose_level - 1);




	FREE_int(block1);
	FREE_int(block2);

	if (f_v) {
		cout << "handle_input_file done" << endl;
	}
}
