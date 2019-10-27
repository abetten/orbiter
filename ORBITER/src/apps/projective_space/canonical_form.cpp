// canonical_form.cpp
// 
// Anton Betten
// December 22, 2017
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	int f_n = FALSE;
	int n;
	int f_poly = FALSE;
	const char *poly = NULL;

	int f_init_incidence_structure = TRUE;

	int f_input = FALSE;
	data_input_stream *Data_input_stream = NULL;

	int f_all_k_subsets = FALSE;
	int k = 0;

	int f_save_incma_in_and_out = FALSE;

	int f_prefix = FALSE;
	const char *prefix = "";

	int f_save = FALSE;
	const char *output_prefix = "";

	int fixed_structure_order_list_sz = 0;
	int fixed_structure_order_list[1000];

	int f_classify_nauty = FALSE;

	int f_latex = FALSE;

	int f_max_TDO_depth = FALSE;
	int max_TDO_depth = INT_MAX;

	os_interface Os;

	t0 = Os.os_ticks();


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
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data_input_stream = NEW_OBJECT(data_input_stream);
			i += Data_input_stream->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-input" << endl;
			}


		else if (strcmp(argv[i], "-all_k_subsets") == 0) {
			f_all_k_subsets = TRUE;
			k = atoi(argv[++i]);
			cout << "-all_k_subsets " << k << endl;
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
		else if (strcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = atoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
			}
		else if (strcmp(argv[i], "-classify_nauty") == 0) {
			f_classify_nauty = TRUE;
			cout << "-classify_nauty " << endl;
			}
		else if (strcmp(argv[i], "-latex") == 0) {
			f_latex = TRUE;
			cout << "-latex " << endl;
			}
		else if (strcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = atoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
			}
		}


	//int f_v = (verbose_level >= 1);

	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_input) {
		cout << "please use option -input ... -end" << endl;
		exit(1);
		}

	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0);
	
	int f_semilinear;
	number_theory_domain NT;
	

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);
	
	PA->init(F, n, 
		f_semilinear, 
		f_init_incidence_structure, 
		0 /* verbose_level */);




	if (f_classify_nauty) {
		// classify:

		classify_bitvectors *CB;

		CB = NEW_OBJECT(classify_bitvectors);
	

		cout << "canonical_form.cpp before PA->classify_objects_using_nauty" << endl;
		PA->classify_objects_using_nauty(Data_input_stream,
			CB,
			f_save_incma_in_and_out, prefix, 
			verbose_level - 1);
		cout << "canonical_form.cpp after PA->classify_objects_using_nauty" << endl;



		cout << "canonical_form.cpp We found " << CB->nb_types << " types" << endl;


		compute_and_print_ago_distribution_with_classes(cout,
				CB, verbose_level);


		cout << "canonical_form.cpp In the ordering of canonical forms, they are" << endl;
		CB->print_reps();
		cout << "We found " << CB->nb_types << " types:" << endl;
		for (i = 0; i < CB->nb_types; i++) {

			object_in_projective_space_with_action *OiPA;
			object_in_projective_space *OiP;
		
			cout << i << " / " << CB->nb_types << " is "
				<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
			OiPA = (object_in_projective_space_with_action *)
					CB->Type_extra_data[i];
			OiP = OiPA->OiP;
			if (OiP->type != t_PAC) {
				OiP->print(cout);
				}

#if 0
			for (j = 0; j < rep_len; j++) {
				cout << (int) Type_data[i][j];
				if (j < rep_len - 1) {
					cout << ", ";
					}
				}
#endif
			cout << endl;
			}



		if (f_save) {
			cout << "Saving the classification with output prefix "
					<< output_prefix << endl;
			PA->save(output_prefix, CB, verbose_level);
			CB->save(output_prefix, 
				OiPA_encode, OiPA_group_order, 
				NULL /* void *global_data */, 
				verbose_level);
			}


	

		if (f_latex) {

			cout << "Producing a latex report:" << endl;

			char fname[1000];

			if (prefix == NULL) {
				cout << "please use option -prefix <prefix> to set the "
						"prefix for the tex file" << endl;
				exit(1);
				}
			sprintf(fname, "%s_classification.tex", prefix);


			PA->latex_report(fname,
					output_prefix,
					CB,
					f_save_incma_in_and_out,
					fixed_structure_order_list_sz,
					fixed_structure_order_list,
					max_TDO_depth,
					verbose_level);

			}// f_latex

		} // if (f_classify_nauty)



	the_end(t0);
}

