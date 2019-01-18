// projective_space_main.C
// 
// Anton Betten
// December 22, 2017
//
// 
//
//

#include "orbiter.h"





// global data:

int t0; // the system time when the program started



void canonical_form(data_input_stream *Data,
	int nb_objects_to_test, 
	projective_space_with_action *PA, 
	int f_save_incma_in_and_out, const char *prefix, 
	int verbose_level);

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

	int order_list_sz = 0;
	int order_list[1000];

	int f_classify_nauty = FALSE;
	int f_classify_backtrack = FALSE;

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
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data_input_stream = NEW_OBJECT(data_input_stream);
			i += Data_input_stream->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

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
			order_list[order_list_sz] = atoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< order_list[order_list_sz] << endl;
			order_list_sz++;
			}
		else if (strcmp(argv[i], "-classify_nauty") == 0) {
			f_classify_nauty = TRUE;
			cout << "-classify_nauty " << endl;
			}
		else if (strcmp(argv[i], "-classify_backtrack") == 0) {
			f_classify_backtrack = TRUE;
			cout << "-classify_backtrack " << endl;
			}
		}


	int f_v = (verbose_level >= 1);

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
	

	if (is_prime(q)) {
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


	int nb_objects_to_test;
	
	cout << "before count_number_of_objects_to_test" << endl;
	nb_objects_to_test = Data_input_stream->count_number_of_objects_to_test(
		verbose_level - 1);



	cout << "nb_objects_to_test=" << nb_objects_to_test << endl;




	if (f_classify_nauty) {
		// classify:

		classify_bitvectors *CB;

		CB = NEW_OBJECT(classify_bitvectors);
	

		cout << "before classify_objects_using_nauty" << endl;
		PA->classify_objects_using_nauty(Data_input_stream,
			nb_objects_to_test, 
			CB,
			f_save_incma_in_and_out, prefix, 
			verbose_level - 1);



		cout << "We found " << CB->nb_types << " types" << endl;


		compute_and_print_ago_distribution_with_classes(cout,
				CB, verbose_level);


		cout << "In the ordering of canonical forms, they are" << endl;
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
			CB->save(output_prefix, 
				OiPA_encode, OiPA_group_order, 
				NULL /* void *global_data */, 
				verbose_level);
			}


		cout << "The orbits in more detail:" << endl;
		int j;
	
		char fname[1000];

		if (prefix == NULL) {
			cout << "please use option -prefix <prefix> toset the "
					"prefix for the tex file" << endl;
			exit(1);
			}
		sprintf(fname, "%s_classification.tex", prefix);


		{
		ofstream fp(fname);
	
		latex_head_easy(fp);

		int *Table;
		int width = 4;
		int *row_labels;
		int *col_labels;
		int row_part_first[2], row_part_len[1];
		int nb_row_parts = 1;
		int col_part_first[2], col_part_len[1];
		int nb_col_parts = 1;



		row_part_first[0] = 0;
		row_part_first[1] = CB->nb_types;
		row_part_len[0] = CB->nb_types;

		col_part_first[0] = 0;
		col_part_first[1] = width;
		col_part_len[0] = width;

		Table = NEW_int(CB->nb_types * width);
		int_vec_zero(Table, CB->nb_types * width);

		row_labels = NEW_int(CB->nb_types);
		col_labels = NEW_int(width);
		for (i = 0; i < CB->nb_types; i++) {
			row_labels[i] = i;
			}
		for (j = 0; j < width; j++) {
			col_labels[j] = j;
			}

		for (i = 0; i < CB->nb_types; i++) {

			j = CB->perm[i];
			Table[i * width + 0] = CB->Type_rep[j];
			Table[i * width + 1] = CB->Type_mult[j];
			Table[i * width + 2] = 0; // group order
			Table[i * width + 3] = 0; // object list
			}

		fp << "\\section{Summary of Orbits}" << endl;
		fp << "$$" << endl;
		int_matrix_print_with_labels_and_partition(fp,
				Table, CB->nb_types, 4,
			row_labels, col_labels, 
			row_part_first, row_part_len, nb_row_parts,  
			col_part_first, col_part_len, nb_col_parts,  
			print_summary_table_entry, 
			CB /*void *data*/, 
			TRUE /* f_tex */);
		fp << "$$" << endl;

		compute_and_print_ago_distribution_with_classes(fp,
				CB, verbose_level);

		for (i = 0; i < CB->nb_types; i++) {

			j = CB->perm[i];
			object_in_projective_space_with_action *OiPA;
			object_in_projective_space *OiP;
		
			cout << "###################################################"
					"#############################" << endl;
			cout << "Orbit " << i << " / " << CB->nb_types
					<< " is canonical form no " << j
					<< ", original object no " << CB->Type_rep[j]
					<< ", frequency " << CB->Type_mult[j]
					<< " : " << endl;


			{
			int *Input_objects;
			int nb_input_objects;
			CB->C_type_of->get_class_by_value(Input_objects,
				nb_input_objects, j, 0 /*verbose_level */);

			cout << "This isomorphism type appears " << nb_input_objects
					<< " times, namely for the following "
							"input objects:" << endl;
			int_vec_print_as_matrix(cout, Input_objects,
					nb_input_objects, 10 /* width */,
					FALSE /* f_tex */);

			FREE_int(Input_objects);
			}

			OiPA = (object_in_projective_space_with_action *)
					CB->Type_extra_data[j];
			OiP = OiPA->OiP;
			if (OiP->type != t_PAC) {
				OiP->print(cout);
				}

			//OiP->init_point_set(PA->P, (int *)CB->Type_extra_data[j],
			//sz, 0 /* verbose_level*/);



			strong_generators *SG;
			longinteger_object go;
			char save_incma_in_and_out_prefix[1000];
	
			if (f_save_incma_in_and_out) {
				sprintf(save_incma_in_and_out_prefix,
						"%s_iso_%d_%d", prefix, i, j);
				}
	
	
			uchar *canonical_form;
			int canonical_form_len;


			SG = PA->set_stabilizer_of_object(
				OiP, 
				f_save_incma_in_and_out, save_incma_in_and_out_prefix, 
				TRUE /* f_compute_canonical_form */,
				canonical_form, canonical_form_len,
				0 /* verbose_level */);

			SG->group_order(go);
	
			fp << "\\section*{Orbit " << i << " / "
				<< CB->nb_types << "}" << endl;
			fp << "Orbit " << i << " / " << CB->nb_types <<  " stored at "
				<< j << " is represented by input object "
				<< CB->Type_rep[j] << " and appears "
				<< CB->Type_mult[j] << " times: \\\\" << endl;
			if (OiP->type != t_PAC) {
				OiP->print(fp);
				fp << "\\\\" << endl;
				}
			//int_vec_print(fp, OiP->set, OiP->sz);
			fp << "Group order " << go << "\\\\" << endl;

			fp << "Stabilizer:" << endl;
			SG->print_generators_tex(fp);

			{
			int *Input_objects;
			int nb_input_objects;
			CB->C_type_of->get_class_by_value(Input_objects,
					nb_input_objects, j, 0 /*verbose_level */);
			int_vec_heapsort(Input_objects, nb_input_objects);

			fp << "This isomorphism type appears " << nb_input_objects
					<< " times, namely for the following "
					<< nb_input_objects << " input objects: " << endl;
			if (nb_input_objects < 10) {
				fp << "$" << endl;
				int_set_print_tex(fp, Input_objects, nb_input_objects);
				fp << "$\\\\" << endl;
				}
			else {
				fp << "$$" << endl;
				int_vec_print_as_matrix(fp, Input_objects,
					nb_input_objects, 10 /* width */, TRUE /* f_tex */);
				fp << "$$" << endl;
				}

			FREE_int(Input_objects);
			}


			int *Incma;
			int nb_rows, nb_cols;
			int *partition;
			incidence_structure *Inc;
			partitionstack *Stack;


			OiP->encode_incma_and_make_decomposition(
				Incma, nb_rows, nb_cols, partition, 
				Inc, 
				Stack, 
				verbose_level);
			FREE_int(Incma);
			FREE_int(partition);
	#if 0
			cout << "set ";
			int_vec_print(cout, OiP->set, OiP->sz);
			cout << " go=" << go << endl;

			cout << "Stabilizer:" << endl;
			SG->print_generators_tex(cout);


			incidence_structure *Inc;
			partitionstack *Stack;
	
			int Sz[1];
			int *Subsets[1];

			Sz[0] = OiP->sz;
			Subsets[0] = OiP->set;
		
			cout << "computing decomposition:" << endl;
			PA->P->decomposition(1 /* nb_subsets */, Sz, Subsets, 
				Inc, 
				Stack, 
				verbose_level);

	#if 0
			cout << "the decomposition is:" << endl;
			Inc->get_and_print_decomposition_schemes(*Stack);
			Stack->print_classes(cout);
	#endif




	#if 0
			fp << "canonical form: ";
			for (i = 0; i < canonical_form_len; i++) {
				fp << (int)canonical_form[i];
				if (i < canonical_form_len - 1) {
					fp << ", ";
					}
				}
			fp << "\\\\" << endl;
	#endif
	#endif
	
		
			Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				fp, TRUE /* f_enter_math */,
				TRUE /* f_print_subscripts */, *Stack);

	#if 0
			Inc->get_and_print_tactical_decomposition_scheme_tex(
				fp, TRUE /* f_enter_math */,
				*Stack);
	#endif



			int f_refine_prev, f_refine, h;
			int f_print_subscripts = TRUE;

			f_refine_prev = TRUE;
			for (h = 0; ; h++) {
				if (EVEN(h)) {
					f_refine = Inc->refine_column_partition_safe(
							*Stack, verbose_level - 3);
					}
				else {
					f_refine = Inc->refine_row_partition_safe(
							*Stack, verbose_level - 3);
					}

				if (f_v) {
					cout << "incidence_structure::compute_TDO_safe "
							"h=" << h << " after refine" << endl;
					}
				if (EVEN(h)) {
					//int f_list_incidences = FALSE;
					Inc->get_and_print_column_tactical_decomposition_scheme_tex(
						fp, TRUE /* f_enter_math */,
						f_print_subscripts, *Stack);
					//get_and_print_col_decomposition_scheme(
					//PStack, f_list_incidences, FALSE);
					//PStack.print_classes_points_and_lines(cout);
					}
				else {
					//int f_list_incidences = FALSE;
					Inc->get_and_print_row_tactical_decomposition_scheme_tex(
						fp, TRUE /* f_enter_math */,
						f_print_subscripts, *Stack);
					//get_and_print_row_decomposition_scheme(
					//PStack, f_list_incidences, FALSE);
					//PStack.print_classes_points_and_lines(cout);
					}
		
				if (!f_refine_prev && !f_refine) {
					break;
					}
				f_refine_prev = f_refine;
				}

			cout << "Classes of the partition:\\\\" << endl; 
			Stack->print_classes_tex(fp);



			OiP->klein(verbose_level);

		
			sims *Stab;
			int *Elt;
			int nb_trials;
			int max_trials = 100;

			Stab = SG->create_sims(verbose_level);
			Elt = NEW_int(PA->A->elt_size_in_int);
		
			for (h = 0; h < order_list_sz; h++) {
				if (Stab->find_element_of_given_order_int(Elt,
						order_list[h], nb_trials, max_trials,
						verbose_level)) {
					fp << "We found an element of order "
							<< order_list[h] << ", which is:" << endl;
					fp << "$$" << endl;
					PA->A->element_print_latex(Elt, fp);
					fp << "$$" << endl;
					PA->report_fixed_objects_in_PG_3_tex(
						Elt, fp, 
						verbose_level);
					}
				else {
					fp << "We could not find an element of order "
						<< order_list[h] << "\\\\" << endl;
					}
				}


			FREE_int(Elt);
			FREE_OBJECT(Stack);
			FREE_OBJECT(Inc);
			FREE_OBJECT(SG);

			}


		latex_foot(fp);
		}

		cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
		
		//FREE_int(perm);
		//FREE_int(v);

		} // if (f_classify_nauty)

	else if (f_classify_backtrack) {

		
		canonical_form(Data_input_stream,
			nb_objects_to_test, 
			PA, 
			f_save_incma_in_and_out, prefix, 
			verbose_level);

		
		}

	
	the_end(t0);
}

void canonical_form(data_input_stream *Data,
	int nb_objects_to_test, 
	projective_space_with_action *PA, 
	int f_save_incma_in_and_out, const char *prefix, 
	int verbose_level)
// ToDo finish this for types lines and packings
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int input_idx;
	
	if (f_v) {
		cout << "canonical_form" << endl;
		}

	for (input_idx = 0; input_idx < Data->nb_inputs; input_idx++) {
		cout << "input " << input_idx << " / " << Data->nb_inputs
				<< " is:" << endl;

		if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
			cout << "input set of points " << Data->input_string[input_idx]
				<< ":" << endl;

			object_in_projective_space *OiP;
			//strong_generators *SG;
			
			OiP = PA->create_object_from_string(t_PTS,
					Data->input_string[input_idx], verbose_level);


			sims *Aut;
			Aut = NEW_OBJECT(sims);
			int f_get_automorphism_group = TRUE;
			int total_backtrack_nodes = 0;
			int *canonical_set;
			int *transporter;


			
			canonical_set = NEW_int(OiP->sz);
			transporter = NEW_int(PA->A->elt_size_in_int);


			PA->A->make_canonical(
				OiP->sz, OiP->set, 
				canonical_set, transporter, 
				total_backtrack_nodes, 
				f_get_automorphism_group, Aut,
				verbose_level);
#if 0
			if (!process_object(PA, CB, OiP, 
				f_save_incma_in_and_out, prefix, 
				nb_objects_to_test, 
				SG, 
				verbose_level)) {
	
				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				}
			else {
				cout << "New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);
				
				OiPA->init(OiP, SG, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				compute_and_print_ago_distribution(cout, PA, CB, verbose_level);
				}
#endif
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {
			cout << "input set of lines " << Data->input_string[input_idx]
				<< ":" << endl;

			object_in_projective_space *OiP;
			//strong_generators *SG;
			
			OiP = PA->create_object_from_string(t_LNS,
					Data->input_string[input_idx], verbose_level);
#if 0
			if (!process_object(PA, CB, OiP, 
				f_save_incma_in_and_out, prefix, 
				nb_objects_to_test, 
				SG, 
				verbose_level)) {
	
				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				}
			else {
				cout << "New isomorphism type! The n e w number of "
						"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);
				
				OiPA->init(OiP, SG, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				compute_and_print_ago_distribution(cout, PA, CB, verbose_level);
				}
#endif
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {
			cout << "input set of packing " << Data->input_string[input_idx]
				<< ":" << endl;

			object_in_projective_space *OiP;
			//strong_generators *SG;
			
			OiP = PA->create_object_from_string(t_PAC,
					Data->input_string[input_idx], verbose_level);
#if 0
			if (!process_object(PA, CB, OiP, 
				f_save_incma_in_and_out, prefix, 
				nb_objects_to_test, 
				SG, 
				verbose_level)) {
	
				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				}
			else {
				cout << "New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);
				
				OiPA->init(OiP, SG, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				compute_and_print_ago_distribution(cout, PA, CB, verbose_level);
				}
#endif
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS ||
				Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES ||
				Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS ||
				Data->input_type[input_idx] ==
					INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
			cout << "input from file " << Data->input_string[input_idx]
				<< ":" << endl;

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			cout << "Reading the file " << Data->input_string[input_idx] << endl;
			SoS->init_from_file(
				PA->P->N_points /* underlying_set_size */,
				Data->input_string[input_idx], verbose_level);
			cout << "Read the file " << Data->input_string[input_idx] << endl;

			int h;


			// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
			int *Spread_table;
			int nb_spreads;
			int spread_size;

			if (Data->input_type[input_idx] ==
				INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				cout << "Reading spread table from file "
						<< Data->input_string2[input_idx] << endl;
				int_matrix_read_csv(Data->input_string2[input_idx],
						Spread_table, nb_spreads, spread_size,
						0 /* verbose_level */);
				cout << "Reading spread table from file "
						<< Data->input_string2[input_idx] << " done" << endl;
				cout << "The spread table contains " << nb_spreads
						<< " spreads" << endl;
				}

			cout << "processing " << SoS->nb_sets << " objects" << endl;
			
			for (h = 0; h < SoS->nb_sets; h++) {


				int *the_set_in;
				int set_size_in;
				object_in_projective_space *OiP;


				set_size_in = SoS->Set_size[h];
				the_set_in = SoS->Sets[h];
		
				cout << "The input set " << h << " / " << SoS->nb_sets
					<< " has size " << set_size_in << ":" << endl;
				cout << "The input set is:" << endl;
				int_vec_print(cout, the_set_in, set_size_in);
				cout << endl;

				OiP = NEW_OBJECT(object_in_projective_space);

				if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_POINTS) {
					OiP->init_point_set(PA->P,
							the_set_in, set_size_in, 0 /* verbose_level*/);
					}
				else if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_LINES) {
					OiP->init_line_set(PA->P,
							the_set_in, set_size_in, 0 /* verbose_level*/);
					}
				else if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS) {
					OiP->init_packing_from_set(PA->P,
							the_set_in, set_size_in, verbose_level);
					}
				else if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
					OiP->init_packing_from_spread_table(PA->P, the_set_in, 
						Spread_table, nb_spreads,
						spread_size, verbose_level);
					}
				else {
					cout << "unknown type" << endl;
					exit(1);
					}

#if 0
				strong_generators *SG;
				if (!process_object(PA, CB, OiP, 
					f_save_incma_in_and_out, prefix, 
					nb_objects_to_test, 
					SG, 
					verbose_level)) {
	
					FREE_OBJECT(OiP);
					FREE_OBJECT(SG);
					}
				else {
					cout << "New isomorphism type! The n e w number of "
							"isomorphism types is " << CB->nb_types << endl;

					int idx;

					object_in_projective_space_with_action *OiPA;

					OiPA = NEW_OBJECT(object_in_projective_space_with_action);
					
					OiPA->init(OiP, SG, verbose_level);
					idx = CB->type_of[CB->n - 1];
					CB->Type_extra_data[idx] = OiPA;


					compute_and_print_ago_distribution(cout,
							PA, CB, verbose_level);
					}
				cout << "after input set " << h << " / " << SoS->nb_sets
					<< ", we have " << CB->nb_types << " isomorphism "
							"types of objects" << endl;
#endif

				}
			if (Data->input_type[input_idx] ==
					INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				FREE_int(Spread_table);
				}
			FREE_OBJECT(SoS);
			}
		else {
			cout << "unknown input type" << endl;
			exit(1);
			}
		}

	if (f_v) {
		cout << "canonical_form done" << endl;
		}
}



