/*
 * projective_space_object_classifier.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




projective_space_object_classifier::projective_space_object_classifier()
{

	Descr = NULL;

	f_projective_space = FALSE;

	PA = NULL;

	IS = NULL;

	CB = NULL;

	Ago = NULL;
	F_reject = NULL;

	nb_orbits = 0;
	Idx_transversal = NULL;
	Ago_transversal = NULL;

	T_Ago = NULL;

}

projective_space_object_classifier::~projective_space_object_classifier()
{
	if (Ago) {
		FREE_lint(Ago);
	}
	if (F_reject) {
		FREE_int(F_reject);
	}
	if (Idx_transversal) {
		FREE_int(Idx_transversal);
	}
	if (Ago_transversal) {
		FREE_lint(Ago_transversal);
	}
	if (T_Ago) {
		FREE_OBJECT(T_Ago);
	}
}

void projective_space_object_classifier::do_the_work(
		projective_space_object_classifier_description *Descr,
		int f_projective_space,
		projective_space_with_action *PA,
		data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work f_projective_space=" << f_projective_space << endl;
	}

	projective_space_object_classifier::f_projective_space = f_projective_space;
	projective_space_object_classifier::PA = PA;
	projective_space_object_classifier::Descr = Descr;
	projective_space_object_classifier::IS = IS;

	//int i;

	CB = NEW_OBJECT(classify_bitvectors);





	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work "
				"before classify_objects_using_nauty" << endl;
	}

	classify_objects_using_nauty(verbose_level - 1);

	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work "
			"after classify_objects_using_nauty" << endl;
	}


	cout << "projective_space_object_classifier::do_the_work We found "
			<< CB->nb_types << " types" << endl;






	if (Descr->f_save_classification) {

		if (TRUE) {
			cout << "projective_space_object_classifier::do_the_work "
					"Saving the classification with "
					"save_prefix " << Descr->save_prefix << endl;
		}

#if 0
		save(Descr->save_prefix, verbose_level);

		CB->save(Descr->save_prefix,
			OiPA_encode, OiPA_group_order,
			NULL /* void *global_data */,
			verbose_level);
#endif

#if 0
		void save(const char *prefix,
			void (*encode_function)(void *extra_data,
				int *&encoding, int &encoding_sz, void *global_data),
			void (*get_group_order_or_NULL)(void *extra_data,
				longinteger_object &go, void *global_data),
			void *global_data,
			int verbose_level);
#endif
	}
	else {
		cout << "projective_space_object_classifier::do_the_work no save" << endl;
	}




	if (Descr->f_report) {

		if (TRUE) {
			cout << "projective_space_object_classifier::do_the_work Producing a latex report:" << endl;
		}


		if (Descr->f_classification_prefix == FALSE) {
			cout << "please use option -classification_prefix <prefix> to set the "
					"prefix for the output file" << endl;
			exit(1);
		}

		string fname;

		fname.assign(Descr->classification_prefix);
		fname.append("_classification.tex");

		if (f_v) {
			cout << "projective_space_object_classifier::do_the_work before latex_report" << endl;
		}

		latex_report(fname,
				Descr->report_prefix,
				Descr->fixed_structure_order_list_sz,
				Descr->fixed_structure_order_list,
				Descr->max_TDO_depth,
				verbose_level);

		if (f_v) {
			cout << "projective_space_object_classifier::do_the_work after latex_report" << endl;
		}


	} // f_report
	else {
		cout << "projective_space_object_classifier::do_the_work no report" << endl;
	}

	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work done" << endl;
	}


}


void projective_space_object_classifier::classify_objects_using_nauty(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx;
	int t0, t1, dt;
	//file_io Fio;
	os_interface Os;
	//string_tools ST;

	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty" << endl;
	}


	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"nb_objects_to_test = " << IS->nb_objects_to_test << endl;
	}


	t0 = Os.os_ticks();


	Ago = NEW_lint(IS->Objects.size());
	F_reject = NEW_int(IS->Objects.size());

	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty "
					"input_idx = " << input_idx << " / " << IS->Objects.size() << endl;
		}

		object_with_canonical_form *OwCF;

		OwCF = (object_with_canonical_form *) IS->Objects[input_idx];
		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty "
					"OwCF:" << endl;
			OwCF->print(cout);
		}


		if (f_projective_space) {
			OwCF->P = PA->P;
		}
		else {
			OwCF->P = NULL;
		}

		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty "
					"before process_any_object" << endl;
		}


		process_any_object(OwCF, input_idx, Ago[input_idx], F_reject[input_idx], verbose_level - 1);

		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty "
					"after process_any_object" << endl;
		}

	}

	nb_orbits = 0;
	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {
		if (!F_reject[input_idx]) {
			nb_orbits++;
		}
	}
	Idx_transversal = NEW_int(nb_orbits);
	Ago_transversal = NEW_lint(nb_orbits);
	int j;
	for (input_idx = 0, j = 0; input_idx < IS->Objects.size(); input_idx++) {
		if (F_reject[input_idx]) {
			continue;
		}
		Idx_transversal[j] = input_idx;
		Ago_transversal[j] = Ago[input_idx];
		j++;
	}
	if (j != nb_orbits) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty j != nb_orbits" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "input object : ago : f_reject" << endl;
		for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {
			cout << setw(3) << input_idx << " : " << setw(5) << Ago[input_idx] << " : " << F_reject[input_idx] << endl;
		}
	}

	if (f_v) {
		cout << "transversal of orbit representatives:" << endl;
		int cnt;
		cout << "iso type : input object : ago" << endl;
		for (input_idx = 0, cnt = 0; input_idx < IS->Objects.size(); input_idx++) {
			if (F_reject[input_idx]) {
				continue;
			}
			cout << setw(3) << cnt << " : " << setw(3) << input_idx << " : " << setw(5) << Ago[input_idx] << endl;
			cnt++;
		}
	}

	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"before CB->finalize" << endl;
	}

	CB->finalize(verbose_level); // computes C_type_of and perm


	T_Ago = NEW_OBJECT(tally);
	T_Ago->init_lint(Ago_transversal, nb_orbits, FALSE, 0);

	if (f_v) {
		cout << "Automorphism group orders of orbit transversal: ";
		T_Ago->print_first(TRUE /* f_backwards */);
		cout << endl;
	}



	if (Descr->f_save_ago) {
		if (f_v) {
			cout << "projective_space_object_classifier::process_multiple_objects_from_file "
					"f_save_ago is TRUE" << endl;
		}
		string ago_fname;
		file_io Fio;
		string_tools ST;

		if (Descr->f_label) {
			ago_fname.assign(Descr->label);
		}
		else {
			ago_fname.assign("classification");

		}
		ST.replace_extension_with(ago_fname, "_ago.csv");

		Fio.lint_vec_write_csv(Ago_transversal, nb_orbits, ago_fname, "Ago");
		if (f_v) {
			cout << "Written file " << ago_fname << " of size " << Fio.file_size(ago_fname) << endl;
		}

	}



	t1 = Os.os_ticks();
	dt = t1 - t0;

	cout << "projective_space_object_classifier::classify_objects_using_nauty Time ";
	Os.time_check_delta(cout, dt);
	cout << endl;



	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty done" << endl;
	}
}


void projective_space_object_classifier::process_any_object(object_with_canonical_form *OwCF,
		int input_idx, long int &ago, int &f_reject, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_any_object" << endl;
	}
	if (f_v) {
		cout << "projective_space_object_classifier::process_any_object "
				"input_idx=" << input_idx << " / " << IS->nb_objects_to_test << endl;
	}
	strong_generators *SG;

	if (f_v) {
		cout << "projective_space_object_classifier::process_any_object "
				"before process_object" << endl;
	}
	int nb_rows, nb_cols;
	int idx;

	OwCF->encoding_size(
			nb_rows, nb_cols,
			verbose_level);

	nauty_output *NO;

	NO = NEW_OBJECT(nauty_output);
	NO->allocate(nb_rows + nb_cols, verbose_level);


	f_reject = process_object(OwCF,
			SG, ago,
			idx,
			NO,
			verbose_level);


	if (f_v) {
		cout << "projective_space_object_classifier::process_any_object "
				"after process_object, f_reject=" << f_reject << endl;
	}


	if (f_reject) {
		//cout << "before FREE_OBJECT(SG)" << endl;
		FREE_OBJECT(SG);
	}
	else {
		if (f_v) {
			cout << "projective_space_object_classifier::process_any_object "
					"New isomorphism type! The current number of "
				"isomorphism types is " << CB->nb_types << endl;
		}
		int idx;

		if (f_projective_space) {
			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OwCF,
					SG->group_order_as_lint(), SG,
					nb_rows, nb_cols,
					NO->canonical_labeling, verbose_level);
			//FREE_OBJECT(SG);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			NO->canonical_labeling = NULL;
		}
		else {
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = SG;

		}

		//compute_and_print_ago_distribution(cout,
		//		CB, verbose_level);
	}

	FREE_OBJECT(NO);

	if (f_v) {
		cout << "projective_space_object_classifier::process_any_object done" << endl;
	}
}


int projective_space_object_classifier::process_object(
	object_with_canonical_form *OwCF,
	strong_generators *&SG, long int &ago,
	int &idx,
	nauty_output *NO,
	int verbose_level)
// returns f_found, which is TRUE if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"n=" << CB->n << endl;
	}

	longinteger_object go;

	bitvector *Canonical_form;




	if (f_projective_space) {
		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"before Nau.set_stabilizer_of_object" << endl;
		}

		nauty_interface_with_group Nau;

		SG = Nau.set_stabilizer_of_object(
				OwCF,
			PA->A,
			TRUE /* f_compute_canonical_form */, Canonical_form,
			NO,
			verbose_level - 2);
		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"after Nau.set_stabilizer_of_object" << endl;
		}


		SG->group_order(go);


	}
	else {
		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"not in projective space" << endl;
		}

		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"before OiP->run_nauty" << endl;
		}
		OwCF->run_nauty(
				TRUE /* f_compute_canonical_form */, Canonical_form,
				NO,
				verbose_level);


		nauty_interface_with_group Nau;
		action *A_perm;

		Nau.automorphism_group_as_permutation_group(
						SG,
						NO,
						A_perm,
						verbose_level);

		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"after OiP->run_nauty" << endl;

			A_perm->Strong_gens->print_generators_in_latex_individually(cout);
			A_perm->Strong_gens->print_generators_in_source_code();
			A_perm->print_base();
		}

	}



	NO->Ago->assign_to(go);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"go = " << go << endl;

		NO->print_stats();


	}



	ago = go.as_lint();

	if (CB->n == 0) {
		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"before CB->init" << endl;
		}
		CB->init(IS->nb_objects_to_test, Canonical_form->get_allocated_length(), verbose_level);
		if (f_v) {
			cout << "projective_space_object_classifier::process_object "
					"after CB->init" << endl;
		}
	}
	int f_found;

	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"before CB->search_and_add_if_new" << endl;
	}
	CB->search_and_add_if_new(Canonical_form->get_data(),
			NULL /* extra_data */,
			f_found, idx,
			verbose_level);
	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"after CB->search_and_add_if_new" << endl;
	}


	//delete SG;

	if (f_v) {
		cout << "projective_space_object_classifier::process_object done" << endl;
	}
	return f_found;
}

int projective_space_object_classifier::process_object_with_known_canonical_labeling(
	object_with_canonical_form *OiP,
	int *canonical_labeling, int canonical_labeling_len,
	int &idx,
	nauty_output *NO,
	int verbose_level)
// returns f_found, which is TRUE if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"n=" << CB->n << endl;
	}

	//longinteger_object go;
	//int *Extra_data;


	//uchar *canonical_form;
	//int canonical_form_len;


#if 0
	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"before PA->set_stabilizer_of_object" << endl;
	}


	SG = PA->set_stabilizer_of_object(
		OiP,
		Descr->f_save_incma_in_and_out, Descr->save_incma_in_and_out_prefix,
		TRUE /* f_compute_canonical_form */,
		canonical_form, canonical_form_len,
		canonical_labeling, canonical_labeling_len,
		verbose_level - 2);


	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"after PA->set_stabilizer_of_object" << endl;
	}
#else

	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"before OiP->canonical_form_given_canonical_labeling" << endl;
	}

	bitvector *Bitvec_canonical_form;

	OiP->canonical_form_given_canonical_labeling(
				canonical_labeling,
				Bitvec_canonical_form,
				verbose_level);


	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"after OiP->canonical_form_given_canonical_labeling" << endl;
	}
#endif

	int i;

	for (i = 0; i < NO->N; i++) {
		NO->canonical_labeling[i] = canonical_labeling[i];
	}


	//SG->group_order(go);



	if (CB->n == 0) {
		CB->init(IS->nb_objects_to_test,
				Bitvec_canonical_form->get_allocated_length() /*canonical_form_len*/,
				verbose_level);
	}
	int f_found;

	CB->search_and_add_if_new(Bitvec_canonical_form->get_data(),
			OiP, f_found, idx, verbose_level);


	//delete SG;

	FREE_OBJECT(Bitvec_canonical_form);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling done" << endl;
	}
	return f_found;
}

void projective_space_object_classifier::save(
		std::string &output_prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_object_classifier::save" << endl;
	}
	fname.assign(output_prefix);
	fname.append("_classified.cvs");

	{
		ofstream fp(fname);
		int i, j;

		fp << "rep,ago,original_file,input_idx,input_set,"
				"nb_rows,nb_cols,canonical_form" << endl;
		for (i = 0; i < CB->nb_types; i++) {

			object_in_projective_space_with_action *OiPA;
			object_with_canonical_form *OwCF;

			//cout << i << " / " << CB->nb_types << " is "
			//	<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
			OiPA = (object_in_projective_space_with_action *) CB->Type_extra_data[i];
			OwCF = OiPA->OwCF;
			if (OwCF == NULL) {
				cout << "OiP == NULL" << endl;
				exit(1);
			}
			if (OwCF->type != t_PAC) {
				//OiP->print(cout);
				}
			//OiP->print(cout);

	#if 0
			for (j = 0; j < rep_len; j++) {
				cout << (int) Type_data[i][j];
				if (j < rep_len - 1) {
					cout << ", ";
					}
				}
	#endif
			//cout << "before writing OiP->set_as_string:" << endl;

			int ago;

			if (OwCF->f_has_known_ago) {
				ago = OwCF->known_ago;
			}
			else {
				ago = 0; //OiPA->Aut_gens->group_order_as_lint();
			}
			fp << i << "," << ago
					<< "," << OwCF->input_fname
					<< "," << OwCF->input_idx
					<< ",\"" << OwCF->set_as_string << "\",";
			//cout << "before writing OiPA->nb_rows:" << endl;
			fp << OiPA->nb_rows << "," << OiPA->nb_cols<< ",";

			//cout << "before writing canonical labeling:" << endl;
			fp << "\"";
			for (j = 0; j < OiPA->nb_rows + OiPA->nb_cols; j++) {
				fp << OiPA->canonical_labeling[j];
				if (j < OiPA->nb_rows + OiPA->nb_cols - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			fp << endl;
			}
		fp << "END" << endl;
	}
	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "projective_space_object_classifier::save done" << endl;
	}
}


void projective_space_object_classifier::latex_report(
		std::string &fname,
		std::string &prefix,
		int fixed_structure_order_list_sz,
		int *fixed_structure_order_list,
		int max_TDO_depth,
		int verbose_level)
{
	int i, j;
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	file_io Fio;
	latex_interface L;

	if (f_v) {
		cout << "projective_space_object_classifier::latex_report" << endl;
	}
	if (f_v) {
		cout << "projective_space_object_classifier::latex_report, CB->nb_types=" << CB->nb_types << endl;
	}
	{
		ofstream fp(fname);
		latex_interface L;

		L.head_easy(fp);

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
		Orbiter->Int_vec.zero(Table, CB->nb_types * width);

		row_labels = NEW_int(CB->nb_types);
		col_labels = NEW_int(width);
		for (i = 0; i < CB->nb_types; i++) {
			row_labels[i] = i;
		}
		for (j = 0; j < width; j++) {
			col_labels[j] = j;
		}

		for (i = 0; i < CB->nb_types; i++) {
			if (f_v) {
				cout << "projective_space_object_classifier::latex_report, i=" << i << endl;
			}
			//j = CB->perm[i];
			j = i;
			if (f_v) {
				cout << "projective_space_object_classifier::latex_report, i=" << i << " j=" << j << endl;
			}
			Table[i * width + 0] = CB->Type_rep[j];
			Table[i * width + 1] = CB->Type_mult[j];
			Table[i * width + 2] = 0; // group order
			Table[i * width + 3] = 0; // object list
		}

		if (f_v) {
			cout << "projective_space_object_classifier::latex_report before Summary of Orbits" << endl;
		}

		fp << "\\section{Summary of Orbits}" << endl;
		fp << "$$" << endl;
		L.int_matrix_print_with_labels_and_partition(fp,
				Table, CB->nb_types, 4,
			row_labels, col_labels,
			row_part_first, row_part_len, nb_row_parts,
			col_part_first, col_part_len, nb_col_parts,
			print_summary_table_entry,
			CB /*void *data*/,
			TRUE /* f_tex */);
		fp << "$$" << endl;

		if (f_v) {
			cout << "projective_space_object_classifier::latex_report after Summary of Orbits" << endl;
		}

		fp << "Ago :";
		T_Ago->print_file_tex(fp, FALSE /* f_backwards*/);
		fp << "\\\\" << endl;

		if (f_v) {
			cout << "projective_space_object_classifier::latex_report before loop" << endl;
		}


		for (i = 0; i < CB->nb_types; i++) {

			//j = CB->perm[i];
			j = i;
			object_in_projective_space_with_action *OiPA;
			object_with_canonical_form *OwCF;

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
				L.int_vec_print_as_matrix(cout, Input_objects,
						nb_input_objects, 10 /* width */,
						FALSE /* f_tex */);

				FREE_int(Input_objects);
			}

			OiPA = (object_in_projective_space_with_action *) CB->Type_extra_data[j];
			OwCF = OiPA->OwCF;

			if (f_v) {
				cout << "OwCF:" << endl;
				OwCF->print(cout);
			}
			if (OwCF->type != t_PAC) {
				OwCF->print(cout);
			}



			strong_generators *SG;
			longinteger_object go;


			bitvector *Canonical_form;

			int nb_r, nb_c;

			OwCF->encoding_size(
					nb_r, nb_c,
					verbose_level);

#if 1
			if (f_v) {
				cout << "projective_space_object_classifier::latex_report before Nau.set_stabilizer_of_object" << endl;
			}

			nauty_interface_with_group Nau;
			nauty_output *NO;

			NO = NEW_OBJECT(nauty_output);
			NO->allocate(nb_r + nb_c, verbose_level);

			SG = Nau.set_stabilizer_of_object(
					OwCF,
				PA->A,
				TRUE /* f_compute_canonical_form */, Canonical_form,
				NO,
				verbose_level - 2);

			if (f_v) {
				cout << "projective_space_object_classifier::latex_report after Nau.set_stabilizer_of_object" << endl;
			}

			FREE_OBJECT(NO);

			SG->group_order(go);
#endif

			fp << "\\section*{Isomorphism type " << i << " / " << CB->nb_types << "}" << endl;
			fp << "Isomorphism type " << i << " / " << CB->nb_types
				//<<  " stored at " << j
				<< " is original object "
				<< CB->Type_rep[j] << " and appears "
				<< CB->Type_mult[j] << " times: \\\\" << endl;
			//if (OiP->type != t_PAC) {

			OwCF->print_tex(fp);
				fp << endl;
				fp << "\\bigskip" << endl;
				fp << endl;
			//	}

			if (OwCF->type == t_PAC) {
				long int *Sets;
				int nb_sets;
				int set_size;
				action *A_on_spreads;
				schreier *Sch;

				OwCF->get_packing_as_set_system(Sets, nb_sets, set_size, verbose_level);


				A_on_spreads = PA->A_on_lines->create_induced_action_on_sets(nb_sets,
						set_size, Sets,
						verbose_level);


				Sch = SG->orbits_on_points_schreier(A_on_spreads, verbose_level);

				fp << "Orbits on spreads:\\\\" << endl;
				Sch->print_and_list_orbits_tex(fp);


				FREE_OBJECT(Sch);
				FREE_OBJECT(A_on_spreads);
				FREE_lint(Sets);
			}
			//int_vec_print(fp, OiP->set, OiP->sz);
			fp << "Group order " << go << "\\\\" << endl;

			//fp << "Stabilizer:" << endl;
			//SG->print_generators_tex(fp);

			{
				int *Input_objects;
				int nb_input_objects;
				CB->C_type_of->get_class_by_value(Input_objects,
						nb_input_objects, j, 0 /*verbose_level */);
				Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

				fp << "This isomorphism type appears " << nb_input_objects
						<< " times, namely for the following "
						<< nb_input_objects << " input objects: " << endl;
				if (nb_input_objects < 10) {
					fp << "$" << endl;
					L.int_set_print_tex(fp, Input_objects, nb_input_objects);
					fp << "$\\\\" << endl;
				}
				else {
					fp << "$$" << endl;
					L.int_vec_print_as_matrix(fp, Input_objects,
						nb_input_objects, 10 /* width */, TRUE /* f_tex */);
					fp << "$$" << endl;
				}

				FREE_int(Input_objects);
			}



#if 0
			if (OiP->type == t_PTS) {
				//long int *set;
				//int sz;

				OiP->print_tex(fp);


				cout << "printing generators in restricted action:" << endl;
				action *A_restricted;

				A_restricted = SG->A->restricted_action(OiP->set, OiP->sz,
						verbose_level);
				SG->print_with_given_action(
						fp, A_restricted);
				FREE_OBJECT(A_restricted);
			}
#endif


			fp << "Stabilizer:\\\\" << endl;
			SG->print_generators_tex(fp);


#if 0
			//fp << "Stabilizer, all elements:\\\\" << endl;
			//SG->print_elements_ost(fp);
			//SG->print_elements_with_special_orthogonal_action_ost(fp);

			{
				action *A_conj;
				sims *Base_group;

				Base_group = SG->create_sims(verbose_level);

				A_conj = PA->A->create_induced_action_by_conjugation(
					Base_group, FALSE /* f_ownership */,
					verbose_level);

				fp << "Generators in conjugation action on the group itself:\\\\" << endl;
				SG->print_with_given_action(fp, A_conj);

				fp << "Elements in conjugation action on the group itself:\\\\" << endl;
				SG->print_elements_with_given_action(fp, A_conj);

				string fname_gap;
				char str[1000];

				fname_gap.assign("class_");

				sprintf(str, "%d", i);

				fname_gap.append(str);
				fname_gap.append(".gap");

				SG->export_permutation_group_to_GAP(fname_gap.c_str(), verbose_level);
				schreier *Sch;

				Sch = SG->orbits_on_points_schreier(A_conj, verbose_level);

				fp << "Orbits on itself by conjugation:\\\\" << endl;
				Sch->print_and_list_orbits_tex(fp);


				FREE_OBJECT(Sch);
				FREE_OBJECT(A_conj);
				FREE_OBJECT(Base_group);
			}
#endif


			encoded_combinatorial_object *Enc;
			incidence_structure *Inc;
			partitionstack *Stack;


			OwCF->encode_incma_and_make_decomposition(
				Enc,
				Inc,
				Stack,
				verbose_level);
			FREE_OBJECT(Enc);
		#if 0
			cout << "set ";
			int_vec_print(cout, OiP->set, OiP->sz);
			cout << " go=" << go << endl;



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
			for (h = 0; h < max_TDO_depth; h++) {
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



			OwCF->klein(verbose_level);

#if 1
			sims *Stab;
			int *Elt;
			int nb_trials;
			int max_trials = 100;

			Stab = SG->create_sims(verbose_level);
			Elt = NEW_int(PA->A->elt_size_in_int);

			for (h = 0; h < fixed_structure_order_list_sz; h++) {
				if (Stab->find_element_of_given_order_int(Elt,
						fixed_structure_order_list[h], nb_trials, max_trials,
						verbose_level)) {
					fp << "We found an element of order "
							<< fixed_structure_order_list[h] << ", which is:" << endl;
					fp << "$$" << endl;
					PA->A->element_print_latex(Elt, fp);
					fp << "$$" << endl;
					PA->report_fixed_points_lines_and_planes(
						Elt, fp,
						verbose_level);
				}
				else {
					fp << "We could not find an element of order "
						<< fixed_structure_order_list[h] << "\\\\" << endl;
				}
			}

			FREE_int(Elt);
#endif

			FREE_OBJECT(SG);

			FREE_OBJECT(Stack);
			FREE_OBJECT(Inc);

		}


		L.foot(fp);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "projective_space_object_classifier::latex_report done" << endl;
	}
}



}}

