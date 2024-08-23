/*
 * classification_of_objects.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace canonical_form_classification {


classification_of_objects::classification_of_objects()
{

	Descr = NULL;

	f_projective_space = false;

	P = NULL;

	IS = NULL;

	CB = NULL;

	Ago = NULL;
	F_reject = NULL;

	nb_orbits = 0;
	Idx_transversal = NULL;
	Ago_transversal = NULL;
	OWCF_transversal = NULL;
	NO_transversal = NULL;

	T_Ago = NULL;

}

classification_of_objects::~classification_of_objects()
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
	if (OWCF_transversal) {
#if 0
		int i;

		for (i = 0; i < nb_orbits; i++) {
			FREE_OBJECT(OWCF[i]);
		}
#endif
		FREE_pvoid((void **) OWCF_transversal);
	}
	if (NO_transversal) {
		FREE_pvoid((void **) NO_transversal);
	}
}

std::string classification_of_objects::get_label()
{
	if (!IS) {
		cout << "classification_of_objects::get_label !IS" << endl;
		exit(1);
	}
	if (!IS->Descr->f_label) {
		cout << "classification_of_objects::get_label !IS->Descr->f_label" << endl;
		exit(1);
	}
	return IS->Descr->label_txt;
}

std::string classification_of_objects::get_label_tex()
{
	if (!IS) {
		cout << "classification_of_objects::get_label_tex !IS" << endl;
		exit(1);
	}
	if (!IS->Descr->f_label) {
		cout << "classification_of_objects::get_label_tex !IS->Descr->f_label" << endl;
		exit(1);
	}
	return IS->Descr->label_tex;
}

void classification_of_objects::perform_classification(
		classification_of_objects_description *Descr,
		int f_projective_space,
		geometry::projective_space *P,
		data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"f_projective_space=" << f_projective_space << endl;
	}

	classification_of_objects::f_projective_space = f_projective_space;
	classification_of_objects::P = P;
	classification_of_objects::Descr = Descr;
	classification_of_objects::IS = IS;

	//int i;

	CB = NEW_OBJECT(classify_bitvectors);





	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"before classify_objects_using_nauty" << endl;
	}

	classify_objects_using_nauty(verbose_level - 1);

	if (f_v) {
		cout << "classification_of_objects::perform_classification "
			"after classify_objects_using_nauty" << endl;
	}


	cout << "classification_of_objects::perform_classification We found "
			<< CB->nb_types << " types" << endl;






	if (f_v) {
		cout << "classification_of_objects::perform_classification done" << endl;
	}


}


void classification_of_objects::classify_objects_using_nauty(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty" << endl;
	}


	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"nb_objects_to_test = " << IS->nb_objects_to_test << endl;
	}


	t0 = Os.os_ticks();


	Ago = NEW_lint(IS->Objects.size());

	F_reject = NEW_int(IS->Objects.size());

	OWCF_transversal = (object_with_canonical_form **) NEW_pvoid(IS->Objects.size());

	NO_transversal = (l1_interfaces::nauty_output **) NEW_pvoid(IS->Objects.size());


	nb_orbits = 0;
	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		if (f_v && (input_idx % 1000) == 0) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"input_idx = " << input_idx << " / " << IS->Objects.size() << endl;
		}

		object_with_canonical_form *OwCF;

		OwCF = (object_with_canonical_form *) IS->Objects[input_idx];
		if (false) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"OwCF:" << endl;
			OwCF->print(cout);
		}


		string object_label;

		if (IS->Descr->f_label) {
			object_label = IS->Descr->label_txt + "_" + std::to_string(input_idx);
		}
		else {
			object_label = "object_" + std::to_string(input_idx);

		}




		OwCF->set_label(object_label);



		if (false) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"before process_any_object" << endl;
		}


		l1_interfaces::nauty_output *NO;
		encoded_combinatorial_object *Enc;

		process_any_object(
					OwCF,
					input_idx,
					Ago[input_idx],
					F_reject[input_idx],
					NO,
					Enc,
					verbose_level - 1);


		FREE_OBJECT(Enc);

		if (false) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"after process_any_object" << endl;
		}

		if (!F_reject[input_idx]) {
			OWCF_transversal[nb_orbits] =
					(object_with_canonical_form *) IS->Objects[input_idx];
			NO_transversal[nb_orbits] = NO;
			nb_orbits++;
		}

	}
	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"nb_orbits = " << nb_orbits << endl;
	}

	Idx_transversal = NEW_int(nb_orbits);
	Ago_transversal = NEW_lint(nb_orbits);

	int iso_idx;

	for (input_idx = 0, iso_idx = 0;

			input_idx < IS->Objects.size();

			input_idx++) {

		if (F_reject[input_idx]) {
			continue;
		}

		Idx_transversal[iso_idx] = input_idx;
		Ago_transversal[iso_idx] = Ago[input_idx];
		iso_idx++;
	}
	if (iso_idx != nb_orbits) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"iso_idx != nb_orbits" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "input object : ago : f_reject" << endl;
		for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {
			cout << setw(3) << input_idx << " : " << setw(5)
					<< Ago[input_idx] << " : " << F_reject[input_idx] << endl;
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
			cout << setw(3) << cnt << " : " << setw(3) << input_idx
					<< " : " << setw(5) << Ago[input_idx] << endl;
			cnt++;
		}
	}

	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"before CB->finalize" << endl;
	}

	CB->finalize(verbose_level); // computes C_type_of and perm


	T_Ago = NEW_OBJECT(data_structures::tally);
	T_Ago->init_lint(Ago_transversal, nb_orbits, false, 0);

	if (f_v) {
		cout << "Automorphism group orders of orbit transversal: ";
		T_Ago->print_first(true /* f_backwards */);
		cout << endl;
	}



	if (Descr->f_save_ago) {
		if (f_v) {
			cout << "classification_of_objects::process_multiple_objects_from_file "
					"f_save_ago is true" << endl;
		}

		save_automorphism_group_order(verbose_level);

	}

	if (Descr->f_save_transversal) {
		if (f_v) {
			cout << "classification_of_objects::process_multiple_objects_from_file "
					"f_save_transversal is true" << endl;
		}

		save_transversal(verbose_level);

	}



	t1 = Os.os_ticks();
	dt = t1 - t0;

	cout << "classification_of_objects::classify_objects_using_nauty Time ";
	Os.time_check_delta(cout, dt);
	cout << endl;



	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty done" << endl;
	}
}

void classification_of_objects::save_automorphism_group_order(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::save_automorphism_group_order " << endl;
	}
	string ago_fname;
	orbiter_kernel_system::file_io Fio;
	data_structures::string_tools ST;

	ago_fname = get_label();
	ST.replace_extension_with(ago_fname, "_ago.csv");

	string label;

	label.assign("Ago");
	Fio.Csv_file_support->lint_vec_write_csv(
			Ago_transversal, nb_orbits, ago_fname, label);
	if (f_v) {
		cout << "Written file " << ago_fname
				<< " of size " << Fio.file_size(ago_fname) << endl;
	}
	if (f_v) {
		cout << "classification_of_objects::save_automorphism_group_order done" << endl;
	}
}

void classification_of_objects::save_transversal(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::save_transversal " << endl;
	}
	string fname;
	orbiter_kernel_system::file_io Fio;
	data_structures::string_tools ST;

	fname = get_label();

	ST.replace_extension_with(fname, "_transversal.csv");
	string label;

	label.assign("Transversal");

	Fio.Csv_file_support->int_vec_write_csv(
			Idx_transversal, nb_orbits, fname, label);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "classification_of_objects::save_transversal done" << endl;
	}
}

void classification_of_objects::process_any_object(
		object_with_canonical_form *OwCF,
		int input_idx, long int &ago, int &f_reject,
		l1_interfaces::nauty_output *&NO,
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::process_any_object, "
				"verbose_level = " << verbose_level << endl;
	}
	if (f_v) {
		cout << "classification_of_objects::process_any_object "
				"input_idx=" << input_idx << " / " << IS->nb_objects_to_test << endl;
	}

	if (f_v) {
		cout << "classification_of_objects::process_any_object "
				"before process_object" << endl;
	}
	int iso_idx_if_found;

	f_reject = process_object(
			OwCF,
			ago,
			iso_idx_if_found,
			NO,
			Enc,
			verbose_level - 1);


	if (f_v) {
		cout << "classification_of_objects::process_any_object "
				"after process_object, f_reject=" << f_reject << endl;
	}


	if (f_reject) {

		//cout << "before FREE_OBJECT(SG)" << endl;

#if 0
		if (f_projective_space) {
			FREE_OBJECT(SG);
		}
		else {
			FREE_OBJECT(A_perm);
		}
#endif

		FREE_OBJECT(NO); // ToDo ???
	}
	else {
		if (f_v) {
			cout << "classification_of_objects::process_any_object "
					"New isomorphism type! The current number of "
				"isomorphism types is " << CB->nb_types << endl;
		}
		//int idx;

		//idx = CB->type_of[CB->n - 1];


#if 0
		if (f_projective_space) {
			object_in_projective_space_with_action *OiPA;


			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OwCF,
					SG->group_order_as_lint(), SG,
					//nb_rows, nb_cols,
					NO->canonical_labeling,
					verbose_level);

			//FREE_OBJECT(SG);

			CB->Type_extra_data[idx] = OiPA;

			NO->canonical_labeling = NULL;
		}
		else {
			CB->Type_extra_data[idx] = A_perm;

		}
#endif


		//compute_and_print_ago_distribution(cout,
		//		CB, verbose_level);
	}


	if (f_v) {
		cout << "classification_of_objects::process_any_object done" << endl;
	}
}


int classification_of_objects::process_object(
		object_with_canonical_form *OwCF,
	long int &ago,
	int &iso_idx_if_found,
	l1_interfaces::nauty_output *&NO,
	encoded_combinatorial_object *&Enc,
	int verbose_level)
// returns f_found, which is true if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"n=" << CB->n << endl;
	}


	data_structures::bitvector *Canonical_form;


	if (f_projective_space) {
		OwCF->P = P;
	}
	else {
		OwCF->P = NULL;
	}


	if (f_v) {
		cout << "classification_of_objects::process_object "
				"before OwCF->run_nauty" << endl;
	}

	OwCF->run_nauty(
			true /* f_compute_canonical_form */,
			Descr->f_save_nauty_input_graphs,
			Canonical_form,
			NO,
			Enc,
			verbose_level);

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"after OwCF->run_nauty" << endl;
	}


	ring_theory::longinteger_object go;

	NO->Ago->assign_to(go);

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"go = " << go << endl;

		NO->print_stats();


	}



	ago = go.as_lint();

	if (CB->n == 0) {
		if (f_v) {
			cout << "classification_of_objects::process_object "
					"before CB->init" << endl;
		}
		CB->init(IS->nb_objects_to_test,

				Canonical_form->get_allocated_length(),

				verbose_level);
		if (f_v) {
			cout << "classification_of_objects::process_object "
					"after CB->init" << endl;
		}
	}
	int f_found;

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"before CB->search_and_add_if_new" << endl;
	}

	CB->search_and_add_if_new(

			Canonical_form->get_data(),

			NULL /* extra_data */,

			f_found, iso_idx_if_found,

			verbose_level);

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"after CB->search_and_add_if_new" << endl;
	}


	//delete SG;

	if (f_v) {
		cout << "classification_of_objects::process_object done" << endl;
	}
	return f_found;
}


#if 0
int classification_of_objects::process_object_with_known_canonical_labeling(
	object_with_canonical_form *OiP,
	int *canonical_labeling, int canonical_labeling_len,
	int &idx,
	nauty_output *NO,
	int verbose_level)
// returns f_found, which is true if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::process_object_with_known_canonical_labeling "
				"n=" << CB->n << endl;
	}

	//longinteger_object go;
	//int *Extra_data;


	//uchar *canonical_form;
	//int canonical_form_len;


#if 0
	if (f_v) {
		cout << "classification_of_objects::process_object_with_known_canonical_labeling "
				"before PA->set_stabilizer_of_object" << endl;
	}


	SG = PA->set_stabilizer_of_object(
		OiP,
		Descr->f_save_incma_in_and_out, Descr->save_incma_in_and_out_prefix,
		true /* f_compute_canonical_form */,
		canonical_form, canonical_form_len,
		canonical_labeling, canonical_labeling_len,
		verbose_level - 2);


	if (f_v) {
		cout << "classification_of_objects::process_object_with_known_canonical_labeling "
				"after PA->set_stabilizer_of_object" << endl;
	}
#else

	if (f_v) {
		cout << "classification_of_objects::process_object_with_known_canonical_labeling "
				"before OiP->canonical_form_given_canonical_labeling" << endl;
	}

	bitvector *Bitvec_canonical_form;

	OiP->canonical_form_given_canonical_labeling(
				canonical_labeling,
				Bitvec_canonical_form,
				verbose_level);


	if (f_v) {
		cout << "classification_of_objects::process_object_with_known_canonical_labeling "
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
		cout << "classification_of_objects::process_object_with_known_canonical_labeling done" << endl;
	}
	return f_found;
}
#endif

#if 0
void classification_of_objects::save(
		std::string &output_prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	file_io Fio;

	if (f_v) {
		cout << "classification_of_objects::save" << endl;
	}
	fname = output_prefix + "_classified.cvs";


#if 0
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

#endif

	if (f_v) {
		cout << "classification_of_objects::save done" << endl;
	}
}
#endif



void classification_of_objects::report_summary_of_orbits(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::report_summary_of_orbits" << endl;
	}
	l1_interfaces::latex_interface L;


	if (f_v) {
		cout << "classification_of_objects::latex_report "
				"before Summary of Orbits" << endl;
	}

	ost << "\\section*{Summary of Orbits}" << endl;

	std::string *Table;
	int nb_rows, nb_cols;

	create_summary_table(
			Table,
			nb_rows, nb_cols,
			verbose_level);


	std::string *headers;

	headers = new string[nb_cols];
	headers[0] = "Iso";
	headers[1] = "Rep";
	headers[2] = "\\#";
	headers[3] = "Ago";
	headers[4] = "Objects";

	ost << "$$" << endl;
	L.print_table_of_strings_with_headers(
			ost, headers, Table, nb_rows, nb_cols);
	ost << "$$" << endl;

#if 0
	ost << "$$" << endl;
	L.int_matrix_print_with_labels_and_partition(ost,
			Table, CB->nb_types, 4,
		row_labels, col_labels,
		row_part_first, row_part_len, nb_row_parts,
		col_part_first, col_part_len, nb_col_parts,
		print_summary_table_entry,
		this /*void *data*/,
		true /* f_tex */);
	ost << "$$" << endl;
#endif

	if (f_v) {
		cout << "classification_of_objects::latex_report "
				"after Summary of Orbits" << endl;
	}

	delete [] Table;

	//FREE_int(Table);

	if (f_v) {
		cout << "classification_of_objects::report_summary_of_orbits done" << endl;
	}

}


void classification_of_objects::create_summary_table(
		std::string *&Table,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::create_summary_table" << endl;
	}
	data_structures::sorting Sorting;

	nb_rows = nb_orbits;
	nb_cols = 5;

	Table = new string[nb_rows * nb_cols];

	int i;

	for (i = 0; i < nb_rows; i++) {

		int j = CB->perm[i];

		string s_idx;
		string s_orbit_rep;
		string s_mult;
		string s_ago;
		string s_input_objects;


		s_idx = std::to_string(i);
		s_orbit_rep = std::to_string(CB->Type_rep[j]);
		s_mult = std::to_string(CB->Type_mult[j]);



		ring_theory::longinteger_object go;
		go.create(Ago_transversal[i]);
		//OiPA->Aut_gens->group_order(go);
		//go.print_to_string(output);

		s_ago = go.stringify();


		int *Input_objects;
		int nb_input_objects;
		if (f_v) {
			cout << "classification_of_objects::create_summary_table "
					"before CB->C_type_of->get_class_by_value" << endl;
		}
		CB->C_type_of->get_class_by_value(
				Input_objects,
			nb_input_objects, j,
			0 /*verbose_level */);
		if (f_v) {
			cout << "classification_of_objects::create_summary_table "
					"after CB->C_type_of->get_class_by_value" << endl;
		}
		Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

		s_input_objects = Int_vec_stringify(Input_objects, nb_input_objects);


		FREE_int(Input_objects);


		Table[i * nb_cols + 0] = s_idx;
		Table[i * nb_cols + 1] = s_orbit_rep;
		Table[i * nb_cols + 2] = s_mult;
		Table[i * nb_cols + 3] = s_ago;
		Table[i * nb_cols + 4] = s_input_objects;

	}

}




}}}


