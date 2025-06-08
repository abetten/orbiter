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
namespace combinatorics {
namespace canonical_form_classification {


classification_of_objects::classification_of_objects()
{
	Record_birth();

	Descr = NULL;

	f_projective_space = false;

	P = NULL;

	IS = NULL;

	Output = NULL;

#if 0
	CB = NULL;

	Ago = NULL;
	F_reject = NULL;

	nb_orbits = 0;
	Idx_transversal = NULL;
	Ago_transversal = NULL;
	OWCF_transversal = NULL;
	NO_transversal = NULL;

	T_Ago = NULL;
#endif

}

classification_of_objects::~classification_of_objects()
{
	Record_death();

	if (Output) {
		FREE_OBJECT(Output);
	}

}


void classification_of_objects::perform_classification(
		classification_of_objects_description *Descr,
		int f_projective_space,
		geometry::projective_geometry::projective_space *P,
		data_input_stream *IS,
		int verbose_level)
// called from
// layer5_applications::apps_combinatorics::combinatorial_object_stream::do_canonical_form
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

	//CB = NEW_OBJECT(classify_bitvectors);

	Output = NEW_OBJECT(data_input_stream_output);

	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"before Output->init" << endl;
	}
	Output->init(this, verbose_level);
	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"after Output->init" << endl;
	}



	if (Descr->f_nauty_control) {
		cout << "classification_of_objects::perform_classification nauty_control: " << endl;
		Descr->Nauty_control->print();
	}



	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"before classify_objects_using_nauty" << endl;
	}

	classify_objects_using_nauty(verbose_level - 1);

	if (f_v) {
		cout << "classification_of_objects::perform_classification "
			"after classify_objects_using_nauty" << endl;
	}


	if (f_v) {
		cout << "classification_of_objects::perform_classification We found "
				<< Output->CB->nb_types << " types" << endl;
	}


	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"before Output->after_classification" << endl;
	}


	Output->after_classification(
			verbose_level);

	if (f_v) {
		cout << "classification_of_objects::perform_classification "
				"after Output->after_classification" << endl;
	}


	if (f_v) {
		cout << "classification_of_objects::perform_classification done" << endl;
	}


}


void classification_of_objects::classify_objects_using_nauty(
	int verbose_level)
// assumes that IS->Objects is available.
// This is an array of pointers to objects of type any_combinatorial_object,
// disguised as void pointers
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty" << endl;
	}
	int f_vv = false;

	int input_idx;
	int t0, t1, dt;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"nb_objects_to_test = " << IS->nb_objects_to_test << endl;
	}


	t0 = Os.os_ticks();

#if 0
	Ago = NEW_lint(IS->Objects.size());

	F_reject = NEW_int(IS->Objects.size());

	OWCF_transversal = (any_combinatorial_object **) NEW_pvoid(IS->Objects.size());

	NO_transversal = (other::l1_interfaces::nauty_output **) NEW_pvoid(IS->Objects.size());
#endif

	Output->nb_orbits = 0;

	for (input_idx = 0; input_idx < Output->nb_input; input_idx++) {

		if (f_v && (input_idx % 1000) == 0) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"input_idx = " << input_idx << " / " << Output->nb_input << endl;
		}

		any_combinatorial_object *OwCF;

		OwCF = (any_combinatorial_object *) IS->Objects[input_idx];
		if (f_vv) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"OwCF:" << endl;
			OwCF->print_brief(cout);
		}


		string object_label;

		if (IS->Descr->f_label) {
			object_label = IS->Descr->label_txt + "_" + std::to_string(input_idx);
		}
		else {
			object_label = "object_" + std::to_string(input_idx);

		}


		if (f_vv) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"object_label = " << object_label << endl;
		}


		OwCF->set_label(object_label);

		//other::l1_interfaces::nauty_output *NO;
		encoded_combinatorial_object *Enc;



		if (f_vv) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"before process_any_object" << endl;
		}

		Output->OWCF[input_idx] = (any_combinatorial_object *) IS->Objects[input_idx];

		process_any_object(
					Output->OWCF[input_idx],
					input_idx,
					Output->Ago[input_idx],
					Output->F_reject[input_idx],
					Output->NO[input_idx],
					Enc,
					verbose_level - 2);

		if (f_vv) {
			cout << "classification_of_objects::classify_objects_using_nauty "
					"after process_any_object" << endl;
		}

		FREE_OBJECT(Enc);

		//Output->NO[input_idx] = NO;

		if (!Output->F_reject[input_idx]) {
			//Output->OWCF_transversal[Output->nb_orbits] = (any_combinatorial_object *) IS->Objects[input_idx];
			Output->nb_orbits++;
		}

	}
	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"nb_orbits = " << Output->nb_orbits << endl;
	}


#if 0
	Output->Idx_transversal = NEW_int(Output->nb_orbits);
	Output->Ago_transversal = NEW_lint(Output->nb_orbits);

	int iso_idx;

	for (input_idx = 0, iso_idx = 0;

			input_idx < IS->Objects.size();

			input_idx++) {

		if (Output->F_reject[input_idx]) {
			continue;
		}

		Output->Idx_transversal[iso_idx] = input_idx;
		Output->Ago_transversal[iso_idx] = Output->Ago[input_idx];
		iso_idx++;
	}
	if (iso_idx != Output->nb_orbits) {
		cout << "classification_of_objects::classify_objects_using_nauty "
				"iso_idx != nb_orbits" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "input object : ago : f_reject" << endl;
		for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {
			cout << setw(3) << input_idx << " : " << setw(5)
					<< Output->Ago[input_idx] << " : " << Output->F_reject[input_idx] << endl;
		}
	}

	if (f_v) {
		cout << "transversal of orbit representatives:" << endl;
		int cnt;
		cout << "iso type : input object : ago" << endl;
		for (input_idx = 0, cnt = 0; input_idx < IS->Objects.size(); input_idx++) {
			if (Output->F_reject[input_idx]) {
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

	Output->CB->finalize(verbose_level); // computes C_type_of and perm


	Output->T_Ago = NEW_OBJECT(other::data_structures::tally);
	Output->T_Ago->init_lint(Output->Ago_transversal, Output->nb_orbits, false, 0);

	if (f_v) {
		cout << "Automorphism group orders of orbit transversal: ";
		Output->T_Ago->print_first(true /* f_backwards */);
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
#endif



	t1 = Os.os_ticks();
	dt = t1 - t0;

	cout << "classification_of_objects::classify_objects_using_nauty Time ";
	Os.time_check_delta(cout, dt);
	cout << endl;



	if (f_v) {
		cout << "classification_of_objects::classify_objects_using_nauty done" << endl;
	}
}

void classification_of_objects::process_any_object(
		any_combinatorial_object *OwCF,
		int input_idx, long int &ago, int &f_reject,
		other::l1_interfaces::nauty_output *&NO,
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

		//FREE_OBJECT(NO); // we keep the NO in all cases
	}
	else {
		if (f_v) {
			cout << "classification_of_objects::process_any_object "
					"New isomorphism type! The current number of "
				"isomorphism types is " << Output->CB->nb_types << endl;
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
		any_combinatorial_object *OwCF,
	long int &ago,
	int &iso_idx_if_found,
	other::l1_interfaces::nauty_output *&NO,
	encoded_combinatorial_object *&Enc,
	int verbose_level)
// returns f_found, which is true if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"n=" << Output->CB->n << endl;
	}


	other::data_structures::bitvector *Canonical_form;


	if (f_projective_space) {
		OwCF->P = P;
	}
	else {
		OwCF->P = NULL;
	}


	other::l1_interfaces::nauty_interface_for_combo NI;


	if (!Descr->f_nauty_control) {
		cout << "classification_of_objects::process_object "
				"please use option -nauty_control" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "classification_of_objects::process_object "
				"before NI.run_nauty_for_combo" << endl;
	}


	NI.run_nauty_for_combo(
			OwCF,
			true /* f_compute_canonical_form */,
			Descr->Nauty_control,
			Canonical_form,
			NO,
			Enc,
			verbose_level);


	if (f_v) {
		cout << "classification_of_objects::process_object "
				"after NI.run_nauty_for_combo" << endl;
	}


	algebra::ring_theory::longinteger_object go;

	NO->Ago->assign_to(go);

	if (f_v) {
		cout << "classification_of_objects::process_object "
				"go = " << go << endl;

		NO->print_stats();


	}



	ago = go.as_lint();

	if (Output->CB->n == 0) {
		if (f_v) {
			cout << "classification_of_objects::process_object "
					"before CB->init" << endl;
		}
		Output->CB->init(IS->nb_objects_to_test,

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

	Output->CB->search_and_add_if_new(

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




}}}}



