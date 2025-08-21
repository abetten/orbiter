/*
 * classification_of_varieties_nauty.cpp
 *
 *  Created on: Jul 15, 2024
 *      Author: betten
 */








#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



classification_of_varieties_nauty::classification_of_varieties_nauty()
{
	Record_birth();
	Classifier = NULL;

	Input = NULL;

	//nb_objects_to_test = 0;
	//Input_Vo = NULL;

	//std::string fname_base;

	// nauty:
	CB = NULL;
	canonical_labeling_len = 0;
	//alpha = NULL;
	//gamma = NULL;
	F_first_time = NULL;
	Iso_idx = NULL;
	Idx_canonical_form = NULL;
	Idx_equation = NULL;
	nb_isomorphism_classes = 0;
	Orbit_input_idx = NULL;

	//Classification_table_nauty = NULL;


	// canonical forms
	Canonical_forms = NULL;
	//std::vector<std::string> Lines;

	Goi = NULL;
	Phi = NULL;


	// auxiliary data:
	nb_times_memory_footprint_reduction = 0;
	Elt_gamma = NULL;
	Elt_gamma_inv = NULL;
	Elt_delta = NULL;
	Elt_phi = NULL;
	eqn2 = NULL;



}

classification_of_varieties_nauty::~classification_of_varieties_nauty()
{
	Record_death();
	if (F_first_time) {
		FREE_int(F_first_time);
	}
	if (Iso_idx) {
		FREE_int(Iso_idx);
	}
	if (Idx_canonical_form) {
		FREE_int(Idx_canonical_form);
	}
	if (Idx_equation) {
		FREE_int(Idx_equation);
	}
	if (Orbit_input_idx) {
		FREE_int(Orbit_input_idx);
	}

#if 0
	if (Classification_table_nauty) {
		FREE_int(Classification_table_nauty);
	}
#endif

	if (Goi) {
		FREE_lint(Goi);
	}
	if (Phi) {

		int i;

		for (i = 0; i < Input->nb_objects_to_test; i++) {
			if (Phi[i]) {
				FREE_int(Phi[i]);
				Phi[i] = NULL;
			}
		}
		FREE_pint(Phi);
		Phi = NULL;
	}


	if (Elt_gamma) {
		FREE_int(Elt_gamma);
	}
	if (Elt_gamma_inv) {
		FREE_int(Elt_gamma_inv);
	}
	if (Elt_delta) {
		FREE_int(Elt_delta);
	}
	if (Elt_phi) {
		FREE_int(Elt_phi);
	}

}

void classification_of_varieties_nauty::prepare_for_classification(
		input_objects_of_type_variety *Input,
		canonical_form_classifier *Classifier,
		int verbose_level)
// called from
// canonical_form_classifier::classify
// canonical_form_global::compute_group_and_tactical_decomposition
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification " << endl;
	}
	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification" << endl;
	}

	classification_of_varieties_nauty::Classifier = Classifier;

	classification_of_varieties_nauty::Input = Input;
	classification_of_varieties_nauty::fname_base = Input->fname_base_out;

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"nb_objects_to_test=" << Input->nb_objects_to_test << endl;
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"fname_base=" << fname_base << endl;
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"Input->nb_objects_to_test=" << Input->nb_objects_to_test << endl;
	}


	// allocate auxiliary data:
	Elt_gamma = NEW_int(Classifier->Ring_with_action->PA->A->elt_size_in_int);
	Elt_gamma_inv = NEW_int(Classifier->Ring_with_action->PA->A->elt_size_in_int);
	Elt_delta = NEW_int(Classifier->Ring_with_action->PA->A->elt_size_in_int);
	Elt_phi = NEW_int(Classifier->Ring_with_action->PA->A->elt_size_in_int);
	eqn2 = NEW_int(Classifier->Ring_with_action->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"before prepare_input" << endl;
	}
	prepare_input(verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"after prepare_input" << endl;
	}

#if 0
	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"before classify_nauty" << endl;
	}
	classify_nauty(verbose_level - 1);
	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"after classify_nauty" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"before write_classification_by_nauty_csv" << endl;
	}
	write_classification_by_nauty_csv(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification "
				"after write_classification_by_nauty_csv" << endl;
	}

#endif


	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_for_classification done" << endl;
	}

}

void classification_of_varieties_nauty::compute_classification(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification "
				"before classify_nauty" << endl;
	}
	classify_nauty(verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification "
				"after classify_nauty" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification "
				"nb_times_memory_footprint_reduction = " << nb_times_memory_footprint_reduction << endl;
	}


	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification "
				"before write_classification_by_nauty_csv" << endl;
	}
	write_classification_by_nauty_csv(
			fname_base,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification "
				"after write_classification_by_nauty_csv" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification "
				"nb_times_memory_footprint_reduction = " << nb_times_memory_footprint_reduction << endl;
	}


	if (f_v) {
		cout << "classification_of_varieties_nauty::compute_classification done" << endl;
	}
}

variety_compute_canonical_form *classification_of_varieties_nauty::get_canonical_form_i(
		int i)
{
	return Canonical_forms[i];
}


void classification_of_varieties_nauty::prepare_input(
		int verbose_level)
// initializes the entries of Canonical_forms[nb_inputs]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_input " << endl;
	}

	if (Input == NULL) {
		cout << "classification_of_varieties_nauty::prepare_input Input == NULL" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_input "
				"Input->nb_objects_to_test" << Input->nb_objects_to_test << endl;
	}


	int input_counter;

	Canonical_forms = (variety_compute_canonical_form **)
			NEW_pvoid(Input->nb_objects_to_test);


	for (input_counter = 0; input_counter < Input->nb_objects_to_test; input_counter++) {


		string fname_case_out;

		fname_case_out = Input->fname_base_out + "_cnt" + std::to_string(input_counter);

		variety_compute_canonical_form *Variety_compute_canonical_form;

		Variety_compute_canonical_form = NEW_OBJECT(variety_compute_canonical_form);

		if (f_v) {
			cout << "classification_of_varieties_nauty::prepare_input "
					"input_counter = " << input_counter << " / " << Classifier->Input->nb_objects_to_test
					<< " before Variety_compute_canonical_form->init" << endl;
		}
		Variety_compute_canonical_form->init(
				Classifier,
				Classifier->Ring_with_action,
				Classifier->Classification_of_varieties_nauty,
				fname_case_out,
				input_counter,
				Classifier->Input->Vo[input_counter],
				verbose_level - 2);



		if (f_v) {
			cout << "classification_of_varieties_nauty::prepare_input "
					"input_counter = " << input_counter << " / " << Classifier->Input->nb_objects_to_test
					<< " after Variety_compute_canonical_form->init" << endl;
		}

		Canonical_forms[input_counter] = Variety_compute_canonical_form;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_input done" << endl;
	}
}


void classification_of_varieties_nauty::classify_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty" << endl;
		cout << "classification_of_varieties_nauty::classify_nauty verbose_level = " << verbose_level << endl;
	}
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty" << endl;
		cout << "classification_of_varieties_nauty::classify_nauty "
				"nb_objects_to_test = " << Input->nb_objects_to_test << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"before allocate_tables" << endl;
	}
	allocate_tables(verbose_level - 10);
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"after allocate_tables" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"before main_loop" << endl;
	}
	main_loop(
			verbose_level - 1);
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"after main_loop" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"The number of graph theoretic canonical forms is "
				<< CB->nb_types << endl;
	}


}

void classification_of_varieties_nauty::allocate_tables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classification_of_varieties_nauty::allocate_tables" << endl;
	}

	CB = NEW_OBJECT(combinatorics::canonical_form_classification::classify_bitvectors);


	// classification by nauty:

	F_first_time = NEW_int(Input->nb_objects_to_test);
	Iso_idx = NEW_int(Input->nb_objects_to_test);
	Idx_canonical_form = NEW_int(Input->nb_objects_to_test);
	Idx_equation = NEW_int(Input->nb_objects_to_test);
	Orbit_input_idx = NEW_int(Input->nb_objects_to_test);
	nb_isomorphism_classes = 0;

	Goi = NEW_lint(Input->nb_objects_to_test);
	Phi = NEW_pint(Input->nb_objects_to_test);
	int i;
	for (i = 0; i < Input->nb_objects_to_test; i++) {
		Phi[i] = NULL;
	}


	if (f_v) {
		cout << "classification_of_varieties_nauty::allocate_tables done" << endl;
	}
}


void classification_of_varieties_nauty::main_loop(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "classification_of_varieties_nauty::main_loop" << endl;
		cout << "classification_of_varieties_nauty::main_loop "
				"verbose_level = " << verbose_level << endl;
		cout << "classification_of_varieties_nauty::main_loop "
				"nb_objects_to_test = " << Input->nb_objects_to_test << endl;
	}


	int input_counter;
	int nb_iso = 0;

	for (input_counter = 0; input_counter < Input->nb_objects_to_test; input_counter++) {


		variety_compute_canonical_form *Variety_compute_canonical_form;

		Variety_compute_canonical_form = Canonical_forms[input_counter];


		if (f_v) {
			cout << "classification_of_varieties_nauty::main_loop "
					"input_counter = " << input_counter << " / " << Input->nb_objects_to_test << endl;
		}


		if (Classifier->skip_this_one(input_counter)) {
			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"skipping case input_counter = " << input_counter << endl;
			}
			//Variety_table[input_counter] = NULL;
			Iso_idx[input_counter] = -1;
			F_first_time[input_counter] = false;
			Goi[input_counter] = -1;
			Phi[input_counter] = NULL;
			//continue;
		}
		else {


			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"before handle_one_input_case" << endl;
			}
			handle_one_input_case(
					input_counter, nb_iso,
					Variety_compute_canonical_form,
					verbose_level - 1);
			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"after handle_one_input_case" << endl;
			}


			if (Classifier->f_nauty_control) {
				if (Classifier->Nauty_interface_control->f_reduce_memory_footprint) {




					string line;

					line = stringify_result(
							input_counter,
							0 /* verbose_level */);
					Lines.push_back(line);

					if (!F_first_time[input_counter]) {

						if (f_v) {
							cout << "classification_of_varieties_nauty::main_loop "
									"memory_footprint_reduction freeing one object, "
									"nb_times_memory_footprint_reduction = " << nb_times_memory_footprint_reduction << endl;
						}

						nb_times_memory_footprint_reduction++;

						FREE_OBJECT(Canonical_forms[input_counter]);
						Canonical_forms[input_counter] = NULL;
					}
				}
			}

		}


	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::main_loop "
				"CB->nb_types = " << CB->nb_types << endl;
	}



	if (f_v) {
		cout << "classification_of_varieties_nauty::main_loop done" << endl;
	}
}

void classification_of_varieties_nauty::handle_one_input_case(
		int input_counter, int &nb_iso,
		variety_compute_canonical_form *Variety_compute_canonical_form,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "classification_of_varieties_nauty::handle_one_input_case" << endl;
		cout << "classification_of_varieties_nauty::handle_one_input_case "
				"verbose_level = " << verbose_level << endl;
	}


	other::l1_interfaces::nauty_interface_control *Nauty_control;



	if (Classifier->f_nauty_control) {
		Nauty_control = Classifier->Nauty_interface_control;
	}
	else {
		cout << "classification_of_varieties_nauty::handle_one_input_case "
				"Classifier->f_nauty_control is false, so no nauty_control" << endl;
		exit(1);
	}


	int f_found_canonical_form = false;
	int idx_canonical_form = -1;
	int idx_equation = -1;
	int f_found_eqn = false;


	if (f_v) {
		cout << "classification_of_varieties_nauty::handle_one_input_case "
				"input_counter = " << input_counter << " / " << Input->nb_objects_to_test
				<< " before Variety_compute_canonical_form->compute_canonical_form_nauty" << endl;
	}
	Variety_compute_canonical_form->compute_canonical_form_nauty(
			Nauty_control,
			f_found_canonical_form,
			idx_canonical_form,
			idx_equation,
			f_found_eqn,
			verbose_level - 1);


	if (f_v) {
		cout << "classification_of_varieties_nauty::handle_one_input_case "
				"input_counter = " << input_counter << " / " << Input->nb_objects_to_test
				<< " after Variety_compute_canonical_form->compute_canonical_form_nauty" << endl;
	}

	Goi[input_counter] = Variety_compute_canonical_form->Variety_stabilizer_compute->Stab_gens_variety->group_order_as_lint();

	if (f_found_canonical_form
			&& f_found_eqn) {

		F_first_time[input_counter] = false;

		actions::action *A;

		A = Classifier->Ring_with_action->PA->A;


		if (f_v) {
			cout << "classification_of_varieties_nauty::handle_one_input_case "
					"input_counter = " << input_counter << " / " << Input->nb_objects_to_test
					<< " is isomorphic to canonical form " << idx_canonical_form
					<< " and equation " << idx_equation << endl;
			cout << "classification_of_varieties_nauty::handle_one_input_case an isomorphism is given by phi=" << endl;

			A->Group_element->element_print(
					Elt_phi, cout);

			cout << "classification_of_varieties_nauty::handle_one_input_case an isomorphism is given by phi=";
			A->Group_element->element_print_for_make_element(
					Elt_phi, cout);
			cout << endl;

		}

		Phi[input_counter] = NEW_int(A->make_element_size);
		A->Group_element->element_code_for_make_element(
				Elt_phi, Phi[input_counter]);


	}
	else if (f_found_canonical_form
			&& !f_found_eqn) {

		F_first_time[input_counter] = true;

	}
	else if (!f_found_canonical_form) {

		F_first_time[input_counter] = true;

	}
	else {
		cout << "classification_of_varieties_nauty::handle_one_input_case illegal combination" << endl;
		exit(1);
	}

	Idx_canonical_form[input_counter] = idx_canonical_form;
	Idx_equation[input_counter] = idx_equation;

	if (F_first_time[input_counter]) {


		Iso_idx[input_counter] = nb_iso;

		int idx, i;


		for (i = 0; i < input_counter; i++) {
			idx = Idx_canonical_form[i];
			if (idx >= idx_canonical_form) {
				Idx_canonical_form[i]++;
			}
		}

		Orbit_input_idx[nb_isomorphism_classes] = input_counter;
		nb_isomorphism_classes++;

		nb_iso++;

	}
	else {
		Iso_idx[input_counter] = Iso_idx[Idx_canonical_form[input_counter]];
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::handle_one_input_case done" << endl;
	}
}

std::string classification_of_varieties_nauty::stringify_result(
		int input_counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::stringify_result" << endl;
	}

	string s;
	string line;

	line = Canonical_forms[input_counter]->stringify_csv_entry_one_line_nauty(
			input_counter, verbose_level);

	s = std::to_string(input_counter) + "," + line;


	if (f_v) {
		cout << "classification_of_varieties_nauty::stringify_result done" << endl;
	}
	return s;

}

void classification_of_varieties_nauty::write_classification_by_nauty_csv(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	std::string fname;


	if (f_v) {
		cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv" << endl;
	}

	fname = fname_base + "_classified.csv";

	{
		ofstream ost(fname);


		string header;

		header = stringify_csv_header_line_nauty(verbose_level);
		ost << header << endl;

		int input_counter, cnt;

		cnt = 0;
		for (input_counter = 0; input_counter < Input->nb_objects_to_test; input_counter++) {

			if (Classifier->skip_this_one(input_counter)) {

				if (f_v) {
					cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv "
							"input_counter=" << input_counter << " / " << Input->nb_objects_to_test << " skipped" << endl;
				}

			}
			else {
				if (f_v) {
					cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv "
							"input_counter=" << input_counter << " / " << Input->nb_objects_to_test << endl;
				}

				if (Classifier->f_nauty_control && Classifier->Nauty_interface_control->f_reduce_memory_footprint) {
					ost << Lines[cnt++] << endl;
				}

				else {

					string line;

					line = Canonical_forms[input_counter]->stringify_csv_entry_one_line_nauty(
							input_counter, verbose_level);

					ost << input_counter << "," << line << endl;
				}
			}

		}
		ost << "END" << endl;
	}


	other::orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv done" << endl;
	}
}



std::string classification_of_varieties_nauty::stringify_csv_header_line_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::stringify_csv_header_line_nauty" << endl;
	}

	std::string header;

	header = "ROW,Q,FO,PO,SO,Iso_idx,F_Fst,Idx_canonical,Idx_eqn,Eqn,Eqn2,nb_pts_on_curve,pts_on_curve,Bitangents";

#if 1
	if (Classifier->has_description()) {
		if (Classifier->get_description()->carry_through.size()) {
			int i;

			for (i = 0; i < Classifier->get_description()->carry_through.size(); i++) {
				header += "," + Classifier->get_description()->carry_through[i];
			}
		}
		else {
			if (f_v) {
				cout << "classification_of_varieties_nauty::stringify_csv_header_line_nauty no carry though" << endl;
			}

		}
	}
#endif

	header += ",NO_N,NO_ago,NO_base_len,NO_aut_cnt,NO_base,NO_tl,NO_aut,NO_cl,NO_stats";
	header += ",nb_eqn,Phi,ago";

	return header;
}


void classification_of_varieties_nauty::report(
		poset_classification::poset_classification_report_options *Report_options,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::report" << endl;
	}

	string fname;


	if (Report_options->f_fname) {
		fname = Report_options->fname;
	}

	fname += "_orbits.tex";



	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(ost);


		if (f_v) {
			cout << "classification_of_varieties_nauty::report "
					"before report_iso_types" << endl;
		}

		report_iso_types(ost, verbose_level);
		if (f_v) {
			cout << "classification_of_varieties_nauty::report "
					"after report_iso_types" << endl;
		}


		L.foot(ost);
	}



	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

#if 0
	{
		string fname_data;

		fname_data = label + "_canonical_form_data.csv";


		if (f_v) {
			cout << "classification_of_varieties_nauty::report "
					"before export_canonical_form_data" << endl;
		}
		export_canonical_form_data(
				fname_data, verbose_level);
		if (f_v) {
			cout << "classification_of_varieties_nauty::report "
					"after export_canonical_form_data" << endl;
		}

		if (f_v) {
			cout << "Written file " << fname_data << " of size "
					<< Fio.file_size(fname_data) << endl;
		}
	}
#endif
	if (f_v) {
		cout << "classification_of_varieties_nauty::report done" << endl;
	}

}


void classification_of_varieties_nauty::report_iso_types(
		std::ostream &ost, int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::report_iso_types" << endl;
	}


	int orbit_index;
	int nb_orbits;

	if (f_v) {
		cout << "classification_of_varieties_nauty::report_iso_types" << endl;
	}


	nb_orbits = nb_isomorphism_classes;


	int idx;

	{


		ost << "Classification\\\\" << endl;
		ost << "$q=" << Classifier->Ring_with_action->PA->F->q << "$\\\\" << endl;
		ost << "Number of isomorphism classes: " << nb_orbits << "\\\\" << endl;


		std::vector<long int> Ago;

		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {
			idx = Orbit_input_idx[orbit_index];


			Ago.push_back(Goi[idx]);
		}

		other::data_structures::tally_lint T;

		T.init_vector_lint(
				Ago,
				false /* f_second */,
				0 /* verbose_level */);
		ost << "Automorphism group order statistic: " << endl;
		//ost << "$";
		T.print_file_tex(ost, true /* f_backwards */);
		ost << "\\\\" << endl;


		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;


		if (f_v) {
			cout << "classification_of_varieties_nauty::report "
					"preparing reps" << endl;
		}
		ost << "The isomorphism classes are:\\\\" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {

			idx = Orbit_input_idx[orbit_index];

			//int *equation;

			if (f_v) {
				cout << "classification_of_varieties_nauty::report_iso_types "
						"orbit_index = " << orbit_index << endl;
			}

			ost << "Isomorphism class " << orbit_index << " / " << nb_orbits
					<< " is input " << idx << ":\\\\" << endl;
			ost << "Automorphism group order " << Goi[idx] << "\\\\" << endl;


			variety_object_with_action *Vo;
				// [nb_objects_to_test]


			Vo = Input->Vo[idx];

			ost << "Number of points " << Vo->Variety_object->Point_sets->Set_size[0] << "\\\\" << endl;

			Vo->Variety_object->report_equations(ost);

			ost << "Points:\\\\" << endl;
			Classifier->Ring_with_action->PA->P->Reporting->print_set_of_points_easy(
					ost,
					Vo->Variety_object->Point_sets->Sets[0],
					Vo->Variety_object->Point_sets->Set_size[0]);

			Canonical_forms[idx]->Variety_stabilizer_compute->report(ost);


			ost << endl;
			ost << "\\bigskip" << endl;
			ost << endl;


			combinatorics_with_groups::combinatorics_with_action CombiA;
			int size_limit_for_printing = 50;
			groups::strong_generators *gens;

			gens = Canonical_forms[idx]->Variety_stabilizer_compute->Stab_gens_variety;


			if (f_v) {
				cout << "classification_of_varieties_nauty::report_iso_types "
						"before CombiA.report_TDO_and_TDA_projective_space" << endl;
			}
			CombiA.report_TDO_and_TDA_projective_space(
					ost,
					Classifier->Ring_with_action->PA->P,
					Vo->Variety_object->Point_sets->Sets[0],
					Vo->Variety_object->Point_sets->Set_size[0],
					Classifier->Ring_with_action->PA->A,
					Classifier->Ring_with_action->PA->A_on_lines,
					gens, size_limit_for_printing,
					verbose_level);
			if (f_v) {
				cout << "classification_of_varieties_nauty::report_iso_types "
						"after CombiA.report_TDO_and_TDA_projective_space" << endl;
			}


		}
	}
	if (f_v) {
		cout << "classification_of_varieties_nauty::report_iso_types done" << endl;
	}
}

void classification_of_varieties_nauty::generate_source_code(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::generate_source_code" << endl;
	}

	int f_vv = (verbose_level >= 2);
	std::string fname;
	int orbit_index;
	int i;

	int nb_orbits;
	int nb_monomials;

	//actions::action *A_on_lines;


	fname = fname_base + ".cpp";


	nb_orbits = nb_isomorphism_classes;
	nb_monomials = Classifier->Ring_with_action->Poly_ring->get_nb_monomials();

	if (f_v) {
		cout << "classification_of_varieties_nauty::generate_source_code "
				"nb_orbits = " << nb_orbits << endl;
		cout << "classification_of_varieties_nauty::generate_source_code "
				"nb_monomials = " << nb_monomials << endl;
	}

	actions::action *A;

	A = Classifier->Ring_with_action->PA->A;
	//A_on_lines = Classifier->PA->A_on_lines;

	{
		ofstream f(fname);


		other::orbiter_kernel_system::os_interface Os;
		string str;

		Os.get_date(str);

		f << "// file: " << fname << endl;
		f << "// created by Orbiter " << endl;
		f << "// creation date: " << str << endl << endl;
		f << "// " << endl;



		f << "static int " << fname_base << "_nb_reps = "
				<< nb_orbits << ";" << endl;
		f << "static int " << fname_base << "_size = "
				<< nb_monomials << ";" << endl;



		if (f_v) {
			cout << "classification_of_varieties_nauty::generate_source_code "
					"preparing reps" << endl;
		}
		f << "// the equations:" << endl;
		f << "static int " << fname_base << "_reps[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			int *equation;

			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			int idx;

			idx = Orbit_input_idx[orbit_index];

			equation = Input->Vo[idx]->Variety_object->eqn;
			//equation = Canonical_forms[idx]->canonical_equation;


			f << "\t";
			for (i = 0; i < nb_monomials; i++) {
				f << equation[i];
				f << ", ";
			}
			f << endl;
		}
		f << "};" << endl;



		if (f_v) {
			cout << "classification_of_varieties_nauty::generate_source_code "
					"preparing stab_order" << endl;
		}
		f << "// the stabilizer orders:" << endl;
		f << "static const char *" << fname_base << "_stab_order[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {

			int idx;

			idx = Orbit_input_idx[orbit_index];



			algebra::ring_theory::longinteger_object ago;

			ago.create(Goi[idx]);

			f << "\t\"";

			ago.print_not_scientific(f);
			f << "\"," << endl;

		}
		f << "};" << endl;





		if (f_v) {
			cout << "classification_of_varieties_nauty::generate_source_code "
					"preparing Bitangents" << endl;
		}
		f << "// the 28 bitangents:" << endl;
		f << "static long int " << fname_base << "_Bitangents[] = { " << endl;


		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			int idx;

			idx = Orbit_input_idx[orbit_index];

			long int *bitangents_orig;

			bitangents_orig = Input->Vo[idx]->Variety_object->Line_sets->Sets[0];
			if (Input->Vo[idx]->Variety_object->Line_sets->Set_size[0] != 28) {
				cout << "classification_of_varieties_nauty::generate_source_code Set_size[0] != 28" << endl;
				cout << "classification_of_varieties_nauty::generate_source_code Set_size[0] = " << Input->Vo[idx]->Variety_object->Point_sets->Set_size[0] << endl;
				exit(1);
			}

			int j;

			f << "\t";
			for (j = 0; j < 28; j++) {
				f << bitangents_orig[j];
				f << ", ";
			}
			f << endl;

#if 0
			if (Variety_table[idx]) {
				long int *bitangents_orig;
				long int *bitangents_canonical;

				bitangents_orig = Variety_table[idx]->Qco->Quartic_curve_object->bitangents28;
				bitangents_canonical = NEW_lint(28);
				for (j = 0; j < 28; j++) {
					bitangents_canonical[j] =
							A_on_lines->Group_element->element_image_of(
									bitangents_orig[j],
							Variety_table[idx]->transporter_to_canonical_form,
							0 /* verbose_level */);
				}




				f << "\t";
				for (j = 0; j < 28; j++) {
					f << bitangents_canonical[j];
					f << ", ";
				}
				f << endl;
			}
			else {
				f << "\t";
				for (j = 0; j < 28; j++) {
					f << 0;
					f << ", ";
				}
				f << "// problem" << endl;

			}
#endif


		}
		f << "};" << endl;

		if (f_v) {
			cout << "classification_of_varieties_nauty::generate_source_code "
					"preparing make_element_size" << endl;
		}

		f << "static int " << fname_base << "_make_element_size = "
				<< A->make_element_size << ";" << endl;

		if (f_v) {
			cout << "classification_of_varieties_nauty::generate_source_code "
					"preparing stabilizer" << endl;
		}
		{
			int *stab_gens_first;
			int *stab_gens_len;
			int fst;


			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"before loop1" << endl;
			}

			stab_gens_first = NEW_int(nb_orbits);
			stab_gens_len = NEW_int(nb_orbits);
			fst = 0;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {


				//groups::strong_generators *gens;

				int idx;

				idx = Orbit_input_idx[orbit_index];
				//idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];

				//canonical_form_substructure *CFS = CFS_table[idx];
				//gens = CFS->Gens_stabilizer_canonical_form;


				if (true /*Canonical_forms[idx]->Canonical_object->f_has_automorphism_group*/) {
					groups::strong_generators *gens;

					//gens = Canonical_forms[idx]->Canonical_object->Stab_gens;
					//groups::strong_generators *gens;

					gens = Canonical_forms[idx]->Variety_stabilizer_compute->Stab_gens_variety;



					if (gens) {
						stab_gens_first[orbit_index] = fst;
						stab_gens_len[orbit_index] = gens->gens->len;
						fst += stab_gens_len[orbit_index];
					}
					else {
						cout << "classification_of_varieties_nauty::generate_source_code "
								"gens not available" << endl;
						stab_gens_first[orbit_index] = fst;
						stab_gens_len[orbit_index] = 0;
						fst += stab_gens_len[orbit_index];
					}
				}
				else {
					stab_gens_first[orbit_index] = fst;
					stab_gens_len[orbit_index] = 0;
					fst += 0;

				}
			}

			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"after loop1" << endl;
			}

			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"preparing stab_gens_fst" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_fst[] = { " << endl << "\t";
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				f << stab_gens_first[orbit_index];
				if (orbit_index < nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;

			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"preparing stab_gens_len" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_len[] = { " << endl << "\t";
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				f << stab_gens_len[orbit_index];
				if (orbit_index < nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;


			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"preparing stab_gens" << endl;
			}
			f << "static int " << fname_base << "_stab_gens[] = {" << endl;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				int j;

				for (j = 0; j < stab_gens_len[orbit_index]; j++) {
					if (f_vv) {
						cout << "classification_of_varieties::generate_source_code "
								"before extract_strong_generators_in_order "
								"generator " << j << " / "
								<< stab_gens_len[orbit_index] << endl;
					}
					f << "\t";

					//groups::strong_generators *gens;

					int idx;

					idx = Orbit_input_idx[orbit_index];
					//idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];

					//canonical_form_substructure *CFS = CFS_table[idx];
					//gens = CFS->Gens_stabilizer_canonical_form;
					if (true /*Canonical_forms[idx]->Canonical_object->f_has_automorphism_group*/) {
						groups::strong_generators *gens;

						//gens = Canonical_forms[idx]->Canonical_object->Stab_gens;
						gens = Canonical_forms[idx]->Variety_stabilizer_compute->Stab_gens_variety;


						if (gens) {
							A->Group_element->element_print_for_make_element(
									gens->gens->ith(j), f);
							f << "," << endl;
						}
						else {
							cout << "classification_of_varieties_nauty::generate_source_code "
									"gens are not available" << endl;
						}
					}
					else {
						f << "// problem" << endl;
					}
				}
			}
			f << "};" << endl;

			if (f_v) {
				cout << "classification_of_varieties_nauty::generate_source_code "
						"after preparing stab_gens" << endl;
			}

			FREE_int(stab_gens_first);
			FREE_int(stab_gens_len);
		}
	}

	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "written file " << fname << " of size "
			<< Fio.file_size(fname.c_str()) << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::generate_source_code done" << endl;
	}
}



}}}

