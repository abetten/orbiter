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
	nb_iso_orbits = 0;
	Orbit_input_idx = NULL;

	Classification_table_nauty = NULL;


	// canonical forms
	Canonical_forms = NULL;

	Elt = NULL;
	eqn2 = NULL;
	Goi = NULL;



}

classification_of_varieties_nauty::~classification_of_varieties_nauty()
{
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

	if (Classification_table_nauty) {
		FREE_int(Classification_table_nauty);
	}

}

void classification_of_varieties_nauty::init(
		input_objects_of_type_variety *Input,
		canonical_form_classifier *Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::init " << endl;
	}
	if (f_v) {
		cout << "classification_of_varieties_nauty::init" << endl;
	}

	classification_of_varieties_nauty::Classifier = Classifier;

	classification_of_varieties_nauty::Input = Input;
	classification_of_varieties_nauty::fname_base = Input->fname_base_out;

	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"nb_objects_to_test=" << Input->nb_objects_to_test << endl;
		cout << "classification_of_varieties_nauty::init "
				"fname_base=" << fname_base << endl;
		cout << "classification_of_varieties_nauty::init "
				"Input->nb_objects_to_test=" << Input->nb_objects_to_test << endl;
	}


	Elt = NEW_int(Classifier->Ring_with_action->PA->A->elt_size_in_int);
	eqn2 = NEW_int(Classifier->Ring_with_action->Poly_ring->get_nb_monomials());
	Goi = NEW_lint(Input->nb_objects_to_test);


	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"before prepare_canonical_forms" << endl;
	}
	prepare_canonical_forms(verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"after prepare_canonical_forms" << endl;
	}

#if 0
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"before classify_nauty" << endl;
	}
	classify_nauty(verbose_level - 1);
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"after classify_nauty" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"before write_classification_by_nauty_csv" << endl;
	}
	write_classification_by_nauty_csv(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"after write_classification_by_nauty_csv" << endl;
	}

#endif


	if (f_v) {
		cout << "classification_of_varieties_nauty::init done" << endl;
	}

}

void classification_of_varieties_nauty::prepare_canonical_forms(
		int verbose_level)
// initializes the entries of Variety_table[nb_inputs]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_canonical_forms " << endl;
	}

	if (Input == NULL) {
		cout << "classification_of_varieties_nauty::prepare_canonical_forms Input == NULL" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_canonical_forms "
				"Input->nb_objects_to_test" << Input->nb_objects_to_test << endl;
	}


	int input_counter;

	Canonical_forms = (variety_compute_canonical_form **)
			NEW_pvoid(Input->nb_objects_to_test);


	for (input_counter = 0; input_counter < Input->nb_objects_to_test; input_counter++) {


		string fname_case_out;

		fname_case_out = Input->fname_base_out + "_cnt" + std::to_string(input_counter);

		variety_compute_canonical_form *Variety;

		Variety = NEW_OBJECT(variety_compute_canonical_form);

		if (f_v) {
			cout << "classification_of_varieties_nauty::prepare_canonical_forms "
					"input_counter = " << input_counter << " / " << Classifier->Input->nb_objects_to_test
					<< " before Variety->init" << endl;
		}
		Variety->init(
				Classifier,
				Classifier->Ring_with_action,
				Classifier->Classification_of_varieties_nauty,
				fname_case_out,
				input_counter,
				Classifier->Input->Vo[input_counter],
				verbose_level - 2);



		if (f_v) {
			cout << "classification_of_varieties_nauty::prepare_canonical_forms "
					"input_counter = " << input_counter << " / " << Classifier->Input->nb_objects_to_test
					<< " after Variety->init" << endl;
		}

		Canonical_forms[input_counter] = Variety;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::prepare_canonical_forms done" << endl;
	}
}


void classification_of_varieties_nauty::classify_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty" << endl;
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
	allocate_tables(verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"after allocate_tables" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"before main_loop" << endl;
	}
	main_loop(
			verbose_level);
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

	CB = NEW_OBJECT(canonical_form_classification::classify_bitvectors);


	// classification by nauty:

	F_first_time = NEW_int(Input->nb_objects_to_test);
	Iso_idx = NEW_int(Input->nb_objects_to_test);
	Idx_canonical_form = NEW_int(Input->nb_objects_to_test);
	Idx_equation = NEW_int(Input->nb_objects_to_test);
	Orbit_input_idx = NEW_int(Input->nb_objects_to_test);
	nb_iso_orbits = 0;



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


#if 0
		string fname_case_out;

		fname_case_out = fname_base + "_cnt" + std::to_string(input_counter);

		variety_compute_canonical_form *Variety;

		Variety = NEW_OBJECT(variety_compute_canonical_form);

		if (f_v) {
			cout << "classification_of_varieties_nauty::main_loop "
					"input_counter = " << input_counter << " / " << nb_objects_to_test
					<< " before Variety->init" << endl;
		}
		Variety->init(
				Classifier,
				Classifier->Ring_with_action,
				Classifier->Classification_of_varieties_nauty,
				fname_case_out,
				input_counter,
				&Input_Vo[input_counter],
				verbose_level - 2);



		if (f_v) {
			cout << "classification_of_varieties_nauty::main_loop "
					"input_counter = " << input_counter << " / " << nb_objects_to_test
					<< " after Variety->init" << endl;
		}

		Canonical_forms[input_counter] = Variety;

#endif

		variety_compute_canonical_form *Variety;

		Variety = Canonical_forms[input_counter];


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
			//continue;
		}
		else {

			int f_save_nauty_input_graphs = false;

			if (Classifier->has_description()) {
				f_save_nauty_input_graphs = Classifier->get_description()->f_save_nauty_input_graphs;
			}
			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"input_counter = " << input_counter << " / " << Input->nb_objects_to_test
						<< " before Variety->compute_canonical_form_nauty_new" << endl;
			}
			Variety->compute_canonical_form_nauty_new(
					f_save_nauty_input_graphs,
					verbose_level);


			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"input_counter = " << input_counter << " / " << Input->nb_objects_to_test
						<< " after Variety->compute_canonical_form_nauty_new" << endl;
			}

			Goi[input_counter] = Variety->Variety_stabilizer_compute->Stab_gens_variety->group_order_as_lint();

			if (Variety->Variety_stabilizer_compute->f_found_canonical_form
					&& Variety->Variety_stabilizer_compute->f_found_eqn) {

				F_first_time[input_counter] = false;

			}
			else if (Variety->Variety_stabilizer_compute->f_found_canonical_form
					&& !Variety->Variety_stabilizer_compute->f_found_eqn) {

				F_first_time[input_counter] = true;

			}
			else if (!Variety->Variety_stabilizer_compute->f_found_canonical_form) {

				F_first_time[input_counter] = true;

			}
			else {
				cout << "classification_of_varieties_nauty::main_loop illegal combination" << endl;
				exit(1);
			}

			Idx_canonical_form[input_counter] = Variety->Variety_stabilizer_compute->idx_canonical_form;
			Idx_equation[input_counter] = Variety->Variety_stabilizer_compute->idx_equation;

			if (F_first_time[input_counter]) {


				Iso_idx[input_counter] = nb_iso;

				int idx, i;


				for (i = 0; i < input_counter; i++) {
					idx = Idx_canonical_form[i];
					if (idx >= Variety->Variety_stabilizer_compute->idx_canonical_form) {
						Idx_canonical_form[i]++;
					}
				}

				Orbit_input_idx[nb_iso_orbits] = input_counter;
				nb_iso_orbits++;

				nb_iso++;

			}
			else {
				Iso_idx[input_counter] = Iso_idx[Idx_canonical_form[input_counter]];
			}
		}


	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::main_loop done" << endl;
	}
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
	fname = fname_base + "_classification_by_nauty.csv";



	{
		ofstream ost(fname);


		string header;

		header = stringify_csv_header_line_nauty(verbose_level);
		ost << header << endl;

		int input_counter;

		for (input_counter = 0; input_counter < Input->nb_objects_to_test; input_counter++) {

			if (f_v) {
				cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv "
						"input_counter=" << input_counter << " / " << Input->nb_objects_to_test << endl;
			}

			string line;

			line = Canonical_forms[input_counter]->stringify_csv_entry_one_line_nauty_new(
					input_counter, verbose_level);

			ost << input_counter << "," << line << endl;


		}
		ost << "END" << endl;
	}


	orbiter_kernel_system::file_io Fio;

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

	header = "ROW,CNT,PO,SO,PO_GO,PO_INDEX,Iso_idx,F_Fst,Idx_canonical,Idx_eqn,Eqn,Eqn2,NPts,Pts,Bitangents";

#if 0
	if (Classifier->Descr->carry_through.size()) {
		int i;

		for (i = 0; i < Classifier->Descr->carry_through.size(); i++) {
			header += "," + Classifier->Descr->carry_through[i];
		}
	}
#endif

	header += ",NO_N,NO_ago,NO_base_len,NO_aut_cnt,NO_base,NO_tl,NO_aut,NO_cl,NO_stats";
	header += ",nb_eqn,ago";

	return header;
}


void classification_of_varieties_nauty::report(
		std::string &fname_base,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::report" << endl;
	}

	string fname;


	fname = fname_base + "_orbits.tex";



	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;

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



	orbiter_kernel_system::file_io Fio;

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


	nb_orbits = nb_iso_orbits;


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

		data_structures::tally_lint T;

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

			ost << "Isomorphism class " << orbit_index << " / " << nb_orbits << " is input " << idx << ":\\\\" << endl;
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
#if 0

	int f_vv = (verbose_level >= 2);
	std::string fname;
	int orbit_index;
	int i;

	int nb_orbits;
	int nb_monomials;

	actions::action *A;
	//actions::action *A_on_lines;


	fname = fname_base + ".cpp";


	nb_orbits = Tally->nb_types;
	nb_monomials = Classifier->Ring_with_action->Poly_ring->get_nb_monomials();


	A = Classifier->Ring_with_action->PA->A;
	//A_on_lines = Classifier->PA->A_on_lines;

	{
		ofstream f(fname);

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

			idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];


			//canonical_form_substructure *CFS = Variety_table[idx];


			if (Variety_table[idx]) {
				//equation = Classification_of_quartic_curves->Reps + orbit_index * Classification_of_quartic_curves->data_set_sz;
				equation = Variety_table[idx]->canonical_equation;

				f << "\t";
				for (i = 0; i < nb_monomials; i++) {
					f << equation[i];
					f << ", ";
				}
				f << endl;
			}
			else {
				f << "\t";
				for (i = 0; i < nb_monomials; i++) {
					f << 0;
					f << ", ";
				}
				f << "// problem" << endl;

			}

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

			ring_theory::longinteger_object ago;

			int idx;

			idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];


			ago.create(Goi[idx]);

			f << "\t\"";

			ago.print_not_scientific(f);
			f << "\"," << endl;

		}
		f << "};" << endl;




#if 0
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

			idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];

			//canonical_form_substructure *CFS = Variety_table[idx];


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

		}
		f << "};" << endl;
#endif

		f << "static int " << fname_base << "_make_element_size = "
				<< A->make_element_size << ";" << endl;

		{
			int *stab_gens_first;
			int *stab_gens_len;
			int fst;



			stab_gens_first = NEW_int(nb_orbits);
			stab_gens_len = NEW_int(nb_orbits);
			fst = 0;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {


				groups::strong_generators *gens;

				int idx;

				idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];

				//canonical_form_substructure *CFS = CFS_table[idx];
				//gens = CFS->Gens_stabilizer_canonical_form;
				if (Variety_table[idx]) {
					gens = Variety_table[idx]->gens_stab_of_canonical_equation;


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

					groups::strong_generators *gens;

					int idx;

					idx = Tally->sorting_perm_inv[Tally->type_first[orbit_index]];

					//canonical_form_substructure *CFS = CFS_table[idx];
					//gens = CFS->Gens_stabilizer_canonical_form;
					if (Variety_table[idx]) {
						gens = Variety_table[idx]->gens_stab_of_canonical_equation;


						if (gens) {
							A->Group_element->element_print_for_make_element(
									gens->gens->ith(j), f);
							f << endl;
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


			FREE_int(stab_gens_first);
			FREE_int(stab_gens_len);
		}
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname.c_str()) << endl;
#endif
	if (f_v) {
		cout << "classification_of_varieties_nauty::generate_source_code done" << endl;
	}
}



}}}
