/*
 * data_input_stream_output.cpp
 *
 *  Created on: Apr 9, 2025
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {


data_input_stream_output::data_input_stream_output()
{
	Record_birth();

	Classification_of_objects = NULL;


	CB = NULL;

	nb_input = 0;

	Ago = NULL;
	F_reject = NULL;
	NO = NULL;
	OWCF = NULL;

	nb_orbits = 0;
	Idx_transversal = NULL;
	Ago_transversal = NULL;

	T_Ago = NULL;


}

data_input_stream_output::~data_input_stream_output()
{
	Record_death();

	if (Ago) {
		FREE_lint(Ago);
	}
	if (F_reject) {
		FREE_int(F_reject);
	}

	if (NO) {
		FREE_pvoid((void **) NO);
	}

	if (OWCF) {
#if 0
		int i;

		for (i = 0; i < nb_orbits; i++) {
			FREE_OBJECT(OWCF[i]);
		}
#endif
		FREE_pvoid((void **) OWCF);
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

void data_input_stream_output::init(
		classification_of_objects *Classification_of_objects, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream_output::init" << endl;
	}

	data_input_stream_output::Classification_of_objects = Classification_of_objects;


	CB = NEW_OBJECT(classify_bitvectors);

	nb_input = Classification_of_objects->IS->Objects.size();

	if (f_v) {
		cout << "data_input_stream_output::init "
				"number of input objects: " << nb_input << endl;
	}


	Ago = NEW_lint(nb_input);

	F_reject = NEW_int(nb_input);

	NO = (other::l1_interfaces::nauty_output **) NEW_pvoid(nb_input);

	OWCF = (any_combinatorial_object **) NEW_pvoid(nb_input);


	if (f_v) {
		cout << "data_input_stream_output::init done" << endl;
	}

}

void data_input_stream_output::after_classification(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream_output::after_classification" << endl;
	}

	Idx_transversal = NEW_int(nb_orbits);
	Ago_transversal = NEW_lint(nb_orbits);

	int iso_idx;
	int input_idx;

	for (input_idx = 0, iso_idx = 0;

			input_idx < Classification_of_objects->IS->Objects.size();

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
		for (input_idx = 0; input_idx < nb_input; input_idx++) {
			cout << setw(3) << input_idx << " : " << setw(5)
					<< Ago[input_idx] << " : " << F_reject[input_idx] << endl;
		}
	}

	if (f_v) {
		cout << "transversal of orbit representatives:" << endl;
		int cnt;
		cout << "iso type : input object : ago" << endl;
		for (input_idx = 0, cnt = 0; input_idx < nb_input; input_idx++) {
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


	T_Ago = NEW_OBJECT(other::data_structures::tally);
	T_Ago->init_lint(Ago_transversal, nb_orbits, false, 0);

	if (f_v) {
		cout << "Automorphism group orders of orbit transversal: ";
		T_Ago->print_first(true /* f_backwards */);
		cout << endl;
	}



	if (Classification_of_objects->Descr->f_save_ago) {
		if (f_v) {
			cout << "classification_of_objects::process_multiple_objects_from_file "
					"f_save_ago is true" << endl;
		}

		save_automorphism_group_order(verbose_level);

	}

	if (Classification_of_objects->Descr->f_save_transversal) {
		if (f_v) {
			cout << "classification_of_objects::process_multiple_objects_from_file "
					"f_save_transversal is true" << endl;
		}

		save_transversal(verbose_level);

	}


	if (f_v) {
		cout << "data_input_stream_output::after_classification done" << endl;
	}

}


void data_input_stream_output::save_automorphism_group_order(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream_output::save_automorphism_group_order " << endl;
	}
	string ago_fname;
	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::string_tools ST;

	ago_fname = Classification_of_objects->get_label();
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
		cout << "data_input_stream_output::save_automorphism_group_order done" << endl;
	}
}

void data_input_stream_output::save_transversal(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream_output::save_transversal " << endl;
	}
	string fname;
	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::string_tools ST;

	fname = Classification_of_objects->get_label();

	ST.replace_extension_with(fname, "_transversal.csv");
	string label;

	label.assign("Transversal");

	Fio.Csv_file_support->int_vec_write_csv(
			Idx_transversal, nb_orbits, fname, label);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "data_input_stream_output::save_transversal done" << endl;
	}
}

void data_input_stream_output::report_summary_of_iso_types(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream_output::report_summary_of_iso_types" << endl;
	}
	other::l1_interfaces::latex_interface L;


	if (f_v) {
		cout << "data_input_stream_output::report_summary_of_iso_types "
				"before Summary of Orbits" << endl;
	}

	ost << "\\section*{Summary of Orbits}" << endl;

	std::string *Table;
	int nb_rows, nb_cols;

	create_summary_of_iso_types_table(
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


	if (f_v) {
		cout << "data_input_stream_output::report_summary_of_iso_types "
				"after Summary of Orbits" << endl;
	}

	delete [] Table;


	if (f_v) {
		cout << "data_input_stream_output::report_summary_of_iso_types done" << endl;
	}

}


void data_input_stream_output::create_summary_of_iso_types_table(
		std::string *&Table,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream_output::create_summary_of_iso_types_table" << endl;
	}
	other::data_structures::sorting Sorting;

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



		algebra::ring_theory::longinteger_object go;
		go.create(Ago_transversal[i]);

		s_ago = go.stringify();


		int *Input_objects;
		int nb_input_objects;

		if (f_v) {
			cout << "data_input_stream_output::create_summary_of_iso_types_table "
					"before CB->C_type_of->get_class_by_value" << endl;
		}
		CB->C_type_of->get_class_by_value(
				Input_objects,
			nb_input_objects, j,
			0 /*verbose_level */);

		if (f_v) {
			cout << "data_input_stream_output::create_summary_of_iso_types_table "
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

	if (f_v) {
		cout << "data_input_stream_output::create_summary_of_iso_types_table done" << endl;
	}
}





}}}}


