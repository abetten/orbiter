/*
 * classify_using_canonical_forms.cpp
 *
 *  Created on: Aug 28, 2021
 *      Author: betten
 */






#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {


classify_using_canonical_forms::classify_using_canonical_forms()
{
	Record_birth();
	nb_input_objects = 0;
}

classify_using_canonical_forms::~classify_using_canonical_forms()
{
	Record_death();
}

void classify_using_canonical_forms::orderly_test(
		any_combinatorial_object *OwCF,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int &f_accept, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test" << endl;
	}

	other::l1_interfaces::nauty_interface_for_combo NI;

	other::l1_interfaces::nauty_output *NO;


	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"before NI.run_nauty_for_combo_basic" << endl;
	}
	NI.run_nauty_for_combo_basic(
			OwCF,
			Nauty_control,
			NO,
			verbose_level);
	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"after NI.run_nauty_for_combo_basic" << endl;
	}

	int nb_rows, nb_cols;

	OwCF->encoding_size(
				nb_rows, nb_cols,
				verbose_level);

	int last_row, last_pt;

	last_row = nb_rows - 1;

	last_pt = NO->canonical_labeling[last_row];

	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"last_row=" << last_row << " last_pt=" << last_pt
				<< endl;
	}

	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"before NO->belong_to_the_same_orbit" << endl;
	}
	if (NO->belong_to_the_same_orbit(last_row, last_pt, 0 /* verbose_level*/)) {
		f_accept = true;
	}
	else {
		f_accept = false;
	}
	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"after NO->belong_to_the_same_orbit f_accept = " << f_accept << endl;
	}

	//cout << "before FREE_OBJECT(NO);" << endl;
	FREE_OBJECT(NO);
	//cout << "after FREE_OBJECT(NO);" << endl;

	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test" << endl;
	}
}

void classify_using_canonical_forms::save_to_csv(
		std::string &fname_base,
		int f_identify_duals_if_possible,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::save_to_csv" << endl;
	}

	std::string *Col_headings;
	std::string *Table;
	int nb_rows, nb_cols;


	make_table_of_strings(
			Col_headings,
			Table, nb_rows, nb_cols,
			f_identify_duals_if_possible, Nauty_control,
			verbose_level - 2);

	std::string fname_out;

	fname_out = fname_base + "_canonical_forms.csv";

	other::orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_out,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "classify_using_canonical_forms::save_to_csv "
				"written file " << fname_out << " of size "
				<< Fio.file_size(fname_out) << endl;
	}

	delete [] Table;
	delete [] Col_headings;

	if (f_v) {
		cout << "classify_using_canonical_forms::save_to_csv done" << endl;
	}

}

void classify_using_canonical_forms::make_table_of_strings(
		std::string *&Col_headings,
		std::string *&Table, int &nb_rows, int &nb_cols,
		int f_identify_duals_if_possible,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
// assumes that the Objects are of type any_combinatorial_object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::make_table_of_strings" << endl;
	}
	int cnt;

	cnt = 0;
	for (std::multimap<uint32_t, int>::const_iterator it = Hashing.begin(); it != Hashing.end(); ++it) {
		cnt++;
		//      std::cout << it->first << " = " << it->second << "; ";
	}

	layer1_foundations::other::data_structures::algorithms Algorithms;

	nb_rows = cnt;
	nb_cols = 10;

	Table = new string [nb_rows * nb_cols];
	Col_headings = new string [nb_cols];

	cnt = 0;
	for (std::multimap<uint32_t, int>::const_iterator it = Hashing.begin(); it != Hashing.end(); ++it) {
		//      std::cout << it->first << " = " << it->second << "; ";

		uint32_t h;
		int idx;

		h = it->first;
		idx = it->second;

		string hash;

		string canonical_form;

		hash = Algorithms.stringify_data_hex_top_down(
				(unsigned char *) &h, sizeof(uint32_t));



		other::data_structures::bitvector *bv;

		bv = Bitvector_array[idx];

		canonical_form = Algorithms.stringify_data_hex(
				(unsigned char *) bv->get_data(), bv->get_allocated_length());


		Table[cnt * nb_cols + 0] = std::to_string(cnt);
		Table[cnt * nb_cols + 1] = hash;
		Table[cnt * nb_cols + 2] = std::to_string(idx);
		Table[cnt * nb_cols + 3] = canonical_form;

		string s;
		any_combinatorial_object *Any_combo;

		Any_combo = (any_combinatorial_object *) Objects[idx];
		s = Any_combo->stringify(0 /*verbose_level*/);
		Table[cnt * nb_cols + 4] = "\"" + s + "\"";

		Table[cnt * nb_cols + 5] = std::to_string(Ago[idx]);
		Table[cnt * nb_cols + 6] = std::to_string(input_index[idx]);


		if (f_identify_duals_if_possible) {
			if (Any_combo->can_dualize(verbose_level - 2)) {
				any_combinatorial_object *Any_combo_dualized;

				if (f_v) {
					cout << "classify_using_canonical_forms::make_table_of_strings before Any_combo->dualize" << endl;
				}
				Any_combo->dualize(Any_combo_dualized, verbose_level);
				if (f_v) {
					cout << "classify_using_canonical_forms::make_table_of_strings after Any_combo->dualize" << endl;
				}
				if (f_v) {
					cout << "classify_using_canonical_forms::make_table_of_strings Any_combo_dualized=" << endl;
					Any_combo_dualized->print();
				}

				int object_idx;
				uint32_t hash;

				int f_found;

				if (f_v) {
					cout << "classify_using_canonical_forms::make_table_of_strings before identify_object" << endl;
				}
				f_found = identify_object(
						Any_combo_dualized,
						Nauty_control,
						object_idx, hash,
						verbose_level);
				if (f_v) {
					cout << "classify_using_canonical_forms::make_table_of_strings after identify_object" << endl;
				}
				if (!f_found) {
					object_idx = -1;
					cout << "classify_using_canonical_forms::make_table_of_strings we did not find the dual object" << endl;
				}
				Table[cnt * nb_cols + 7] = std::to_string(object_idx);

				int f_is_self_dual = false;

				if (object_idx == idx) {
					f_is_self_dual = true;
				}
				Table[cnt * nb_cols + 8] = std::to_string(f_is_self_dual);
				Table[cnt * nb_cols + 9] = std::to_string(hash);

				FREE_OBJECT(Any_combo_dualized);

			}

		}

		cnt++;

	}

	Col_headings[0] = "Line";
	Col_headings[1] = "Hash";
	Col_headings[2] = "IsoIdx";
	Col_headings[3] = "CanonicalForm";
	Col_headings[4] = "Object";
	Col_headings[5] = "Ago";
	Col_headings[6] = "InputIdx";
	Col_headings[7] = "DualIdx";
	Col_headings[8] = "IsSelfDual";
	Col_headings[9] = "DualHash";


	if (f_v) {
		cout << "classify_using_canonical_forms::make_table_of_strings done" << endl;
	}
}

void classify_using_canonical_forms::find_object(
		any_combinatorial_object *OwCF,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int &f_found, int &idx,
		other::l1_interfaces::nauty_output *&NO,
		other::data_structures::bitvector *&Canonical_form,
		int verbose_level)
// computes the canonical form of the combinatorial object.
// Then performs a search for the combinatorial form based on the hash value and the hash table
// if f_found is true, Bitvector_array[idx] agrees with the given object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::find_object" << endl;
	}


	other::l1_interfaces::nauty_interface_for_combo NI;


	encoded_combinatorial_object *Enc;

	if (f_v) {
		cout << "classify_using_canonical_forms::find_object "
				"before NI.run_nauty_for_combo" << endl;
	}
	NI.run_nauty_for_combo(
			OwCF,
			true /* f_compute_canonical_form */,
			Nauty_control,
			Canonical_form,
			NO,
			Enc,
			verbose_level);
	if (f_v) {
		cout << "classify_using_canonical_forms::find_object "
				"after NI.run_nauty_for_combo" << endl;
	}

	FREE_OBJECT(Enc);

	uint32_t h;
	int c;
	other::data_structures::sorting sorting;

	h = Canonical_form->compute_hash();

	map<uint32_t, int>::iterator itr, itr1, itr2;

	itr1 = Hashing.lower_bound(h);
	itr2 = Hashing.upper_bound(h);

	f_found = false;

	// we loop over all hash values which are equal to h
	// and try to find the canonical form in the corresponding Bitvector_array

	for (itr = itr1; itr != itr2; ++itr) {

		idx = itr->second;

    	c = sorting.uchar_vec_compare(
    			Canonical_form->get_data(),
				Bitvector_array[idx]->get_data(),
				Canonical_form->get_allocated_length());

		if (c == 0) {
			if (f_v) {
				cout << "classify_using_canonical_forms::find_object "
						"found object at position " << idx << endl;
			}
			f_found = true;
			break;
		}
    }

	if (f_v) {
		cout << "classify_using_canonical_forms::find_object done" << endl;
	}
}

void classify_using_canonical_forms::add_object(
		any_combinatorial_object *OwCF,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int &f_new_object, int verbose_level)
// calls find_object, which computes the canonical form
// and tries to find it in the table.
// if not found, the canonical form will be added to the table
// and the object will be added to the vector Objects.
// OwCF will either be destroyed (in case the object is a duplicate)
// or stored in the Objects vector.
// The counter nb_input_objects will be incremented in any case.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::add_object" << endl;
	}

	other::l1_interfaces::nauty_output *NO;
	other::data_structures::bitvector *Canonical_form;


	int f_found;
	int idx;

	find_object(
			OwCF,
			Nauty_control,
			f_found, idx,
			NO,
			Canonical_form,
			verbose_level);


	uint32_t h;

	h = Canonical_form->compute_hash();

	if (!f_found) {

		// we append a new object to the table.
		// we save the any_combinatorial_object with it.

		f_new_object = true;
		idx = Bitvector_array.size();
		Bitvector_array.push_back(Canonical_form);

		// store the combinatorial object in the vector Objects:
		Objects.push_back(OwCF);

		Ago.push_back(NO->Ago->as_lint());
		Hashing.insert(pair<uint32_t, int>(h, idx));
		input_index.push_back(nb_input_objects);

	}
	else {

		// since the object has been found,
		// we will delete the object and the canonical form:

		f_new_object = false;
		FREE_OBJECT(Canonical_form);
		FREE_OBJECT(OwCF);
	}

	FREE_OBJECT(NO);
	nb_input_objects++;

	if (f_v) {
		cout << "classify_using_canonical_forms::add_object done" << endl;
	}


}


int classify_using_canonical_forms::identify_object(
		any_combinatorial_object *Any_combo,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int &object_idx, uint32_t &hash, int verbose_level)
// Does not destroy Any_combo
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::identify_object" << endl;
	}

	other::l1_interfaces::nauty_output *NO;
	other::data_structures::bitvector *Canonical_form;


	int f_found;

	find_object(
			Any_combo,
			Nauty_control,
			f_found, object_idx,
			NO,
			Canonical_form,
			verbose_level);



	hash = Canonical_form->compute_hash();

	if (!f_found) {


		if (f_v) {
			cout << "classify_using_canonical_forms::identify_object "
					"we could not find the object" << endl;
		}

	}
	else {

		if (f_v) {
			cout << "classify_using_canonical_forms::identify_object "
					"the object has been found at idx = " << object_idx << " hash = " << hash << endl;
		}
		FREE_OBJECT(Canonical_form);
	}

	FREE_OBJECT(NO);
	nb_input_objects++;

	if (f_v) {
		cout << "classify_using_canonical_forms::identify_object done" << endl;
	}

	return f_found;
}




}}}}



