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


}}}}



