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
namespace data_structures {


classify_using_canonical_forms::classify_using_canonical_forms()
{
	nb_input_objects = 0;
}

classify_using_canonical_forms::~classify_using_canonical_forms()
{
}

void classify_using_canonical_forms::orderly_test(
		geometry::object_with_canonical_form *OwCF,
		int &f_accept, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test" << endl;
	}

	nauty_output *NO;


	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"before OwCF->run_nauty_basic" << endl;
	}
	OwCF->run_nauty_basic(
			NO,
			verbose_level);
	if (f_v) {
		cout << "classify_using_canonical_forms::orderly_test "
				"after OwCF->run_nauty_basic" << endl;
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
		geometry::object_with_canonical_form *OwCF,
		int &f_found, int &idx,
		nauty_output *&NO,
		bitvector *&Canonical_form,
		int verbose_level)
// if f_found is true, B[idx] agrees with the given object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::find_object" << endl;
	}

	if (f_v) {
		cout << "classify_using_canonical_forms::find_object "
				"before OwCF->run_nauty" << endl;
	}
	OwCF->run_nauty(
			true /* f_compute_canonical_form */, Canonical_form,
			NO,
			verbose_level);
	if (f_v) {
		cout << "classify_using_canonical_forms::find_object "
				"after OwCF->run_nauty" << endl;
	}

	uint32_t h;
	int c;
	sorting sorting;

	h = Canonical_form->compute_hash();

	map<uint32_t, int>::iterator itr, itr1, itr2;

	itr1 = Hashing.lower_bound(h);
	itr2 = Hashing.upper_bound(h);
	f_found = false;
	for (itr = itr1; itr != itr2; ++itr) {
    	idx = itr->second;
    	c = sorting.uchar_vec_compare(
    			Canonical_form->get_data(),
    			B[idx]->get_data(),
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
		geometry::object_with_canonical_form *OwCF,
		int &f_new_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::add_object" << endl;
	}

	nauty_output *NO;
	bitvector *Canonical_form;

#if 0
	if (f_v) {
		cout << "classify_using_canonical_forms::add_object "
				"before OwCF->run_nauty" << endl;
	}
	OwCF->run_nauty(
			true /* f_compute_canonical_form */, Canonical_form,
			NO,
			verbose_level);
	if (f_v) {
		cout << "classify_using_canonical_forms::add_object "
				"after OwCF->run_nauty" << endl;
	}

	uint32_t h;
	int f_found, c;
	sorting sorting;

	h = Canonical_form->compute_hash();

	map<uint32_t, int>::iterator itr, itr1, itr2;

	itr1 = Hashing.lower_bound(h);
	itr2 = Hashing.upper_bound(h);
	f_found = false;
	for (itr = itr1; itr != itr2; ++itr) {
    	idx = itr->second;
    	c = sorting.uchar_vec_compare(
    			Canonical_form->get_data(),
    			B[idx]->get_data(),
				Canonical_form->get_allocated_length());
		if (c == 0) {
			f_found = true;
			break;
		}
    }
#else

	int f_found;
	int idx;

	find_object(OwCF,
			f_found, idx,
			NO,
			Canonical_form,
			verbose_level);

#endif

	uint32_t h;

	h = Canonical_form->compute_hash();

	if (!f_found) {
		f_new_object = true;
		idx = B.size();
		B.push_back(Canonical_form);
		Objects.push_back(OwCF);
		Ago.push_back(NO->Ago->as_lint());
		Hashing.insert(pair<uint32_t, int>(h, idx));
		input_index.push_back(nb_input_objects);
	}
	else {
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


}}}


