/*
 * classify_using_canonical_forms.cpp
 *
 *  Created on: Aug 28, 2021
 *      Author: betten
 */






#include "foundations.h"

using namespace std;

namespace orbiter {
namespace foundations {


classify_using_canonical_forms::classify_using_canonical_forms()
{
	nb_input_objects = 0;
}

classify_using_canonical_forms::~classify_using_canonical_forms()
{
}

void classify_using_canonical_forms::add_object(object_in_projective_space *OiP,
		int &f_new_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_using_canonical_forms::add_object" << endl;
	}


	int nb_rows, nb_cols;
	//int canonical_labeling_len;
	//long int *canonical_labeling;


	if (f_v) {
		cout << "classify_using_canonical_forms::add_object before OiP->encoding_size" << endl;
	}
	OiP->encoding_size(nb_rows, nb_cols, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_using_canonical_forms::add_object after OiP->encoding_size" << endl;
		cout << "classify_using_canonical_forms::add_object nb_rows=" << nb_rows << endl;
		cout << "classify_using_canonical_forms::add_object nb_cols=" << nb_cols << endl;
	}
	//canonical_labeling_len = nb_rows + nb_cols;
	//canonical_labeling = NEW_lint(canonical_labeling_len);

	nauty_output *NO;
	bitvector *Canonical_form;

	NO = NEW_OBJECT(nauty_output);
	NO->allocate(nb_rows + nb_cols, verbose_level);

	if (f_v) {
		cout << "classify_using_canonical_forms::add_object "
				"before OiP->run_nauty" << endl;
	}
	OiP->run_nauty(
			TRUE /* f_compute_canonical_form */, Canonical_form,
			//canonical_labeling, canonical_labeling_len,
			NO,
			verbose_level);
	if (f_v) {
		cout << "classify_using_canonical_forms::add_object "
				"after OiP->run_nauty" << endl;
	}

	uint32_t h;
	int f_found, idx, c;
	sorting sorting;

	h = Canonical_form->compute_hash();

	map<uint32_t, int>::iterator itr, itr1, itr2;

	itr1 = Hashing.lower_bound(h);
	itr2 = Hashing.upper_bound(h);
	f_found = FALSE;
	for (itr = itr1; itr != itr2; ++itr) {
    	idx = itr->second;
    	c = sorting.uchar_vec_compare(Canonical_form->get_data(), B[idx]->get_data(), Canonical_form->get_allocated_length());
		if (c == 0) {
			f_found = TRUE;
			break;
		}
    }
	if (!f_found) {
		f_new_object = TRUE;
		idx = B.size();
		B.push_back(Canonical_form);
		Objects.push_back(OiP);
		Ago.push_back(NO->Ago->as_lint());
		Hashing.insert(pair<uint32_t, int>(h, idx));
		input_index.push_back(nb_input_objects);
	}
	else {
		f_new_object = FALSE;
		FREE_OBJECT(Canonical_form);
		FREE_OBJECT(OiP);
	}

	FREE_OBJECT(NO);
	//FREE_lint(canonical_labeling);
	nb_input_objects++;

	if (f_v) {
		cout << "classify_using_canonical_forms::add_object done" << endl;
	}


}


}}

