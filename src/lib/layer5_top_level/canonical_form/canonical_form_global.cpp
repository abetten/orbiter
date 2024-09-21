/*
 * canonical_form_global.cpp
 *
 *  Created on: Mar 29, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



canonical_form_global::canonical_form_global()
{
}

canonical_form_global::~canonical_form_global()
{
}

void canonical_form_global::compute_stabilizer_of_quartic_curve(
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_from_surface
			*Quartic_curve_from_surface,
			int f_save_nauty_input_graphs,
			automorphism_group_of_variety *&Aut_of_variety,
			int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve" << endl;
	}


	Aut_of_variety = NEW_OBJECT(automorphism_group_of_variety);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"before Aut_of_variety->init_and_compute" << endl;
	}

	string input_fname;
	int input_idx;


	input_fname = Quartic_curve_from_surface->label;
	input_idx = 0;

	Aut_of_variety->init_and_compute(
			Quartic_curve_from_surface->SOA->Surf_A->PA->PA2,
			Quartic_curve_from_surface->SOA->Surf_A->AonHPD_4_3,
			input_fname,
			input_idx,
			Quartic_curve_from_surface->curve,
			Quartic_curve_from_surface->Pts_on_curve, Quartic_curve_from_surface->sz_curve,
			f_save_nauty_input_graphs,
			verbose_level);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"after Aut_of_variety->init_and_compute" << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve" << endl;
	}
}

void canonical_form_global::find_isomorphism(
		variety_stabilizer_compute *C1,
		variety_stabilizer_compute *C,
		int *alpha, int *gamma,
		int verbose_level)
// find gamma which maps the points of C1 to the points of C.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_global::find_isomorphism" << endl;
	}

	int *alpha_inv;
	int *beta_inv;
	int i, j;
	int f_found = false;


	int canonical_labeling_len;

	canonical_labeling_len = C1->NO->N;
	alpha_inv = C1->NO->canonical_labeling;
	//alpha_inv = C1->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_global::find_isomorphism "
				"alpha_inv = " << endl;
		Int_vec_print(cout,
				alpha_inv,
				canonical_labeling_len);
		cout << endl;
	}

	beta_inv = C->NO->canonical_labeling;
	//beta_inv = C->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_global::find_isomorphism "
				"beta_inv = " << endl;
		Int_vec_print(
				cout,
				beta_inv,
				canonical_labeling_len);
		cout << endl;
	}

	// compute gamma = alpha * beta^-1 (left to right multiplication),
	// which maps the points on curve C1 to the points on curve C


	if (f_v) {
		cout << "canonical_form_global::find_isomorphism "
				"computing alpha" << endl;
	}
	for (i = 0; i < canonical_labeling_len; i++) {
		j = alpha_inv[i];
		alpha[j] = i;
	}

	if (f_v) {
		cout << "canonical_form_global::find_isomorphism "
				"computing gamma" << endl;
	}
	for (i = 0; i < canonical_labeling_len; i++) {
		gamma[i] = beta_inv[alpha[i]];
	}
	if (f_v) {
		cout << "canonical_form_global::find_isomorphism "
				"gamma = " << endl;
		Int_vec_print(
				cout,
				gamma,
				canonical_labeling_len);
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::find_isomorphism done" << endl;
	}
}


}}}



