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
	//int f_found = false;


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

void canonical_form_global::compute_group_and_tactical_decomposition(
		canonical_form::canonical_form_classifier *Classifier,
		canonical_form::variety_object_with_action *Input_Vo,
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition" << endl;
	}

	canonical_form::classification_of_varieties_nauty *Classification_of_varieties_nauty;

	Classification_of_varieties_nauty = NEW_OBJECT(canonical_form::classification_of_varieties_nauty);



	Classifier->Classification_of_varieties_nauty = Classification_of_varieties_nauty;

	//Classifier->Output_nauty = Nauty;


	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"before Classification_of_varieties_nauty->init" << endl;
	}
	Classification_of_varieties_nauty->init(
			Classifier->Input,
			Classifier,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"after Classification_of_varieties_nauty->init" << endl;
	}

	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"before Classification_of_varieties_nauty->classify_nauty" << endl;
	}
	Classification_of_varieties_nauty->classify_nauty(verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"after Classification_of_varieties_nauty->classify_nauty" << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"before Classification_of_varieties_nauty->report" << endl;
	}
	Classification_of_varieties_nauty->report(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"after Classification_of_varieties_nauty->report" << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"before Classification_of_varieties_nauty->write_classification_by_nauty_csv" << endl;
	}
	Classification_of_varieties_nauty->write_classification_by_nauty_csv(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"after Classification_of_varieties_nauty->write_classification_by_nauty_csv" << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"before copying stabilizer generators" << endl;
	}
	Input_Vo[0].f_has_automorphism_group = true;
	Input_Vo[0].Stab_gens = NEW_OBJECT(groups::strong_generators);

	Input_Vo[0].Stab_gens->init_copy(
			Classification_of_varieties_nauty->Canonical_forms[0]->Variety_stabilizer_compute->Stab_gens_variety,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"after copying stabilizer generators" << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"before Input_Vo[0].compute_tactical_decompositions" << endl;
	}
	Input_Vo[0].compute_tactical_decompositions(verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition "
				"after Input_Vo[0].compute_tactical_decompositions" << endl;
	}


	FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "canonical_form_global::compute_group_and_tactical_decomposition done" << endl;
	}

}

}}}



