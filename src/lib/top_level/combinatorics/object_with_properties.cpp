/*
 * object_with_properties.cpp
 *
 *  Created on: Dec 8, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


object_with_properties::object_with_properties()
{
	OwCF = NULL;

	NO = NULL;

	SG = NULL;

	A_perm = NULL;

}

object_with_properties::~object_with_properties()
{
}


void object_with_properties::init(object_with_canonical_form *OwCF,
		nauty_output *NO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::init" << endl;
	}

	object_with_properties::OwCF = OwCF;
	object_with_properties::NO = NO;

	nauty_interface_with_group Nau;

	if (f_v) {
		cout << "object_with_properties::init before Nau.automorphism_group_as_permutation_group" << endl;
	}
	Nau.automorphism_group_as_permutation_group(
					NO,
					A_perm,
					verbose_level);

	if (f_v) {
		cout << "object_with_properties::init after Nau.automorphism_group_as_permutation_group" << endl;
	}

	if (f_v) {
		cout << "object_with_properties::init "
				"A_perm:" << endl;

		A_perm->Strong_gens->print_generators_in_latex_individually(cout);
		A_perm->Strong_gens->print_generators_in_source_code();
		A_perm->print_base();
	}


	if (f_v) {
		cout << "object_with_properties::init done" << endl;
	}
}

void object_with_properties::init_object_in_projective_space(
		object_with_canonical_form *OwCF,
		nauty_output *NO,
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space" << endl;
	}

	object_with_properties::OwCF = OwCF;
	object_with_properties::NO = NO;


	nauty_interface_with_group Nau;
	action *A_linear;

	A_linear = PA->A;

	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space before Nau.reverse_engineer_linear_group_from_permutation_group" << endl;
	}
	Nau.reverse_engineer_linear_group_from_permutation_group(
			A_linear,
			PA->P,
			SG,
			A_perm,
			NO,
			verbose_level);
	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space after Nau.reverse_engineer_linear_group_from_permutation_group" << endl;
	}


	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space done" << endl;
	}

}

void object_with_properties::latex_report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::latex_report" << endl;
	}

	//action *A_perm;


	//A_perm = OwP[object_idx].A_perm;
	//A_perm = (action *) CB->Type_extra_data[object_idx];


	ost << "Generators for the automorphism group: \\\\" << endl;
	A_perm->Strong_gens->print_generators_in_latex_individually(ost);

	schreier *Sch;


	if (f_v) {
		cout << "object_with_properties::latex_report before orbits_on_points_schreier" << endl;
	}
	Sch = A_perm->Strong_gens->orbits_on_points_schreier(A_perm,
			verbose_level);
	if (f_v) {
		cout << "object_with_properties::latex_report after orbits_on_points_schreier" << endl;
	}


	ost << "Decomposition by automorphism group:\\\\" << endl;

	if (f_v) {
		cout << "object_with_properties::latex_report before Sch->print_TDA" << endl;
	}
	Sch->print_TDA(ost, OwCF, verbose_level);
	if (f_v) {
		cout << "object_with_properties::latex_report after Sch->print_TDA" << endl;
	}

	ost << "Canonical labeling:\\\\" << endl;
	encoded_combinatorial_object *Enc;

	OwCF->encode_incma(Enc, verbose_level);

	int canonical_row;
	int canonical_orbit;

	canonical_row = NO->canonical_labeling[Enc->nb_rows - 1];

	canonical_orbit = Sch->orbit_number(canonical_row);

	ost << "canonical row = " << canonical_row << "\\\\" << endl;
	ost << "canonical orbit number = " << canonical_orbit << "\\\\" << endl;

	int v = Enc->nb_rows;
	int b = Enc->nb_cols;

	std::string *point_labels;
	std::string *block_labels;


	point_labels = new string [v];
	block_labels = new string [b];

	int i, j, a;

	char str[1000];


	for (i = 0; i < v; i++) {

		a = NO->canonical_labeling[i];
		if (Sch->orbit_number(a) == canonical_orbit) {
			sprintf(str, "*%d", a);
		}
		else {
			sprintf(str, "%d", a);
		}
		point_labels[i].assign(str);
	}
	for (j = 0; j < b; j++) {
		sprintf(str, "%d", NO->canonical_labeling[v + j]);
		block_labels[j].assign(str);
	}



	Enc->latex_canonical_form_with_labels(ost, NO,
			point_labels,
			block_labels,
			verbose_level);

	delete [] point_labels;
	delete [] block_labels;

	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "object_with_properties::latex_report done" << endl;
	}

}

}}
