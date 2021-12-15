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

	//std::string label;

	NO = NULL;

	f_projective_space = FALSE;
	PA = NULL;

	SG = NULL;

	A_perm = NULL;

	Flags = NULL;
	Anti_Flags = NULL;

}

object_with_properties::~object_with_properties()
{

	if (Flags) {
		FREE_OBJECT(Flags);
	}
	if (Anti_Flags) {
		FREE_OBJECT(Anti_Flags);
	}
}


void object_with_properties::init(object_with_canonical_form *OwCF,
		nauty_output *NO,
		int f_projective_space, projective_space_with_action *PA,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::init" << endl;
	}

	object_with_properties::OwCF = OwCF;
	object_with_properties::NO = NO;
	object_with_properties::f_projective_space = f_projective_space;
	object_with_properties::PA = PA;
	object_with_properties::label.assign(label);

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

	if (f_projective_space) {
		//strong_generators *SG;
		action *A_perm;

		nauty_interface_with_group Naug;

		if (f_v) {
			cout << "object_with_properties::init "
					"before Naug.reverse_engineer_linear_group_from_permutation_group" << endl;
		}

		Naug.reverse_engineer_linear_group_from_permutation_group(
				PA->A /* A_linear */,
				PA->P,
				SG,
				A_perm,
				NO,
				verbose_level);

		if (f_v) {
			cout << "object_with_properties::init "
					"after Naug.reverse_engineer_linear_group_from_permutation_group" << endl;
		}


		FREE_OBJECT(A_perm);
	}
	else {

		Flags = NEW_OBJECT(flag_orbits_incidence_structure);
		Anti_Flags = NEW_OBJECT(flag_orbits_incidence_structure);

		if (f_v) {
			cout << "object_with_properties::init "
					"before Flags->init" << endl;
		}
		Flags->init(this, FALSE, A_perm, A_perm->Strong_gens, verbose_level);
		if (f_v) {
			cout << "object_with_properties::init "
					"after Flags->init" << endl;
		}

		if (f_v) {
			cout << "object_with_properties::init "
					"before Anti_Flags->init" << endl;
		}
		Anti_Flags->init(this, TRUE, A_perm, A_perm->Strong_gens, verbose_level);
		if (f_v) {
			cout << "object_with_properties::init "
					"after Anti_Flags->init" << endl;
		}

	}

	if (f_v) {
		cout << "object_with_properties::init done" << endl;
	}
}

void object_with_properties::init_object_in_projective_space(
		object_with_canonical_form *OwCF,
		nauty_output *NO,
		projective_space_with_action *PA,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space" << endl;
	}

	object_with_properties::OwCF = OwCF;
	object_with_properties::NO = NO;
	object_with_properties::label.assign(label);


	nauty_interface_with_group Nau;
	action *A_linear;

	A_linear = PA->A;

	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space "
				"before Nau.reverse_engineer_linear_group_from_permutation_group" << endl;
	}
	Nau.reverse_engineer_linear_group_from_permutation_group(
			A_linear,
			PA->P,
			SG,
			A_perm,
			NO,
			verbose_level);
	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space "
				"after Nau.reverse_engineer_linear_group_from_permutation_group" << endl;
	}


	if (f_v) {
		cout << "object_with_properties::init_object_in_projective_space done" << endl;
	}

}

void object_with_properties::latex_report(std::ostream &ost,
		int f_show_incma, int verbose_level)
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
	Sch->print_TDA(ost, OwCF, f_show_incma, verbose_level);
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


	if (f_show_incma) {

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
	}

	if (!f_projective_space) {

		ost << "Flag orbits:\\\\" << endl;
		Flags->report(ost, verbose_level);

		ost << "Anti-Flag orbits:\\\\" << endl;
		Anti_Flags->report(ost, verbose_level);

		export_TDA_with_flag_orbits(ost,
				Sch,
				verbose_level);
	}
	if (f_v) {
		cout << "object_with_properties::latex_report done" << endl;
	}

}

void object_with_properties::export_TDA_with_flag_orbits(std::ostream &ost,
		schreier *Sch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits" << endl;
	}

	encoded_combinatorial_object *Enc;

	OwCF->encode_incma(Enc, verbose_level);


	int *Inc2;
	int i0, j0;
	int i, j;
	int idx;
	int nb_orbits_on_flags;
	int orbit_idx;

	Inc2 = NEW_int(Enc->nb_rows * Enc->nb_cols);
	Orbiter->Int_vec.zero(Inc2, Enc->nb_rows * Enc->nb_cols);

	nb_orbits_on_flags = Flags->Orb->Sch->nb_orbits;


	// +1 avoids the color white

	for (i = 0; i < Enc->nb_rows; i++) {
		i0 = Sch->orbit[i];
		for (j = 0; j < Enc->nb_cols; j++) {
			j0 = Sch->orbit[Enc->nb_rows + j] - Enc->nb_rows;
			if (Enc->Incma[i0 * Enc->nb_cols + j0]) {
				idx = Flags->find_flag(i0, j0 + Enc->nb_rows);
				orbit_idx = Flags->Orb->Sch->orbit_number(idx);
				Inc2[i * Enc->nb_cols + j] = orbit_idx + 1;
			}
			else {
				idx = Anti_Flags->find_flag(i0, j0 + Enc->nb_rows);
				orbit_idx = Anti_Flags->Orb->Sch->orbit_number(idx);
				Inc2[i * Enc->nb_cols + j] = nb_orbits_on_flags + orbit_idx + 1;
			}
		}
	}

	file_io Fio;
	string fname;

	fname.assign(label);
	fname.append("_TDA_flag_orbits.csv");

	Fio.int_matrix_write_csv(fname, Inc2, Enc->nb_rows, Enc->nb_cols);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(Inc2);
	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits done" << endl;
	}
}


}}
