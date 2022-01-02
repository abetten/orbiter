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

	TDO = NULL;

	Flags = NULL;
	Anti_Flags = NULL;

}

object_with_properties::~object_with_properties()
{

	if (TDO) {
		FREE_OBJECT(TDO);
	}
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
		int max_TDO_depth,
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
					verbose_level - 2);

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
		if (f_v) {
			cout << "object_with_properties::init "
					"before lift_generators_to_matrix_group" << endl;
		}

		lift_generators_to_matrix_group(verbose_level - 2);

		if (f_v) {
			cout << "object_with_properties::init "
					"after lift_generators_to_matrix_group" << endl;
		}
	}

	if (f_v) {
		cout << "object_with_properties::init "
				"before compute_flag_orbits" << endl;
	}
	compute_flag_orbits(verbose_level - 2);
	if (f_v) {
		cout << "object_with_properties::init "
				"after compute_flag_orbits" << endl;
	}

	if (f_v) {
		cout << "object_with_properties::init "
				"before compute_TDO" << endl;
	}
	compute_TDO(max_TDO_depth, verbose_level - 2);
	if (f_v) {
		cout << "object_with_properties::init "
				"after compute_TDO" << endl;
	}

	if (f_v) {
		cout << "object_with_properties::init done" << endl;
	}
}

void object_with_properties::compute_flag_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::compute_flag_orbits" << endl;
	}

	Flags = NEW_OBJECT(flag_orbits_incidence_structure);
	Anti_Flags = NEW_OBJECT(flag_orbits_incidence_structure);

	if (f_v) {
		cout << "object_with_properties::compute_flag_orbits "
				"before Flags->init" << endl;
	}
	Flags->init(this, FALSE, A_perm, A_perm->Strong_gens, verbose_level - 2);
	if (f_v) {
		cout << "object_with_properties::compute_flag_orbits "
				"after Flags->init" << endl;
	}

	if (f_v) {
		cout << "object_with_properties::compute_flag_orbits "
				"before Anti_Flags->init" << endl;
	}
	Anti_Flags->init(this, TRUE, A_perm, A_perm->Strong_gens, verbose_level - 2);
	if (f_v) {
		cout << "object_with_properties::compute_flag_orbits "
				"after Anti_Flags->init" << endl;
	}

	if (f_v) {
		cout << "object_with_properties::compute_flag_orbits done" << endl;
	}
}

void object_with_properties::lift_generators_to_matrix_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::lift_generators_to_matrix_group" << endl;
	}
	//strong_generators *SG;
	action *A_perm;

	nauty_interface_with_group Naug;

	if (f_v) {
		cout << "object_with_properties::lift_generators_to_matrix_group "
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
		cout << "object_with_properties::lift_generators_to_matrix_group "
				"after Naug.reverse_engineer_linear_group_from_permutation_group" << endl;
	}


	FREE_OBJECT(A_perm);

	if (f_v) {
		cout << "object_with_properties::lift_generators_to_matrix_group done" << endl;
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
		classification_of_objects_report_options *Report_options,
		int verbose_level)
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

	if (f_projective_space) {
		ost << "Generators for the automorphism group as matrix group: \\\\" << endl;
		SG->print_generators_in_latex_individually(ost);
	}

	if (Report_options->f_export_group) {

		std::string fname;
		std::string label_txt;
		std::string label_tex;


		fname.assign(label);
		fname.append("_aut.makefile");
		label_txt.assign(label);
		label_txt.append("_aut");
		label_tex.assign(label);
		label_tex.append("\\_aut");

		A_perm->Strong_gens->export_to_orbiter_as_bsgs(
				A_perm,
				fname, label, label_tex,
				verbose_level);
	}




	schreier *Sch;


	if (f_v) {
		cout << "object_with_properties::latex_report before orbits_on_points_schreier" << endl;
	}
	Sch = A_perm->Strong_gens->orbits_on_points_schreier(A_perm,
			verbose_level);
	if (f_v) {
		cout << "object_with_properties::latex_report after orbits_on_points_schreier" << endl;
	}


	if (Report_options->f_export_flag_orbits) {
		export_INP_with_flag_orbits(ost,
				Sch,
				verbose_level);
		export_TDA_with_flag_orbits(ost,
				Sch,
				verbose_level);
	}

	if (Report_options->f_show_TDO) {

		ost << "Decomposition by combinatorial refinement:\\\\" << endl;

		if (f_v) {
			cout << "object_with_properties::latex_report before Sch->print_TDA" << endl;
		}
		print_TDO(ost, Report_options);
		if (f_v) {
			cout << "object_with_properties::latex_report after Sch->print_TDA" << endl;
		}
	}

	if (Report_options->f_show_TDA) {

		ost << "Decomposition by automorphism group:\\\\" << endl;

		if (f_v) {
			cout << "object_with_properties::latex_report before Sch->print_TDA" << endl;
		}
		Sch->print_TDA(ost, OwCF, Report_options, verbose_level);
		if (f_v) {
			cout << "object_with_properties::latex_report after Sch->print_TDA" << endl;
		}
	}

	ost << "Canonical labeling:\\\\" << endl;
	encoded_combinatorial_object *Enc;
	encoded_combinatorial_object *Enc2;

	OwCF->encode_incma(Enc, verbose_level);


	Enc2 = NEW_OBJECT(encoded_combinatorial_object);

	Enc2->init_canonical_form(Enc, NO, verbose_level);



	int canonical_row;
	int canonical_orbit;

	canonical_row = NO->canonical_labeling[Enc->nb_rows - 1];

	canonical_orbit = Sch->orbit_number(canonical_row);

	ost << "canonical row = " << canonical_row << "\\\\" << endl;
	ost << "canonical orbit number = " << canonical_orbit << "\\\\" << endl;

	Enc2->latex_set_system_by_rows(ost, verbose_level);

	FREE_OBJECT(Enc2);

	if (Report_options->f_show_incidence_matrices) {

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

	ost << "Flag orbits:\\\\" << endl;
	Flags->report(ost, verbose_level);

	ost << "Anti-Flag orbits:\\\\" << endl;
	Anti_Flags->report(ost, verbose_level);


	if (Report_options->f_lex_least) {

		if (f_v) {
			cout << "object_with_properties::latex_report f_lex_least" << endl;
		}
		int idx;

		idx = Orbiter->find_symbol(Report_options->lex_least_geometry_builder);

		symbol_table_object_type t;

		t = Orbiter->get_object_type(idx);
		if (t != t_geometry_builder) {
			cout << "object_with_properties::latex_report "
				<< Report_options->lex_least_geometry_builder
				<< " is not of type geometry_builder" << endl;
			exit(1);
		}

		geometry_builder *GB;
		int f_found;
		nauty_output *NO;
		bitvector *Canonical_form;

		GB = (geometry_builder *) Orbiter->get_object(idx);


		if (f_v) {
			cout << "object_with_properties::latex_report before find_object, "
					"OwCF->v=" << OwCF->v << endl;
		}

		GB->gg->inc->iso_type_at_line[OwCF->v - 1]->Canonical_forms->find_object(
				OwCF,
				f_found, idx,
				NO,
				Canonical_form,
				verbose_level);

		if (f_v) {
			cout << "object_with_properties::latex_report after find_object" << endl;
		}

		// if f_found is TRUE, B[idx] agrees with the given object


		if (!f_found) {
			cout << "object_with_properties::latex_report "
					"cannot find object in geometry_builder" << endl;
			exit(1);
		}

		object_with_canonical_form *OwCF2 = (object_with_canonical_form *)
				GB->gg->inc->iso_type_at_line[OwCF->v - 1]->Canonical_forms->Objects[idx];

		if (f_v) {
			cout << "object_with_properties::latex_report before FREE_OBJECT(NO)" << endl;
		}
		FREE_OBJECT(NO);
		if (f_v) {
			cout << "object_with_properties::latex_report after FREE_OBJECT(NO)" << endl;
		}
		FREE_OBJECT(Canonical_form);
		if (f_v) {
			cout << "object_with_properties::latex_report after FREE_OBJECT(Canonical_form)" << endl;
		}

		ost << "Is isomorphic to object " << idx << " in the list:\\\\" << endl;
		ost << "Lex-least form is:\\\\" << endl;

		OwCF2->print_tex_detailed(ost,
				Report_options->f_show_incidence_matrices,
				verbose_level);
	}


	if (f_v) {
		cout << "object_with_properties::latex_report done" << endl;
	}

}

void object_with_properties::compute_TDO(int max_TDO_depth, int verbose_level)
{
	encoded_combinatorial_object *Enc;

	OwCF->encode_incma(Enc, verbose_level);



	TDO = NEW_OBJECT(tdo_scheme_compute);

	TDO->init(Enc, max_TDO_depth, verbose_level);


	//latex_TDA(ost, Enc, verbose_level);

	FREE_OBJECT(Enc);

}

void object_with_properties::print_TDO(std::ostream &ost,
		classification_of_objects_report_options *Report_options)
{

	TDO->print_schemes(ost);

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
	int *Inc_flag_orbits;
	int i0, j0;
	int i, j;
	int idx;
	int nb_orbits_on_flags;
	int orbit_idx;

	Inc2 = NEW_int(Enc->nb_rows * Enc->nb_cols);
	Inc_flag_orbits = NEW_int(Enc->nb_rows * Enc->nb_cols);
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
				Inc2[i * Enc->nb_cols + j] = 1;
				Inc_flag_orbits[i * Enc->nb_cols + j] = orbit_idx + 1;
			}
			else {
				idx = Anti_Flags->find_flag(i0, j0 + Enc->nb_rows);
				orbit_idx = Anti_Flags->Orb->Sch->orbit_number(idx);
				Inc2[i * Enc->nb_cols + j] = 0;
				Inc_flag_orbits[i * Enc->nb_cols + j] = nb_orbits_on_flags + orbit_idx + 1;
			}
		}
	}

	file_io Fio;
	string fname;

	fname.assign(label);
	fname.append("_TDA.csv");

	Fio.int_matrix_write_csv(fname, Inc2, Enc->nb_rows, Enc->nb_cols);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	fname.assign(label);
	fname.append("_TDA_flag_orbits.csv");

	Fio.int_matrix_write_csv(fname, Inc_flag_orbits, Enc->nb_rows, Enc->nb_cols);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(Inc2);
	FREE_int(Inc_flag_orbits);
	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits done" << endl;
	}
}

void object_with_properties::export_INP_with_flag_orbits(std::ostream &ost,
		schreier *Sch,
		int verbose_level)
// INP = input geometry
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::export_INP_with_flag_orbits" << endl;
	}

	encoded_combinatorial_object *Enc;

	OwCF->encode_incma(Enc, verbose_level);


	int *Inc2;
	int *Inc_flag_orbits;
	int i0, j0;
	int i, j;
	int idx;
	int nb_orbits_on_flags;
	int orbit_idx;

	Inc2 = NEW_int(Enc->nb_rows * Enc->nb_cols);
	Inc_flag_orbits = NEW_int(Enc->nb_rows * Enc->nb_cols);
	Orbiter->Int_vec.zero(Inc2, Enc->nb_rows * Enc->nb_cols);

	nb_orbits_on_flags = Flags->Orb->Sch->nb_orbits;


	// +1 avoids the color white

	for (i = 0; i < Enc->nb_rows; i++) {
		i0 = i;
		for (j = 0; j < Enc->nb_cols; j++) {
			j0 = j;
			if (Enc->Incma[i0 * Enc->nb_cols + j0]) {
				idx = Flags->find_flag(i0, j0 + Enc->nb_rows);
				orbit_idx = Flags->Orb->Sch->orbit_number(idx);
				Inc2[i * Enc->nb_cols + j] = 1;
				Inc_flag_orbits[i * Enc->nb_cols + j] = orbit_idx + 1;
			}
			else {
				idx = Anti_Flags->find_flag(i0, j0 + Enc->nb_rows);
				orbit_idx = Anti_Flags->Orb->Sch->orbit_number(idx);
				Inc2[i * Enc->nb_cols + j] = 0;
				Inc_flag_orbits[i * Enc->nb_cols + j] = nb_orbits_on_flags + orbit_idx + 1;
			}
		}
	}

	file_io Fio;
	string fname;

	fname.assign(label);
	fname.append("_INP.csv");

	Fio.int_matrix_write_csv(fname, Inc2, Enc->nb_rows, Enc->nb_cols);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	fname.assign(label);
	fname.append("_INP_flag_orbits.csv");

	Fio.int_matrix_write_csv(fname, Inc_flag_orbits, Enc->nb_rows, Enc->nb_cols);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(Inc2);
	FREE_int(Inc_flag_orbits);
	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "object_with_properties::export_INP_with_flag_orbits done" << endl;
	}
}


}}
