/*
 * object_with_properties.cpp
 *
 *  Created on: Dec 8, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


object_with_properties::object_with_properties()
{
	OwCF = NULL;

	//std::string label;

	NO = NULL;

	f_projective_space = false;
	PA = NULL;

	SG = NULL;

	A_perm = NULL;

	TDO = NULL;

	GA_on_CO = NULL;

	//Flags = NULL;
	//Anti_Flags = NULL;

}

object_with_properties::~object_with_properties()
{

	if (TDO) {
		FREE_OBJECT(TDO);
	}
	if (GA_on_CO) {
		FREE_OBJECT(GA_on_CO);
	}
#if 0
	if (Flags) {
		FREE_OBJECT(Flags);
	}
	if (Anti_Flags) {
		FREE_OBJECT(Anti_Flags);
	}
#endif
}


void object_with_properties::init(
		geometry::object_with_canonical_form *OwCF,
		l1_interfaces::nauty_output *NO,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
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

	interfaces::nauty_interface_with_group Nau;

	if (f_v) {
		cout << "object_with_properties::init "
				"before Nau.automorphism_group_as_permutation_group" << endl;
	}
	Nau.automorphism_group_as_permutation_group(
					NO,
					A_perm,
					verbose_level - 2);

	if (f_v) {
		cout << "object_with_properties::init "
				"after Nau.automorphism_group_as_permutation_group" << endl;
	}

	if (false) {
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


	GA_on_CO = NEW_OBJECT(data_structures_groups::group_action_on_combinatorial_object);


	if (f_v) {
		cout << "object_with_properties::init "
				"before GA_on_CO->init" << endl;
	}
	GA_on_CO->init(
			label,
			label,
			OwCF,
			A_perm,
			verbose_level);
	if (f_v) {
		cout << "object_with_properties::init "
				"after GA_on_CO->init" << endl;
	}


#if 0
	if (f_v) {
		cout << "object_with_properties::init "
				"before compute_flag_orbits" << endl;
	}
	compute_flag_orbits(verbose_level - 2);
	if (f_v) {
		cout << "object_with_properties::init "
				"after compute_flag_orbits" << endl;
	}
#endif

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


void object_with_properties::lift_generators_to_matrix_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::lift_generators_to_matrix_group" << endl;
	}
	//strong_generators *SG;
	actions::action *A_perm;

	interfaces::nauty_interface_with_group Naug;

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
		geometry::object_with_canonical_form *OwCF,
		l1_interfaces::nauty_output *NO,
		projective_geometry::projective_space_with_action *PA,
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


	interfaces::nauty_interface_with_group Nau;
	actions::action *A_linear;

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

void object_with_properties::latex_report(
		std::ostream &ost,
		combinatorics::classification_of_objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::latex_report" << endl;
	}

	ost << "Generators for the automorphism group: \\\\" << endl;
	if (A_perm->degree < 100) {
		A_perm->Strong_gens->print_generators_in_latex_individually(ost);

		ost << "\\begin{verbatim}" << endl;
		A_perm->Strong_gens->print_generators_gap(ost);
		ost << "\\end{verbatim}" << endl;

	}
	else {
		ost << "permutation degree is too large to print. \\\\" << endl;

	}

	if (f_projective_space) {

		ost << "Generators for the automorphism group as matrix group: \\\\" << endl;

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before SG->print_generators_in_latex_individually" << endl;
		}
		SG->print_generators_in_latex_individually(ost);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after SG->print_generators_in_latex_individually" << endl;
		}
	}


	if (Report_options->f_export_group_orbiter) {

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"f_export_group_orbiter" << endl;
		}

		std::string fname;
		std::string label_txt;
		std::string label_tex;


		fname = label + "_aut.makefile";
		label_txt = label + "_aut";
		label_tex = label + "\\_aut";

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before A_perm->Strong_gens->export_to_orbiter_as_bsgs" << endl;
		}
		A_perm->Strong_gens->export_to_orbiter_as_bsgs(
				A_perm,
				fname, label, label_tex,
				verbose_level);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after A_perm->Strong_gens->export_to_orbiter_as_bsgs" << endl;
		}
	}

	if (Report_options->f_export_group_GAP) {

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"f_export_group_GAP" << endl;
		}

		std::string fname;


		fname = label + "_aut.gap";

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before A_perm->Strong_gens->export_permutation_group_to_GAP" << endl;
		}
		A_perm->Strong_gens->export_permutation_group_to_GAP(fname,
				A_perm, verbose_level);


		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after A_perm->Strong_gens->export_permutation_group_to_GAP" << endl;
		}
	}



#if 1
	groups::schreier *Sch;


	if (f_v) {
		cout << "object_with_properties::latex_report "
				"before compute_all_point_orbits_schreier" << endl;
	}
	Sch = A_perm->Strong_gens->compute_all_point_orbits_schreier(A_perm,
			verbose_level);
	if (f_v) {
		cout << "object_with_properties::latex_report "
				"after compute_all_point_orbits_schreier" << endl;
	}
#endif


	if (Report_options->f_export_flag_orbits) {

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"f_export_flag_orbits" << endl;
		}

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before GA_on_CO->export_INP_with_flag_orbits" << endl;
		}
		GA_on_CO->export_INP_with_flag_orbits(ost,
				//Sch,
				verbose_level);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after GA_on_CO->export_INP_with_flag_orbits" << endl;
		}

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before GA_on_CO->export_TDA_with_flag_orbits" << endl;
		}
		GA_on_CO->export_TDA_with_flag_orbits(ost,
				//Sch,
				verbose_level);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after GA_on_CO->export_TDA_with_flag_orbits" << endl;
		}
	}

	if (Report_options->f_show_TDO) {

		ost << "Decomposition by combinatorial refinement:\\\\" << endl;

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before Sch->print_TDO" << endl;
		}
		print_TDO(ost, Report_options);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after Sch->print_TDO" << endl;
		}
	}

	if (Report_options->f_show_TDA) {

		ost << "Decomposition by automorphism group:\\\\" << endl;

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before Sch->print_TDA" << endl;
		}
		Sch->print_TDA(ost, OwCF, Report_options, verbose_level);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after Sch->print_TDA" << endl;
		}
	}
	if (Report_options->f_export_labels) {

		combinatorics::encoded_combinatorial_object *Enc;

		OwCF->encode_incma(Enc, verbose_level);

		//latex_TDA(ost, Enc, verbose_level);
		//ost << "\\\\" << endl;

		int *point_labels;
		int *block_labels;

		Enc->compute_labels(
				Sch->nb_orbits, Sch->orbit_first, Sch->orbit_len, Sch->orbit,
				point_labels, block_labels,
				verbose_level);

		orbiter_kernel_system::file_io Fio;

		string fname;

		fname = "point_labels.csv";
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, point_labels, Enc->nb_rows, 1);

		cout << "object_with_properties::latex_report "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		fname = "block_labels.csv";
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, block_labels, Enc->nb_cols, 1);

		cout << "object_with_properties::latex_report "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		FREE_int(point_labels);
		FREE_int(block_labels);

		FREE_OBJECT(Enc);
	}

	ost << "Canonical labeling:\\\\" << endl;
	combinatorics::encoded_combinatorial_object *Enc;
	combinatorics::encoded_combinatorial_object *Enc2;

	if (f_v) {
		cout << "object_with_properties::latex_report "
				"before OwCF->encode_incma" << endl;
	}
	OwCF->encode_incma(Enc, verbose_level);
	if (f_v) {
		cout << "object_with_properties::latex_report "
				"after OwCF->encode_incma" << endl;
	}


	Enc2 = NEW_OBJECT(combinatorics::encoded_combinatorial_object);

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


		if (f_v) {
			cout << "object_with_properties::latex_report "
					"f_show_incidence_matrices" << endl;
		}

		int v = Enc->nb_rows;
		int b = Enc->nb_cols;

		std::string *point_labels;
		std::string *block_labels;


		point_labels = new string [v];
		block_labels = new string [b];

		int i, j, a;

		for (i = 0; i < v; i++) {

			a = NO->canonical_labeling[i];
			if (Sch->orbit_number(a) == canonical_orbit) {
				point_labels[i] = "*" + std::to_string(a);
			}
			else {
				point_labels[i] = std::to_string(a);
			}
		}
		for (j = 0; j < b; j++) {
			block_labels[j] = std::to_string(NO->canonical_labeling[v + j]);
		}

		Enc->latex_canonical_form_with_labels(ost, NO,
				point_labels,
				block_labels,
				verbose_level);

		delete [] point_labels;
		delete [] block_labels;

		FREE_OBJECT(Enc);
	}

	GA_on_CO->report_flag_orbits(
			ost, verbose_level);



	if (Report_options->f_lex_least) {

		if (f_v) {
			cout << "object_with_properties::latex_report f_lex_least" << endl;
		}
		int idx;

		idx = orbiter_kernel_system::Orbiter->find_symbol(Report_options->lex_least_geometry_builder);

		layer1_foundations::orbiter_kernel_system::symbol_table_object_type t;

		t = orbiter_kernel_system::Orbiter->get_object_type(idx);
		if (t != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_geometry_builder) {
			cout << "object_with_properties::latex_report "
				<< Report_options->lex_least_geometry_builder
				<< " is not of type geometry_builder" << endl;
			exit(1);
		}

		geometry_builder::geometry_builder *GB;
		int f_found;
		l1_interfaces::nauty_output *NO;
		data_structures::bitvector *Canonical_form;

		GB = (geometry_builder::geometry_builder *)
				orbiter_kernel_system::Orbiter->get_object(idx);


		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before find_object, "
					"OwCF->v=" << OwCF->v << endl;
		}

		GB->gg->inc->iso_type_at_line[OwCF->v - 1]->Canonical_forms->find_object(
				OwCF,
				f_found, idx,
				NO,
				Canonical_form,
				verbose_level);

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after find_object" << endl;
		}

		// if f_found is true, B[idx] agrees with the given object


		if (!f_found) {
			cout << "object_with_properties::latex_report "
					"cannot find object in geometry_builder" << endl;
			exit(1);
		}

		geometry::object_with_canonical_form *OwCF2 =
				(geometry::object_with_canonical_form *)
				GB->gg->inc->iso_type_at_line[OwCF->v - 1]->Canonical_forms->Objects[idx];

		if (f_v) {
			cout << "object_with_properties::latex_report "
					"before FREE_OBJECT(NO)" << endl;
		}
		FREE_OBJECT(NO);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after FREE_OBJECT(NO)" << endl;
		}
		FREE_OBJECT(Canonical_form);
		if (f_v) {
			cout << "object_with_properties::latex_report "
					"after FREE_OBJECT(Canonical_form)" << endl;
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

void object_with_properties::compute_TDO(
		int max_TDO_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::compute_TDO" << endl;
	}
	combinatorics::encoded_combinatorial_object *Enc;

	OwCF->encode_incma(Enc, verbose_level);



	TDO = NEW_OBJECT(combinatorics::tdo_scheme_compute);

	TDO->init(Enc, max_TDO_depth, verbose_level);


	//latex_TDA(ost, Enc, verbose_level);

	FREE_OBJECT(Enc);
	if (f_v) {
		cout << "object_with_properties::compute_TDO done" << endl;
	}

}

void object_with_properties::print_TDO(
		std::ostream &ost,
		combinatorics::classification_of_objects_report_options
			*Report_options)
{

	TDO->print_schemes(ost);

}

#if 0
void object_with_properties::export_TDA_with_flag_orbits(
		std::ostream &ost,
		groups::schreier *Sch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits" << endl;
	}

	combinatorics::encoded_combinatorial_object *Enc;

	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits "
				"before OwCF->encode_incma" << endl;
	}
	OwCF->encode_incma(Enc, 0 /*verbose_level*/);
	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits "
				"after OwCF->encode_incma" << endl;
	}



	{
		if (f_v) {
			cout << "object_with_properties::export_TDA_with_flag_orbits Enc->nb_rows = " << Enc->nb_rows << endl;
			cout << "object_with_properties::export_TDA_with_flag_orbits Enc->nb_cols = " << Enc->nb_cols << endl;
		}

		actions::action *A_on_points;
		actions::action *A_on_lines;

		long int *points;
		long int *lines;

		points = NEW_lint(Enc->nb_rows);
		lines = NEW_lint(Enc->nb_cols);

		int i;

		for (i = 0; i < Enc->nb_rows; i++) {
			points[i] = i;
		}
		for (i = 0; i < Enc->nb_cols; i++) {
			lines[i] = Enc->nb_rows + i;
		}

		std::string label1, label2;

		label1 = "points";
		label2 = "lines";


		A_on_points = A_perm->Induced_action->restricted_action(
				points, Enc->nb_rows,
				label1,
				verbose_level);

		A_on_lines = A_perm->Induced_action->restricted_action(
				lines, Enc->nb_cols,
				label2,
				verbose_level);

		actions::action_global Action_global;

		Action_global.report_TDA_combinatorial_object(
				ost,
				Enc,
				A_on_points, A_on_lines,
				A_perm->Strong_gens, 25 /* size_limit_for_printing */,
				verbose_level);


		FREE_lint(points);
		FREE_lint(lines);
		FREE_OBJECT(A_on_points);
		FREE_OBJECT(A_on_lines);
	}

#if 0
	orbiter_kernel_system::file_io Fio;
	string fname;


	int i0, j0;
	int i, j;


	if (Flags->f_flag_orbits_have_been_computed) {

		if (f_v) {
			cout << "object_with_properties::export_TDA_with_flag_orbits "
					"f_flag_orbits_have_been_computed" << endl;
		}

		int *Inc_flag_orbits;
		int *Inc_TDA;
		int nb_orbits_on_flags;
		int idx;
		int orbit_idx;

		Inc_flag_orbits = NEW_int(Enc->nb_rows * Enc->nb_cols);
		Inc_TDA = NEW_int(Enc->nb_rows * Enc->nb_cols);
		nb_orbits_on_flags = Flags->Orb->Sch->nb_orbits;
		for (i = 0; i < Enc->nb_rows; i++) {
			i0 = Sch->orbit[i];
			for (j = 0; j < Enc->nb_cols; j++) {
				j0 = Sch->orbit[Enc->nb_rows + j] - Enc->nb_rows;
				if (Enc->get_incidence_ij(i0, j0)) {
					idx = Flags->find_flag(i0, j0 + Enc->nb_rows);
					orbit_idx = Flags->Orb->Sch->orbit_number(idx);
					Inc_flag_orbits[i * Enc->nb_cols + j] = orbit_idx + 1;
					Inc_TDA[i * Enc->nb_cols + j] = 1;
				}
				else {
					idx = Anti_Flags->find_flag(i0, j0 + Enc->nb_rows);
					orbit_idx = Anti_Flags->Orb->Sch->orbit_number(idx);
					Inc_flag_orbits[i * Enc->nb_cols + j] = nb_orbits_on_flags + orbit_idx + 1;
					Inc_TDA[i * Enc->nb_cols + j] = 0;
				}
			}
		}

		fname = label + "_TDA.csv";

		Fio.int_matrix_write_csv(fname, Inc_TDA, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}


		fname = label + "_TDA_flag_orbits.csv";

		Fio.int_matrix_write_csv(fname, Inc_flag_orbits, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}

		FREE_int(Inc_TDA);
		FREE_int(Inc_flag_orbits);

	}

	else {

		if (f_v) {
			cout << "object_with_properties::export_TDA_with_flag_orbits "
					"flag orbits have not been computed" << endl;
		}

		int *Inc2;
		Inc2 = NEW_int(Enc->nb_rows * Enc->nb_cols);
		Int_vec_zero(Inc2, Enc->nb_rows * Enc->nb_cols);

		// +1 avoids the color white

		for (i = 0; i < Enc->nb_rows; i++) {
			i0 = Sch->orbit[i];
			for (j = 0; j < Enc->nb_cols; j++) {
				j0 = Sch->orbit[Enc->nb_rows + j] - Enc->nb_rows;
				if (Enc->get_incidence_ij(i0, j0)) {
					Inc2[i * Enc->nb_cols + j] = 1;
				}
				else {
					Inc2[i * Enc->nb_cols + j] = 0;
				}
			}

		}


		fname = label + "_TDA.csv";

		Fio.int_matrix_write_csv(
				fname, Inc2, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}
		FREE_int(Inc2);

	}

	FREE_OBJECT(Enc);
#endif

	if (f_v) {
		cout << "object_with_properties::export_TDA_with_flag_orbits done" << endl;
	}
}
#endif

#if 0
void object_with_properties::export_INP_with_flag_orbits(
		std::ostream &ost,
		groups::schreier *Sch,
		int verbose_level)
// INP = input geometry
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_properties::export_INP_with_flag_orbits" << endl;
	}

	combinatorics::encoded_combinatorial_object *Enc;

	if (f_v) {
		cout << "object_with_properties::export_INP_with_flag_orbits "
				"before OwCF->encode_incma" << endl;
	}
	OwCF->encode_incma(Enc, 0 /*verbose_level*/);
	if (f_v) {
		cout << "object_with_properties::export_INP_with_flag_orbits "
				"after OwCF->encode_incma" << endl;
		cout << "object_with_properties::export_INP_with_flag_orbits "
				"Enc->nb_rows = " << Enc->nb_rows << endl;
		cout << "object_with_properties::export_INP_with_flag_orbits "
				"Enc->nb_cols = " << Enc->nb_cols << endl;
		//Enc->print_incma();
	}

	orbiter_kernel_system::file_io Fio;
	string fname;


	int i0, j0;
	int i, j;


	if (Flags->f_flag_orbits_have_been_computed) {

		if (f_v) {
			cout << "object_with_properties::export_INP_with_flag_orbits "
					"f_flag_orbits_have_been_computed" << endl;
		}

		int *Inc_flag_orbits;
		int *Inc;
		int nb_orbits_on_flags;
		int idx;
		int orbit_idx;

		Inc = NEW_int(Enc->nb_rows * Enc->nb_cols);
		Inc_flag_orbits = NEW_int(Enc->nb_rows * Enc->nb_cols);
		nb_orbits_on_flags = Flags->Orb->Sch->nb_orbits;
		for (i = 0; i < Enc->nb_rows; i++) {
			i0 = i;
			for (j = 0; j < Enc->nb_cols; j++) {
				j0 = j;
				if (Enc->get_incidence_ij(i0, j0)) {
					idx = Flags->find_flag(i0, j0 + Enc->nb_rows);
					orbit_idx = Flags->Orb->Sch->orbit_number(idx);
					Inc_flag_orbits[i * Enc->nb_cols + j] = orbit_idx + 1;
					Inc[i * Enc->nb_cols + j] = 1;
				}
				else {
					idx = Anti_Flags->find_flag(i0, j0 + Enc->nb_rows);
					orbit_idx = Anti_Flags->Orb->Sch->orbit_number(idx);
					Inc_flag_orbits[i * Enc->nb_cols + j] = nb_orbits_on_flags + orbit_idx + 1;
					Inc[i * Enc->nb_cols + j] = 0;
				}
			}
		}

		fname = label + "_INP.csv";

		Fio.int_matrix_write_csv(fname, Inc, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

		fname = label + "_INP_flag_orbits.csv";

		Fio.int_matrix_write_csv(fname, Inc_flag_orbits, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

		FREE_int(Inc);
		FREE_int(Inc_flag_orbits);

	}
	else {

		if (f_v) {
			cout << "object_with_properties::export_INP_with_flag_orbits "
					"flag_orbits have not been computed" << endl;
		}

		int *Inc2;

		Inc2 = NEW_int(Enc->nb_rows * Enc->nb_cols);
		Int_vec_zero(Inc2, Enc->nb_rows * Enc->nb_cols);
		// +1 avoids the color white

		for (i = 0; i < Enc->nb_rows; i++) {
			i0 = i;
			for (j = 0; j < Enc->nb_cols; j++) {
				j0 = j;
				if (Enc->get_incidence_ij(i0, j0)) {
					Inc2[i * Enc->nb_cols + j] = 1;
				}
				else {
					Inc2[i * Enc->nb_cols + j] = 0;
				}
			}

		}

		fname = label + "_INP.csv";

		Fio.int_matrix_write_csv(
				fname, Inc2, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}
		FREE_int(Inc2);
	}


	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "object_with_properties::export_INP_with_flag_orbits done" << endl;
	}
}
#endif


}}}
