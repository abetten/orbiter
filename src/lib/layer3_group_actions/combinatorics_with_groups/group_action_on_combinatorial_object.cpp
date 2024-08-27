/*
 * group_action_on_combinatorial_object.cpp
 *
 *  Created on: Jul 7, 2023
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace combinatorics_with_groups {


group_action_on_combinatorial_object::group_action_on_combinatorial_object()
{
	OwCF = NULL;

	//std::string label_txt;
	//std::string label_tex;

	Enc = NULL;
	//Sch = NULL;
	A_perm = NULL;
	gens = NULL;

	A_on_points = NULL;
	A_on_lines = NULL;

	points = NULL;
	lines = NULL;

	Decomposition = NULL;

	Sch_points = NULL;
	Sch_lines = NULL;


	Flags = NULL;
	Anti_Flags = NULL;

}


group_action_on_combinatorial_object::~group_action_on_combinatorial_object()
{
	if (points) {
		FREE_lint(points);
	}
	if (lines) {
		FREE_lint(lines);
	}
}

void group_action_on_combinatorial_object::init(
		std::string &label_txt,
		std::string &label_tex,
		canonical_form_classification::object_with_canonical_form *OwCF,
		actions::action *A_perm,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init" << endl;
	}

	group_action_on_combinatorial_object::label_txt = label_txt;
	group_action_on_combinatorial_object::label_tex = label_tex;

	group_action_on_combinatorial_object::OwCF = OwCF;
	group_action_on_combinatorial_object::A_perm = A_perm;


	if (!A_perm->f_has_strong_generators) {
		cout << "group_action_on_combinatorial_object::init !A_perm->f_has_strong_generators" << endl;
		exit(1);
	}
	groups::strong_generators *gens = A_perm->Strong_gens;


	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before OwCF->encode_incma" << endl;
	}
	OwCF->encode_incma(Enc, 0 /*verbose_level*/);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after OwCF->encode_incma" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"Enc->nb_rows = " << Enc->nb_rows << endl;
		cout << "group_action_on_combinatorial_object::init "
				"Enc->nb_cols = " << Enc->nb_cols << endl;
	}

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
	std::string label1_tex, label2_tex;

	label1 = label_txt + "_points";
	label2 = label_txt + "_lines";
	label1_tex = label_tex + "\\_points";
	label2_tex = label_tex + "\\_lines";


	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before A_perm->Induced_action->restricted_action" << endl;
	}
	A_on_points = A_perm->Induced_action->restricted_action(
			points, Enc->nb_rows,
			label1, label1_tex,
			verbose_level);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after A_perm->Induced_action->restricted_action" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before A_perm->Induced_action->restricted_action" << endl;
	}
	A_on_lines = A_perm->Induced_action->restricted_action(
			lines, Enc->nb_cols,
			label2, label2_tex,
			verbose_level);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after A_perm->Induced_action->restricted_action" << endl;
	}

	geometry::incidence_structure *Inc;

	Inc = NEW_OBJECT(geometry::incidence_structure);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Inc->init_by_matrix" << endl;
	}
	Inc->init_by_matrix(
			Enc->nb_rows,
			Enc->nb_cols,
			Enc->get_Incma(), 0 /* verbose_level*/);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Inc->init_by_matrix" << endl;
	}


	Decomposition = NEW_OBJECT(combinatorics::decomposition);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Decomposition->init_incidence_structure" << endl;
	}
	Decomposition->init_incidence_structure(Inc, verbose_level - 1);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Decomposition->init_incidence_structure" << endl;
	}




	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Decomposition->compute_TDO_deep" << endl;
	}
	Decomposition->compute_TDO_deep(verbose_level - 1);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Decomposition->compute_TDO_deep" << endl;
	}

#if 0
	int TDO_depth = N;
	//int TDO_ht;


	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Decomposition->I->compute_TDO_safe" << endl;
	}
	Decomposition->I->compute_TDO_safe(*Decomposition->Stack, TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Decomposition->I->compute_TDO_safe" << endl;
	}
#endif

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"computing orbits on points" << endl;
	}

	if (gens == NULL) {
		cout << "group_action_on_combinatorial_object::init gens == NULL" << endl;
		exit(1);
	}

	Sch_points = NEW_OBJECT(groups::schreier);
	Sch_points->init(A_on_points, verbose_level - 2);
	Sch_points->initialize_tables();
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Sch_points->init_generators" << endl;
	}
	Sch_points->init_generators(
			*gens->gens /* *generators */, verbose_level - 2);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Sch_points->init_generators" << endl;
	}
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Sch_points->compute_all_point_orbits" << endl;
	}
	Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Sch_points->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"found " << Sch_points->nb_orbits
				<< " orbits on points" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"computing orbits on lines or blocks" << endl;
	}

	Sch_lines = NEW_OBJECT(groups::schreier);
	Sch_lines->init(A_on_lines, verbose_level - 2);
	Sch_lines->initialize_tables();
	Sch_lines->init_generators(
			*gens->gens /* *generators */, verbose_level - 2);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Sch_lines->compute_all_point_orbits" << endl;
	}
	Sch_lines->compute_all_point_orbits(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Sch_lines->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"found " << Sch_lines->nb_orbits
				<< " orbits on lines" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before S->split_by_orbit_partition" << endl;
	}

	Decomposition->Stack->split_by_orbit_partition(
			Sch_points->nb_orbits,
		Sch_points->orbit_first, Sch_points->orbit_len, Sch_points->orbit,
		0 /* offset */,
		verbose_level - 2);

	Decomposition->Stack->split_by_orbit_partition(
			Sch_lines->nb_orbits,
		Sch_lines->orbit_first, Sch_lines->orbit_len, Sch_lines->orbit,
		Decomposition->Inc->nb_points() /* offset */,
		verbose_level - 2);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after S->split_by_orbit_partition" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Decomposition->compute_the_decomposition" << endl;
	}
	Decomposition->compute_the_decomposition(
			verbose_level - 1);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Decomposition->compute_the_decomposition" << endl;
	}


	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Decomposition->get_row_scheme" << endl;
	}
	Decomposition->get_row_scheme(verbose_level);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Decomposition->get_row_scheme" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before Decomposition->get_col_scheme" << endl;
	}
	Decomposition->get_col_scheme(verbose_level);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after Decomposition->get_col_scheme" << endl;
	}





	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"before compute_flag_orbits" << endl;
	}
	compute_flag_orbits(verbose_level);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::init "
				"after compute_flag_orbits" << endl;
	}


	if (f_v) {
		cout << "group_action_on_combinatorial_object::init done" << endl;
	}
}


void group_action_on_combinatorial_object::print_schemes(
		std::ostream &ost,
		canonical_form_classification::classification_of_objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::print_schemes" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::print_schemes "
				"before Decomposition->print_schemes" << endl;
	}

	Decomposition->print_schemes(ost, Report_options, verbose_level);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::print_schemes "
				"after Decomposition->print_schemes" << endl;
	}



	if (f_v) {
		cout << "group_action_on_combinatorial_object::print_schemes done" << endl;
	}
}


void group_action_on_combinatorial_object::compute_flag_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::compute_flag_orbits" << endl;
	}

	Flags = NEW_OBJECT(flag_orbits_incidence_structure);
	Anti_Flags = NEW_OBJECT(flag_orbits_incidence_structure);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::compute_flag_orbits "
				"before Flags->init" << endl;
	}
	Flags->init(
			OwCF, false, A_perm, A_perm->Strong_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::compute_flag_orbits "
				"after Flags->init" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::compute_flag_orbits "
				"before Anti_Flags->init" << endl;
	}
	Anti_Flags->init(
			OwCF, true, A_perm, A_perm->Strong_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "group_action_on_combinatorial_object::compute_flag_orbits "
				"after Anti_Flags->init" << endl;
	}

	if (f_v) {
		cout << "group_action_on_combinatorial_object::compute_flag_orbits done" << endl;
	}
}



void group_action_on_combinatorial_object::report_flag_orbits(
		std::ostream &ost, int verbose_level)
{
	ost << "Flag orbits:\\\\" << endl;
	Flags->report(ost, verbose_level);

	ost << "Anti-Flag orbits:\\\\" << endl;
	Anti_Flags->report(ost, verbose_level);

}


void group_action_on_combinatorial_object::export_TDA_with_flag_orbits(
		std::ostream &ost,
		int verbose_level)
// TDA = tactical decomposition by automorphism group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::export_TDA_with_flag_orbits" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;


	int i0, j0;
	int i, j;


	if (Flags->f_flag_orbits_have_been_computed) {

		if (f_v) {
			cout << "group_action_on_combinatorial_object::export_TDA_with_flag_orbits "
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
			i0 = Sch_points->orbit[i];
			for (j = 0; j < Enc->nb_cols; j++) {
				j0 = Sch_lines->orbit[j];
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

		fname = label_txt + "_TDA.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Inc_TDA, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}


		fname = label_txt + "_TDA_flag_orbits.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Inc_flag_orbits, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}

		FREE_int(Inc_TDA);
		FREE_int(Inc_flag_orbits);

	}

	else {

		if (f_v) {
			cout << "group_action_on_combinatorial_object::export_TDA_with_flag_orbits "
					"flag orbits have not been computed" << endl;
		}

		int *Inc2;

		Inc2 = NEW_int(Enc->nb_rows * Enc->nb_cols);
		Int_vec_zero(Inc2, Enc->nb_rows * Enc->nb_cols);

		// +1 avoids the color white

		for (i = 0; i < Enc->nb_rows; i++) {
			i0 = Sch_points->orbit[i];
			for (j = 0; j < Enc->nb_cols; j++) {
				j0 = Sch_lines->orbit[j];
				if (Enc->get_incidence_ij(i0, j0)) {
					Inc2[i * Enc->nb_cols + j] = 1;
				}
				else {
					Inc2[i * Enc->nb_cols + j] = 0;
				}
			}

		}


		fname = label_txt + "_TDA.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Inc2, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}
		FREE_int(Inc2);

	}

	//FREE_OBJECT(Enc);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::export_TDA_with_flag_orbits done" << endl;
	}

}

void group_action_on_combinatorial_object::export_INP_with_flag_orbits(
		std::ostream &ost,
		int verbose_level)
// INP = input geometry
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::export_INP_with_flag_orbits" << endl;
	}


	orbiter_kernel_system::file_io Fio;
	string fname;


	int i0, j0;
	int i, j;


	if (Flags->f_flag_orbits_have_been_computed) {

		if (f_v) {
			cout << "group_action_on_combinatorial_object::export_INP_with_flag_orbits "
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

		fname = label_txt + "_INP.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Inc, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

		fname = label_txt + "_INP_flag_orbits.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Inc_flag_orbits, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

		FREE_int(Inc);
		FREE_int(Inc_flag_orbits);

	}
	else {

		if (f_v) {
			cout << "group_action_on_combinatorial_object::export_INP_with_flag_orbits "
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

		fname = label_txt + "_INP.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Inc2, Enc->nb_rows, Enc->nb_cols);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}
		FREE_int(Inc2);
	}


	//FREE_OBJECT(Enc);

	if (f_v) {
		cout << "group_action_on_combinatorial_object::export_INP_with_flag_orbits done" << endl;
	}
}



}}}




