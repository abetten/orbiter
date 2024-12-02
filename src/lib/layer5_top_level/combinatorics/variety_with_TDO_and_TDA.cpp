/*
 * variety_with_TDO_and_TDA.cpp
 *
 *  Created on: Aug 27, 2024
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {




variety_with_TDO_and_TDA::variety_with_TDO_and_TDA()
{
	Record_birth();
	PA = NULL;

	Variety_object = NULL;

	A_on_points = NULL;
	A_on_lines = NULL;

	Aut_gens = NULL;

	f_has_TDO_TDA = false;

	Decomposition_scheme_TDO = NULL;
	Decomposition_scheme_TDA = NULL;

	Variety_with_TDO = NULL;
	Variety_with_TDA = NULL;
}



variety_with_TDO_and_TDA::~variety_with_TDO_and_TDA()
{
	Record_death();
	if (Decomposition_scheme_TDO) {
		FREE_OBJECT(Decomposition_scheme_TDO);
	}
	if (Decomposition_scheme_TDA) {
		FREE_OBJECT(Decomposition_scheme_TDA);
	}
	if (Variety_with_TDO) {
		FREE_OBJECT(Variety_with_TDO);
	}
	if (Variety_with_TDA) {
		FREE_OBJECT(Variety_with_TDA);
	}
}



void variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions(
		projective_geometry::projective_space_with_action *PA,
		geometry::algebraic_geometry::variety_object *Variety_object,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;


	variety_with_TDO_and_TDA::PA = PA;
	variety_with_TDO_and_TDA::Variety_object = Variety_object;

	A_on_points = PA->A;
	A_on_lines = PA->A_on_lines;

	variety_with_TDO_and_TDA::Aut_gens = Aut_gens;

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"before Combi.compute_TDO_decomposition_of_projective_space" << endl;
	}
	Decomposition_scheme_TDO = Combi.compute_TDO_decomposition_of_projective_space(
			PA->P,
			Variety_object->Point_sets->Sets[0],
			Variety_object->Point_sets->Set_size[0],
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0],
			verbose_level);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"after Combi.compute_TDO_decomposition_of_projective_space" << endl;
	}


	if (Decomposition_scheme_TDO == NULL) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"the space is too large, we abort the computation of tactical decompositions" << endl;
		f_has_TDO_TDA = false;
		return;
	}

	f_has_TDO_TDA = true;

	combinatorics_with_groups::combinatorics_with_action CombiA;

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"before CombiA.refine_decomposition_by_group_orbits" << endl;
	}
	CombiA.refine_decomposition_by_group_orbits(
			Decomposition_scheme_TDO->Decomposition,
			A_on_points,
			A_on_lines,
			Aut_gens,
			verbose_level);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"after CombiA.refine_decomposition_by_group_orbits" << endl;
	}

	//combinatorics::decomposition_scheme *Decomposition_scheme_TDA;

	Decomposition_scheme_TDA = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition_scheme);

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"before Decomposition_scheme->init_row_and_col_schemes" << endl;
	}
	Decomposition_scheme_TDA->init_row_and_col_schemes(
			Decomposition_scheme_TDO->Decomposition,
		verbose_level);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"after Decomposition_scheme->init_row_and_col_schemes" << endl;
	}


	Variety_with_TDO = NEW_OBJECT(geometry::algebraic_geometry::variety_object);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"before Variety_with_TDO->init_set_of_sets" << endl;
	}
	Variety_with_TDO->init_set_of_sets(
			PA->P,
			Variety_object->Ring,
			Variety_object->eqn,
			Decomposition_scheme_TDO->SoS_points,
			Decomposition_scheme_TDO->SoS_lines,
			verbose_level);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"after Variety_with_TDO->init_set_of_sets" << endl;
	}

	Variety_with_TDA = NEW_OBJECT(geometry::algebraic_geometry::variety_object);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"before Variety_with_TDA->init_set_of_sets" << endl;
	}
	Variety_with_TDA->init_set_of_sets(
			PA->P,
			Variety_object->Ring,
			Variety_object->eqn,
			Decomposition_scheme_TDA->SoS_points,
			Decomposition_scheme_TDA->SoS_lines,
			verbose_level);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions "
				"after Variety_with_TDA->init_set_of_sets" << endl;
	}


	if (f_v) {
		cout << "variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions done" << endl;
	}
}

void variety_with_TDO_and_TDA::report_decomposition_schemes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes" << endl;
	}

	int upper_bound_on_size_for_printing = 100;

	std::string label_scheme;

	label_scheme = "TDO";


	if (Decomposition_scheme_TDO == NULL) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"Decomposition_scheme_TDO not available" << endl;
		return;
	}
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"before Decomposition_scheme_TDO->report_latex_with_external_files" << endl;
	}
	Decomposition_scheme_TDO->report_latex_with_external_files(
			ost,
			label_scheme,
			Variety_object->label_txt,
			upper_bound_on_size_for_printing,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"after Decomposition_scheme_TDO->report_latex_with_external_files" << endl;
	}


	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"before Decomposition_scheme_TDO->export_csv" << endl;
	}
	Decomposition_scheme_TDO->export_csv(
			label_scheme,
			Variety_object->label_txt,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"after Decomposition_scheme_TDO->export_csv" << endl;
	}


	if (Decomposition_scheme_TDA == NULL) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"Decomposition_scheme_TDA not available" << endl;
		return;
	}

	label_scheme = "TDA";

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"before Decomposition_scheme_TDA->report_latex_with_external_files" << endl;
	}
	Decomposition_scheme_TDA->report_latex_with_external_files(
			ost,
			label_scheme,
			Variety_object->label_txt,
			upper_bound_on_size_for_printing,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"after Decomposition_scheme_TDA->report_latex_with_external_files" << endl;
	}


	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"before Decomposition_scheme_TDA->export_csv" << endl;
	}
	Decomposition_scheme_TDA->export_csv(
			label_scheme,
			Variety_object->label_txt,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"after Decomposition_scheme_TDA->export_csv" << endl;
	}


	ost << "TDO classes:\\\\" << endl;

	label_scheme = "TDO";

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"before Decomposition_scheme_TDO->report_classes_with_external_files" << endl;
	}
	Decomposition_scheme_TDO->report_classes_with_external_files(
			ost,
			label_scheme,
			Variety_object->label_txt,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"after Decomposition_scheme_TDO->report_classes_with_external_files" << endl;
	}

	ost << "TDA classes:\\\\" << endl;

	label_scheme = "TDA";

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"before Decomposition_scheme_TDA->report_classes_with_external_files" << endl;
	}
	Decomposition_scheme_TDA->report_classes_with_external_files(
			ost,
			label_scheme,
			Variety_object->label_txt,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes "
				"after Decomposition_scheme_TDA->report_classes_with_external_files" << endl;
	}

	if (f_v) {
		cout << "variety_with_TDO_and_TDA::report_decomposition_schemes done" << endl;
	}
}






}}}


