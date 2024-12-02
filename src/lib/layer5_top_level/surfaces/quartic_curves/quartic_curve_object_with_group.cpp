/*
 * quartic_curve_object_with_group.cpp
 *
 *  Created on: May 22, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {



quartic_curve_object_with_group::quartic_curve_object_with_group()
{
	Record_birth();
	DomA = NULL;
	QO = NULL;
	Aut_gens = NULL;
	f_has_nice_gens = false;
	nice_gens = NULL;
	projectivity_group_gens = NULL;
	Syl = NULL;
	A_on_points = NULL;
	Orbits_on_points = NULL;
}

quartic_curve_object_with_group::~quartic_curve_object_with_group()
{
	Record_death();
}

void quartic_curve_object_with_group::init(
		quartic_curve_domain_with_action *DomA,
		geometry::algebraic_geometry::quartic_curve_object *QO,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_group::init" << endl;
	}
	quartic_curve_object_with_group::DomA = DomA;
	quartic_curve_object_with_group::QO = QO;
	quartic_curve_object_with_group::Aut_gens = Aut_gens;

	if (f_v) {
		cout << "quartic_curve_object_with_group::init done" << endl;
	}
}

void quartic_curve_object_with_group::export_something(
		std::string &what,
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_group::export_something" << endl;
	}

	other::data_structures::string_tools ST;
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	if (ST.stringcmp(what, "points") == 0) {

		fname = fname_base + "_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->Pts, 1, QO->nb_pts);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "equation") == 0) {

		fname = fname_base + "_equation.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, QO->eqn15, 1, 15);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "bitangents") == 0) {

		fname = fname_base + "_bitangents.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->bitangents28, 1, 28);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "Kovalevski_points") == 0) {

		fname = fname_base + "_Kovalevski_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->QP->Kovalevski->Kovalevski_points,
				1, QO->QP->Kovalevski->nb_Kovalevski);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "singular_points") == 0) {

		fname = fname_base + "_singular_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->QP->singular_pts, 1, QO->QP->nb_singular_pts);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else {
		cout << "quartic_curve_object_with_group::export_something "
				"unrecognized export target: " << what << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object_with_group::export_something done" << endl;
	}

}





}}}}

