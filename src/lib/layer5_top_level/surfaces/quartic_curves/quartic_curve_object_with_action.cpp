/*
 * quartic_curve_object_with_action.cpp
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



quartic_curve_object_with_action::quartic_curve_object_with_action()
{
	F = NULL;
	DomA = NULL;
	QO = NULL;
	Aut_gens = NULL;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;
	projectivity_group_gens = NULL;
	Syl = NULL;
	A_on_points = NULL;
	Orbits_on_points = NULL;
}

quartic_curve_object_with_action::~quartic_curve_object_with_action()
{
}

void quartic_curve_object_with_action::init(quartic_curve_domain_with_action *DomA,
		algebraic_geometry::quartic_curve_object *QO,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_action::init" << endl;
	}
	quartic_curve_object_with_action::DomA = DomA;
	quartic_curve_object_with_action::QO = QO;
	quartic_curve_object_with_action::Aut_gens = Aut_gens;
	F = DomA->Dom->F;

	if (f_v) {
		cout << "quartic_curve_object_with_action::init done" << endl;
	}
}

void quartic_curve_object_with_action::export_something(std::string &what,
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_action::export_something" << endl;
	}

	data_structures::string_tools ST;
	string fname;
	orbiter_kernel_system::file_io Fio;

#if 0
	if (ST.stringcmp(what, "points") == 0) {

		fname.assign(fname_base);
		fname.append("_points.csv");

		//Fio.write_set_to_file(fname, Pts, nb_pts, 0 /*verbose_level*/);
		Fio.lint_matrix_write_csv(fname, Pts, 1, nb_pts);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "points_off") == 0) {

		fname.assign(fname_base);
		fname.append("_points_off.csv");

		long int *Pts_off;
		int nb_pts_off;

		nb_pts_off = Surf->P->N_points - nb_pts;

		Pts_off = NEW_lint(Surf->P->N_points);

		Lint_vec_complement_to(Pts, Pts_off, Surf->P->N_points, nb_pts);

		//Fio.write_set_to_file(fname, Pts_off, nb_pts_off, 0 /*verbose_level*/);
		Fio.lint_matrix_write_csv(fname, Pts_off, 1, nb_pts_off);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		FREE_lint(Pts_off);
	}
	else if (ST.stringcmp(what, "lines") == 0) {

		fname.assign(fname_base);
		fname.append("_lines.csv");

		//Fio.write_set_to_file(fname, Pts, nb_pts, 0 /*verbose_level*/);
		Fio.lint_matrix_write_csv(fname, Lines, nb_lines, 1);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "Eckardt_points") == 0) {

		fname.assign(fname_base);
		fname.append("_Eckardt_points.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Eckardt_points, 1, SOP->nb_Eckardt_points);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "Eckardt_points") == 0) {

		fname.assign(fname_base);
		fname.append("_Eckardt_points.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Eckardt_points, 1, SOP->nb_Eckardt_points);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "Hesse_planes") == 0) {

		fname.assign(fname_base);
		fname.append("_Hesse_planes.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Hesse_planes, 1, SOP->nb_Hesse_planes);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "axes") == 0) {

		fname.assign(fname_base);
		fname.append("_axes.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Axes_line_rank, 1, SOP->nb_axes);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "double_points") == 0) {

		fname.assign(fname_base);
		fname.append("_double_points.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Double_points, 1, SOP->nb_Double_points);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "single_points") == 0) {

		fname.assign(fname_base);
		fname.append("_single_points.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Single_points, 1, SOP->nb_Single_points);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "singular_points") == 0) {

		fname.assign(fname_base);
		fname.append("_singular_points.csv");

		Fio.lint_matrix_write_csv(fname, SOP->singular_pts, 1, SOP->nb_singular_pts);

		cout << "quartic_curve_object_with_action::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else {
		cout << "quartic_curve_object_with_action::export_something unrecognized export target: " << what << endl;
	}
#endif

	if (f_v) {
		cout << "quartic_curve_object_with_action::export_something done" << endl;
	}

}


}}}}

