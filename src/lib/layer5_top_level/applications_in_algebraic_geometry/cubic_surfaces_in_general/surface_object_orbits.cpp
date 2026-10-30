/*
 * surface_object_orbits.cpp
 *
 *  Created on: Apr 13, 2026
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {



surface_object_orbits::surface_object_orbits()
{
	Record_birth();

	SOG = NULL;

	A_on_points = NULL;
	A_on_Eckardt_points = NULL;
	A_on_Double_points = NULL;
	A_on_Single_points = NULL;
	A_on_the_lines = NULL;
	A_single_sixes = NULL;
	A_double_sixes = NULL;
	A_on_tritangent_planes = NULL;
	A_on_Hesse_planes = NULL;
	A_on_axes = NULL;
	A_on_trihedral_pairs = NULL;
	A_on_pts_not_on_lines = NULL;


	Orbits_on_points = NULL;
	Orbits_on_Eckardt_points = NULL;
	Orbits_on_Double_points = NULL;
	Orbits_on_Single_points = NULL;
	Orbits_on_lines = NULL;
	Orbits_on_single_sixes = NULL;
	Orbits_on_double_sixes = NULL;
	Orbits_on_tritangent_planes = NULL;
	Orbits_on_Hesse_planes = NULL;
	Orbits_on_axes = NULL;
	Orbits_on_trihedral_pairs = NULL;
	Orbits_on_points_not_on_lines = NULL;

	print_interval = 10000;

}

surface_object_orbits::~surface_object_orbits()
{
	Record_death();
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::~surface_object_orbits" << endl;
	}

	if (A_on_points) {
		if (f_v) {
			cout << "surface_object_orbits::~surface_object_orbits before FREE_OBJECT(A_on_points)" << endl;
		}
		FREE_OBJECT(A_on_points);
	}
	if (A_on_Eckardt_points) {
		FREE_OBJECT(A_on_Eckardt_points);
	}
	if (A_on_Double_points) {
		FREE_OBJECT(A_on_Double_points);
	}
	if (A_on_Single_points) {
		FREE_OBJECT(A_on_Single_points);
	}
	if (A_on_the_lines) {
		FREE_OBJECT(A_on_the_lines);
	}
	if (A_single_sixes) {
		FREE_OBJECT(A_single_sixes);
	}
	if (A_double_sixes) {
		FREE_OBJECT(A_double_sixes);
	}
	if (A_on_tritangent_planes) {
		FREE_OBJECT(A_on_tritangent_planes);
	}
	if (A_on_Hesse_planes) {
		FREE_OBJECT(A_on_Hesse_planes);
	}
	if (A_on_axes) {
		if (f_v) {
			cout << "surface_object_orbits::~surface_object_orbits before FREE_OBJECT(A_on_axes)" << endl;
		}
		FREE_OBJECT(A_on_axes);
	}
	if (A_on_trihedral_pairs) {
		FREE_OBJECT(A_on_trihedral_pairs);
	}
	if (A_on_pts_not_on_lines) {
		FREE_OBJECT(A_on_pts_not_on_lines);
	}
	if (Orbits_on_points) {
		FREE_OBJECT(Orbits_on_points);
	}
	if (Orbits_on_Eckardt_points) {
		FREE_OBJECT(Orbits_on_Eckardt_points);
	}
	if (Orbits_on_Double_points) {
		FREE_OBJECT(Orbits_on_Double_points);
	}
	if (Orbits_on_Single_points) {
		FREE_OBJECT(Orbits_on_Single_points);
	}
	if (Orbits_on_lines) {
		FREE_OBJECT(Orbits_on_lines);
	}
	if (Orbits_on_single_sixes) {
		FREE_OBJECT(Orbits_on_single_sixes);
	}
	if (Orbits_on_double_sixes) {
		if (f_v) {
			cout << "surface_object_orbits::~surface_object_orbits before FREE_OBJECT(Orbits_on_double_sixes)" << endl;
		}
		FREE_OBJECT(Orbits_on_double_sixes);
	}
	if (Orbits_on_tritangent_planes) {
		FREE_OBJECT(Orbits_on_tritangent_planes);
	}
	if (Orbits_on_Hesse_planes) {
		FREE_OBJECT(Orbits_on_Hesse_planes);
	}
	if (Orbits_on_axes) {
		FREE_OBJECT(Orbits_on_axes);
	}
	if (Orbits_on_trihedral_pairs) {
		FREE_OBJECT(Orbits_on_trihedral_pairs);
	}
	if (Orbits_on_points_not_on_lines) {
		FREE_OBJECT(Orbits_on_points_not_on_lines);
	}
	if (f_v) {
		cout << "surface_object_orbits::~surface_object_orbits done" << endl;
	}
}

void surface_object_orbits::init(
		surface_object_with_group *SOG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init" << endl;
	}
	surface_object_orbits::SOG = SOG;

	if (f_v) {
		cout << "surface_object_orbits::init "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_orbits::init "
				"after compute_orbits_of_automorphism_group" << endl;
	}


	if (f_v) {
		cout << "surface_object_orbits::init done" << endl;
	}
}


void surface_object_orbits::compute_orbits_of_automorphism_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group" << endl;
	}

	// orbits on points:

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"orbits on points" << endl;
	}
	init_orbits_on_points(verbose_level - 1);


	// orbits on Eckardt points:

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"orbits on Eckardt points" << endl;
	}
	init_orbits_on_Eckardt_points(verbose_level - 1);


	// orbits on Double points:

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"orbits on double points" << endl;
	}
	init_orbits_on_Double_points(verbose_level - 1);

	// orbits on Single points:

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"orbits on single points" << endl;
	}
	init_orbits_on_Single_points(verbose_level - 1);


	// orbits on lines:

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"orbits on lines" << endl;
	}
	init_orbits_on_lines(verbose_level);



	if (SOG->SO->Variety_object->Line_sets->Set_size[0] == 27) {

		// orbits on half double sixes:

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"before init_orbits_on_half_double_sixes" << endl;
		}
		init_orbits_on_half_double_sixes(verbose_level - 1);

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"after init_orbits_on_half_double_sixes" << endl;
		}

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"before init_orbits_on_double_sixes" << endl;
		}
		init_orbits_on_double_sixes(verbose_level - 1);

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"after init_orbits_on_double_sixes" << endl;
		}

		// orbits on tritangent planes:

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"before init_orbits_on_tritangent_planes" << endl;
		}
		init_orbits_on_tritangent_planes(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"after init_orbits_on_tritangent_planes" << endl;
		}


		// orbits on Hesse planes:

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"before init_orbits_on_Hesse_planes" << endl;
		}
		init_orbits_on_Hesse_planes(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"after init_orbits_on_Hesse_planes" << endl;
		}

		// orbits on axes:

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"before init_orbits_on_axes" << endl;
		}
		init_orbits_on_axes(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"after init_orbits_on_axes" << endl;
		}


		// orbits on trihedral pairs:

		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"before init_orbits_on_trihedral_pairs" << endl;
		}
		init_orbits_on_trihedral_pairs(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
					"after init_orbits_on_trihedral_pairs" << endl;
		}

	}



	// orbits on points not on lines:

	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"before init_orbits_on_points_not_on_lines" << endl;
	}
	init_orbits_on_points_not_on_lines(verbose_level - 1);
	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group "
				"after init_orbits_on_points_not_on_lines" << endl;
	}


	if (f_v) {
		cout << "surface_object_orbits::compute_orbits_of_automorphism_group done" << endl;
	}
}


void surface_object_orbits::init_orbits_on_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points action: ";
		SOG->Surf_A->A->print_info();
	}


	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Pts");
	label_of_set_tex.assign("\\_Pts");

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points "
				"creating action on points:" << endl;
	}

	A_on_points = SOG->Surf_A->A->Induced_action->restricted_action(
			SOG->SO->Variety_object->Point_sets->Sets[0],
			SOG->SO->Variety_object->Point_sets->Set_size[0],
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points "
				"creating action on points done" << endl;
	}


	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points "
				"computing orbits on points:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_orbits::init_orbits_on_points "
					"computing orbits on points using nice gens:" << endl;
		}
		Orbits_on_points = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_points,
				print_interval,
				0 /*verbose_level*/);

	}
	else {
		if (f_v) {
			cout << "surface_object_orbits::init_orbits_on_points "
					"computing orbits on points using Aut_gens:" << endl;
			SOG->Aut_gens->print_generators(cout, verbose_level - 1);
		}
		if (f_v) {
			cout << "surface_object_orbits::init_orbits_on_points "
					"before Aut_gens->compute_all_point_orbits_schreier" << endl;
		}

		Orbits_on_points = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_points, print_interval, verbose_level - 2);
		if (f_v) {
			cout << "surface_object_orbits::init_orbits_on_points "
					"after Aut_gens->compute_all_point_orbits_schreier" << endl;
		}
	}
	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points "
				"We found " << Orbits_on_points->Forest->nb_orbits
				<< " orbits on points" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_Eckardt_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Eckardt_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Eckardt");
	label_of_set_tex.assign("\\_Eckardt");

	if (f_v) {
		cout << "creating action on Eckardt points:" << endl;
	}
	A_on_Eckardt_points = SOG->Surf_A->A->Induced_action->restricted_action(
			SOG->SO->SOP->Eckardt_points,
			SOG->SO->SOP->nb_Eckardt_points,
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Eckardt points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Eckardt points:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_orbits::init_orbits_on_Eckardt_points "
					"computing orbits on points using nice gens:" << endl;
		}
		Orbits_on_Eckardt_points = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_Eckardt_points, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Eckardt_points = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_Eckardt_points, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Eckardt_points->Forest->nb_orbits
				<< " orbits on Eckardt points" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Eckardt_points done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_Double_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Double_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Double_pts");
	label_of_set_tex.assign("\\_Double\\_pts");

	if (f_v) {
		cout << "creating action on Double points:" << endl;
	}
	A_on_Double_points = SOG->Surf_A->A->Induced_action->restricted_action(
			SOG->SO->SOP->Double_points,
			SOG->SO->SOP->nb_Double_points,
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Double points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Double points:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		Orbits_on_Double_points = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_Double_points, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Double_points = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_Double_points, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Double_points->Forest->nb_orbits
				<< " orbits on Double points" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Double_points done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_Single_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Single_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Single_pts");
	label_of_set_tex.assign("\\_Single\\_pts");

	if (f_v) {
		cout << "creating action on Single points:" << endl;
	}
	A_on_Single_points = SOG->Surf_A->A->Induced_action->restricted_action(
			SOG->SO->SOP->Single_points,
			SOG->SO->SOP->nb_Single_points,
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Single points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Single points:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		Orbits_on_Single_points = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_Single_points, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Single_points = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_Single_points, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Single_points->Forest->nb_orbits
				<< " orbits on Single points" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Single_points done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_lines" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Lines");
	label_of_set_tex.assign("\\_Lines");

	if (f_v) {
		cout << "creating restricted action "
				"on the lines:" << endl;
	}
	A_on_the_lines = SOG->Surf_A->A2->Induced_action->restricted_action(
			SOG->SO->Variety_object->Line_sets->Sets[0],
			SOG->SO->Variety_object->Line_sets->Set_size[0],
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating restricted action "
				"on the lines done" << endl;
	}

	if (f_v) {
		cout << "computing orbits on lines:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		Orbits_on_lines = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_the_lines, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_lines = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_the_lines, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_lines->Forest->nb_orbits
				<< " orbits on lines" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_lines done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_half_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_half_double_sixes" << endl;
	}

	if (f_v) {
		cout << "creating action on half double sixes:" << endl;
	}
	A_single_sixes = A_on_the_lines->Induced_action->create_induced_action_on_sets(
			72, 6,
			SOG->Surf->Schlaefli->Schlaefli_double_six->Double_six,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on half double sixes done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on single sixes:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		Orbits_on_single_sixes = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_single_sixes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_single_sixes = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_single_sixes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "computing orbits on single sixes done" << endl;
	}
	if (f_v) {
		cout << "We found " << Orbits_on_single_sixes->Forest->nb_orbits
				<< " orbits on single sixes" << endl;
	}

	//nb_orbits_on_single_sixes = Orbits_on_single_sixes->nb_orbits;

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_half_double_sixes done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_double_sixes" << endl;
	}

	long int double_six_sets[72];
	int i, j;

	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			double_six_sets[i * 2 + j] = i * 2 + j;
		}
	}

	if (f_v) {
		cout << "creating action on half double sixes:" << endl;
	}
	A_double_sixes = A_single_sixes->Induced_action->create_induced_action_on_sets(
			36, 2,
			double_six_sets,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on half double sixes done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on double sixes:" << endl;
	}
	if (SOG->f_has_nice_gens) {
		Orbits_on_double_sixes = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_double_sixes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_double_sixes = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_double_sixes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "computing orbits on double sixes done" << endl;
	}
	if (f_v) {
		cout << "We found " << Orbits_on_double_sixes->Forest->nb_orbits
				<< " orbits on double sixes" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_double_sixes done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_tritangent_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_tritangent_planes" << endl;
	}

	if (f_v) {
		cout << "creating action on tritangent planes:" << endl;
		cout << "SO->SOP->nb_tritangent_planes = "
				<< SOG->SO->SOP->SmoothProperties->nb_tritangent_planes << endl;
	}
	A_on_tritangent_planes =
			A_on_the_lines->Induced_action->create_induced_action_on_sets(
					SOG->SO->SOP->SmoothProperties->nb_tritangent_planes, 3,
					SOG->Surf->Schlaefli->Schlaefli_tritangent_planes->Lines_in_tritangent_planes,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "action on tritangent planes done" << endl;
	}

	if (SOG->f_has_nice_gens) {
		Orbits_on_tritangent_planes = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_tritangent_planes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_tritangent_planes = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_tritangent_planes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_tritangent_planes->Forest->nb_orbits
				<< " orbits on the set of " << SOG->SO->SOP->SmoothProperties->nb_tritangent_planes
				<< " tritangent planes" << endl;
	}

	if (f_vv) {
		Orbits_on_tritangent_planes->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_tritangent_planes done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_Hesse_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Hesse_planes" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Hesse_planes");
	label_of_set_tex.assign("\\_Hesse\\_planes");

	if (f_v) {
		cout << "creating action on Hesse planes:" << endl;
		cout << "SO->SOP->nb_Hesse_planes = "
				<< SOG->SO->SOP->nb_Hesse_planes << endl;
	}
	A_on_Hesse_planes = SOG->Surf_A->A_on_planes->Induced_action->restricted_action(
			SOG->SO->SOP->Hesse_planes,
			SOG->SO->SOP->nb_Hesse_planes,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "action on Hesse planes done" << endl;
	}

	if (SOG->f_has_nice_gens) {
		Orbits_on_Hesse_planes = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_Hesse_planes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Hesse_planes = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_Hesse_planes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Hesse_planes->Forest->nb_orbits
				<< " orbits on the set of " << SOG->SO->SOP->nb_Hesse_planes
				<< " Hesse planes" << endl;
	}

	if (f_vv) {
		Orbits_on_Hesse_planes->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_Hesse_planes done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_axes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_axes" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set = "_axes";
	label_of_set_tex = "\\_axes";

	if (f_v) {
		cout << "creating action on axes:" << endl;
		cout << "SO->SOP->nb_axes = "
				<< SOG->SO->SOP->nb_axes << endl;
		cout << "Axes_line_rank:";
		Lint_vec_print(cout, SOG->SO->SOP->Axes_line_rank, SOG->SO->SOP->nb_axes);
		cout << endl;
	}
	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_axes "
				"before Surf_A->A2->restricted_action" << endl;
	}
	A_on_axes = SOG->Surf_A->A2->Induced_action->restricted_action(
			SOG->SO->SOP->Axes_line_rank,
			SOG->SO->SOP->nb_axes,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_axes "
				"after Surf_A->A2->restricted_action" << endl;
	}

	if (SOG->f_has_nice_gens) {
		Orbits_on_axes = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_axes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_axes = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_axes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_axes->Forest->nb_orbits
				<< " orbits on the set of " << SOG->SO->SOP->nb_axes
				<< " axes" << endl;
	}

	if (f_vv) {
		Orbits_on_axes->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_axes done" << endl;
	}
}


void surface_object_orbits::init_orbits_on_trihedral_pairs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "creating action on trihedral pairs:" << endl;
	}
	A_on_trihedral_pairs =
			A_on_tritangent_planes->Induced_action->create_induced_action_on_sets(
					120, 6,
					SOG->Surf->Schlaefli->Schlaefli_trihedral_pairs->Axes,
					0 /*verbose_level*/);
	if (f_v) {
		cout << "action on trihedral pairs created" << endl;
	}

	if (SOG->f_has_nice_gens) {
		Orbits_on_trihedral_pairs = SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
				SOG->Surf_A->A,
				A_on_trihedral_pairs, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_trihedral_pairs = SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
				A_on_trihedral_pairs, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_trihedral_pairs->Forest->nb_orbits
				<< " orbits on trihedral pairs" << endl;
	}

	if (f_vv) {
		Orbits_on_trihedral_pairs->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_trihedral_pairs done" << endl;
	}
}

void surface_object_orbits::init_orbits_on_points_not_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points_not_on_lines" << endl;
	}


	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set = "_pts_not_on_lines";
	label_of_set_tex = "\\_pts\\_not\\_on\\_lines";

	if (f_v) {
		cout << "creating action on points not on lines:" << endl;
	}
	A_on_pts_not_on_lines = SOG->Surf_A->A->Induced_action->restricted_action(
			SOG->SO->SOP->Pts_not_on_lines,
			SOG->SO->SOP->nb_pts_not_on_lines,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on points not on lines done" << endl;
	}

	if (SOG->f_has_nice_gens) {
		Orbits_on_points_not_on_lines =
				SOG->nice_gens->compute_all_point_orbits_schreier_with_print_interval(
						SOG->Surf_A->A,
						A_on_pts_not_on_lines, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_points_not_on_lines =
				SOG->Aut_gens->compute_all_point_orbits_schreier_with_print_interval(
						A_on_pts_not_on_lines,  print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_points_not_on_lines->Forest->nb_orbits
				<< " orbits on points not on lines" << endl;
	}

	if (f_vv) {
		Orbits_on_points_not_on_lines->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_orbits::init_orbits_on_points_not_on_lines done" << endl;
	}
}



void surface_object_orbits::report_orbits_on_lines(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_lines" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_lines;

	SoS_Orbits_on_lines = Orbits_on_lines->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Lines}" << endl;

	//SoS_Orbits_on_lines->print_table_tex(ost);

	{
		other::l1_interfaces::latex_interface L;
		int i, j, idx, len;

		//cout << "set of sets with " << nb_sets << " sets :" << endl;
		for (i = 0; i < SoS_Orbits_on_lines->nb_sets; i++) {

			len = SoS_Orbits_on_lines->Set_size[i];

			ost << "Set " << i << " has size " << len << " : ";

			ost << "$";

			L.lint_set_print_tex(ost, SoS_Orbits_on_lines->Sets[i], len);
			//SO->Surf->print_lines_tex(ost, SoS_Orbits_on_lines->Sets[i], len);

			ost << " = ";
			ost << "\\{ ";
			for (j = 0; j < len; j++) {
				idx = SoS_Orbits_on_lines->Sets[i][j];
				ost << SOG->SO->Surf->Schlaefli->Labels->Line_label_tex[idx];
				if (j < len - 1) {
					ost << ", ";
				}
			}
			ost << " \\}";


			ost << "$";


			ost << "\\\\" << endl;
		}
	}


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_lines done" << endl;
	}
}

void surface_object_orbits::report_orbits_on_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_points;

	SoS_Orbits_on_points = Orbits_on_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Points}" << endl;

	SoS_Orbits_on_points->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_points done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_Eckardt_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Eckardt_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Eckardt_points;

	SoS_Orbits_on_Eckardt_points = Orbits_on_Eckardt_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Eckardt Points}" << endl;

	if (SoS_Orbits_on_Eckardt_points->nb_sets == 0) {
		ost << "There are no orbits on Eckardt points.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_Eckardt_points->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Eckardt_points done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_Double_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Double_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Double_points;

	SoS_Orbits_on_Double_points = Orbits_on_Double_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Double Points}" << endl;

	if (SoS_Orbits_on_Double_points->nb_sets == 0) {
		ost << "There are no orbits on double points.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_Double_points->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Double_points done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_Single_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Single_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Single_points;

	SoS_Orbits_on_Single_points = Orbits_on_Single_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Single Points}" << endl;

	if (SoS_Orbits_on_Single_points->nb_sets == 0) {
		ost << "There are no orbits on single points.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_Single_points->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Single_points done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_Zero_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Zero_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_points_not_on_lines;

	SoS_Orbits_on_points_not_on_lines = Orbits_on_points_not_on_lines->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Zero Points}" << endl;

	if (SoS_Orbits_on_points_not_on_lines->nb_sets == 0) {
		ost << "There are no orbits on points not on lines.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_points_not_on_lines->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Zero_points done" << endl;
	}

}


void surface_object_orbits::report_orbits_on_Hesse_planes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Hesse_planes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Hesse_planes;

	SoS_Orbits_on_Hesse_planes = Orbits_on_Hesse_planes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Hesse Planes}" << endl;

	SoS_Orbits_on_Hesse_planes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Hesse_planes done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_axes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_axes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_axes;

	SoS_Orbits_on_axes = Orbits_on_axes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on axes}" << endl;

	SoS_Orbits_on_axes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_axes done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_Single_Sixes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Single_Sixes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_single_sixes;

	SoS_Orbits_on_single_sixes = Orbits_on_single_sixes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on single sixes}" << endl;

	SoS_Orbits_on_single_sixes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Single_Sixes done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_Double_Sixes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Double_Sixes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_double_sixes;

	SoS_Orbits_on_double_sixes = Orbits_on_double_sixes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on double sixes}" << endl;

	SoS_Orbits_on_double_sixes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_Double_Sixes done" << endl;
	}

}

void surface_object_orbits::report_orbits_on_tritangent_planes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_tritangent_planes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_tritangent_planes;

	SoS_Orbits_on_tritangent_planes = Orbits_on_tritangent_planes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on tritangent planes}" << endl;

	//SoS_Orbits_on_tritangent_planes->print_table_tex(ost);

	{
		other::l1_interfaces::latex_interface L;
		int i, j, idx, len;

		//cout << "set of sets with " << nb_sets << " sets :" << endl;
		for (i = 0; i < SoS_Orbits_on_tritangent_planes->nb_sets; i++) {

			len = SoS_Orbits_on_tritangent_planes->Set_size[i];

			ost << "Set " << i << " has size " << len << " : ";

			ost << "$";

			L.lint_set_print_tex(ost, SoS_Orbits_on_tritangent_planes->Sets[i], len);
			//SO->Surf->print_lines_tex(ost, SoS_Orbits_on_lines->Sets[i], len);

			ost << " = ";
			ost << "\\{ ";
			for (j = 0; j < len; j++) {
				idx = SoS_Orbits_on_tritangent_planes->Sets[i][j];
				//ost << SO->Surf->Schlaefli->Labels->Tritangent_plane_label_tex[idx];

				ost << "\\pi_{" << SOG->SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx] << "}";
				//ost << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx];
				//ost << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx];


				if (j < len - 1) {
					ost << ", ";
				}
			}
			ost << " \\}";


			ost << "$";


			ost << "\\\\" << endl;
		}
	}



	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_tritangent_planes done" << endl;
	}

}


void surface_object_orbits::report_orbits_on_trihedral_pairs(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_trihedral_pairs" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_trihedral_pairs;

	SoS_Orbits_on_trihedral_pairs = Orbits_on_trihedral_pairs->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on trihedral pairs}" << endl;

	SoS_Orbits_on_trihedral_pairs->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_orbits::report_orbits_on_trihedral_pairs done" << endl;
	}

}



void surface_object_orbits::print_summary(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_summary" << endl;
	}

	ost << "\\subsection*{Summary}" << endl;

	string s_orbits_lines;
	string s_orbits_points;
	string s_orbits_Eckardt_points;
	string s_orbits_Double_points;
	string s_orbits_Single_points;
	string s_orbits_Zero_points;
	string s_orbits_Hesse_planes;
	string s_orbits_Axes;
	string s_orbits_single_sixes;
	string s_orbits_double_sixes;
	string s_orbits_tritangent_planes;
	string s_orbits_trihedral_pairs;

	if (f_v) {
		cout << "surface_object_orbits::print_summary 1" << endl;
	}
	if (Orbits_on_lines) {
		Orbits_on_lines->Forest->print_orbit_length_distribution_to_string(s_orbits_lines);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 2" << endl;
	}
	if (Orbits_on_points) {
		Orbits_on_points->Forest->print_orbit_length_distribution_to_string(s_orbits_points);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 3" << endl;
	}
	if (Orbits_on_Eckardt_points) {
		Orbits_on_Eckardt_points->Forest->print_orbit_length_distribution_to_string(s_orbits_Eckardt_points);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 4" << endl;
	}
	if (Orbits_on_Double_points) {
		Orbits_on_Double_points->Forest->print_orbit_length_distribution_to_string(s_orbits_Double_points);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 5" << endl;
	}
	if (Orbits_on_Single_points) {
		Orbits_on_Single_points->Forest->print_orbit_length_distribution_to_string(s_orbits_Single_points);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 6" << endl;
	}
	if (Orbits_on_points_not_on_lines) {
		Orbits_on_points_not_on_lines->Forest->print_orbit_length_distribution_to_string(s_orbits_Zero_points);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 7" << endl;
	}
	if (Orbits_on_Hesse_planes) {
		Orbits_on_Hesse_planes->Forest->print_orbit_length_distribution_to_string(s_orbits_Hesse_planes);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 8" << endl;
	}
	if (Orbits_on_axes) {
		Orbits_on_axes->Forest->print_orbit_length_distribution_to_string(s_orbits_Axes);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 9" << endl;
	}
	if (Orbits_on_single_sixes) {
		Orbits_on_single_sixes->Forest->print_orbit_length_distribution_to_string(s_orbits_single_sixes);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 10" << endl;
	}
	if (Orbits_on_double_sixes) {
		Orbits_on_double_sixes->Forest->print_orbit_length_distribution_to_string(s_orbits_double_sixes);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 11" << endl;
	}
	if (Orbits_on_tritangent_planes) {
		Orbits_on_tritangent_planes->Forest->print_orbit_length_distribution_to_string(s_orbits_tritangent_planes);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 12" << endl;
	}
	if (Orbits_on_trihedral_pairs) {
		Orbits_on_trihedral_pairs->Forest->print_orbit_length_distribution_to_string(s_orbits_trihedral_pairs);
	}
	if (f_v) {
		cout << "surface_object_orbits::print_summary 13" << endl;
	}


	int nb_lines;
	int nb_points;

	if (SOG->SO->Variety_object->Line_sets) {
		nb_lines = SOG->SO->Variety_object->Line_sets->Set_size[0];
	}
	else {
		nb_lines = -1;
	}
	if (SOG->SO->Variety_object->Point_sets) {
		nb_points = SOG->SO->Variety_object->Point_sets->Set_size[0];
	}
	else {
		nb_points = -1;
	}

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Object} & \\mbox{Number}  & \\mbox{Orbit type} \\\\";
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Lines} & " << nb_lines << " & " << s_orbits_lines;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Points on surface} & " << nb_points << " & " << s_orbits_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Singular points} & " << SOG->SO->SOP->nb_singular_pts << " & \\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Eckardt points} & " << SOG->SO->SOP->nb_Eckardt_points << " & " << s_orbits_Eckardt_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Double points} & " << SOG->SO->SOP->nb_Double_points << " & " << s_orbits_Double_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Single points} & " << SOG->SO->SOP->nb_Single_points << " & " << s_orbits_Single_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Points off lines} & " << SOG->SO->SOP->nb_pts_not_on_lines << " & " << s_orbits_Zero_points;
	ost << "\\\\" << endl;

	ost << "\\hline" << endl;
	ost << "\\mbox{Hesse planes} & " << SOG->SO->SOP->nb_Hesse_planes << " & " << s_orbits_Hesse_planes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Axes} & " << SOG->SO->SOP->nb_axes << " & " << s_orbits_Axes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Single sixes} & " << 72 << " & " << s_orbits_single_sixes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Double sixes} & " << 36 << " & " << s_orbits_double_sixes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Tritangent planes} & " << 45 << " & " << s_orbits_tritangent_planes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Trihedral pairs} & " << 120 << " & " << s_orbits_trihedral_pairs;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;




	ost << "\\mbox{Type of points on lines} & ";
	SOG->SO->SOP->Type_pts_on_lines->print_bare_tex(ost, true);
	ost << " & \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of lines on points} & ";
	SOG->SO->SOP->Type_lines_on_point->print_bare_tex(ost, true);
	ost << " & \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
#if 0
	ost << "Points on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_bare_tex(ost, true);
	ost << "$$" << endl;
	ost << "Lines on points:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_bare_tex(ost, true);
	ost << "$$" << endl;
#endif

	if (f_v) {
		cout << "surface_object_orbits::print_summary done" << endl;
	}


}


void surface_object_orbits::print_generators_on_lines(
		ostream &ost,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_generators_on_lines" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_generators_tex_with_point_labels(
			A_on_the_lines,
			ost,
			SOG->Surf->Schlaefli->Labels->Line_label_tex,
			//algebraic_geometry::callback_surface_domain_sstr_line_label,
			SOG->Surf);

}

void surface_object_orbits::print_elements_on_lines(
		std::ostream &ost,
		groups::strong_generators *Aut_gens,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_elements_on_lines" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_elements_latex_ost_with_point_labels(
			A_on_the_lines,
			ost,
			SOG->Surf->Schlaefli->Labels->Line_label_tex,
			max_nb_elements_printed,
			SOG->Surf);

}

void surface_object_orbits::print_elements_on_tritangent_planes(
		std::ostream &ost,
		groups::strong_generators *Aut_gens,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_elements_on_tritangent_planes" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_elements_latex_ost_with_point_labels(
			A_on_tritangent_planes,
			ost,
			SOG->Surf->Schlaefli->Labels->Tritangent_plane_label_tex,
			max_nb_elements_printed,
			SOG->Surf);

}



void surface_object_orbits::print_automorphism_group(
	std::ostream &ost,
	int verbose_level)
// called from surface_object_with_group::cheat_sheet
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_automorphism_group" << endl;
	}


	algebra::ring_theory::longinteger_object go;
	other::l1_interfaces::latex_interface L;

	SOG->Aut_gens->group_order(go);

	ost << "\\section*{Orbits of the automorphism group}" << endl;
	ost << "surface\\_object\\_with\\_group::print\\_automorphism\\_group" << endl;
	ost << "The automorphism group has order " << go << endl;
	ost << "\\bigskip" << endl;
	ost << "\\subsection*{Orbits on points}" << endl;
	//Orbits_on_points->print_and_list_orbits_and_
	//stabilizer_sorted_by_length(ost, true, Surf_A->A, go);
	Orbits_on_points->print_and_list_orbits_with_original_labels_tex(ost);
	Orbits_on_points->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);


	if (f_v) {
		cout << "surface_object_orbits::print_automorphism_group done" << endl;
	}
}



void surface_object_orbits::print_orbits_of_automorphism_group(
	std::ostream &ost,
	int f_print_orbits, std::string &fname_mask,
	other::graphics::draw_options *Opt,
	int max_nb_elements_printed,
	int verbose_level)
// called from surface_object_with_group::cheat_sheet
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_orbits_of_automorphism_group" << endl;
	}


	algebra::ring_theory::longinteger_object go;
	other::l1_interfaces::latex_interface L;

	ost << "\\subsection*{Orbits on Eckardt points}" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_with_original_labels_tex(ost);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_Eckardt_points";

		Orbits_on_Eckardt_points->Forest->make_orbit_trees(ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	Orbits_on_Eckardt_points->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);


	ost << "\\subsection*{Orbits on Double points}" << endl;
	Orbits_on_Double_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on Single points}" << endl;
	Orbits_on_Single_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on points not on lines}" << endl;
	//Orbits_on_points_not_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);
	Orbits_on_points_not_on_lines->print_and_list_orbits_with_original_labels_tex(ost);

	SOG->print_full_del_Pezzo(ost, verbose_level);


	ost << "\\subsection*{Orbits on lines}" << endl;
	Orbits_on_lines->Forest->print_and_list_orbits_tex(ost);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_on_lines";

		Orbits_on_lines->Forest->make_orbit_trees(ost,
				my_fname_mask, Opt,
				verbose_level);
	}


	ost << "\\bigskip" << endl;

	SOG->Surf->Schlaefli->latex_table_of_Schlaefli_labeling_of_lines(ost);

	ost << "\\bigskip" << endl;

#if 0
	Orbits_on_lines->print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		int i, action *default_action,
		strong_generators *gens, std::ostream &ost);
#endif

	Orbits_on_lines->print_and_list_orbits_with_original_labels_tex(ost);
	Orbits_on_lines->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);
	int orbit_idx;
	for (orbit_idx = 0; orbit_idx < Orbits_on_lines->Forest->nb_orbits; orbit_idx++) {

		groups::strong_generators *gens_stab;

		Orbits_on_lines->get_stabilizer_orbit_rep(
			orbit_idx,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			gens_stab);

		ost << "The stabilizer of orbit " << orbit_idx << " in the action on the lines:\\\\" << endl;
		print_elements_on_lines(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		print_elements_on_tritangent_planes(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		FREE_OBJECT(gens_stab);
	}

	int *Decomp_scheme;
	int nb;
	int block_width = 10;
	nb = Orbits_on_lines->Forest->nb_orbits;

	Orbits_on_lines->Forest->get_orbit_decomposition_scheme_of_graph(
			SOG->SO->SOP->Adj_line_intersection_graph,
			SOG->SO->Variety_object->Line_sets->Set_size[0],
			Decomp_scheme,
			0 /*verbose_level*/);

	ost << "\\subsection*{Decomposition scheme of line intersection graph}" << endl;
	ost << "Decomposition scheme of line intersection graph:" << endl;
	L.print_integer_matrix_tex_block_by_block(ost,
			Decomp_scheme, nb, nb, block_width);
	FREE_int(Decomp_scheme);


	if (SOG->SO->Variety_object->Line_sets->Set_size[0] == 27) {

		if (f_v) {
			cout << "surface_object_orbits::print_orbits_of_automorphism_group "
					"before print_orbits_on_schlaefli_related_things" << endl;
		}
		print_orbits_on_schlaefli_related_things(ost,
				f_print_orbits, fname_mask,
				Opt,
				max_nb_elements_printed,
				verbose_level);
		if (f_v) {
			cout << "surface_object_orbits::print_orbits_of_automorphism_group "
					"after print_orbits_on_schlaefli_related_things" << endl;
		}
	}


	ost << "\\clearpage" << endl;

	if (f_v) {
		cout << "surface_object_orbits::print_orbits_of_automorphism_group done" << endl;
	}
}

void surface_object_orbits::print_orbits_on_schlaefli_related_things(
	std::ostream &ost,
	int f_print_orbits, std::string &fname_mask,
	other::graphics::draw_options *Opt,
	int max_nb_elements_printed,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::print_orbits_on_schlaefli_related_things" << endl;
	}

	ost << "\\subsection*{Orbits on single sixes}" << endl;
	Orbits_on_single_sixes->Forest->print_and_list_orbits_tex(ost);

	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_single_sixes";

		Orbits_on_single_sixes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	ost << "\\subsection*{Orbits on double sixes}" << endl;
	Orbits_on_double_sixes->Forest->print_and_list_orbits_tex(ost);
	Orbits_on_double_sixes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);

	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_double_sixes";

		Orbits_on_double_sixes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	int orbit_idx;

	for (orbit_idx = 0; orbit_idx < Orbits_on_lines->Forest->nb_orbits; orbit_idx++) {

		groups::strong_generators *gens_stab;

		Orbits_on_double_sixes->get_stabilizer_orbit_rep(
			orbit_idx,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			gens_stab);

		ost << "The stabilizer of orbit " << orbit_idx << " in the action on double sixes:\\\\" << endl;
		print_elements_on_lines(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		print_elements_on_tritangent_planes(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		FREE_OBJECT(gens_stab);
	}

	ost << "\\subsection*{Orbits on tritangent planes}" << endl;
	Orbits_on_tritangent_planes->Forest->print_and_list_orbits_tex(ost);
	Orbits_on_tritangent_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_tritangent_planes";

		Orbits_on_tritangent_planes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	ost << "\\subsection*{Orbits on Hesse planes}" << endl;
	Orbits_on_Hesse_planes->Forest->print_and_list_orbits_tex(ost);
	Orbits_on_Hesse_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_Hesse_planes";

		Orbits_on_Hesse_planes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}
	Orbits_on_Hesse_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost,
			SOG->Surf_A->A,
			SOG->Aut_gens,
			verbose_level);

	ost << "\\subsection*{Orbits on trihedral pairs}" << endl;
	Orbits_on_trihedral_pairs->Forest->print_and_list_orbits_tex(ost);

	if (f_v) {
		cout << "surface_object_orbits::print_orbits_on_schlaefli_related_things done" << endl;
	}

}

void surface_object_orbits::cheat_sheet_basic(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_basic" << endl;
	}


	algebra::ring_theory::longinteger_object ago;
	SOG->Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic "
				"before Aut_gens->"
				"print_generators_tex" << endl;
	}
	SOG->Aut_gens->print_generators_tex(ost);


	if (SOG->f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		SOG->nice_gens->print_tex(ost);

	}

	ost << "Orbits on Eckardt points:\\\\" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\bigskip" << endl;

	if (SOG->SO->Variety_object->Line_sets->Set_size[0] == 27) {
		ost << "Orbits on half double-sixes:\\\\" << endl;
		int i, idx;

		for (i = 0; i < Orbits_on_single_sixes->Forest->nb_orbits; i++) {

			//ost << "\\bigskip" << endl;
			//ost << "" << endl;
			ost << "Orbit " << i << " / " << Orbits_on_single_sixes->Forest->nb_orbits
					<< " of length " << Orbits_on_single_sixes->Forest->orbit_len[i]
					<< " consists of the following half double sixes:" << endl;


			ost << "$$" << endl;
			L.int_set_print_tex(ost,
				Orbits_on_single_sixes->Forest->orbit +
					Orbits_on_single_sixes->Forest->orbit_first[i],
				Orbits_on_single_sixes->Forest->orbit_len[i]);
			ost << "$$" << endl;

			idx = Orbits_on_single_sixes->Forest->orbit[Orbits_on_single_sixes->Forest->orbit_first[i]];

			ost << "orbit rep:" << endl;
			ost << "$$" << endl;
			SOG->Surf->Schlaefli->Schlaefli_double_six->latex_half_double_six(ost, idx);
			ost << "$$" << endl;

		}
	}

	ost << "\\bigskip" << endl;

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_basic done" << endl;
	}
}


void surface_object_orbits::cheat_sheet_group_elements(
		std::ostream &ost,
		std::string &fname_csv,
		std::string &col_heading,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_group_elements" << endl;
		cout << "surface_object_orbits::cheat_sheet_group_elements "
				"verbose_level = " << verbose_level << endl;
	}

	other::orbiter_kernel_system::file_io Fio;

	other::data_structures::string_tools ST;

	other::data_structures::spreadsheet S;

	S.read_spreadsheet(fname_csv, 0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_group_elements "
				"S.nb_rows = " << S.nb_rows << endl;
		cout << "surface_object_orbits::cheat_sheet_group_elements "
				"S.nb_cols = " << S.nb_cols << endl;
	}


	int idx_input;
	int row;

	int *Data;
	int *data;
	int sz;
	int N;


	N = S.nb_rows - 1;
	Data = NEW_int(N * SOG->Surf_A->A->make_element_size);

	idx_input = S.find_column(col_heading);

	for (row = 0; row < N; row++) {

		string s_data;
		string s_data2;


		s_data = S.get_entry_ij(row + 1, idx_input);

		ST.drop_quotes(s_data, s_data2);

		Int_vec_scan(s_data2, data, sz);
		cout << "read:" << s_data2 << endl;
		if (sz != SOG->Surf_A->A->make_element_size) {
			cout << "data size mismatch" << endl;
			cout << "sz = " << sz << endl;
			cout << "Surf_A->A->make_element_size = " << SOG->Surf_A->A->make_element_size << endl;
			exit(1);
		}
		Int_vec_copy(data,
				Data + row * SOG->Surf_A->A->make_element_size,
				SOG->Surf_A->A->make_element_size);
	}

	int *Elt;
	int i;

	Elt = NEW_int(SOG->Surf_A->A->elt_size_in_int);

	int nb_col = 13;
	int nb;
	string *Table;

	nb = S.nb_cols + nb_col;
	Table = new string[N * nb];

	for (i = 0; i < N; i++) {

		int j;


		for (j = 0; j < S.nb_cols; j++) {

			Table[i * nb + j] = S.get_entry_ij(i + 1, j);
		}
		cout << "Element " << i << " / " << N << " is:" << endl;
		ost << "$" << endl;

		SOG->Surf_A->A->Group_element->make_element(
				Elt, Data + i * SOG->Surf_A->A->make_element_size,
				0 /*verbose_level*/);

		SOG->Surf_A->A->Group_element->element_print_latex(Elt, ost);

#if 0
		actions::action *A_on_points;
		actions::action *A_on_Eckardt_points;
		actions::action *A_on_Double_points;
		actions::action *A_on_Single_points;
		actions::action *A_on_the_lines;
		actions::action *A_single_sixes;
		actions::action *A_double_sixes;
		actions::action *A_on_tritangent_planes;
		actions::action *A_on_Hesse_planes;
		actions::action *A_on_axes;
		actions::action *A_on_trihedral_pairs;
		actions::action *A_on_pts_not_on_lines;
#endif

#if 0
		A_on_points->Group_element->element_order_and_cycle_type_verbose(
				void *elt, int *cycle_type, int verbose_level)
		// cycle_type[i - 1] is the number of cycles of length i for 1 le i le n
#endif

		ost << "$" << endl;
		ost << "\\\\" << endl;
		ost << "\\bigskip" << endl;

		data_structures_groups::vector_ge *gens;

		gens = NEW_OBJECT(data_structures_groups::vector_ge);

		gens->init_single(SOG->Surf_A->A, Elt, verbose_level);


		groups::schreier **Orbits;

		Orbits = (groups::schreier **) NEW_pvoid(nb_col);

		// ToDo:
		if (f_v) {
			cout << "surface_object_orbits::cheat_sheet_group_elements "
					"before compute_all_point_orbits_schreier" << endl;
		}
		Orbits[0] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, SOG->Surf_A->A, print_interval, verbose_level - 2);
		Orbits[1] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_points, print_interval, verbose_level - 2);
		Orbits[2] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_Eckardt_points, print_interval, verbose_level - 2);
		Orbits[3] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_Double_points, print_interval, verbose_level - 2);
		Orbits[4] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_Single_points, print_interval, verbose_level - 2);
		Orbits[5] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_the_lines, print_interval, verbose_level - 2);
		Orbits[6] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_single_sixes, print_interval, verbose_level - 2);
		Orbits[7] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_double_sixes, print_interval, verbose_level - 2);
		Orbits[8] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_tritangent_planes, print_interval, verbose_level - 2);
		Orbits[9] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_Hesse_planes, print_interval, verbose_level - 2);
		Orbits[10] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_axes, print_interval, verbose_level - 2);
		Orbits[11] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_trihedral_pairs, print_interval, verbose_level - 2);
		Orbits[12] = gens->compute_all_point_orbits_schreier_with_print_interval(SOG->Surf_A->A, A_on_pts_not_on_lines, print_interval, verbose_level - 2);

		for (j = 0; j < nb_col; j++) {

			string s;
			other::data_structures::tally *Classify_orbits_by_length;

			Classify_orbits_by_length = NEW_OBJECT(other::data_structures::tally);

			Classify_orbits_by_length->init(Orbits[j]->Forest->orbit_len, Orbits[j]->Forest->nb_orbits, false, 0);

			cout << "j=" << j << " : ";
			s = Classify_orbits_by_length->stringify_bare_tex(false);
			cout << s << endl;

			Table[i * nb + S.nb_cols + j] = "\"" + s + "\"";


			FREE_OBJECT(Classify_orbits_by_length);
		}

		for (j = 0; j < nb_col; j++) {
			FREE_OBJECT(Orbits[j]);
		}

		FREE_pvoid((void **) Orbits);

		if (f_v) {
			cout << "surface_object_orbits::cheat_sheet_group_elements "
					"after compute_all_point_orbits_schreier" << endl;
		}



	}

	string fname;
	string *Headings;
	string headings;

	Headings = new string[nb];


	int j;

	for (j = 0; j < S.nb_cols; j++) {

		Headings[j] = S.get_entry_ij(0, j);
	}
	for (j = 0; j < nb_col; j++) {
		Headings[S.nb_cols + j] = "ORB" + std::to_string(j);
	}

	for (j = 0; j < nb; j++) {
		headings += Headings[j];
		if (j < nb - 1) {
			headings += ",";
		}
	}

	fname = fname_csv;
	ST.chop_off_extension(fname);
	fname += "_properties.csv";

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_group_elements "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(
			fname,
			N, nb, Table,
			headings,
			verbose_level);

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_group_elements "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}

	delete [] Headings;
	delete [] Table;

	FREE_int(Elt);
	FREE_int(Data);

	if (f_v) {
		cout << "surface_object_orbits::cheat_sheet_group_elements "
				"done" << verbose_level << endl;
	}
}



void surface_object_orbits::print_automorphism_group_generators(
		std::ostream &ost,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "surface_object_orbits::print_automorphism_group_generators" << endl;
	}

	if (SOG->Aut_gens == NULL) {
		if (f_v) {
			cout << "surface_object_orbits::print_automorphism_group_generators "
					"the automorphism group is not available" << endl;
		}
		return;
	}
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_orbits::print_automorphism_group_generators "
				"before Aut_gens->print_generators_tex" << endl;
	}
	SOG->Aut_gens->print_generators_tex(ost);

	if (f_v) {
		cout << "surface_object_orbits::print_automorphism_group_generators "
				"before Aut_gens->print_generators_in_different_action_tex" << endl;
		A_on_the_lines->print_info();
	}
	SOG->Aut_gens->print_generators_in_different_action_tex(
			ost, A_on_the_lines);


	if (SOG->f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		SOG->nice_gens->print_tex(ost);

	}


	if (SOG->projectivity_group_gens) {

		if (f_v) {
			cout << "surface_object_orbits::print_automorphism_group_generators "
					"before print_projectivity_group" << endl;
		}

		print_projectivity_group(ost, max_nb_elements_printed, verbose_level - 2);

		if (f_v) {
			cout << "surface_object_orbits::print_automorphism_group_generators "
					"after print_projectivity_group" << endl;
		}

		if (SOG->Syl) {

			if (f_v) {
				cout << "surface_object_orbits::print_automorphism_group_generators "
						"before print_sylow_groups_of_projectivity_group" << endl;
			}

			SOG->print_sylow_groups_of_projectivity_group(ost, verbose_level - 2);

			if (f_v) {
				cout << "surface_object_orbits::print_automorphism_group_generators "
						"after print_sylow_groups_of_projectivity_group" << endl;
			}
		}

	}



}

void surface_object_orbits::print_projectivity_group(
		std::ostream &ost,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_orbits::print_projectivity_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_orbits::print_projectivity_group "
				"projectivity stabilizer" << endl;
	}
	algebra::ring_theory::longinteger_object go;
	SOG->projectivity_group_gens->group_order(go);
	ost << "The projectivity group has order "
			<< go << "\\\\" << endl;
	ost << "The projectivity group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_group::print_projectivity_group "
				"before projectivity_group_gens->"
				"print_generators_tex" << endl;
	}
	SOG->projectivity_group_gens->print_generators_tex(ost);
	SOG->projectivity_group_gens->print_generators_in_different_action_tex(ost, A_on_the_lines);


	ost << "The projectivity group in the action on the lines:\\\\" << endl;
	print_generators_on_lines(
			ost,
			SOG->projectivity_group_gens,
			verbose_level);

#if 1
	ost << "The elements of the projectivity group "
			"in the action on the lines:\\\\" << endl;
	print_elements_on_lines(
			ost,
			SOG->projectivity_group_gens,
			max_nb_elements_printed,
			verbose_level);
#endif



	string label_group;

	label_group = SOG->SO->label_txt + "_proj_grp";
	SOG->projectivity_group_gens->export_group_and_copy_to_latex(label_group,
			ost,
			SOG->projectivity_group_gens->A,
			verbose_level - 2);

	label_group = SOG->SO->label_txt + "_proj_grp_on_lines";
	SOG->projectivity_group_gens->export_group_and_copy_to_latex(label_group,
			ost,
			A_on_the_lines,
			verbose_level - 2);

	label_group = SOG->SO->label_txt + "_proj_grp_on_tritangent_planes";
	SOG->projectivity_group_gens->export_group_and_copy_to_latex(label_group,
			ost,
			A_on_tritangent_planes,
			verbose_level - 2);



	if (f_v) {
		cout << "surface_object_orbits::print_projectivity_group done" << endl;
	}

}


void surface_object_orbits::print_element_in_different_actions(
		std::ostream &ost,
		int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_orbits::print_element_in_different_actions" << endl;
	}

	ost << "Action on points: \\\\" << endl;
	A_on_points->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;


	ost << "Action on Eckardt points: \\\\" << endl;
	A_on_Eckardt_points->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on Double points: \\\\" << endl;
	A_on_Double_points->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on Single points: \\\\" << endl;
	A_on_Single_points->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on lines: \\\\" << endl;
	A_on_the_lines->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on single sixes: \\\\" << endl;
	A_single_sixes->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on tritangent planes: \\\\" << endl;
	A_on_tritangent_planes->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on Hesse planes: \\\\" << endl;
	A_on_Hesse_planes->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;


	ost << "Action on trihedral pairs: \\\\" << endl;
	A_on_trihedral_pairs->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "Action on points not on lines: \\\\" << endl;
	A_on_pts_not_on_lines->Group_element->element_print_as_permutation(Elt, ost);
	ost << "\\bigskip" << endl;
	ost << endl;

	if (f_v) {
		cout << "surface_object_orbits::print_element_in_different_actions done" << endl;
	}
}



}}}}


