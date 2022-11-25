// surface_object_with_action.cpp
// 
// Anton Betten
//
// October 4, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {



surface_object_with_action::surface_object_with_action()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;
	SO = NULL;
	Aut_gens = NULL;

	f_has_nice_gens = FALSE;
	nice_gens = NULL;


	projectivity_group_gens = NULL;
	Syl = NULL;

	A_on_points = NULL;
	A_on_Eckardt_points = NULL;
	A_on_Double_points = NULL;
	A_on_Single_points = NULL;
	A_on_the_lines = NULL;
	A_single_sixes = NULL;
	A_on_tritangent_planes = NULL;
	A_on_Hesse_planes = NULL;
	A_on_trihedral_pairs = NULL;
	A_on_pts_not_on_lines = NULL;


	Orbits_on_points = NULL;
	Orbits_on_Eckardt_points = NULL;
	Orbits_on_Double_points = NULL;
	Orbits_on_Single_points = NULL;
	Orbits_on_lines = NULL;
	Orbits_on_single_sixes = NULL;
	Orbits_on_tritangent_planes = NULL;
	Orbits_on_Hesse_planes = NULL;
	Orbits_on_trihedral_pairs = NULL;
	Orbits_on_points_not_on_lines = NULL;
}

surface_object_with_action::~surface_object_with_action()
{
	if (projectivity_group_gens) {
		FREE_OBJECT(projectivity_group_gens);
	}
	if (Syl) {
		FREE_OBJECT(Syl);
	}
	if (A_on_points) {
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
	if (A_on_tritangent_planes) {
		FREE_OBJECT(A_on_tritangent_planes);
	}
	if (A_on_Hesse_planes) {
		FREE_OBJECT(A_on_Hesse_planes);
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
	if (Orbits_on_tritangent_planes) {
		FREE_OBJECT(Orbits_on_tritangent_planes);
	}
	if (Orbits_on_Hesse_planes) {
		FREE_OBJECT(Orbits_on_Hesse_planes);
	}
	if (Orbits_on_trihedral_pairs) {
		FREE_OBJECT(Orbits_on_trihedral_pairs);
	}
	if (Orbits_on_points_not_on_lines) {
		FREE_OBJECT(Orbits_on_points_not_on_lines);
	}
}

void surface_object_with_action::init_equation(
	surface_with_action *Surf_A, int *eqn,
	groups::strong_generators *Aut_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_equation" << endl;
	}

	surface_object_with_action::Surf_A = Surf_A;
	surface_object_with_action::Aut_gens = Aut_gens;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;

	SO = NEW_OBJECT(algebraic_geometry::surface_object);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(Surf_A->Surf, eqn, verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"after SO->init_equation" << endl;
	}

#if 0
	if (SO->nb_lines != 27) {
		cout << "surface_object_with_action::init_equation "
				"the surface does not have 27 lines" << endl;
		return FALSE;
	}
#endif

	
	compute_projectivity_group(verbose_level);

	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"after compute_orbits_of_automorphism_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_equation done" << endl;
	}
}



void surface_object_with_action::init_with_group(
		surface_with_action *Surf_A,
	long int *Lines, int nb_lines, int *eqn,
	groups::strong_generators *Aut_gens,
	int f_find_double_six_and_rearrange_lines,
	int f_has_nice_gens,
	data_structures_groups::vector_ge *nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_with_group" << endl;
	}

	algebraic_geometry::surface_object *SO;

	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (nb_lines == 27) {
		if (f_v) {
			cout << "surface_object_with_action::init_with_group "
					"before SO->init_with_27_lines" << endl;
		}
		SO->init_with_27_lines(Surf_A->Surf, Lines, eqn,
				f_find_double_six_and_rearrange_lines, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::init_with_group "
					"after SO->init_with_27_lines" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "surface_object_with_action::init_with_group "
					"before SO->init_equation" << endl;
		}
		SO->init_equation(Surf_A->Surf, eqn,
				verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::init_with_group "
					"after SO->init_equation" << endl;
		}

	}

	if (f_v) {
		cout << "surface_object_with_action::init_with_group "
				"before SO->init_with_surface_object" << endl;
	}

	init_with_surface_object(Surf_A,
			SO,
			Aut_gens,
			f_has_nice_gens, nice_gens,
			verbose_level);

	if (f_v) {
		cout << "surface_object_with_action::init_with_group "
				"after SO->init_with_surface_object" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_with_group done" << endl;
	}
}


void surface_object_with_action::init_with_surface_object(
		surface_with_action *Surf_A,
		algebraic_geometry::surface_object *SO,
		groups::strong_generators *Aut_gens,
		int f_has_nice_gens,
		data_structures_groups::vector_ge *nice_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object" << endl;
	}

	surface_object_with_action::Surf_A = Surf_A;
	surface_object_with_action::f_has_nice_gens = f_has_nice_gens;
	surface_object_with_action::nice_gens = nice_gens;
	surface_object_with_action::SO = SO;
	surface_object_with_action::Aut_gens = Aut_gens;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;

	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object "
				"testing Aut_gens" << endl;
	}
	Aut_gens->test_if_set_is_invariant_under_given_action(
			Surf_A->A2, SO->Lines, SO->nb_lines, verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object "
				"testing Aut_gens done" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object "
				"before compute_projectivity_group" << endl;
	}
	compute_projectivity_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object "
				"after compute_projectivity_group" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object "
				"after compute_orbits_of_automorphism_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_with_surface_object done" << endl;
	}
}


void surface_object_with_action::init_surface_object(
	surface_with_action *Surf_A, algebraic_geometry::surface_object *SO,
	groups::strong_generators *Aut_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_surface_object" << endl;
	}
	surface_object_with_action::Surf_A = Surf_A;
	surface_object_with_action::SO = SO;
	surface_object_with_action::Aut_gens = Aut_gens;


	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;


	
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"before compute_projectivity_group" << endl;
	}
	compute_projectivity_group(verbose_level - 5);
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"after compute_projectivity_group" << endl;
	}





	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"after compute_orbits_of_automorphism_group" << endl;
	}
	
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"done" << endl;
	}
}

void surface_object_with_action::compute_projectivity_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::compute_projectivity_group" << endl;
		cout << "surface_object_with_action::compute_projectivity_group "
				"verbose_level=" << verbose_level << endl;
	}

	Surf_A->A->compute_projectivity_subgroup(projectivity_group_gens,
			Aut_gens, verbose_level);




	if (f_v) {
		cout << "surface_object_with_action::compute_projectivity_group "
				"computing Sylow structure" << endl;
	}
	// compute the Sylow structure:
	groups::sims *S = NULL;

	if (projectivity_group_gens) {
		S = projectivity_group_gens->create_sims(0 /*verbose_level */);
	}
	else {
		if (Aut_gens) {
			S = Aut_gens->create_sims(0 /*verbose_level */);
		}
	}

	if (S) {
		if (f_v) {
			cout << "surface_object_with_action::compute_projectivity_group "
					"before Syl->init" << endl;
		}
		Syl = NEW_OBJECT(groups::sylow_structure);
		Syl->init(S, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::compute_projectivity_group "
					"after Syl->init" << endl;
		}
	}


	if (f_v) {
		cout << "surface_object_with_action::compute_projectivity_group done" << endl;
	}
}

void surface_object_with_action::compute_orbits_of_automorphism_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group" << endl;
	}

	// orbits on points:
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on points" << endl;
	}
	init_orbits_on_points(verbose_level - 1);


	// orbits on Eckardt points:
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on Eckardt points" << endl;
	}
	init_orbits_on_Eckardt_points(verbose_level - 1);


	// orbits on Double points:
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on double points" << endl;
	}
	init_orbits_on_Double_points(verbose_level - 1);

	// orbits on Single points:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on single points" << endl;
	}
	init_orbits_on_Single_points(verbose_level - 1);


	// orbits on lines:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on lines" << endl;
	}
	init_orbits_on_lines(verbose_level);



	if (SO->nb_lines == 27) {

		// orbits on half double sixes:

		if (f_v) {
			cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
					"orbits on half double sixes" << endl;
		}
		init_orbits_on_half_double_sixes(verbose_level);


		// orbits on tritangent planes:

		if (f_v) {
			cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
					"orbits on tritangent planes" << endl;
		}
		init_orbits_on_tritangent_planes(verbose_level);


		// orbits on Hesse planes:

		if (f_v) {
			cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
					"orbits on Hesse planes" << endl;
		}
		init_orbits_on_Hesse_planes(verbose_level);


		// orbits on trihedral pairs:

		if (f_v) {
			cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
					"orbits on trihedral pairs" << endl;
		}
		init_orbits_on_trihedral_pairs(verbose_level);
	}



	// orbits on points not on lines:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on points not on lines" << endl;
	}
	init_orbits_on_points_not_on_lines(verbose_level);


	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action action on points:" << endl;
	}
	A_on_points = Surf_A->A->restricted_action(
			SO->Pts, SO->nb_pts, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_with_action action "
				"on points done" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_points "
				"computing orbits on points:" << endl;
	}
	if (f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_with_action::init_orbits_on_points "
					"computing orbits on points using nice gens:" << endl;
		}
		Orbits_on_points = nice_gens->orbits_on_points_schreier(
				A_on_points, 0 /*verbose_level*/);

	}
	else {
		if (f_v) {
			cout << "surface_object_with_action::init_orbits_on_points "
					"computing orbits on points using Aut_gens:" << endl;
		}
		Orbits_on_points = Aut_gens->orbits_on_points_schreier(
				A_on_points, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_points "
				"We found " << Orbits_on_points->nb_orbits
				<< " orbits on points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_points done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_Eckardt_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Eckardt_points" << endl;
	}

	if (f_v) {
		cout << "creating action on Eckardt points:" << endl;
	}
	A_on_Eckardt_points = Surf_A->A->restricted_action(
			SO->SOP->Eckardt_points, SO->SOP->nb_Eckardt_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Eckardt points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Eckardt points:" << endl;
	}
	if (f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_with_action::init_orbits_on_Eckardt_points "
					"computing orbits on points using nice gens:" << endl;
		}
		Orbits_on_Eckardt_points = nice_gens->orbits_on_points_schreier(
				A_on_Eckardt_points, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Eckardt_points = Aut_gens->orbits_on_points_schreier(
				A_on_Eckardt_points, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Eckardt_points->nb_orbits
				<< " orbits on Eckardt points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Eckardt_points done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_Double_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Double_points" << endl;
	}

	if (f_v) {
		cout << "creating action on Double points:" << endl;
	}
	A_on_Double_points = Surf_A->A->restricted_action(
			SO->SOP->Double_points, SO->SOP->nb_Double_points,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Double points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Double points:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_Double_points = nice_gens->orbits_on_points_schreier(
				A_on_Double_points, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Double_points = Aut_gens->orbits_on_points_schreier(
				A_on_Double_points, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Double_points->nb_orbits
				<< " orbits on Double points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Double_points done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_Single_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Single_points" << endl;
	}

	if (f_v) {
		cout << "creating action on Single points:" << endl;
	}
	A_on_Single_points = Surf_A->A->restricted_action(
			SO->SOP->Single_points, SO->SOP->nb_Single_points,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Single points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Single points:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_Single_points = nice_gens->orbits_on_points_schreier(
				A_on_Single_points, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Single_points = Aut_gens->orbits_on_points_schreier(
				A_on_Single_points, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Single_points->nb_orbits
				<< " orbits on Single points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Single_points done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_lines" << endl;
	}

	if (f_v) {
		cout << "creating restricted action "
				"on the lines:" << endl;
	}
	A_on_the_lines = Surf_A->A2->restricted_action(
			SO->Lines, SO->nb_lines, 0 /*verbose_level*/);
	if (f_v) {
		cout << "creating restricted action "
				"on the lines done" << endl;
	}

	if (f_v) {
		cout << "computing orbits on lines:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_lines = nice_gens->orbits_on_points_schreier(
				A_on_the_lines, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_lines = Aut_gens->orbits_on_points_schreier(
				A_on_the_lines, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_lines->nb_orbits
				<< " orbits on lines" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_lines done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_half_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_half_double_sixes" << endl;
	}

	if (f_v) {
		cout << "creating action on half double sixes:" << endl;
	}
	A_single_sixes = A_on_the_lines->create_induced_action_on_sets(
			72, 6, Surf->Schlaefli->Double_six, 0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on half double sixes done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on single sixes:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_single_sixes = nice_gens->orbits_on_points_schreier(
				A_single_sixes, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_single_sixes = Aut_gens->orbits_on_points_schreier(
				A_single_sixes, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "computing orbits on single sixes done" << endl;
	}
	if (f_v) {
		cout << "We found " << Orbits_on_single_sixes->nb_orbits
				<< " orbits on single sixes" << endl;
	}

	//nb_orbits_on_single_sixes = Orbits_on_single_sixes->nb_orbits;

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_half_double_sixes done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_tritangent_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_tritangent_planes" << endl;
	}

	if (f_v) {
		cout << "creating action on tritangent planes:" << endl;
		cout << "SO->SOP->nb_tritangent_planes = "
				<< SO->SOP->nb_tritangent_planes << endl;
	}
	A_on_tritangent_planes = A_on_the_lines->create_induced_action_on_sets(
			SO->SOP->nb_tritangent_planes, 3,
			//SO->Lines_in_tritangent_planes,
			Surf->Schlaefli->Lines_in_tritangent_planes,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "action on tritangent planes done" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_tritangent_planes = nice_gens->orbits_on_points_schreier(
				A_on_tritangent_planes, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_tritangent_planes = Aut_gens->orbits_on_points_schreier(
				A_on_tritangent_planes, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_tritangent_planes->nb_orbits
				<< " orbits on the set of " << SO->SOP->nb_tritangent_planes
				<< " tritangent planes" << endl;
	}

	Orbits_on_tritangent_planes->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_tritangent_planes done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_Hesse_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Hesse_planes" << endl;
	}

	if (f_v) {
		cout << "creating action on Hesse planes:" << endl;
		cout << "SO->SOP->nb_Hesse_planes = "
				<< SO->SOP->nb_Hesse_planes << endl;
	}
	A_on_Hesse_planes = Surf_A->A_on_planes->restricted_action(
			SO->SOP->Hesse_planes, SO->SOP->nb_Hesse_planes, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action on Hesse planes done" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_Hesse_planes = nice_gens->orbits_on_points_schreier(
				A_on_Hesse_planes, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Hesse_planes = Aut_gens->orbits_on_points_schreier(
				A_on_Hesse_planes, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Hesse_planes->nb_orbits
				<< " orbits on the set of " << SO->SOP->nb_Hesse_planes
				<< " Hesse planes" << endl;
	}

	Orbits_on_Hesse_planes->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_Hesse_planes done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_trihedral_pairs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "creating action on trihedral pairs:" << endl;
	}
	A_on_trihedral_pairs =
			A_on_tritangent_planes->create_induced_action_on_sets(
					120, 6,
					Surf->Schlaefli->Trihedral_to_Eckardt,
					0 /*verbose_level*/);
	if (f_v) {
		cout << "action on trihedral pairs created" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_trihedral_pairs = nice_gens->orbits_on_points_schreier(
				A_on_trihedral_pairs, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_trihedral_pairs = Aut_gens->orbits_on_points_schreier(
				A_on_trihedral_pairs, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_trihedral_pairs->nb_orbits
				<< " orbits on trihedral pairs" << endl;
	}

	Orbits_on_trihedral_pairs->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_trihedral_pairs done" << endl;
	}
}

void surface_object_with_action::init_orbits_on_points_not_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_points_not_on_lines" << endl;
	}

	if (f_v) {
		cout << "creating action on points not on lines:" << endl;
	}
	A_on_pts_not_on_lines = Surf_A->A->restricted_action(
			SO->SOP->Pts_not_on_lines, SO->SOP->nb_pts_not_on_lines,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on points not on lines done" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_points_not_on_lines =
				nice_gens->orbits_on_points_schreier(
						A_on_pts_not_on_lines,  0 /*verbose_level*/);
	}
	else {
		Orbits_on_points_not_on_lines =
				Aut_gens->orbits_on_points_schreier(
						A_on_pts_not_on_lines,  0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_points_not_on_lines->nb_orbits
				<< " orbits on points not on lines" << endl;
	}

	Orbits_on_points_not_on_lines->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_points_not_on_lines done" << endl;
	}
}


void surface_object_with_action::print_generators_on_lines(
		ostream &ost,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::print_generators_on_lines" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_generators_tex_with_print_point_function(
			A_on_the_lines,
			ost,
			algebraic_geometry::callback_surface_domain_sstr_line_label,
			Surf);

}

void surface_object_with_action::print_elements_on_lines(
		ostream &ost,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::print_elements_on_lines" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_elements_latex_ost_with_print_point_function(
			A_on_the_lines,
			ost,
			algebraic_geometry::callback_surface_domain_sstr_line_label,
			Surf);

}

void surface_object_with_action::print_automorphism_group(
	std::ostream &ost,
	int f_print_orbits, std::string &fname_mask,
	graphics::layered_graph_draw_options *Opt,
	int verbose_level)
{
	ring_theory::longinteger_object go;
	orbiter_kernel_system::latex_interface L;

	Aut_gens->group_order(go);
	
	ost << "\\section*{Orbits of the automorphism group}" << endl;
	ost << "The automorphism group has order " << go << endl;
	ost << "\\bigskip" << endl;
	ost << "\\subsection*{Orbits on points}" << endl;
	//Orbits_on_points->print_and_list_orbits_and_
	//stabilizer_sorted_by_length(ost, TRUE, Surf_A->A, go);
	Orbits_on_points->print_and_list_orbits_with_original_labels_tex(ost);



	ost << "\\subsection*{Orbits on Eckardt points}" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_with_original_labels_tex(ost);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask.assign(fname_mask);
		my_fname_mask.append("_Eckardt_points");

		Orbits_on_Eckardt_points->make_orbit_trees(ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	Orbits_on_Eckardt_points->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);


	ost << "\\subsection*{Orbits on Double points}" << endl;
	Orbits_on_Double_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on Single points}" << endl;
	Orbits_on_Single_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on points not on lines}" << endl;
	//Orbits_on_points_not_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);
	Orbits_on_points_not_on_lines->print_and_list_orbits_with_original_labels_tex(ost);

	print_full_del_Pezzo(ost, verbose_level);


	ost << "\\subsection*{Orbits on lines}" << endl;
	Orbits_on_lines->print_and_list_orbits_tex(ost);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask.assign(fname_mask);
		my_fname_mask.append("_on_lines");

		Orbits_on_lines->make_orbit_trees(ost,
				my_fname_mask, Opt,
				verbose_level);
	}


	ost << "\\bigskip" << endl;

	Surf->Schlaefli->latex_table_of_Schlaefli_labeling_of_lines(ost);

	ost << "\\bigskip" << endl;

#if 0
	Orbits_on_lines->print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		int i, action *default_action,
		strong_generators *gens, std::ostream &ost);
#endif

	Orbits_on_lines->print_and_list_orbits_with_original_labels_tex(ost);

	int *Decomp_scheme;
	int nb;
	int block_width = 10;
	nb = Orbits_on_lines->nb_orbits;
	Orbits_on_lines->get_orbit_decomposition_scheme_of_graph(
			SO->SOP->Adj_line_intersection_graph, SO->nb_lines, Decomp_scheme,
			0 /*verbose_level*/);
	ost << "\\subsection*{Decomposition scheme of line intersection graph}" << endl;
	ost << "Decomposition scheme of line intersection graph:" << endl;
	L.print_integer_matrix_tex_block_by_block(ost,
			Decomp_scheme, nb, nb, block_width);
	FREE_int(Decomp_scheme);
	

	if (SO->nb_lines == 27) {
		ost << "\\subsection*{Orbits on single sixes}" << endl;
		Orbits_on_single_sixes->print_and_list_orbits_tex(ost);

		if (f_print_orbits) {


			string my_fname_mask;

			my_fname_mask.assign(fname_mask);
			my_fname_mask.append("_single_sixes");

			Orbits_on_single_sixes->make_orbit_trees(ost,
					my_fname_mask, Opt,
					verbose_level);
		}
	

		ost << "\\subsection*{Orbits on tritangent planes}" << endl;
		Orbits_on_tritangent_planes->print_and_list_orbits_tex(ost);
		if (f_print_orbits) {

			string my_fname_mask;

			my_fname_mask.assign(fname_mask);
			my_fname_mask.append("_tritangent_planes");

			Orbits_on_tritangent_planes->make_orbit_trees(ost,
					my_fname_mask, Opt,
					verbose_level);
		}

		ost << "\\subsection*{Orbits on Hesse planes}" << endl;
		Orbits_on_Hesse_planes->print_and_list_orbits_tex(ost);
		if (f_print_orbits) {

			string my_fname_mask;

			my_fname_mask.assign(fname_mask);
			my_fname_mask.append("_Hesse_planes");

			Orbits_on_Hesse_planes->make_orbit_trees(ost,
					my_fname_mask, Opt,
					verbose_level);
		}
		Orbits_on_Hesse_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
				ost, Surf_A->A, Aut_gens,
				verbose_level);

		ost << "\\subsection*{Orbits on trihedral pairs}" << endl;
		Orbits_on_trihedral_pairs->print_and_list_orbits_tex(ost);
	}


	ost << "\\clearpage" << endl;

}

void surface_object_with_action::cheat_sheet_basic(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic" << endl;
	}


	ring_theory::longinteger_object ago;
	Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic "
				"before Aut_gens->"
				"print_generators_tex" << endl;
	}
	Aut_gens->print_generators_tex(ost);


	if (f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		nice_gens->print_tex(ost);

	}

	ost << "Orbits on Eckardt points:\\\\" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\bigskip" << endl;

	if (SO->nb_lines == 27) {
		ost << "Orbits on half double-sixes:\\\\" << endl;
		int i, idx;

		for (i = 0; i < Orbits_on_single_sixes->nb_orbits; i++) {

			//ost << "\\bigskip" << endl;
			//ost << "" << endl;
			ost << "Orbit " << i << " / " << Orbits_on_single_sixes->nb_orbits
					<< " of length " << Orbits_on_single_sixes->orbit_len[i]
					<< " consists of the following half double sixes:" << endl;


			ost << "$$" << endl;
			L.int_set_print_tex(ost,
				Orbits_on_single_sixes->orbit +
					Orbits_on_single_sixes->orbit_first[i],
				Orbits_on_single_sixes->orbit_len[i]);
			ost << "$$" << endl;

			idx = Orbits_on_single_sixes->orbit[Orbits_on_single_sixes->orbit_first[i]];

			ost << "orbit rep:" << endl;
			ost << "$$" << endl;
			Surf->Schlaefli->latex_half_double_six(ost, idx);
			ost << "$$" << endl;

		}
	}

	ost << "\\bigskip" << endl;

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic done" << endl;
	}
}

void surface_object_with_action::cheat_sheet(std::ostream &ost,
		std::string &label_txt,
		std::string &label_tex,
		int f_print_orbits, std::string &fname_mask,
		graphics::layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet" << endl;
		cout << "surface_object_with_action::cheat_sheet verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_equation" << endl;
	}
	SO->SOP->print_equation(ost);
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"after SO->print_equation" << endl;
	}


	ring_theory::longinteger_object ago;
	Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;




	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before print_everything" << endl;
	}

	print_everything(ost, verbose_level - 1);

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"after print_everything" << endl;
	}

	print_automorphism_group_generators(ost, verbose_level);



	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before print_automorphism_group" << endl;
	}
	print_automorphism_group(ost, f_print_orbits, fname_mask, Opt, verbose_level - 1);


#if 0
	if (SO->nb_pts_not_on_lines) {

		if (f_v) {
			cout << "surface_object_with_action::cheat_sheet "
					"before cheat_sheet_quartic_curve" << endl;
		}
		cheat_sheet_quartic_curve(ost,
			label_txt, label_tex, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::cheat_sheet "
					"after cheat_sheet_quartic_curve" << endl;
		}

	}
#endif

	ost << "\\clearpage\\subsection*{The Elements of "
			"the Automorphism Group}" << endl;
	Aut_gens->print_elements_latex_ost(ost);

	ost << "\\clearpage\\subsection*{The Group Table}" << endl;
	long int go;
	int block_width = 12;

	go = Aut_gens->group_order_as_lint();
	if (go < 50) {
		orbiter_kernel_system::latex_interface L;
		int *Table;
		Aut_gens->create_group_table(Table, go, verbose_level - 1);
		L.print_integer_matrix_tex_block_by_block(ost,
				Table, go, go, block_width);
		FREE_int(Table);
	}
	else {
		ost << "Too big to print." << endl;
	}


	Aut_gens->export_group_and_copy_to_latex(label_txt,
			ost,
			A_on_the_lines,
			verbose_level - 2);


	if (Aut_gens->A->degree < 500) {

		Aut_gens->export_group_and_copy_to_latex(label_txt,
				ost,
				Aut_gens->A,
				verbose_level - 2);

	}
	else {
		cout << "permutation degree is too large, "
				"skipping export to magma and GAP" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet done" << endl;
	}


}

void surface_object_with_action::print_automorphism_group_generators(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "surface_object_with_action::print_automorphism_group_generators" << endl;
	}

	if (Aut_gens == NULL) {
		if (f_v) {
			cout << "surface_object_with_action::print_automorphism_group_generators "
					"the automorphism group is not available" << endl;
		}
		return;
	}
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_action::print_automorphism_group_generators "
				"before Aut_gens->print_generators_tex" << endl;
	}
	Aut_gens->print_generators_tex(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_automorphism_group_generators "
				"before Aut_gens->print_generators_in_different_action_tex" << endl;
		A_on_the_lines->print_info();
	}
	Aut_gens->print_generators_in_different_action_tex(ost, A_on_the_lines);


	if (f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		nice_gens->print_tex(ost);

	}


	if (projectivity_group_gens) {
		if (f_v) {
			cout << "surface_object_with_action::print_automorphism_group_generators "
					"projectivity stabilizer" << endl;
		}
		ring_theory::longinteger_object go;
		projectivity_group_gens->group_order(go);
		ost << "The projectivity group has order "
				<< go << "\\\\" << endl;
		ost << "The projectivity group is generated by:\\\\" << endl;
		if (f_v) {
			cout << "surface_object_with_action::print_automorphism_group_generators "
					"before projectivity_group_gens->"
					"print_generators_tex" << endl;
		}
		projectivity_group_gens->print_generators_tex(ost);
		projectivity_group_gens->print_generators_in_different_action_tex(ost, A_on_the_lines);


		ost << "The projectivity group in the action on the lines:\\\\" << endl;
		print_generators_on_lines(
				ost,
				projectivity_group_gens,
				verbose_level);

#if 1
		ost << "The elements of the projectivity group "
				"in the action on the lines:\\\\" << endl;
		print_elements_on_lines(
				ost,
				projectivity_group_gens,
				verbose_level);
#endif

		string label_group;

		label_group.assign("label_txt_proj_grp");
		projectivity_group_gens->export_group_and_copy_to_latex(label_group,
				ost,
				projectivity_group_gens->A,
				verbose_level - 2);

		label_group.assign("label_txt_proj_grp_on_lines");
		projectivity_group_gens->export_group_and_copy_to_latex(label_group,
				ost,
				A_on_the_lines,
				verbose_level - 2);

		label_group.assign("label_txt_proj_grp_on_tritangent_planes");
		projectivity_group_gens->export_group_and_copy_to_latex(label_group,
				ost,
				A_on_tritangent_planes,
				verbose_level - 2);



	}

	if (Syl && projectivity_group_gens) {
		if (f_v) {
			cout << "surface_object_with_action::print_automorphism_group_generators "
					"Sylow subgroups" << endl;
		}
		int idx;

		for (idx = 0; idx < Syl->nb_primes; idx++) {
			if (f_v) {
				cout << "surface_object_with_action::print_automorphism_group_generators "
						"idx=" << idx << " / " << Syl->nb_primes << endl;
			}
			ost << "The " << Syl->primes[idx]
				<< "-Sylow subgroup is generated by:\\\\" << endl;
			Syl->Sub[idx].SG->print_generators_tex(ost);


			if (f_v) {
				cout << "surface_object_with_action::print_automorphism_group_generators "
						"idx=" << idx << " / " << Syl->nb_primes
						<< " making label_group" << endl;
			}

			string label_group;
			char str[1000];


			label_group.assign("label_txt_proj_grp_syl_");
			snprintf(str, sizeof(str), "%d", Syl->primes[idx]);
			label_group.append(str);

			if (f_v) {
				cout << "surface_object_with_action::print_automorphism_group_generators "
						"idx=" << idx << " / " << Syl->nb_primes
						<< " label_group=" << label_group << endl;
			}

			if (f_v) {
				cout << "surface_object_with_action::print_automorphism_group_generators "
						"idx=" << idx << " / " << Syl->nb_primes
						<< " before export_group_and_copy_to_latex" << endl;
			}

			Syl->Sub[idx].SG->export_group_and_copy_to_latex(label_group,
					ost,
					projectivity_group_gens->A,
					verbose_level - 2);

			label_group.assign("label_txt_proj_grp_syl_");
			snprintf(str, sizeof(str), "%d", Syl->primes[idx]);
			label_group.append(str);
			label_group.append("_on_lines");

			if (f_v) {
				cout << "surface_object_with_action::print_automorphism_group_generators "
						"idx=" << idx << " / " << Syl->nb_primes
						<< " label_group=" << label_group << endl;
			}

			if (f_v) {
				cout << "surface_object_with_action::print_automorphism_group_generators "
						"idx=" << idx << " / " << Syl->nb_primes
						<< " before export_group_and_copy_to_latex" << endl;
			}
			Syl->Sub[idx].SG->export_group_and_copy_to_latex(label_group,
					ost,
					A_on_the_lines,
					verbose_level - 2);

		}
	}


}


void surface_object_with_action::investigate_surface_and_write_report(
		graphics::layered_graph_draw_options *Opt,
		actions::action *A,
		surface_create *SC,
		cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report" << endl;
	}

	string fname;
	string fname_mask;
	string label;
	string label_tex;


	fname.assign("surface_");
	fname.append(SC->prefix);
	fname.append("_with_group.tex");


	label.assign("surface_");
	label.append(SC->label_txt);

	label_tex.assign("surface_");
	label_tex.append(SC->label_tex);

	fname_mask.assign("surface_");
	fname_mask.append(SC->prefix);
	fname_mask.append("_orbit_%d");

	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		L.head_easy(fp);

		investigate_surface_and_write_report2(
					fp,
					Opt,
					A,
					SC,
					Six_arcs,
					fname_mask,
					label,
					label_tex,
					verbose_level);


		L.foot(fp);
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


}




void surface_object_with_action::investigate_surface_and_write_report2(
		std::ostream &ost,
		graphics::layered_graph_draw_options *Opt,
		actions::action *A,
		surface_create *SC,
		cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs,
		std::string &fname_mask,
		std::string &label,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report2" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report2 "
				"before cheat_sheet" << endl;
	}
	ost << "\\section{The Cubic Surface " << SC->label_tex
			<< " over $\\mathbb F_{" << SC->F->q << "}$}" << endl;

	cheat_sheet(ost,
		label,
		label_tex,
		TRUE /* f_print_orbits */,
		fname_mask /* const char *fname_mask*/,
		Opt,
		verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report2 "
				"after cheat_sheet" << endl;
	}



	ost << "\\bigskip" << endl;

	ost << "\\section{The Finite Field $\\mathbb F_{" << SC->F->q << "}$}" << endl;
	SC->F->cheat_sheet(ost, verbose_level);





	ost << "\\setlength{\\parindent}{0pt}" << endl;

#if 0
	if (f_surface_clebsch) {

		if (f_v) {
			cout << "surface_object_with_action::investigate_surface_and_write_report2 f_surface_clebsch" << endl;
		}

		//surface_object *SO;
		//SO = SoA->SO;

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
		ost << "\\section{Points on the surface}" << endl;
		ost << endl;

		SO->SOP->print_affine_points_in_source_code(ost);


		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		ost << "\\section{Clebsch maps}" << endl;

		SC->Surf->Schlaefli->latex_table_of_clebsch_maps(ost);


		ost << endl;
		ost << "\\clearpage" << endl;
		ost << endl;



		ost << "\\section{Six-arcs not on a conic}" << endl;
		ost << endl;


		//ost << "The six-arcs not on a conic are:\\\\" << endl;
		Six_arcs->report_latex(ost);




	}
	else {
		if (f_v) {
			cout << "surface_object_with_action::investigate_surface_and_write_report2 !f_surface_clebsch" << endl;
		}

	}



	if (f_surface_quartic) {

		if (f_v) {
			cout << "surface_object_with_action::investigate_surface_and_write_report2 f_surface_quartic" << endl;
		}

		{
			ofstream ost_quartics("quartics.txt");



			all_quartic_curves(ost, ost_quartics, verbose_level);
		}

	}
	else {
		if (f_v) {
			cout << "surface_object_with_action::investigate_surface_and_write_report2 !f_surface_quartic" << endl;
		}


	}




	if (f_surface_codes) {

		if (f_v) {
			cout << "surface_object_with_action::investigate_surface_and_write_report2 f_surface_codes" << endl;
		}

		homogeneous_polynomial_domain *HPD;

		HPD = NEW_OBJECT(homogeneous_polynomial_domain);

		HPD->init(SC->F, 3, 2 /* degree */,
				TRUE /* f_init_incidence_structure */,
				t_PART,
				verbose_level);

		action *A_on_poly;

		A_on_poly = NEW_OBJECT(action);
		A_on_poly->induced_action_on_homogeneous_polynomials(A,
			HPD,
			FALSE /* f_induce_action */, NULL,
			verbose_level);

		cout << "created action A_on_poly" << endl;
		A_on_poly->print_info();

		schreier *Sch;
		longinteger_object full_go;

		//Sch = new schreier;
		//A2->all_point_orbits(*Sch, verbose_level);

		cout << "computing orbits:" << endl;

		Sch = A->Strong_gens->orbits_on_points_schreier(A_on_poly, verbose_level);

		//SC->Sg->
		//Sch = SC->Sg->orbits_on_points_schreier(A_on_poly, verbose_level);

		orbit_transversal *T;

		A->group_order(full_go);
		T = NEW_OBJECT(orbit_transversal);

		cout << "before T->init_from_schreier" << endl;

		T->init_from_schreier(
				Sch,
				A,
				full_go,
				verbose_level);

		cout << "after T->init_from_schreier" << endl;

		Sch->print_orbit_reps(cout);

		cout << "orbit reps:" << endl;

		ost << "\\section{Orbits on conics}" << endl;
		ost << endl;

		T->print_table_latex(
				ost,
				TRUE /* f_has_callback */,
				HPD_callback_print_function2,
				HPD /* callback_data */,
				TRUE /* f_has_callback */,
				HPD_callback_print_function,
				HPD /* callback_data */,
				verbose_level);


	}
	else {
		if (f_v) {
			cout << "surface_object_with_action::investigate_surface_and_write_report2 !f_surface_codes" << endl;
		}


	}
#endif



	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report2 done" << endl;
	}
}

void surface_object_with_action::all_quartic_curves(
		std::string &surface_label_txt,
		std::string &surface_label_tex,
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int f_TDO = FALSE;

	if (f_v) {
		cout << "surface_object_with_action::all_quartic_curves surface_label_txt=" << surface_label_txt << endl;
	}
	int pt_orbit;


	ost << "Orbits on points not on lines nb orbits = " << Orbits_on_points_not_on_lines->nb_orbits << "\\\\" << endl;

	for (pt_orbit = 0; pt_orbit < Orbits_on_points_not_on_lines->nb_orbits; pt_orbit++) {

		ost << "\\section{Quartic curve associated with orbit " << pt_orbit
				<< " / " << Orbits_on_points_not_on_lines->nb_orbits << "}" << endl;


		quartic_curves::quartic_curve_from_surface *QC;

		QC = NEW_OBJECT(quartic_curves::quartic_curve_from_surface);

		QC->init(this, verbose_level);

		QC->init_labels(surface_label_txt, surface_label_tex, verbose_level);


		if (f_v) {
			cout << "surface_object_with_action::all_quartic_curves before QC->quartic" << endl;
		}
		QC->quartic(pt_orbit, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::all_quartic_curves after QC->quartic" << endl;
		}

		// the quartic curve is now in QC->curve
		// as a Surf->Poly4_x123

		if (f_v) {
			cout << "surface_object_with_action::all_quartic_curves before QC->compute_stabilizer" << endl;
		}
		QC->compute_stabilizer(verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::all_quartic_curves after QC->compute_stabilizer" << endl;
		}


		if (f_v) {
			cout << "surface_object_with_action::all_quartic_curves before QC->cheat_sheet_quartic_curve" << endl;
		}
		QC->cheat_sheet_quartic_curve(ost, f_TDO, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::all_quartic_curves after QC->cheat_sheet_quartic_curve" << endl;
		}

		FREE_OBJECT(QC);
	}
	if (f_v) {
		cout << "surface_object_with_action::all_quartic_curves done" << endl;
	}
}

void surface_object_with_action::export_all_quartic_curves(
		std::ostream &ost_quartics_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_TDO = FALSE;

	if (f_v) {
		cout << "surface_object_with_action::export_all_quartic_curves" << endl;
	}
	int pt_orbit;

	ost_quartics_csv << "orbit,curve,pts_on_curve,bitangents,go" << endl;
	for (pt_orbit = 0; pt_orbit < Orbits_on_points_not_on_lines->nb_orbits; pt_orbit++) {

		cout << "Quartic curve associated with surface and with orbit " << pt_orbit
				<< " / " << Orbits_on_points_not_on_lines->nb_orbits << "}" << endl;


		quartic_curves::quartic_curve_from_surface *QC;

		QC = NEW_OBJECT(quartic_curves::quartic_curve_from_surface);

		QC->init(this, verbose_level);


		if (f_v) {
			cout << "surface_object_with_action::export_all_quartic_curves before QC->quartic" << endl;
		}
		QC->quartic(pt_orbit, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::export_all_quartic_curves after QC->quartic" << endl;
		}


#if 0
		// the quartic curve is now in QC->curve
		// as a Surf->Poly4_x123

		if (f_v) {
			cout << "surface_object_with_action::export_all_quartic_curves before QC->compute_stabilizer" << endl;
		}
		QC->compute_stabilizer(verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::export_all_quartic_curves after QC->compute_stabilizer" << endl;
		}

#endif

		ost_quartics_csv << pt_orbit;

		int i;
		{
			ostringstream s;


			for (i = 0; i < Surf->Poly4_x123->get_nb_monomials(); i++) {
				s << QC->curve[i];
				if (i < Surf->Poly4_x123->get_nb_monomials() - 1) {
					s << ",";
				}
			}
			ost_quartics_csv << ",\"" << s.str() << "\"";
		}
		{
			ostringstream s;


			for (i = 0; i < QC->sz_curve; i++) {
				s << QC->Pts_on_curve[i];
				if (i < QC->sz_curve - 1) {
					s << ",";
				}
			}
			ost_quartics_csv << ",\"" << s.str() << "\"";
		}
		{
			ostringstream s;

			for (i = 0; i < QC->nb_bitangents; i++) {
				s << QC->Bitangents[i];
				if (i < QC->nb_bitangents - 1) {
					s << ",";
				}
			}
			ost_quartics_csv << ",\"" << s.str() << "\"";
		}
		{
			//longinteger_object go;

			//QC->Stab_gens_quartic->group_order(go);
			ost_quartics_csv << "," << -1;
		}

		ost_quartics_csv << endl;

		FREE_OBJECT(QC);
	}
	ost_quartics_csv << "END" << endl;
	if (f_v) {
		cout << "surface_object_with_action::export_all_quartic_curves done" << endl;
	}
}

void surface_object_with_action::print_full_del_Pezzo(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, P_idx, P_idx_local;
	long int P;

	if (f_v) {
		cout << "surface_object_with_action::print_full_del_Pezzo" << endl;
	}


	//schreier *Orbits_on_points_not_on_lines;

	ost << "Full del Pezzo surfaces:\\\\" << endl;
	ost << "testing all " << Orbits_on_points_not_on_lines->nb_orbits << " orbits:\\\\" << endl;

	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|}" << endl;
	for (i = 0; i < Orbits_on_points_not_on_lines->nb_orbits; i++) {
		f = Orbits_on_points_not_on_lines->orbit_first[i];
		P_idx_local = Orbits_on_points_not_on_lines->orbit[f];
		P = SO->SOP->Pts_not_on_lines[P_idx_local];
		if (!SO->find_point(P, P_idx)) {
			cout << "surface_object_with_action::print_full_del_Pezzo could not find point" << endl;
			exit(1);
		}
		ost << i << " & " << P_idx << " & "  << P << " & ";

		int *f_deleted;
		int j, f_first;

		SO->SOP->compute_reduced_set_of_points_not_on_lines_wrt_P(P_idx, f_deleted, verbose_level);
		// P_idx = index into SO->Pts[]

		ost << "\\{";
		f_first = TRUE;
		for (j = 0; j < SO->SOP->nb_pts_not_on_lines; j++) {
			if (!f_deleted[j]) {
				if (f_first) {
					f_first = FALSE;
				}
				else {
					ost << ",";
				}
				ost << j;
			}
		}
		ost << "\\}";
		ost << " & ";
		if (SO->SOP->test_full_del_pezzo(P_idx, f_deleted, verbose_level)) {
			ost << " \\mbox{is full}\\\\" << endl;
		}
		else {
			ost << " \\mbox{is not full}\\\\" << endl;
		}


		//, SO->SOP->nb_pts_not_on_lines,

		FREE_int(f_deleted);
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	if (f_v) {
		cout << "surface_object_with_action::print_full_del_Pezzo done" << endl;
	}
}

void surface_object_with_action::print_everything(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::print_everything" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_equation" << endl;
	}
	SO->SOP->print_equation(ost);
	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"after print_equation" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_summary" << endl;
	}
	print_summary(ost);


	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_lines" << endl;
	}
	SO->SOP->print_lines(ost);


	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_points" << endl;
	}
	SO->SOP->print_points(ost);


	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_lines_with_points_on_them" << endl;
	}
	SO->SOP->print_lines_with_points_on_them(ost);



	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before SO->print_line_intersection_graph" << endl;
	}
	SO->SOP->print_line_intersection_graph(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_adjacency_matrix_with_intersection_points" << endl;
	}
	SO->SOP->print_adjacency_matrix_with_intersection_points(ost);


	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_neighbor_sets" << endl;
	}
	SO->SOP->print_neighbor_sets(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_tritangent_planes" << endl;
	}
	SO->SOP->print_tritangent_planes(ost);


	//SO->print_planes_in_trihedral_pairs(ost);

#if 0
	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_generalized_quadrangle" << endl;
	}
	SO->SOP->print_generalized_quadrangle(ost);
#endif

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_double sixes" << endl;
	}
	SO->SOP->print_double_sixes(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_half_double_sixes" << endl;
	}
	SO->SOP->print_half_double_sixes(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_half_double_sixes_numerically" << endl;
	}
	SO->SOP->print_half_double_sixes_numerically(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_trihedral_pairs" << endl;
	}

	SO->SOP->print_trihedral_pairs(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything "
				"before print_trihedral_pairs_numerically" << endl;
	}

	SO->SOP->print_trihedral_pairs_numerically(ost);

	if (f_v) {
		cout << "surface_object_with_action::print_everything done" << endl;
	}
}


void surface_object_with_action::print_summary(std::ostream &ost)
{
	ost << "\\subsection*{Summary}" << endl;


	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Object} & \\mbox{Number}  & \\mbox{Orbit type} \\\\";
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Lines} & " << SO->nb_lines << " & ";
	{
		string str;
		Orbits_on_lines->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Points on surface} & " << SO->nb_pts << " & ";
	{
		string str;
		Orbits_on_points->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Singular points} & " << SO->SOP->nb_singular_pts << " & \\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Eckardt points} & " << SO->SOP->nb_Eckardt_points << " & ";
	{
		string str;
		Orbits_on_Eckardt_points->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Double points} & " << SO->SOP->nb_Double_points << " & ";
	{
		string str;
		Orbits_on_Double_points->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Single points} & " << SO->SOP->nb_Single_points << " & ";
	{
		string str;
		Orbits_on_Single_points->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Points off lines} & " << SO->SOP->nb_pts_not_on_lines << " & ";
	{
		string str;
		Orbits_on_points_not_on_lines->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;

	ost << "\\hline" << endl;
	ost << "\\mbox{Hesse planes} & " << SO->SOP->nb_Hesse_planes << " & ";
	{
		string str;
		Orbits_on_Hesse_planes->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Axes} & " << SO->SOP->nb_axes << " & \\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Single sixes} & " << 72 << " & ";
	{
		string str;
		Orbits_on_single_sixes->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Tritangent planes} & " << 45 << " & ";
	{
		string str;
		Orbits_on_tritangent_planes->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Trihedral pairs} & " << 120 << " & ";
	{
		string str;
		Orbits_on_trihedral_pairs->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;




	ost << "\\mbox{Type of points on lines} & ";
	SO->SOP->Type_pts_on_lines->print_naked_tex(ost, TRUE);
	ost << " & \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of lines on points} & ";
	SO->SOP->Type_lines_on_point->print_naked_tex(ost, TRUE);
	ost << " & \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
#if 0
	ost << "Points on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
	ost << "Lines on points:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
#endif
}




}}}}

