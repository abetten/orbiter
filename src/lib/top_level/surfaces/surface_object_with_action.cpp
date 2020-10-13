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
namespace top_level {


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
	A_on_the_lines = NULL;
	A_single_sixes = NULL;
	A_on_tritangent_planes = NULL;
	A_on_trihedral_pairs = NULL;
	A_on_pts_not_on_lines = NULL;


	Orbits_on_points = NULL;
	Orbits_on_Eckardt_points = NULL;
	Orbits_on_Double_points = NULL;
	Orbits_on_lines = NULL;
	Orbits_on_single_sixes = NULL;
	Orbits_on_tritangent_planes = NULL;
	Orbits_on_trihedral_pairs = NULL;
	Orbits_on_points_not_on_lines = NULL;
	null();
}

surface_object_with_action::~surface_object_with_action()
{
	freeself();
}

void surface_object_with_action::null()
{
}

void surface_object_with_action::freeself()
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
	if (A_on_the_lines) {
		FREE_OBJECT(A_on_the_lines);
	}
	if (A_single_sixes) {
		FREE_OBJECT(A_single_sixes);
	}
	if (A_on_tritangent_planes) {
		FREE_OBJECT(A_on_tritangent_planes);
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
	if (Orbits_on_lines) {
		FREE_OBJECT(Orbits_on_lines);
	}
	if (Orbits_on_single_sixes) {
		FREE_OBJECT(Orbits_on_single_sixes);
	}
	if (Orbits_on_tritangent_planes) {
		FREE_OBJECT(Orbits_on_tritangent_planes);
	}
	if (Orbits_on_trihedral_pairs) {
		FREE_OBJECT(Orbits_on_trihedral_pairs);
	}
	if (Orbits_on_points_not_on_lines) {
		FREE_OBJECT(Orbits_on_points_not_on_lines);
	}
	null();
}

int surface_object_with_action::init_equation(
	surface_with_action *Surf_A, int *eqn,
	strong_generators *Aut_gens, int verbose_level)
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

	SO = NEW_OBJECT(surface_object);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(Surf_A->Surf, eqn, verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"after SO->init_equation" << endl;
	}

	if (SO->nb_lines != 27) {
		cout << "surface_object_with_action::init_equation "
				"the surface does not have 27 lines" << endl;
		return FALSE;
	}

	
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
	return TRUE;
}


void surface_object_with_action::init_with_surface_object(surface_with_action *Surf_A,
		surface_object *SO,
		strong_generators *Aut_gens,
		int f_has_nice_gens, vector_ge *nice_gens,
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
			Surf_A->A2, SO->Lines, 27, verbose_level);
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


void surface_object_with_action::init_with_27_lines(surface_with_action *Surf_A,
	long int *Lines27, int *eqn,
	strong_generators *Aut_gens,
	int f_find_double_six_and_rearrange_lines,
	int f_has_nice_gens, vector_ge *nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_with_27_lines" << endl;
	}

	surface_object *SO;
	SO = NEW_OBJECT(surface_object);
	if (f_v) {
		cout << "surface_object_with_action::init_with_27_lines "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(Surf_A->Surf, Lines27, eqn,
			f_find_double_six_and_rearrange_lines, verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_with_27_lines "
				"after SO->init_with_27_lines" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_with_27_lines "
				"before SO->init_with_surface_object" << endl;
	}

	init_with_surface_object(Surf_A,
			SO,
			Aut_gens,
			f_has_nice_gens, nice_gens,
			verbose_level);

	if (f_v) {
		cout << "surface_object_with_action::init_with_27_lines "
				"after SO->init_with_surface_object" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_action::init_with_27_lines done" << endl;
	}
}


void surface_object_with_action::init_surface_object(
	surface_with_action *Surf_A, surface_object *SO,
	strong_generators *Aut_gens, int verbose_level)
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
	if (Surf_A->A->is_semilinear_matrix_group()) {
		if (f_v) {
			cout << "surface_object_with_action::compute_projectivity_group "
					"computing projectivity subgroup" << endl;
		}

		projectivity_group_gens = NEW_OBJECT(strong_generators);
		{
			sims *S;

			if (f_v) {
				cout << "surface_object_with_action::compute_projectivity_group "
						"before Aut_gens->create_sims" << endl;
			}
			S = Aut_gens->create_sims(0 /*verbose_level */);
			if (f_v) {
				cout << "surface_object_with_action::compute_projectivity_group "
						"after Aut_gens->create_sims" << endl;
			}
			if (f_v) {
				cout << "surface_object_with_action::compute_projectivity_group "
						"before projectivity_group_gens->projectivity_subgroup" << endl;
			}
			projectivity_group_gens->projectivity_subgroup(S, verbose_level - 3);
			if (f_v) {
				cout << "surface_object_with_action::compute_projectivity_group "
						"after projectivity_group_gens->projectivity_subgroup" << endl;
			}
			FREE_OBJECT(S);
		}
		if (f_v) {
			cout << "surface_object_with_action::compute_projectivity_group "
					"computing projectivity subgroup done" << endl;
		}
	}
	else {
		projectivity_group_gens = NULL;
	}


	if (f_v) {
		cout << "surface_object_with_action::compute_projectivity_group "
				"computing Sylow structure" << endl;
	}
	// compute the Sylow structure:
	sims *S = NULL;

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
		Syl = NEW_OBJECT(sylow_structure);
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


	// orbits on lines:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on lines" << endl;
	}
	init_orbits_on_lines(verbose_level);


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


	// orbits on trihedral pairs:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_automorphism_group "
				"orbits on trihedral pairs" << endl;
	}
	init_orbits_on_trihedral_pairs(verbose_level);



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
			SO->Lines, 27, 0 /*verbose_level*/);
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
			72, 6, Surf->Double_six, 0 /*verbose_level*/);
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
			Surf->Lines_in_tritangent_planes,
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
					//SO->Trihedral_pairs_as_tritangent_planes,
					Surf->Trihedral_to_Eckardt,
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
		strong_generators *Aut_gens,
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
			callback_surface_domain_sstr_line_label,
			Surf);

}

void surface_object_with_action::print_elements_on_lines(
		ostream &ost,
		strong_generators *Aut_gens,
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
			callback_surface_domain_sstr_line_label,
			Surf);

}

void surface_object_with_action::print_automorphism_group(
	ostream &ost,
	int f_print_orbits, const char *fname_mask)
{
	longinteger_object go;
	latex_interface L;

	Aut_gens->group_order(go);
	
	ost << "\\clearpage" << endl;
	ost << "\\section*{Orbits of the automorphism group}" << endl;
	ost << "The automorphism group has order " << go << endl;
	ost << "\\bigskip" << endl;
	ost << "\\subsection*{Orbits on points}" << endl;
	//Orbits_on_points->print_and_list_orbits_and_
	//stabilizer_sorted_by_length(ost, TRUE, Surf_A->A, go);
	Orbits_on_points->print_and_list_orbits_with_original_labels_tex(ost);



	ost << "\\subsection*{Orbits on Eckardt points}" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_with_original_labels_tex(ost);


	ost << "\\subsection*{Orbits on Double points}" << endl;
	Orbits_on_Double_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on points not on lines}" << endl;
	//Orbits_on_points_not_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);
	Orbits_on_points_not_on_lines->print_and_list_orbits_with_original_labels_tex(ost);


	ost << "\\subsection*{Orbits on lines}" << endl;
	Orbits_on_lines->print_and_list_orbits_tex(ost);

	ost << "\\bigskip" << endl;

	Surf->latex_table_of_Schlaefli_labeling_of_lines(ost);

	ost << "\\bigskip" << endl;

	Orbits_on_lines->print_and_list_orbits_with_original_labels_tex(ost);

	int *Decomp_scheme;
	int nb;
	int block_width = 10;
	nb = Orbits_on_lines->nb_orbits;
	Orbits_on_lines->get_orbit_decomposition_scheme_of_graph(
			SO->SOP->Adj_line_intersection_graph, 27, Decomp_scheme,
			0 /*verbose_level*/);
	ost << "\\subsection*{Decomposition scheme of line intersection graph}" << endl;
	ost << "Decomposition scheme of line intersection graph:" << endl;
	L.print_integer_matrix_tex_block_by_block(ost,
			Decomp_scheme, nb, nb, block_width);
	FREE_int(Decomp_scheme);
	

	ost << "\\subsection*{Orbits on single sixes}" << endl;
	Orbits_on_single_sixes->print_and_list_orbits_tex(ost);

	if (f_print_orbits) {

		int xmax = 1000000;
		int ymax = 1000000;
		int f_circletext = TRUE;
		int rad = 22000;
		int f_embedded = FALSE;
		int f_sideways = FALSE;
		double scale = 0.33;
		double line_width = 0.5;
		int f_has_point_labels = FALSE;
		long int *point_labels = NULL;
	
		Orbits_on_single_sixes->draw_forest(fname_mask, 
			xmax, ymax, 
			f_circletext, rad, 
			f_embedded, f_sideways, 
			scale, line_width, 
			f_has_point_labels, point_labels, 
			0 /*verbose_level*/);


		int i;
		for (i = 0; i < Orbits_on_single_sixes->nb_orbits; i++) {
			char fname[1000];

			sprintf(fname, fname_mask, i);
			ost << "" << endl; 
			ost << "\\bigskip" << endl; 
			ost << "" << endl; 
			ost << "Orbit " << i << " consisting of the following "
					<< Orbits_on_single_sixes->orbit_len[i]
					<< " half double sixes:" << endl;
			ost << "$$" << endl;
			L.int_set_print_tex(ost,
				Orbits_on_single_sixes->orbit + 
					Orbits_on_single_sixes->orbit_first[i], 
				Orbits_on_single_sixes->orbit_len[i]);
			ost << "$$" << endl;
			ost << "" << endl; 
			ost << "\\begin{center}" << endl;
			ost << "\\input " << fname << endl; 
			ost << "\\end{center}" << endl;
			ost << "" << endl; 
			}


		}

	
	ost << "\\subsection*{Orbits on tritangent planes}" << endl;
	Orbits_on_tritangent_planes->print_and_list_orbits_tex(ost);

	ost << "\\subsection*{Orbits on trihedral pairs}" << endl;
	Orbits_on_trihedral_pairs->print_and_list_orbits_tex(ost);

}

void surface_object_with_action::cheat_sheet_basic(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;
	latex_interface L;

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic" << endl;
	}


	longinteger_object ago;
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
		Surf->latex_half_double_six(ost, idx);
		ost << "$$" << endl;

	}

	ost << "\\bigskip" << endl;

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic done" << endl;
	}
}

void surface_object_with_action::cheat_sheet(ostream &ost, 
	const char *label_txt, const char *label_tex, 
	int f_print_orbits, const char *fname_mask, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet" << endl;
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


	longinteger_object ago;
	Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before Aut_gens->"
				"print_generators_tex" << endl;
	}
	Aut_gens->print_generators_tex(ost);


	if (f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		nice_gens->print_tex(ost);

	}


	if (projectivity_group_gens) {
		longinteger_object go;
		projectivity_group_gens->group_order(go);
		ost << "The projectivity group has order "
				<< go << "\\\\" << endl;
		ost << "The projectivity group is generated by:\\\\" << endl;
		if (f_v) {
			cout << "surface_object_with_action::cheat_sheet "
					"before projectivity_group_gens->"
					"print_generators_tex" << endl;
		}
		projectivity_group_gens->print_generators_tex(ost);

		ost << "The projectivity group in the action on the lines:\\\\" << endl;
		print_generators_on_lines(
				ost,
				projectivity_group_gens,
				verbose_level);

#if 0
		ost << "The elements of the projectivity group "
				"in the action on the lines:\\\\" << endl;
		print_elements_on_lines(
				ost,
				projectivity_group_gens,
				verbose_level);
#endif

		char label_group[1000];

		sprintf(label_group, "label_txt_proj_grp");
		projectivity_group_gens->export_group_and_copy_to_latex(label_group,
				ost,
				verbose_level - 2);


	}

	if (Syl) {
		int idx;

		for (idx = 0; idx < Syl->nb_primes; idx++) {
			ost << "The " << Syl->primes[idx]
				<< "-Sylow subgroup is generated by:\\\\" << endl;
			Syl->Sub[idx].SG->print_generators_tex(ost);
			char label_group[1000];

			sprintf(label_group, "label_txt_proj_grp_syl_%d", Syl->primes[idx]);
			Syl->Sub[idx].SG->export_group_and_copy_to_latex(label_group,
					ost,
					verbose_level - 2);

		}
	}


	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_everything" << endl;
	}

	SO->SOP->print_everything(ost, verbose_level - 1);

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"after SO->print_everything" << endl;
	}




	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before print_automorphism_group" << endl;
	}
	print_automorphism_group(ost, f_print_orbits, fname_mask);
	

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
	int block_width = 24;

	go = Aut_gens->group_order_as_lint();
	if (go < 50) {
		latex_interface L;
		int *Table;
		Aut_gens->create_group_table(Table, go, verbose_level - 1);
		L.print_integer_matrix_tex_block_by_block(ost,
				Table, go, go, block_width);
		FREE_int(Table);
	}
	else {
		ost << "Too big to print." << endl;
	}


	if (Aut_gens->A->degree < 500) {

		Aut_gens->export_group_and_copy_to_latex(label_txt,
				ost,
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


void surface_object_with_action::investigate_surface_and_write_report(
		action *A,
		surface_create *SC,
		six_arcs_not_on_a_conic *Six_arcs,
		int f_surface_clebsch,
		int f_surface_codes,
		int f_surface_quartic,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report" << endl;
	}

	char fname[2000];
	char fname_mask[2000];
	char label[2000];
	char label_tex[2000];

	snprintf(fname, 2000, "surface_%s.tex", SC->prefix.c_str());
	snprintf(label, 2000, "surface_%s", SC->label_txt.c_str());
	snprintf(label_tex, 2000, "surface %s", SC->label_tex.c_str());
	snprintf(fname_mask, 2000, "surface_%s_orbit_%%d", SC->prefix.c_str());
	{
		ofstream fp(fname);
		latex_interface L;

		L.head_easy(fp);

		investigate_surface_and_write_report2(
					fp,
					A,
					SC,
					Six_arcs,
					f_surface_clebsch,
					f_surface_codes,
					f_surface_quartic,
					fname_mask,
					label,
					label_tex,
					verbose_level);


		L.foot(fp);
	}
	file_io Fio;

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


}




void surface_object_with_action::investigate_surface_and_write_report2(
		ostream &ost,
		action *A,
		surface_create *SC,
		six_arcs_not_on_a_conic *Six_arcs,
		int f_surface_clebsch,
		int f_surface_codes,
		int f_surface_quartic,
		char fname_mask[2000],
		char label[2000],
		char label_tex[2000],
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report2" << endl;
	}

	ost << "\\section{The Finite Field $\\mathbb F_{" << SC->F->q << "}$}" << endl;
	SC->F->cheat_sheet(ost, verbose_level);

	ost << "\\bigskip" << endl;

	cheat_sheet(ost,
		label,
		label_tex,
		TRUE /* f_print_orbits */,
		fname_mask /* const char *fname_mask*/,
		verbose_level);

	ost << "\\setlength{\\parindent}{0pt}" << endl;

	if (f_surface_clebsch) {

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

		SC->Surf->latex_table_of_clebsch_maps(ost);


		ost << endl;
		ost << "\\clearpage" << endl;
		ost << endl;



		ost << "\\section{Six-arcs not on a conic}" << endl;
		ost << endl;


		//ost << "The six-arcs not on a conic are:\\\\" << endl;
		Six_arcs->report_latex(ost);


		if (f_surface_codes) {

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


#if 0

		int *Arc_iso; // [72]
		int *Clebsch_map; // [nb_pts]
		int *Clebsch_coeff; // [nb_pts * 4]
		//int line_a, line_b;
		//int transversal_line;
		int tritangent_plane_rk;
		int plane_rk_global;
		int ds, ds_row;

		fp << endl;
		fp << "\\clearpage" << endl;
		fp << endl;

		fp << "\\section{Clebsch maps in detail}" << endl;
		fp << endl;




		Arc_iso = NEW_int(72);
		Clebsch_map = NEW_int(SO->nb_pts);
		Clebsch_coeff = NEW_int(SO->nb_pts * 4);

		for (ds = 0; ds < 36; ds++) {
			for (ds_row = 0; ds_row < 2; ds_row++) {
				SC->Surf->prepare_clebsch_map(
						ds, ds_row,
						line_a, line_b,
						transversal_line,
						0 /*verbose_level */);


				ost << endl;
				ost << "\\bigskip" << endl;
				ost << endl;
				ost << "\\subsection{Clebsch map for double six "
						<< ds << ", row " << ds_row << "}" << endl;
				ost << endl;



				cout << "computing clebsch map:" << endl;
				SO->compute_clebsch_map(line_a, line_b,
					transversal_line,
					tritangent_plane_rk,
					Clebsch_map,
					Clebsch_coeff,
					verbose_level);


				plane_rk_global = SO->Tritangent_planes[
					SO->Eckardt_to_Tritangent_plane[
						tritangent_plane_rk]];

				int Arc[6];
				int Arc2[6];
				int Blown_up_lines[6];
				int perm[6];

				SO->clebsch_map_find_arc_and_lines(
						Clebsch_map,
						Arc,
						Blown_up_lines,
						0 /* verbose_level */);

				for (j = 0; j < 6; j++) {
					perm[j] = j;
					}

				int_vec_heapsort_with_log(Blown_up_lines, perm, 6);
				for (j = 0; j < 6; j++) {
					Arc2[j] = Arc[perm[j]];
					}


				ost << endl;
				ost << "\\bigskip" << endl;
				ost << endl;
				//ost << "\\section{Clebsch map}" << endl;
				//ost << endl;
				ost << "Line 1 = $";
				ost << SC->Surf->Line_label_tex[line_a];
				ost << "$\\\\" << endl;
				ost << "Line 2 = $";
				ost << SC->Surf->Line_label_tex[line_b];
				ost << "$\\\\" << endl;
				ost << "Transversal line $";
				ost << SC->Surf->Line_label_tex[transversal_line];
				ost << "$\\\\" << endl;
				ost << "Image plane $\\pi_{" << tritangent_plane_rk
						<< "}=" << plane_rk_global << "=$\\\\" << endl;
				ost << "$$" << endl;

				ost << "\\left[" << endl;
				SC->Surf->Gr3->print_single_generator_matrix_tex(
						ost, plane_rk_global);
				ost << "\\right]," << endl;

				ost << "$$" << endl;
				ost << "Arc $";
				int_set_print_tex(ost, Arc2, 6);
				ost << "$\\\\" << endl;
				ost << "Half double six: $";
				int_set_print_tex(ost, Blown_up_lines, 6);
				ost << "=\\{";
				for (j = 0; j < 6; j++) {
					ost << SC->Surf->Line_label_tex[Blown_up_lines[j]];
					ost << ", ";
					}
				ost << "\\}$\\\\" << endl;

				ost << "The arc consists of the following "
						"points:\\\\" << endl;
				display_table_of_projective_points(ost,
						SC->F, Arc2, 6, 3);

				int orbit_at_level, idx;
				Six_arcs->Gen->gen->identify(Arc2, 6,
						transporter, orbit_at_level,
						0 /*verbose_level */);


				if (!int_vec_search(Six_arcs->Not_on_conic_idx,
					Six_arcs->nb_arcs_not_on_conic,
					orbit_at_level,
					idx)) {
					cout << "could not find orbit" << endl;
					exit(1);
					}

				ost << "The arc is isomorphic to arc " << orbit_at_level
						<< " in the original classification.\\\\" << endl;
				ost << "The arc is isomorphic to arc " << idx
						<< " in the list.\\\\" << endl;
				Arc_iso[2 * ds + ds_row] = idx;



				SO->clebsch_map_latex(ost, Clebsch_map, Clebsch_coeff);

				//SO->clebsch_map_print_fibers(Clebsch_map);
				}
			}



		ost << "The isomorphism type of arc associated with "
				"each half-double six is:" << endl;
		ost << "$$" << endl;
		print_integer_matrix_with_standard_labels(ost,
				Arc_iso, 36, 2, TRUE);
		ost << "$$" << endl;

		FREE_int(Arc_iso);
		FREE_int(Clebsch_map);
		FREE_int(Clebsch_coeff);
#endif


#if 0
		ost << endl;
		ost << "\\clearpage" << endl;
		ost << endl;


		ost << "\\section{Clebsch maps in detail by orbits "
				"on half-double sixes}" << endl;
		ost << endl;



		ost << "There are " << SoA->Orbits_on_single_sixes->nb_orbits
				<< "orbits on half double sixes\\\\" << endl;

		Arc_iso = NEW_int(SoA->Orbits_on_single_sixes->nb_orbits);
		Clebsch_map = NEW_int(SO->nb_pts);
		Clebsch_coeff = NEW_int(SO->nb_pts * 4);

		int j, f, l, k;

		for (j = 0; j < SoA->Orbits_on_single_sixes->nb_orbits; j++) {

			int line1, line2, transversal_line;

			if (f_v) {
				cout << "surface_with_action::arc_lifting_and_classify "
					"orbit on single sixes " << j << " / "
					<< SoA->Orbits_on_single_sixes->nb_orbits << ":" << endl;
			}

			fp << "\\subsection*{Orbit on single sixes " << j << " / "
				<< SoA->Orbits_on_single_sixes->nb_orbits << "}" << endl;

			f = SoA->Orbits_on_single_sixes->orbit_first[j];
			l = SoA->Orbits_on_single_sixes->orbit_len[j];
			if (f_v) {
				cout << "orbit f=" << f <<  " l=" << l << endl;
				}
			k = SoA->Orbits_on_single_sixes->orbit[f];

			if (f_v) {
				cout << "The half double six is no " << k << " : ";
				int_vec_print(cout, SoA->Surf->Half_double_sixes + k * 6, 6);
				cout << endl;
				}

			int h;

			fp << "The half double six is no " << k << "$ = "
					<< Surf->Half_double_six_label_tex[k] << "$ : $";
			int_vec_print(ost, Surf->Half_double_sixes + k * 6, 6);
			ost << " = \\{" << endl;
			for (h = 0; h < 6; h++) {
				ost << Surf->Line_label_tex[
						Surf->Half_double_sixes[k * 6 + h]];
				if (h < 6 - 1) {
					ost << ", ";
					}
				}
			ost << "\\}$\\\\" << endl;

			ds = k / 2;
			ds_row = k % 2;

			SC->Surf->prepare_clebsch_map(
					ds, ds_row,
					line1, line2,
					transversal_line,
					0 /*verbose_level */);

			ost << endl;
			ost << "\\bigskip" << endl;
			ost << endl;
			ost << "\\subsection{Clebsch map for double six "
					<< ds << ", row " << ds_row << "}" << endl;
			ost << endl;



			cout << "computing clebsch map:" << endl;
			SO->compute_clebsch_map(line1, line2,
				transversal_line,
				tritangent_plane_rk,
				Clebsch_map,
				Clebsch_coeff,
				verbose_level);


			plane_rk_global = SO->Tritangent_planes[
				SO->Eckardt_to_Tritangent_plane[
					tritangent_plane_rk]];

			int Arc[6];
			int Arc2[6];
			int Blown_up_lines[6];
			int perm[6];

			SO->clebsch_map_find_arc_and_lines(
					Clebsch_map,
					Arc,
					Blown_up_lines,
					0 /* verbose_level */);

			for (h = 0; h < 6; h++) {
				perm[h] = h;
				}

			Sorting.int_vec_heapsort_with_log(Blown_up_lines, perm, 6);
			for (h = 0; h < 6; h++) {
				Arc2[h] = Arc[perm[h]];
				}


			ost << endl;
			ost << "\\bigskip" << endl;
			ost << endl;
			//ost << "\\section{Clebsch map}" << endl;
			//ost << endl;
			ost << "Line 1 = $";
			ost << SC->Surf->Line_label_tex[line1];
			ost << "$\\\\" << endl;
			ost << "Line 2 = $";
			ost << SC->Surf->Line_label_tex[line2];
			ost << "$\\\\" << endl;
			ost << "Transversal line $";
			ost << SC->Surf->Line_label_tex[transversal_line];
			ost << "$\\\\" << endl;
			ost << "Image plane $\\pi_{" << tritangent_plane_rk
					<< "}=" << plane_rk_global << "=$\\\\" << endl;
			ost << "$$" << endl;

			ost << "\\left[" << endl;
			SC->Surf->Gr3->print_single_generator_matrix_tex(
					ost, plane_rk_global);
			ost << "\\right]," << endl;

			ost << "$$" << endl;
			ost << "Arc $";
			int_set_print_tex(ost, Arc2, 6);
			ost << "$\\\\" << endl;
			ost << "Half double six: $";
			int_set_print_tex(ost, Blown_up_lines, 6);
			ost << "=\\{";
			for (h = 0; h < 6; h++) {
				ost << SC->Surf->Line_label_tex[Blown_up_lines[h]];
				ost << ", ";
				}
			ost << "\\}$\\\\" << endl;

			ost << "The arc consists of the following "
					"points:\\\\" << endl;
			SC->F->display_table_of_projective_points(ost,
					Arc2, 6, 3);

			int orbit_at_level, idx;
			Six_arcs->Gen->gen->identify(Arc2, 6,
					transporter, orbit_at_level,
					0 /*verbose_level */);


			if (!Sorting.int_vec_search(Six_arcs->Not_on_conic_idx,
				Six_arcs->nb_arcs_not_on_conic,
				orbit_at_level,
				idx)) {
				cout << "could not find orbit" << endl;
				exit(1);
				}

			ost << "The arc is isomorphic to arc " << orbit_at_level
					<< " in the original classification.\\\\" << endl;
			ost << "The arc is isomorphic to arc " << idx
					<< " in the list.\\\\" << endl;
			Arc_iso[j] = idx;



			SO->clebsch_map_latex(ost, Clebsch_map, Clebsch_coeff);

		} // next j

		ost << "The isomorphism type of arc associated with "
				"each half-double six is:" << endl;
		ost << "$$" << endl;
		int_vec_print(fp,
				Arc_iso, SoA->Orbits_on_single_sixes->nb_orbits);
		ost << "$$" << endl;



		FREE_int(Arc_iso);
		FREE_int(Clebsch_map);
		FREE_int(Clebsch_coeff);

#endif



		if (f_surface_quartic) {

			surface_object_tangent_cone *SOT;

			SOT = NEW_OBJECT(surface_object_tangent_cone);

			SOT->init(this, verbose_level);
			SOT->quartic(ost, verbose_level);

			FREE_OBJECT(SOT);

			//SoA->quartic(ost, verbose_level);
		}


	}

	if (f_v) {
		cout << "surface_object_with_action::investigate_surface_and_write_report2 done" << endl;
	}
}





}}
