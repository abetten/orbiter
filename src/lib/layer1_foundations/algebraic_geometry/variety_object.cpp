/*
 * variety_object.cpp
 *
 *  Created on: Nov 17, 2023
 *      Author: betten
 */





#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {



variety_object::variety_object()
{
	Projective_space = NULL;

	Ring = NULL;

	eqn = NULL;

	Point_sets = NULL;

	Line_sets = NULL;

}

variety_object::~variety_object()
{

}

void variety_object::init_equation(
		geometry::projective_space *Projective_space,
		ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_equation" << endl;
	}

	variety_object::Projective_space = Projective_space;
	variety_object::Ring = Ring;


	eqn = NEW_int(Ring->get_nb_monomials());
	Int_vec_copy(
			equation, eqn, Ring->get_nb_monomials());

	if (f_v) {
		cout << "variety_object::init_equation "
				"before enumerate_points" << endl;
	}
	enumerate_points(verbose_level - 1);
	if (f_v) {
		cout << "variety_object::init_equation "
				"after enumerate_points" << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation done" << endl;
	}
}

void variety_object::init_set_of_sets(
		geometry::projective_space *Projective_space,
		ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		data_structures::set_of_sets *Point_sets,
		data_structures::set_of_sets *Line_sets,
		int verbose_level)
// takes a copy of Point_sets and Line_sets.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_set_of_sets" << endl;
	}

	init_equation(
			Projective_space, Ring, equation, verbose_level - 1);

	variety_object::Point_sets = Point_sets->copy();
	variety_object::Line_sets = Line_sets->copy();
	if (f_v) {
		cout << "variety_object::init_set_of_sets done" << endl;
	}
}


void variety_object::enumerate_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::enumerate_points" << endl;
	}

	long int *Pts;
	int nb_pts;

	if (f_v) {
		cout << "variety_object::enumerate_points before "
				"Ring->enumerate_points" << endl;
	}
	Ring->enumerate_points_lint(
			eqn, Pts, nb_pts,
			0/*verbose_level - 1*/);

	if (f_v) {
		cout << "variety_object::enumerate_points after "
				"Ring->enumerate_points" << endl;
	}
	if (f_v) {
		cout << "variety_object::enumerate_points The variety "
				"has " << nb_pts << " points" << endl;
	}

	Point_sets = NEW_OBJECT(data_structures::set_of_sets);

	Point_sets->init_single(
			Projective_space->Subspaces->N_points /* underlying_set_size */,
			Pts, nb_pts, 0 /* verbose_level */);

	FREE_lint(Pts);


	if (f_v) {
		cout << "quartic_curve_object::enumerate_points done" << endl;
	}
}



}}}

