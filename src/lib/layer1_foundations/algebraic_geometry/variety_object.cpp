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

	//std::string eqn_txt;

	eqn = NULL;

	Point_sets = NULL;

	Line_sets = NULL;

}

variety_object::~variety_object()
{

}

void variety_object::init_from_string(
		geometry::projective_space *Projective_space,
		ring_theory::homogeneous_polynomial_domain *Ring,
		std::string &eqn_txt,
		std::string &pts_txt, std::string &bitangents_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_from_string" << endl;
	}

	variety_object::Projective_space = Projective_space;
	variety_object::Ring = Ring;

	variety_object::eqn_txt = eqn_txt;


	if (std::isalpha(eqn_txt[0])) {

		if (f_v) {
			cout << "variety_object::init_from_string "
					"reading formula" << endl;
		}

		ring_theory::ring_theory_global R;
		int *coeffs;

		if (f_v) {
			cout << "variety_object::init_from_string "
					"before R.parse_equation_easy" << endl;
		}

		R.parse_equation_easy(
				Ring,
				eqn_txt,
				coeffs,
				verbose_level - 1);

		if (f_v) {
			cout << "variety_object::init_from_string "
					"after R.parse_equation_easy" << endl;
		}

		eqn = NEW_int(Ring->get_nb_monomials());
		Int_vec_copy(
				coeffs, eqn, Ring->get_nb_monomials());

		FREE_int(coeffs);

	}
	else {

		if (f_v) {
			cout << "variety_object::init_from_string "
					"reading coefficients numerically" << endl;
		}

		int sz;

		Int_vec_scan(eqn_txt, eqn, sz);

		if (sz != Ring->get_nb_monomials()) {
			cout << "variety_object::init_from_string "
					"the equation does not have the required number of terms" << endl;
			exit(1);
		}

	}

	long int *Pts;
	int nb_pts;

	Lint_vec_scan(pts_txt, Pts, nb_pts);

	int nb_bitangents;
	long int *Bitangents;

	Lint_vec_scan(bitangents_txt, Bitangents, nb_bitangents);


	Point_sets = NEW_OBJECT(data_structures::set_of_sets);
	Line_sets = NEW_OBJECT(data_structures::set_of_sets);

	Point_sets->init_single(
			Projective_space->Subspaces->N_points /* underlying_set_size */,
			Pts, nb_pts, 0 /* verbose_level*/);

	Line_sets->init_single(
			Projective_space->Subspaces->N_points /* underlying_set_size */,
			Bitangents, nb_bitangents, 0 /* verbose_level*/);


	FREE_lint(Pts);
	FREE_lint(Bitangents);


	if (f_v) {
		cout << "variety_object::init_from_string done" << endl;
	}
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
		cout << "variety_object::enumerate_points done" << endl;
	}
}

void variety_object::print(
		std::ostream &ost)
{
	ost << " eqn = " << eqn_txt << " = ";
	Int_vec_print(ost, eqn, Ring->get_nb_monomials());
	ost << " pts=";
	Lint_vec_print(ost, Point_sets->Sets[0], Point_sets->Set_size[0]);
	ost << " bitangents=";
	Lint_vec_print(ost, Line_sets->Sets[0], Line_sets->Set_size[0]);
	ost << endl;
}


void variety_object::stringify(
		std::string &s_Eqn, std::string &s_Pts, std::string &s_Bitangents)
{
	s_Eqn = Int_vec_stringify(
			eqn,
			Ring->get_nb_monomials());
	s_Pts = Lint_vec_stringify(
			Point_sets->Sets[0],
			Point_sets->Set_size[0]);

	s_Bitangents = Lint_vec_stringify(
			Line_sets->Sets[0],
			Line_sets->Set_size[0]);

}




}}}

