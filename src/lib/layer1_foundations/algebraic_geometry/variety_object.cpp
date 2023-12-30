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

	f_second_equation = false;
	//std::string eqn2;

	eqn = NULL;
	eqn2 = NULL;


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
		int f_second_equation, std::string &eqn2_txt,
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
	variety_object::f_second_equation = f_second_equation;
	variety_object::eqn2_txt = eqn2_txt;


	if (f_v) {
		cout << "variety_object::init_from_string "
				"before parse_equation" << endl;
	}
	parse_equation(
			eqn_txt,
			eqn,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object::init_from_string "
				"after parse_equation" << endl;
	}

	if (f_second_equation) {
		if (f_v) {
			cout << "variety_object::init_from_string "
					"before parse_equation (2)" << endl;
		}
		parse_equation(
				eqn2_txt,
				eqn2,
				verbose_level - 2);
		if (f_v) {
			cout << "variety_object::init_from_string "
					"after parse_equation (2)" << endl;
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

void variety_object::parse_equation(
		std::string &equation_txt,
		int *&equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::parse_equation" << endl;
	}
	if (f_v) {
		cout << "variety_object::parse_equation equation = " << equation_txt << endl;
	}

	if (std::isalpha(equation_txt[0])) {

		if (f_v) {
			cout << "variety_object::parse_equation "
					"reading formula" << endl;
		}

		ring_theory::ring_theory_global R;
		int *coeffs;

		if (f_v) {
			cout << "variety_object::parse_equation "
					"before R.parse_equation_easy" << endl;
		}

		R.parse_equation_easy(
				Ring,
				equation_txt,
				coeffs,
				verbose_level - 1);

		if (f_v) {
			cout << "variety_object::parse_equation "
					"after R.parse_equation_easy" << endl;
		}

		equation = NEW_int(Ring->get_nb_monomials());
		Int_vec_copy(
				coeffs, equation, Ring->get_nb_monomials());

		FREE_int(coeffs);

	}
	else {

		if (f_v) {
			cout << "variety_object::parse_equation "
					"reading coefficients numerically" << endl;
		}

		int sz;

		Int_vec_scan(equation_txt, equation, sz);

		if (sz != Ring->get_nb_monomials()) {
			cout << "variety_object::parse_equation "
					"the equation does not have the required number of terms" << endl;
			exit(1);
		}

	}

	if (f_v) {
		cout << "variety_object::parse_equation done" << endl;
	}
}

void variety_object::init_equation_only(
		geometry::projective_space *Projective_space,
		ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_equation_only" << endl;
	}

	variety_object::Projective_space = Projective_space;
	variety_object::Ring = Ring;


	eqn = NEW_int(Ring->get_nb_monomials());
	Int_vec_copy(
			equation, eqn, Ring->get_nb_monomials());


	if (f_v) {
		cout << "variety_object::init_equation_only done" << endl;
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

	init_equation_only(Projective_space, Ring, equation, verbose_level);

#if 0
	variety_object::Projective_space = Projective_space;
	variety_object::Ring = Ring;


	eqn = NEW_int(Ring->get_nb_monomials());
	Int_vec_copy(
			equation, eqn, Ring->get_nb_monomials());
#endif

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


	if (f_second_equation) {
		if (f_v) {
			cout << "variety_object::enumerate_points before "
					"Ring->enumerate_points" << endl;
		}
		Ring->enumerate_points_in_intersection_lint(
				eqn, eqn2, Pts, nb_pts,
				0/*verbose_level - 1*/);

		if (f_v) {
			cout << "variety_object::enumerate_points after "
					"Ring->enumerate_points" << endl;
		}
	}
	else {
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

void variety_object::enumerate_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::enumerate_lines" << endl;
	}

	geometry::geometry_global Geo;
	vector<long int> Points;
	vector<long int> The_Lines;
	int i;

	for (i = 0; i < Point_sets->Set_size[0]; i++) {
		Points.push_back(Point_sets->Sets[0][i]);
	}

	if (f_v) {
		cout << "surface_object::enumerate_lines before "
				"Geo.find_lines_which_are_contained" << endl;
	}
	Geo.find_lines_which_are_contained(
			Projective_space,
			Points, The_Lines, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_object::enumerate_lines after "
				"Geo.find_lines_which_are_contained" << endl;
	}

	long int *Lines;
	int nb_lines;

	nb_lines = The_Lines.size();
	Lines = NEW_lint(nb_lines);
	for (i = 0; i < nb_lines; i++) {
		Lines[i] = The_Lines[i];
	}

	if (f_v) {
		cout << "variety_object::enumerate_lines The variety "
				"has " << nb_lines << " lines" << endl;
	}

	Line_sets = NEW_OBJECT(data_structures::set_of_sets);

	Line_sets->init_single(
			Projective_space->Subspaces->N_lines /* underlying_set_size */,
			Lines, nb_lines, 0 /* verbose_level */);

	FREE_lint(Lines);


	if (f_v) {
		cout << "variety_object::enumerate_lines done" << endl;
	}
}

void variety_object::print(
		std::ostream &ost)
{
	ost << " eqn = " << eqn_txt << " = ";
	Int_vec_print(ost, eqn, Ring->get_nb_monomials());

	cout << "equation1 is: " << eqn_txt << " = ";
	Ring->print_equation_simple(
			cout, eqn);
	cout << endl;
	if (f_second_equation) {
		cout << "equation2 is: " << eqn2_txt << " = ";
		Ring->print_equation_simple(
				cout, eqn2);
		cout << endl;
	}

	ost << " pts=";
	Lint_vec_print(ost, Point_sets->Sets[0], Point_sets->Set_size[0]);
	ost << " bitangents=";
	Lint_vec_print(ost, Line_sets->Sets[0], Line_sets->Set_size[0]);
	ost << endl;
}


void variety_object::stringify(
		std::string &s_Eqn1, std::string &s_Eqn2,
		std::string &s_Pts, std::string &s_Bitangents)
{
	s_Eqn1 = Int_vec_stringify(
			eqn,
			Ring->get_nb_monomials());
	if (f_second_equation) {
		s_Eqn2 = Int_vec_stringify(
				eqn2,
				Ring->get_nb_monomials());
	}
	s_Pts = Lint_vec_stringify(
			Point_sets->Sets[0],
			Point_sets->Set_size[0]);

	s_Bitangents = Lint_vec_stringify(
			Line_sets->Sets[0],
			Line_sets->Set_size[0]);

}

void variety_object::report_equations(
		std::ostream &ost)
{
	report_equation(ost);
	if (f_second_equation) {
		report_equation2(ost);
	}
}

void variety_object::report_equation(
		std::ostream &ost)
{
	ost << "Equation ";
	ost << "\\verb'";
	ost << eqn_txt;
	ost << "'";
	ost << "\\\\" << endl;
	ost << "Equation ";
	Int_vec_print(ost,
			eqn,
			Ring->get_nb_monomials());
	ost << "\\\\" << endl;

}

void variety_object::report_equation2(
		std::ostream &ost)
{
	ost << "Equation2 ";
	ost << "\\verb'";
	ost << eqn2_txt;
	ost << "'";
	ost << "\\\\" << endl;
	ost << "Equation2 ";
	Int_vec_print(ost,
			eqn2,
			Ring->get_nb_monomials());
	ost << "\\\\" << endl;

}

int variety_object::find_point(
		long int P, int &idx)
{
	data_structures::sorting Sorting;

	if (Sorting.lint_vec_search(Point_sets->Sets[0], Point_sets->Set_size[0], P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
}


std::string variety_object::stringify_eqn()
{
	string s;

	s = Int_vec_stringify(eqn, Ring->get_nb_monomials());
	return s;
}



std::string variety_object::stringify_Pts()
{
	string s;

	s = Lint_vec_stringify(
			Point_sets->Sets[0],
			Point_sets->Set_size[0]);
	return s;
}

std::string variety_object::stringify_Lines()
{
	string s;

	s = Lint_vec_stringify(
			Line_sets->Sets[0],
			Line_sets->Set_size[0]);
	return s;
}

void variety_object::identify_lines(
		long int *lines, int nb_lines,
	int *line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "variety_object::identify_lines" << endl;
	}
	for (i = 0; i < nb_lines; i++) {
		if (!Sorting.lint_vec_search_linear(
				Line_sets->Sets[0], Line_sets->Set_size[0], lines[i], idx)) {
			cout << "variety_object::identify_lines could "
					"not find lines[" << i << "]=" << lines[i]
					<< " in Lines[]" << endl;
			exit(1);
		}
		line_idx[i] = idx;
	}
	if (f_v) {
		cout << "variety_object::identify_lines done" << endl;
	}
}



}}}

