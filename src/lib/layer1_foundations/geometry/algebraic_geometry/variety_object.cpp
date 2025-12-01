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
namespace geometry {
namespace algebraic_geometry {



variety_object::variety_object()
{
	Record_birth();
	Descr = NULL;

	Projective_space = NULL;

	Ring = NULL;

	//std::string label_txt;
	//std::string label_tex;


#if 0
	//std::string eqn_txt;

	f_second_equation = false;
	//std::string eqn2;
#endif

	eqn = NULL;
	//eqn2 = NULL;


	Point_sets = NULL;

	Line_sets = NULL;

	f_has_singular_points = false;
	//std::vector<long int> Singular_points;

}

variety_object::~variety_object()
{
	Record_death();

}

void variety_object::init(
		variety_description *Descr,
		int verbose_level)
// Does not perform the transformations.
// Called from variety_object_with_action::create_variety
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init" << endl;
	}

	variety_object::Descr = Descr;


	if (Descr->f_projective_space_pointer) {
		Projective_space = Descr->Projective_space_pointer;
	}
	else {
		cout << "variety_object::init f_has_projective_space_pointer is false" << endl;
		exit(1);
	}

	if (Descr->f_ring) {
		Ring = Get_ring(Descr->ring_label);
	}
	else if (Descr->f_ring_pointer) {
		Ring = Descr->Ring_pointer;
	}
	else {
		cout << "variety_object::init please use option -ring" << endl;
		exit(1);
	}




	if (Descr->f_equation_in_algebraic_form) {
		if (f_v) {
			cout << "variety_object::init "
					"before parse_equation_in_algebraic_form" << endl;
			cout << "variety_object::init_from_string "
					"equation = " << Descr->equation_in_algebraic_form_text << endl;
		}

		string *equation_text;

		equation_text = new string;

		*equation_text = Get_string(Descr->equation_in_algebraic_form_text);

		if (Descr->f_set_parameters) {
			if (f_v) {
				cout << "variety_object::init "
						"before parse_equation_in_algebraic_form" << endl;
			}
			parse_equation_in_algebraic_form_with_parameters(
					*equation_text,
					Descr->set_parameters_label,
					Descr->set_parameters_label_tex,
					Descr->set_parameters_values,
					eqn,
					verbose_level - 2);
			if (f_v) {
				cout << "variety_object::init "
						"after parse_equation_in_algebraic_form" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "variety_object::init "
						"before parse_equation_in_algebraic_form" << endl;
			}
			parse_equation_in_algebraic_form(
					*equation_text,
					eqn,
					verbose_level - 2);
			if (f_v) {
				cout << "variety_object::init "
						"after parse_equation_in_algebraic_form" << endl;
			}

		}
	}
	else if (Descr->f_equation_by_coefficients) {
		if (f_v) {
			cout << "variety_object::init "
					"before parse_equation_by_coefficients" << endl;
			cout << "variety_object::init "
					"equation = " << Descr->equation_in_algebraic_form_text << endl;
		}
		parse_equation_by_coefficients(
				Descr->equation_by_coefficients_text,
				eqn,
				verbose_level - 2);
		if (f_v) {
			cout << "variety_object::init "
					"after parse_equation_by_coefficients" << endl;
		}
	}
	else if (Descr->f_equation_by_rank) {
		if (f_v) {
			cout << "variety_object::init "
					"before parse_equation_by_rank" << endl;
			cout << "variety_object::init "
					"equation_by_rank = " << Descr->equation_by_rank_text << endl;
		}
		parse_equation_by_rank(
				Descr->equation_by_rank_text,
				eqn,
				verbose_level - 2);
		if (f_v) {
			cout << "variety_object::init "
					"after parse_equation_by_coefficients" << endl;
		}
	}

	else {
		cout << "variety_object::init please specify an equation" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "variety_object::init "
				"eqn = ";
		Int_vec_print_fully(cout, eqn, Ring->get_nb_monomials());
		cout << endl;
	}


	if (Descr->f_points) {
		if (f_v) {
			cout << "variety_object::init f_points" << endl;
		}

		long int *Pts;
		int nb_pts;

		Lint_vec_scan(Descr->points_txt, Pts, nb_pts);
		Point_sets = NEW_OBJECT(other::data_structures::set_of_sets);
		Point_sets->init_single(
				Projective_space->Subspaces->N_points /* underlying_set_size */,
				Pts, nb_pts, 0 /* verbose_level*/);

		FREE_lint(Pts);
	}
	else {
		if (f_v) {
			cout << "variety_object::init computing the set of rational points" << endl;
		}
		if (f_v) {
			cout << "variety_object::init before enumerate_points" << endl;
		}
		enumerate_points(verbose_level);
		if (f_v) {
			cout << "variety_object::init after enumerate_points" << endl;
		}
	}

	if (Descr->f_bitangents) {

		if (f_v) {
			cout << "variety_object::init f_bitangents" << endl;
		}

		long int *Bitangents;
		int nb_bitangents;

		Lint_vec_scan(Descr->bitangents_txt, Bitangents, nb_bitangents);


		Line_sets = NEW_OBJECT(other::data_structures::set_of_sets);

		Line_sets->init_single(
				Projective_space->Subspaces->N_points /* underlying_set_size */,
				Bitangents, nb_bitangents, 0 /* verbose_level*/);


		FREE_lint(Bitangents);

	}
	else {
		if (f_v) {
			cout << "variety_object::init !f_bitangents" << endl;
		}

		if (Descr->f_compute_lines) {

			if (f_v) {
				cout << "variety_object::init f_compute_lines" << endl;
			}

			if (f_v) {
				cout << "variety_object::init "
						"before enumerate_lines" << endl;
			}
			enumerate_lines(verbose_level - 2);
			if (f_v) {
				cout << "variety_object::init "
						"after enumerate_lines" << endl;
			}
			if (f_v) {
				cout << "variety_object::init The variety "
						"has " << Line_sets->Set_size[0] << " lines" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "variety_object::init !f_compute_lines" << endl;
			}


		}
		//cout << "variety_object::init please specify the set of bitangents" << endl;
		//exit(1);
	}
	if (f_v) {
		cout << "variety_object::init "
				"nb_pts = " << get_nb_points()
				<< " nb_bitangents=" << get_nb_lines() << endl;
	}

	if (Descr->f_label_txt) {
		label_txt = Descr->label_txt;
	}
	else {
		label_txt = "variety_object";
	}

	if (Descr->f_label_tex) {
		label_tex = Descr->label_tex;
	}
	else {
		label_tex = "variety\\_object";
	}


	int i;
	for (i = 0; i < Descr->transformations.size(); i++) {
		if (Descr->transformation_inverse[i]) {
			cout << "-transform_inverse " << Descr->transformations[i] << endl;
		}
		else {
			cout << "-transform " << Descr->transformations[i] << endl;
		}
	}


	if (f_v) {
		cout << "variety_object::init done" << endl;
	}
}

int variety_object::get_nb_points()
{
	if (Point_sets) {
		return Point_sets->Set_size[0];
	}
	else {
		return -1;
	}
}

int variety_object::get_nb_lines()
{
	if (Line_sets) {
		return Line_sets->Set_size[0];
	}
	else {
		return -1;
	}
}

#if 0
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
				"before parse_equation_in_algebraic_form" << endl;
	}
	parse_equation_in_algebraic_form(
			eqn_txt,
			eqn,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object::init_from_string "
				"after parse_equation_in_algebraic_form" << endl;
	}

	if (f_second_equation) {
		if (f_v) {
			cout << "variety_object::init_from_string "
					"before parse_equation_in_algebraic_form (2)" << endl;
		}
		parse_equation_in_algebraic_form(
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

	if (f_v) {
		cout << "variety_object::init_from_string "
				"nb_pts = " << nb_pts << " nb_bitangents=" << nb_bitangents << endl;
	}

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
#endif

void variety_object::parse_equation_by_coefficients(
		std::string &equation_txt,
		int *&equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::parse_equation_by_coefficients" << endl;
	}
	if (f_v) {
		cout << "variety_object::parse_equation_by_coefficients "
				"equation = " << equation_txt << endl;
	}
	if (f_v) {
		cout << "variety_object::parse_equation_by_coefficients "
				"reading coefficients numerically" << endl;
	}

	int sz;

	Int_vec_scan(equation_txt, equation, sz);

	if (sz != Ring->get_nb_monomials()) {
		cout << "variety_object::parse_equation_by_coefficients "
				"the equation does not have the required number of terms" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "variety_object::parse_equation_by_coefficients done" << endl;
	}
}

void variety_object::parse_equation_by_rank(
		std::string &rank_txt,
		int *&equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::parse_equation_by_rank" << endl;
	}
	if (f_v) {
		cout << "variety_object::parse_equation_by_rank "
				"rank_txt = " << rank_txt << endl;
	}

	long int *equation_rk;

	int sz;

	Lint_vec_scan(rank_txt, equation_rk, sz);

	if (sz != 1) {
		cout << "variety_object::parse_equation_by_rank sz != 1" << endl;
		exit(1);
	}


	equation = NEW_int(Ring->get_nb_monomials());

	Ring->unrank_coeff_vector(
			equation, equation_rk[0]);

	FREE_lint(equation_rk);

	if (f_v) {
		cout << "variety_object::parse_equation_by_rank done" << endl;
	}
}





void variety_object::parse_equation_in_algebraic_form(
		std::string &equation_txt,
		int *&equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form" << endl;
	}
	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form "
				"equation = " << equation_txt << endl;
	}

	//if (std::isalpha(equation_txt[0])) {

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form "
				"reading formula" << endl;
	}

	algebra::ring_theory::ring_theory_global R;
	int *coeffs;

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form "
				"before R.parse_equation_easy" << endl;
	}

	R.parse_equation_easy(
			Ring,
			equation_txt,
			coeffs,
			verbose_level - 1);

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form "
				"after R.parse_equation_easy" << endl;
	}

	equation = NEW_int(Ring->get_nb_monomials());
	Int_vec_copy(
			coeffs, equation, Ring->get_nb_monomials());

	FREE_int(coeffs);


	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form done" << endl;
	}
}

void variety_object::parse_equation_in_algebraic_form_with_parameters(
		std::string &equation_txt,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::string &equation_parameter_values,
		int *&equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form_with_parameters" << endl;
	}
	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form_with_parameters "
				"equation = " << equation_txt << endl;
	}

	//if (std::isalpha(equation_txt[0])) {

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form_with_parameters "
				"reading formula" << endl;
	}

	algebra::ring_theory::ring_theory_global R;
	int *coeffs;

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form_with_parameters "
				"before R.parse_equation_with_parameters" << endl;
	}

	R.parse_equation_with_parameters(
			Ring,
			equation_txt,
			equation_parameters,
			equation_parameters_tex,
			equation_parameter_values,
			coeffs,
			verbose_level - 2);

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form_with_parameters "
				"after R.parse_equation_with_parameters" << endl;
	}

	equation = NEW_int(Ring->get_nb_monomials());
	Int_vec_copy(
			coeffs, equation, Ring->get_nb_monomials());

	FREE_int(coeffs);

	if (f_v) {
		cout << "variety_object::parse_equation_in_algebraic_form_with_parameters done" << endl;
	}
}

void variety_object::init_equation_only(
		geometry::projective_geometry::projective_space *Projective_space,
		algebra::ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_equation_only" << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation_only 1" << endl;
	}
	variety_object::Projective_space = Projective_space;
	if (f_v) {
		cout << "variety_object::init_equation_only 2" << endl;
	}
	variety_object::Ring = Ring;
	if (f_v) {
		cout << "variety_object::init_equation_only 3" << endl;
	}


	eqn = NEW_int(Ring->get_nb_monomials());
	if (f_v) {
		cout << "variety_object::init_equation_only 4" << endl;
	}
	Int_vec_copy(
			equation, eqn, Ring->get_nb_monomials());


	if (f_v) {
		cout << "variety_object::init_equation_only done" << endl;
	}
}

void variety_object::init_equation(
		geometry::projective_geometry::projective_space *Projective_space,
		algebra::ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_equation" << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation "
				"before init_equation_only" << endl;
	}
	init_equation_only(Projective_space, Ring, equation, verbose_level);
	if (f_v) {
		cout << "variety_object::init_equation "
				"after init_equation_only" << endl;
	}

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

void variety_object::init_equation_and_points_and_lines_and_labels(
		geometry::projective_geometry::projective_space *Projective_space,
		algebra::ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		long int *Pts, int nb_pts,
		long int *Bitangents, int nb_bitangents,
		std::string &label_txt,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels" << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"Projective_space = " << Projective_space->label_txt << endl;
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"Ring = " << Ring->get_label_txt() << endl;
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"equation = ";
		Int_vec_print(cout, equation, Ring->get_nb_monomials());
		cout << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"before init_equation_only" << endl;
	}
	init_equation_only(Projective_space, Ring, equation, verbose_level);
	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"after init_equation_only" << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"nb_pts = " << nb_pts << endl;
	}
	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"before Point_sets->init_single" << endl;
	}
	Point_sets = NEW_OBJECT(other::data_structures::set_of_sets);
	Point_sets->init_single(
			Projective_space->Subspaces->N_points /* underlying_set_size */,
			Pts, nb_pts, 0 /* verbose_level*/);
	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"before Point_sets->init_single" << endl;
	}


	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"nb_bitangents = " << nb_bitangents << endl;
	}

	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"before Line_sets->init_single" << endl;
	}
	Line_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Line_sets->init_single(
			Projective_space->Subspaces->N_points /* underlying_set_size */,
			Bitangents, nb_bitangents, 0 /* verbose_level*/);
	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels "
				"after Line_sets->init_single" << endl;
	}

	variety_object::label_txt = label_txt;
	variety_object::label_tex = label_tex;

	if (f_v) {
		cout << "variety_object::init_equation_and_points_and_lines_and_labels done" << endl;
	}

}



void variety_object::init_set_of_sets(
		geometry::projective_geometry::projective_space *Projective_space,
		algebra::ring_theory::homogeneous_polynomial_domain *Ring,
		int *equation,
		other::data_structures::set_of_sets *Point_sets,
		other::data_structures::set_of_sets *Line_sets,
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

#if 0
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

	}
#endif
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

	//}
	if (f_v) {
		cout << "variety_object::enumerate_points The variety "
				"has " << nb_pts << " points" << endl;
	}

	Point_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	if (f_v) {
		cout << "variety_object::enumerate_points "
				"before Point_sets->init_single" << endl;
	}

	Point_sets->init_single(
			Projective_space->Subspaces->N_points /* underlying_set_size */,
			Pts, nb_pts, 0 /* verbose_level */);

	if (f_v) {
		cout << "variety_object::enumerate_points "
				"after Point_sets->init_single" << endl;
	}

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

	geometry::other_geometry::geometry_global Geo;
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



	if (f_v) {
		cout << "surface_object::enumerate_lines before "
				"find_real_lines" << endl;
	}
	find_real_lines(
			The_Lines,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_object::enumerate_lines after "
				"find_real_lines" << endl;
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


	if (f_v) {
		cout << "variety_object::enumerate_lines "
				"before set_lines" << endl;
	}

	set_lines(
			Lines, nb_lines,
			verbose_level);

	if (f_v) {
		cout << "variety_object::enumerate_lines "
				"after set_lines" << endl;
	}

#if 0
	Line_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Line_sets->init_single(
			Projective_space->Subspaces->N_lines /* underlying_set_size */,
			Lines, nb_lines, 0 /* verbose_level */);
#endif

	FREE_lint(Lines);


	if (f_v) {
		cout << "variety_object::enumerate_lines done" << endl;
	}
}

void variety_object::set_lines(
		long int *Lines, int nb_lines,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::set_lines" << endl;
	}

	Line_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Line_sets->init_single(
			Projective_space->Subspaces->N_lines /* underlying_set_size */,
			Lines, nb_lines, 0 /* verbose_level */);



	if (f_v) {
		cout << "variety_object::set_lines done" << endl;
	}

}

long int *variety_object::get_points()
{
	if (Point_sets == NULL) {
		cout << "variety_object::get_points Point_sets == NULL" << endl;
		exit(1);
	}
	return Point_sets->Sets[0];
}

long int variety_object::get_point(
		int idx)
{
	if (Point_sets == NULL) {
		cout << "variety_object::get_point Point_sets == NULL" << endl;
		exit(1);
	}
	return Point_sets->Sets[0][idx];
}

void variety_object::set_point(
		int idx, long int rk)
{
	if (Point_sets == NULL) {
		cout << "variety_object::set_point Point_sets == NULL" << endl;
		exit(1);
	}
	Point_sets->Sets[0][idx] = rk;
}

long int *variety_object::get_lines()
{
	if (Line_sets == NULL) {
		cout << "variety_object::get_lines Line_sets == NULL" << endl;
		exit(1);
	}
	return Line_sets->Sets[0];
}

long int variety_object::get_line(
		int idx)
{
	if (Line_sets == NULL) {
		cout << "variety_object::get_line Line_sets == NULL" << endl;
		exit(1);
	}
	return Line_sets->Sets[0][idx];
}

void variety_object::set_line(
		int idx, long int rk)
{
	if (Line_sets == NULL) {
		cout << "variety_object::set_line Line_sets == NULL" << endl;
		exit(1);
	}
	Line_sets->Sets[0][idx] = rk;
}


int variety_object::find_point(
		long int P, int &idx)
{
	other::data_structures::sorting Sorting;

	if (Point_sets == NULL) {
		cout << "variety_object::find_point Point_sets == NULL" << endl;
		exit(1);
	}

	if (Sorting.lint_vec_search(
			Point_sets->Sets[0], Point_sets->Set_size[0], P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
}

int variety_object::find_line(
		long int P, int &idx)
{
	other::data_structures::sorting Sorting;

	if (Line_sets == NULL) {
		cout << "variety_object::find_point Line_sets == NULL" << endl;
		exit(1);
	}

	if (Sorting.lint_vec_search(
			Line_sets->Sets[0], Line_sets->Set_size[0], P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
}


void variety_object::print(
		std::ostream &ost)
{
	ost << " eqn = ";
	Int_vec_print(ost, eqn, Ring->get_nb_monomials());

	cout << "equation1 is: ";
	Ring->print_equation_simple(
			cout, eqn);
	cout << endl;
#if 0
	if (f_second_equation) {
		cout << "equation2 is: " << eqn2_txt << " = ";
		Ring->print_equation_simple(
				cout, eqn2);
		cout << endl;
	}
#endif

	if (Point_sets) {
		ost << "nb_pts = " << Point_sets->Set_size[0] << " pts=";
		Lint_vec_print_fully(ost, Point_sets->Sets[0], Point_sets->Set_size[0]);
		ost << endl;
	}
	if (Line_sets) {
		ost << "nb_lines = " << Line_sets->Set_size[0] << " Lines=";
		Lint_vec_print(ost, Line_sets->Sets[0], Line_sets->Set_size[0]);
		ost << endl;
	}
}

void variety_object::print_equation_with_line_breaks_tex(
		std::ostream &ost, int *coeffs)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;
	Ring->print_equation_with_line_breaks_tex(
			ost, coeffs, 10 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
	ost << "=0" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}

void variety_object::print_equation(
		std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the variety ";
	ost << " is :" << endl;


	algebra::field_theory::finite_field *F;
	int *coeffs; // [Ring->get_nb_monomials()]
	int nb_monomials;


	F = Projective_space->Subspaces->F;

	nb_monomials = Ring->get_nb_monomials();

	coeffs = NEW_int(nb_monomials);

	Int_vec_copy(eqn, coeffs, nb_monomials);
	F->Projective_space_basic->PG_element_normalize_from_front(
			coeffs, 1, nb_monomials);

#if 0
	ost << "$$" << endl;
	SO->Surf->print_equation_tex(ost, SO->eqn);
	ost << endl << "=0\n$$" << endl;
#else
	print_equation_with_line_breaks_tex(ost, coeffs);
#endif


	Int_vec_print(ost, coeffs, nb_monomials);
	ost << "\\\\" << endl;


	print_equation_verbatim(coeffs, ost);

#if 0
	ost << "\\begin{verbatim}" << endl;

	Ring->print_equation_relaxed(ost, coeffs);
	ost << endl;
	ost << "\\end{verbatim}" << endl;
#endif


	FREE_int(coeffs);

	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}

void variety_object::print_equation_verbatim(
		int *coeffs,
		std::ostream &ost)
{
	ost << "\\begin{verbatim}" << endl;
	Ring->print_equation_relaxed(ost, coeffs);
	ost << endl;
	ost << "\\end{verbatim}" << endl;
}

std::string variety_object::stringify_points()
{
	string s;

	if (Point_sets) {

		other::data_structures::sorting Sorting;
		Sorting.lint_vec_heapsort(Point_sets->Sets[0], Point_sets->Set_size[0]);
		s = Lint_vec_stringify(
			Point_sets->Sets[0],
			Point_sets->Set_size[0]);
	}
	return s;
}

std::string variety_object::stringify_lines()
{
	string s;

	if (Line_sets) {

		other::data_structures::sorting Sorting;
		Sorting.lint_vec_heapsort(Line_sets->Sets[0], Line_sets->Set_size[0]);

		s = Lint_vec_stringify(
				Line_sets->Sets[0],
				Line_sets->Set_size[0]);
	}
	return s;
}

std::string variety_object::stringify_equation()
{
	string eqn_txt;

	eqn_txt = Int_vec_stringify(
			eqn,
			Ring->get_nb_monomials());
	return eqn_txt;
}


void variety_object::stringify(
		std::string &s_Eqn1, //std::string &s_Eqn2,
		std::string &s_nb_Pts,
		std::string &s_Pts,
		std::string &s_Bitangents)
{
	s_Eqn1 = Int_vec_stringify(
			eqn,
			Ring->get_nb_monomials());
#if 0
	if (f_second_equation) {
		s_Eqn2 = Int_vec_stringify(
				eqn2,
				Ring->get_nb_monomials());
	}
#endif
	s_nb_Pts = std::to_string(Point_sets->Set_size[0]);
	s_Pts = Lint_vec_stringify(
			Point_sets->Sets[0],
			Point_sets->Set_size[0]);

	if (Line_sets) {
		s_Bitangents = Lint_vec_stringify(
				Line_sets->Sets[0],
				Line_sets->Set_size[0]);
	}
	else {
		s_Bitangents = "";
	}
}

void variety_object::report_equations(
		std::ostream &ost)
{
	report_equation(ost);
#if 0
	if (f_second_equation) {
		report_equation2(ost);
	}
#endif
}

void variety_object::report_equation(
		std::ostream &ost)
{
#if 0
	ost << "Equation ";
	ost << "\\verb'";
	ost << eqn_txt;
	ost << "'";
#endif
	ost << "\\\\" << endl;
	ost << "Equation ";
	Int_vec_print(ost,
			eqn,
			Ring->get_nb_monomials());
	ost << "\\\\" << endl;

}

#if 0
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
#endif

#if 0
int variety_object::find_point(
		long int P, int &idx)
{
	other::data_structures::sorting Sorting;

	if (Sorting.lint_vec_search(
			Point_sets->Sets[0], Point_sets->Set_size[0], P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
}
#endif

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
	other::data_structures::sorting Sorting;

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

void variety_object::find_real_lines(
		std::vector<long int> &The_Lines,
		int verbose_level)
// adapted from surface_object::find_real_lines
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::find_real_lines" << endl;
	}

	int i, j, d;
	long int rk;
	int *M; // [2 * d]
	int *coeff_out; // [Ring->degree + 1]


	d = Projective_space->Subspaces->n + 1;
	M = NEW_int(2 * d);
	coeff_out = NEW_int(Ring->degree + 1);

	if (f_v) {
		cout << "variety_object::find_real_lines d = " << d << endl;
		cout << "variety_object::find_real_lines Ring->nb_variables = " << Ring->nb_variables << endl;
		cout << "variety_object::find_real_lines Ring->get_nb_monomials() = " << Ring->get_nb_monomials() << endl;
		cout << "variety_object::find_real_lines Ring->degree + 1 = " << Ring->degree + 1 << endl;
	}


	for (i = 0, j = 0; i < The_Lines.size(); i++) {
		rk = The_Lines[i];
		Projective_space->Subspaces->Grass_lines->unrank_lint_here(
				M, rk, 0 /* verbose_level */);
		if (f_v) {
			cout << "variety_object::find_real_lines testing line" << endl;
			Int_matrix_print(M, 2, d);
		}

		Ring->substitute_line(
			eqn /* coeff_in */,
			coeff_out,
			M /* Pt1_coeff */, M + d /* Pt2_coeff */,
			verbose_level - 3);
		// coeff_in[nb_monomials], coeff_out[degree + 1]

		if (f_v) {
			cout << "variety_object::find_real_lines coeff_out=";
			Int_vec_print(cout, coeff_out, Ring->degree + 1);
			cout << endl;
		}
		if (!Int_vec_is_zero(coeff_out, Ring->degree + 1)) {
			if (f_v) {
				cout << "variety_object::find_real_lines not a real line" << endl;
			}
		}
		else {
			The_Lines[j] = rk;
			j++;
		}
	}
	The_Lines.resize(j);

	FREE_int(M);
	FREE_int(coeff_out);

	if (f_v) {
		cout << "variety_object::find_real_lines done" << endl;
	}
}

void variety_object::compute_singular_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object::compute_singular_points" << endl;
	}
	if (f_v) {
		cout << "variety_object::compute_singular_points "
				"before Poly_ring->compute_singular_points_projectively" << endl;
	}
	Ring->compute_singular_points_projectively(
			Projective_space,
			eqn,
			Singular_points,
			verbose_level);
	if (f_v) {
		cout << "variety_object::compute_singular_points "
				"after Poly_ring->compute_singular_points_projectively" << endl;
	}

	f_has_singular_points = true;

	if (f_v) {
		cout << "variety_object::compute_singular_points done" << endl;
	}
}



}}}}

