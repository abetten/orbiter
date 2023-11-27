/*
 * quartic_curve_object.cpp
 *
 *  Created on: May 20, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {



quartic_curve_object::quartic_curve_object()
{
	Dom = NULL;


	Pts = NULL;
	nb_pts = 0;


	//eqn15[15]

	f_has_bitangents = false;
	//bitangents28[28]

	QP = NULL;

}

quartic_curve_object::~quartic_curve_object()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::~quartic_curve_object" << endl;
	}
	if (Pts) {
		FREE_lint(Pts);
	}
	if (QP) {
		FREE_OBJECT(QP);
	}



	if (f_v) {
		cout << "quartic_curve_object::~quartic_curve_object done" << endl;
	}
}

void quartic_curve_object::init_from_string(
		ring_theory::homogeneous_polynomial_domain *Poly_ring,
		std::string &eqn_txt,
		std::string &pts_txt, std::string &bitangents_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_from_string" << endl;
	}

	quartic_curve_object::Dom = NULL;


	if (std::isalpha(eqn_txt[0])) {

		if (f_v) {
			cout << "quartic_curve_object::init_from_string "
					"reading formula" << endl;
		}

		if (Poly_ring->get_nb_monomials() != 15) {
			cout << "quartic_curve_object::init_from_string "
					"the ring should have 15 monomials" << endl;
			exit(1);
		}
		ring_theory::ring_theory_global R;
		int *coeffs;

		if (f_v) {
			cout << "quartic_curve_object::init_from_string "
					"before R.parse_equation_easy" << endl;
		}

		R.parse_equation_easy(
				Poly_ring,
				eqn_txt,
				coeffs,
				verbose_level - 1);

		if (f_v) {
			cout << "quartic_curve_object::init_from_string "
					"after R.parse_equation_easy" << endl;
		}

		Int_vec_copy(coeffs, eqn15, 15);
		FREE_int(coeffs);

	}
	else {

		if (f_v) {
			cout << "quartic_curve_object::init_from_string "
					"reading coefficients numerically" << endl;
		}

		int sz;
		int *eqn;

		Int_vec_scan(eqn_txt, eqn, sz);

		if (sz != 15) {
			cout << "quartic_curve_object::init_from_string "
					"the equation must have 15 terms" << endl;
			exit(1);
		}

		Int_vec_copy(eqn, eqn15, 15);
		FREE_int(eqn);
	}

	Lint_vec_scan(pts_txt, Pts, nb_pts);

	int nb_bitangents;
	long int *Bitangents;

	Lint_vec_scan(bitangents_txt, Bitangents, nb_bitangents);

	if (nb_bitangents == 28) {

		if (f_v) {
			cout << "quartic_curve_object::init_from_string "
					"with 28 bitangents" << endl;
		}

		Lint_vec_copy(Bitangents, bitangents28, 28);

		f_has_bitangents = true;
	}
	else {

		if (f_v) {
			cout << "quartic_curve_object::init_from_string "
					"no bitangents" << endl;
		}

		f_has_bitangents = false;

	}

	FREE_lint(Bitangents);


	if (f_v) {
		cout << "quartic_curve_object::init_from_string done" << endl;
	}
}

void quartic_curve_object::allocate_points(
		int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::allocate_points" << endl;
	}


	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "quartic_curve_object::allocate_points done" << endl;
	}
}

void quartic_curve_object::init_equation_but_no_bitangents(
		quartic_curve_domain *Dom,
		int *eqn15,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents equation:" << endl;
		Int_vec_print(cout, eqn15, 15);
		cout << endl;
	}


	quartic_curve_object::Dom = Dom;

	f_has_bitangents = false;
	Int_vec_copy(eqn15, quartic_curve_object::eqn15, 15);



	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"before enumerate_points" << endl;
	}
	enumerate_points(Dom->Poly4_3, verbose_level - 1);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"after enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"before compute_properties" << endl;
	}
	compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"after compute_properties" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents done" << endl;
	}
}

void quartic_curve_object::init_equation_and_bitangents(
		quartic_curve_domain *Dom,
		int *eqn15, long int *bitangents28,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents" << endl;
		cout << "eqn15:";
		Int_vec_print(cout, eqn15, 15);
		cout << endl;
		cout << "bitangents28:";
		Lint_vec_print(cout, bitangents28, 28);
		cout << endl;
	}

	quartic_curve_object::Dom = Dom;

	f_has_bitangents = true;
	Int_vec_copy(eqn15, quartic_curve_object::eqn15, 15);
	Lint_vec_copy(bitangents28, quartic_curve_object::bitangents28, 28);



	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"before enumerate_points" << endl;
	}
	enumerate_points(Dom->Poly4_3, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"after enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents done" << endl;
	}
}


void quartic_curve_object::init_equation_and_bitangents_and_compute_properties(
		quartic_curve_domain *Dom,
		int *eqn15, long int *bitangents28,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"before init_equation_and_bitangents" << endl;
	}
	init_equation_and_bitangents(
			Dom, eqn15, bitangents28, verbose_level);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"after init_equation_and_bitangents" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"before compute_properties" << endl;
	}
	compute_properties(verbose_level);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"after compute_properties" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties done" << endl;
	}
}



void quartic_curve_object::enumerate_points(
		ring_theory::homogeneous_polynomial_domain *Poly_ring,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points before "
				"Dom->Poly4_3->enumerate_points" << endl;
	}
	Poly_ring->enumerate_points_lint(
			eqn15, Pts, nb_pts, 0/*verbose_level - 1*/);

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points after "
				"Dom->Poly4_3->enumerate_points" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_object::enumerate_points The curve "
				"has " << nb_pts << " points" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::enumerate_points done" << endl;
	}
}



void quartic_curve_object::compute_properties(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::compute_properties" << endl;
	}

	QP = NEW_OBJECT(quartic_curve_object_properties);

	QP->init(this, verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::compute_properties done" << endl;
	}
}

void quartic_curve_object::recompute_properties(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::recompute_properties" << endl;
	}


	if (QP) {
		FREE_OBJECT(QP);
		QP = NULL;
	}

	QP = NEW_OBJECT(quartic_curve_object_properties);

	QP->init(this, verbose_level);


	if (f_v) {
		cout << "quartic_curve_object::recompute_properties done" << endl;
	}
}










void quartic_curve_object::identify_lines(
		long int *lines, int nb_lines,
	int *line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "quartic_curve_object::identify_lines" << endl;
	}
	for (i = 0; i < nb_lines; i++) {
		if (!Sorting.lint_vec_search_linear(
				bitangents28, 28, lines[i], idx)) {
			cout << "quartic_curve_object::identify_lines could "
					"not find lines[" << i << "]=" << lines[i]
					<< " in bitangents28[]" << endl;
			exit(1);
		}
		line_idx[i] = idx;
	}
	if (f_v) {
		cout << "quartic_curve_object::identify_lines done" << endl;
	}
}



int quartic_curve_object::find_point(
		long int P, int &idx)
{
	data_structures::sorting Sorting;

	if (Sorting.lint_vec_search(
			Pts, nb_pts, P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
}

void quartic_curve_object::print(
		std::ostream &ost)
{
	ost << " eqn=";
	Int_vec_print(ost, eqn15, 15);
	ost << " pts=";
	Lint_vec_print(ost, Pts, nb_pts);
	ost << " bitangents=";
	Lint_vec_print(ost, bitangents28, 28);
	ost << endl;
}

void quartic_curve_object::stringify(
		std::string &s_Eqn, std::string &s_Pts, std::string &s_Bitangents)
{
	s_Eqn = Int_vec_stringify(
			eqn15,
			15);
	s_Pts = Lint_vec_stringify(
			Pts,
			nb_pts);

	s_Bitangents = Lint_vec_stringify(
			bitangents28, 28);

}



}}}

