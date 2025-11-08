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
namespace geometry {
namespace algebraic_geometry {



quartic_curve_object::quartic_curve_object()
{
	Record_birth();
	Dom = NULL;

	//std::string eqn_txt;

#if 0
	Pts = NULL;
	nb_pts = 0;


	//eqn15[15]

	f_has_bitangents = false;
	//bitangents28[28]
#else

	Variety_object = NULL;

	f_has_bitangents = false;

#endif


	QP = NULL;

}

quartic_curve_object::~quartic_curve_object()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::~quartic_curve_object" << endl;
	}
#if 0
	if (Pts) {
		FREE_lint(Pts);
	}
#else
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
#endif
	if (QP) {
		FREE_OBJECT(QP);
	}



	if (f_v) {
		cout << "quartic_curve_object::~quartic_curve_object done" << endl;
	}
}

#if 0
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

	quartic_curve_object::eqn_txt = eqn_txt;


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
#endif

#if 0
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
#endif

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

	Variety_object = NEW_OBJECT(geometry::algebraic_geometry::variety_object);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"before Variety_object->init_equation_only" << endl;
	}

	Variety_object->init_equation_only(
			Dom->P,
			Dom->Poly4_3,
			eqn15,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"after Variety_object->init_equation_only" << endl;
	}

	f_has_bitangents = false;


	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"before enumerate_points" << endl;
	}
	Variety_object->enumerate_points(verbose_level - 1);
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

	//f_has_bitangents = true;
	//Int_vec_copy(eqn15, quartic_curve_object::eqn15, 15);
	//Lint_vec_copy(bitangents28, quartic_curve_object::bitangents28, 28);


	Variety_object = NEW_OBJECT(geometry::algebraic_geometry::variety_object);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"before Variety_object->init_equation_only" << endl;
	}

	Variety_object->init_equation_only(
			Dom->P,
			Dom->Poly4_3,
			eqn15,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"after Variety_object->init_equation_only" << endl;
	}



	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"before Variety_object->set_lines" << endl;
	}

	Variety_object->set_lines(
			bitangents28, 28,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"after Variety_object->set_lines" << endl;
	}

	f_has_bitangents = true;

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"before Variety_object->enumerate_points" << endl;
	}
	Variety_object->enumerate_points(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"after Variety_object->enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents done" << endl;
	}
}


void quartic_curve_object::init_equation_and_bitangents_and_compute_properties(
		quartic_curve_domain *Dom,
		int *eqn15,
		long int *bitangents28,
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
			Dom, eqn15, bitangents28,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"after init_equation_and_bitangents" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"nb_points = " << get_nb_points() << endl;
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"nb_lines = " << get_nb_lines() << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"before compute_properties" << endl;
	}
	compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"after compute_properties" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties done" << endl;
	}
}

int quartic_curve_object::get_nb_points()
{
	if (!Variety_object) {
		cout << "quartic_curve_object::get_nb_points !Variety_object" << endl;
	}
	return Variety_object->get_nb_points();
}


long int quartic_curve_object::get_point(
		int idx)
{
	if (Variety_object == NULL) {
		cout << "quartic_curve_object::get_point Variety_object == NULL" << endl;
		exit(1);
	}
	return Variety_object->get_point(idx);
}

void quartic_curve_object::set_point(
		int idx, long int rk)
{
	if (Variety_object == NULL) {
		cout << "quartic_curve_object::set_point Variety_object == NULL" << endl;
		exit(1);
	}
	return Variety_object->set_point(idx, rk);
}


long int *quartic_curve_object::get_points()
{
	if (Variety_object == NULL) {
		cout << "quartic_curve_object::get_point Variety_object == NULL" << endl;
		exit(1);
	}
	return Variety_object->get_points();
}



int quartic_curve_object::get_nb_lines()
{
	if (!Variety_object) {
		cout << "quartic_curve_object::get_nb_lines !Variety_object" << endl;
	}
	return Variety_object->get_nb_lines();
}


long int quartic_curve_object::get_line(
		int idx)
{
	if (Variety_object == NULL) {
		cout << "quartic_curve_object::get_line Variety_object == NULL" << endl;
		exit(1);
	}
	return Variety_object->get_line(idx);
}

void quartic_curve_object::set_line(
		int idx, long int rk)
{
	if (Variety_object == NULL) {
		cout << "quartic_curve_object::get_line Variety_object == NULL" << endl;
		exit(1);
	}
	return Variety_object->set_line(idx, rk);
}

long int *quartic_curve_object::get_lines()
{
	if (Variety_object == NULL) {
		cout << "quartic_curve_object::get_line Variety_object == NULL" << endl;
		exit(1);
	}
	return Variety_object->get_lines();
}




void quartic_curve_object::enumerate_points(
		algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points" << endl;
	}

#if 0
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
#else

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points "
				"before Variety_object->enumerate_points" << endl;
	}
	Variety_object->enumerate_points(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object::enumerate_points "
				"after Variety_object->enumerate_points" << endl;
	}

#endif

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points The curve "
				"has " << get_nb_points() << " points" << endl;
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

	if (f_v) {
		cout << "quartic_curve_object::compute_properties "
				"before QP->init" << endl;
	}
	QP->init(this, verbose_level);
	if (f_v) {
		cout << "quartic_curve_object::compute_properties "
				"after QP->init" << endl;
	}

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

	if (f_v) {
		cout << "quartic_curve_object::recompute_properties "
				"before QP->init" << endl;
	}
	QP->init(this, verbose_level);
	if (f_v) {
		cout << "quartic_curve_object::recompute_properties "
				"after QP->init" << endl;
	}


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
	//other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "quartic_curve_object::identify_lines" << endl;
	}
	for (i = 0; i < nb_lines; i++) {
		if (!find_line(lines[i], idx) /*Sorting.lint_vec_search_linear(
				bitangents28, 28, lines[i], idx)*/ ) {
			cout << "quartic_curve_object::identify_lines could "
					"not find lines[" << i << "]=" << lines[i]
					<< " in the list of lines" << endl;
			exit(1);
		}
		line_idx[i] = idx;
	}
	if (f_v) {
		cout << "quartic_curve_object::identify_lines done" << endl;
	}
}

int quartic_curve_object::find_line(
		long int P, int &idx)
{

	return Variety_object->find_line(
			P, idx);
}


int quartic_curve_object::find_point(
		long int P, int &idx)
{

	return Variety_object->find_point(
			P, idx);

#if 0

	other::data_structures::sorting Sorting;

	if (Sorting.lint_vec_search(
			Variety_object->get_points(), Variety_object->get_nb_points(), P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
#endif

}

void quartic_curve_object::print(
		std::ostream &ost)
{
	ost << " eqn=";
	Int_vec_print(ost, Variety_object->eqn, 15);
	ost << " pts=";
	Lint_vec_print(ost, Variety_object->get_points(), Variety_object->get_nb_points());
	ost << " bitangents=";
	Lint_vec_print(ost, Variety_object->get_lines(), Variety_object->get_nb_lines());
	ost << endl;
}

void quartic_curve_object::stringify(
		std::string &s_Eqn, std::string &s_Pts, std::string &s_Bitangents)
{
	s_Eqn = Int_vec_stringify(
			Variety_object->eqn, 15);
	s_Pts = Lint_vec_stringify(
			Variety_object->get_points(), Variety_object->get_nb_points());

	s_Bitangents = Lint_vec_stringify(
			Variety_object->get_lines(), Variety_object->get_nb_lines());

}



}}}}

