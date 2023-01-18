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
	q = 0;
	F = NULL;
	Dom = NULL;


	Pts = NULL;
	nb_pts = 0;

	//Lines = NULL;
	//nb_lines = 0;

	//eqn15[15]

	f_has_bitangents = FALSE;
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
#if 0
	if (Lines) {
		FREE_lint(Lines);
	}
#endif
	if (QP) {
		FREE_OBJECT(QP);
	}



	if (f_v) {
		cout << "quartic_curve_object::~quartic_curve_object done" << endl;
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
		Int_vec_print(cout, eqn15, 15);
		cout << endl;
	}

	quartic_curve_object::Dom = Dom;
	F = Dom->P->F;
	q = F->q;

	f_has_bitangents = FALSE;
	Int_vec_copy(eqn15, quartic_curve_object::eqn15, 15);



	if (f_v) {
		cout << "quartic_curve_object::init_equation_but_no_bitangents "
				"before enumerate_points" << endl;
	}
	enumerate_points(verbose_level - 1);
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
	F = Dom->P->F;
	q = F->q;

	f_has_bitangents = TRUE;
	Int_vec_copy(eqn15, quartic_curve_object::eqn15, 15);
	Lint_vec_copy(bitangents28, quartic_curve_object::bitangents28, 28);



	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents "
				"before enumerate_points" << endl;
	}
	enumerate_points(0/*verbose_level - 1*/);
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
	init_equation_and_bitangents(Dom, eqn15, bitangents28, verbose_level);
	if (f_v) {
		cout << "quartic_curve_object::init_equation_and_bitangents_and_compute_properties "
				"after init_equation_and_bitangents" << endl;
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



void quartic_curve_object::enumerate_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object::enumerate_points before "
				"Dom->Poly4_3->enumerate_points" << endl;
	}
	Dom->Poly4_3->enumerate_points_lint(eqn15, Pts, nb_pts, 0/*verbose_level - 1*/);

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



void quartic_curve_object::compute_properties(int verbose_level)
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

void quartic_curve_object::recompute_properties(int verbose_level)
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










void quartic_curve_object::identify_lines(long int *lines, int nb_lines,
	int *line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "quartic_curve_object::identify_lines" << endl;
		}
	for (i = 0; i < nb_lines; i++) {
		if (!Sorting.lint_vec_search_linear(bitangents28, 28, lines[i], idx)) {
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



int quartic_curve_object::find_point(long int P, int &idx)
{
	data_structures::sorting Sorting;

	if (Sorting.lint_vec_search(Pts, nb_pts, P,
			idx, 0 /* verbose_level */)) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}




}}}

