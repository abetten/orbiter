/*
 * variety_object_with_action.cpp
 *
 *  Created on: Dec 11, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



variety_object_with_action::variety_object_with_action()
{
	cnt = 0;
	po_go = 0;
	po_index = 0;
	po = 0;
	so = 0;

	Variety_object = NULL;

}

variety_object_with_action::~variety_object_with_action()
{
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
}


void variety_object_with_action::init(
		int cnt, int po_go, int po_index, int po, int so,
		geometry::projective_space *Projective_space,
		ring_theory::homogeneous_polynomial_domain *Poly_ring,
		std::string &eqn_txt,
		int f_second_equation, std::string &eqn2_txt,
		std::string &pts_txt, std::string &bitangents_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::init" << endl;
	}

	data_structures::string_tools ST;

	if (false) {
		cout << "pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
	}

	variety_object_with_action::cnt = cnt;
	variety_object_with_action::po_go = po_go;
	variety_object_with_action::po_index = po_index;
	variety_object_with_action::po = po;
	variety_object_with_action::so = so;

	Variety_object = NEW_OBJECT(algebraic_geometry::variety_object);


	if (f_v) {
		cout << "variety_object_with_action::init "
				"before Variety_object->init_from_string" << endl;
	}
	Variety_object->init_from_string(
			Projective_space,
			Poly_ring,
			eqn_txt,
			f_second_equation, eqn2_txt,
			pts_txt, bitangents_txt,
			verbose_level);
	if (f_v) {
		cout << "variety_object_with_action::init "
				"after Variety_object->init_from_string" << endl;
	}

	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::init done" << endl;
	}
}

void variety_object_with_action::init_image_of(
		variety_object_with_action *old_one,
		int *Elt,
		actions::action *A,
		actions::action *A_on_lines,
		int *eqn2,
		int verbose_level)
// we are not mapping the equation
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::init_image_of" << endl;
	}
	data_structures::sorting Sorting;



	cnt = old_one->cnt;
	po = old_one->po;
	so = old_one->so;

	Variety_object = NEW_OBJECT(algebraic_geometry::variety_object);

#if 0
	Variety_object->allocate_points(
			old_one->Variety_object->nb_pts,
			verbose_level);

	Int_vec_copy(eqn2, Quartic_curve_object->eqn15, 15);

	int i;

	for (i = 0; i < old_one->Quartic_curve_object->nb_pts; i++) {
		Quartic_curve_object->Pts[i] =
				A->Group_element->element_image_of(
				old_one->Quartic_curve_object->Pts[i],
				Elt, 0 /* verbose_level */);
	}

	// after mapping, the points are not in increasing order
	// Therefore, we sort the points:

	Sorting.lint_vec_heapsort(
			Quartic_curve_object->Pts,
			old_one->Quartic_curve_object->nb_pts);


	for (i = 0; i < 28; i++) {
		Quartic_curve_object->bitangents28[i] =
				A_on_lines->Group_element->element_image_of(
				old_one->Quartic_curve_object->bitangents28[i],
				Elt, 0 /* verbose_level */);
	}

	// We don't sort the lines because the lines are often in the Schlaefli ordering
#endif

	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::init_image_of done" << endl;
	}
}


void variety_object_with_action::print(
		std::ostream &ost)
{
	ost << "cnt=" << cnt;
	ost << " po=" << po;
	ost << " so=" << so;

	Variety_object->print(ost);
}

std::string variety_object_with_action::stringify_Pts()
{
	std::string s;


	s = Lint_vec_stringify(
			Variety_object->Point_sets->Sets[0],
			Variety_object->Point_sets->Set_size[0]);

	return s;

}

std::string variety_object_with_action::stringify_bitangents()
{
	std::string s;

	s = Lint_vec_stringify(
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0]);
	return s;

}


}}}




