/*
 * quartic_curve_object_with_action.cpp
 *
 *  Created on: Oct 9, 2022
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {



quartic_curve_object_with_action::quartic_curve_object_with_action()
{
	cnt = 0;
	po_go = 0;
	po_index = 0;
	po = 0;
	so = 0;

	Quartic_curve_object = NULL;

}

quartic_curve_object_with_action::~quartic_curve_object_with_action()
{
	if (Quartic_curve_object) {
		FREE_OBJECT(Quartic_curve_object);
	}
}


void quartic_curve_object_with_action::init(
		int cnt, int po_go, int po_index, int po, int so,
		std::string &eqn_txt,
		std::string &pts_txt, std::string &bitangents_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_action::init" << endl;
	}

	data_structures::string_tools ST;

	if (false) {
		cout << "pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
	}

	quartic_curve_object_with_action::cnt = cnt;
	quartic_curve_object_with_action::po_go = po_go;
	quartic_curve_object_with_action::po_index = po_index;
	quartic_curve_object_with_action::po = po;
	quartic_curve_object_with_action::so = so;

	Quartic_curve_object = NEW_OBJECT(algebraic_geometry::quartic_curve_object);


	Quartic_curve_object->init_from_string(
			eqn_txt,
			pts_txt, bitangents_txt,
			verbose_level);

	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "quartic_curve_object_with_action::init done" << endl;
	}
}

void quartic_curve_object_with_action::init_image_of(
		quartic_curve_object_with_action *old_one,
		int *Elt,
		actions::action *A,
		actions::action *A_on_lines,
		int *eqn2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_action::init_image_of" << endl;
	}
	data_structures::sorting Sorting;



	cnt = old_one->cnt;
	po = old_one->po;
	so = old_one->so;

	Quartic_curve_object = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

	Quartic_curve_object->allocate_points(
			old_one->Quartic_curve_object->nb_pts,
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

	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "quartic_curve_object_with_action::init_image_of done" << endl;
	}
}


void quartic_curve_object_with_action::print(
		std::ostream &ost)
{
	ost << "cnt=" << cnt;
	ost << " po=" << po;
	ost << " so=" << so;

	Quartic_curve_object->print(ost);
}

std::string quartic_curve_object_with_action::stringify_Pts()
{
	std::string s;

	s = Lint_vec_stringify(
			Quartic_curve_object->Pts,
			Quartic_curve_object->nb_pts);
	return s;

}

std::string quartic_curve_object_with_action::stringify_bitangents()
{
	std::string s;

	s = Lint_vec_stringify(
			Quartic_curve_object->bitangents28,
			28);
	return s;

}


}}}}



