/*
 * quartic_curve_object.cpp
 *
 *  Created on: Oct 9, 2022
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



quartic_curve_object::quartic_curve_object()
{
	cnt = 0;
	po = 0;
	so = 0;

	eqn = NULL;
	sz = 0;
	pts = NULL;
	nb_pts = 0;
	bitangents = NULL;
	nb_bitangents = 0;
}

quartic_curve_object::~quartic_curve_object()
{
}


void quartic_curve_object::init(
		int cnt, int po, int so,
		std::string &eqn_txt,
		std::string &pts_txt, std::string &bitangents_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init" << endl;
	}

	data_structures::string_tools ST;

	if (false) {
		cout << "pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
	}

	quartic_curve_object::cnt = cnt;
	quartic_curve_object::po = po;
	quartic_curve_object::so = so;

	Int_vec_scan(eqn_txt, eqn, sz);
	Lint_vec_scan(pts_txt, pts, nb_pts);
	Lint_vec_scan(bitangents_txt, bitangents, nb_bitangents);

	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "quartic_curve_object::init done" << endl;
	}
}

void quartic_curve_object::init_image_of(quartic_curve_object *old_one,
		int *Elt,
		actions::action *A,
		actions::action *A_on_lines,
		int *eqn2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object::init_image_of" << endl;
	}

	int i;


	cnt = old_one->cnt;
	po = old_one->po;
	so = old_one->so;

	eqn = NEW_int(old_one->sz);
	sz = old_one->sz;
	Int_vec_copy(eqn2, eqn, old_one->sz);

	pts = NEW_lint(old_one->nb_pts);
	bitangents = NEW_lint(old_one->nb_bitangents);

	nb_pts = old_one->nb_pts;
	nb_bitangents = old_one->nb_bitangents;

	for (i = 0; i < old_one->nb_pts; i++) {
		pts[i] = A->Group_element->element_image_of(old_one->pts[i], Elt, 0 /* verbose_level */);
	}
	for (i = 0; i < old_one->nb_bitangents; i++) {
		bitangents[i] = A_on_lines->Group_element->element_image_of(old_one->bitangents[i], Elt, 0 /* verbose_level */);
	}


	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "quartic_curve_object::init_image_of done" << endl;
	}
}


void quartic_curve_object::print(std::ostream &ost)
{
	ost << "cnt=" << cnt;
	ost << " po=" << po;
	ost << " so=" << so;
	ost << " eqn=";
	Int_vec_print(ost, eqn, sz);
	ost << " pts=";
	Lint_vec_print(ost, pts, nb_pts);
	ost << " bitangents=";
	Lint_vec_print(ost, bitangents, nb_bitangents);
	ost << endl;
}

}}}


