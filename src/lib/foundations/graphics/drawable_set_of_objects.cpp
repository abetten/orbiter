/*
 * drawable_set_of_objects.cpp
 *
 *  Created on: Apr 10, 2020
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace foundations {



drawable_set_of_objects::drawable_set_of_objects()
{
	group_idx = 0;

	type = 0;
		// 1 = sphere
		// 2 = cylinder
		// 3 = prisms (faces)
		// 4 = planes
		// 5 = lines
		// 6 = cubics
		// 7 = quadrics
		// 8 = quartics
		// 9 = label

	d = 0.;
	d2 = 0.;

	properties = NULL;
}

drawable_set_of_objects::~drawable_set_of_objects()
{
	//freeself();
}

void drawable_set_of_objects::init_spheres(int group_idx,
		double rad, const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_spheres" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 1;
	d = rad;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_spheres done" << endl;
	}
}

void drawable_set_of_objects::init_cylinders(int group_idx,
		double rad, const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_cylinders" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 2;
	d = rad;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_cylinders done" << endl;
	}
}

void drawable_set_of_objects::init_prisms(int group_idx,
		double thickness, const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_prisms" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 3;
	d = thickness;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_prisms done" << endl;
	}
}

void drawable_set_of_objects::init_planes(int group_idx,
		const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_planes" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 4;
	//d = thickness;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_planes done" << endl;
	}
}

void drawable_set_of_objects::init_lines(int group_idx,
		double rad, const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_lines" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 5;
	d = rad;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_lines done" << endl;
	}
}

void drawable_set_of_objects::init_cubics(int group_idx,
		const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_cubics" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 6;
	//d = rad;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_cubics done" << endl;
	}
}

void drawable_set_of_objects::init_quadrics(int group_idx,
		const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_quadrics" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 7;
	//d = rad;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_quadrics done" << endl;
	}
}

void drawable_set_of_objects::init_quartics(int group_idx,
		const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_quartics" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 8;
	//d = rad;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_quartics done" << endl;
	}
}

void drawable_set_of_objects::init_labels(int group_idx,
		double thickness_half, double scale, const char *properties, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "drawable_set_of_objects::init_labels" << endl;
	}
	drawable_set_of_objects::group_idx = group_idx;
	type = 9;
	d = thickness_half;
	d2 = scale;
	drawable_set_of_objects::properties = properties;


	if (f_v) {
		cout << "drawable_set_of_objects::init_labels done" << endl;
	}
}


void drawable_set_of_objects::draw(animate *Anim, ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int sz;
	int *Selection;
	int j;

	if (f_v) {
		cout << "drawable_set_of_objects::draw" << endl;
	}
	if (f_v) {
		cout << "group_idx = " << group_idx << endl;
	}
	sz = Anim->S->group_of_things[group_idx].size();
	if (f_v) {
		cout << "sz = " << sz << endl;
	}
	Selection = NEW_int(sz);
	for (j = 0; j < sz; j++) {
		Selection[j] = Anim->S->group_of_things[group_idx][j];
	}
	if (f_v) {
		cout << "Selection: " << endl;
		int_vec_print(cout, Selection, sz);
		cout << endl;
	}
	if (type == 1) {
		if (f_v) {
			cout << "type == 1" << endl;
		}
		Anim->S->draw_points_with_selection(Selection, sz,
				d, properties, ost);
	}
	else if (type == 2) {
		if (f_v) {
			cout << "type == 2 cylinders = edges" << endl;
		}
		Anim->S->line_radius = d;
		Anim->S->draw_edges_with_selection(Selection, sz,
				properties, ost);
	}
	else if (type == 3) {
		if (f_v) {
			cout << "type == 3 prisms = faces" << endl;
		}
		Anim->S->draw_faces_with_selection(Selection, sz,
				d, properties, ost);
	}
	else if (type == 4) {
		if (f_v) {
			cout << "type == 4 planes" << endl;
		}
		Anim->S->draw_planes_with_selection(Selection, sz,
				properties, ost);
	}
	else if (type == 5) {
		if (f_v) {
			cout << "type == 5 lines" << endl;
		}
		Anim->S->line_radius = d;
		Anim->S->draw_lines_with_selection(Selection, sz,
				properties, ost);
	}
	else if (type == 6) {
		if (f_v) {
			cout << "type == 6 cubics" << endl;
		}
		Anim->S->draw_cubic_with_selection(Selection, sz,
				properties, ost);
	}
	else if (type == 7) {
		if (f_v) {
			cout << "type == 7 quadrics" << endl;
		}
		Anim->S->draw_quadric_with_selection(Selection, sz,
				properties, ost);
	}
	else if (type == 8) {
		if (f_v) {
			cout << "type == 8 quartics" << endl;
		}
		Anim->S->draw_quartic_with_selection(Selection, sz,
				properties, ost);
	}
	else if (type == 9) {
		if (f_v) {
			cout << "type == 9 labels" << endl;
		}
		Anim->draw_text_with_selection(Selection, sz,
				d /* thickness_half */, 0. /* extra_spacing */,
				d2 /* scale */,
				0. /* off_x */, 0. /* off_y */, 0. /* off_z */,
				properties, "" /* group_options */,
				ost, verbose_level);
	}
	else {
		cout << "drawable type unrecognized" << endl;
		exit(1);
	}
	FREE_int(Selection);
	if (f_v) {
		cout << "drawable_set_of_objects::draw done" << endl;
	}
}

}}

