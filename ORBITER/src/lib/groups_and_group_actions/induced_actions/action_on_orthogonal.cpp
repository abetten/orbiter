// action_on_orthogonal.C
//
// Anton Betten
// September 27, 2012

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

namespace orbiter {

action_on_orthogonal::action_on_orthogonal()
{
	null();
}

action_on_orthogonal::~action_on_orthogonal()
{
	free();
}

void action_on_orthogonal::null()
{
	O = NULL;
	v1 = NULL;
	v2 = NULL;
	w1 = NULL;
	w2 = NULL;
	f_on_points = FALSE;
	f_on_lines = FALSE;
	f_on_points_and_lines = FALSE;
}

void action_on_orthogonal::free()
{
	if (v1) {
		FREE_int(v1);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (w1) {
		FREE_int(w1);
		}
	if (w2) {
		FREE_int(w2);
		}
	null();
}

void action_on_orthogonal::init(action *original_action, orthogonal *O, int f_on_points, int f_on_lines, int f_on_points_and_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_orthogonal::init" << endl;
		cout << "f_on_lines=" << f_on_lines << endl;
		}
	if (!original_action->f_is_linear) {
		cout << "action_on_orthogonal::init original_action not of linear type" << endl;
		cout << "action " << original_action->group_prefix << endl;
		exit(1);
		}
	action_on_orthogonal::original_action = original_action;
	action_on_orthogonal::O = O;
	action_on_orthogonal::f_on_points = f_on_points;
	action_on_orthogonal::f_on_lines = f_on_lines;
	action_on_orthogonal::f_on_points_and_lines = f_on_points_and_lines;
	low_level_point_size = O->n + 1;
		
	v1 = NEW_int(low_level_point_size);
	v2 = NEW_int(low_level_point_size);
	w1 = NEW_int(low_level_point_size);
	w2 = NEW_int(low_level_point_size);
	
	if (f_on_points) {
		degree = O->nb_points;
		}
	else if (f_on_lines) {
		degree = O->nb_lines;
		}
	else if (f_on_points_and_lines) {
		degree = O->nb_points + O->nb_lines;
		}
	else {
		cout << "action_on_orthogonal::init no type of action given" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action_on_orthogonal::init degree=" << degree << endl;
		}
	
	if (f_v) {
		cout << "action_on_orthogonal::init done" << endl;
		}
}

int action_on_orthogonal::map_a_point(int *Elt, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;
	action *A;
	
	if (f_v) {
		cout << "action_on_orthogonal::map_a_point" << endl;
		}
	A = original_action;
	O->unrank_point(v1, 1 /* stride */, i, 0 /* verbose_level */);
	A->element_image_of_low_level(v1, w1, Elt, verbose_level - 1);
	j = O->rank_point(w1, 1 /* stride */, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_on_orthogonal::map_a_point done" << endl;
		}
	return j;
}

int action_on_orthogonal::map_a_line(int *Elt, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int j, p1, p2, q1, q2;
	action *A;
	
	if (f_v) {
		cout << "action_on_orthogonal::map_a_line" << endl;
		}
	A = original_action;
	O->unrank_line(p1, p2, i, 0 /*verbose_level */);
	O->unrank_point(v1, 1 /* stride */, p1, 0 /* verbose_level */);
	O->unrank_point(v2, 1 /* stride */, p2, 0 /* verbose_level */);
	A->element_image_of_low_level(v1, w1, Elt, verbose_level - 1);
	A->element_image_of_low_level(v2, w2, Elt, verbose_level - 1);
	q1 = O->rank_point(w1, 1 /* stride */, 0 /* verbose_level */);
	q2 = O->rank_point(w2, 1 /* stride */, 0 /* verbose_level */);
	j = O->rank_line(q1, q2, 0 /*verbose_level */);
	if (f_vv) {
		cout << "action_on_orthogonal::map_a_line i=" << i << " p1=" << p1 << " p2=" << p2 << " q1=" << q1 << " q2=" << q2 << " j=" << j << endl;
		}
	if (f_v) {
		cout << "action_on_orthogonal::map_a_line done" << endl;
		}
	return j;
}

int action_on_orthogonal::compute_image_int(int *Elt, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;
	//action *A;
	
	//A = original_action;

	if (f_v) {
		cout << "action_on_orthogonal::compute_image_int i = " << i << endl;
		//cout << "A->low_level_point_size=" << A->low_level_point_size << endl;
		//cout << "using action " << A->label << endl;
		}

	
	if (i >= degree) {
		cout << "action_on_orthogonal::compute_image_int i >= degree" << endl;
		}
	if (f_on_points) {
		j = map_a_point(Elt, i, verbose_level - 1);
		}
	else if (f_on_lines) {
		j = map_a_line(Elt, i, verbose_level - 1);
		}
	else if (f_on_points_and_lines) {
		if (i >= O->nb_points) {
			i -= O->nb_points;
			j = map_a_line(Elt, i, verbose_level - 1);
			j += O->nb_points;
			}
		else {
			j = map_a_point(Elt, i, verbose_level - 1);
			}
		}
	else {
		cout << "action_on_orthogonal::compute_image_int need to know the type of action" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "action_on_orthogonal::compute_image_int image of " << i << " is " << j << endl;
		}

	return j;
}

}


