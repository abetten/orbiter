// action_on_orthogonal.cpp
//
// Anton Betten
// September 27, 2012

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_orthogonal::action_on_orthogonal()
{
	original_action = NULL;
	O = NULL;
	v1 = NULL;
	v2 = NULL;
	w1 = NULL;
	w2 = NULL;
	f_on_points = FALSE;
	f_on_lines = FALSE;
	f_on_points_and_lines = FALSE;
	low_level_point_size = 0;
	degree = 0;
}

action_on_orthogonal::~action_on_orthogonal()
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
}

void action_on_orthogonal::init(actions::action *original_action,
		orthogonal_geometry::orthogonal *O,
		int f_on_points,
		int f_on_lines,
		int f_on_points_and_lines,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_orthogonal::init" << endl;
		cout << "f_on_lines=" << f_on_lines << endl;
	}
	if (!original_action->f_is_linear) {
		cout << "action_on_orthogonal::init "
				"original_action not of linear type" << endl;
		cout << "action " << original_action->label << endl;
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
		degree = O->Hyperbolic_pair->nb_points;
	}
	else if (f_on_lines) {
		degree = O->Hyperbolic_pair->nb_lines;
	}
	else if (f_on_points_and_lines) {
		degree = O->Hyperbolic_pair->nb_points + O->Hyperbolic_pair->nb_lines;
	}
	else {
		cout << "action_on_orthogonal::init "
				"no type of action given" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "action_on_orthogonal::init "
				"degree=" << degree << endl;
	}
	
	if (f_v) {
		cout << "action_on_orthogonal::init done" << endl;
	}
}

void action_on_orthogonal::unrank_point(int *v, int rk)
{
	O->Hyperbolic_pair->unrank_point(v, 1 /* stride */, rk, 0 /* verbose_level */);
}

int action_on_orthogonal::rank_point(int *v)
{
	int rk;

	rk = O->Hyperbolic_pair->rank_point(v, 1 /* stride */, 0 /* verbose_level */);
	return rk;
}

long int action_on_orthogonal::map_a_point(
		int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int j;
	actions::action *A;
	
	if (f_v) {
		cout << "action_on_orthogonal::map_a_point" << endl;
	}
	A = original_action;

	O->Hyperbolic_pair->unrank_point(v1, 1 /* stride */, i, 0 /* verbose_level */);

	A->element_image_of_low_level(v1, w1, Elt, verbose_level - 1);

	j = O->Hyperbolic_pair->rank_point(w1, 1 /* stride */, 0 /* verbose_level */);

	if (f_v) {
		cout << "action_on_orthogonal::map_a_point done" << endl;
	}
	return j;
}

long int action_on_orthogonal::map_a_line(int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int j;
	long int p1, p2, q1, q2;
	actions::action *A;
	
	if (f_v) {
		cout << "action_on_orthogonal::map_a_line" << endl;
	}
	A = original_action;

	O->Hyperbolic_pair->unrank_line(p1, p2, i, 0 /*verbose_level */);

	O->Hyperbolic_pair->unrank_point(v1, 1 /* stride */, p1, 0 /* verbose_level */);
	O->Hyperbolic_pair->unrank_point(v2, 1 /* stride */, p2, 0 /* verbose_level */);

	A->element_image_of_low_level(v1, w1, Elt, verbose_level - 1);
	A->element_image_of_low_level(v2, w2, Elt, verbose_level - 1);

	q1 = O->Hyperbolic_pair->rank_point(w1, 1 /* stride */, 0 /* verbose_level */);
	q2 = O->Hyperbolic_pair->rank_point(w2, 1 /* stride */, 0 /* verbose_level */);

	j = O->Hyperbolic_pair->rank_line(q1, q2, 0 /*verbose_level */);

	if (f_vv) {
		cout << "action_on_orthogonal::map_a_line i=" << i
				<< " p1=" << p1 << " p2=" << p2
				<< " q1=" << q1 << " q2=" << q2 << " j=" << j << endl;
	}
	if (f_v) {
		cout << "action_on_orthogonal::map_a_line done" << endl;
	}
	return j;
}

long int action_on_orthogonal::compute_image_int(
		int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int j;
	//action *A;
	
	//A = original_action;

	if (f_v) {
		cout << "action_on_orthogonal::compute_image_int "
				"i = " << i << endl;
		//cout << "A->low_level_point_size="
		//<< A->low_level_point_size << endl;
		//cout << "using action " << A->label << endl;
	}

	
	if (i >= degree) {
		cout << "action_on_orthogonal::compute_image_int "
				"i >= degree" << endl;
	}
	if (f_on_points) {
		j = map_a_point(Elt, i, verbose_level - 1);
	}
	else if (f_on_lines) {
		j = map_a_line(Elt, i, verbose_level - 1);
	}
	else if (f_on_points_and_lines) {
		if (i >= O->Hyperbolic_pair->nb_points) {

			i -= O->Hyperbolic_pair->nb_points;

			j = map_a_line(Elt, i, verbose_level - 1);

			j += O->Hyperbolic_pair->nb_points;
		}
		else {
			j = map_a_point(Elt, i, verbose_level - 1);
		}
	}
	else {
		cout << "action_on_orthogonal::compute_image_int "
				"need to know the type of action" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "action_on_orthogonal::compute_image_int "
				"image of " << i << " is " << j << endl;
	}

	return j;
}

}}}



