// action_on_bricks.cpp
//
// Anton Betten
// Jan 10, 2013

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_bricks::action_on_bricks()
{
	Record_birth();
	A = NULL;
	B = NULL;
	degree = 0;
	f_linear_action = false;
}

action_on_bricks::~action_on_bricks()
{
	Record_death();
}

void action_on_bricks::init(
		actions::action *A,
		combinatorics::puzzles::brick_domain *B,
	int f_linear_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_bricks::init q=" << B->q
				<< " f_linear_action=" << f_linear_action << endl;
		}
	action_on_bricks::A = A;
	action_on_bricks::B = B;
	action_on_bricks::f_linear_action = f_linear_action;
	degree = B->nb_bricks;
	if (f_v) {
		cout << "action_on_bricks::init degree=" << degree << endl;
		}
	if (f_v) {
		cout << "action_on_bricks::init done" << endl;
		}
}

long int action_on_bricks::compute_image(
		int *Elt,
		long int i, int verbose_level)
{
	long int j;

	if (f_linear_action) {
		j = compute_image_linear_action(Elt, i, verbose_level);
		}
	else {
		j = compute_image_permutation_action(Elt, i, verbose_level);
		}
	return j;
}

long int action_on_bricks::compute_image_linear_action(
		int *Elt,
		long int i, int verbose_level)
{
	//verbose_level = 3; // !!!
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int v[3], w[3];
	long int rk_v, rk_w;
	int vv[3], ww[3];
	long int rk_vv, rk_ww;
	long int j;

	if (f_v) {
		cout << "action_on_bricks::compute_image "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_bricks::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	B->unrank_coordinates(i, v[0], v[1], w[0], w[1], 0);
	v[2] = 1;
	w[2] = 1;
	if (f_v) {
		cout << "action_on_bricks::compute_image v=";
		Int_vec_print(cout, v, 3);
		cout << endl;
		cout << "action_on_bricks::compute_image w=";
		Int_vec_print(cout, w, 3);
		cout << endl;
		}
	
	B->F->Projective_space_basic->PG_element_rank_modified(
			v, 1, 3, rk_v);
	B->F->Projective_space_basic->PG_element_rank_modified(
			w, 1, 3, rk_w);
	if (f_v) {
		cout << "action_on_bricks::compute_image rk_v=" << rk_v << endl;
		cout << "action_on_bricks::compute_image rk_w=" << rk_w << endl;
		cout << "action_on_bricks::compute_image A=" << endl;
		A->Group_element->element_print_quick(Elt, cout);
		}
	rk_vv = A->Group_element->image_of(Elt, rk_v);
	rk_ww = A->Group_element->image_of(Elt, rk_w);
	if (f_v) {
		cout << "action_on_bricks::compute_image rk_vv=" << rk_vv << endl;
		cout << "action_on_bricks::compute_image rk_ww=" << rk_ww << endl;
		}
	B->F->Projective_space_basic->PG_element_unrank_modified(
			vv, 1, 3, rk_vv);
	B->F->Projective_space_basic->PG_element_unrank_modified(
			ww, 1, 3, rk_ww);
	if (f_v) {
		cout << "action_on_bricks::compute_image vv=";
		Int_vec_print(cout, vv, 3);
		cout << endl;
		cout << "action_on_bricks::compute_image ww=";
		Int_vec_print(cout, ww, 3);
		cout << endl;
		}
	if (vv[2] == 0) {
		cout << "action_on_bricks::compute_image vv[2] == 0" << endl;
		exit(1);
		}
	if (ww[2] == 0) {
		cout << "action_on_bricks::compute_image ww[2] == 0" << endl;
		exit(1);
		}
	B->F->Projective_space_basic->PG_element_normalize(
			vv, 1, 3);
	B->F->Projective_space_basic->PG_element_normalize(
			ww, 1, 3);
	if (f_v) {
		cout << "action_on_bricks::compute_image after normalize vv=";
		Int_vec_print(cout, vv, 3);
		cout << endl;
		cout << "action_on_bricks::compute_image after normalize ww=";
		Int_vec_print(cout, ww, 3);
		cout << endl;
		}
	j = B->rank_coordinates(vv[0], vv[1], ww[0], ww[1], 0);
	if (j < 0 || j >= degree) {
		cout << "action_on_bricks::compute_image "
				"j = " << j << " out of range" << endl;
		exit(1);
		}
	return j;
}

long int action_on_bricks::compute_image_permutation_action(
		int *Elt, long int i, int verbose_level)
{
	//verbose_level = 3; // !!!
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int x0, y0, x1, y1, x2, y2, x3, y3;
	int a, b, c, d;
	long int j;

	if (f_v) {
		cout << "action_on_bricks::compute_image_permutation_action "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_bricks::compute_image_permutation_action "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	B->unrank_coordinates(i, x0, y0, x1, y1, 0);
	a = x0 * B->q + y0;
	b = x1 * B->q + y1;

	if (f_v) {
		cout << "action_on_bricks::compute_image_permutation_action "
				"a=" << a << endl;
		cout << "action_on_bricks::compute_image_permutation_action "
				"b=" << b << endl;
		cout << "action_on_bricks::compute_image_permutation_action "
				"A=" << endl;
		A->Group_element->element_print_quick(Elt, cout);
		}
	c = A->Group_element->image_of(Elt, a);
	d = A->Group_element->image_of(Elt, b);
	if (f_v) {
		cout << "action_on_bricks::compute_image_permutation_action "
				"c=" << c << endl;
		cout << "action_on_bricks::compute_image_permutation_action "
				"d=" << d << endl;
		}
	y2 = c % B->q;
	x2 = c / B->q;
	y3 = d % B->q;
	x3 = d / B->q;

	j = B->rank_coordinates(x2, y2, x3, y3, 0);
	if (j < 0 || j >= degree) {
		cout << "action_on_bricks::compute_image_permutation_action "
				"j = " << j << " out of range" << endl;
		exit(1);
		}
	return j;
}


}}}

