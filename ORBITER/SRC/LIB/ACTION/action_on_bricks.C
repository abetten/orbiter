// action_on_bricks.C
//
// Anton Betten
// Jan 10, 2013

#include "galois.h"
#include "action.h"

action_on_bricks::action_on_bricks()
{
	null();
}

action_on_bricks::~action_on_bricks()
{
	free();
}

void action_on_bricks::null()
{
	A = NULL;
	B = NULL;
}

void action_on_bricks::free()
{
	null();
}

void action_on_bricks::init(action *A, brick_domain *B, 
	INT f_linear_action, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_bricks::init q=" << B->q << " f_linear_action=" << f_linear_action << endl;
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

void action_on_bricks::compute_image(INT *Elt, INT i, INT &j, INT verbose_level)
{
	if (f_linear_action) {
		compute_image_linear_action(Elt, i, j, verbose_level);
		}
	else {
		compute_image_permutation_action(Elt, i, j, verbose_level);
		}
}

void action_on_bricks::compute_image_linear_action(INT *Elt, INT i, INT &j, INT verbose_level)
{
	//verbose_level = 3; // !!!
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT v[3], w[3], rk_v, rk_w;
	INT vv[3], ww[3], rk_vv, rk_ww;

	if (f_v) {
		cout << "action_on_bricks::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_bricks::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	B->unrank_coordinates(i, v[0], v[1], w[0], w[1], 0);
	v[2] = 1;
	w[2] = 1;
	if (f_v) {
		cout << "action_on_bricks::compute_image v=";
		INT_vec_print(cout, v, 3);
		cout << endl;
		cout << "action_on_bricks::compute_image w=";
		INT_vec_print(cout, w, 3);
		cout << endl;
		}
	
	PG_element_rank_modified(*B->F, v, 1, 3, rk_v);
	PG_element_rank_modified(*B->F, w, 1, 3, rk_w);
	if (f_v) {
		cout << "action_on_bricks::compute_image rk_v=" << rk_v << endl;
		cout << "action_on_bricks::compute_image rk_w=" << rk_w << endl;
		cout << "action_on_bricks::compute_image A=" << endl;
		A->element_print_quick(Elt, cout);
		}
	rk_vv = A->image_of(Elt, rk_v);
	rk_ww = A->image_of(Elt, rk_w);
	if (f_v) {
		cout << "action_on_bricks::compute_image rk_vv=" << rk_vv << endl;
		cout << "action_on_bricks::compute_image rk_ww=" << rk_ww << endl;
		}
	PG_element_unrank_modified(*B->F, vv, 1, 3, rk_vv);
	PG_element_unrank_modified(*B->F, ww, 1, 3, rk_ww);
	if (f_v) {
		cout << "action_on_bricks::compute_image vv=";
		INT_vec_print(cout, vv, 3);
		cout << endl;
		cout << "action_on_bricks::compute_image ww=";
		INT_vec_print(cout, ww, 3);
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
	PG_element_normalize(*B->F, vv, 1, 3);
	PG_element_normalize(*B->F, ww, 1, 3);
	if (f_v) {
		cout << "action_on_bricks::compute_image after normalize vv=";
		INT_vec_print(cout, vv, 3);
		cout << endl;
		cout << "action_on_bricks::compute_image after normalize ww=";
		INT_vec_print(cout, ww, 3);
		cout << endl;
		}
	j = B->rank_coordinates(vv[0], vv[1], ww[0], ww[1], 0);
	if (j < 0 || j >= degree) {
		cout << "action_on_bricks::compute_image j = " << j << " out of range" << endl;
		exit(1);
		}
}

void action_on_bricks::compute_image_permutation_action(INT *Elt, INT i, INT &j, INT verbose_level)
{
	//verbose_level = 3; // !!!
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT x0, y0, x1, y1, x2, y2, x3, y3;
	INT a, b, c, d;

	if (f_v) {
		cout << "action_on_bricks::compute_image_permutation_action i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_bricks::compute_image_permutation_action i = " << i << " out of range" << endl;
		exit(1);
		}
	B->unrank_coordinates(i, x0, y0, x1, y1, 0);
	a = x0 * B->q + y0;
	b = x1 * B->q + y1;

	if (f_v) {
		cout << "action_on_bricks::compute_image_permutation_action a=" << a << endl;
		cout << "action_on_bricks::compute_image_permutation_action b=" << b << endl;
		cout << "action_on_bricks::compute_image_permutation_action A=" << endl;
		A->element_print_quick(Elt, cout);
		}
	c = A->image_of(Elt, a);
	d = A->image_of(Elt, b);
	if (f_v) {
		cout << "action_on_bricks::compute_image_permutation_action c=" << c << endl;
		cout << "action_on_bricks::compute_image_permutation_action d=" << d << endl;
		}
	y2 = c % B->q;
	x2 = c / B->q;
	y3 = d % B->q;
	x3 = d / B->q;

	j = B->rank_coordinates(x2, y2, x3, y3, 0);
	if (j < 0 || j >= degree) {
		cout << "action_on_bricks::compute_image_permutation_action j = " << j << " out of range" << endl;
		exit(1);
		}
}



