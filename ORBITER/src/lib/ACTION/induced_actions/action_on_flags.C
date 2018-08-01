// action_on_flags.C
//
// Anton Betten
// May 21, 2016

#include "GALOIS/galois.h"
#include "action.h"

action_on_flags::action_on_flags()
{
	null();
}

action_on_flags::~action_on_flags()
{
	free();
}

void action_on_flags::null()
{
	A = NULL;
	type = NULL;
	type_len = 0;
}

void action_on_flags::free()
{
	null();
}

void action_on_flags::init(action *A, INT *type, INT type_len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_flags::init" << endl;
		}
	action_on_flags::A = A;
	action_on_flags::type = type;
	action_on_flags::type_len = type_len;
	if (!A->f_is_linear) {
		cout << "action_on_flags::init the action must be linear but is not" << endl;
		exit(1);
		}
	n = A->dimension;
	if (A->type_G == matrix_group_t) {
		M = A->G.matrix_grp;
		}
	else {
		action *sub = A->subaction;
		M = sub->G.matrix_grp;
		}
	F = M->GFq;
	if (f_v) {
		cout << "action_on_flags::init n=" << n << " q=" << F->q << endl;
		}

	Flag = new flag;
	Flag->init(n, type, type_len, F, verbose_level);

	degree = Flag->N;
	if (f_v) {
		cout << "action_on_flags::init degree = " << degree << endl;
		}

	M1 = NEW_INT(n * n);
	M2 = NEW_INT(n * n);
	if (f_v) {
		cout << "action_on_flags::init done" << endl;
		}
}

INT action_on_flags::compute_image(INT *Elt, INT i, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h, j;

	if (f_v) {
		cout << "action_on_flags::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_flags::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "Elt=" << endl;
		A->element_print_quick(Elt, cout);
		}
	Flag->unrank(i, M1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action_on_flags::compute_image M1=" << endl;
		INT_matrix_print(M1, Flag->K, n);
		}
	if (f_v) {
		cout << "action_on_flags::compute_image before image_of_low_level" << endl;
		}
	for (h = 0; h < Flag->K; h++) {
		A->image_of_low_level(Elt, M1 + h * n, M2 + h * n);
		}
	if (f_v) {
		cout << "action_on_flags::compute_image after image_of_low_level" << endl;
		}
	if (f_v) {
		cout << "action_on_flags::compute_image M2=" << endl;
		INT_matrix_print(M2, Flag->K, n);
		}
	j = Flag->rank(M2, 0 /*verbose_level*/);

	if (f_v) {
		cout << "action_on_flags::compute_image " << i << " maps to " << j << endl;
		}
	return j;
}



