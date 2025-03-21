// action_on_flags.cpp
//
// Anton Betten
// May 21, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_flags::action_on_flags()
{
	Record_birth();
	A = NULL;
	n = 0;
	F = NULL;
	type = NULL;
	type_len = 0;
	Flag = NULL;
	M = NULL;
	degree = 0;
	M1 = NULL;
	M2 = NULL;
}


action_on_flags::~action_on_flags()
{
	Record_death();
}

void action_on_flags::init(
		actions::action *A,
		int *type,
		int type_len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_flags::init" << endl;
		}
	action_on_flags::A = A;
	action_on_flags::type = type;
	action_on_flags::type_len = type_len;
	if (!A->f_is_linear) {
		cout << "action_on_flags::init the action must be "
				"linear but is not" << endl;
		exit(1);
		}
	n = A->dimension;
	if (A->type_G == matrix_group_t) {
		M = A->G.matrix_grp;
		}
	else {
		actions::action *sub = A->subaction;
		M = sub->G.matrix_grp;
		}
	F = M->GFq;
	if (f_v) {
		cout << "action_on_flags::init n=" << n << " q=" << F->q << endl;
		}

	Flag = NEW_OBJECT(geometry::other_geometry::flag);
	Flag->init(n, type, type_len, F, verbose_level);

	degree = Flag->N;
	if (f_v) {
		cout << "action_on_flags::init degree = " << degree << endl;
		}

	M1 = NEW_int(n * n);
	M2 = NEW_int(n * n);
	if (f_v) {
		cout << "action_on_flags::init done" << endl;
		}
}

long int action_on_flags::compute_image(
		int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int h, j;

	if (f_v) {
		cout << "action_on_flags::compute_image "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_flags::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "Elt=" << endl;
		A->Group_element->element_print_quick(Elt, cout);
		}
	Flag->unrank(i, M1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action_on_flags::compute_image M1=" << endl;
		Int_matrix_print(M1, Flag->K, n);
		}
	if (f_v) {
		cout << "action_on_flags::compute_image "
				"before image_of_low_level" << endl;
		}
	for (h = 0; h < Flag->K; h++) {
		A->Group_element->image_of_low_level(Elt,
				M1 + h * n, M2 + h * n, verbose_level - 1);
		}
	if (f_v) {
		cout << "action_on_flags::compute_image "
				"after image_of_low_level" << endl;
		}
	if (f_v) {
		cout << "action_on_flags::compute_image M2=" << endl;
		Int_matrix_print(M2, Flag->K, n);
		}
	j = Flag->rank(M2, 0 /*verbose_level*/);

	if (f_v) {
		cout << "action_on_flags::compute_image "
				<< i << " maps to " << j << endl;
		}
	return j;
}


}}}

