/*
 * action_on_interior_direct_product.cpp
 *
 *  Created on: Aug 22, 2021
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_interior_direct_product::action_on_interior_direct_product()
{
	A = NULL;
	nb_rows = 0;
	nb_cols = 0;
	degree = 0;
}

action_on_interior_direct_product::~action_on_interior_direct_product()
{
}

void action_on_interior_direct_product::init(
		actions::action *A,
		int nb_rows, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_interior_direct_product::init" << endl;
	}

	action_on_interior_direct_product::A = A;
	if (f_v) {
		cout << "action_on_interior_direct_product::init subaction = ";
		A->print_info();
	}
	action_on_interior_direct_product::nb_rows = nb_rows;
	action_on_interior_direct_product::nb_cols = A->degree - nb_rows;
	degree = nb_rows * nb_cols;
	if (f_v) {
		cout << "action_on_interior_direct_product::init nb_rows = " << nb_rows << endl;
		cout << "action_on_interior_direct_product::init nb_cols = " << nb_cols << endl;
		cout << "action_on_interior_direct_product::init degree = " << degree << endl;
	}

	if (f_v) {
		cout << "action_on_interior_direct_product::init done" << endl;
	}
}

long int action_on_interior_direct_product::compute_image(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, ii, jj, b;

	if (f_v) {
		cout << "action_on_interior_direct_product::compute_image "
				"verbose_level=" << verbose_level << " a=" << a << endl;
	}

	i = a / nb_cols;
	j = a % nb_cols;

	if (f_v) {
		cout << "action_on_interior_direct_product::compute_image "
				"computing image of " << i << endl;
	}
	ii = A->Group_element->element_image_of(i, Elt, verbose_level - 1);
	if (f_v) {
		cout << "action_on_interior_direct_product::compute_image "
				"computing image of " << j << endl;
	}
	jj = A->Group_element->element_image_of(j + nb_rows, Elt, verbose_level - 1) - nb_rows;
	b = ii * nb_cols + jj;

	if (f_v) {
		cout << "action_on_interior_direct_product::compute_image "
				"done " << a << "->" << b << endl;
	}
	return b;
}

}}}



