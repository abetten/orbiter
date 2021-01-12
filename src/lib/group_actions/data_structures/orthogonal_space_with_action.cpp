/*
 * orthogonal_space_with_action.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


orthogonal_space_with_action::orthogonal_space_with_action()
{
	Descr = NULL;
	O = NULL;
}

orthogonal_space_with_action::~orthogonal_space_with_action()
{
}

void orthogonal_space_with_action::init(
		orthogonal_space_with_action_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init" << endl;
	}
	orthogonal_space_with_action::Descr = Descr;

	O = NEW_OBJECT(orthogonal);





	if (f_v) {
		cout << "orthogonal_space_with_action::init before O->init" << endl;
	}
	O->init(Descr->epsilon, Descr->n, Descr->F, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::init after O->init" << endl;
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::init before init_group" << endl;
	}
	init_group(verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::init after init_group" << endl;
	}


	if (f_v) {
		cout << "orthogonal_space_with_action::init done" << endl;
	}
}

void orthogonal_space_with_action::init_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group" << endl;
	}

#if 0
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"creating linear group" << endl;
	}

	vector_ge *nice_gens;

	A = NEW_OBJECT(action);
	A->init_linear_group(
		F, d,
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear,
		FALSE /* f_special */,
		nice_gens,
		0 /* verbose_level*/);
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"creating linear group done" << endl;
	}
	FREE_OBJECT(nice_gens);


	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"creating action on lines" << endl;
	}
	A_on_lines = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"creating action on lines done" << endl;
	}
#endif



	if (f_v) {
		cout << "orthogonal_space_with_action::init_group done" << endl;
	}
}

}}
