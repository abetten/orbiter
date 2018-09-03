// object_in_projective_space_with_action.C
// 
// Anton Betten
//
// December 30, 2017
//
//
// 
//
//

#include "orbiter.h"


object_in_projective_space_with_action::object_in_projective_space_with_action()
{
	null();
}

object_in_projective_space_with_action::~object_in_projective_space_with_action()
{
	freeself();
}

void object_in_projective_space_with_action::null()
{
	OiP = NULL;
	Aut_gens = NULL;
}

void object_in_projective_space_with_action::freeself()
{
	null();
}

void object_in_projective_space_with_action::init(
	object_in_projective_space *OiP,
	strong_generators *Aut_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space_with_action::init" << endl;
		}

	object_in_projective_space_with_action::OiP = OiP;
	object_in_projective_space_with_action::Aut_gens = Aut_gens;
	
	if (f_v) {
		cout << "object_in_projective_space_with_action::init done" << endl;
		}
}


