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

#include "foundations/foundations.h"
#include "group_actions.h"


namespace orbiter {
namespace group_actions {

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
	nb_rows = nb_cols = 0;
	canonical_labeling = NULL;
}

void object_in_projective_space_with_action::freeself()
{
	if (canonical_labeling) {
		FREE_int(canonical_labeling);
	}
	null();
}

void object_in_projective_space_with_action::init(
	object_in_projective_space *OiP,
	strong_generators *Aut_gens,
	int nb_rows, int nb_cols,
	int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space_with_action::init" << endl;
		}

	object_in_projective_space_with_action::OiP = OiP;
	object_in_projective_space_with_action::Aut_gens = Aut_gens;
	object_in_projective_space_with_action::nb_rows = nb_rows;
	object_in_projective_space_with_action::nb_cols = nb_cols;
	object_in_projective_space_with_action::canonical_labeling = canonical_labeling;
	
	if (f_v) {
		cout << "object_in_projective_space_with_action::init done" << endl;
		}
}


}}

