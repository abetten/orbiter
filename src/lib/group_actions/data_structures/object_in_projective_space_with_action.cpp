// object_in_projective_space_with_action.cpp
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

using namespace std;



namespace orbiter {
namespace group_actions {

object_in_projective_space_with_action::object_in_projective_space_with_action()
{
	OiP = NULL;
	Aut_gens = NULL;
	ago = 0;
	nb_rows = nb_cols = 0;
	canonical_labeling = NULL;
	//null();
}

object_in_projective_space_with_action::~object_in_projective_space_with_action()
{
	freeself();
}

void object_in_projective_space_with_action::null()
{
}

void object_in_projective_space_with_action::freeself()
{
	if (canonical_labeling) {
		FREE_lint(canonical_labeling);
	}
	null();
}

void object_in_projective_space_with_action::init(
	object_in_projective_space *OiP,
	long int ago,
	strong_generators *Aut_gens,
	int nb_rows, int nb_cols,
	long int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space_with_action::init" << endl;
	}

	object_in_projective_space_with_action::OiP = OiP;
	object_in_projective_space_with_action::Aut_gens = Aut_gens;
	object_in_projective_space_with_action::ago = ago;
	object_in_projective_space_with_action::nb_rows = nb_rows;
	object_in_projective_space_with_action::nb_cols = nb_cols;
	object_in_projective_space_with_action::canonical_labeling = canonical_labeling;
	OiP->f_has_known_ago = TRUE;
	OiP->known_ago = ago; //Aut_gens->group_order_as_lint();
	if (f_v) {
		cout << "object_in_projective_space_with_action::init done" << endl;
	 }
}




}}

