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
		FREE_lint(canonical_labeling);
	}
	null();
}

void object_in_projective_space_with_action::init(
	object_in_projective_space *OiP,
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
	object_in_projective_space_with_action::nb_rows = nb_rows;
	object_in_projective_space_with_action::nb_cols = nb_cols;
	object_in_projective_space_with_action::canonical_labeling = canonical_labeling;
	OiP->f_has_known_ago = TRUE;
	OiP->known_ago = Aut_gens->group_order_as_int();
	if (f_v) {
		cout << "object_in_projective_space_with_action::init done" << endl;
		}
}

void object_in_projective_space_with_action::init_known_ago(
	object_in_projective_space *OiP,
	int known_ago,
	int nb_rows, int nb_cols,
	long int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space_with_action::init_known_ago" << endl;
		}

	object_in_projective_space_with_action::OiP = OiP;
	object_in_projective_space_with_action::Aut_gens = NULL;
	object_in_projective_space_with_action::nb_rows = nb_rows;
	object_in_projective_space_with_action::nb_cols = nb_cols;
	object_in_projective_space_with_action::canonical_labeling = canonical_labeling;
	OiP->f_has_known_ago = TRUE;
	OiP->known_ago = known_ago;
	if (f_v) {
		cout << "object_in_projective_space_with_action::init_known_ago done" << endl;
		}
}


}}

