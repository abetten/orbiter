// subgroup.C
//
// Anton Betten
// April 29, 2017

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

subgroup::subgroup()
{
	null();
}

subgroup::~subgroup()
{
	freeself();
}

void subgroup::null()
{
	Elements = NULL;
	gens = NULL;
}

void subgroup::freeself()
{
	if (Elements) {
		FREE_INT(Elements);
		}
	if (gens) {
		FREE_INT(gens);
		}
	null();
}

void subgroup::init(INT *Elements, INT group_order, INT *gens, INT nb_gens)
{
	subgroup::Elements = NEW_INT(group_order);
	subgroup::gens = NEW_INT(nb_gens);
	subgroup::group_order = group_order;
	subgroup::nb_gens = nb_gens;
	INT_vec_copy(Elements, subgroup::Elements, group_order);
	INT_vec_copy(gens, subgroup::gens, nb_gens);
}

void subgroup::print()
{
	cout << "group of order " << group_order << " : ";
	INT_vec_print(cout, Elements, group_order);
	cout << " gens: ";
	INT_vec_print(cout, gens, nb_gens);
	cout << endl;
}

INT subgroup::contains_this_element(INT elt)
{
	INT idx;
	
	if (INT_vec_search(Elements, group_order, elt, idx)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


