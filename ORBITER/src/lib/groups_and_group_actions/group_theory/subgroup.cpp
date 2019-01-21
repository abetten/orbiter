// subgroup.C
//
// Anton Betten
// April 29, 2017

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

namespace orbiter {

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
		FREE_int(Elements);
		}
	if (gens) {
		FREE_int(gens);
		}
	null();
}

void subgroup::init(int *Elements, int group_order, int *gens, int nb_gens)
{
	subgroup::Elements = NEW_int(group_order);
	subgroup::gens = NEW_int(nb_gens);
	subgroup::group_order = group_order;
	subgroup::nb_gens = nb_gens;
	int_vec_copy(Elements, subgroup::Elements, group_order);
	int_vec_copy(gens, subgroup::gens, nb_gens);
}

void subgroup::print()
{
	cout << "group of order " << group_order << " : ";
	int_vec_print(cout, Elements, group_order);
	cout << " gens: ";
	int_vec_print(cout, gens, nb_gens);
	cout << endl;
}

int subgroup::contains_this_element(int elt)
{
	int idx;
	
	if (int_vec_search(Elements, group_order, elt, idx)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

}

