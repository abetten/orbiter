// poset.C
//
// Anton Betten
// November 19, 2018

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"



poset::poset()
{
	null();
}

poset::~poset()
{
	freeself();
}

void poset::null()
{
	description = NULL;

	A = NULL;
	A2 = NULL;
	Strong_gens = NULL;

	longinteger_object go;

}

void poset::freeself()
{
	null();
}

void poset::init_subset_lattice(action *A, action *A2,
		strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::init_subset_lattice" << endl;
		}
	f_subset_lattice = TRUE;
	n = A2->degree;
	f_subspace_lattice = FALSE;
	poset::A = A;
	poset::A2 = A2;
	poset::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	if (f_v) {
		cout << "poset::init_subset_lattice done" << endl;
		}
}

void poset::init_subspace_lattice(action *A, action *A2,
		strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::init_subspace_lattice" << endl;
		}
	f_subset_lattice = FALSE;
	n = A2->degree;
	f_subspace_lattice = TRUE;
	poset::A = A;
	poset::A2 = A2;
	poset::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	if (f_v) {
		cout << "poset::init_subspace_lattice done" << endl;
		}
}

void poset::init(
		poset_description *description,
		action *A, // the action in which the group is given
		action *A2, // the action in which we do the search
		strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset::init" << endl;
		}
	poset::description = description;
	f_subset_lattice = description->f_subset_lattice;
	n = description->n;
	f_subspace_lattice = description->f_subspace_lattice;
	poset::A = A;
	poset::A2 = A2;
	poset::Strong_gens = Strong_gens;
	Strong_gens->group_order(go);
	if (f_v) {
		cout << "poset::init Action A:" << endl;
		A->print_info();
		}
	if (f_v) {
		cout << "poset::init Action A2:" << endl;
		A2->print_info();
		}
	if (f_v) {
		cout << "poset::init gnerators for a group of order " << go
				<< " and degree " << A2->degree << endl;
		}
	if (f_v) {
		cout << "poset::init done" << endl;
		}
}

void poset::print()
{
	if (f_subset_lattice) {
		cout << "poset of subsets" << endl;
	}
	if (f_subspace_lattice) {
		cout << "poset of subspaces" << endl;
	}
	A->print_info();
	A2->print_info();
	cout << "group order " << go << endl;
}
