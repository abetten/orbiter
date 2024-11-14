// subgroup.cpp
//
// Anton Betten
// April 29, 2017

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



subgroup::subgroup()
{
	Subgroup_lattice = NULL;
	//A = NULL;
	Elements = NULL;
	group_order = 0;
	gens = NULL;
	nb_gens = 0;
	Sub = NULL;
	SG = NULL;
}

subgroup::~subgroup()
{
	if (Elements) {
		FREE_int(Elements);
	}
	if (gens) {
		FREE_int(gens);
	}
	if (Sub) {
		FREE_OBJECT(Sub);
	}
	if (SG) {
		FREE_OBJECT(SG);
	}
}

void subgroup::init_from_sims(
		groups::subgroup_lattice *Subgroup_lattice,
		sims *Sub,
		strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	long int i, rk;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "subgroup::init_from_sims" << endl;
	}
	//A = S->A;
	subgroup::Subgroup_lattice = Subgroup_lattice;
	subgroup::Sub = Sub;
	subgroup::SG = SG;
	group_order = SG->group_order_as_lint();
	if (f_v) {
		cout << "subgroup::init_from_sims "
				"group_order=" << group_order << endl;
	}
	Elt = NEW_int(Subgroup_lattice->A->elt_size_in_int);
	Elements = NEW_int(group_order);
	for (i = 0; i < group_order; i++) {
		Sub->element_unrank_lint(i, Elt);
		rk = Subgroup_lattice->Sims->element_rank_lint(Elt);
		Elements[i] = rk;
	}
	Sorting.int_vec_heapsort(Elements, group_order);

	if (f_v) {
		cout << "subgroup::init_from_sims done" << endl;
	}
}

void subgroup::init(
		groups::subgroup_lattice *Subgroup_lattice,
		int *Elements,
		int group_order, int *gens, int nb_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup::init" << endl;
	}

	subgroup::Subgroup_lattice = Subgroup_lattice;
	subgroup::Elements = NEW_int(group_order);
	subgroup::gens = NEW_int(nb_gens);
	subgroup::group_order = group_order;
	subgroup::nb_gens = nb_gens;
	Int_vec_copy(Elements, subgroup::Elements, group_order);
	Int_vec_copy(gens, subgroup::gens, nb_gens);
	if (f_v) {
		cout << "subgroup::init done" << endl;
	}
}

void subgroup::init_trivial_subgroup(
		groups::subgroup_lattice *Subgroup_lattice)
{
	subgroup::Subgroup_lattice = Subgroup_lattice;
	group_order = 1;
	Elements = NEW_int(group_order);
	Elements[0] = 0;
	nb_gens = 0;
	gens = NEW_int(nb_gens);
	group_order = 1;
}

void subgroup::print()
{
	cout << "group of order " << group_order << " : ";
	Int_vec_print(cout, Elements, group_order);
	cout << " gens: ";
	Int_vec_print(cout, gens, nb_gens);
	cout << endl;
}

int subgroup::contains_this_element(
		int elt)
{
	int idx;
	data_structures::sorting Sorting;
	
	if (Sorting.int_vec_search(Elements, group_order, elt, idx)) {
		return true;
	}
	else {
		return false;
	}
}

void subgroup::report(
		std::ostream &ost)
{
	SG->print_generators_tex(ost);
}

uint32_t subgroup::compute_hash()
// performs a sort of the group elements before hashing
{
	uint32_t hash;
	data_structures::sorting Sorting;
	data_structures::data_structures_global Data;

	Sorting.int_vec_heapsort(Elements, group_order);
	hash = Data.int_vec_hash(Elements, group_order);
	return hash;

}

int subgroup::is_subgroup_of(
		subgroup *Subgroup2)
{
	data_structures::sorting Sorting;
	int ret;

	ret = Sorting.int_vec_is_subset_of(
			Elements, group_order, Subgroup2->Elements, Subgroup2->group_order);
	return ret;
}



}}}


