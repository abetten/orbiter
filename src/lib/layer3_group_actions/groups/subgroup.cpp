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
	Record_birth();
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
	Record_death();
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
	other::data_structures::sorting Sorting;

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

void subgroup::create_sims(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup::create_sims" << endl;
	}

#if 0

	data_structures_groups::vector_ge *gen_vec;

	gen_vec = NEW_OBJECT(data_structures_groups::vector_ge);

	gen_vec->init(Subgroup_lattice->A, verbose_level);
	gen_vec->allocate(nb_gens, 0 /* verbose_level */);

	int i;

	for (i = 0; i < nb_gens; i++) {
		 Subgroup_lattice->Sims->element_unrank_lint(gens[i], gen_vec->ith(i));
	}

	long int target_go;

	target_go = group_order;

	if (f_v) {
		cout << "subgroup::create_sims "
				"before Subgroup_lattice->A->create_sims_from_generators_with_target_group_order_lint" << endl;
	}
	Sub = Subgroup_lattice->A->create_sims_from_generators_with_target_group_order_lint(
			gen_vec,
			target_go,
			verbose_level - 2);
	if (f_v) {
		cout << "subgroup::create_sims "
				"after Subgroup_lattice->A->create_sims_from_generators_with_target_group_order_lint" << endl;
	}
#endif

	group_theory_global Group_theory_global;
	long int *gens_long;

	gens_long = NEW_lint(nb_gens);
	Int_vec_copy_to_lint(gens, gens_long, nb_gens);

	if (f_v) {
		cout << "subgroup::create_sims "
				"before Group_theory_global.create_sims_for_subgroup_given_by_generator_ranks" << endl;
	}

	Sub = Group_theory_global.create_sims_for_subgroup_given_by_generator_ranks(
			Subgroup_lattice->A,
			Subgroup_lattice->Sims,
			gens_long, nb_gens, group_order /* subgroup_order */,
			verbose_level);

	if (f_v) {
		cout << "subgroup::create_sims "
				"after Group_theory_global.create_sims_for_subgroup_given_by_generator_ranks" << endl;
	}

	FREE_lint(gens_long);

	if (f_v) {
		cout << "subgroup::create_sims done" << endl;
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
		cout << "subgroup::init before create_sims" << endl;
	}
	create_sims(verbose_level - 2);
	if (f_v) {
		cout << "subgroup::init after create_sims" << endl;
	}


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
	other::data_structures::sorting Sorting;
	
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

void subgroup::report_elements_to_file(
		std::string &label, std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup::report_elements_to_file" << endl;
	}

	if (f_v) {
		cout << "subgroup::report_elements_to_file "
				"before Subgroup_sims->report_all_group_elements_to_file" << endl;
	}

	Sub->report_all_group_elements_to_file(
			label, label_tex,
			verbose_level);

	if (f_v) {
		cout << "subgroup::report_elements_to_file "
				"after Subgroup_sims->report_all_group_elements_to_file" << endl;
	}


	if (f_v) {
		cout << "subgroup::report_elements_to_file done" << endl;
	}
}

uint32_t subgroup::compute_hash()
// performs a sort of the group elements before hashing
{
	uint32_t hash;
	other::data_structures::sorting Sorting;
	other::data_structures::data_structures_global Data;

	Sorting.int_vec_heapsort(Elements, group_order);
	hash = Data.int_vec_hash(Elements, group_order);
	return hash;

}

int subgroup::is_subgroup_of(
		subgroup *Subgroup2)
{
	other::data_structures::sorting Sorting;
	int ret;

	ret = Sorting.int_vec_is_subset_of(
			Elements, group_order, Subgroup2->Elements, Subgroup2->group_order);
	return ret;
}



}}}


