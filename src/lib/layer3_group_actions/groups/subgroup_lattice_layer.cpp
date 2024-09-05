/*
 * subgroup_lattice_layer.cpp
 *
 *  Created on: Aug 31, 2024
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



subgroup_lattice_layer::subgroup_lattice_layer()
{
	Subgroup_lattice = NULL;

	layer_idx = 0;

	//std::vector<long int> Divisors;

	Hash_table_subgroups = NULL;

	A_on_groups = NULL;

	Sch_on_groups = NULL;

#if 0
	//std::vector<void *> Subgroups;

	//std::multimap<uint32_t, int> Hashing;
#endif

}


subgroup_lattice_layer::~subgroup_lattice_layer()
{
}

void subgroup_lattice_layer::init(
		subgroup_lattice *Subgroup_lattice,
		int layer_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::init" << endl;
	}

	subgroup_lattice_layer::Subgroup_lattice = Subgroup_lattice;
	subgroup_lattice_layer::layer_idx = layer_idx;


	Hash_table_subgroups = NEW_OBJECT(data_structures_groups::hash_table_subgroups);

	if (f_v) {
		cout << "subgroup_lattice_layer::init done" << endl;
	}
}

groups::subgroup *subgroup_lattice_layer::get_subgroup(
		int group_idx)
{
	groups::subgroup *Subgroup;

	Subgroup = Hash_table_subgroups->get_subgroup(group_idx);
	//Subgroup = (groups::subgroup *) Subgroups[group_idx];

	return Subgroup;
}


void subgroup_lattice_layer::print(
		std::ostream &ost)
{
	long int *Go;
	int i;

	Go = NEW_lint(nb_subgroups());
	for (i = 0; i < nb_subgroups(); i++) {

		Go[i] = Hash_table_subgroups->get_subgroup(i)->group_order;

	}

	data_structures::tally_lint T;

	T.init(
			Go,
			nb_subgroups(), false /* f_second */,
			0 /* verbose_level */);

	cout << "layer " << layer_idx << " : ";
	Lint_vec_stl_print(cout, Divisors);
	cout << " : " << nb_subgroups();
	cout << " : ";
	T.print_bare(true /*f_backwards*/);
	cout << endl;

	FREE_lint(Go);
}

int subgroup_lattice_layer::add_subgroup(
		groups::subgroup *Subgroup,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::add_subgroup" << endl;
	}
	int f_new_group;

	f_new_group = Hash_table_subgroups->add_subgroup(Subgroup, verbose_level);


#if 0
	//long int order;

	//order = Subgroup->group_order;

	uint32_t hash;
	int pos;

	hash = Subgroup->compute_hash();

	if (!find_subgroup(Subgroup, pos, hash)) {

		if (f_v) {
			cout << "subgroup_lattice_layer::add_subgroup "
					"the subgroup is new" << endl;
		}

		Hashing.insert(pair<uint32_t, int>(hash, Subgroups.size()));

		Subgroups.push_back(Subgroup);
		if (f_v) {
			cout << "subgroup_lattice_layer::add_subgroup "
					"number of subgroups at layer " << layer_idx
					<< " is " << Subgroups.size() << endl;
		}
		f_new_group = true;

	}
	else {
		if (f_v) {
			cout << "subgroup_lattice_layer::add_subgroup "
					"the subgroup is already present" << endl;
		}
		f_new_group = false;

	}
#endif

	if (f_v) {
		cout << "subgroup_lattice_layer::add_subgroup done" << endl;
	}
	return f_new_group;
}

int subgroup_lattice_layer::find_subgroup(
		groups::subgroup *Subgroup,
		int &pos, uint32_t &hash, int verbose_level)
{
	int f_found;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup" << endl;
	}
	f_found = Hash_table_subgroups->find_subgroup(Subgroup, pos, hash, verbose_level - 1);
	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup done" << endl;
	}

#if 0
	data_structures::sorting Sorting;
	data_structures::data_structures_global Data;

	Sorting.int_vec_heapsort(Subgroup->Elements, Subgroup->group_order);
	hash = Data.int_vec_hash(Subgroup->Elements, Subgroup->group_order);

	map<uint32_t, int>::iterator itr, itr1, itr2;

	itr1 = Hashing.lower_bound(hash);
	itr2 = Hashing.upper_bound(hash);
	f_found = false;
	for (itr = itr1; itr != itr2; ++itr) {
    	pos = itr->second;

		groups::subgroup *Subgroup1;

		Subgroup1 = get_subgroup(pos);

		if (Subgroup1->group_order != Subgroup->group_order) {
			continue;
		}

        if (Sorting.int_vec_compare(
        		Subgroup->Elements, Subgroup1->Elements, Subgroup1->group_order) == 0) {
        	f_found = true;
        	break;
        }
	}
#endif

	return f_found;
}

int subgroup_lattice_layer::nb_subgroups()
{
	return Hash_table_subgroups->Subgroups.size();
}

void subgroup_lattice_layer::orbits_under_conjugation(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer = " << layer_idx << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer = " << layer_idx << " SG=" << endl;
		Subgroup_lattice->SG->print_generators(cout, verbose_level);
	}

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " before creating A_on_groups" << endl;
	}


	A_on_groups = Subgroup_lattice->A->Induced_action->create_induced_action_on_subgroups(
			Subgroup_lattice->Sims,
		Hash_table_subgroups,
		verbose_level);
	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " after creating A_on_groups" << endl;
	}



	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " before SG->compute_all_point_orbits_schreier" << endl;
	}
	Sch_on_groups = Subgroup_lattice->SG->compute_all_point_orbits_schreier(
			A_on_groups, verbose_level);

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " after SG->compute_all_point_orbits_schreier" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " The conjugacy classes of groups have the following lengths: ";
		Sch_on_groups->print_orbit_length_distribution(cout);
	}

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer = " << layer_idx << " done" << endl;
	}
}


int subgroup_lattice_layer::extend_layer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::extend_layer layer = " << layer_idx << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::extend_layer layer = " << layer_idx
				<< " before orbits_under_conjugation" << endl;
	}
	orbits_under_conjugation(
			verbose_level);
	if (f_v) {
		cout << "subgroup_lattice::extend_layer layer = " << layer_idx
				<< " after orbits_under_conjugation" << endl;
	}

	int group_idx;
	int *Nb_new_groups;
	int nb_new_groups_total = 0;

	Nb_new_groups = NEW_int(nb_subgroups());

	for (group_idx = 0; group_idx < nb_subgroups(); group_idx++) {
		if (f_v) {
			cout << "subgroup_lattice::extend_layer layer = " << layer_idx
					<< " before extend_group " << group_idx << " / " << nb_subgroups() << endl;
		}
		Nb_new_groups[group_idx] = extend_group(group_idx, verbose_level);
		nb_new_groups_total += Nb_new_groups[group_idx];
		if (f_v) {
			cout << "subgroup_lattice::extend_layer layer = " << layer_idx
					<< " after extend_group " << group_idx << " / " << nb_subgroups()
					<< " : nb_new_groups = " << Nb_new_groups[group_idx]
					<< " : " << nb_new_groups_total << endl;
			Subgroup_lattice->print();
		}

		if (f_v) {
			cout << "subgroup_lattice::extend_layer "
					"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
					<< " before save_csv" << endl;
		}
		Subgroup_lattice->save_csv(
				verbose_level);
		if (f_v) {
			cout << "subgroup_lattice::extend_layer "
					"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
					<< " after save_csv" << endl;
		}

	}

	if (f_v) {
		cout << "subgroup_lattice::extend_layer layer = " << layer_idx << " done" << endl;
	}
	return nb_new_groups_total;
}

int subgroup_lattice_layer::extend_group(
		int group_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::extend_group" << endl;
	}

	groups::subgroup *Subgroup1;

	Subgroup1 = get_subgroup(group_idx);

	int z;
	int *gens;
	int nb_gens;
	int *cosets;
	int *group;
	int group_sz;
	int nb_new_groups;

	cosets = NEW_int(Subgroup_lattice->group_order);
	group = NEW_int(Subgroup_lattice->group_order);
	gens = NEW_int(Subgroup_lattice->group_order);

	nb_new_groups = 0;

	for (z = 0; z < Subgroup_lattice->Zuppos.size(); z++) {

		int f_new_group;

		nb_gens = Subgroup1->nb_gens;
		Int_vec_copy(Subgroup1->gens, gens, Subgroup1->nb_gens);


		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size() << ":" << endl;
		}
		Subgroup_lattice->Sims->dimino(
				Subgroup1->Elements,
				Subgroup1->group_order,
				gens, nb_gens,
			cosets,
			Subgroup_lattice->Zuppos[z] /* new_gen*/,
			group, group_sz,
			0 /* verbose_level */);

		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size()
					<< ": found a group of order " << group_sz << " : generators: ";
			Int_vec_print(cout, gens, nb_gens);
			cout << endl;
		}

		groups::subgroup *Subgroup;

		Subgroup = NEW_OBJECT(groups::subgroup);

		Subgroup->init(
				Subgroup_lattice,
				group, group_sz, gens, nb_gens,
				verbose_level - 1);

		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size()
					<< ": before Subgroup_lattice->add_subgroup" << endl;
		}
		f_new_group = Subgroup_lattice->add_subgroup(
				Subgroup,
				verbose_level);
		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size()
					<< ": after Subgroup_lattice->add_subgroup f_new_group=" << f_new_group << endl;
		}
		if (!f_new_group) {
			FREE_OBJECT(Subgroup);
		}
		else {
			nb_new_groups++;
		}

	}

	FREE_int(cosets);
	FREE_int(group);
	FREE_int(gens);


	if (f_v) {
		cout << "subgroup_lattice::extend_group done" << endl;
	}
	return nb_new_groups;
}



}}}



