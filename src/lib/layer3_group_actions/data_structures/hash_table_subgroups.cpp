/*
 * hash_table_subgroups.cpp
 *
 *  Created on: Sep 3, 2024
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {

hash_table_subgroups::hash_table_subgroups()
{
	Record_birth();

	//std::vector<void *> Subgroups;

	//std::multimap<uint32_t, int> Hashing;

}

hash_table_subgroups::~hash_table_subgroups()
{
	Record_death();
}


int hash_table_subgroups::nb_groups()
{
	return Subgroups.size();
}

int hash_table_subgroups::add_subgroup(
		groups::subgroup *Subgroup,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hash_table_subgroups::add_subgroup" << endl;
	}


	uint32_t hash;
	int pos;
	int f_new_group;

	hash = Subgroup->compute_hash();

	if (!find_subgroup(Subgroup, pos, hash, verbose_level - 1)) {

		if (f_v) {
			cout << "hash_table_subgroups::add_subgroup "
					"the subgroup is new" << endl;
		}

		Hashing.insert(pair<uint32_t, int>(hash, Subgroups.size()));

		Subgroups.push_back(Subgroup);
		if (f_v) {
			cout << "hash_table_subgroups::add_subgroup "
					"number of subgroups is " << Subgroups.size() << endl;
		}
		f_new_group = true;

	}
	else {
		if (f_v) {
			cout << "hash_table_subgroups::add_subgroup "
					"the subgroup is already present" << endl;
		}
		f_new_group = false;

	}
	if (f_v) {
		cout << "hash_table_subgroups::add_subgroup done" << endl;
	}
	return f_new_group;
}

int hash_table_subgroups::find_subgroup(
		groups::subgroup *Subgroup,
		int &pos, uint32_t &hash, int verbose_level)
{
	other::data_structures::sorting Sorting;
	other::data_structures::data_structures_global Data;

	Sorting.int_vec_heapsort(Subgroup->Elements, Subgroup->group_order);
	hash = Data.int_vec_hash(Subgroup->Elements, Subgroup->group_order);

	map<uint32_t, int>::iterator itr, itr1, itr2;
	int f_found;

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
	return f_found;
}

int hash_table_subgroups::find_subgroup_direct(
		int *Elements, int group_order,
		int &pos, uint32_t &hash, int verbose_level)
{
	other::data_structures::sorting Sorting;
	other::data_structures::data_structures_global Data;

	Sorting.int_vec_heapsort(Elements, group_order);
	hash = Data.int_vec_hash(Elements, group_order);

	map<uint32_t, int>::iterator itr, itr1, itr2;
	int f_found;

	itr1 = Hashing.lower_bound(hash);
	itr2 = Hashing.upper_bound(hash);
	f_found = false;
	for (itr = itr1; itr != itr2; ++itr) {
    	pos = itr->second;

		groups::subgroup *Subgroup1;

		Subgroup1 = get_subgroup(pos);

		if (Subgroup1->group_order != group_order) {
			continue;
		}

        if (Sorting.int_vec_compare(
        		Elements, Subgroup1->Elements, Subgroup1->group_order) == 0) {
        	f_found = true;
        	break;
        }
	}
	return f_found;
}


groups::subgroup *hash_table_subgroups::get_subgroup(
		int group_idx)
{
	groups::subgroup *Subgroup;

	Subgroup = (groups::subgroup *) Subgroups[group_idx];

	return Subgroup;
}




}}}


