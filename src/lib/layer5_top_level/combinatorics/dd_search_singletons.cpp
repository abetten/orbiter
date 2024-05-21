/*
 * dd_search_singletons.cpp
 *
 *  Created on: May 18, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {






dd_search_singletons::dd_search_singletons()
{
	DD_lifting = NULL;
	DD = NULL;

	orbit_idx = 0;
	target_depth = 0;
	level = 0;
	Live_points = NULL;
	chosen_set = NULL;
	index = NULL;
	nb_nodes = 0;
	//std::vector<std::vector<int>> Solutions;

}


dd_search_singletons::~dd_search_singletons()
{

}


void dd_search_singletons::search_case_singletons(
		dd_lifting *DD_lifting,
		int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "dd_search_singletons::search_case_singletons "
				<< orbit_idx << " / " << DD_lifting->ODF->nb_cases << endl;
	}

	dd_search_singletons::DD_lifting = DD_lifting;
	dd_search_singletons::DD = DD_lifting->DD;
	dd_search_singletons::orbit_idx = orbit_idx;


	target_depth = DD->Descr->K - DD->Descr->singletons_starter_size;

	if (f_v) {
		cout << "dd_search_singletons::search_case_singletons "
				"target_depth=" << target_depth << endl;
		cout << "dd_search_singletons::search_case_singletons "
				"nb_live_points=" << DD->nb_live_points << endl;
	}


	Live_points = NEW_OBJECT(data_structures::set_of_sets);

	Live_points->init_basic_constant_size(
			DD->nb_live_points,
			target_depth + 1, DD->nb_live_points,
			0 /*verbose_level*/);

	Lint_vec_copy(DD->live_points, Live_points->Sets[0], DD->nb_live_points);
	Live_points->Set_size[0] = DD->nb_live_points;





	chosen_set = NEW_lint(target_depth);
	index = NEW_int(target_depth);

	nb_nodes = 0;

	if (f_v) {
		cout << "dd_search_singletons::search_case_singletons "
				<< orbit_idx << " / " << DD_lifting->ODF->nb_cases
				<< " before search_case_singletons_recursion" << endl;
	}

	search_case_singletons_recursion(
			0 /* level */,
			verbose_level);

	if (f_v) {
		cout << "dd_search_singletons::search_case_singletons "
				<< orbit_idx << " / " << DD_lifting->ODF->nb_cases
				<< " after search_case_singletons_recursion" << endl;
	}

	if (f_v) {
		cout << "dd_search_singletons::search_case_singletons done" << endl;
	}
}




void dd_search_singletons::search_case_singletons_recursion(
		int level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (level == target_depth) {

		if (f_v) {
			cout << "dd_search_singletons::search_case_singletons "
					"solution: ";
			Lint_vec_print(cout, chosen_set, target_depth);
			cout << endl;
		}

		std::vector<int> v;
		int i;

		for (i = 0; i < target_depth; i++) {
			v.push_back(chosen_set[i]);
		}
		Solutions.push_back(v);

		return;
	}

	nb_nodes++;

	if ((nb_nodes % (1 << 20)) == 0) {
		cout << "dd_search_singletons::search_case_singletons "
				"level " << level << " nb_nodes=" << nb_nodes
				<< " nb_live = " << Live_points->Set_size[level] << " : ";
		cout << "dd_search_singletons::search_case_singletons "
				"case " << index[0] << " / " << Live_points->Set_size[0] << endl;
	}

	//cout << "level " << level << " nb_live = " << Live_points->Set_size[level] << endl;

	for (index[level] = 0; index[level] < Live_points->Set_size[level]; index[level]++) {

		//int i, o;
		long int c;

		c = Live_points->Sets[level][index[level]];


		DD->increase_orbit_covering_firm(
				DD_lifting->ODF->sets[orbit_idx],
				DD->Descr->singletons_starter_size,
				c);
		DD->increase_orbit_covering_firm(chosen_set, level, c);



		chosen_set[level] = c;

		// compute live points for the next level:

		Live_points->Set_size[level + 1] = 0;

		int h, flag;
		long int d;

		for (h = index[level] + 1; h < Live_points->Set_size[level]; h++) {

			Int_vec_copy(DD->orbit_covered, DD->orbit_covered2, DD->Orbits_on_pairs->nb_orbits);

			d = Live_points->Sets[level][h];

			flag = DD->try_to_increase_orbit_covering_based_on_two_sets(
					DD_lifting->ODF->sets[orbit_idx],
					DD->Descr->singletons_starter_size,
					chosen_set, level + 1,
					d);
			if (flag) {
				Live_points->Sets[level + 1][Live_points->Set_size[level + 1]++] = d;
			}

		}

		search_case_singletons_recursion(
				level + 1,
				verbose_level);

		c = chosen_set[level]; // redundant

		DD->decrease_orbit_covering(
				DD_lifting->ODF->sets[orbit_idx],
				DD->Descr->singletons_starter_size,
				c);
		DD->decrease_orbit_covering(chosen_set, level, c);


	}
}


}}}
