/*
 * group_table_and_generators.cpp
 *
 *  Created on: Sep 22, 2024
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


group_table_and_generators::group_table_and_generators()
{
	Table = NULL;
	group_order = 0;
	gens = NULL;
	nb_gens = 0;
}

group_table_and_generators::~group_table_and_generators()
{

}

void group_table_and_generators::init(
		groups::sims *Sims,
		data_structures_groups::vector_ge *generators,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_table_and_generators::init" << endl;
	}

	long int n;
	int i;

	if (f_v) {
		cout << "group_table_and_generators::init "
				"before Sims->create_group_table" << endl;
	}
	Sims->create_group_table(
			Table, n,
			verbose_level);
	if (f_v) {
		cout << "group_table_and_generators::init "
				"after Sims->create_group_table" << endl;
	}

	actions::action *A;
	int *Elt1;
	int k;

	A = Sims->A;

	Elt1 = NEW_int(A->elt_size_in_int);

	group_order = n;
	nb_gens = generators->len;

	gens = NEW_int(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->move(generators->ith(i), Elt1);
		k = Sims->element_rank_lint(Elt1);
		gens[i] = k;
	}


	FREE_int(Elt1);



	if (f_v) {
		cout << "group_table_and_generators::init done" << endl;
	}
}

void group_table_and_generators::init_basic(
		int *Table,
		int group_order,
		int *gens,
		int nb_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_table_and_generators::init_basic" << endl;
	}
	group_table_and_generators::Table = Table;
	group_table_and_generators::group_order = group_order;
	group_table_and_generators::gens = gens;
	group_table_and_generators::nb_gens = nb_gens;

	if (f_v) {
		cout << "group_table_and_generators::init_basic done" << endl;
	}
}



}}}

