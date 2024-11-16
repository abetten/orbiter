/*
 * export_group.cpp
 *
 *  Created on: Nov 16, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {

export_group::export_group()
{
	A_base = NULL;
	A_induced = NULL;
	Strong_gens = NULL;

}

export_group::~export_group()
{
}

void export_group::init(
		actions::action *A_base,
		actions::action *A_induced,
		groups::strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "export_group::init" << endl;
	}

	export_group::A_base = A_base;
	export_group::A_induced = A_induced;
	export_group::Strong_gens = Strong_gens;


	if (f_v) {
		cout << "export_group::init done" << endl;
	}
}


}}}

