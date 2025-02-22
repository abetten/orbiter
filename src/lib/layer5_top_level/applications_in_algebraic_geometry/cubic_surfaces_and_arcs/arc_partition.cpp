/*
 * arc_partition.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {



arc_partition::arc_partition()
{
	Record_birth();
	OP = NULL;

	A = NULL;
	A_on_arc = NULL;

	pair_orbit_idx = -1;
	The_pair = NULL;

	//int arc_remainder[4];

	A_on_rest = NULL;
	A_on_partition = NULL;

	Orbits_on_partition = NULL;

	nb_orbits_on_partition = 0;
}


arc_partition::~arc_partition()
{
	Record_death();
	if (The_pair) {
		FREE_OBJECT(The_pair);
	}
	if (A_on_rest) {
		FREE_OBJECT(A_on_rest);
	}
	if (A_on_partition) {
		FREE_OBJECT(A_on_partition);
	}
	if (Orbits_on_partition) {
		FREE_OBJECT(Orbits_on_partition);
	}
}

void arc_partition::init(
	arc_orbits_on_pairs *OP, int pair_orbit_idx,
	actions::action *A, actions::action *A_on_arc,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "arc_partition::init" << endl;
	}

	arc_partition::OP = OP;
	arc_partition::pair_orbit_idx = pair_orbit_idx;
	arc_partition::A = A;
	arc_partition::A_on_arc = A_on_arc;


	if (f_v) {
		cout << "arc_partition::init "
				"creating The_arc" << endl;
	}

	The_pair = OP->Orbits_on_pairs->get_set_and_stabilizer(
			2 /* level */,
			pair_orbit_idx,
			0 /* verbose_level */);


	other::orbiter_kernel_system::Orbiter->Lint_vec->complement(The_pair->data, arc_remainder, 6, 2);
	if (f_v) {
		cout << "arc_partition::init "
				"the pair is :";
		Lint_vec_print(cout, The_pair->data, 2);
		cout << endl;
		cout << "arc_partition::init "
				"the remainder is :";
		Lint_vec_print(cout, arc_remainder, 4);
		cout << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set = "_remaining4";
	label_of_set_tex = "\\_remaining4";


	if (f_v) {
		cout << "arc_partition::init "
				"creating restricted action on the arc" << endl;
	}

	A_on_rest = A_on_arc->Induced_action->restricted_action(
			arc_remainder, 4 /* nb_points */,
			label_of_set, label_of_set_tex,
			verbose_level);

	if (f_v) {
		cout << "arc_partition::init "
				"creating action on the partition" << endl;
	}

	A_on_partition = A_on_rest->Induced_action->induced_action_on_set_partitions(
			2,
			verbose_level);

	Orbits_on_partition = NEW_OBJECT(groups::schreier);

	if (f_v) {
		cout << "arc_partition::init "
				"before A_on_rest->all_point_orbits_from_generators" << endl;
	}

	actions::action_global AcGl;

	AcGl.all_point_orbits_from_generators(
			A_on_partition,
			*Orbits_on_partition,
			The_pair->Strong_gens,
			verbose_level);

	nb_orbits_on_partition = Orbits_on_partition->nb_orbits;

	if (f_v) {
		cout << "arc_partition::init done" << endl;
	}
}

void arc_partition::recognize(
		int *partition, int *transporter,
		int &orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int partition_idx;
	combinatorics::other_combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "arc_partition::recognize" << endl;
	}


	partition_idx = Combi.set_partition_4_into_2_rank(partition);

	if (f_v) {
		cout << "arc_partition::recognize partition_idx=" << partition_idx << endl;
	}

	Orbits_on_partition->transporter_from_point_to_orbit_rep(partition_idx,
			orbit_idx, transporter, verbose_level - 2);

	if (f_v) {
		cout << "arc_partition::recognize orbit_idx=" << orbit_idx << endl;
	}


	if (f_v) {
		cout << "arc_partition::recognize done" << endl;
	}
}



}}}}

