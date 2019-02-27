/*
 * arc_orbits_on_pairs.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

arc_orbits_on_pairs::arc_orbits_on_pairs()
{
	null();
}

arc_orbits_on_pairs::~arc_orbits_on_pairs()
{
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (Orbits_on_pairs) {
		FREE_OBJECT(Orbits_on_pairs);
	}
	if (The_arc) {
		FREE_OBJECT(The_arc);
	}
	if (A_on_arc) {
		FREE_OBJECT(A_on_arc);
	}
	if (Table_orbits_on_partition) {
		FREE_OBJECTS(Table_orbits_on_partition);
	}
	freeself();
}

void arc_orbits_on_pairs::null()
{
	SAL = NULL;

	A = NULL;

	The_arc = NULL;
	A_on_arc = NULL;

	arc_idx = -1;
	Poset = NULL;
	Orbits_on_pairs = NULL;

	nb_orbits_on_pairs = -1;
	Table_orbits_on_partition = NULL;
	total_nb_orbits_on_partitions = -1;
}

void arc_orbits_on_pairs::freeself()
{
	null();
}

void arc_orbits_on_pairs::init(
	surfaces_arc_lifting *SAL, int arc_idx,
	action *A,
	int argc, const char **argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "arc_orbits_on_pairs::init" << endl;
		}

	arc_orbits_on_pairs::SAL = SAL;
	arc_orbits_on_pairs::arc_idx = arc_idx;
	arc_orbits_on_pairs::A = A;


	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"creating The_arc" << endl;
		}

	The_arc = SAL->Six_arcs->Gen->gen->get_set_and_stabilizer(
			6 /* level */,
			SAL->Six_arcs->Not_on_conic_idx[arc_idx],
			0 /* verbose_level */);


	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"creating restricted action on the arc" << endl;
		}

	A_on_arc = A->restricted_action(The_arc->data, 6 /* nb_points */,
			verbose_level);

	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"creating poset" << endl;
		}
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A_on_arc,
			The_arc->Strong_gens,
			verbose_level);

	Orbits_on_pairs = NEW_OBJECT(poset_classification);
	Orbits_on_pairs->init(Poset,
		2 /* sz */, verbose_level);




	Orbits_on_pairs->f_print_function = FALSE;
	Orbits_on_pairs->print_function = NULL;
	Orbits_on_pairs->print_function_data = (void *) this;


	int nb_nodes = 20;

	if (f_v) {
		cout << "arc_orbits_on_pairs::init calling init_poset_orbit_node with "
				<< nb_nodes << " nodes" << endl;
		}

	Orbits_on_pairs->init_poset_orbit_node(nb_nodes, verbose_level - 1);

	if (f_v) {
		cout << "arc_orbits_on_pairs::init after init_poset_orbit_node" << endl;
		}

	if (f_v) {
		cout << "arc_orbits_on_pairs::init before init_root_node" << endl;
		}

	Orbits_on_pairs->root[0].init_root_node(
			Orbits_on_pairs, 0/*verbose_level - 2*/);


	if (f_v) {
		cout << "arc_orbits_on_pairs::init after init_root_node" << endl;
		}

	int t0 = os_ticks();

	if (f_v) {
		cout << "arc_orbits_on_pairs::init before Orbits_on_pairs->main" << endl;
		}
	Orbits_on_pairs->depth = 2;
	Orbits_on_pairs->main(t0, 2 /*schreier_depth */,
				TRUE /* f_use_invariant_subset_if_available */,
				FALSE /* f_debug*/,
				verbose_level);
	if (f_v) {
		cout << "arc_orbits_on_pairs::init after Orbits_on_pairs->main" << endl;
		}

	nb_orbits_on_pairs = Orbits_on_pairs->nb_orbits_at_level(2);
	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"nb_orbits_on_pairs=" << nb_orbits_on_pairs << endl;
		}


	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"computing orbits on partition" << endl;
		}


	Table_orbits_on_partition =
			NEW_OBJECTS(arc_partition, nb_orbits_on_pairs);
	int pair_orbit_idx;

	total_nb_orbits_on_partitions = 0;

	for (pair_orbit_idx = 0; pair_orbit_idx < nb_orbits_on_pairs; pair_orbit_idx++) {

		if (f_v) {
			cout << "arc_orbits_on_pairs::init "
					"before Table_orbits_on_pairs[" << arc_idx << "].init" << endl;
			}
		Table_orbits_on_partition[pair_orbit_idx].init(this, pair_orbit_idx,
				A, A_on_arc,
				argc, argv,
				verbose_level);
		total_nb_orbits_on_partitions +=
				Table_orbits_on_partition[pair_orbit_idx].
					nb_orbits_on_partition;
	}
	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"computing orbits on partition done" << endl;
		}


	if (f_v) {
		cout << "arc_orbits_on_pairs::init done" << endl;
		}
}

}}

