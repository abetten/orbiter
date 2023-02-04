/*
 * arc_orbits_on_pairs.cpp
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


arc_orbits_on_pairs::arc_orbits_on_pairs()
{
	SAL = NULL;

	A = NULL;

	The_arc = NULL;
	A_on_arc = NULL;

	arc_idx = -1;
	Poset = NULL;
	Control = NULL;
	Orbits_on_pairs = NULL;

	nb_orbits_on_pairs = -1;
	Table_orbits_on_partition = NULL;
	total_nb_orbits_on_partitions = -1;

	partition_orbit_first = NULL;
	partition_orbit_len = NULL;
}

arc_orbits_on_pairs::~arc_orbits_on_pairs()
{
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	if (Control) {
		FREE_OBJECT(Control);
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
	if (partition_orbit_first) {
		FREE_int(partition_orbit_first);
	}
	if (partition_orbit_len) {
		FREE_int(partition_orbit_len);
	}
}

void arc_orbits_on_pairs::init(
	surfaces_arc_lifting *SAL, int arc_idx,
	actions::action *A,
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
		cout << "arc_orbits_on_pairs::init creating The_arc" << endl;
		}

	The_arc = SAL->Six_arcs->Gen->gen->get_set_and_stabilizer(
			6 /* level */,
			SAL->Six_arcs->Not_on_conic_idx[arc_idx],
			verbose_level);


	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"creating restricted action on the arc" << endl;
		}

	A_on_arc = A->Induced_action->restricted_action(
			The_arc->data, 6 /* nb_points */,
			verbose_level - 2);

	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"creating poset" << endl;
		}

	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Control->f_depth = TRUE;
	Control->depth = 2;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(A, A_on_arc,
			The_arc->Strong_gens,
			verbose_level);

	Poset->f_print_function = FALSE;
	Poset->print_function = NULL;
	Poset->print_function_data = (void *) this;


	Orbits_on_pairs = NEW_OBJECT(poset_classification::poset_classification);
	Orbits_on_pairs->initialize_and_allocate_root_node(Control, Poset,
		2 /* sz */, verbose_level);






	if (f_v) {
		cout << "arc_orbits_on_pairs::init after init_root_node" << endl;
	}
	orbiter_kernel_system::os_interface Os;

	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "arc_orbits_on_pairs::init before Orbits_on_pairs->main" << endl;
	}
	//Orbits_on_pairs->depth = 2;
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

	partition_orbit_first = NEW_int(nb_orbits_on_pairs);
	partition_orbit_len = NEW_int(nb_orbits_on_pairs);

	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"computing orbits on partition" << endl;
	}


	Table_orbits_on_partition = NEW_OBJECTS(arc_partition, nb_orbits_on_pairs);
	int pair_orbit_idx;

	total_nb_orbits_on_partitions = 0;

	for (pair_orbit_idx = 0;
			pair_orbit_idx < nb_orbits_on_pairs;
			pair_orbit_idx++) {

		if (f_v) {
			cout << "arc_orbits_on_pairs::init "
					"pair_orbit_idx = " << pair_orbit_idx << " / " << nb_orbits_on_pairs << ":" << endl;
		}

		int nb;

		if (f_v) {
			cout << "arc_orbits_on_pairs::init "
					"before Table_orbits_on_pairs[" << arc_idx << "].init" << endl;
		}
		Table_orbits_on_partition[pair_orbit_idx].init(this, pair_orbit_idx,
				A, A_on_arc,
				verbose_level);

		nb = Table_orbits_on_partition[pair_orbit_idx].nb_orbits_on_partition;

		if (f_v) {
			cout << "arc_orbits_on_pairs::init "
					"pair_orbit_idx = " << pair_orbit_idx << " / " << nb_orbits_on_pairs
					<< " has " << nb << " orbits on partitions" << endl;
		}

		partition_orbit_first[pair_orbit_idx] = total_nb_orbits_on_partitions;
		partition_orbit_len[pair_orbit_idx] = nb;

		total_nb_orbits_on_partitions += nb;
	}
	if (f_v) {
		cout << "arc_orbits_on_pairs::init "
				"computing orbits on partition done" << endl;
	}


	if (f_v) {
		cout << "arc_orbits_on_pairs::init done" << endl;
	}
}

void arc_orbits_on_pairs::print()
{
	int pair_orbit_idx, nb;

	cout << "pair_orbit_idx : nb_orbits_on_partitions" << endl;
	for (pair_orbit_idx = 0;
			pair_orbit_idx < nb_orbits_on_pairs;
			pair_orbit_idx++) {

		nb = Table_orbits_on_partition[pair_orbit_idx].nb_orbits_on_partition;

		cout << pair_orbit_idx << " & " << nb << endl;
	}
}

void arc_orbits_on_pairs::recognize(long int *pair, int *transporter,
		int &orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_orbits_on_pairs::recognize" << endl;
	}

	Orbits_on_pairs->get_Orbit_tracer()->identify(pair, 2,
		transporter,
		orbit_idx,
		0 /*verbose_level */);


	if (f_v) {
		cout << "arc_orbits_on_pairs::recognize done" << endl;
	}
}



}}}}


