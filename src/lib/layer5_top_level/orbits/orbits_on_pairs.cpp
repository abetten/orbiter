/*
 * orbits_on_pairs.cpp
 *
 *  Created on: May 18, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {



orbits_on_pairs::orbits_on_pairs()
{
	Record_birth();

	Strong_gens = NULL;
	A = NULL;
	A0 = NULL;

	V = 0;

	Control = NULL;
	Poset = NULL;
	Poset_classification = NULL;

	pair_orbit = NULL;
	nb_orbits = 0;
	transporter = NULL;
	tmp_Elt = NULL;
	orbit_length = NULL;

}

orbits_on_pairs::~orbits_on_pairs()
{
	Record_death();
	if (pair_orbit) {
		FREE_int(pair_orbit);
	}
	if (transporter) {
		FREE_int(transporter);
	}
	if (tmp_Elt) {
		FREE_int(tmp_Elt);
	}
	if (orbit_length) {
		FREE_int(orbit_length);
	}

}

void orbits_on_pairs::init(
		std::string &control_label,
		groups::strong_generators *Strong_gens,
		actions::action *A,
		actions::action *A0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_pairs::init" << endl;
	}

	orbits_on_pairs::Strong_gens = Strong_gens;
	orbits_on_pairs::A = A;
	orbits_on_pairs::A0 = A0;

	V = A->degree;

	Control = Get_poset_classification_control(
			control_label);


	if (f_v) {
		cout << "orbits_on_pairs::init before compute_orbits_on_pairs" << endl;
	}
	compute_orbits_on_pairs(verbose_level);
	if (f_v) {
		cout << "orbits_on_pairs::init after compute_orbits_on_pairs" << endl;
	}

	if (f_v) {
		cout << "orbits_on_pairs::init done" << endl;
	}
}


void orbits_on_pairs::compute_orbits_on_pairs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs" << endl;
	}
	Poset_classification = NEW_OBJECT(poset_classification::poset_classification);


	Control->f_depth = true;
	Control->depth = 2;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(
			A0, A, Strong_gens,
			verbose_level);


	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs "
				"before Poset_classification->init" << endl;
	}
	Poset_classification->initialize_and_allocate_root_node(
			Control, Poset,
			2 /* sz */, verbose_level);
	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs "
				"after Poset_classification->init" << endl;
	}



	int f_use_invariant_subset_if_available;
	int f_debug;

	f_use_invariant_subset_if_available = true;
	f_debug = false;


	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs "
				"before Pairs->poset_classification_main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
	}


	//Pairs->f_allowed_to_show_group_elements = true;

	Control->f_depth = true;
	Control->depth = 2;

	//Pairs->depth = 2;
	Poset_classification->poset_classification_main(
			t0,
		2 /* schreier_depth */,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs "
				"after Poset_classification->poset_classification_main" << endl;
	}


	nb_orbits = Poset_classification->nb_orbits_at_level(2);

	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs "
				"nb_orbits = "
				<< nb_orbits << endl;
	}

	transporter = NEW_int(A0->elt_size_in_int);
	tmp_Elt = NEW_int(A0->elt_size_in_int);

	orbit_length = NEW_int(nb_orbits);



	for (i = 0; i < nb_orbits; i++) {

		orbit_length[i] = Poset_classification->orbit_length_as_int(
				i /* orbit_at_level*/, 2 /* level*/);

	}
	cout << "i : orbit_length[i] " << endl;
	for (i = 0; i < nb_orbits; i++) {
		cout << i << " : " << orbit_length[i]
			<< endl;
	}

	compute_pair_orbit_table(verbose_level);
	//write_pair_orbit_file(verbose_level);
	if (f_v) {
		cout << "orbits_on_pairs::compute_orbits_on_pairs done" << endl;
	}
}

void orbits_on_pairs::compute_pair_orbit_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;


	if (f_v) {
		cout << "orbits_on_pairs::compute_pair_orbit_table" << endl;
	}
	pair_orbit = NEW_int(V * V);
	Int_vec_zero(pair_orbit, V * V);
	for (i = 0; i < V; i++) {
		for (j = i + 1; j < V; j++) {
			k = find_pair_orbit_by_tracing(i, j, 0 /*verbose_level - 2*/);
			pair_orbit[i * V + j] = k;
			pair_orbit[j * V + i] = k;
		}
		if ((i % 100) == 0) {
			cout << "i=" << i << endl;
		}
	}
	if (f_v) {
		cout << "orbits_on_pairs::compute_pair_orbit_table done" << endl;
	}
}

int orbits_on_pairs::find_pair_orbit_by_tracing(
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_no;
	long int set[2];
	long int canonical_set[2];

	if (f_v) {
		cout << "orbits_on_pairs::find_pair_orbit_by_tracing" << endl;
	}
	if (i == j) {
		cout << "orbits_on_pairs::find_pair_orbit_by_tracing "
				"i = j = " << j << endl;
		exit(1);
	}
	set[0] = i;
	set[1] = j;
	orbit_no = Poset_classification->trace_set(
			set, 2, 2,
		canonical_set, transporter,
		verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_pairs::find_pair_orbit_by_tracing "
				"done" << endl;
	}
	return orbit_no;
}


int orbits_on_pairs::find_pair_orbit(
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_no;

	if (f_v) {
		cout << "orbits_on_pairs::find_pair_orbit" << endl;
	}
	if (i == j) {
		cout << "orbits_on_pairs::find_pair_orbit "
				"i = j = " << j << endl;
		exit(1);
	}
	orbit_no = pair_orbit[i * V + j];
	if (f_v) {
		cout << "orbits_on_pairs::find_pair_orbit done" << endl;
	}
	return orbit_no;
}




}}}

