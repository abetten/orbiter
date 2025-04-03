/*
 * shallow_schreier_ai.cpp
 *
 *  Created on: Jun 2, 2019
 *      Author: sajeeb
 */

#include "shallow_schreier_ai.h"

#if 0
void shallow_schreier_ai::generate_shallow_tree(
		groups::schreier& sch, int verbose_level) {

	int f_v = (verbose_level >= 1);
	int *Elt1;

	if (f_v) {
		cout << "schreier::shallow_tree_generators_ai" << endl;
	}


	if (sch.Generators_and_images->A->degree == 0) {
		if (f_v) {
			cout << "schreier::shallow_tree_generators_ai degree is zero, returning" << endl;
		}
		return;
	}
	Elt1 = NEW_int(sch.Generators_and_images->A->elt_size_in_int);


	// Make a copy of the current generators
	data_structures_groups::vector_ge* gens2 = NEW_OBJECT(data_structures_groups::vector_ge);
	gens2->init(sch.Generators_and_images->A, verbose_level - 2);
	for (int el = 0; el < sch.Generators_and_images->gens.len; el++)
		gens2->append(sch.Generators_and_images->gens.ith(el), verbose_level - 2);



	int print_interval = 10000;

	// Create a new schreier forest with the same generators
	groups::schreier* S = NEW_OBJECT(groups::schreier);
	S->init(sch.Generators_and_images->A, verbose_level - 2);
	S->Generators_and_images->init_generators_recycle_images(
			*gens2, sch.Generators_and_images->images, verbose_level - 2);
	S->compute_all_point_orbits(print_interval, 0);




	for (int step = 0, ns = gens2->len, gen_idx = 0; step < ns; ++step, ++gen_idx) {

		groups::schreier* previous_schreier = S;
		if (S->Forest->nb_orbits == 0) {
			printf("S->nb_orbits=%d\n", S->Forest->nb_orbits);
			break;
		}

		//auto total_points_in_old_forest = S->get_num_points();

		//
		other::orbiter_kernel_system::os_interface Os;
		int random_orbit_idx = Os.random_integer(S->Forest->nb_orbits);
		int random_orbit_idx_cpy = random_orbit_idx;
		int random_point_idx = Os.random_integer(S->Forest->orbit_len[random_orbit_idx]);
		int random_point = S->Forest->orbit[S->Forest->orbit_first[random_orbit_idx]
				+ random_point_idx];
		int random_generator_idx = gen_idx; //random_integer(gens2->len);


		sch.Generators_and_images->transporter_from_orbit_rep_to_point(
				random_point,
				random_orbit_idx_cpy, Elt1, 0 /*verbose_level*/);




		// Create a new generating set with the new element
		data_structures_groups::vector_ge* new_gens = NEW_OBJECT(data_structures_groups::vector_ge);
		new_gens->init(sch.Generators_and_images->A, verbose_level - 2);
		for (int el = 0; el < gens2->len; el++) {
			(el != random_generator_idx) ?  new_gens->append(gens2->ith(el), verbose_level - 2) :
											new_gens->append(Elt1, verbose_level - 2);
		}
		FREE_OBJECT(gens2);
		gens2 = new_gens;



		// Create a new schreier tree with the new generating set
		S = NEW_OBJECT(groups::schreier);
		S->init(sch.Generators_and_images->A, verbose_level - 2);

		S->Generators_and_images->init_generators_recycle_images(
				previous_schreier->Generators_and_images->nb_images,
				gens2->ith(0),
				previous_schreier->Generators_and_images->images,
				random_generator_idx);


		int print_interval = 10000;

		S->compute_all_point_orbits(print_interval, 0 /*verbose_level*/);



		// if the number of points in the new forest is not
		// equal to the number of nodes in the old forest,
		// then an invalid move has been made.

		if (/*S->get_num_points() != total_points_in_old_forest
				|| */ S->Forest->nb_orbits != sch.Forest->nb_orbits) {
			FREE_OBJECT(S);
			if (true) {
				cout << "schreier::shallow_tree_generators_ai reverting to previous schreier" << endl;
			}
			S = previous_schreier;
			this->nb_revert_backs += 1;
			break;
		}


		FREE_OBJECT(previous_schreier);

	}



	sch.init(sch.Generators_and_images->A, verbose_level - 2);
	sch.Generators_and_images->init_generators_recycle_images(
			S->Generators_and_images->gens,
			S->Generators_and_images->images,
			verbose_level - 2);
	sch.compute_all_point_orbits(print_interval, verbose_level);



	FREE_OBJECT(S);
	FREE_OBJECT(gens2);
	FREE_int(Elt1);


	if (f_v) {
		cout << "schreier::shallow_tree_generators_ai done" << endl;
	}
}


void shallow_schreier_ai::get_degree_sequence(
		groups::schreier& sch, int vl) {

	nb_nodes = sch.Generators_and_images->degree;
	if (deg_seq) delete [] deg_seq;
	deg_seq = new int [nb_nodes] ();

	for (int i=0; i<nb_nodes; ++i) {
		int pt = sch.Forest->orbit[i];
		int parent = sch.Forest->prev[pt];
		bool root_node = (parent == -1);
		if (!root_node) {
			deg_seq[parent] += 1;
		}
	}

	s = &sch;

}


void shallow_schreier_ai::print_degree_sequence () {
	cout << __FILE__ << ":" << __LINE__
			<< ":shallow_schreier_ai::print_degree_sequence" << endl;

	for (int i=0; i<this->nb_nodes; ++i) {
		cout << "Node " << s->Forest->orbit[i] << " -> " << this->deg_seq[i] << endl;
	}

	cout << __FILE__ << ":" << __LINE__
			<< ":shallow_schreier_ai::print_degree_sequence Done." << endl;
}

shallow_schreier_ai::~shallow_schreier_ai() {
	if (deg_seq) {
		delete [] deg_seq;
	}
}
#endif


