/*
 * shallow_schreier_ai.cpp
 *
 *  Created on: Jun 2, 2019
 *      Author: sajeeb
 */

#include "shallow_schreier_ai.h"

void shallow_schreier_ai::generate_shallow_tree(schreier& sch, int verbose_level) {

	int f_v = (verbose_level >= 1);
	int fst, len, root, cnt, l;
	int i, a, f, o;
	int *Elt1;

	if (f_v) {

		cout << "schreier::shallow_tree_generators_ai" << endl;

	}


	if (sch.A->degree == 0) {
		if (f_v) {
			cout << "schreier::shallow_tree_generators_ai degree is zero, returning" << endl;
		}
		return;
	}
	Elt1 = NEW_int(sch.A->elt_size_in_int);


	// Make a copy of the current generators
	vector_ge* gens2 = NEW_OBJECT(vector_ge);
	gens2->init(sch.A, verbose_level - 2);
	for (int el = 0; el < sch.gens.len; el++)
		gens2->append(sch.gens.ith(el), verbose_level - 2);



	// Create a new schreier forest with the same generators
	schreier* S = NEW_OBJECT(schreier);
	S->init(sch.A, verbose_level - 2);
	S->init_generators_recycle_images(*gens2, sch.images, verbose_level - 2);
	S->compute_all_point_orbits(0);




	for (int step = 0, ns = gens2->len, gen_idx = 0; step < ns; ++step, ++gen_idx) {

		schreier* previous_schreier = S;
		if (S->nb_orbits == 0) {
			printf("S->nb_orbits=%d\n", S->nb_orbits);
			break;
		}

		//auto total_points_in_old_forest = S->get_num_points();

		//
		int random_orbit_idx = random_integer(S->nb_orbits);
		int random_orbit_idx_cpy = random_orbit_idx;
		int random_point_idx = random_integer(S->orbit_len[random_orbit_idx]);
		int random_point = S->orbit[S->orbit_first[random_orbit_idx]
				+ random_point_idx];
		int random_generator_idx = gen_idx; //random_integer(gens2->len);


		sch.transporter_from_orbit_rep_to_point(random_point,
				random_orbit_idx_cpy, Elt1, 0 /*verbose_level*/);




		// Create a new generating set with the new element
		vector_ge* new_gens = NEW_OBJECT(vector_ge);
		new_gens->init(sch.A, verbose_level - 2);
		for (int el = 0; el < gens2->len; el++) {
			(el != random_generator_idx) ?  new_gens->append(gens2->ith(el), verbose_level - 2) :
											new_gens->append(Elt1, verbose_level - 2);
		}
		FREE_OBJECT(gens2);
		gens2 = new_gens;



		// Create a new schreier tree with the new generating set
		S = NEW_OBJECT(schreier);
		S->init(sch.A, verbose_level - 2);

		S->init_generators_recycle_images(
				previous_schreier->nb_images,
				gens2->ith(0),
				previous_schreier->images,
				random_generator_idx);


		S->compute_all_point_orbits(0 /*verbose_level*/);



		// if the number of points in the new forest is not
		// equal to the number of nodes in the old forest,
		// then an invalid move has been made.

		if (/*S->get_num_points() != total_points_in_old_forest
				|| */ S->nb_orbits != sch.nb_orbits) {
			FREE_OBJECT(S);
			if (TRUE) {
				cout << "schreier::shallow_tree_generators_ai reverting to previous schreier" << endl;
			}
			S = previous_schreier;
			this->nb_revert_backs += 1;
			break;
		}


		FREE_OBJECT(previous_schreier);

	}


	sch.init(sch.A, verbose_level - 2);
	sch.init_generators_recycle_images(S->gens, S->images, verbose_level - 2);
	sch.compute_all_point_orbits(verbose_level);



	FREE_OBJECT(S);
	FREE_OBJECT(gens2);
	FREE_int(Elt1);


	if (f_v) {
		cout << "schreier::shallow_tree_generators_ai done" << endl;
	}
}


void shallow_schreier_ai::get_degree_sequence (schreier& sch, int vl) {

	nb_nodes = sch.degree;
	if (deg_seq) delete [] deg_seq;
	deg_seq = new int [nb_nodes] ();

	for (int i=0; i<nb_nodes; ++i) {
		int pt = sch.orbit[i];
		int parent = sch.prev[pt];
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
		cout << "Node " << s->orbit[i] << " -> " << this->deg_seq[i] << endl;
	}

	cout << __FILE__ << ":" << __LINE__
			<< ":shallow_schreier_ai::print_degree_sequence Done." << endl;
}

shallow_schreier_ai::~shallow_schreier_ai() {
	if (deg_seq) {
		delete [] deg_seq;
	}
}
