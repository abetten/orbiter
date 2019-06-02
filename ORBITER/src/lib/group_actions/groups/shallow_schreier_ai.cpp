/*
 * shallow_schreier_ai.cpp
 *
 *  Created on: Jun 2, 2019
 *      Author: sajeeb
 */

#include "shallow_schreier_ai.h"

shallow_schreier_ai::shallow_schreier_ai(schreier& sch, int vl) {

	int f_v = (vl >= 1);
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
	gens2->init(sch.A);
	for (int el = 0; el < sch.gens.len; el++)
		gens2->append(sch.gens.ith(el));



	// Create a new schreier forest with the same generators
	schreier* S = NEW_OBJECT(schreier);
	S->init(sch.A);
	S->init_generators_recycle_images(*gens2, sch.images);
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
		new_gens->init(sch.A);
		for (int el = 0; el < gens2->len; el++) {
			(el != random_generator_idx) ?  new_gens->append(gens2->ith(el)) :
											new_gens->append(Elt1);
		}
		FREE_OBJECT(gens2);
		gens2 = new_gens;



		// Create a new schreier tree with the new generating set
		S = NEW_OBJECT(schreier);
		S->init(sch.A);

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
			break;
		}


		FREE_OBJECT(previous_schreier);

	}


	sch.init(sch.A);
	sch.init_generators_recycle_images(S->gens, S->images);
	sch.compute_all_point_orbits(vl);



	FREE_OBJECT(S);
	FREE_OBJECT(gens2);
	FREE_int(Elt1);


	if (f_v) {
		cout << "schreier::shallow_tree_generators_ai done" << endl;
	}
}

shallow_schreier_ai::~shallow_schreier_ai() {

}
