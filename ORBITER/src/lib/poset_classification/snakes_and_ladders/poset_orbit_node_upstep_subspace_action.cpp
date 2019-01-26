// poset_orbit_node_upstep_subspace_action.C
//
// Anton Betten
// Jan 25, 2010

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "poset_classification/poset_classification.h"

namespace orbiter {

void poset_orbit_node::orbit_representative_and_coset_rep_inv_subspace_action(
	poset_classification *gen,
	int lvl, int pt_to_trace,
	int &pt0, int *&cosetrep, int verbose_level)
// called by poset_orbit_node::trace_next_point
// creates action_on_factor_space AF;
// and action A_factor_space;
// and disposes them at the end.
{
	int f_v = (verbose_level >= 1);
	int projected_pt, projected_pt0;

	action_on_factor_space *AF;
	action *A_factor_space;

	//int f_check_image = FALSE;


	A_factor_space = A_on_upset;
	AF = A_factor_space->G.AF;


	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_"
				"rep_inv_subspace_action" << endl;
		cout << "node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "setting up factor space action" << endl;
		}

#if 0

	//if (TRUE /*gen->f_early_test_func*/) {
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_"
					"rep_inv_subspace_action "
					"before setup_factor_space_action_light" << endl;
		}
		gen->root[node].setup_factor_space_action_light(gen,
				AF, lvl, verbose_level - 4);
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_"
					"rep_inv_subspace_action "
					"after setup_factor_space_action_light" << endl;
		}

			// poset_orbit_node_downstep_subspace_action.C


#if 1
		gen->root[node].setup_factor_space_action_with_early_test(
			gen,
			AF, A_factor_space, lvl, 
			verbose_level - 2);
#endif
	//}

#if 0
	else {
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_"
					"rep_inv_subspace_action "
					"before setup_factor_space_action" << endl;
		}
		gen->root[node].setup_factor_space_action(gen,
			AF, A_factor_space, lvl,
			FALSE /*f_compute_tables*/,
			verbose_level - 4);
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_"
					"rep_inv_subspace_action "
					"after setup_factor_space_action" << endl;
		}
	}
#endif
#endif


	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_"
				"rep_inv_subspace_action "
				"before project_onto_Gauss_reduced_vector" << endl;
	}
	//projected_pt = AF.project(pt_to_trace, verbose_level - 2);
	projected_pt = AF->project_onto_Gauss_reduced_vector(
			pt_to_trace, verbose_level - 4);


	if (f_v) {
		cout << "poset_orbit_node::representative_and_coset_rep_inv_subspace_action "
				"lvl=" << lvl << " pt_to_trace=" << pt_to_trace
				<< " projects onto " << projected_pt << endl;
		}
	if (nb_strong_generators == 0) {

		cosetrep = gen->Elt1;
		gen->Poset->A->element_one(gen->Elt1, 0);
		projected_pt0 = projected_pt;
		

		//pt0 = AF.preimage(projected_pt0, verbose_level - 2);
		pt0 = AF->lexleast_element_in_coset(
				projected_pt0,
				verbose_level - 4);

		if (f_v) {
			cout << "poset_orbit_node::representative_and_coset_rep_inv_subspace_"
					"action lvl=" << lvl << " stabilizer is trivial, "
					"projected_pt0=" << projected_pt0
					<< " pt0=" << pt0 << endl;
			}
		return;
		}
	if (Schreier_vector) {

		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action before "
					"Schreier_vector_handler->coset_rep_inv" << endl;
		}
		gen->Schreier_vector_handler->coset_rep_inv(
				Schreier_vector,
				projected_pt,
				projected_pt0,
				verbose_level - 4);
		cosetrep = gen->Schreier_vector_handler->cosetrep;
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action after "
					"Schreier_vector_handler->coset_rep_inv" << endl;
		}

		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action before "
					"AF.lexleast_element_in_coset" << endl;
		}
		pt0 = AF->lexleast_element_in_coset(
				projected_pt0, verbose_level - 4);

		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action after "
					"AF.lexleast_element_in_coset" << endl;
		}
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action with schreier vector: "
					"pt_to_trace=" << pt_to_trace
				<< " projected_pt0=" << projected_pt0
				<< " preimage=" << pt0 << endl;
			}

#if 0
		int a;
		a = gen->Poset->A2->element_image_of(pt_to_trace, gen->Elt1, 0);
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action " << pt_to_trace << "->" << a << endl;
			}
#endif
		}
	else {
		cout << "Node " << node
				<< " poset_orbit_node::orbit_representative_and_"
				"coset_rep_inv_subspace_action sv not "
				"available (fatal)" << endl;
		cout << "node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
		cout << "pt_to_trace=" << pt_to_trace << endl;
		cout << "verbose_level=" << verbose_level << endl;
		exit(1);
		}
	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_"
				"rep_inv_subspace_action" << endl;
		cout << "node=" << node << " prev=" << prev
				<< " pt=" << pt << "done" << endl;
		}
}

}


