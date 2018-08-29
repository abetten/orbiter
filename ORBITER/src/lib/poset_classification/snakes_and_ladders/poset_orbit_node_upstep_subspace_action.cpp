// poset_orbit_node_upstep_subspace_action.C
//
// Anton Betten
// Jan 25, 2010

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

void poset_orbit_node::orbit_representative_and_coset_rep_inv_subspace_action(
	poset_classification *gen,
	INT lvl, INT pt_to_trace,
	INT &pt0, INT *&cosetrep, INT verbose_level)
// called by oracle::trace_next_point
{
	INT f_v = (verbose_level >= 1);
	action_on_factor_space AF;
	action A_factor_space;
	INT projected_pt, projected_pt0;
	INT f_check_image = FALSE;

	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_"
				"rep_inv_subspace_action" << endl;
		cout << "node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "setting up factor space action" << endl;
		}

	if (gen->f_early_test_func) {
		gen->root[node].setup_factor_space_action_light(gen,
				AF, lvl, verbose_level - 4);

			// poset_orbit_node_downstep_subspace_action.C


#if 0
		gen->root[node].setup_factor_space_action_with_early_test(
			gen,
			AF, A_factor_space, lvl, 
			verbose_level - 2);
#endif
		}
	else {
		gen->root[node].setup_factor_space_action(gen,
			AF, A_factor_space, lvl,
			FALSE /*f_compute_tables*/,
			verbose_level - 4);
		}

	//projected_pt = AF.project(pt_to_trace, verbose_level - 2);
	projected_pt = AF.project_onto_Gauss_reduced_vector(
			pt_to_trace, verbose_level - 4);


	//G.point_stabilizer_with_action(&A_factor_space,
	// H, projected_pt, 0);

	if (f_v) {
		cout << "poset_orbit_node::representative_and_coset_rep_inv_subspace_action "
				"lvl=" << lvl << " pt_to_trace=" << pt_to_trace
				<< " projects onto " << projected_pt << endl;
		}
	cosetrep = gen->Elt1;
	if (nb_strong_generators == 0) {
		gen->A->element_one(gen->Elt1, 0);
		projected_pt0 = projected_pt;
		

		//pt0 = AF.preimage(projected_pt0, verbose_level - 2);
		pt0 = AF.lexleast_element_in_coset(
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
	if (sv) {
		INT f_trivial_group;
		
		if (nb_strong_generators) 
			f_trivial_group = FALSE;
		else 
			f_trivial_group = TRUE;
		//cout << "Node " << node << " poset_orbit_node::orbit_representative_and_"
		//"coset_rep_inv_subspace_action calling schreier_vector_"
		//"coset_rep_inv" << endl;
		schreier_vector_coset_rep_inv(
			gen->A2 /*&A_factor_space*/,
			sv, 
			hdl_strong_generators, 
			projected_pt, 
			projected_pt0, 
			gen->Elt1, gen->Elt2, gen->Elt3, gen->Elt4, 
			f_trivial_group, 
			TRUE /* f_compact */,
			f_check_image, 
			verbose_level - 4);
		// gen->Elt1 contains the element that maps pt_to_trace to pt0
		//cout << "Node " << node << " poset_orbit_node::orbit_representative_and_"
		//"coset_rep_inv_subspace_action schreier_vector_coset_"
		//"rep_inv done" << endl;
		
		//pt0 = AF.preimage(projected_pt0, verbose_level - 2);
		pt0 = AF.lexleast_element_in_coset(
				projected_pt0, verbose_level - 4);

		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action with schreier vector: "
					"pt_to_trace=" << pt_to_trace
				<< " projected_pt0=" << projected_pt0
				<< " preimage=" << pt0 << endl;
			}
		INT a;
		a = gen->A2->element_image_of(pt_to_trace, gen->Elt1, 0);
		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
					"subspace_action " << pt_to_trace << "->" << a << endl;
			}
		return;
		}
	else {
		cout << "Node " << node << " poset_orbit_node::orbit_representative_and_"
				"coset_rep_inv_subspace_action sv not "
				"available (fatal)" << endl;
		cout << "node=" << node << " prev=" << prev
				<< " pt=" << pt << endl;
		cout << "pt_to_trace=" << pt_to_trace << endl;
		cout << "verbose_level=" << verbose_level << endl;
		exit(1);
		}
#if 0
	projected_pt0 =
			A_factor_space.least_image_of_point_generators_by_handle(
		nb_strong_generators, 
		hdl_strong_generators, 
		projected_pt, 
		gen->Elt1, 
		verbose_level - 1);
	pt0 = AF.preimage(projected_pt0, verbose_level - 2);
	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
				"pt_to_trace=" << pt_to_trace
			<< " projected_pt0=" << projected_pt0
			<< " preimage=" << pt0 << endl;
		}
	//return;
	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv_"
				"subspace_action done" << endl;
		}
#endif
}


