/*
 * group_theoretic_activity.cpp
 *
 *  Created on: May 5, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


group_theoretic_activity::group_theoretic_activity()
{
	Descr = NULL;

	AG = NULL;

	A1 = NULL;
	A2 = NULL;

}

group_theoretic_activity::~group_theoretic_activity()
{

}


void group_theoretic_activity::init_group(group_theoretic_activity_description *Descr,
		any_group *AG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::init_group" << endl;
	}

	group_theoretic_activity::Descr = Descr;
	group_theoretic_activity::AG = AG;


	A1 = AG->A_base;
	A2 = AG->A;

	if (f_v) {
		cout << "group_theoretic_activity::init_group A1 = " << A1->label << endl;
		cout << "group_theoretic_activity::init_group A2 = " << A2->label << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::init_group done" << endl;
	}
}



void group_theoretic_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::perform_activity" << endl;
	}


	if (Descr->f_multiply) {
		multiply(verbose_level);
	}

	if (Descr->f_inverse) {
		inverse(verbose_level);
	}

	if (Descr->f_raise_to_the_power) {
		raise_to_the_power(verbose_level);
	}

	if (Descr->f_export_orbiter) {
		AG->do_export_orbiter(A2, verbose_level);
	}

	if (Descr->f_export_gap) {
		AG->do_export_gap(verbose_level);
	}

	if (Descr->f_export_magma) {
		AG->do_export_magma(verbose_level);
	}

	if (Descr->f_canonical_image) {
		AG->do_canonical_image_GAP(Descr->canonical_image_input_set, verbose_level);
	}

	if (Descr->f_classes_based_on_normal_form) {
		AG->classes_based_on_normal_form(verbose_level);
	}

	if (Descr->f_classes) {
		AG->classes(verbose_level);
	}

	if (Descr->f_group_table) {
		AG->create_group_table(verbose_level);
	}

	if (Descr->f_normalizer) {
		AG->normalizer(verbose_level);
	}

	if (Descr->f_centralizer_of_element) {
		AG->centralizer(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}

	if (Descr->f_conjugacy_class_of_element) {
		AG->do_conjugacy_class_of_element(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}
	if (Descr->f_orbits_on_group_elements_under_conjugation) {
		AG->do_orbits_on_group_elements_under_conjugation(
				Descr->orbits_on_group_elements_under_conjugation_fname,
				Descr->orbits_on_group_elements_under_conjugation_transporter_fname,
				verbose_level);
	}



	if (Descr->f_normalizer_of_cyclic_subgroup) {
		AG->normalizer_of_cyclic_subgroup(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}

	if (Descr->f_find_subgroup) {
		AG->do_find_subgroups(Descr->find_subgroup_order, verbose_level);
	}





	if (Descr->f_report) {

		if (!Orbiter->f_draw_options) {
			cout << "for a report of the group, please use -draw_options" << endl;
			exit(1);
		}

		AG->create_latex_report(
				Orbiter->draw_options,
				Descr->f_sylow, Descr->f_group_table, Descr->f_classes,
				verbose_level);

	}

	if (Descr->f_print_elements) {
		AG->print_elements(verbose_level);
	}

	if (Descr->f_print_elements_tex) {

		AG->print_elements_tex(
				verbose_level);
	}

	if (Descr->f_order_of_products) {

		AG->order_of_products_of_elements(
				Descr->order_of_products_elements,
				verbose_level);
	}

	if (Descr->f_save_elements_csv) {
		AG->save_elements_csv(Descr->save_elements_csv_fname, verbose_level);
	}

	if (Descr->f_multiply_elements_csv_column_major_ordering) {
		AG->multiply_elements_csv(
				Descr->multiply_elements_csv_column_major_ordering_fname1,
				Descr->multiply_elements_csv_column_major_ordering_fname2,
				Descr->multiply_elements_csv_column_major_ordering_fname3,
				TRUE, verbose_level);
	}
	if (Descr->f_multiply_elements_csv_row_major_ordering) {
		AG->multiply_elements_csv(
				Descr->multiply_elements_csv_row_major_ordering_fname1,
				Descr->multiply_elements_csv_row_major_ordering_fname2,
				Descr->multiply_elements_csv_row_major_ordering_fname3,
				FALSE, verbose_level);
	}
	if (Descr->f_apply_elements_csv_to_set) {


		AG->apply_elements_to_set_csv(
				Descr->apply_elements_csv_to_set_fname1,
				Descr->apply_elements_csv_to_set_fname2,
				Descr->apply_elements_csv_to_set_set,
				verbose_level);
	}


	if (Descr->f_find_singer_cycle) {
		AG->find_singer_cycle(verbose_level);
	}
	if (Descr->f_search_element_of_order) {
		AG->search_element_of_order(Descr->search_element_order, verbose_level);
	}

	if (Descr->f_find_standard_generators) {
		AG->find_standard_generators(Descr->find_standard_generators_order_a,
				Descr->find_standard_generators_order_b,
				Descr->find_standard_generators_order_ab,
				verbose_level);

	}

	if (Descr->f_element_rank) {
		AG->element_rank(Descr->element_rank_data, verbose_level);
	}
	if (Descr->f_element_unrank) {
		AG->element_unrank(Descr->element_unrank_data, verbose_level);
	}
	if (Descr->f_conjugacy_class_of) {
		AG->conjugacy_class_of(Descr->conjugacy_class_of_data, verbose_level);
	}
	if (Descr->f_isomorphism_Klein_quadric) {
		AG->isomorphism_Klein_quadric(Descr->isomorphism_Klein_quadric_fname, verbose_level);
	}



	if (Descr->f_orbits_on_set_system_from_file) {
		AG->orbits_on_set_system_from_file(
				Descr->orbits_on_set_system_from_file_fname,
				Descr->orbits_on_set_system_number_of_columns,
				Descr->orbits_on_set_system_first_column,
				verbose_level);
	}

	if (Descr->f_orbit_of_set_from_file) {
		AG->orbits_on_set_from_file(Descr->orbit_of_set_from_file_fname, verbose_level);
	}

	if (Descr->f_orbit_of) {
		AG->orbit_of(Descr->orbit_of_point_idx, verbose_level);
	}
	else if (Descr->f_orbits_on_subsets) {

		poset_classification::poset_classification_control *Control;

		if (Descr->f_poset_classification_control) {
			Control = Descr->Control;
		}
		else {
			cout << "please use option -poset_classification_control" << endl;
			exit(1);
			//Control = NEW_OBJECT(poset_classification_control);
		}

		poset_classification::poset_classification *PC;

		AG->orbits_on_subsets(Control, PC,
				Descr->orbits_on_subsets_size, verbose_level);

		FREE_OBJECT(PC);
	}

	// generic orbits on points or subspaces:

	else if (Descr->f_orbits_on_points) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity f_orbits_on_points" << endl;
		}


		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->orbits_on_points" << endl;
		}

		AG->orbits_on_points(Orb, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->orbits_on_points" << endl;
		}


		if (Descr->f_stabilizer) {

			int orbit_idx = 0;

			if (f_v) {
				cout << "group_theoretic_activity::perform_activity before Orb->stabilizer_of" << endl;
			}
			Orb->stabilizer_of(orbit_idx, verbose_level);
			if (f_v) {
				cout << "group_theoretic_activity::perform_activity after Orb->stabilizer_of" << endl;
			}
		}

		if (Descr->f_stabilizer_of_orbit_rep) {

			int orbit_idx = Descr->stabilizer_of_orbit_rep_orbit_idx;

			if (f_v) {
				cout << "group_theoretic_activity::perform_activity before Orb->stabilizer_of" << endl;
			}
			Orb->stabilizer_of(orbit_idx, verbose_level);
			if (f_v) {
				cout << "group_theoretic_activity::perform_activity after Orb->stabilizer_of" << endl;
			}
		}


		if (Descr->f_report) {

			Orb->create_latex_report(verbose_level);

		}

		if (Descr->f_export_trees) {
			string fname_tree_mask;
			int orbit_idx;

			fname_tree_mask.assign(A2->label);
			fname_tree_mask.append("_%d.layered_graph");

			for (orbit_idx = 0; orbit_idx < Orb->Sch->nb_orbits; orbit_idx++) {
				cout << "orbit " << orbit_idx << " / " <<  Orb->Sch->nb_orbits
						<< " before Sch->export_tree_as_layered_graph" << endl;
				Orb->Sch->export_tree_as_layered_graph(0 /* orbit_no */,
						fname_tree_mask,
						verbose_level - 1);
			}
		}

		FREE_OBJECT(Orb);

	}
	else if (Descr->f_orbits_on_subspaces) {

		if (!Descr->f_poset_classification_control) {
			cout << "please use option -poset_classification_control descr -end" << endl;
			exit(1);
		}

		AG->do_orbits_on_subspaces(this, Descr->Control, Descr->orbits_on_subspaces_depth, verbose_level);
	}
	else if (Descr->f_reverse_isomorphism_exterior_square) {
		AG->do_reverse_isomorphism_exterior_square(verbose_level);
	}





	// classification of:


	// linear codes:


	if (Descr->f_linear_codes) {

		if (!Descr->f_poset_classification_control) {
			cout << "Please use option -poset_classification_control <descr> -end" << endl;
			exit(1);
		}

		AG->do_linear_codes(
				Descr->Control,
				Descr->linear_codes_minimum_distance,
				Descr->linear_codes_target_size, verbose_level);
	}












	// tensors:

	else if (Descr->f_tensor_classify) {
		if (!Descr->f_poset_classification_control) {
			cout << "please use option -poset_classification_control descr -end" << endl;
			exit(1);
		}

		AG->do_tensor_classify(Descr->Control, Descr->tensor_classify_depth, verbose_level);

	}
	else if (Descr->f_tensor_permutations) {
		AG->do_tensor_permutations(verbose_level);
	}


	else if (Descr->f_classify_ovoids) {
		AG->do_classify_ovoids(Descr->Control, Descr->Ovoid_classify_description, verbose_level);
	}



	else if (Descr->f_orbits_on_polynomials) {

		algebra_global_with_action Algebra;


		if (Descr->f_orbits_on_polynomials_draw_tree &&
				!Orbiter->f_draw_options) {
			cout << "please use -draw_options ... -end" << endl;
			exit(1);
		}

		if (!AG->f_linear_group) {
			cout << "Descr->f_orbits_on_polynomials group must be linear" << endl;
			exit(1);
		}

		Algebra.do_orbits_on_polynomials(
				AG->LG,
				Descr->orbits_on_polynomials_degree,
				Descr->f_recognize_orbits_on_polynomials,
				Descr->recognize_orbits_on_polynomials_text,
				Descr->f_orbits_on_polynomials_draw_tree,
				Descr->orbits_on_polynomials_draw_tree_idx,
				Orbiter->draw_options,
				verbose_level);
	}

	else if (Descr->f_representation_on_polynomials) {

		algebra_global_with_action Algebra;

		if (!AG->f_linear_group) {
			cout << "Descr->f_representation_on_polynomials group must be linear" << endl;
			exit(1);
		}

		Algebra.representation_on_polynomials(
				AG->LG,
				Descr->representation_on_polynomials_degree,
				verbose_level);
	}

	else if (Descr->f_Andre_Bruck_Bose_construction) {
		do_Andre_Bruck_Bose_construction(
					Descr->Andre_Bruck_Bose_construction_spread_no,
					FALSE /* f_Fano */, FALSE /* f_arcs */, FALSE /* f_depth */, 0 /* depth */,
					Descr->Andre_Bruck_Bose_construction_label,
					verbose_level);
	}


	if (f_v) {
		cout << "group_theoretic_activity::perform_activity done" << endl;
	}
}




void group_theoretic_activity::multiply(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::multiply" << endl;
	}

	A1->multiply_based_on_text(Descr->multiply_a,
			Descr->multiply_b, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::multiply done" << endl;
	}
}

void group_theoretic_activity::inverse(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::inverse" << endl;
	}

	A1->inverse_based_on_text(Descr->inverse_a, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::inverse done" << endl;
	}
}

void group_theoretic_activity::raise_to_the_power(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::raise_to_the_power" << endl;
	}

	A1->raise_to_the_power_based_on_text(Descr->raise_to_the_power_a_text,
			Descr->raise_to_the_power_exponent_text, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::raise_to_the_power done" << endl;
	}
}






#if 0
void group_theoretic_activity::search_subgroup(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::search_subgroup" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;
	int i, cnt;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif
		if (Elt[7] == 0 && Elt[8] == 0 &&
				Elt[11] == 0 && Elt[14] == 0 &&
				Elt[12] == 0 && Elt[19] == 0 &&
				Elt[22] == 0 && Elt[23] == 0) {
			cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << " = " << cnt << ":" << endl;
			A2->element_print(Elt, cout);
			cout << endl;
			//A->element_print_as_permutation(Elt, cout);
			//cout << endl;
			cnt++;
		}
	}
	cout << "we found " << cnt << " group elements of the special form" << endl;

	FREE_int(Elt);
	if (f_v) {
		cout << "group_theoretic_activity::search_subgroup done" << endl;
	}
}
#endif


void group_theoretic_activity::do_Andre_Bruck_Bose_construction(int spread_no,
		int f_Fano, int f_arcs, int f_depth, int depth,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *spread_elements_numeric; // do not free
	actions::action *An;
	actions::action *An1;
	data_structures_groups::vector_ge *gens;
	spreads::translation_plane_via_andre_model *Andre;
	groups::matrix_group *M; // do not free
	field_theory::finite_field *F;

	int f_semilinear = FALSE;
	int n, k, q;

	string stab_order;
	ring_theory::longinteger_object stab_go;
	int order_of_plane;
	number_theory::number_theory_domain NT;
	knowledge_base K;

	if (f_v) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction" << endl;
	}

	An = A1;
	An1 = A2;
	F = An->matrix_group_finite_field();
	M = An->get_matrix_group();
	f_semilinear = An->is_semilinear_matrix_group();
	n = An->matrix_group_dimension();
	if (ODD(n)) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction "
				"dimension must be even" << endl;
		exit(1);
	}
	k = n >> 1;
	q = F->q;

	if (f_v) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction "
				"n=" << n << " k=" << k << " q=" << q << endl;
	}

	order_of_plane = NT.i_power_j(q, k);


	int sz;
	//vector_ge *nice_gens;

	spread_elements_numeric = K.Spread_representative(q, k, spread_no, sz);




	An->stabilizer_of_spread_representative(q, k, spread_no,
			gens, stab_order, verbose_level);

	stab_go.create_from_base_10_string(stab_order);

	if (f_v) {
		cout << "Spread stabilizer has order " << stab_go << endl;
	}

	Andre = NEW_OBJECT(spreads::translation_plane_via_andre_model);

	Andre->init(spread_elements_numeric, k, An, An1,
		gens /*spread_stab_gens*/, stab_go, label, verbose_level);

	Andre->create_latex_report(verbose_level);


	if (f_Fano) {
		char prefix[1000];
		int nb_subplanes;

		sprintf(prefix, "Fano_TP_%d_", spread_no);

		Andre->classify_subplanes(prefix, verbose_level);

		int target_depth;

		if (f_depth) {
			target_depth = depth;
			}
		else {
			target_depth = 7;
			}

		nb_subplanes = Andre->arcs->nb_orbits_at_level(target_depth);

		cout << "Translation plane " << q << "#" << spread_no << " has "
				<<  nb_subplanes << " partial Fano subplanes "
						"(up to isomorphism) at depth "
				<< target_depth << endl;
		}
	else if (f_arcs) {
		char prefix[1000];
		int nb;

		int target_depth;

		if (f_depth) {
			target_depth = depth;
			}
		else {
			target_depth = order_of_plane + 2;
				// we are looking for hyperovals
			}


		sprintf(prefix, "Arcs_TP_%d_", spread_no);

		Andre->classify_arcs(prefix, target_depth, verbose_level);


		nb = Andre->arcs->nb_orbits_at_level(target_depth);

		cout << "Translation plane " << q << "#" << spread_no << " has "
				<<  nb << " Arcs of size " << target_depth
				<< " (up to isomorphism)" << endl;
		}

	FREE_OBJECT(Andre);
	FREE_OBJECT(gens);
	if (f_v) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction done" << endl;
	}
}




}}}


