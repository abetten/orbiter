// poset_orbit_node_group_theory.cpp
//
// Anton Betten
//
// moved out of poset_orbit_node.cpp: Nov 9, 2018
// December 27, 2004

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


void poset_orbit_node::store_strong_generators(
		poset_classification *gen,
		groups::strong_generators *Strong_gens)
{
	int i;

	nb_strong_generators = Strong_gens->gens->len;
	if (nb_strong_generators == 0) {
		first_strong_generator_handle = -1;
		//hdl_strong_generators = NULL;
		tl = NULL;
	}
	else {
		//hdl_strong_generators = NEW_int(nb_strong_generators);
		tl = NEW_int(gen->get_A()->base_len());
		for (i = 0; i < nb_strong_generators; i++) {

			if (i == 0) {
				first_strong_generator_handle =
					gen->get_A()->Group_element->element_store(Strong_gens->gens->ith(i), false);
			}
			else {
				gen->get_A()->Group_element->element_store(Strong_gens->gens->ith(i), false);
			}
		}
		Int_vec_copy(Strong_gens->tl, tl, gen->get_A()->base_len());
	}
}



#if 0
void poset_orbit_node::get_stabilizer_order(poset_classification *gen, longinteger_object &go)
{
	strong_generators *Strong_gens;

	get_stabilizer_generators(gen,
		Strong_gens, 0 /*verbose_level*/);
	Strong_gens->group_order(go);
	// ToDo: free Strong_gens
}
#else
void poset_orbit_node::get_stabilizer_order(
		poset_classification *PC, ring_theory::longinteger_object &go)
{
	if (nb_strong_generators) {
		go.create_product(PC->get_poset()->A->base_len(), tl);
	}
	else {
		go.create(1, __FILE__, __LINE__);
	}
}
#endif




long int poset_orbit_node::get_stabilizer_order_lint(
		poset_classification *PC)
{
	ring_theory::longinteger_object go;

	if (nb_strong_generators) {
		go.create_product(PC->get_poset()->A->base_len(), tl);
	}
	else {
		go.create(1, __FILE__, __LINE__);
	}
	return go.as_lint();
}

void poset_orbit_node::get_stabilizer(
	poset_classification *PC,
	data_structures_groups::group_container &G,
	ring_theory::longinteger_object &go_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::get_stabilizer, nb_strong_generators=" << nb_strong_generators << endl;
	}
	G.init(PC->get_A(), verbose_level - 2);



#if 0
	G.init_strong_generators_by_hdl(
			nb_strong_generators, hdl_strong_generators, tl, 0);
#else

	std::vector<int> gen_handle;
	std::vector<int> tl;

	get_strong_generators_handle(gen_handle, verbose_level - 2);
	get_tl(tl, PC, verbose_level - 2);

	G.init_strong_generators_by_handle_and_with_tl(
				gen_handle,
				tl, 0 /*verbose_level*/);
#endif
	if (f_v) {
		cout << "poset_orbit_node::get_stabilizer "
				"calling schreier_sims for stabilizer with "
			<< nb_strong_generators << " strong generators" << endl;
	}
	G.schreier_sims(verbose_level - 3);
	G.group_order(go_G);
	if (f_v) {
		cout << "poset_orbit_node::get_stabilizer "
				"stabilizer has order "
				<< go_G << endl;
	}
}

int poset_orbit_node::test_if_stabilizer_is_trivial()
{
	if (nb_strong_generators == 0) {
		return true;
	}
	else {
		return false;
	}
}

void poset_orbit_node::get_stabilizer_generators(
	poset_classification *PC,
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "poset_orbit_node::get_stabilizer_generators" << endl;
		cout << "poset_orbit_node::get_stabilizer_generators "
				"nb_strong_generators=" << nb_strong_generators << endl;
	}
	Strong_gens = NEW_OBJECT(groups::strong_generators);

#if 0
	Strong_gens->init_by_hdl(PC->get_A(),
			hdl_strong_generators, nb_strong_generators, 0);
	if (nb_strong_generators == 0) {
		for (i = 0; i < PC->get_A()->base_len(); i++) {
			Strong_gens->tl[i] = 1;
		}
	}
	else {
		for (i = 0; i < PC->get_A()->base_len(); i++) {
			Strong_gens->tl[i] = poset_orbit_node::tl[i];
		}
	}
#else
	std::vector<int> gen_handle;
	std::vector<int> tl;

	get_strong_generators_handle(gen_handle, verbose_level - 2);
	get_tl(tl, PC, verbose_level - 2);

	Strong_gens->init_by_hdl_and_with_tl(PC->get_A(),
			gen_handle,
			tl,
			verbose_level - 3);
#endif

}

void poset_orbit_node::init_extension_node_prepare_G(
	poset_classification *PC,
	int prev, int prev_ex, int size,
	data_structures_groups::group_container &G,
	ring_theory::longinteger_object &go_G,
	int verbose_level)
// sets up the group G using the strong generators that are stored
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "poset_orbit_node::init_extension_node_prepare_G" << endl;
	}
	poset_orbit_node *Op = PC->get_node(prev);


	if (f_v) {
		cout << "poset_orbit_node::init_extension_node_prepare_G before Op->get_stabilizer" << endl;
	}
	Op->get_stabilizer(PC, G, go_G, 0 /*verbose_level */);
	if (f_v) {
		cout << "poset_orbit_node::init_extension_node_prepare_G after Op->get_stabilizer" << endl;
	}

#if 0
	G.init(PC->get_A(), verbose_level - 2);
	if (f_vv) {
		PC->print_level_extension_info(size - 1, prev, prev_ex);
		lint_vec_print(cout, PC->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_G "
				"calling init_strong_generators_by_hdl" << endl;
#if 0
		int_vec_print(cout,
				Op->hdl_strong_generators,
				Op->nb_strong_generators);
		cout << endl;
#endif
		cout << "verbose_level=" << verbose_level << endl;
	}

	G.init_strong_generators_by_hdl(
			Op->nb_strong_generators,
			Op->hdl_strong_generators,
			Op->tl, verbose_level - 1);
#endif

	if (f_vvv) {
		PC->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, PC->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_G "
				"the strong generators are:" << endl;
		G.print_strong_generators(cout,
				false /* f_print_as_permutation */);
	}

#if 0
	if (f_vv) {
		PC->print_level_extension_info(size - 1, prev, prev_ex);
		lint_vec_print(cout, PC->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_G "
				"before schreier_sims for stabilizer with "
			<< Op->nb_strong_generators << " strong generators" << endl;
	}
	G.schreier_sims(0 /*verbose_level - 2*/);
	if (f_vv) {
		PC->print_level_extension_info(size - 1, prev, prev_ex);
		lint_vec_print(cout, PC->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_G "
				"after schreier_sims" << endl;
	}

	//G.group_order(go_G);
#endif
	if (f_vv) {
		PC->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, PC->get_S(), size);
		cout << "_{" << go_G << "}, previous stabilizer "
				"has been reconstructed" << endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::init_extension_node_prepare_G "
				"done" << endl;
	}

}


void poset_orbit_node::init_extension_node_prepare_H(
	poset_classification *gen,
	int prev, int prev_ex, int size,
	data_structures_groups::group_container &G,
	ring_theory::longinteger_object &go_G,
	data_structures_groups::group_container &H,
	ring_theory::longinteger_object &go_H,
	long int pt, int pt_orbit_len,
	int verbose_level)
// sets up the group H which is the stabilizer of the point pt in G
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_v5 = (verbose_level >= 5);
	//int f_v10 = (verbose_level >= 10);


	if (f_v) {
		cout << "poset_orbit_node::init_extension_node_prepare_H, "
				"verbose_level = " << verbose_level
				<< " pt = " << pt
				<< " pt_orbit_len = " << pt_orbit_len
				<< endl;
	}


	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_H "
				"computing stabilizer of point " << pt
			<< " (of index " << pt_orbit_len
			<< " in a group of order " << go_G;
		if (G.f_has_sims && !go_G.is_one()) {
			cout << " = ";
			G.S->print_group_order_factored(cout);
		}
		cout << ")" << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

	//cout << "computing point stabilizer" << endl;
	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_H "
				"computing stabilizer of point " << pt
				<< " in group of order " << go_G << endl;
	}

	if (gen->get_poset()->f_subspace_lattice) {

		if (f_vv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "poset_orbit_node::init_extension_node_prepare_H "
					"before compute_point_stabilizer_in_subspace_setting"
					<< endl;
		}
		compute_point_stabilizer_in_subspace_setting(gen,
			prev, prev_ex, size,
			G, go_G,
			H, go_H,
			pt, pt_orbit_len,
			verbose_level - 3);
		if (f_vv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "poset_orbit_node::init_extension_node_prepare_H "
					"after compute_point_stabilizer_in_subspace_setting"
					<< endl;
		}

	}
	else {
		// action on sets:

		if (f_vv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "poset_orbit_node::init_extension_node_prepare_H "
					"before compute_point_stabilizer_in_standard_setting"
					<< endl;
		}
		compute_point_stabilizer_in_standard_setting(gen,
			prev, prev_ex, size,
			G, go_G,
			H,
			pt, pt_orbit_len,
			verbose_level - 1);
		if (f_vv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "poset_orbit_node::init_extension_node_prepare_H "
					"after compute_point_stabilizer_in_standard_setting"
					<< endl;
		}

	}
	// now H has strong generators only

	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "poset_orbit_node::init_extension_node_prepare_H "
				"calling schreier_sims for point stabilizer" << endl;
	}
	H.schreier_sims(0);

	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "poset_orbit_node::init_extension_node_prepare_H "
				"after schreier_sims for point stabilizer" << endl;
	}



	ring_theory::longinteger_object q, r;
	ring_theory::longinteger_domain D;


	H.group_order(go_H);
	D.integral_division(go_G, go_H, q, r, 0);
	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_H "
				"point stabilizer has order ";
		H.print_group_order(cout);
		//cout << endl;
		cout << ", of index = " << q << " in " << go_G << endl;
		//H.S->print(true);
	}
	if (q.as_int() != pt_orbit_len) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_H: "
				"fatal: q != pt_orbit_len" << endl;
		cout << "go_G = " << go_G << endl;
		cout << "go_H = " << go_H << endl;
		cout << "q = " << q << endl;
		cout << "pt_orbit_len = " << pt_orbit_len << endl;
		exit(1);
	}
	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "poset_orbit_node::init_extension_node_prepare_H "
				"point stabilizer is generated by:" << endl;
		int f_print_as_permutation = false;
		if (/*f_v10 &&*/ gen->get_A2()->degree < 100) {
			f_print_as_permutation = true;
		}
		H.print_strong_generators(cout, f_print_as_permutation);
	}

	if (f_v) {
		cout << "poset_orbit_node::init_extension_node_prepare_H done" << endl;
	}
}

void poset_orbit_node::compute_point_stabilizer_in_subspace_setting(
	poset_classification *gen,
	int prev, int prev_ex, int size,
	data_structures_groups::group_container &G,
	ring_theory::longinteger_object &go_G,
	data_structures_groups::group_container &H,
	ring_theory::longinteger_object &go_H,
	long int pt, int pt_orbit_len,
	int verbose_level)
// we are at the new node, and prev is the node from which we came.
// prev_ex is the extension that we are currently processing.
// size is the new size.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int projected_pt;


	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "poset_orbit_node::compute_point_stabilizer_"
				"in_subspace_setting, "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "generators for G of order " << go_G << endl;
		G.SG->print_with_given_action(
				cout, gen->get_A2());
	}

	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "poset_orbit_node::compute_point_stabilizer_"
				"in_subspace_setting, before H.init()" << endl;
	}
	H.init(gen->get_A(), verbose_level - 2);
	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "poset_orbit_node::compute_point_stabilizer_"
				"in_subspace_setting, after H.init()" << endl;
	}



#if 0
	action_on_factor_space *AF;
	action *A_factor_space;

	if (gen->root[prev].A_on_upset == NULL) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "poset_orbit_node::compute_point_stabilizer_"
				"in_subspace_setting gen->root[prev].A_on_upset == NULL" << endl;
		exit(1);
	}
	A_factor_space = gen->root[prev].A_on_upset;
	AF = A_factor_space->G.AF;
	if (f_v) {
		cout << "generators for G of order " << go_G
				<< " in factor space action" << endl;
		G.SG->print_with_given_action(
				cout, A_factor_space);
	}
#endif


	{
#if 1
		induced_actions::action_on_factor_space *AF;
		actions::action *A_factor_space;
		poset_orbit_node *Op = gen->get_node(prev);

		AF = NEW_OBJECT(induced_actions::action_on_factor_space);
		if (false /*gen->f_early_test_func*/) {
#if 0
			int i;

			if (f_v) {
				cout << "poset_orbit_node::compute_point_stabilizer_"
						"in_subspace_setting, "
						"with early test function, "
						"before Op->setup_factor_space_action_with_early_test"
						<< endl;
				}
			Op->setup_factor_space_action_with_early_test(gen,
				AF, A_factor_space, size - 1,
				verbose_level - 4);
			if (f_v) {
				cout << "poset_orbit_node::compute_point_stabilizer_"
						"in_subspace_setting after "
						"Op->setup_factor_space_action_with_early_test"
						<< endl;
				}
			for (i = 0; i < AF.nb_cosets; i++) {
				if (AF.preimage(i, 0) == pt) {
					if (f_vv) {
						cout << "poset_orbit_node::compute_point_stabilizer_"
								"in_subspace_setting: point pt=" << pt
								<< " is coset " << i << endl;
						}
					break;
					}
				}
			if (i == AF.nb_cosets) {
				cout << "poset_orbit_node::compute_point_stabilizer_"
						"in_subspace_setting "
						"fatal: could not find the coset corresponding "
						"to point " << pt << endl;
				exit(1);
				}
			projected_pt = i;
#endif
		}
		else {
			// no early_test_func:

			if (f_vvv) {
				gen->print_level_extension_info(size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_"
						"in_subspace_setting, "
						"without early test function, setting up factor "
						"space action:" << endl;
			}
			Op->setup_factor_space_action(
				gen,
				AF,
				A_factor_space,
				size - 1,
				true /*f_compute_tables*/,
				verbose_level - 4);
			// now AF is part of A_factor_space
			// and should not be freed.
			projected_pt = AF->project(pt, verbose_level - 4);
		}
#endif

#if 0
		if (f_vvv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			cout << " poset_orbit_node::compute_point_stabilizer_"
					"in_subspace_setting "
					" before AF->project_onto_Gauss_reduced_vector"
					<< endl;
		}
#endif
		//projected_pt = AF->project(pt, verbose_level - 4);
		//projected_pt = AF->project_onto_Gauss_reduced_vector(
		//		pt, verbose_level - 4);

		if (f_vvv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			cout << " poset_orbit_node::compute_point_stabilizer_"
					"in_subspace_setting "
					" pt=" << pt << " projected_pt=" << projected_pt << endl;
		}
		if (projected_pt == -1) {
			cout << "poset_orbit_node::compute_point_stabilizer_"
					"in_subspace_setting "
					"fatal: projected_pt == -1" << endl;
			exit(1);
		}
		if (f_vvv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			cout << " poset_orbit_node::compute_point_stabilizer_"
					"in_subspace_setting "
					"calling G.point_stabilizer_with_action "
					"verbose_level=" << verbose_level << endl;
		}

#if 0
		if (size == 2 && prev == 2 && prev_ex == 6) {
			verbose_level += 10;
			cout << "START" << endl;
		}
#endif
		G.point_stabilizer_with_action(
				A_factor_space,
				H,
				projected_pt,
				verbose_level - 4);
		if (f_vvv) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			cout << " poset_orbit_node::compute_point_stabilizer_"
					"in_subspace_setting "
					"G.point_stabilizer_with_action done" << endl;
		}

		if (f_v) {
			cout << "poset_orbit_node::compute_point_stabilizer_"
					"in_subspace_setting "
					"before freeing A_factor_space" << endl;
		}
		FREE_OBJECT(A_factor_space);
	}
	if (f_v) {
		cout << "poset_orbit_node::compute_point_stabilizer_"
				"in_subspace_setting "
				"done" << endl;
	}

}

void poset_orbit_node::compute_point_stabilizer_in_standard_setting(
	poset_classification *gen,
	int prev, int prev_ex, int size,
	data_structures_groups::group_container &G,
	ring_theory::longinteger_object &go_G,
	data_structures_groups::group_container &H, /*longinteger_object &go_H, */
	int pt, int pt_orbit_len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r;
	ring_theory::longinteger_object go_H;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting, "
				"verbose_level = " << verbose_level << endl;
	}

	D.integral_division_by_int(go_G, pt_orbit_len, go_H, r);
	if (r != 0) {
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting "
				"r != 0" << endl;
		cout << "go_G=" << go_G << endl;
		cout << "pt_orbit_len=" << pt_orbit_len << endl;
		cout << "go_H=" << go_H << endl;
		exit(1);
	}

	H.init(gen->get_A(), verbose_level - 2);

	poset_orbit_node *Op = gen->get_node(prev);


	if (f_v) {
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting, "
				"verbose_level = " << verbose_level << endl;
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting, "
				"go_G = " << go_G << endl;
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting, "
				"pt_orbit_len = " << pt_orbit_len << endl;
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting, "
				"go_H = " << go_H << endl;
	}

	if (Op->Schreier_vector) {
		if (f_v) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting "
					"setting up restricted action from the previous "
					"schreier vector:" << endl;
		}


		if (Op->nb_strong_generators) {
			// if G is non-trivial

			actions::action *AR;

			if (f_v) {
				gen->print_level_extension_info(size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting "
						"before gen->get_A2()->Induced_action->induced_action_by_restriction_on_orbit_with_schreier_vector" << endl;
			}
			AR = gen->get_A2()->Induced_action->induced_action_by_restriction_on_orbit_with_schreier_vector(
				false /* f_induce_action */,
				NULL /* old_G */,
				Op->Schreier_vector /* Op->sv*/,
				pt,
				0 /*verbose_level - 1*/);
			if (f_v) {
				gen->print_level_extension_info(size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting "
						"after gen->get_A2()->Induced_action->induced_action_by_restriction_on_orbit_with_schreier_vector" << endl;
			}
			if (f_v) {
				gen->print_level_extension_info(size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting created action of degree "
						<< AR->degree << endl;
			}
			if (f_v) {
				gen->print_level_extension_info(
						size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting calling "
						"G.point_stabilizer_with_action"
						<< endl;
			}
			G.point_stabilizer_with_action(
					AR, H, AR->G.ABR->idx_of_root_node /* 0 */ /*pt */,
					0 /*verbose_level - 3*/);
			if (f_v) {
				gen->print_level_extension_info(size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting after "
						"G.point_stabilizer_with_action"
						<< endl;
			}

			ring_theory::longinteger_object go_H1;
			H.group_order(go_H1);
			ring_theory::longinteger_domain D;
			if (D.compare(go_H, go_H1) != 0) {
				cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting "
						"go_H is incorrect" << endl;
				cout << "go_H=" << go_H << endl;
				cout << "go_H1=" << go_H1 << endl;
				exit(1);
			}

			if (f_v) {
				gen->print_level_extension_info(size - 1, prev, prev_ex);
				cout << " poset_orbit_node::compute_point_stabilizer_in_standard_setting "
						"G.point_stabilizer_with_action done"
						<< endl;
			}
			if (f_v) {
				cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting "
						"before FREE_OBJECT(AR)" << endl;
			}
			FREE_OBJECT(AR);
			if (f_v) {
				cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting "
						"after FREE_OBJECT(AR)" << endl;
			}
		}
		else {
			// do nothing, the stabilizer is trivial (since G is trivial)
			data_structures_groups::vector_ge stab_gens;
			int *tl;
			int i;

			stab_gens.init(gen->get_A(), verbose_level - 2);
			stab_gens.allocate(0, verbose_level - 2);
			tl = NEW_int(gen->get_A()->base_len());
			for (i = 0; i < gen->get_A()->base_len(); i++) {
				tl[i] = 1;
			}

			H.init(gen->get_A(), verbose_level - 2);
			H.init_strong_generators(stab_gens, tl, verbose_level - 2);
			FREE_int(tl);
		}
	}
	else {
		if (f_v) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			cout << " previous schreier vector not available. "
					"Before G.point_stabilizer_with_action" << endl;
		}
		G.point_stabilizer_with_action(gen->get_A2(), H, pt, 0);
	}

	if (f_v) {
		cout << "poset_orbit_node::compute_point_stabilizer_in_standard_setting done" << endl;
	}

}

void poset_orbit_node::create_schreier_vector_wrapper(
	poset_classification *gen,
	int f_create_schreier_vector,
	groups::schreier *Schreier,
	int verbose_level)
// calls Schreier.get_schreier_vector
// called from poset_orbit_node_downstep.cpp and from
// poset_orbit_node_downstep_subspace_action.cpp
// ToDo add an enum for the shallow schreier tree strategy
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int gen_hdl_first;

	if (f_v) {
		cout << "poset_orbit_node::create_schreier_vector_wrapper" << endl;
	}
	if (f_create_schreier_vector) {

		if (f_vv) {
			cout << "poset_orbit_node::create_schreier_vector_wrapper "
					"calling get_schreier_vector" << endl;
		}
		if (f_vv) {
			//int i;
			cout << "poset_orbit_node::create_schreier_vector_wrapper " << endl;
#if 0
			for (i = 0; i < nb_strong_generators; i++) {
				cout << "hdl_strong_generators[" << i << "]="
						<< hdl_strong_generators[i] << endl;
			}
#endif

		}
		if (nb_strong_generators == 0) {
			gen_hdl_first = -1;
		}
		else {
			//gen_hdl_first = hdl_strong_generators[0];
			gen_hdl_first = first_strong_generator_handle;
		}

		// ToDo: set the shallow schreier strategy

		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				//shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;
		Schreier_vector = Schreier->get_schreier_vector(
				gen_hdl_first,
				nb_strong_generators,
				Shallow_schreier_tree_strategy,
				verbose_level - 1);
		if (Schreier_vector->f_has_local_generators) {
			Schreier_vector->local_gens->A = gen->get_schreier_vector_handler()->A2;
		}
	}
	else {
		Schreier_vector = NULL;
	}
	if (f_v) {
		cout << "poset_orbit_node::create_schreier_vector_wrapper "
				"done" << endl;
	}
}


void poset_orbit_node::create_schreier_vector_wrapper_subspace_action(
	poset_classification *gen,
	int f_create_schreier_vector,
	groups::schreier &Schreier,
	actions::action *A_factor_space,
	induced_actions::action_on_factor_space *AF,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 10);

	if (f_v) {
		cout << "poset_orbit_node::create_schreier_vector_wrapper_subspace_action"
				<< endl;
	}
	if (f_create_schreier_vector) {

		if (f_vv) {
			cout << "calling get_schreier_vector" << endl;
		}
		int gen_hdl_first;
		if (nb_strong_generators == 0) {
			gen_hdl_first = -1;
		}
		else {
			//gen_hdl_first = hdl_strong_generators[0];
			gen_hdl_first = first_strong_generator_handle;
		}

		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				//shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;

		Schreier_vector = Schreier.get_schreier_vector(
				gen_hdl_first,
				nb_strong_generators,
				Shallow_schreier_tree_strategy,
				0 /*verbose_level - 1*/);

		if (f_vv) {
			cout << "schreier vector before relabeling :" << endl;
			if (Schreier_vector->get_number_of_points() < 100) {
				Int_vec_print(cout, Schreier_vector->points(),
						Schreier_vector->get_number_of_points());
				cout << endl;
			}
			else {
				cout << "too large to print" << endl;
			}
		}
		if (f_v) {
			cout << "poset_orbit_node::create_schreier_vector_wrapper_subspace_action "
					"changing point labels:" << endl;
		}
		Schreier_vector->relabel_points(AF, 0 /*verbose_level - 4*/);
		if (f_v) {
			cout << "poset_orbit_node::create_schreier_vector_wrapper_subspace_action "
					"changing point labels done" << endl;
		}
		if (f_vv) {
			cout << "schreier vector after relabeling :" << endl;
			if (Schreier_vector->get_number_of_points() < 100) {
				Int_vec_print(cout, Schreier_vector->points(),
						Schreier_vector->get_number_of_points());
				cout << endl;
			}
			else {
				cout << "too large to print" << endl;
			}
		}
	}
	else {
		Schreier_vector = NULL;
	}
	if (f_v) {
		cout << "poset_orbit_node::create_schreier_vector_wrapper_subspace_action "
				"done" << endl;
	}
}

}}}


