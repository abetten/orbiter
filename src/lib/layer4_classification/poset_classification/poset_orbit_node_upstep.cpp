// poset_orbit_node_upstep.cpp
//
// Anton Betten
// December 27, 2004
// July 23, 2007

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


int poset_orbit_node::apply_isomorphism(
		poset_classification *gen,
	int lvl, int current_node, 
	int current_extension, int len, int f_tolerant,
	int *Elt_tmp1, int *Elt_tmp2,
	int verbose_level)
// returns next_node
{
	int f_v = (verbose_level >= 1);
	int next_node;
	long int *set;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism" << endl;
		}

	set = NEW_lint(len + 1); // this call should be eliminated
	//set = gen->get_tmp_set_apply_fusion();

	gen->get_A()->element_retrieve(
			E[current_extension].get_data(),
			Elt_tmp1, 0);
		// A Betten March 18 2012, this was gen->A2 previously
	
	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism "
				"applying fusion element" << endl;
		if (gen->allowed_to_show_group_elements()) {
			gen->get_A2()->element_print_quick(Elt_tmp1, cout);
			}
		cout << "in action " << gen->get_A2()->label << ":" << endl;
		if (gen->allowed_to_show_group_elements()) {
			gen->get_A2()->element_print_as_permutation(Elt_tmp1, cout);
			}
		cout << "to the set ";
		Lint_vec_print(cout, gen->get_set_i(lvl + 1), len + 1);
		cout << endl;
		}
	gen->get_A2()->map_a_set(
			gen->get_set_i(lvl + 1),
			set,
			len + 1,
			Elt_tmp1, 0);
	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism the set becomes: ";
		Lint_vec_print(cout, set, len + 1);
		cout << endl;
		}

	gen->get_A2()->element_mult(
			gen->get_transporter()->ith(lvl + 1),
			Elt_tmp1, Elt_tmp2, 0);
	if (f_v) {
		Lint_vec_print(cout,
				gen->get_set_i(lvl + 1), len + 1);
		cout << endl;
		}
	gen->get_A2()->move(Elt_tmp2,
			gen->get_transporter()->ith(lvl + 1));

	if (gen->get_poset()->f_subspace_lattice) {
		next_node = gen->find_node_for_subspace_by_rank(
				set,
				lvl + 1,
				verbose_level - 1);

		Lint_vec_copy(set, gen->get_set_i(lvl + 1), len + 1);
		}
	else {
		Sorting.lint_vec_heapsort(set, lvl + 1);
		Lint_vec_copy(set, gen->get_set_i(lvl + 1), len + 1);
		if (f_v) {
			cout << "poset_orbit_node::apply_isomorphism after sorting: ";
			}
		if (f_v) {
			cout << "poset_orbit_node::apply_isomorphism "
					"calling find_poset_orbit_node_for_set: ";
			Lint_vec_print(cout, gen->get_set_i(lvl + 1), lvl + 1);
			cout << endl;
			}
		next_node = gen->find_poset_orbit_node_for_set(
				lvl + 1,
				gen->get_set_i(lvl + 1),
				f_tolerant, 0);
		}

	FREE_lint(set);
	if (f_v) {
		cout << "poset_orbit_node::apply_isomorphism the set ";
		Lint_vec_print(cout, gen->get_set_i(lvl + 1), lvl + 1);
		cout << " is node " << next_node << endl;
		}
	return next_node;
}

void poset_orbit_node::install_fusion_node(
	poset_classification *gen,
	int lvl, int current_node, 
	int my_node, int my_extension, int my_coset, 
	long int pt0, int current_extension,
	int f_debug, int f_implicit_fusion, 
	int *Elt_tmp,
	int verbose_level)
// Called from poset_orbit_node::handle_last_level
// current_node is the same as poset_orbit_node::node !!!
// pt0 is the same as E[current_extension].pt !!!
{
	int f_v = (verbose_level >= 1);
	//int f_v10 = (verbose_level >= 10);
	int hdl, cmp;	
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node "
				" verbose_level = " << verbose_level <<
				"lvl=" << lvl
			<< " node=" << node 
			<< " current_node=" << current_node 
			<< " my_node=" << my_node 
			<< " current_extension=" << current_extension 
			<< " pt0=" << pt0 
			<< " E[current_extension].get_pt()=" << E[current_extension].get_pt()
			<< endl;
		}
		
	if (E[current_extension].get_pt() != pt0) {
		cout << "poset_orbit_node::install_fusion_node "
				"E[current_extension].pt != pt0" << endl;
		exit(1);
		}
	if (current_node != node) {
		cout << "poset_orbit_node::install_fusion_node "
				"current_node != node" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node: "
				"unprocessed extension, ";
		cout << "we will now install a fusion node at node " << node 
			<< " , extension " << current_extension << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node" << endl;
		cout << "transporter[lvl + 1]=" << endl;
		gen->get_A()->element_print_quick(
				gen->get_transporter()->ith(lvl + 1), cout);
		gen->get_A2()->element_print_as_permutation_verbose(
				gen->get_transporter()->ith(lvl + 1), cout, 0);
	}
	gen->get_A()->element_invert(
			gen->get_transporter()->ith(lvl + 1),
			Elt_tmp, FALSE);
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node" << endl;
		cout << "transporter[lvl + 1]^-1=Elt1=" << endl;
		gen->get_A()->element_print_quick(Elt_tmp, cout);
		gen->get_A2()->element_print_as_permutation_verbose(
				Elt_tmp, cout, 0);
	}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node: "
				"fusion element:" << endl;
		if (gen->allowed_to_show_group_elements()) {
			gen->get_A()->element_print_quick(Elt_tmp, cout);
			gen->get_A2()->element_print_as_permutation(Elt_tmp, cout);
			cout << endl;
			}
		}
	hdl = gen->get_A()->element_store(Elt_tmp, FALSE);
	//E[current_extension].type = EXTENSION_TYPE_FUSION;
	gen->get_Poo()->change_extension_type(lvl,
			current_node, current_extension,
			EXTENSION_TYPE_FUSION,
			0/* verbose_level*/);
	E[current_extension].set_data(hdl);
	E[current_extension].set_data1(my_node);
	E[current_extension].set_data2(my_extension);
	if (f_v) {
		cout << "FUSION NODE at lvl " << lvl
				<< " node/extension=" << current_node
				<< "/" << current_extension << " pt=" << pt0
				<< " hdl=" << hdl << " to node/extension="
				<< E[current_extension].get_data1() /*my_node*/
				<< "/" << E[current_extension].get_data2() /*my_extension*/
				<< " : ";
		Lint_vec_print(cout, gen->get_set0(), lvl + 1);
		cout << endl;
#if 0
		if (current_node == 9 && pt0 == 39371) {
			gen->A->element_print_quick(gen->Elt1, cout);
			gen->A2->element_print_as_permutation(gen->Elt1, cout);
			cout << endl;
			}
#endif
		//cout << "FUSION from=" << current_node << " / "
		//<< current_extension << " hdl=" << hdl << " to="
		// << my_node << "/" << my_extension << endl;
		}

	
	// we check it:
	store_set_to(gen, lvl - 1, gen->get_set1());
	gen->get_set1()[lvl] = pt0;
			
#if 0
	if (node == my_node || f_v) {
		cout << "poset_orbit_node::install_fusion_node "
				"fusion element stored in Node " << node
				<< ", extension " << current_extension
				<< " my_node = " << my_node << endl;
		gen->A->element_print_verbose(gen->Elt1, cout);
		cout << endl;
		cout << "Node " << node << " fusion from ";
		int_set_print(cout, gen->set1, lvl + 1);
		cout << " to ";
		int_set_print(cout, gen->set0, lvl + 1);
		cout << endl;
		if (node == my_node) {
			exit(1);
			}
		}
#endif

	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node set1=";
		Lint_vec_print(cout, gen->get_set1(), lvl + 1);
		cout << endl;
		cout << "Elt1=" << endl;
		gen->get_A()->element_print_quick(Elt_tmp, cout);
		cout << "before map_a_set" << endl;
	}
	gen->get_A2()->map_a_set(
			gen->get_set1(),
			gen->get_set3(),
			lvl + 1,
			Elt_tmp, 0);
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node "
				"after map_a_set set3=";
		Lint_vec_print(cout, gen->get_set3(), lvl + 1);
		cout << endl;
	}

	if (gen->get_poset()->f_subspace_lattice) {
		cmp = gen->get_VS()->compare_subspaces_ranked(
				gen->get_set3(), gen->get_set0(), lvl + 1, verbose_level);
#if 0
		cmp = gen->F->compare_subspaces_ranked_with_unrank_function(
			gen->set3, gen->set0, lvl + 1, 
			gen->vector_space_dimension, 
			gen->unrank_point_func,
			gen->rank_point_data, 
			verbose_level);
#endif
		}
	else {
		Sorting.lint_vec_heapsort(gen->get_set3(), lvl);
		cmp = Sorting.lint_vec_compare(gen->get_set3(), gen->get_set0(), lvl + 1);
		}


	if (cmp != 0) {
		cout << "poset_orbit_node::install_fusion_node "
				"something is wrong" << endl;
		cout << "comparing ";
		Lint_vec_print(cout, gen->get_set3(), lvl + 1);
		cout << " with ";
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, gen->get_set0(), lvl + 1);
		cout << endl;
		if (gen->get_poset()->f_subspace_lattice) {
			int *v;
			int i;

			v = NEW_int(gen->get_VS()->dimension);
			orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, gen->get_set3(), lvl + 1);
			cout << " is " << endl;
			for (i = 0; i < lvl + 1; i++) {
				gen->unrank_point(v, gen->get_set3()[i]);
				Int_vec_print(cout, v, gen->get_VS()->dimension);
				cout << endl;
				}
			orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, gen->get_set0(), lvl + 1);
			cout << " is " << endl;
			for (i = 0; i < lvl + 1; i++) {
				gen->unrank_point(v, gen->get_set0()[i]);
				Int_vec_print(cout, v, gen->get_VS()->dimension);
				cout << endl;
				}

			FREE_int(v);			
			}
		exit(1);
		}
	if (f_v) {
		cout << "poset_orbit_node::install_fusion_node done" << endl;
	}
}

int poset_orbit_node::trace_next_point_wrapper(
	poset_classification *gen,
	int lvl, int current_node,
	int len, int f_implicit_fusion,
	int *cosetrep,
	int &f_failure_to_find_point,
	int verbose_level)
// Called from upstep_work::recognize_recursion
// applies the permutation which maps the point with index lvl 
// (i.e. the lvl+1-st point) to its orbit representative.
// also maps all the other points under that permutation.
// we are dealing with a set of size len + 1
// returns FALSE if we are using implicit fusion
// nodes and the set becomes lexicographically
// less than before, in which case trace has to be restarted.
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper" << endl;
		cout << "poset_orbit_node::trace_next_point_wrapper current_node = " << current_node << endl;
		cout << "poset_orbit_node::trace_next_point_wrapper len = " << len << endl;
		cout << "poset_orbit_node::trace_next_point_wrapper lvl = " << lvl << endl;
		cout << "poset_orbit_node::trace_next_point_wrapper gen->get_transporter()->len = " << gen->get_transporter()->len << endl;
	}

	int *transporter_lvl;
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper setting transporter_lvl" << endl;
	}
	transporter_lvl = gen->get_transporter_i(lvl);

	int *transporter_lvlp1;

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper "
				"before trace_next_point setting transporter_lvlp1" << endl;
	}
	transporter_lvlp1 = gen->get_transporter_i(lvl + 1);

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper "
				"before trace_next_point" << endl;
	}

	ret = trace_next_point(gen,
		lvl,
		current_node,
		len + 1,
		gen->get_set_i(lvl),
		gen->get_set_i(lvl + 1),
		transporter_lvl,
		transporter_lvlp1,
		cosetrep,
		f_implicit_fusion,
		f_failure_to_find_point,
		0 /*verbose_level*/);

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper "
				"gen->get_set_i(" << lvl << ")=";
		Lint_vec_print(cout, gen->get_set_i(lvl), len + 1);
		cout << endl;
		cout << "poset_orbit_node::trace_next_point_wrapper "
				"gen->get_set_i(" << lvl + 1 << ")=";
		Lint_vec_print(cout, gen->get_set_i(lvl + 1), len + 1);
		cout << endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_wrapper done" << endl;
	}
	return ret;
}

int poset_orbit_node::trace_next_point_in_place(
	poset_classification *gen,
	int lvl,
	int current_node,
	int size,
	long int *cur_set,
	long int *tmp_set,
	int *cur_transporter,
	int *tmp_transporter,
	int *cosetrep,
	int f_implicit_fusion,
	int &f_failure_to_find_point,
	int verbose_level)
// called by poset_classification::trace_set_recursion
{
	int ret;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		//cout << "poset_orbit_node::trace_next_point_in_place" << endl;
		cout << "poset_orbit_node::trace_next_point_in_place "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_in_place, "
				"before trace_next_point" << endl;
	}

	ret = trace_next_point(gen,
		lvl,
		current_node,
		size,
		cur_set,
		tmp_set,
		cur_transporter,
		tmp_transporter,
		cosetrep,
		f_implicit_fusion,
		f_failure_to_find_point,
		verbose_level - 1);

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_in_place, "
				"after trace_next_point" << endl;
	}

	Lint_vec_copy(tmp_set, cur_set, size);

	gen->get_A()->element_move(tmp_transporter,
			cur_transporter, 0);

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point_in_place done" << endl;
	}
	return ret;
}

void poset_orbit_node::trace_starter(
	poset_classification *gen, int size,
	long int *cur_set, long int *next_set,
	int *cur_transporter, int *next_transporter, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt;
	int i;

	if (f_v) {
		cout << "poset_orbit_node::trace_starter" << endl;
		cout << "set:" << endl;
		Lint_vec_print(cout, cur_set, size);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	Elt = gen->get_Base_case()->Elt;
	
	gen->get_Base_case()->invoke_recognition(cur_set, size, Elt, verbose_level - 1);

	if (f_vv) {
		cout << "applying:" << endl;
		if (gen->allowed_to_show_group_elements()) {
			gen->get_A2()->element_print(Elt, cout);
			cout << endl;
			}
		}
		
	for (i = 0; i < size; i++) {
		next_set[i] = gen->get_A2()->element_image_of(
				cur_set[i], Elt, FALSE);
		}

	gen->get_A()->element_mult(cur_transporter,
			Elt,
			next_transporter,
			FALSE);

	if (f_v) {
		cout << "after canonize:" << endl;
		Lint_vec_print(cout, next_set, size);
		cout << endl;
		}
	if (f_v) {
		cout << "poset_orbit_node::trace_starter done" << endl;
		}
}


int poset_orbit_node::trace_next_point(
	poset_classification *gen,
	int lvl, int current_node, int size, 
	long int *cur_set, long int *next_set,
	int *cur_transporter, int *next_transporter, 
	int *cosetrep,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// Called by poset_orbit_node::trace_next_point_wrapper
// and by poset_orbit_node::trace_next_point_in_place
// returns FALSE only if f_implicit_fusion is TRUE and
// the set becomes lexicographically less
{
	long int the_point, pt0;
	int i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_v10 = (verbose_level >= 10);
	int ret;
	
	f_failure_to_find_point = FALSE;
	the_point = cur_set[lvl];

	if (f_v) {
		cout << "poset_orbit_node::trace_next_point lvl = " << lvl
				<< " the_point=" << the_point << endl;
		cout << "poset_orbit_node::trace_next_point node=" << node
				<< " prev=" << prev << " pt=" << pt << endl;
		cout << "poset_orbit_node::trace_next_point verbose_level = "
				<< verbose_level << endl;
	}
	
	if (gen->get_poset()->f_subspace_lattice) {
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point before "
					"orbit_representative_and_coset_rep_inv_subspace_action" << endl;
		}
		orbit_representative_and_coset_rep_inv_subspace_action(
			gen, lvl,
			the_point, pt0, cosetrep,
			verbose_level - 1);

			// poset_orbit_node_upstep_subspace_action.cpp
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point lvl = " << lvl
					<< " the_point=" << the_point
					<< " is traced to " << pt0 << endl;
		}
	}
	else {
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"before orbit_representative_and_coset_rep_inv" << endl;
		}
		if (!orbit_representative_and_coset_rep_inv(
			gen, lvl,
			the_point, pt0, cosetrep,
			verbose_level - 1)) {
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point "
						"orbit_representative_and_coset_rep_inv returns FALSE, "
						"f_failure_to_find_point = TRUE" << endl;
			}
			f_failure_to_find_point = TRUE;
			return TRUE;
		}
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"after orbit_representative_and_coset_rep_inv" << endl;
		}
	}
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point lvl = " << lvl
				<< " mapping "
				<< the_point << " -> " << pt0
				<< " under the element " << endl;
		gen->get_A2()->element_print_quick(cosetrep, cout);
		cout << "in action " << gen->get_A2()->label << endl;
		if (gen->allowed_to_show_group_elements()) {
			gen->get_A2()->element_print_as_permutation_verbose(
					cosetrep, cout, 0);
			cout << endl;
		}
	}
	if (pt0 == the_point) {
		if (f_v) {
			cout << "Since the image point is equal "
					"to the original point, "
					"we apply no element and copy the set "
					"and the transporter over:" << endl;
		}
		Lint_vec_copy(cur_set, next_set, size);
		gen->get_A2()->element_move(cur_transporter, next_transporter, FALSE);
	}
	else {
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point lvl = " << lvl
					<< " applying:" << endl;
			gen->get_A2()->element_print_quick(cosetrep, cout);

			cout << "in action " << gen->get_A2()->label << endl;

			if (gen->allowed_to_show_group_elements()) {
				gen->get_A2()->element_print_as_permutation_verbose(
						cosetrep, cout, 0);
				cout << endl;
			}
			cout << "poset_orbit_node::trace_next_point cur_set: ";
			Lint_vec_print(cout, cur_set, size);
			cout << endl;
		}
		
		Lint_vec_copy(cur_set, next_set, lvl);

		next_set[lvl] = pt0;

		for (i = lvl + 1; i < size; i++) {
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point lvl = " << lvl
						<< " mapping point " << i << " / " << size
						<< "cur_set[i]=" << cur_set[i] << endl;
			}
			next_set[i] = gen->get_A2()->element_image_of(
					cur_set[i], cosetrep, 0 /*verbose_level*/);
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point lvl = " << lvl
						<< " mapping point " << i << " / " << size
						<< "next_set[i]=" << next_set[i] << endl;
			}
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point "
						"lvl = " << lvl << ": ";
				cout << "mapping " << i << "-th point: "
						<< cur_set[i] << "->" << next_set[i] << endl;
			}
		}
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point next_set: ";
			Lint_vec_print(cout, next_set, size);
			cout << endl;
		}

		//gen->A->map_a_set(gen->set[lvl],
		// gen->set[lvl + 1], len + 1, cosetrep);

		//int_vec_sort(len, gen->set[lvl + 1]);
		// we keep the last point extra

#if 0
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"before element_mult" << endl;
			cout << "cur_transporter:" << endl;
			gen->A->element_print_quick(cur_transporter, cout);
			cout << "cosetrep:" << endl;
			gen->A->element_print_quick(cosetrep, cout);
			}
#endif
		gen->get_A()->element_mult(cur_transporter,
				cosetrep, next_transporter, 0);
#if 0
		if (f_v) {
			cout << "poset_orbit_node::trace_next_point "
					"after element_mult" << endl;
			}
#endif
		}
	
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point lvl = " << lvl
			<< " mapping " << the_point << "->" << pt0
			<< " done, the set becomes ";
		orbiter_kernel_system::Orbiter->Lint_vec->set_print(cout, next_set, size);
		cout << endl;

		if (gen->get_poset()->f_print_function && f_vv) {
			gen->get_poset()->invoke_print_function(cout, size, next_set);
		}
		if (gen->allowed_to_show_group_elements() && f_vv) {
			cout << "poset_orbit_node::trace_next_point the n e w "
					"transporter is" << endl;
			gen->get_A2()->element_print_quick(next_transporter, cout);
			gen->get_A2()->element_print_as_permutation(
					next_transporter, cout);
			cout << endl;
		}
		
	}
	
	if (f_implicit_fusion) {
		// this is needed if implicit fusion nodes are used
	
		if (lvl > 0 && next_set[lvl] < next_set[lvl - 1]) {
			if (f_v) {
				cout << "poset_orbit_node::trace_next_point the set becomes "
						"lexicographically less" << endl;
			}
			ret = FALSE;
		}
		else {
			ret = TRUE;
		}
	}
	else {
		ret = TRUE;
	}
	if (f_v) {
		cout << "poset_orbit_node::trace_next_point "
				"lvl = " << lvl << " done, ret=" << ret << endl;
	}
	return ret;
}

int poset_orbit_node::orbit_representative_and_coset_rep_inv(
	poset_classification *gen,
	int lvl, long int pt_to_trace,
	long int &pt0, int *cosetrep,
	int verbose_level)
// called by poset_orbit_node::trace_next_point
{
	int f_v = (verbose_level >= 1);
	//int f_check_image = FALSE;
	//int f_allow_failure = TRUE;

	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
				"lvl=" << lvl
				<< " pt_to_trace=" << pt_to_trace << endl;
	}
	if (nb_strong_generators == 0) {
		//cosetrep = gen->get_Elt1();
		gen->get_A()->element_one(cosetrep, FALSE);
		pt0 = pt_to_trace;
		return TRUE;
	}

	if (Schreier_vector) {

		//cout << "Node " << node
		//<< " poset_orbit_node::orbit_representative_and_"
		//"coset_rep_inv calling schreier_vector_coset_rep_inv" << endl;


		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
					"before gen->Schreier_vector_handler->coset_rep_inv_lint "
					"verbose_level=" << verbose_level
					<< endl;
		}
		if (!gen->get_schreier_vector_handler()->coset_rep_inv_lint(
				Schreier_vector,
				pt_to_trace,
				pt0,
				verbose_level - 1)) {

			if (f_v) {
				cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
						"schreier_vector_coset_rep_inv_lint returns FALSE, "
						"point not found" << endl;
			}
			return FALSE;
		}

		if (f_v) {
			cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
					"after gen->Schreier_vector_handler->coset_rep_inv"
					<< endl;
		}

		//cosetrep = gen->get_schreier_vector_handler()->cosetrep;

		gen->get_A2()->element_move(gen->get_schreier_vector_handler()->cosetrep, cosetrep, 0);


		// gen->Elt1 contains the element that maps pt_to_trace to pt0
		//cout << "Node " << node << " poset_orbit_node::orbit_representative_and_"
		//"coset_rep_inv schreier_vector_coset_rep_inv done" << endl;
	}
	else {
		if (f_v) {
			cout << "Node " << node
					<< " poset_orbit_node::orbit_representative_and_coset_rep_inv "
							"Schreier_vector not available, "
							"calling least_image_of_point_generators_by_handle" << endl;
		}
		//cosetrep = gen->get_Elt1();


		std::vector<int> gen_handle;
		actions::action_global AcGl;

		get_strong_generators_handle(gen_handle, verbose_level - 2);


		pt0 = AcGl.least_image_of_point_generators_by_handle(
				gen->get_A2(),
			gen_handle,
			pt_to_trace,
			cosetrep,
			verbose_level - 1);

	}

	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv "
				"pt_to_trace=" << pt_to_trace
				<< " pt0=" << pt0 <<  " done" << endl;
	}

	if (f_v) {
		cout << "poset_orbit_node::orbit_representative_and_coset_rep_inv cosetrep:" << endl;
		gen->get_A2()->element_print_quick(cosetrep, cout);
	}
	return TRUE;
}

}}}



