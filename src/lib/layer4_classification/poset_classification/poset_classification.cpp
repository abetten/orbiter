// poset_classification.cpp
//
// Anton Betten
// December 29, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {

poset_of_orbits *poset_classification::get_Poo()
{
	return Poo;
}

orbit_tracer *poset_classification::get_Orbit_tracer()
{
	return Orbit_tracer;
}

std::string &poset_classification::get_problem_label_with_path()
{
	return problem_label_with_path;
}

std::string &poset_classification::get_problem_label()
{
	return problem_label;
}

int poset_classification::first_node_at_level(
		int i)
{
	return Poo->first_node_at_level(i);
}

poset_orbit_node *poset_classification::get_node(
		int node_idx)
{
	return Poo->get_node(node_idx);
}

data_structures_groups::vector_ge *poset_classification::get_transporter()
{
	return Orbit_tracer->get_transporter();
}

int *poset_classification::get_transporter_i(
		int i)
{
	return Orbit_tracer->get_transporter()->ith(i);
}

int poset_classification::get_sz()
{
	return sz;
}

int poset_classification::get_max_set_size()
{
	return max_set_size;
}

long int *poset_classification::get_S()
{
	return set_S;
}

long int *poset_classification::get_set_i(
		int i)
{
	return Orbit_tracer->get_set_i(i);
}

long int *poset_classification::get_set0()
{
	return Poo->get_set0();
}

long int *poset_classification::get_set1()
{
	return Poo->get_set1();
}

long int *poset_classification::get_set3()
{
	return Poo->get_set3();
}

int poset_classification::allowed_to_show_group_elements()
{
	return Control->f_allowed_to_show_group_elements;
}

int poset_classification::do_group_extension_in_upstep()
{
	return Control->f_do_group_extension_in_upstep;
}

poset_with_group_action *poset_classification::get_poset()
{
	return Poset;
}

poset_classification_control *poset_classification::get_control()
{
	return Control;
}

actions::action *poset_classification::get_A()
{
	return Poset->A;
}

actions::action *poset_classification::get_A2()
{
	return Poset->A2;
}

algebra::linear_algebra::vector_space *poset_classification::get_VS()
{
	return Poset->VS;
}

data_structures_groups::schreier_vector_handler *poset_classification::get_schreier_vector_handler()
{
	return Schreier_vector_handler;
}

int &poset_classification::get_depth()
{
	return depth;
}

int poset_classification::has_base_case()
{
	return f_base_case;
}

int poset_classification::has_invariant_subset_for_root_node()
{
	return Control->f_has_invariant_subset_for_root_node;
}

int poset_classification::size_of_invariant_subset_for_root_node()
{
	return Control->invariant_subset_for_root_node_size;
}

int *poset_classification::get_invariant_subset_for_root_node()
{
	return Control->invariant_subset_for_root_node;
}



classification_base_case *poset_classification::get_Base_case()
{
	return Base_case;
}

int poset_classification::node_has_schreier_vector(
		int node_idx)
{
	if (Poo->get_node(node_idx)->has_Schreier_vector()) {
		return true;
	}
	else {
		return false;
	}
}

int poset_classification::max_number_of_orbits_to_print()
{
	return Control->downstep_orbits_print_max_orbits;
}

int poset_classification::max_number_of_points_to_print_in_orbit()
{
	return Control->downstep_orbits_print_max_points_per_orbit;
}

void poset_classification::invoke_early_test_func(
		long int *the_set, int lvl,
		long int *candidates,
		int nb_candidates,
		long int *good_candidates,
		int &nb_good_candidates,
		int verbose_level)
{
	Poset->early_test_func(
			the_set, lvl,
			candidates,
			nb_candidates,
			good_candidates,
			nb_good_candidates,
			verbose_level - 2);

}

int poset_classification::nb_orbits_at_level(
		int level)
{
	return Poo->nb_orbits_at_level(level);
}

long int poset_classification::nb_flag_orbits_up_at_level(
		int level)
{
	return Poo->nb_flag_orbits_up_at_level(level);
}

poset_orbit_node *poset_classification::get_node_ij(
		int level, int node)
{
	return Poo->get_node_ij(level, node);
}

int poset_classification::poset_structure_is_contained(
		long int *set1, int sz1, long int *set2, int sz2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_contained;
	int i, rk1, rk2;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "poset_classification::poset_structure_is_contained" << endl;
	}
	if (f_vv) {
		cout << "set1: ";
		Lint_vec_print(cout, set1, sz1);
		cout << " ; ";
		cout << "set2: ";
		Lint_vec_print(cout, set2, sz2);
		cout << endl;
	}
	if (sz1 > sz2) {
		f_contained = false;
	}
	else {
		if (Poset->f_subspace_lattice) {
			int *B1, *B2;
			int dim = Poset->VS->dimension;

			B1 = NEW_int(sz1 * dim);
			B2 = NEW_int((sz1 + sz2) * dim);

			for (i = 0; i < sz1; i++) {
				unrank_point(B1 + i * dim, set1[i]);
			}
			for (i = 0; i < sz2; i++) {
				unrank_point(B2 + i * dim, set2[i]);
			}

			rk1 = Poset->VS->F->Linear_algebra->Gauss_easy(B1, sz1, dim);
			if (rk1 != sz1) {
				cout << "poset_classification::poset_structure_is_contained "
						"rk1 != sz1" << endl;
				exit(1);
			}
			
			rk2 = Poset->VS->F->Linear_algebra->Gauss_easy(B2, sz2, dim);
			if (rk2 != sz2) {
				cout << "poset_classification::poset_structure_is_contained "
						"rk2 != sz2" << endl;
				exit(1);
			}
			Int_vec_copy(B1,
					B2 + sz2 * dim,
					sz1 * dim);
			rk2 = Poset->VS->F->Linear_algebra->Gauss_easy(B2, sz1 + sz2, dim);
			if (rk2 > sz2) {
				f_contained = false;
			}
			else {
				f_contained = true;
			}

			FREE_int(B1);
			FREE_int(B2);
		}
		else {
			f_contained = Sorting.lint_vec_sort_and_test_if_contained(
					set1, sz1, set2, sz2);
		}
	}
	return f_contained;
}

data_structures_groups::orbit_transversal *poset_classification::get_orbit_transversal(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::orbit_transversal *T;
	int orbit_at_level;

	if (f_v) {
		cout << "poset_classification::get_orbit_transversal" << endl;
	}
	T = NEW_OBJECT(data_structures_groups::orbit_transversal);
	T->A = Poset->A;
	T->A2 = Poset->A2;


	T->nb_orbits = nb_orbits_at_level(level);


	if (f_v) {
		cout << "poset_classification::get_orbit_transversal "
				"processing " << T->nb_orbits
				<< " orbit representatives" << endl;
	}


	T->Reps = NEW_OBJECTS(data_structures_groups::set_and_stabilizer, T->nb_orbits);

	for (orbit_at_level = 0;
			orbit_at_level < T->nb_orbits;
			orbit_at_level++) {

		data_structures_groups::set_and_stabilizer *SaS;

		SaS = get_set_and_stabilizer(level,
				orbit_at_level, verbose_level);



		T->Reps[orbit_at_level].init_everything(
				Poset->A, Poset->A2, SaS->data, level,
				SaS->Strong_gens, 0 /* verbose_level */);

		SaS->data = NULL;
		SaS->Strong_gens = NULL;

		FREE_OBJECT(SaS);

	}



	if (f_v) {
		cout << "poset_classification::get_orbit_transversal done" << endl;
	}
	return T;
}

int poset_classification::test_if_stabilizer_is_trivial(
		int level, int orbit_at_level, int verbose_level)
{
	poset_orbit_node *O;

	O = get_node_ij(level, orbit_at_level);
	return O->test_if_stabilizer_is_trivial();
}

data_structures_groups::set_and_stabilizer *poset_classification::get_set_and_stabilizer(
		int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::set_and_stabilizer *SaS;

	if (f_v) {
		cout << "poset_classification::get_set_and_stabilizer" << endl;
	}

	SaS = NEW_OBJECT(data_structures_groups::set_and_stabilizer);

	SaS->init(Poset->A, Poset->A2, 0 /*verbose_level */);

	SaS->allocate_data(level, 0 /* verbose_level */);

	get_set_by_level(level, orbit_at_level, SaS->data);

	get_stabilizer_generators(SaS->Strong_gens,
		level, orbit_at_level, 0 /* verbose_level */);

	SaS->Strong_gens->group_order(SaS->target_go);

	SaS->Stab = SaS->Strong_gens->create_sims(0 /*verbose_level*/);
	if (f_v) {
		cout << "poset_classification::get_set_and_stabilizer done" << endl;
	}
	return SaS;
}

void poset_classification::get_set_by_level(
		int level, int node, long int *set)
{
	int size;
	poset_orbit_node *O;
	
	O = get_node_ij(level, node);
	size = O->depth_of_node(this);
	if (size != level) {
		cout << "poset_classification::get_set_by_level "
				"size != level" << endl;
		exit(1);
	}
	//root[n].store_set_to(this, size - 1, set);
	O->store_set_to(this, size - 1, set);
}

void poset_classification::get_set(
		int node, long int *set, int &size)
{
	Poo->get_set(node, set, size);
}

void poset_classification::get_set(
		int level, int orbit, long int *set, int &size)
{
	Poo->get_set(level, orbit, set, size);
}

int poset_classification::find_poset_orbit_node_for_set(
		int len,
		long int *set, int f_tolerant, int verbose_level)
// finds the node that represents s_0,...,s_{len - 1}
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "poset_classification::find_poset_orbit_node_for_set ";
		Lint_vec_print(cout, set, len);
		cout << endl;
	}
	if (f_base_case) {
		int i, j, h;
		if (len < Base_case->size) {
			cout << "poset_classification::find_poset_orbit_node_for_set "
					"len < starter_size" << endl;
			cout << "len=" << len << endl;
			exit(1);
		}
		for (i = 0; i < Base_case->size; i++) {
			for (j = i; j < len; j++) {
				if (set[j] == Base_case->orbit_rep[i]) {
					if (f_v) {
						cout << "found " << i << "-th element "
								"of the starter which is " << Base_case->orbit_rep[i]
							<< " at position " << j << endl;
					}
					break;
				}
			}
			if (j == len) {
				cout << "poset_classification::find_poset_orbit_node_for_set "
						"did not find " << i << "-th element "
						"of the starter" << endl;
			}
			for (h = j; h > i; h--) {
				set[h] = set[h - 1];
			}
			set[i] = Base_case->orbit_rep[i];
		}
		int from = Base_case->size;
		int node = Base_case->size;
		ret = find_poset_orbit_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
	}
	else {
		int from = 0;
		int node = 0;
		ret = find_poset_orbit_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
	}
	if (ret == -1) {
		if (f_tolerant) {
			if (f_v) {
				cout << "poset_classification::find_poset_orbit_node_for_set ";
				Lint_vec_print(cout, set, len);
				cout << " extension not found, "
						"we are tolerant, returnning -1" << endl;
			}
			return -1;
		}
		else {
			cout << "poset_classification::find_poset_orbit_node_for_set "
					"we should not be here" << endl;
			exit(1);
		}
	}
	return ret;
	
}

int poset_classification::find_poset_orbit_node_for_set_basic(
		int from,
		int node, int len, long int *set, int f_tolerant,
		int verbose_level)
{
	int i, j;
	long int pt;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);

	if (f_vv) {
		cout << "poset_classification::"
				"find_poset_orbit_node_for_set_basic "
				"looking for set ";
		Lint_vec_print(cout, set, len);
		cout << endl;
		cout << "node=" << node << endl;
		cout << "from=" << from << endl;
		cout << "len=" << len << endl;
		cout << "f_tolerant=" << f_tolerant << endl;
	}
	for (i = from; i < len; i++) {
		pt = set[i];
		if (f_vv) {
			cout << "pt=" << pt << endl;
			cout << "calling root[node].find_extension_from_point" << endl;
		}
		j = Poo->find_extension_from_point(
				node, pt, 0 /* verbose_level */);

		//j = root[node].find_extension_from_point(this, pt, false);

		if (j == -1) {
			if (f_v) {
				cout << "poset_classification::"
						"find_poset_orbit_node_for_set_basic "
						"depth " << i << " no extension for point "
						<< pt << " found" << endl;
			}
			if (f_tolerant) {
				if (f_v) {
					cout << "poset_classification::"
							"find_poset_orbit_node_for_set_basic "
							"since we are tolerant, we return -1" << endl;
				}
				return -1;
			}
			else {
				cout << "poset_classification::"
						"find_poset_orbit_node_for_set_basic "
						"failure in find_extension_from_point" << endl;
				Lint_vec_print(cout, set, len);
				cout << endl;
				cout << "node=" << node << endl;
				cout << "from=" << from << endl;
				cout << "i=" << i << endl;
				cout << "pt=" << pt << endl;
				Poo->get_node(node)->print_extensions(this);
				exit(1);
			}
		}
		if (Poo->get_node(node)->get_E(j)->get_pt() != pt) {
			cout << "poset_classification::"
					"find_poset_orbit_node_for_set_basic "
					"root[node].E[j].pt != pt" << endl;
			exit(1);
		}
		if (Poo->get_node(node)->get_E(j)->get_type() != EXTENSION_TYPE_EXTENSION &&
				Poo->get_node(node)->get_E(j)->get_type() != EXTENSION_TYPE_PROCESSING) {
			cout << "poset_classification::"
					"find_poset_orbit_node_for_set_basic "
					"root[node].get_E(j)->type != "
					"EXTENSION_TYPE_EXTENSION" << endl;
			cout << "root[node].get_E(j)->type="
					<< Poo->get_node(node)->get_E(j)->get_type() << " = ";
			print_extension_type(cout, Poo->get_node(node)->get_E(j)->get_type());
			cout << endl;
			cout << "poset_classification::"
					"find_poset_orbit_node_for_set_basic "
					"looking for set ";
			Lint_vec_print(cout, set, len);
			cout << endl;
			cout << "node=" << node << endl;
			cout << "from=" << from << endl;
			cout << "i=" << i << endl;
			cout << "node=" << node << endl;
			cout << "f_tolerant=" << f_tolerant << endl;
			cout << "node=" << node << endl;
			cout << "pt=" << pt << endl;
			cout << "j=" << j << endl;
			exit(1);
		}
		node = Poo->get_node(node)->get_E(j)->get_data();
		if (f_v) {
			cout << "depth " << i << " extension " << j
					<< " new node " << node << endl;
		}
	}
	return node;
}


long int poset_classification::count_extension_nodes_at_level(
		int lvl)
{
	return Poo->count_extension_nodes_at_level(lvl);
}

double poset_classification::level_progress(
		int lvl)
{
	return Poo->level_progress(lvl);
}



void poset_classification::count_automorphism_group_orders(
	int lvl, int &nb_agos,
	algebra::ring_theory::longinteger_object *&agos,
	int *&multiplicities,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l, j, c, h, f_added;
	algebra::ring_theory::longinteger_object ago;
	algebra::ring_theory::longinteger_object *tmp_agos;
	int *tmp_multiplicities;
	algebra::ring_theory::longinteger_domain D;
	
	l = nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "collecting the automorphism group orders of "
				<< l << " orbits" << endl;
	}
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
	for (i = 0; i < l; i++) {
		get_stabilizer_order(lvl, i, ago);
		f_added = false;
		for (j = 0; j < nb_agos; j++) {
			c = D.compare_unsigned(ago, agos[j]);
			//cout << "comparing " << ago << " with " << agos[j]
			// << " yields " << c << endl;
			if (c >= 0) {
				if (c == 0) {
					multiplicities[j]++;
				}
				else {
					tmp_agos = agos;
					tmp_multiplicities = multiplicities;
					agos = NEW_OBJECTS(algebra::ring_theory::longinteger_object, nb_agos + 1);
					multiplicities = NEW_int(nb_agos + 1);
					for (h = 0; h < j; h++) {
						tmp_agos[h].swap_with(agos[h]);
						multiplicities[h] = tmp_multiplicities[h];
					}
					ago.swap_with(agos[j]);
					multiplicities[j] = 1;
					for (h = j; h < nb_agos; h++) {
						tmp_agos[h].swap_with(agos[h + 1]);
						multiplicities[h + 1] = tmp_multiplicities[h];
					}
					nb_agos++;
					if (tmp_agos) {
						FREE_OBJECTS(tmp_agos);
						FREE_int(tmp_multiplicities);
					}
				}
				f_added = true;
				break;
			}
		}
		if (!f_added) {
			// add at the end (including the case that the list is empty)
			tmp_agos = agos;
			tmp_multiplicities = multiplicities;
			agos = NEW_OBJECTS(algebra::ring_theory::longinteger_object, nb_agos + 1);
			multiplicities = NEW_int(nb_agos + 1);
			for (h = 0; h < nb_agos; h++) {
				tmp_agos[h].swap_with(agos[h]);
				multiplicities[h] = tmp_multiplicities[h];
			}
			ago.swap_with(agos[nb_agos]);
			multiplicities[nb_agos] = 1;
			nb_agos++;
			if (tmp_agos) {
				FREE_OBJECTS(tmp_agos);
				FREE_int(tmp_multiplicities);
			}
		}
	}
}

void poset_classification::compute_and_print_automorphism_group_orders(
		int lvl, std::ostream &ost)
{

	int j, nb_agos;
	algebra::ring_theory::longinteger_object *agos;
	int *multiplicities;
	int N, r, h;
	algebra::ring_theory::longinteger_object S, S1, Q;
	algebra::ring_theory::longinteger_domain D;
	
	count_automorphism_group_orders(lvl, nb_agos, agos,
			multiplicities, false);
	S.create(0);
	N = 0;
	for (j = 0; j < nb_agos; j++) {
		N += multiplicities[j];
		for (h = 0; h < multiplicities[j]; h++) {
			D.add(S, agos[j], S1);
			S1.assign_to(S);
		}
	}
	D.integral_division_by_int(S, N, Q, r);
	

	ost << "(";
	for (j = 0; j < nb_agos; j++) {
		ost << agos[j];
		if (multiplicities[j] == 1) {
		}
		else if (multiplicities[j] >= 10) {
			ost << "^{" << multiplicities[j] << "}";
		}
		else  {
			ost << "^" << multiplicities[j];
		}
		if (j < nb_agos - 1) {
			ost << ", ";
		}
	}
	ost << ") average is " << Q << " + " << r << " / " << N << endl;
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
	}
}

void poset_classification::stabilizer_order(
		int node, algebra::ring_theory::longinteger_object &go)
{
#if 0
	if (root[node].get_nb_strong_generators()) {
		go.create_product(Poset->A->base_len(), root[node].tl);
	}
	else {
		go.create(1, __FILE__, __LINE__);
	}
#else
	Poo->get_node(node)->get_stabilizer_order(this, go);
#endif
}


void poset_classification::orbit_length(
		int orbit_at_level,
		int level, algebra::ring_theory::longinteger_object &len)
// uses poset_classification::go for the group order
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object stab_order, quo, rem;

	get_stabilizer_order(level, orbit_at_level, stab_order);
	D.integral_division(Poset->go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "poset_classification::orbit_length stabilizer order does "
				"not divide group order" << endl;
		exit(1);
	}
}

void poset_classification::get_orbit_length_and_stabilizer_order(
		int node,
		int level, algebra::ring_theory::longinteger_object &stab_order,
		algebra::ring_theory::longinteger_object &len)
// uses poset_classification::go for the group order
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object quo, rem;

	get_stabilizer_order(level, node, stab_order);
	D.integral_division(Poset->go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "poset_classification::orbit_length "
				"stabilizer order "
				"does not divide group order" << endl;
		exit(1);
	}
}

int poset_classification::orbit_length_as_int(
		int orbit_at_level, int level)
{
	algebra::ring_theory::longinteger_object len;

	orbit_length(orbit_at_level, level, len);
	return len.as_int();
	
}


void poset_classification::recreate_schreier_vectors_up_to_level(
		int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "poset_classification::recreate_schreier_vectors_up_to_level "
				"creating Schreier vectors up to "
				"level " << lvl << endl;
	}
	for (i = 0; i <= lvl; i++) {
		if (f_v) {
			cout << "poset_classification::recreate_schreier_vectors_up_to_level "
					"creating Schreier vectors at "
					"level " << i << endl;
		}
		recreate_schreier_vectors_at_level(i, verbose_level - 1);
	}
	if (f_v) {
		cout << "poset_classification::recreate_schreier_vectors_up_to_level done" << endl;
	}
}

void poset_classification::recreate_schreier_vectors_at_level(
		int level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = false;//(verbose_level >= 3);
	int f, l, prev, u;
	int f_recreate_extensions = false;
	int f_dont_keep_sv = false;

	if (f_v) {
		cout << "poset_classification::recreate_schreier_vectors_at_level "
				"level = " << level << endl;
	}
	f = Poo->first_node_at_level(level);
	if (f_v) {
		cout << "poset_classification::recreate_schreier_vectors_at_level "
				"f = " << f << endl;
	}
	//cur = Poo->first_node_at_level(level + 1);
	//l = cur - f;
	l = Poo->nb_orbits_at_level(level);

	if (f_vv) {
		cout << "creating Schreier vectors at depth " << level
				<< " for " << l << " orbits" << endl;
	}
	if (f_vv) {
		cout << "poset_classification::recreate_schreier_vectors_at_level "
				"Testing if a schreier vector file exists" << endl;
	}
	if (test_sv_level_file_binary(level, problem_label_with_path)) {

		if (f_vv) {
			cout << "poset_classification::recreate_schreier_vectors_at_level "
					"Yes, a schreier vector file exists. "
					"We will read this file" << endl;
		}

		read_sv_level_file_binary(level, problem_label_with_path, false, 0, 0,
			f_recreate_extensions, f_dont_keep_sv, 
			verbose_level - 2);
		if (f_vv) {
			cout << "read Schreier vectors at depth " << level
					<< " from file" << endl;
		}
		return;
	}


	if (f_vv) {
		cout << "poset_classification::recreate_schreier_vectors_at_level "
				"No, a schreier vector file does not exist. "
				"We will create such a file now" << endl;
	}



	for (u = 0; u < l; u++) {
			
		prev = f + u;
			
		if (f_vv && !f_vvv) {
			cout << ".";
			if (((u + 1) % 50) == 0) {
				cout << "; " << u + 1 << " / " << l << endl;
			}
			if (((u + 1) % 1000) == 0) {
				cout << " " << u + 1 << endl;
			}
		}
		if (f_vv) {
			cout << "poset_classification::recreate_schreier_vectors_at_level "
				<< level << " node " << u << " / " << l
				<< " before compute_schreier_vector" << endl;
		}
			
		Poo->get_node(prev)->compute_schreier_vector(
				this, level,
				verbose_level - 1);
	}
	write_sv_level_file_binary(
			level, problem_label_with_path, false, 0, 0,
			verbose_level);
	if (f_vv) {
		cout << "poset_classification::recreate_schreier_vectors_at_level "
				"Written a file with Schreier "
				"vectors at depth " << level << endl;
	}
	if (f_vv) {
		cout << endl;
	}
	if (f_v) {
		cout << "poset_classification::recreate_schreier_vectors_at_level done" << endl;
	}
}


void poset_classification::find_node_by_stabilizer_order(
		int level, int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_nodes, node, i, j, elt_order;
	algebra::ring_theory::longinteger_object ago;
	long int set[300];
	
	if (f_v) {
		cout << "poset_classification::find_node_by_stabilizer_order" << endl;
	}
	nb_nodes = nb_orbits_at_level(level);
	for (i = 0; i < nb_nodes; i++) {
		node = Poo->first_node_at_level(level) + i;

		Poo->get_node(node)->get_stabilizer_order(this, ago);

		if (ago.as_int() == order) {
			cout << "found a node whose automorphism group is order "
					<< order << endl;
			cout << "the node is # " << i << " at level "
					<< level << endl;
			get_set(Poo->first_node_at_level(level) + i,
					set, level);
			Lint_vec_print(cout, set, level);
			cout << endl;
			
			groups::strong_generators *Strong_gens;
			
			get_stabilizer_generators(Strong_gens,
				level, i, 0  /* verbose_level */);
				
			for (j = 0; j < Strong_gens->gens->len; j++) {
				elt_order = Poset->A->Group_element->element_order(
						Strong_gens->gens->ith(j));
				cout << "poset_classification " << j << " of order "
						<< elt_order << ":" << endl;
				if (order == elt_order) {
					cout << "CYCLIC" << endl;
					}
				Poset->A->Group_element->element_print(
						Strong_gens->gens->ith(j), cout);
				Poset->A->Group_element->element_print_as_permutation(
						Strong_gens->gens->ith(j), cout);
			}
			FREE_OBJECT(Strong_gens);
		}
	}
}

void poset_classification::get_all_stabilizer_orders_at_level(
		int level,
		long int *&Ago, int &nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::get_all_stabilizer_orders_at_level, "
				"level = " << level << endl;
	}
	int i;

	nb = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_classification::get_all_stabilizer_orders_at_level "
				"nb = " << nb << endl;
	}
	Ago = NEW_lint(nb);
	for (i = 0; i < nb; i++) {
		Ago[i] = get_stabilizer_order_lint(level, i);
	}
	if (f_v) {
		cout << "poset_classification::get_all_stabilizer_orders_at_level done" << endl;
	}
}

void poset_classification::get_stabilizer_order(
		int level,
		int orbit_at_level, algebra::ring_theory::longinteger_object &go)
{
	poset_orbit_node *O;

	O = get_node_ij(level, orbit_at_level);


#if 0
	if (O->nb_strong_generators == 0) {
		go.create(1, __FILE__, __LINE__);
	}
	else {
		longinteger_domain D;

		D.multiply_up(go, O->tl, Poset->A->base_len(), 0 /* verbose_level */);
	}
#else
	O->get_stabilizer_order(this, go);
#endif
}

long int poset_classification::get_stabilizer_order_lint(
		int level,
		int orbit_at_level)
{
	poset_orbit_node *O;

	O = get_node_ij(level, orbit_at_level);
	return O->get_stabilizer_order_lint(this);
}

void poset_classification::get_stabilizer_group(
		data_structures_groups::group_container *&G,
	int level, int orbit_at_level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	poset_orbit_node *O;
	//int node;

	if (f_v) {
		cout << "poset_classification::get_stabilizer_group "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
	}

	O = get_node_ij(level, orbit_at_level);


#if 0
	G = NEW_OBJECT(group);
	//node = first_poset_orbit_node_at_level[level] + orbit_at_level;
	//O = root + node;

	G->init(Poset->A, verbose_level - 2);
	if (f_vv) {
		cout << "poset_classification::"
				"get_stabilizer_group before "
				"G->init_strong_generators_by_hdl" << endl;
	}
	G->init_strong_generators_by_hdl(O->nb_strong_generators,
			O->hdl_strong_generators, O->tl, false);
	G->schreier_sims(0);
#else
	algebra::ring_theory::longinteger_object go;

	G = NEW_OBJECT(data_structures_groups::group_container);
	O->get_stabilizer(
		this,
		*G, go,
		verbose_level - 2);
#endif
	
	if (f_v) {
		cout << "poset_classification::get_stabilizer_group "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
	}
}

void poset_classification::get_stabilizer_generators_cleaned_up(
		groups::strong_generators *&gens,
	int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::"
				"get_stabilizer_generators_cleaned_up "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
	}
	data_structures_groups::group_container *G;

	get_stabilizer_group(G,
			level, orbit_at_level, verbose_level - 1);

	gens = NEW_OBJECT(groups::strong_generators);

	gens->init_from_sims(G->S, 0 /* verbose_level */);
	FREE_OBJECT(G);
	if (f_v) {
		cout << "poset_classification::"
				"get_stabilizer_generators_cleaned_up "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
	}

}

void poset_classification::get_stabilizer_generators(
		groups::strong_generators *&gens,
	int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::get_stabilizer_generators "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
	}

	poset_orbit_node *O;
	//int node;

	//node = first_poset_orbit_node_at_level[level] + orbit_at_level;
	//O = root + node;
	O = get_node_ij(level, orbit_at_level);

	O->get_stabilizer_generators(this, gens, verbose_level);

	if (f_v) {
		cout << "poset_classification::get_stabilizer_generators "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
	}
}



void poset_classification::orbit_element_unrank(
		int depth,
		int orbit_idx, long int rank, long int *set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	int *Elt2;
	long int *the_set;
	poset_orbit_node *O;


	if (f_v) {
		cout << "poset_classification::orbit_element_unrank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx
				<< " rank=" << rank << endl;
	}

	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	Elt2 = NEW_int(Poset->A->elt_size_in_int);
	the_set = NEW_lint(depth);
	
	//O = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	O = get_node_ij(
			depth, orbit_idx);
	coset_unrank(
			depth, orbit_idx, rank, Elt1, 0 /*verbose_level*/);

	Poset->A->Group_element->element_invert(
			Elt1, Elt2, 0);
	O->store_set_to(this, depth - 1, the_set);
	Poset->A2->Group_element->map_a_set(
			the_set, set, depth, Elt2,
			0 /*verbose_level*/);

	FREE_lint(the_set);
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "poset_classification::orbit_element_unrank ";
		Lint_vec_print(cout, set, depth);
		cout << endl;
	}
}

void poset_classification::orbit_element_rank(
	int depth,
	int &orbit_idx, long int &rank, long int *set,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	long int *the_set;
	long int *canonical_set;
	int i;


	if (f_v) {
		cout << "poset_classification::orbit_element_rank "
				"depth=" << depth << " ";
		Lint_vec_print(cout, set, depth);
		cout << endl;
	}

	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	the_set = NEW_lint(depth);
	canonical_set = NEW_lint(depth);
	for (i = 0; i < depth; i++) {
		the_set[i] = set[i];
	}
	
	orbit_idx = trace_set(the_set, depth, depth, 
		canonical_set, Elt1, 
		verbose_level - 3);

	// now Elt1 is the transporter element that moves 
	// the given set to the orbit representative

	if (f_vv) {
		cout << "poset_classification::orbit_element_rank "
				"after trace_set, "
				"orbit_idx = " << orbit_idx << endl;
		cout << "transporter:" << endl;
		Poset->A->Group_element->element_print_quick(Elt1, cout);
		cout << "as permutation:" << endl;
		Poset->A2->Group_element->element_print_as_permutation(Elt1, cout);
	}
	if (f_v) {
		cout << "calling coset_rank" << endl;
	}
	rank = coset_rank(depth, orbit_idx, Elt1, verbose_level);
	if (f_v) {
		cout << "after coset_rank, rank=" << rank << endl;
	}
		
	FREE_int(Elt1);
	FREE_lint(the_set);
	FREE_lint(canonical_set);
	if (f_v) {
		cout << "poset_classification::orbit_element_rank "
				"orbit_idx="
				<< orbit_idx << " rank=" << rank << endl;
	}
}

void poset_classification::coset_unrank(
		int depth, int orbit_idx,
		long int rank, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *the_set;
	data_structures_groups::group_container *G1, *G2;
	int *Elt_gk;
	algebra::ring_theory::longinteger_object G_order, U_order;
	poset_orbit_node *O1, *O2;

	if (f_v) {
		cout << "poset_classification::coset_unrank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		Poset->A->print_info();
		cout << "action A2:" << endl;
		Poset->A2->print_info();
	}

	//O1 = &root[0];
	//O2 = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	O1 = get_node_ij(0, 0);
	O2 = get_node_ij(depth, orbit_idx);


	
	G1 = NEW_OBJECT(data_structures_groups::group_container);
	G2 = NEW_OBJECT(data_structures_groups::group_container);
	the_set = NEW_lint(depth);
	Elt_gk = NEW_int(Poset->A->elt_size_in_int);
	
	O2->store_set_to(this, depth - 1, the_set);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		Lint_vec_print(cout, the_set, depth);
		cout << endl;
	}
	
	O1->get_stabilizer(this, *G1, G_order, verbose_level - 2);
	O2->get_stabilizer(this, *G2, U_order, verbose_level - 2);


	Poset->A->coset_unrank(G1->S, G2->S, rank, Elt, verbose_level);

	FREE_OBJECT(G1);
	FREE_OBJECT(G2);
	FREE_lint(the_set);
	FREE_int(Elt_gk);

}

long int poset_classification::coset_rank(
		int depth, int orbit_idx,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rank;
	long int *the_set;
	data_structures_groups::group_container *G1, *G2;
	int *Elt_gk;
	algebra::ring_theory::longinteger_object G_order, U_order;
	poset_orbit_node *O1, *O2;

	if (f_v) {
		cout << "poset_classification::coset_rank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		Poset->A->print_info();
		cout << "action A2:" << endl;
		Poset->A2->print_info();
	}

	//O1 = &root[0];
	//O2 = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	O1 = get_node_ij(0, 0);
	O2 = get_node_ij(depth, orbit_idx);


	
	G1 = NEW_OBJECT(data_structures_groups::group_container);
	G2 = NEW_OBJECT(data_structures_groups::group_container);
	the_set = NEW_lint(depth);
	Elt_gk = NEW_int(Poset->A->elt_size_in_int);
	
	O2->store_set_to(this, depth - 1, the_set);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		Lint_vec_print(cout, the_set, depth);
		cout << endl;
	}
	
	O1->get_stabilizer(this, *G1, G_order, verbose_level - 2);
	O2->get_stabilizer(this, *G2, U_order, verbose_level - 2);


	rank = Poset->A->coset_rank(
			G1->S, G2->S, Elt, verbose_level);

	FREE_OBJECT(G1);
	FREE_OBJECT(G2);
	FREE_lint(the_set);
	FREE_int(Elt_gk);
	
	return rank;
}

void poset_classification::list_all_orbits_at_level(
	int depth,
	int f_has_print_function, 
	void (*print_function)(
			std::ostream &ost,
			int len, long int *S, void *data),
	void *print_function_data, 
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int l, i;

	l = nb_orbits_at_level(depth);

	cout << "poset_classification::list_all_orbits_at_level "
			"listing all orbits "
			"at depth " << depth << ":" << endl;
	for (i = 0; i < l; i++) {
		cout << "poset_classification::list_all_orbits_at_level "
			"listing orbit "
			<< i << " / " << l << endl;
		list_whole_orbit(depth, i, 
			f_has_print_function, print_function, print_function_data, 
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
	}
}

void poset_classification::compute_integer_property_of_selected_list_of_orbits(
	int depth,
	int nb_orbits, int *Orbit_idx, 
	int (*compute_function)(
			int len, long int *S, void *data),
	void *compute_function_data,
	int *&Data)
{
	int l, i, j, d;
	long int *set;

	set = NEW_lint(depth);
	l = nb_orbits_at_level(depth);

	Data = NEW_int(nb_orbits);
	
	cout << "computing integer property for a set of "
			<< nb_orbits << " orbits at "
			"depth " << depth << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {
		i = Orbit_idx[j];
		if (i >= l) {
			cout << "orbit idx is out of range" << endl;
			exit(1);
		}
		cout << "Orbit " << j << " / " << nb_orbits
				<< " which is no " << i << ":" << endl;

		get_set_by_level(depth, i, set);

		d = (*compute_function)(depth, set, compute_function_data);
		Data[j] = d;
	}

	FREE_lint(set);
}

void poset_classification::list_selected_set_of_orbits_at_level(
	int depth,
	int nb_orbits, int *Orbit_idx, 
	int f_has_print_function, 
	void (*print_function)(std::ostream &ost,
			int len, long int *S, void *data),
	void *print_function_data, 
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int l, i, j;

	l = nb_orbits_at_level(depth);

	cout << "listing a set of " << nb_orbits
			<< " orbits at depth " << depth << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {
		i = Orbit_idx[j];
		if (i >= l) {
			cout << "orbit idx is out of range" << endl;
			exit(1);
		}
		cout << "Orbit " << j << " / " << nb_orbits
				<< " which is no " << i << ":" << endl;
		list_whole_orbit(depth, i, 
			f_has_print_function, print_function, print_function_data, 
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
	}
}

void poset_classification::test_property(
		int depth,
	int (*test_property_function)(
			int len, long int *S, void *data),
	void *test_property_data, 
	int &nb, int *&Orbit_idx)
{
	int N, i;
	long int *set;

	set = NEW_lint(depth);
	N = nb_orbits_at_level(depth);
	Orbit_idx = NEW_int(N);
	nb = 0;
	for (i = 0; i < N; i++) {
		get_set_by_level(depth, i, set);
		if ((*test_property_function)(depth, set, test_property_data)) {
			Orbit_idx[nb++] = i;
		}
	}
	FREE_lint(set);
}

#if 0
void poset_classification::print_schreier_vectors_at_depth(
		int depth, int verbose_level)
{
	int i, l;

	l = nb_orbits_at_level(depth);
	for (i = 0; i < l; i++) {
		print_schreier_vector(depth, i, verbose_level);
		}
}

void poset_classification::print_schreier_vector(int depth,
		int orbit_idx, int verbose_level)
{
	int *set;
	int len;
	//strong_generators *Strong_gens;
	longinteger_object Len, L, go;
	//longinteger_domain D;
	
	set = NEW_int(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_int(orbit_idx, depth);
	L.create(len);
	
	get_stabilizer_order(depth, orbit_idx, go);


	cout << "orbit " << orbit_idx << " / " << nb_orbits_at_level(depth)
			<< " (=node " << first_poset_orbit_node_at_level[depth] + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	int_set_print(cout, set, depth);
	cout << "_" << go << endl;

	cout << "schreier tree:" << endl;

	int *sv;


	sv = root[first_poset_orbit_node_at_level[depth] + orbit_idx].sv;

	if (sv == NULL) {
		cout << "No schreier vector available" << endl;
		}

	schreier_vector_print_tree(sv, 0 /*verbose_level */);
}
#endif

void poset_classification::list_whole_orbit(
	int depth, int orbit_idx,
	int f_has_print_function, 
	void (*print_function)(std::ostream &ost,
			int len, long int *S, void *data),
	void *print_function_data, 
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	long int *set;
	int rank, len;
	groups::strong_generators *Strong_gens;
	algebra::ring_theory::longinteger_object Len, L, go;
	algebra::ring_theory::longinteger_domain D;
	
	set = NEW_lint(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_int(orbit_idx, depth);
	L.create(len);
	
	get_stabilizer_order(depth, orbit_idx, go);


	cout << "poset_classification::list_whole_orbit "
			"depth " << depth
			<< " orbit " << orbit_idx
			<< " / " << nb_orbits_at_level(depth)
			<< " (=node " << Poo->first_node_at_level(depth) + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	Lint_vec_print(cout, set, depth);
	cout << "_" << go << " ";

	//print_lex_rank(set, depth);
	cout << endl;

	if (f_has_print_function) {
		(*print_function)(cout, depth, set, print_function_data);
	}

	get_stabilizer_generators(Strong_gens, depth, orbit_idx, 0 /* verbose_level*/);


	if (f_show_orbit_decomposition) {
		if (Poset->f_subset_lattice) {
			cout << "poset_classification::list_whole_orbit "
					"orbits on the set:" << endl;

			// ToDo:
			//Strong_gens->compute_and_print_orbits_on_a_given_set(
			//		Poset->A2, set, depth, 0 /* verbose_level*/);
		}
		else {
			cout << "subspace_lattice not yet implemented" << endl;
		}
	
		cout << "poset_classification::list_whole_orbit "
				"orbits in the original "
				"action on the whole space:" << endl;
		Strong_gens->compute_and_print_orbits(Poset->A,
				0 /* verbose_level*/);
	}
	
	if (f_show_stab) {
		cout << "The stabilizer is generated by:" << endl;
		Strong_gens->print_generators(cout, 0 /* verbose_level*/);
	}

	if (f_save_stab) {
		string fname;

		fname = problem_label_with_path + "_stab_" + std::to_string(depth) + "_" + std::to_string(orbit_idx) + ".bin";

		cout << "saving stabilizer poset_classifications "
				"to file " << fname << endl;
		Strong_gens->write_file(fname, Control->verbose_level);
	}


	if (f_show_whole_orbit) {
		int max_len;
		if (len > 1000) {
			max_len = 10;
		}
		else {
			max_len = len;
		}

		if (D.compare(L, Len) != 0) {
			cout << "orbit is too long to show" << endl;
		}
		else {
			for (rank = 0; rank < max_len; rank++) {
				orbit_element_unrank(depth, orbit_idx,
						rank, set, 0 /* verbose_level */);
				cout << setw(5) << rank << " : ";
				Lint_vec_set_print(
						cout, set, depth);
				cout << endl;
			}
			if (max_len < len) {
				cout << "output truncated" << endl;
			}
		}
	}

	FREE_lint(set);
	FREE_OBJECT(Strong_gens);
	cout << "poset_classification::list_whole_orbit done" << endl;
}

void poset_classification::get_whole_orbit(
	int depth, int orbit_idx,
	long int *&Orbit, int &orbit_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rank;
	algebra::ring_theory::longinteger_object Len, L, go;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "poset_classification::get_whole_orbit" << endl;
	}
	poset_classification::orbit_length(orbit_idx, depth, Len);
	orbit_length = orbit_length_as_int(orbit_idx, depth);
	L.create(orbit_length);

	if (f_v) {
		cout << "poset_classification::get_whole_orbit "
				"orbit_length=" << orbit_length << endl;
	}
	if (D.compare(L, Len) != 0) {
		cout << "poset_classification::get_whole_orbit "
				"orbit is too long" << endl;
		exit(1);
	}

	Orbit = NEW_lint(orbit_length * depth);
	for (rank = 0; rank < orbit_length; rank++) {
		if (f_v) {
			cout << "poset_classification::get_whole_orbit "
					"element " << rank << " / " << orbit_length << endl;
		}
		orbit_element_unrank(
				depth, orbit_idx,
				rank,
				Orbit + rank * depth,
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "poset_classification::get_whole_orbit done" << endl;
	}
}

void poset_classification::map_to_canonical_k_subset(
	long int *the_set, int set_size,
	int subset_size, int subset_rk,
	long int *reduced_set, int *transporter, int &local_idx,
	int verbose_level)
// fills reduced_set[set_size - subset_size], transporter and local_idx
// local_idx is the index of the orbit that the subset belongs to 
// (in the list of orbit of subsets of size subset_size)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_classification::map_to_canonical_k_subset" << endl;
	}
	int *our_set;
	long int *subset;
	long int *canonical_subset;
	int *Elt1;
	int i; //, j, k;
	int reduced_set_size;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	our_set = NEW_int(set_size);
	subset = NEW_lint(set_size);
	canonical_subset = NEW_lint(set_size);
	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	reduced_set_size = set_size - subset_size;

	// unrank the k-subset and its complement to our_set[set_size]:
	Combi.unrank_k_subset_and_complement(subset_rk,
			our_set, set_size, subset_size);

	if (f_v) {
		cout << "poset_classification::map_to_canonical_k_subset "
				"our_set=";
		Int_vec_print(cout, our_set, set_size);
		cout << endl;
	}

	for (i = 0; i < set_size; i++) {
		subset[i] = the_set[our_set[i]];
		//set[0][i] = subset[i]; // ToDo
	}
	for (i = 0; i < sz; i++) {
		//set[0][i] = subset[i]; // ToDo
	}
	if (f_v) {
		cout << "poset_classification::map_to_canonical_k_subset "
				"subset=";
		Lint_vec_print(cout, subset, set_size);
		cout << endl;
	}
	
	// ToDo
	//Poset->A->element_one(poset_classification::transporter->ith(0), false);


	// trace the subset:
	
	if (f_v) {
		cout << "poset_classification::map_to_canonical_k_subset "
				"before trace_set" << endl;
	}

	if (set_size > max_set_size) {
		cout << "poset_classification::map_to_canonical_k_subset "
				"set_size > max_set_size" << endl;
		cout << "poset_classification::map_to_canonical_k_subset "
				"set_size = " << set_size << endl;
		cout << "poset_classification::map_to_canonical_k_subset "
				"max_set_size = " << max_set_size << endl;
		exit(1);
	}

	local_idx = trace_set(
		subset, set_size, subset_size,
		canonical_subset, Elt1, 
		verbose_level);
	if (f_v) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset "
				"after trace_set local_idx=" << local_idx << endl;
		cout << "poset_classification::map_to_canonical_k_subset "
				"canonical_subset=";
		Lint_vec_print(cout, canonical_subset, set_size);
		cout << endl;
	}


	if (f_v) {
		cout << "the transporter is" << endl;
		Poset->A->Group_element->element_print(Elt1, cout);
		cout << endl;
	}
	Poset->A->Group_element->element_move(Elt1, transporter, false);

	for (i = 0; i < reduced_set_size; i++) {
		reduced_set[i] = canonical_subset[subset_size + i];
	}
	if (f_v) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset reduced set = ";
		Lint_vec_print(cout, reduced_set, reduced_set_size);
		cout << endl;
	}
	FREE_int(Elt1);
	FREE_int(our_set);
	FREE_lint(subset);
	FREE_lint(canonical_subset);
	
	if (f_v) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset done" << endl;
	}
}

void poset_classification::get_representative_of_subset_orbit(
	long int *set, int size, int local_orbit_no,
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int fst, node, sz;
	poset_orbit_node *O;

	if (f_v) {
		cout << "poset_classification::get_representative_of_subset_orbit "
				"verbose_level=" << verbose_level << endl;
	}
	fst = Poo->first_node_at_level(size);
	node = fst + local_orbit_no;
	if (f_vv) {
		cout << "poset_classification::get_representative_of_subset_orbit "
				"before get_set" << endl;
	}
	get_set(node, set, sz);
	if (sz != size) {
		cout << "get_representative_of_subset_orbit: "
				"sz != size" << endl;
		exit(1);
	}
	//O = root + node;
	O = get_node_ij(size, local_orbit_no);
	if (f_vv) {
		cout << "poset_classification::get_representative_of_subset_orbit "
				"before get_stabilizer_poset_classifications" << endl;
	}
	O->get_stabilizer_generators(this, Strong_gens, 0);
	if (f_v) {
		cout << "poset_classification::get_representative_of_subset_orbit done" << endl;
	}
}

void poset_classification::find_interesting_k_subsets(
	long int *the_set, int n, int k,
	int *&interesting_sets, int &nb_interesting_sets,
	int &orbit_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::tally *C;
	int j, t, f, l, l_min, t_min = 0;

	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"n = " << n << " k = " << k << endl;
	}
	

	classify_k_subsets(the_set, n, k, C, verbose_level);


	if (f_v) {
		C->print_bare(false);
		cout << endl;
	}

	l_min = INT_MAX;
	f = 0;
	for (t = 0; t < C->nb_types; t++) {
		f = C->type_first[t];
		l = C->type_len[t];
		if (l < l_min) {
			l_min = l;
			t_min = t;
		}
	}
	interesting_sets = NEW_int(l_min);
	nb_interesting_sets = l_min;
	for (j = 0; j < l_min; j++) {
		interesting_sets[j] = C->sorting_perm_inv[f + j];
	}
	orbit_idx = C->data_sorted[f];
	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"l_min = " << l_min << " t_min = " << t_min
				<< " orbit_idx = " << orbit_idx << endl;
	}
	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"interesting set of size "
				<< nb_interesting_sets << " : ";
		Int_vec_print(cout, interesting_sets, nb_interesting_sets);
		cout << endl;
	}

	FREE_OBJECT(C);
	
	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
	}
}

void poset_classification::classify_k_subsets(
		long int *the_set, int n, int k,
		other::data_structures::tally *&C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int nCk;
	int *isotype;

	if (f_v) {
		cout << "poset_classification::classify_k_subsets "
				"n = " << n << " k = " << k << endl;
	}
	
	trace_all_k_subsets(the_set, n, k, nCk, isotype, verbose_level);
	
	C = NEW_OBJECT(other::data_structures::tally);

	C->init(isotype, nCk, false, 0);

	if (f_v) {
		cout << "poset_classification::classify_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
	}
}

void poset_classification::trace_all_k_subsets_and_compute_frequencies(
		long int *the_set,
		int n, int k, int &nCk,
		int *&isotype, int *&orbit_frequencies, int &nb_orbits,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;


	if (f_v) {
		cout << "poset_classification::trace_all_k_subsets_and_compute_frequencies "
				"n = " << n << " k = " << k << " nCk=" << nCk << endl;
	}

	trace_all_k_subsets(
			the_set,
			n, k, nCk, isotype,
			verbose_level);

	nb_orbits = nb_orbits_at_level(k);
	orbit_frequencies = NEW_int(nb_orbits);
	Int_vec_zero(orbit_frequencies, nb_orbits);

	for (i = 0; i < nCk; i++) {
		a = isotype[i];
		orbit_frequencies[a]++;
	}

	if (f_v) {
		cout << "poset_classification::trace_all_k_subsets_and_compute_frequencies done" << endl;
	}
}

void poset_classification::trace_all_k_subsets(
		long int *the_set,
		int n, int k, int &nCk, int *&isotype,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);
	int *index_set;
	long int *subset;
	long int *canonical_subset;
	int *Elt;
	long int subset_rk, local_idx, i;
	//int f_implicit_fusion = true;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	nCk = Combi.int_n_choose_k(n, k);
	if (f_v) {
		cout << "poset_classification::trace_all_k_subsets "
				"n = " << n << " k = " << k
				<< " nCk = " << nCk << endl;
	}

	Elt = NEW_int(Poset->A->elt_size_in_int);

	index_set = NEW_int(k);
	subset = NEW_lint(k);
	canonical_subset = NEW_lint(k);
	isotype = NEW_int(nCk);
	
	Int_vec_zero(isotype, nCk);

	Combi.first_k_subset(index_set, n, k);
	subset_rk = 0;

	while (true) {
		if (true && ((subset_rk % 10000) == 0)) {
			cout << "poset_classification::trace_all_k_subsets "
					"k=" << k
				<< " testing set " << subset_rk << " / " << nCk 
				<< " = " << 100. * (double) subset_rk /
				(double) nCk << " % : ";
			Int_vec_print(cout, index_set, k);
			cout << endl;
		}
		for (i = 0; i < k; i++) {
			subset[i] = the_set[index_set[i]];
		}
		//Lint_vec_copy(subset, set[0], k);

		if (false /*f_v2*/) {
			cout << "poset_classification::trace_all_k_subsets "
					"corresponding to set ";
			Lint_vec_print(cout, subset, k);
			cout << endl;
		}
		//Poset->A->element_one(transporter->ith(0), 0);
		
		if (k == 0) {
			isotype[0] = 0;
		}
		else {

			if (false) {
				cout << "poset_classification::trace_all_k_subsets "
						"before trace_set" << endl;
			}
			local_idx = trace_set(subset, k, k, 
				canonical_subset, Elt, 
				0 /*verbose_level - 3*/);
			if (false) {
				cout << "poset_classification::trace_all_k_subsets "
						"after trace_set, local_idx = "
						<< local_idx << endl;
			}
			
			if (false /*f_vvv*/) {
				cout << "poset_classification::trace_all_k_subsets "
						"local_idx=" << local_idx << endl;
			}
			isotype[subset_rk] = local_idx;
			if (false) {
				cout << "poset_classification::trace_all_k_subsets "
						"the transporter is" << endl;
				Poset->A->Group_element->element_print(Elt, cout);
				cout << endl;
			}

		}
		subset_rk++;
		if (!Combi.next_k_subset(index_set, n, k)) {
			break;
		}
	}
	if (subset_rk != nCk) {
		cout << "poset_classification::trace_all_k_subsets "
				"subset_rk != nCk" << endl;
		exit(1);
	}


	FREE_int(index_set);
	FREE_lint(subset);
	FREE_lint(canonical_subset);
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_classification::trace_all_k_subsets done" << endl;
	}
}

void poset_classification::get_orbit_representatives(
		int level,
		int &nb_orbits, long int *&Orbit_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_classification::get_orbit_representatives" << endl;
	}
	nb_orbits = nb_orbits_at_level(level);
	if (f_v) {
		cout << "orbits_on_k_sets: we found " << nb_orbits
				<< " orbits on " << level << "-sets" << endl;
	}
	Orbit_reps = NEW_lint(nb_orbits * level);
	for (i = 0; i < nb_orbits; i++) {
		get_set_by_level(level, i, Orbit_reps + i * level);
	}
	
	if (f_v) {
		cout << "poset_classification::get_orbit_representatives done" << endl;
	}
}

void poset_classification::unrank_point(
		int *v, long int rk)
{
	Poset->unrank_point(v, rk);
}

long int poset_classification::rank_point(
		int *v)
{
	long int rk;

	rk = Poset->rank_point(v);
	return rk;
}

void poset_classification::unrank_basis(
		int *Basis, long int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		unrank_point(Basis + i * Poset->VS->dimension, S[i]);
	}
}

void poset_classification::rank_basis(
		int *Basis, long int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		S[i] = rank_point(Basis + i * Poset->VS->dimension);
	}
}

}}}





