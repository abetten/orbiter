// poset_classification_init.cpp
//
// Anton Betten
// December 29, 2003
//
// moved here from poset_classification.cpp: July 29, 2014


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

poset_classification::poset_classification()
{
	t0 = 0;

	Control = NULL;

	//problem_label[0] = 0;
	//problem_label_with_path[0] = 0;
	
	Elt_memory = NULL;

	Poset = NULL;

	f_base_case = FALSE;
	Base_case = NULL;

	depth = 0;

	Schreier_vector_handler = NULL;
	S = NULL;
	sz = 0; // = depth

	Elt_memory = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Elt5 = NULL;
	Elt6 = NULL;


	tmp_set_apply_fusion = NULL;
	tmp_find_node_for_subspace_by_rank1 = NULL;
	tmp_find_node_for_subspace_by_rank2 = NULL;



	//nb_times_trace = 0;
	//nb_times_trace_was_saved = 0;
	
	transporter = NULL;
	set = NULL;
	
	
	nb_poset_orbit_nodes_used = 0;
	nb_poset_orbit_nodes_allocated = 0;
	poset_orbit_nodes_increment = 0;
	poset_orbit_nodes_increment_last = 0;

	root = NULL;
	first_poset_orbit_node_at_level = NULL;
	set0 = NULL;
	set1 = NULL;
	set3 = NULL;
	nb_extension_nodes_at_level_total = NULL;
	nb_extension_nodes_at_level = NULL;
	nb_fusion_nodes_at_level = NULL;
	nb_unprocessed_nodes_at_level = NULL;



	f_has_invariant_subset_for_root_node = FALSE;
	invariant_subset_for_root_node = NULL;
	invariant_subset_for_root_node_size = 0;
	
	
	

	
	f_do_group_extension_in_upstep = TRUE;

	f_allowed_to_show_group_elements = FALSE;	
	downstep_orbits_print_max_orbits = 25;
	downstep_orbits_print_max_points_per_orbit = 50;

	nb_times_image_of_called0 = 0;
	nb_times_mult_called0 = 0;
	nb_times_invert_called0 = 0;
	nb_times_retrieve_called0 = 0;
	nb_times_store_called0 = 0;

	progress_last_time = 0.;
	progress_epsilon = 0.;



	os_interface Os;

	t0 = Os.os_ticks();
}

poset_classification::~poset_classification()
{
	freeself();
}

void poset_classification::null()
{
}

void poset_classification::freeself()
{
	int i;
	int f_v = FALSE;
	
	if (f_v) {
		cout << "poset_classification::freeself" << endl;
	}
	if (Elt_memory) {
		FREE_int(Elt_memory);
	}

	// do not free Strong_gens
	

	if (f_v) {
		cout << "poset_classification::freeself deleting S" << endl;
	}
	if (S) {
		FREE_lint(S);
	}
	if (Schreier_vector_handler) {
		FREE_OBJECT(Schreier_vector_handler);
	}
	if (tmp_set_apply_fusion) {
		FREE_lint(tmp_set_apply_fusion);
	}
	if (tmp_find_node_for_subspace_by_rank1) {
		FREE_int(tmp_find_node_for_subspace_by_rank1);
	}
	if (tmp_find_node_for_subspace_by_rank2) {
		FREE_int(tmp_find_node_for_subspace_by_rank2);
	}

	if (f_v) {
		cout << "poset_classification::freeself "
				"deleting transporter and set[]" << endl;
	}
	if (transporter) {
		FREE_OBJECT(transporter);
		for (i = 0; i <= sz; i++) {
			FREE_lint(set[i]);
		}
		FREE_plint(set);
	}
	if (f_v) {
		cout << "poset_classification::freeself "
				"before exit_poset_orbit_node" << endl;
	}
	exit_poset_orbit_node();
	if (f_v) {
		cout << "poset_classification::freeself "
				"after exit_poset_orbit_node" << endl;
	}


	if (f_v) {
		cout << "poset_classification::freeself done" << endl;
	}
	null();
}


void poset_classification::init_internal(
	poset_classification_control *PC_control,
	poset *Poset,
	int sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v6 = (verbose_level >= 6);
	int i;
	
	if (f_v) {
		cout << "poset_classification::init_internal sz=" << sz << endl;
		cout << "poset_classification::init_internal Control=" << endl;
		Control->print();
	}
	
	Control = PC_control;
	poset_classification::Poset = Poset;
	poset_classification::sz = sz;

	problem_label.assign(Control->problem_label);
	problem_label_with_path.assign(Control->path);
	problem_label_with_path.append(Control->problem_label);
	//Control->init_labels(problem_label, problem_label_with_path);

	if (f_v) {
		cout << "poset_classification::init_internal, problem_label=" << problem_label << endl;
		cout << "poset_classification::init_internal, problem_label_with_path=" << problem_label_with_path << endl;
	}

	if (Poset == NULL) {
		cout << "poset_classification::init_internal "
				"Poset == NULL" << endl;
		exit(1);
	}
	if (Poset->A == NULL) {
		cout << "poset_classification::init_internal "
				"Poset->A == NULL" << endl;
		exit(1);
	}
	if (Poset->A2 == NULL) {
		cout << "poset_classification::init_internal "
				"Poset->A2 == NULL" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_classification::init_internal "
				"sz = " << sz << endl;
		cout << "poset_classification::init_internal "
				"A->degree=" << Poset->A->degree << endl;
		cout << "poset_classification::init_internal "
				"A2->degree=" << Poset->A2->degree << endl;
	}

	if (Poset->Orbit_based_testing) {
		Poset->Orbit_based_testing->PC = this;
	}

	os_interface Os;

	t0 = Os.os_ticks();

	progress_epsilon = 0.005;

	
	if (f_vv) {
		cout << "poset_classification::init_internal action A:" << endl;
		Poset->A->print_info();
		cout << "poset_classification::init_internal action A2:" << endl;
		Poset->A2->print_info();
	}


	if (f_v) {
		cout << "poset_classification::init_internal group order is ";
		cout << Poset->go << endl;
	}
	
	Schreier_vector_handler = NEW_OBJECT(schreier_vector_handler);
	Schreier_vector_handler->init(Poset->A, Poset->A2,
			TRUE /* f_allow_failure */,
			verbose_level);
	
	if (f_v) {
		cout << "poset_classification::init_internal sz = " << sz << endl;
	}
	
	if (f_vv) {
		cout << "poset_classification::init_internal "
				"allocating S of size " << sz << endl;
	}
	S = NEW_lint(sz);
	for (i = 0; i < sz; i++) {
		S[i] = i;
	}

	tmp_set_apply_fusion = NEW_lint(sz + 1);

	if (f_vv) {
		cout << "poset_classification::init_internal "
				"allocating Elt_memory" << endl;
	}


	Elt_memory = NEW_int(6 * Poset->A->elt_size_in_int);
	Elt1 = Elt_memory + 0 * Poset->A->elt_size_in_int;
	Elt2 = Elt_memory + 1 * Poset->A->elt_size_in_int;
	Elt3 = Elt_memory + 2 * Poset->A->elt_size_in_int;
	Elt4 = Elt_memory + 3 * Poset->A->elt_size_in_int;
	Elt5 = Elt_memory + 4 * Poset->A->elt_size_in_int;
	Elt6 = Elt_memory + 5 * Poset->A->elt_size_in_int;
	
	transporter = NEW_OBJECT(vector_ge);
	transporter->init(Poset->A, verbose_level - 2);
	transporter->allocate(sz + 1, verbose_level - 2);
	Poset->A->element_one(transporter->ith(0), FALSE);
	
	set = NEW_plint(sz + 1);
	for (i = 0; i <= sz; i++) {
		set[i] = NEW_lint(sz);
	}

		
	nb_poset_orbit_nodes_used = 0;
	nb_poset_orbit_nodes_allocated = 0;

	nb_times_image_of_called0 = Poset->A->ptr->nb_times_image_of_called;
	nb_times_mult_called0 = Poset->A->ptr->nb_times_mult_called;
	nb_times_invert_called0 = Poset->A->ptr->nb_times_invert_called;
	nb_times_retrieve_called0 = Poset->A->ptr->nb_times_retrieve_called;
	nb_times_store_called0 = Poset->A->ptr->nb_times_store_called;


	if (Poset->f_subspace_lattice) {
		tmp_find_node_for_subspace_by_rank1 = NEW_int(Poset->VS->dimension);
		tmp_find_node_for_subspace_by_rank2 = NEW_int(sz * Poset->VS->dimension);
		//tmp_find_node_for_subspace_by_rank3 =
		//		NEW_int(Poset->VS->dimension);
	}

	if (f_v) {
		cout << "poset_classification::init_internal done" << endl;
	}
}

void poset_classification::initialize_and_allocate_root_node(
	poset_classification_control *PC_control,
	poset *Poset,
	int depth,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_classification::initialize_and_allocate_root_node" << endl;
		cout << "poset_classification::initialize_and_allocate_root_node "
				"depth = " << depth << endl;
	}


	poset_classification::depth = depth;
	poset_classification::Control = PC_control;
	downstep_orbits_print_max_orbits = 50;
	downstep_orbits_print_max_points_per_orbit = INT_MAX;
	

	// !!!
	//f_allowed_to_show_group_elements = TRUE;

	if (f_vv) {
		cout << "poset_classification::initialize_and_allocate_root_node "
				"before init_internal" << endl;
	}
	init_internal(PC_control,
		Poset,
		depth,
		verbose_level - 2);
	if (f_vv) {
		cout << "poset_classification::initialize_and_allocate_root_node "
				"after init_internal" << endl;
	}
	
	int nb_poset_orbit_nodes = 1000;
	
	if (f_vv) {
		cout << "poset_classification::initialize_and_allocate_root_node "
				"calling gen->init_poset_orbit_node" << endl;
	}
	init_poset_orbit_node(nb_poset_orbit_nodes, verbose_level - 1);
	if (f_vv) {
		cout << "poset_classification::initialize_and_allocate_root_node "
				"calling gen->init_root_node" << endl;
	}
	init_root_node(verbose_level - 1);


	if (f_v) {
		cout << "poset_classification::initialize_and_allocate_root_node done" << endl;
	}
}


void poset_classification::initialize_with_base_case(
	poset_classification_control *PC_control,
	poset *Poset,
	int depth, 
	classification_base_case *Base_case,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_classification::initialize_with_base_case" << endl;
	}

	Control = PC_control;

	poset_classification::depth = depth;
	downstep_orbits_print_max_orbits = 50;
	downstep_orbits_print_max_points_per_orbit = INT_MAX;
	

	// !!!
	//f_allowed_to_show_group_elements = TRUE;

	if (f_vv) {
		cout << "poset_classification::initialize_with_base_case "
				"before init_internal" << endl;
	}
	init_internal(PC_control, Poset,
		depth, verbose_level - 2);
	if (f_vv) {
		cout << "poset_classification::initialize_with_base_case "
				"after init_internal" << endl;
	}
	

	if (f_vv) {
		cout << "poset_classification::initialize_with_base_case "
				"calling init_starter" << endl;
	}
	init_base_case(Base_case, verbose_level);

	int nb_poset_orbit_nodes = 1000;
	
	if (f_vv) {
		cout << "poset_classification::initialize_with_base_case "
				"calling gen->init_poset_orbit_node" << endl;
	}
	init_poset_orbit_node(nb_poset_orbit_nodes, verbose_level - 1);
	if (f_vv) {
		cout << "poset_classification::initialize_with_base_case "
				"calling gen->init_root_node" << endl;
	}
	init_root_node(verbose_level);

	if (f_v) {
		cout << "poset_classification::initialize_with_base_case done" << endl;
	}
}

void poset_classification::init_root_node_invariant_subset(
	int *invariant_subset, int invariant_subset_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_classification::init_root_node_"
				"invariant_subset" << endl;
	}
	f_has_invariant_subset_for_root_node = TRUE;
	invariant_subset_for_root_node = invariant_subset;
	invariant_subset_for_root_node_size = invariant_subset_size;
	if (f_v) {
		cout << "poset_classification::init_root_node_"
				"invariant_subset "
				"installed invariant subset of size "
				<< invariant_subset_size << endl;
	}
}

void poset_classification::init_root_node_from_base_case(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_classification::init_root_node_from_base_case" << endl;
	}

	if (Base_case == NULL) {
		cout << "poset_classification::init_root_node_from_base_case Base_case == NULL" << endl;
		exit(1);
	}

	first_poset_orbit_node_at_level[0] = 0;
	root[0].freeself();
	root[0].node = 0;
	root[0].prev = -1;
	root[0].nb_strong_generators = 0;
	root[0].Schreier_vector = NULL;

	for (i = 0; i < Base_case->size; i++) {

		nb_extension_nodes_at_level_total[i] = 0;
		nb_extension_nodes_at_level[i] = 0;
		nb_fusion_nodes_at_level[i] = 0;
		nb_unprocessed_nodes_at_level[i] = 0;

		if (f_v) {
			cout << "poset_classification::init_root_node_from_base_case "
					"initializing node at level " << i << endl;
		}
		first_poset_orbit_node_at_level[i + 1] =
				first_poset_orbit_node_at_level[i] + 1;
		root[i].E = NEW_OBJECTS(extension, 1);
		root[i].nb_extensions = 1;
		root[i].E[0].type = EXTENSION_TYPE_EXTENSION;
		root[i].E[0].data = i + 1;
		root[i + 1].freeself();
		root[i + 1].node = i + 1;
		root[i + 1].prev = i;
		root[i + 1].pt = Base_case->orbit_rep[i];
		root[i + 1].nb_strong_generators = 0;
		root[i + 1].Schreier_vector = NULL;
	}
	if (f_v) {
		cout << "poset_classification::init_root_node_from_base_case "
				"storing strong poset_classifications" << endl;
	}
	root[Base_case->size].store_strong_generators(
			this, Base_case->Stab_gens);
	first_poset_orbit_node_at_level[Base_case->size + 1] =
			Base_case->size + 1;
	if (f_v) {
		cout << "i : first_poset_orbit_node_at_level[i]" << endl;
		for (i = 0; i <= Base_case->size + 1; i++) {
			cout << i << " : "
					<< first_poset_orbit_node_at_level[i] << endl;
		}
	}

	if (f_v) {
		cout << "poset_classification::init_root_node_from_base_case done" << endl;
	}
}

void poset_classification::init_root_node(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_classification::init_root_node" << endl;
	}
	if (f_base_case) {

		init_root_node_from_base_case(verbose_level);

	}
	else {
		root[0].init_root_node(this, verbose_level - 1);
	}
	if (f_v) {
		cout << "poset_classification::init_root_node done" << endl;
	}
}

void poset_classification::init_poset_orbit_node(
		int nb_poset_orbit_nodes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_classification::init_poset_orbit_node" << endl;
	}
	root = NEW_OBJECTS(poset_orbit_node, nb_poset_orbit_nodes);
	for (i = 0; i < nb_poset_orbit_nodes; i++) {
		root[i].node = i;
	}
	nb_poset_orbit_nodes_allocated = nb_poset_orbit_nodes;
	nb_poset_orbit_nodes_used = 0;
	poset_orbit_nodes_increment = nb_poset_orbit_nodes;
	poset_orbit_nodes_increment_last = nb_poset_orbit_nodes;
	first_poset_orbit_node_at_level = NEW_int(sz + 2);
	first_poset_orbit_node_at_level[0] = 0;
	first_poset_orbit_node_at_level[1] = 1;
	set0 = NEW_lint(sz + 1);
	set1 = NEW_lint(sz + 1);
	set3 = NEW_lint(sz + 1);
	nb_extension_nodes_at_level_total = NEW_int(sz + 1);
	nb_extension_nodes_at_level = NEW_int(sz + 1);
	nb_fusion_nodes_at_level = NEW_int(sz + 1);
	nb_unprocessed_nodes_at_level = NEW_int(sz + 1);
	for (i = 0; i < sz + 1; i++) {
		nb_extension_nodes_at_level_total[i] = 0;
		nb_extension_nodes_at_level[i] = 0;
		nb_fusion_nodes_at_level[i] = 0;
		nb_unprocessed_nodes_at_level[i] = 0;
	}
	if (f_v) {
		cout << "poset_classification::init_poset_orbit_node done" << endl;
	}
}


void poset_classification::exit_poset_orbit_node()
{
	if (root) {
		FREE_OBJECTS(root);
		root = NULL;
	}
	if (set0) {
		FREE_lint(set0);
		set0 = NULL;
	}
	if (set1) {
		FREE_lint(set1);
		set1 = NULL;
	}
	if (set3) {
		FREE_lint(set3);
		set3 = NULL;
	}
	if (first_poset_orbit_node_at_level) {
		FREE_int(first_poset_orbit_node_at_level);
		first_poset_orbit_node_at_level = NULL;
	}

	if (nb_extension_nodes_at_level_total) {
		FREE_int(nb_extension_nodes_at_level_total);
		nb_extension_nodes_at_level_total = NULL;
	}
	if (nb_extension_nodes_at_level) {
		FREE_int(nb_extension_nodes_at_level);
		nb_extension_nodes_at_level = NULL;
	}
	if (nb_fusion_nodes_at_level) {
		FREE_int(nb_fusion_nodes_at_level);
		nb_fusion_nodes_at_level = NULL;
	}
	if (nb_unprocessed_nodes_at_level) {
		FREE_int(nb_unprocessed_nodes_at_level);
		nb_unprocessed_nodes_at_level = NULL;
	}
}

void poset_classification::reallocate()
{
	int increment_new;
	int verbose_level = 0;
	
	increment_new = poset_orbit_nodes_increment +
			poset_orbit_nodes_increment_last;
	reallocate_to(nb_poset_orbit_nodes_allocated +
			poset_orbit_nodes_increment, verbose_level - 1);
	poset_orbit_nodes_increment_last = poset_orbit_nodes_increment;
	poset_orbit_nodes_increment = increment_new;
	
}

void poset_classification::reallocate_to(int new_number_of_nodes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	poset_orbit_node *new_root;
	
	if (f_v) {
		cout << "poset_classification::reallocate_to" << endl;
	}
	if (new_number_of_nodes < nb_poset_orbit_nodes_allocated) {
		cout << "poset_classification::reallocate_to "
				"new_number_of_nodes < "
				"nb_poset_orbit_nodes_allocated" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_classification::reallocate_to from "
				<< nb_poset_orbit_nodes_allocated
				<< " to " << new_number_of_nodes << endl;
	}
	new_root = NEW_OBJECTS(poset_orbit_node, new_number_of_nodes);
	for (i = 0; i < nb_poset_orbit_nodes_allocated; i++) {
		new_root[i] = root[i];
		root[i].null();
	}
	FREE_OBJECTS(root);
	root = new_root;
	nb_poset_orbit_nodes_allocated = new_number_of_nodes;
	if (f_v) {
		cout << "poset_classification::reallocate_to done" << endl;
	}
}


void poset_classification::init_base_case(classification_base_case *Base_case,
	int verbose_level)
// Does not initialize the first starter nodes.
// This is done in init_root_node
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::init_base_case" << endl;
	}
	f_base_case = TRUE;
	poset_classification::Base_case = Base_case;
}


}}


