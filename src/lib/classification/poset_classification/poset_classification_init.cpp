// poset_classification_init.cpp
//
// Anton Betten
// December 29, 2003
//
// moved here from poset_classification.cpp: July 29, 2014


#include "foundations/foundations.h"
#include "discreta/discreta.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


poset_classification::poset_classification()
{
	t0 = 0;

	Control = NULL;

	//problem_label[0] = 0;
	//problem_label_with_path[0] = 0;
	
	//Elt_memory = NULL;

	Poset = NULL;

	f_base_case = FALSE;
	Base_case = NULL;

	depth = 0;

	Schreier_vector_handler = NULL;
	set_S = NULL;
	sz = 0; // = depth
	max_set_size = 0;


	//tmp_set_apply_fusion = NULL;
	//tmp_find_node_for_subspace_by_rank1 = NULL;
	//tmp_find_node_for_subspace_by_rank2 = NULL;


	
	Orbit_tracer = NULL;
	
	Poo = NULL;


	nb_times_image_of_called0 = 0;
	nb_times_mult_called0 = 0;
	nb_times_invert_called0 = 0;
	nb_times_retrieve_called0 = 0;
	nb_times_store_called0 = 0;

	progress_last_time = 0.;
	progress_epsilon = 0.;



	orbiter_kernel_system::os_interface Os;

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
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::freeself" << endl;
	}

	

	if (set_S) {
		if (f_v) {
			cout << "poset_classification::freeself deleting S" << endl;
		}
		FREE_lint(set_S);
	}
	if (Schreier_vector_handler) {
		FREE_OBJECT(Schreier_vector_handler);
	}

#if 0
	if (tmp_set_apply_fusion) {
		FREE_lint(tmp_set_apply_fusion);
	}
	if (tmp_find_node_for_subspace_by_rank1) {
		FREE_int(tmp_find_node_for_subspace_by_rank1);
	}
	if (tmp_find_node_for_subspace_by_rank2) {
		FREE_int(tmp_find_node_for_subspace_by_rank2);
	}
#endif

	if (f_v) {
		cout << "poset_classification::freeself "
				"deleting transporter and set[]" << endl;
	}

	if (Orbit_tracer) {
		FREE_OBJECT(Orbit_tracer);
	}

	if (Poo) {
		FREE_OBJECT(Poo);
	}


	if (f_v) {
		cout << "poset_classification::freeself done" << endl;
	}
	null();
}


void poset_classification::init_internal(
	poset_classification_control *PC_control,
	poset_with_group_action *Poset,
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
		cout << "poset_classification::init_internal, "
				"problem_label=" << problem_label << endl;
		cout << "poset_classification::init_internal, "
				"problem_label_with_path=" << problem_label_with_path << endl;
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

	max_set_size = Poset->A2->degree;
	if (f_v) {
		cout << "poset_classification::init_internal max_set_size=" << max_set_size << endl;
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

	orbiter_kernel_system::os_interface Os;

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
	
	Schreier_vector_handler = NEW_OBJECT(data_structures_groups::schreier_vector_handler);
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
	set_S = NEW_lint(sz);
	for (i = 0; i < sz; i++) {
		set_S[i] = i;
	}

#if 0
	//tmp_set_apply_fusion = NEW_lint(sz + 1);
#endif

	if (f_vv) {
		cout << "poset_classification::init_internal "
				"allocating Elt_memory" << endl;
	}

#if 0
	Elt_memory = NEW_int(6 * Poset->A->elt_size_in_int);
	Elt1 = Elt_memory + 0 * Poset->A->elt_size_in_int;
	Elt2 = Elt_memory + 1 * Poset->A->elt_size_in_int;
	Elt3 = Elt_memory + 2 * Poset->A->elt_size_in_int;
	Elt4 = Elt_memory + 3 * Poset->A->elt_size_in_int;
	Elt5 = Elt_memory + 4 * Poset->A->elt_size_in_int;
	Elt6 = Elt_memory + 5 * Poset->A->elt_size_in_int;
#endif

	if (sz == 0) {
		cout << "poset_classification::init_internal sz == 0" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "poset_classification::init_internal "
				"allocating Orbit_tracer" << endl;
	}
	Orbit_tracer = NEW_OBJECT(orbit_tracer);

	if (f_vv) {
		cout << "poset_classification::init_internal "
				"before Orbit_tracer->init" << endl;
	}
	Orbit_tracer->init(this, verbose_level);


	int nb_poset_orbit_nodes = 1000;

	Poo = NEW_OBJECT(poset_of_orbits);
	if (f_vv) {
		cout << "poset_classification::init_internal "
				"before Poo->init" << endl;
	}
	Poo->init(this, nb_poset_orbit_nodes, sz, max_set_size, t0, verbose_level);
	if (f_vv) {
		cout << "poset_classification::init_internal "
				"after Poo->init" << endl;
	}

	if (f_v) {
		cout << "poset_classification::init_internal before Control->prepare" << endl;
	}
	Control->prepare(this, verbose_level);
	if (f_v) {
		cout << "poset_classification::init_internal after Control->prepare" << endl;
	}

	nb_times_image_of_called0 = Poset->A->ptr->nb_times_image_of_called;
	nb_times_mult_called0 = Poset->A->ptr->nb_times_mult_called;
	nb_times_invert_called0 = Poset->A->ptr->nb_times_invert_called;
	nb_times_retrieve_called0 = Poset->A->ptr->nb_times_retrieve_called;
	nb_times_store_called0 = Poset->A->ptr->nb_times_store_called;


#if 0
	if (Poset->f_subspace_lattice) {
		tmp_find_node_for_subspace_by_rank1 = NEW_int(Poset->VS->dimension);
		tmp_find_node_for_subspace_by_rank2 = NEW_int(sz * Poset->VS->dimension);
		//tmp_find_node_for_subspace_by_rank3 =
		//		NEW_int(Poset->VS->dimension);
	}
#endif

	if (f_v) {
		cout << "poset_classification::init_internal done" << endl;
	}
}

void poset_classification::initialize_and_allocate_root_node(
	poset_classification_control *PC_control,
	poset_with_group_action *Poset,
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
	//downstep_orbits_print_max_orbits = 50;
	//downstep_orbits_print_max_points_per_orbit = INT_MAX;
	

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
	
	Poo->init_root_node(verbose_level - 1);


	if (f_v) {
		cout << "poset_classification::initialize_and_allocate_root_node done" << endl;
	}
}


void poset_classification::initialize_with_base_case(
	poset_classification_control *PC_control,
	poset_with_group_action *Poset,
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
	//downstep_orbits_print_max_orbits = 50;
	//downstep_orbits_print_max_points_per_orbit = INT_MAX;
	

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

	Poo->init_root_node(verbose_level);

	if (f_v) {
		cout << "poset_classification::initialize_with_base_case done" << endl;
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


}}}


