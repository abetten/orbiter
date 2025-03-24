// upstep_work.cpp
//
// Anton Betten
// March 10, 2010

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


static void print_coset_table(coset_table_entry *coset_table, int len);

upstep_work::upstep_work()
{
	Record_birth();
	gen = NULL;
	size = 0;
	prev = 0;
	prev_ex = 0;
	cur = 0;
	nb_fusion_nodes = 0;
	nb_fuse_cur = 0;
	nb_ext_cur = 0;
	f_debug = false;
	f_implicit_fusion = false;
	f_indicate_not_canonicals = false;
	mod_for_printing = 1;
	pt = 0;
	pt_orbit_len = 0;
	path = NULL;
	O_prev = NULL;
	O_cur = NULL;
	G = NULL;
	H = NULL;
	coset = 0;
	nb_cosets = 0;
	nb_cosets_processed = 0;
	coset_table = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;


}

upstep_work::~upstep_work()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "upstep_work::~upstep_work" << endl;
		}
	if (G) {
		if (f_v) {
			cout << "upstep_work::~upstep_work "
					"before FREE_OBJECT(G)" << endl;
			}
		FREE_OBJECT(G);
		G = NULL;
		}
	if (H) {
		if (f_v) {
			cout << "upstep_work::~upstep_work "
					"before FREE_OBJECT(H)" << endl;
			}
		FREE_OBJECT(H);
		H = NULL;
		}
	if (coset_table) {
		if (f_v) {
			cout << "upstep_work::~upstep_work "
					"before FREE_OBJECT(coset_table)" << endl;
			}
		FREE_OBJECTS(coset_table);
		coset_table = NULL;
		}
	if (path) {
		if (f_v) {
			cout << "upstep_work::~upstep_work "
					"before FREE_int(path)" << endl;
			}
		FREE_int(path);
		path = NULL;
		}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (f_v) {
		cout << "upstep_work::~upstep_work done" << endl;
		}
}

void upstep_work::init(
		poset_classification *gen,
	int size,
	int prev,
	int prev_ex,
	int cur,
	int f_debug,
	int f_implicit_fusion,
	int f_indicate_not_canonicals, 
	int verbose_level)
// called from poset_classification::extend_node
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "upstep_work::init size=" << size
				<< " prev=" << prev << " prev_ex="
				<< prev_ex << " cur=" << cur << endl;
		}
	upstep_work::gen = gen;
	upstep_work::size = size;
	upstep_work::prev = prev;
	upstep_work::prev_ex = prev_ex;
	upstep_work::cur = cur;
	upstep_work::f_debug = f_debug;
	upstep_work::f_implicit_fusion = f_implicit_fusion;
	upstep_work::f_indicate_not_canonicals = f_indicate_not_canonicals;

	O_prev = gen->get_node(prev);


	if (O_prev->get_nb_of_extensions() > 25) {
		mod_for_printing = 25;
		}
	if (O_prev->get_nb_of_extensions() > 50) {
		mod_for_printing = 50;
		}
	if (O_prev->get_nb_of_extensions() > 100) {
		mod_for_printing = 100;
		}
	if (O_prev->get_nb_of_extensions() > 500) {
		mod_for_printing = 500;
		}

	path = NEW_int(size + 1);
	path[size] = prev;
	for (i = size - 1; i >= 0; i--) {
		path[i] = gen->get_node(path[i + 1])->get_prev();
		}
	if (f_v) {
		cout << "upstep_work::init path: ";
		Int_vec_print(cout, path, size + 1);
		cout << endl;
		}

	Elt1 = NEW_int(gen->get_poset()->A->elt_size_in_int);
	Elt2 = NEW_int(gen->get_poset()->A->elt_size_in_int);
	Elt3 = NEW_int(gen->get_poset()->A->elt_size_in_int);

	if (f_v) {
		cout << "upstep_work::init done" << endl;
		}
}

void upstep_work::handle_extension(
		int &nb_fuse_cur,
		int &nb_ext_cur, int verbose_level)
// called from poset_classification::extend_node
// Calls handle_extension_fusion_type 
// or handle_extension_unprocessed_type
//
// Handles the extension 'cur_ex' in node 'prev'.
// We are extending a set of size 'size' to a set of size 'size' + 1. 
// Calls poset_orbit_node::init_extension_node for the new node
// that is (possibly) created
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int type;

	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "upstep_work::handle_extension verbose_level = "
				<< verbose_level << endl;
		cout << "prev=" << prev << " prev_ex=" << prev_ex << endl;
		}
	pt = O_prev->get_E(prev_ex)->get_pt();
	type = O_prev->get_E(prev_ex)->get_type();

	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "type ";
		print_extension_type(cout, type);
		cout << endl;
		}

	if (type == EXTENSION_TYPE_FUSION) {
		if (f_v) {
			cout << "upstep_work::handle_extension fusion type" << endl;
			}
		handle_extension_fusion_type(verbose_level - 2);	
		nb_fuse_cur++;
		}
	else if (type == EXTENSION_TYPE_UNPROCESSED) {
		if (f_v) {
			cout << "upstep_work::handle_extension unprocessed type" << endl;
			}
		handle_extension_unprocessed_type(verbose_level);
		nb_ext_cur++;
		}
	else {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << endl;
		cout << "upstep_work::handle_extension extension "
				"not of unprocessed type, error" << endl;
		cout << "type is ";
		print_extension_type(cout, type);
		cout << endl;
		exit(1);
		}
	if (f_v) {
		cout << "upstep_work::handle_extension prev=" << prev
				<< " prev_ex=" << prev_ex << " done" << endl;
		}
}

void upstep_work::handle_extension_fusion_type(
		int verbose_level)
// called from upstep_work::handle_extension
// Handles the extension 'cur_ex' in node 'prev'.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	// fusion node, nothing to do
	
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::handle_extension_fusion_type" << endl;
		}		
	if (f_vv) {
		long int *set;
		set = NEW_lint(size + 1);
		O_prev->store_set_to(gen, size - 1, set /*gen->S1*/);
			// store_set_to(k) stores a set of size k+1
			// so here we store the first size points of the set
			// namely the current set.
								
			// next we store the size+1 th point:
		set[size] = pt; //gen->S1[size] = pt;
			// so, we really have a set of size size + 1

		gen->print_level_extension_info(size, prev, prev_ex);
		cout << " point " << pt << " ";
		Lint_vec_set_print(cout, set /*gen->S1*/, size + 1);
		cout << " is a fusion node, skipping" << endl;
		FREE_lint(set);
#if 0
		if (f_vvv) {
			if (gen->f_print_function) {
				(*gen->print_function)(cout, size + 1,
						gen->S1, gen->print_function_data);
				}
			gen->generator_apply_isomorphism_no_transporter(
				size, size + 1, prev, prev_ex, 
				gen->S1, gen->S2, 
				verbose_level - 3);
			cout << "fusion elt: " << endl;
			//A->element_print_quick(Elt1, cout);
			cout << "maps it to: ";
			int_set_print(cout, gen->S2, size + 1);
			cout << endl;
			if (gen->f_print_function) {
				(*gen->print_function)(cout, size + 1,
						gen->S2, gen->print_function_data);
				}
			}
#endif
		}
}

void upstep_work::handle_extension_unprocessed_type(
		int verbose_level)
// called from upstep_work::handle_extension
// calls init_extension_node
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	int ret, type;
	
	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "upstep_work::handle_extension_unprocessed_type" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	type = O_prev->get_E(prev_ex)->get_type();
		
	if (f_vv) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "with point " << pt << " : " << endl;
	}
	if (type != EXTENSION_TYPE_UNPROCESSED) {
		cout << "upstep_work::handle_extension_unprocessed_type "
				"extension not of unprocessed type, error" << endl;
		cout << "type is ";
		print_extension_type(cout, type);
		cout << endl;
		exit(1);
	}
				
	// process the node and create a new set orbit at level size + 1:
				
	pt_orbit_len = O_prev->get_E(prev_ex)->get_orbit_len();

	size++;
		// here, size is incremented so we need to subtract
		// one if we want to use gen->print_level_extension_info
		
	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "with point " << pt << ", pt_orbit_len=" << pt_orbit_len
				<< " : before init_extension_node" << endl;
	}
	
	ret = init_extension_node(verbose_level - 3);

	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "with point " << pt << " : after init_extension_node ";
		cout << "nb_cosets_processed=" << nb_cosets_processed << endl;
	}
	if (f_vvv) {
		cout << "upstep_work::handle_extension_unprocessed_type "
				"coset_table:" << endl;
		print_coset_table(coset_table, nb_cosets_processed);
	}
	if (ret) {
		if (f_vv) {
			cout << "init_extension_node returns true" << endl;
		}
	}
	else {
		if (f_vv) {
			cout << "init_extension_node returns false, "
					"the set is not canonical" << endl;
		//cout << "u=" << gen->split_case << " @(" << prev
			//<< "," << prev_ex << ") not canonical" << endl;
		}
		if (f_vv) {
			cout << "the set is not canonical, we skip it" << endl;
		}
		cout << "setting type of extension to "
				"EXTENSION_TYPE_NOT_CANONICAL" << endl;
		O_prev->get_E(prev_ex)->set_type(EXTENSION_TYPE_NOT_CANONICAL);
		cur--;
		cout << "reducing cur to " << cur << endl;
	}


	cur++;
	size--;
		// the original value of size is restored

	if (f_vvv) {
		cout << "cur=" << cur << endl;
	}

	if (f_v) {
		gen->print_level_extension_info(size, prev, prev_ex);
		cout << "with point " << pt << " done" << endl;
		cout << "upstep_work::handle_extension_unprocessed_type done" << endl;
	}
}

int upstep_work::init_extension_node(
		int verbose_level)
// size has been incremented
// Called from upstep_work::handle_extension_unprocessed_type
// Calls upstep_subspace_action or upstep_for_sets, 
// depending on the type of action
// then changes the type of the extension to EXTENSION_TYPE_EXTENSION
//
// Establishes a new node at depth 'size'
// (i.e., a set of size 'size') as an extension
// of a previous node (prev) at depth size - 1 
// with respect to a given point (pt).
// This function is to be called for the next
// free poset_orbit_node which will
// become the descendant of the previous node (prev).
// the extension node corresponds to the point pt. 
// returns false if the set is not canonical
// (provided f_indicate_not_canonicals is true)
{

#if 0
	if (prev == 12 && prev_ex == 3) {
		cout << "upstep_work::init_extension_node we are at node (12,3)" << endl;
		verbose_level += 10;
	}
#endif

#if 0
	if (cur == 26) {
		cout << "upstep_work::init_extension_node Node=26" << endl;
	}
#endif

	//longinteger_domain D;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	


	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "upstep_work::init_extension_node cur=" << cur 
			<< " size=" << size
			<< " verbose_level=" << verbose_level << endl;
	}

	if (cur == -1) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "upstep_work::init_extension_node cur=" << cur << endl;
		exit(1);
	}

	O_cur = gen->get_node(cur);
		

	O_cur->init_node(cur, prev, pt, verbose_level);

	//if (f_v) {cout << "after freeself" << endl;}
	O_cur->store_set(gen, size - 1);
		// stores a set of size 'size' to gen->S
	
	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "upstep_work::init_extension_node "
				"initializing Node " << cur << " ";
		Lint_vec_print(cout, gen->get_S(), size);
		cout << " f_indicate_not_canonicals="
				<< f_indicate_not_canonicals;
		cout << " verbose_level=" << verbose_level;
		cout << endl;
	}

	if (f_vv) {
		cout << "point " << pt << " lies in an orbit of length "
				<< pt_orbit_len
				<< " verbose_level = " << verbose_level << endl;
	}


	if (G) {
		cout << "upstep_work::init_extension_node "
				"G is already allocated" << endl;
		exit(1);
	}
	if (H) {
		cout << "upstep_work::init_extension_node "
				"H is already allocated" << endl;
		exit(1);
	}
	G = NEW_OBJECT(data_structures_groups::group_container);
	H = NEW_OBJECT(data_structures_groups::group_container);
	

	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << "upstep_work::init_extension_node "
				"before O_cur->init_extension_node_prepare_G" << endl;
	}
	O_cur->init_extension_node_prepare_G(gen, 
		prev, prev_ex, size, *G, go_G, 
		verbose_level - 4);
	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << "upstep_work::init_extension_node "
				"after O_cur->init_extension_node_prepare_G" << endl;
	}

	

	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << endl;
	}
	if (f_vvv) {
		if (gen->get_poset()->f_print_function) {
			gen->get_poset()->invoke_print_function(cout, size, gen->get_S());
		}
	}
	if (f_vv) {
		cout << "(orbit length = " << pt_orbit_len << ")" << endl;
	}
	
	O_prev->get_E(prev_ex)->set_type(EXTENSION_TYPE_PROCESSING);
		// currently processing
	O_prev->get_E(prev_ex)->set_data(cur);
	

	//group H;


	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << "upstep_work::init_extension_node "
				"before O_cur->init_extension_node_prepare_H" << endl;
	}
	
	O_cur->init_extension_node_prepare_H(gen, 
		prev, prev_ex, size, 
		*G, go_G,
		*H, go_H, 
		pt, pt_orbit_len, 
		verbose_level - 2);


#if 0
	if (cur == 26) {
		cout << "upstep_work::init_extension_node Node=26" << endl;
		cout << "go_G=" << go_G << endl;
		cout << "go_H=" << go_H << endl;
	}
#endif
	
	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << "upstep_work::init_extension_node "
				"after O_cur->init_extension_node_prepare_H" << endl;
	}

	
	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "upstep_work::init_extension_node calling upstep" << endl;
	}

	if (gen->get_poset()->f_subspace_lattice) {
		if (f_v) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "upstep_work::init_extension_node "
					"calling upstep_subspace_action" << endl;
		}
		if (!upstep_subspace_action(verbose_level - 2)) {

			if (f_indicate_not_canonicals) {
				if (f_vv) {
					cout << "the set is not canonical" << endl;
				}
				return false;
			}
			cout << "upstep_subspace_action returns false, "
					"the set is not canonical, this should not happen"
					<< endl;
			exit(1);
		}
		if (f_v) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "upstep_work::init_extension_node "
					"after upstep_subspace_action" << endl;
		}
	}
	else {
		if (f_v) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "upstep_work::init_extension_node calling "
					"upstep_for_sets, verbose_level = "
					<< verbose_level - 2 << endl;
		}
		if (!upstep_for_sets(verbose_level - 2)) {
			if (f_indicate_not_canonicals) {
				if (f_vv) {
					cout << "the set is not canonical" << endl;
				}
				return false;
			}
			cout << "upstep_for_sets returns false, "
					"the set is not canonical, "
					"this should not happen" << endl;
			exit(1);
		}
		if (f_v) {
			gen->print_level_extension_info(size - 1, prev, prev_ex);
			Lint_vec_print(cout, gen->get_S(), size);
			cout << "upstep_work::init_extension_node after "
					"upstep_for_sets" << endl;
		}
	}
	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "extension with point " << pt << " : " << endl;
		cout << "after upstep_for_sets/upstep_subspace_action" << endl;
		//print_coset_table(coset_table, nb_cosets_processed);
	}

	gen->get_Poo()->change_extension_type(size - 1, prev, prev_ex,
			EXTENSION_TYPE_EXTENSION, 0/* verbose_level*/);


	if (f_vv) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "_{";
		H->print_group_order(cout);
		cout << "}" << endl;
	}

	groups::strong_generators *Strong_gens;

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(H->S, 0);

#if 0
	if (cur == 26) {
		longinteger_object go;
		cout << "upstep_work::init_extension_node Node=26 before "
				"upstep_subspace_action group order=";
		Strong_gens->group_order(go);
		cout << go;
		cout << endl;
		cout << "generators are:" << endl;
		Strong_gens->print_generators();
		cout << "tl:";
		int_vec_print(cout, Strong_gens->tl, gen->A->base_len);
		cout << endl;
	}
#endif

	O_cur->store_strong_generators(gen, Strong_gens);
	FREE_OBJECT(Strong_gens);
	

	if (f_v) {
		algebra::ring_theory::longinteger_object go;
		
		gen->stabilizer_order(cur, go);
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		Lint_vec_print(cout, gen->get_S(), size);
		cout << "_{";
		cout << go;
		cout << "} (double check)" << endl;
		cout << "upstep_work::init_extension_node done" << endl;
	}
	return true;
}

int upstep_work::upstep_for_sets(
		int verbose_level)
// This routine is called from upstep_work::init_extension_node
// It is testing a set of size 'size'.
// The newly added point is in gen->S[size - 1]
// returns false if the set is not canonical
// (provided f_indicate_not_canonicals is true)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);
	int f_v5 = (verbose_level >= 5);
	groups::schreier up_orbit;
	int possible_image;
	int *aut, idx;
	trace_result r;
	actions::action *A_by_restriction;
	int final_node, final_ex;
	data_structures_groups::union_find UF;
	
	O_cur->store_set(gen, size - 1); // stores a set of size 'size'
	if (f_v) {
		gen->print_level_extension_info(size - 1, prev, prev_ex);
		cout << "upstep for set ";
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << " verbose_level=" << verbose_level;
		cout << " f_indicate_not_canonicals="
				<< f_indicate_not_canonicals << endl;
		//cout << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_upstep_work");
	label_of_set_tex.assign("\\_upstep\\_work");

	A_by_restriction = gen->get_A2()->Induced_action->create_induced_action_by_restriction(
		NULL /*sims *old_G */,
		size, gen->get_S(), label_of_set, label_of_set_tex,
		false /* f_induce_action */,
		0 /*verbose_level - 2*/);
	
	// the newly added point:
	if (gen->get_S()[size - 1] != pt) {
		cout << "upstep_work::upstep_for_sets fatal: "
				"gen->S[size - 1] != pt" << endl;
		exit(1);
	}

	if (f_v) {
		print_level_extension_info();
		cout << "initializing up_orbit with restricted action ";
		A_by_restriction->print_info();
	}
	up_orbit.init(A_by_restriction, verbose_level - 2);
	//up_orbit.init(gen->A2);
	if (f_v) {
		print_level_extension_info();
		cout << "initializing up_orbit with generators" << endl;
	}
	up_orbit.init_generators(*H->SG, verbose_level - 2);



	int print_interval = 10000;

	if (f_v) {
		print_level_extension_info();
		cout << "computing orbit of point " << pt << endl;
	}
	up_orbit.compute_point_orbit(size - 1 /*pt*/, print_interval, 0);
		// the orbits of the group H
		// up_orbit will be extended as soon 
		// as new automorphisms are found



	if (f_vv) {
		cout << "upstep_work::upstep_for_sets "
				"initializing union_find:" << endl;
	}
	UF.init(A_by_restriction, 0 /*verbose_level - 8*/);
	if (f_vv) {
		cout << "upstep_work::upstep_for_sets "
				"adding generators to union_find:" << endl;
	}
	UF.add_generators(H->SG, 0 /*verbose_level - 8*/);
	if (f_vv) {
		cout << "upstep_work::upstep_for_sets "
				"initializing union_find done" << endl;
	}
	if (f_vvv) {
		UF.print();
	}


	if (coset_table) {
		cout << "upstep_work::upstep_for_sets "
				"coset_table is allocated" << endl;
		exit(1);
	}
	nb_cosets = size;
	nb_cosets_processed = 0;
	coset_table = NEW_OBJECTS(coset_table_entry, nb_cosets);
	
	for (coset = 0; coset < size - 1; coset++) { 
		if (f_v) {
			cout << "upstep_work::upstep_for_sets "
					"coset=" << coset << " / " << nb_cosets << endl;
		}
		// for all the previous (=old) points
		possible_image = gen->get_S()[coset];
		if (f_vv) {
			print_level_extension_coset_info();
			cout << " we are trying to map " << possible_image << " to " << pt << endl;
		}


		idx = UF.ancestor(coset);
		if (idx < coset) {
			//gen->nb_times_trace_was_saved++;
			if (f_vv) {
				print_level_extension_coset_info();
				cout << "coset " << coset << " / " << nb_cosets
						<< " is at " << idx << " which has already "
								"been done, so we save one trace" << endl;
			}
			continue;
		}




		if (f_v4) {
			print_level_extension_coset_info();
			cout << " orbit length upstep so far: " << up_orbit.Forest->orbit_len[0]
				<< " checking possible image " << possible_image << endl;
		}


		// initialize set[0] and transporter[0] for the tracing
		Lint_vec_copy(gen->get_S(), gen->get_set_i(0), size);
#if 0
		for (h = 0; h < size; h++) {
			gen->set[0][h] = gen->S[h];
		}
#endif
		gen->get_set_i(0)[coset] = pt;
		gen->get_set_i(0)[size - 1] = possible_image;
		gen->get_A()->Group_element->element_one(gen->get_transporter()->ith(0), 0);


		if (f_v4) {
			print_level_extension_coset_info();
			cout << "exchanged set: ";
			Lint_vec_set_print(cout, gen->get_set_i(0), size);
			cout << endl;
			cout << "upstep_work::upstep_for_sets "
					"calling recognize, verbose_level="
					<< verbose_level << endl;
		}
		
		int nb_times_image_of_called0 = gen->get_A()->ptr->nb_times_image_of_called;
		int nb_times_mult_called0 = gen->get_A()->ptr->nb_times_mult_called;
		int nb_times_invert_called0 = gen->get_A()->ptr->nb_times_invert_called;
		int nb_times_retrieve_called0 = gen->get_A()->ptr->nb_times_retrieve_called;
		
		r = recognize(final_node, 
				final_ex, 
				true /*f_tolerant*/, 
				verbose_level - 4);

		if (f_v) {
			cout << "upstep_work::upstep_for_sets coset "
					<< coset << " / " << nb_cosets
					<< " recognize returns "
					<< trace_result_as_text(r) << endl;
		}
		


		coset_table[nb_cosets_processed].coset = coset;
		coset_table[nb_cosets_processed].type = r;
		coset_table[nb_cosets_processed].node = final_node;
		coset_table[nb_cosets_processed].ex = final_ex;
		coset_table[nb_cosets_processed].nb_times_image_of_called = 
			gen->get_A()->ptr->nb_times_image_of_called - nb_times_image_of_called0;
		coset_table[nb_cosets_processed].nb_times_mult_called = 
			gen->get_A()->ptr->nb_times_mult_called - nb_times_mult_called0;
		coset_table[nb_cosets_processed].nb_times_invert_called = 
			gen->get_A()->ptr->nb_times_invert_called - nb_times_invert_called0;
		coset_table[nb_cosets_processed].nb_times_retrieve_called = 
			gen->get_A()->ptr->nb_times_retrieve_called - nb_times_retrieve_called0;
		nb_cosets_processed++;

		if (f_vvv) {
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_for_sets calling find_automorphism "
					"returns " << trace_result_as_text(r) << endl;
		}
		
		
		if (r == found_automorphism) {
			aut = gen->get_transporter()->ith(size);
			if (f_vvv) {
				print_level_extension_coset_info();
				cout << "upstep_work::upstep_for_sets found automorphism "
						"mapping " << possible_image << " to " << pt << endl;
				//gen->A->element_print_as_permutation(aut, cout);
				if (gen->allowed_to_show_group_elements() && f_v5) {
					gen->get_A()->Group_element->element_print_quick(aut, cout);
					cout << endl;
				}
			}
			if (gen->get_A2()->Group_element->element_image_of(possible_image, aut, 0) != pt) {
				cout << "upstep_work::upstep_for_sets image of possible_"
						"image is not pt" << endl;
				exit(1);
			}
			UF.add_generator(aut, 0 /*verbose_level - 5*/);
			up_orbit.extend_orbit(aut, verbose_level - 5);
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets new orbit length "
						"upstep = " << up_orbit.Forest->orbit_len[0] << endl;
			}
		}
		else if (r == not_canonical) {
			if (f_indicate_not_canonicals) {
				if (f_vvv) {
					cout << "upstep_work::upstep_for_sets not canonical"
							<< endl;
				}
				FREE_OBJECT(A_by_restriction);
				return false;
			}
#if 0
			print_level_extension_coset_info();
			cout << "upstep_work::upstep_for_sets fatal: find_automorphism_"
					"by_tracing returns not_canonical, this should "
					"not happen" << endl;
			exit(1);
#endif
		}
		else if (r == no_result_extension_not_found) {
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets "
						"no_result_extension_not_found" << endl;
			}
		}
		else if (r == no_result_fusion_node_installed) {
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets "
						"no_result_fusion_node_installed" << endl;
			}
		}
		else if (r == no_result_fusion_node_already_installed) {
			if (f_vvv) {
				cout << "upstep_work::upstep_for_sets "
						"no_result_fusion_node_already_installed" << endl;
			}
		}
	} // next j
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_for_sets upstep orbit "
				"length for set ";
		Lint_vec_set_print(cout, gen->get_S(), size);
		cout << " is " << up_orbit.Forest->orbit_len[0] << endl;

		cout << "coset_table of length " << nb_cosets_processed
				<< ":" << endl;
		print_coset_table(coset_table, nb_cosets_processed);
	}
	data_structures_groups::vector_ge SG_extension;
	int *tl_extension = NEW_int(gen->get_A()->base_len());
	int f_tolerant = true;
	
	if (f_vvv) {
		cout << "upstep_work::upstep_for_sets H->S->transitive_extension_tolerant "
				"up_orbit.orbit_len[0]="
				<< up_orbit.Forest->orbit_len[0] << endl;
	}
	H->S->transitive_extension_tolerant(
			up_orbit,
			SG_extension, tl_extension, f_tolerant,
			0 /*verbose_level - 3*/);
	H->delete_strong_generators();
	H->init_strong_generators(SG_extension, tl_extension, verbose_level - 2);
	
	FREE_int(tl_extension);
	
	if (f_v) {
		print_level_extension_info();
		cout << "upstep_work::upstep_for_sets done "
				"nb_cosets_processed = " << nb_cosets_processed << endl;
	}
	FREE_OBJECT(A_by_restriction);
	return true;
}



#if 0
void upstep_work::print_level_extension_info_original_size()
{
	gen->print_level_extension_info(size, prev, prev_ex);
}
#endif

void upstep_work::print_level_extension_info()
{
	gen->print_level_extension_info(size - 1, prev, prev_ex);
}

void upstep_work::print_level_extension_coset_info()
{
	gen->print_level_extension_coset_info(size - 1,
			prev, prev_ex, coset, nb_cosets);
}



static void print_coset_table(coset_table_entry *coset_table, int len)
{
	int i;
	
	cout << "coset table" << endl;
	cout << "i : coset : node : ex : nb_times_mult : nb_times_invert : "
			"nb_times_retrieve : trace_result" << endl;
	for (i = 0; i < len; i++) {
		cout << setw(3) << i << " : " 
			<< setw(5) << coset_table[i].coset << " : " 
			<< setw(5) << coset_table[i].node << " : " 
			<< setw(5) << coset_table[i].ex << " : (" 
			<< coset_table[i].nb_times_mult_called << "/" 
			<< coset_table[i].nb_times_invert_called << "/" 
			<< coset_table[i].nb_times_retrieve_called << ") : " 
			<< setw(5) << trace_result_as_text((trace_result)
					coset_table[i].type) << endl;
	}
}


}}}



