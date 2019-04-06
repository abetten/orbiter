// poset_classification_classify.C
//
// Anton Betten
//
// moved here from poset_classification.C
// July 19, 2014


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {


void poset_classification::compute_orbits_on_subsets(
	int target_depth,
	const char *prefix,
	int f_W, int f_w,
	poset *Poset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_poset_orbit_nodes = 1000;
	int schreier_depth = target_depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0 = os_ticks();


	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"verbose_level=" << verbose_level << endl;
		}
	//gen = NEW_OBJECT(poset_classification);


	poset_classification::f_W = f_W;
	depth = target_depth;
	downstep_orbits_print_max_orbits = 50;
	downstep_orbits_print_max_points_per_orbit = INT_MAX;


	// !!!
	f_allowed_to_show_group_elements = FALSE;

	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"calling gen->init" << endl;
		}
	init(Poset,
		target_depth, verbose_level - 1);

	strcpy(fname_base, prefix);


	init_poset_orbit_node(nb_poset_orbit_nodes, verbose_level - 1);
	init_root_node(verbose_level - 1);

	main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);

	int i, fst, len;

	if (f_v) {
		cout << "compute_orbits_on_subsets done" << endl;
		cout << "depth : number of orbits" << endl;
		}
	for (i = 0; i < target_depth + 1; i++) {
		fst = first_poset_orbit_node_at_level[i];
		len = first_poset_orbit_node_at_level[i + 1] - fst;
		if (f_v) {
			cout << i << " : " << len << endl;
			}
		}
	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"done" << endl;
		}
}



int poset_classification::compute_orbits(int from_level, int to_level, 
	int verbose_level)
// returns the last level that was computed.
{
	int f_v = (verbose_level >= 1);
	int level;
	int f_create_schreier_vector = TRUE;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int f_write_files;


	if (f_v) {
		cout << "poset_classification::compute_orbits from "
				<< from_level << " to " << to_level << endl;
		cout << "f_lex=" << f_lex << endl;
		cout << "fname_base=" << fname_base << endl;
		}


	for (level = from_level; level < to_level; level++) {

		if (f_v) {
			cout << "poset_classification::compute_orbits: ";
			print_problem_label();
			cout << " calling extend_level " << level << endl;
			}

		int f_write_candidate_file = FALSE;

#if 1
		if (f_W && level) {
			f_write_candidate_file = TRUE;
			}

		if (f_w && level == to_level - 1) {
			f_write_candidate_file = TRUE;
			}
#endif

		extend_level(level, 
			f_create_schreier_vector, 
			f_use_invariant_subset_if_available, 
			f_debug,
			f_write_candidate_file, 
			verbose_level - 2);
		
		
		f_write_files = (f_W || (f_w && level == to_level - 1));
	
		

		if (f_write_data_files) {
			housekeeping(level + 1, f_write_files,
					os_ticks(), verbose_level - 1);
		}
		else {
			housekeeping_no_data_file(level + 1,
					os_ticks(), verbose_level - 1);
		}

		int nb_nodes;
		nb_nodes = nb_orbits_at_level(level + 1);
		if (nb_nodes == 0) {
			int j;
			for (j = level + 2; j <= to_level + 1; j++) {
				first_poset_orbit_node_at_level[j] =
						first_poset_orbit_node_at_level[j - 1];
				}
			return level;
			}	
			
		} // next level

	
	if (f_v) {
		cout << "poset_classification::compute_orbits from "
				<< from_level << " to " << to_level << " done" << endl;
		}
	return to_level;
}

int poset_classification::main(int t0, 
	int schreier_depth, 
	int f_use_invariant_subset_if_available, 
	int f_debug, 
	int verbose_level)
// f_use_invariant_subset_if_available
// is an option that affects the downstep.
// if FALSE, the orbits of the stabilizer on all points are computed. 
// if TRUE, the orbits of the stabilizer on the set of points that were 
// possible in the previous level are computed only 
// (using Schreier.orbits_on_invariant_subset_fast).
// The set of possible points is stored 
// inside the schreier vector data structure (sv).
{
	int f_v = (verbose_level >= 1);
	int size, depth_completed = 0;
	int f_create_schreier_vector;
	int vl;
	int target_depth;
	int f_write_files;
	int f_embedded = TRUE;
	
	if (f_v) {
		cout << "poset_classification::main" << endl;
		cout << "poset_classification::main ";
		print_problem_label();
		cout << " depth = " << depth << endl;
		cout << "f_W = " << f_W << endl;
		cout << "f_w = " << f_w << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	if (f_recover) {
		if (f_v) {
			cout << "poset_classification::main: recovering from file "
					<< recover_fname << endl;
			}


		int t1, dt;
		t1 = os_ticks();
		dt = t1 - t0;
	
		cout << "Time ";
		time_check_delta(cout, dt);
		cout << endl;


		recover(recover_fname, depth_completed, verbose_level - 1);
		
		if (f_v) {
			cout << "depth_completed = " << depth_completed << endl;
			cout << "poset_classification::main: "
					"recreating schreier vectors "
					"to depth " << depth_completed - 1 << endl;
			}
	
		recreate_schreier_vectors_up_to_level(depth_completed - 1, 
			verbose_level /*MINIMUM(verbose_level, 1)*/);
		}
	if (f_print_only) {
		print_tree();
		write_treefile_and_draw_tree(
			fname_base, depth_completed, 
			xmax, ymax, 
			radius, f_embedded, verbose_level - 1);

		return 0;
		}
	if (f_starter) {
		depth_completed = starter_size;
		}
		
	
	if (f_max_depth) {
		target_depth = max_depth;
		}
	else {
		target_depth = depth;
		}
	if (f_v) {
		cout << "poset_classification::main "
				"target_depth=" << target_depth << endl;
		}
	
	for (size = depth_completed; size < target_depth; size++) {

		int f_write_candidate_file = FALSE;

#if 1
		if (f_W && size) {
			f_write_candidate_file = TRUE;
			}

		if (f_w && size == target_depth - 1) {
			f_write_candidate_file = TRUE;
			}
#endif
		if (f_v) {
			cout << "poset_classification::main: ";
			print_problem_label();
			cout << " calling extend_level " << size
					<< " f_write_candidate_file="
					<< f_write_candidate_file << endl;
			}

		if (size <= schreier_depth) {
			f_create_schreier_vector = TRUE;
			if (f_v) {
				cout << "we will store schreier vectors "
						"for this level" << endl;
				}
			}
		else {
			if (f_v) {
				cout << "we will NOT store schreier vectors "
						"for this level" << endl;
				}
			f_create_schreier_vector = FALSE;
			}

#if 0
		if (size == 1) {
			verbose_level += 10;
			}
#endif

		extend_level(size,
			f_create_schreier_vector, 
			f_use_invariant_subset_if_available, 
			f_debug, 
			f_write_candidate_file, 
			verbose_level - 2);
		
		
		if (size + 1 == sz) {
			vl = verbose_level;
			}
		else {
			vl = verbose_level - 1;
			}

		f_write_files = (f_W || (f_w && size == target_depth - 1));
	
		
		if (f_write_data_files) {
			housekeeping(size + 1, f_write_files,
					os_ticks(), verbose_level - 1);
		}
		else {
			housekeeping_no_data_file(size + 1,
					os_ticks(), verbose_level - 1);
		}


		int nb_nodes;
		nb_nodes = nb_orbits_at_level(size + 1);
		if (nb_nodes == 0) {
			int j;
			for (j = size + 2; j <= target_depth + 1; j++) {
				first_poset_orbit_node_at_level[j] =
						first_poset_orbit_node_at_level[j - 1];
				}
			return size + 1;
			}	
			
		} // next size
	if (f_v) {
		cout << "poset_classification::main done" << endl;
		}
	return size;
}

void poset_classification::extend_level(int size,
	int f_create_schreier_vector, 
	int f_use_invariant_subset_if_available, 
	int f_debug, 
	int f_write_candidate_file, 
	int verbose_level)
// calls downstep, upstep
{
	int f_v = (verbose_level >= 1);
	//int f, cur; //, l;

	if (f_v) {
		cout << "####################################################"
				"##############################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "poset_classification::extend_level "
				"constructing orbits at depth "
				<< size + 1 << endl;
		cout << "poset_classification::extend_level from "
				<< nb_orbits_at_level(size)
				<< " nodes at depth " << size << endl;
		//cout << "f_create_schreier_vector="
		//<< f_create_schreier_vector << endl;
		//cout << "f_use_invariant_subset_if_available="
		//<< f_use_invariant_subset_if_available << endl;
		//cout << "f_debug=" << f_debug << endl;
		//cout << "f_write_candidate_file="
		//<< f_write_candidate_file << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}

	if (f_v) {
		cout << "poset_classification::extend_level size = " << size
				<< " calling compute_flag_orbits" << endl;
		}
	compute_flag_orbits(size,
		f_create_schreier_vector,
		f_use_invariant_subset_if_available, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_classification::extend_level size = " << size <<
				"after compute_flag_orbits" << endl;
		}

	if (f_write_candidate_file) {
		if (f_v) {
			cout << "poset_classification::extend_level "
					"size = " << size
					<< " before write_candidates_binary_using_sv" << endl;
			}
		write_candidates_binary_using_sv(fname_base,
				size, t0, verbose_level - 1);
		if (f_v) {
			cout << "poset_classification::extend_level "
					"size = " << size
					<< " after write_candidates_binary_using_sv" << endl;
			}
		}

	if (f_v) {
		cout << "poset_classification::extend_level "
				"calling upstep" << endl;
		}
	upstep(size, 
		f_debug, 
		verbose_level - 1);
	if (f_v) {
		cout << "poset_classification::extend_level "
				"after upstep" << endl;
		}


}

void poset_classification::compute_flag_orbits(int size,
	int f_create_schreier_vector,
	int f_use_invariant_subset_if_available, 
	int verbose_level)
// calls root[prev].downstep_subspace_action 
// or root[prev].downstep
{
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);
	int f, cur, l, prev, u;
	int f_print = f_v;
	double progress;

	f = first_poset_orbit_node_at_level[size];
	cur = first_poset_orbit_node_at_level[size + 1];
	l = cur - f;

	if (f_v) {
		cout << "################################################"
				"##################################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "compute_flag_orbits depth " << size
				<< " creating orbits at level " << size + 1
				<< " verbose_level=" << verbose_level << endl;
		}
	progress_last_time = 0;
	progress = 0;
	
	for (u = 0; u < l; u++) {
		

		
		prev = f + u;
		
		if (f_print) {
			print_level_info(size, prev);
			cout << " compute_flag_orbits node " << u << " / " << l
					<< " starting" << endl;
			}
			
		if (Poset->f_subspace_lattice) {
			root[prev].compute_flag_orbits_subspace_action(this, size,
				f_create_schreier_vector,
				f_use_invariant_subset_if_available, 
				f_lex, 
				verbose_level - 2);
			}
		else {
			root[prev].compute_flag_orbits(this, size,
				f_create_schreier_vector,
				f_use_invariant_subset_if_available, 
				f_lex, 
				verbose_level - 2);
			}
		if (f_print) {
			//cout << endl;
			print_level_info(size, prev);
			cout << " compute_flag_orbits node " << u << " / " << l
					<< " finished : ";
			if (root[prev].Schreier_vector) {
				//int nb = root[prev].sv[0];
				int nb = root[prev].get_nb_of_live_points();
				cout << " found " << nb << " live points in "
					<< root[prev].nb_extensions << " orbits : ";
				}
			if (f_v3) {
				root[prev].print_extensions(this);
				}
			print_progress(progress);
			//cout << endl;
			}

		progress = (double) u / (double) l;

		if (f_v && ABS(progress - progress_last_time) > progress_epsilon) {
			f_print = TRUE;
			progress_last_time = progress;
			}
		else {
			f_print = FALSE;
			}
		
		
		}
		
}

void poset_classification::upstep(int size, 
	int f_debug, 
	int verbose_level)
// calls extend_node
{
	int f_v = (verbose_level >= 1);
	int f_v4 = (verbose_level >= 4);
	int f, cur, l, prev, u;
	int f_indicate_not_canonicals = FALSE;
	FILE *fp = NULL;
	int f_print = f_v;

	if (f_v) {
		cout << "poset_classification::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	


	f = first_poset_orbit_node_at_level[size];
	cur = first_poset_orbit_node_at_level[size + 1];
	l = cur - f;

	progress_last_time = 0;

	if (f_v) {
		cout << "#################################################"
				"#################################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "extension step depth " << size << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "f_indicate_not_canonicals="
				<< f_indicate_not_canonicals << endl;
		}
	count_extension_nodes_at_level(size);
	if (f_v) {
		cout << "with " << nb_extension_nodes_at_level_total[size]
			<< " extension nodes" << endl;
		}
	for (u = 0; u < l; u++) {

		if (f_v4) {
			cout << "poset_classification::upstep "
					"case " << u << " / " << l << endl;
			}
		prev = f + u;
			
		if (f_print) {
			print_level_info(size, prev);
			cout << " Upstep : " << endl;
			}
		else {
			}

#if 1
		if (f_v4) {
			cout << "poset_classification::upstep "
					"before extend_node" << endl;
			print_extensions_at_level(cout, size);
			}
#endif

		extend_node(size, prev, cur, 
			f_debug, 
			f_indicate_not_canonicals, 
			fp, 
			verbose_level - 2);

#if 1
		if (f_v4) {
			cout << "poset_classification::upstep "
					"after extend_node, size="
					<< size << endl;
			}
#endif
			
		double progress;
	
	
		progress = level_progress(size);

		if (f_print) {
			print_level_info(size, prev);
			cout << " Upstep : ";
			print_progress(progress);
			}

		if (f_v &&
				ABS(progress - progress_last_time) > progress_epsilon) {
			f_print = TRUE;
			progress_last_time = progress;
			}
		else {
			f_print = FALSE;
			}

		}

	first_poset_orbit_node_at_level[size + 2] = cur;
	nb_poset_orbit_nodes_used = cur;




}

void poset_classification::extend_node(
	int size, int prev, int &cur,
	int f_debug, 
	int f_indicate_not_canonicals, 
	FILE *fp, 
	int verbose_level)
// called by poset_classification::upstep
// Uses an upstep_work structure to handle the work.
// Calls upstep_work::handle_extension
{
	int nb_fuse_cur, nb_ext_cur, prev_ex;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);

	if (f_v4) {
		cout << "poset_classification::extend_node prev=" << prev
				<< " cur=" << cur << endl;
		}
	
	while (cur + root[prev].nb_extensions + 10 >=
			nb_poset_orbit_nodes_allocated) {
		print_level_info(size, prev);
		if (f_v) {
			cout << "poset_classification::extend_node "
					"running out of nodes" << endl;
			cout << "cur = " << cur << endl;
			cout << "allocated nodes = "
					<< nb_poset_orbit_nodes_allocated << endl;
			cout << "reallocating" << endl;
			}
		reallocate();
		if (f_v) {
			cout << "allocated nodes = "
					<< nb_poset_orbit_nodes_allocated << endl;
			}
		}
			
	nb_fuse_cur = 0;
	nb_ext_cur = 0;
			
	if (f_vv) {
		longinteger_object go;
		
		print_level_info(size, prev);
		//cout << "Level " << size << " Node " << cur << " : ";
		cout << " extending set ";

		print_set(prev);
		if (root[prev].Schreier_vector) {
			//int nb = root[prev].sv[0];
			int nb = root[prev].get_nb_of_live_points();
			cout << " with " << nb << " live points" << endl;
#if 0
			if (f_vvv && root[prev].Schreier_vector) {
				cout << " : ";
				int_vec_print(cout, root[prev].sv + 1, nb);
				cout << endl;
				}
			else {
				cout << endl;
				}
#endif
			}

		cout << " with " << root[prev].nb_extensions
				<< " extensions" << endl;
		cout << " verbose_level=" << verbose_level << endl;
		if (f_vvv) {
			cout << "poset_classification::extend_node prev=" << prev
					<< " cur=" << cur << " extensions:" << endl;
			//print_set_verbose(prev);
			//root[prev].print_node(this);
			root[prev].print_extensions(this);
			}
		}





	for (prev_ex = 0; prev_ex < root[prev].nb_extensions; prev_ex++) {
		

		if (f_vvv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ":" << endl;
			}
	

		{
		upstep_work Work;


#if 0
		if (FALSE /*prev == 32 && prev_ex == 3*/) { 
			cout << "poset_classification::extend_node "
					"we are at node (32,3)" << endl;
			verbose_level_down = verbose_level + 20; 
			}
		else {
			verbose_level_down = verbose_level - 2;
			}
#endif
		//verbose_level_down = verbose_level - 4;

		if (f_vvv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ": before Work.init" << endl;
			}

		Work.init(this, size, prev, prev_ex, cur, 
			f_debug, 
			f_lex, 
			f_indicate_not_canonicals, fp, 
			verbose_level - 4);

		if (f_vvv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ": after Work.init" << endl;
			}
		


		if (f_vvv) {
			if ((prev_ex % Work.mod_for_printing) == 0 && prev_ex) {
				print_progress_by_extension(size, cur,
						prev, prev_ex, nb_ext_cur, nb_fuse_cur);
				}
			}
		if (f_vvv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ": before Work.handle_extension nb_ext_cur="
					<< nb_ext_cur << endl;
			}
		Work.handle_extension(nb_fuse_cur, nb_ext_cur, 
			verbose_level - 4);
		// in upstep_work.cpp

		if (f_vvv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node after "
					"Work.handle_extension" << endl;
			}

		
		cur = Work.cur;

		} // end of upstep_work Work


		if (f_vvv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ":" << endl;
			cout << "poset_classification::extend_node "
					"after freeing Work" << endl;
			}

		}
			
			
	if (f_v) {

		print_progress(size, cur, prev, nb_ext_cur, nb_fuse_cur);
		//cout << "cur=" << cur << endl;

		}
}

}}



