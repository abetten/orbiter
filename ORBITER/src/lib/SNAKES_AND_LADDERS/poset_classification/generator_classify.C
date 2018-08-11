// generator_classify.C
//
// Anton Betten
//
// moved here from generator.C
// July 19, 2014


#include "GALOIS/galois.h"
#include "ACTION/action.h"
#include "SNAKES_AND_LADDERS/snakesandladders.h"

INT generator::compute_orbits(INT from_level, INT to_level, 
	INT f_write_candidate_file, 
	INT verbose_level)
// returns TRUE if there is at least one orbit at level to_level,
// FALSE otherwise
{
	INT f_v = (verbose_level >= 1);
	INT level;
	INT f_create_schreier_vector = TRUE;
	INT f_compact = TRUE;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	INT f_write_files;


	if (f_v) {
		cout << "generator::compute_orbits from "
				<< from_level << " to " << to_level << endl;
		cout << "f_lex=" << f_lex << endl;
		cout << "f_write_candidate_file="
				<< f_write_candidate_file << endl;
		cout << "fname_base=" << fname_base << endl;
		}


	for (level = from_level; level < to_level; level++) {

		if (f_v) {
			cout << "generator::compute_orbits: ";
			print_problem_label();
			cout << " calling extend_level " << level << endl;
			}


		extend_level(level, 
			f_create_schreier_vector, 
			f_compact, 
			f_use_invariant_subset_if_available, 
			f_debug,
			f_write_candidate_file, 
			verbose_level - 2);
		
		
		f_write_files = (f_W || (f_w && level == to_level - 1));
	
		
		housekeeping(level + 1, f_write_files,
				os_ticks(), verbose_level - 1);

		INT nb_nodes;
		nb_nodes = nb_orbits_at_level(level + 1);
		if (nb_nodes == 0) {
			INT j;
			for (j = level + 2; j <= to_level + 1; j++) {
				first_oracle_node_at_level[j] =
						first_oracle_node_at_level[j - 1];
				}
			return level;
			}	
			
		} // next level

	
	if (f_v) {
		cout << "generator::compute_orbits from "
				<< from_level << " to " << to_level << " done" << endl;
		}
	return to_level;
}

INT generator::main(INT t0, 
	INT schreier_depth, 
	INT f_use_invariant_subset_if_available, 
	INT f_debug, 
	INT verbose_level)
// f_use_invariant_subset_if_available
// is an option that affects the downstep.
// if FALSE, the orbits of the stabilizer on all points are computed. 
// if TRUE, the orbits of the stabilizer on the set of points that were 
// possible in the previous level are computed only 
// (using Schreier.orbits_on_invariant_subset_fast).
// The set of possible points is stored 
// inside the schreier vector data structure (sv).
{
	INT f_v = (verbose_level >= 1);
	INT i, depth_completed = 0;
	INT f_create_schreier_vector;
	INT f_compact = TRUE;
	INT vl;
	INT target_depth;
	INT f_write_files;
	INT f_embedded = TRUE;
	
	if (f_v) {
		cout << "generator::main" << endl;
		cout << "generator::main ";
		print_problem_label();
		cout << " depth = " << depth << endl;
		cout << "f_W = " << f_W << endl;
		cout << "f_w = " << f_w << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	if (f_recover) {
		if (f_v) {
			cout << "generator::main: recovering from file "
					<< recover_fname << endl;
			}


		INT t1, dt;
		t1 = os_ticks();
		dt = t1 - t0;
	
		cout << "Time ";
		time_check_delta(cout, dt);
		cout << endl;


		recover(recover_fname, depth_completed, verbose_level - 1);
		
		if (f_v) {
			cout << "depth_completed = " << depth_completed << endl;
			cout << "generator::main: recreating schreier vectors "
					"to depth " << depth_completed - 1 << endl;
			}
	
		recreate_schreier_vectors_up_to_level(depth_completed - 1, 
			TRUE /* f_compact */, 
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
		cout << "generator::main target_depth=" << target_depth << endl;
		}
	
	for (i = depth_completed; i < target_depth; i++) {

		INT f_write_candidate_file = FALSE;

#if 1
		if (f_W && i) {
			f_write_candidate_file = TRUE;
			}

		if (f_w && i == target_depth - 1) {
			f_write_candidate_file = TRUE;
			}
#endif
		if (f_v) {
			cout << "generator::main: ";
			print_problem_label();
			cout << " calling extend_level " << i
					<< " f_write_candidate_file="
					<< f_write_candidate_file << endl;
			}

		if (i <= schreier_depth) {
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
		if (i == 1) {
			verbose_level += 10;
			}
#endif

		extend_level(i, 
			f_create_schreier_vector, 
			f_compact, 
			f_use_invariant_subset_if_available, 
			f_debug, 
			f_write_candidate_file, 
			verbose_level - 2);
		
		
		if (i + 1 == sz) {
			vl = verbose_level;
			}
		else {
			vl = verbose_level - 1;
			}

		f_write_files = (f_W || (f_w && i == target_depth - 1));
	
		
		housekeeping(i + 1, f_write_files, t0, vl);

		INT nb_nodes;
		nb_nodes = nb_orbits_at_level(i + 1);	
		if (nb_nodes == 0) {
			INT j;
			for (j = i + 2; j <= target_depth + 1; j++) {
				first_oracle_node_at_level[j] =
						first_oracle_node_at_level[j - 1];
				}
			return i + 1;
			}	
			
		} // next i
	if (f_v) {
		cout << "generator::main done" << endl;
		}
	return i;
}

void generator::extend_level(INT size,
	INT f_create_schreier_vector, 
	INT f_compact, 
	INT f_use_invariant_subset_if_available, 
	INT f_debug, 
	INT f_write_candidate_file, 
	INT verbose_level)
// calls downstep, upstep
{
	INT f_v = (verbose_level >= 1);
	//INT f, cur; //, l;

	if (f_v) {
		cout << "####################################################"
				"##############################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "generator::extend_level constructing nodes at depth "
				<< size + 1 << endl;
		cout << "generator::extend_level from "
				<< nb_orbits_at_level(size)
				<< " nodes at depth " << size << endl;
		//cout << "f_create_schreier_vector="
		//<< f_create_schreier_vector << endl;
		//cout << "f_use_invariant_subset_if_available="
		//<< f_use_invariant_subset_if_available << endl;
		//cout << "f_compact=" << f_compact << endl;
		//cout << "f_debug=" << f_debug << endl;
		//cout << "f_write_candidate_file="
		//<< f_write_candidate_file << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	//f = first_oracle_node_at_level[size];
	//cur = first_oracle_node_at_level[size + 1];
	//l = cur - f;

	if (f_v) {
		cout << "generator::extend_level " << size
				<< " calling downstep" << endl;
		}
	downstep(size, f_create_schreier_vector, f_compact, 
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		verbose_level - 1);
	if (f_v) {
		cout << "generator::extend_level after downstep" << endl;
		}

	if (f_write_candidate_file) {
		if (f_v) {
			cout << "generator::extend_level size = " << size
					<< " before write_candidates_binary_using_sv" << endl;
			}
		write_candidates_binary_using_sv(fname_base,
				size, t0, 1 /*verbose_level */);
		}

	if (f_v) {
		cout << "generator::extend_level calling upstep" << endl;
		}
	upstep(size, 
		f_debug, 
		verbose_level - 1);
	if (f_v) {
		cout << "generator::extend_level after upstep" << endl;
		}


}

void generator::downstep(INT size, 
	INT f_create_schreier_vector, INT f_compact, 
	INT f_use_invariant_subset_if_available, 
	INT verbose_level)
// calls root[prev].downstep_subspace_action 
// or root[prev].downstep
{
	INT f_v = (verbose_level >= 1);
	INT f_v3 = (verbose_level >= 3);
	INT f, cur, l, prev, u;
	INT f_print = f_v;
	double progress;

	f = first_oracle_node_at_level[size];
	cur = first_oracle_node_at_level[size + 1];
	l = cur - f;

	if (f_v) {
		cout << "################################################"
				"##################################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "downstep depth " << size
				<< " verbose_level=" << verbose_level << endl;
		}
	progress_last_time = 0;
	progress = 0;
	
	for (u = 0; u < l; u++) {
		

		
		prev = f + u;
		
		if (f_print) {
			print_level_info(size + 1, prev);
			cout << " Downstep node starting" << endl;
			}
			
		if (f_on_subspaces) {
			root[prev].downstep_subspace_action(this, size, 
				f_create_schreier_vector, f_compact, 
				f_use_invariant_subset_if_available, 
				f_lex, 
				verbose_level - 2);
			}
		else {
			root[prev].downstep(this, size, 
				f_create_schreier_vector, f_compact, 
				f_use_invariant_subset_if_available, 
				f_lex, 
				verbose_level - 2);
			}
		if (f_print) {
			cout << endl;
			//print_level_info(size + 1, prev);
			cout << "Downstep node finished : ";
			if (root[prev].sv) {
				INT nb = root[prev].sv[0];
				cout << " found " << nb << " live points in "
					<< root[prev].nb_extensions << " orbits : ";
				}
			if (f_v3) {
				root[prev].print_extensions(this);
				}
			print_progress(size + 1, progress);
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

void generator::upstep(INT size, 
	INT f_debug, 
	INT verbose_level)
// calls extend_node
{
	INT f_v = (verbose_level >= 1);
	INT f_v4 = (verbose_level >= 4);
	INT f, cur, l, prev, u;
	INT f_indicate_not_canonicals = FALSE;
	FILE *fp = NULL;
	INT f_print = f_v;

	if (f_v) {
		cout << "generator::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	


	f = first_oracle_node_at_level[size];
	cur = first_oracle_node_at_level[size + 1];
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
			cout << "generator::upstep case " << u << " / " << l << endl;
			}
		prev = f + u;
			
		if (f_print) {
			print_level_info(size + 1, prev);
			cout << " Upstep : " << endl;
			}
		else {
			}

#if 1
		if (f_v4) {
			cout << "generator::upstep before extend_node" << endl;
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
			cout << "generator::upstep after extend_node, size="
					<< size << endl;
			}
#endif
			
		double progress;
	
	
		progress = level_progress(size);

		if (f_print) {
			print_level_info(size + 1, prev);
			cout << " Upstep : ";
			print_progress(size + 1, progress);
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

	first_oracle_node_at_level[size + 2] = cur;
	nb_oracle_nodes_used = cur;




}

void generator::extend_node(INT size, INT prev, INT &cur, 
	INT f_debug, 
	INT f_indicate_not_canonicals, 
	FILE *fp, 
	INT verbose_level)
// called by generator::upstep
// Uses an upstep_work structure to handle the work.
// Calls upstep_work::handle_extension
{
	INT nb_fuse_cur, nb_ext_cur, prev_ex;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_v4 = (verbose_level >= 4);
	INT verbose_level_down;

	if (f_v4) {
		cout << "generator::extend_node prev=" << prev
				<< " cur=" << cur << endl;
		}
	
	while (cur + root[prev].nb_extensions + 10 >=
			nb_oracle_nodes_allocated) {
		print_level_info(size + 1, prev);
		if (f_v) {
			cout << "generator::extend_node running out of nodes" << endl;
			cout << "cur = " << cur << endl;
			cout << "allocated nodes = "
					<< nb_oracle_nodes_allocated << endl;
			cout << "reallocating" << endl;
			}
		reallocate();
		if (f_v) {
			cout << "allocated nodes = "
					<< nb_oracle_nodes_allocated << endl;
			}
		}
			
	nb_fuse_cur = 0;
	nb_ext_cur = 0;
			
	if (f_vv) {
		longinteger_object go;
		
		print_level_info(size + 1, prev);
		//cout << "Level " << size << " Node " << cur << " : ";
		cout << " extending set ";

		print_set(prev);
		if (root[prev].sv) {
			INT nb = root[prev].sv[0];
			cout << " with " << nb << " live points";
#if 1
			if (f_vvv && root[prev].sv) {
				cout << " : ";
				INT_vec_print(cout, root[prev].sv + 1, nb);
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
		if (FALSE /*f_vvv*/) {
			//print_set_verbose(prev);
			//root[prev].print_node(this);
			root[prev].print_extensions(this);
			}
		}





	for (prev_ex = 0; prev_ex < root[prev].nb_extensions; prev_ex++) {
		

		if (f_vvv) {
			print_level_info(size + 1, prev);
			cout << "generator::extend_node working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ":" << endl;
			}
	

		{
		upstep_work Work;


#if 0
		if (FALSE /*prev == 32 && prev_ex == 3*/) { 
			cout << "generator::extend_node we are at node (32,3)" << endl;
			verbose_level_down = verbose_level + 20; 
			}
		else {
			verbose_level_down = verbose_level - 2;
			}
#endif
		verbose_level_down = verbose_level - 4;

		if (f_vvv) {
			print_level_info(size + 1, prev);
			cout << "generator::extend_node working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ": before Work.init" << endl;
			}

		Work.init(this, size, prev, prev_ex, cur, 
			f_debug, 
			f_lex, 
			f_indicate_not_canonicals, fp, 
			verbose_level_down);

		if (f_vvv) {
			print_level_info(size + 1, prev);
			cout << "generator::extend_node working on extension "
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
			print_level_info(size + 1, prev);
			cout << "generator::extend_node working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ": before Work.handle_extension nb_ext_cur="
					<< nb_ext_cur << endl;
			}
		Work.handle_extension(nb_fuse_cur, nb_ext_cur, 
			verbose_level_down);

		if (f_vvv) {
			print_level_info(size + 1, prev);
			cout << "generator::extend_node after "
					"Work.handle_extension" << endl;
			}

		
		cur = Work.cur;

		} // end of upstep_work Work


		if (f_vvv) {
			print_level_info(size + 1, prev);
			cout << "generator::extend_node working on extension "
					<< prev_ex << " / " << root[prev].nb_extensions
					<< ":" << endl;
			cout << "generator::extend_node after freeing Work" << endl;
			}

		}
			
			
	if (f_v) {

		print_progress(size, cur, prev, nb_ext_cur, nb_fuse_cur);
		//cout << "cur=" << cur << endl;

		}
}


