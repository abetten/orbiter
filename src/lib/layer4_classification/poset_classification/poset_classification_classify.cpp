// poset_classification_classify.cpp
//
// Anton Betten
//
// moved here from poset_classification.cpp
// July 19, 2014


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


void poset_classification::compute_orbits_on_subsets(
	int target_depth,
	poset_classification_control *PC_control,
	poset_with_group_action *Poset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int nb_poset_orbit_nodes = 1000;
	int schreier_depth = target_depth;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	other::orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();


	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"verbose_level=" << verbose_level << endl;
	}

	depth = target_depth;
	//downstep_orbits_print_max_orbits = 50;
	//downstep_orbits_print_max_points_per_orbit = INT_MAX;


	// !!!
	//f_allowed_to_show_group_elements = false;

	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"calling initialize_and_allocate_root_node" << endl;
	}
	initialize_and_allocate_root_node(PC_control,
		Poset,
		target_depth, verbose_level - 1);


	//init_poset_orbit_node(nb_poset_orbit_nodes, verbose_level - 1);
	//init_root_node(verbose_level - 1);

	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"calling main" << endl;
	}
	main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);
	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"after main" << endl;
	}

	int i, fst, len;

	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets done" << endl;
		cout << "depth : number of orbits" << endl;
	}
	for (i = 0; i < target_depth + 1; i++) {
		fst = Poo->first_node_at_level(i);
		len = Poo->first_node_at_level(i + 1) - fst;
		if (f_v) {
			cout << i << " : " << len << endl;
		}
	}

	long int N, F, level;

	N = 0;
	F = 0;
	for (level = 0; level <= target_depth; level++) {
		N += nb_orbits_at_level(level);
	}
	for (level = 0; level < target_depth; level++) {
		F += nb_flag_orbits_up_at_level(level);
	}
	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets N=" << N << endl;
		cout << "poset_classification::compute_orbits_on_subsets F=" << F << endl;
	}

	if (f_v) {
		cout << "poset_classification::compute_orbits_on_subsets "
				"done" << endl;
	}
}



int poset_classification::main(
		int t0,
	int schreier_depth, 
	int f_use_invariant_subset_if_available, 
	int f_debug, 
	int verbose_level)
// f_use_invariant_subset_if_available
// is an option that affects the downstep.
// if false, the orbits of the stabilizer on all points are computed. 
// if true, the orbits of the stabilizer on the set of points that were 
// possible in the previous level are computed only 
// (using Schreier.orbits_on_invariant_subset_fast).
// The set of possible points is stored 
// inside the schreier vector data structure (sv).
{
	int f_v = (verbose_level >= 1);
	int size, depth_completed = 0;
	//int f_create_schreier_vector;
	int target_depth;
	//int f_write_files;
	//int f_embedded = true;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "poset_classification::main" << endl;
		cout << "poset_classification::main ";
		print_problem_label();
		cout << " depth = " << depth << endl;
		cout << "f_W = " << Control->f_W << endl;
		cout << "f_w = " << Control->f_w << endl;
		cout << "verbose_level = " << verbose_level << endl;
		Control->print();
	}
	if (Control->f_recover) {
		if (f_v) {
			cout << "poset_classification::main: recovering from file "
					<< Control->recover_fname << endl;
		}


		int t1, dt;
		t1 = Os.os_ticks();
		dt = t1 - t0;
	
		cout << "Time ";
		Os.time_check_delta(cout, dt);
		cout << endl;


		recover(Control->recover_fname, depth_completed, verbose_level - 1);
		
		if (f_v) {
			cout << "depth_completed = " << depth_completed << endl;
			cout << "poset_classification::main: "
					"recreating schreier vectors "
					"to depth " << depth_completed - 1 << endl;
		}
	
		recreate_schreier_vectors_up_to_level(depth_completed - 1, 
			verbose_level /*MINIMUM(verbose_level, 1)*/);
	}
	if (f_base_case) {
		depth_completed = Base_case->size;
	}
		
	
	if (Control->f_depth) {
		target_depth = Control->depth;
	}
	else {
		target_depth = depth;
	}
	if (f_v) {
		cout << "poset_classification::main "
				"target_depth=" << target_depth << endl;
	}
	


	if (f_v) {
		cout << "poset_classification::main before compute_orbits" << endl;
	}
	size = compute_orbits(
			depth_completed, target_depth,
			schreier_depth,
			f_use_invariant_subset_if_available,
			verbose_level);
	if (f_v) {
		cout << "poset_classification::main after compute_orbits" << endl;
	}



#if 0
	if (f_v) {
		cout << "poset_classification::main before post_processing" << endl;
	}
	post_processing(size, verbose_level);
	if (f_v) {
		cout << "poset_classification::main after post_processing" << endl;
	}
#endif

	if (f_v) {
		cout << "poset_classification::main done" << endl;
	}
	return size;
}

int poset_classification::compute_orbits(
		int from_level, int to_level,
		int schreier_depth,
		int f_use_invariant_subset_if_available,
		int verbose_level)
// returns the last level that has at least one orbit
{
	int f_v = (verbose_level >= 1);
	int level;
	int f_create_schreier_vector = true;
	int f_debug = false;
	int f_write_files;
	other::orbiter_kernel_system::os_interface Os;


	if (f_v) {
		cout << "poset_classification::compute_orbits ";
		print_problem_label();
		cout << " from " << from_level << " to " << to_level << endl;
		cout << "f_lex=" << Control->f_lex << endl;
		cout << "problem_label_with_path=" << problem_label_with_path << endl;
		cout << "schreier_depth=" << schreier_depth << endl;
		cout << "f_use_invariant_subset_if_available=" << f_use_invariant_subset_if_available << endl;
		cout << "poset_classification_control:" << endl;
		Control->print();
	}


	for (level = from_level; level < to_level; level++) {

		if (f_v) {
			cout << "poset_classification::compute_orbits: ";
			print_problem_label();
			cout << " level " << level << endl;
		}

		int f_write_candidate_file = false;

#if 1
		if (Control->f_W && level) {
			f_write_candidate_file = true;
		}

		if (Control->f_w && level == to_level - 1) {
			f_write_candidate_file = true;
		}
#endif

		if (level <= schreier_depth) {
			f_create_schreier_vector = true;
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"we will store schreier vectors "
						"for this level" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"we will NOT store schreier vectors "
						"for this level" << endl;
			}
			f_create_schreier_vector = false;
		}

		if (f_v) {
			cout << "poset_classification::compute_orbits: ";
			print_problem_label();
			cout << " before extend_level" << endl;
		}
		extend_level(level,
			f_create_schreier_vector,
			f_use_invariant_subset_if_available,
			f_debug,
			f_write_candidate_file,
			verbose_level - 1);
		if (f_v) {
			cout << "poset_classification::compute_orbits: ";
			print_problem_label();
			cout << " after extend_level" << endl;
		}


		f_write_files = (Control->f_W || (Control->f_w && level == to_level - 1));


		if (f_write_files) {

			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"before write_reps_csv" << endl;
			}
			write_reps_csv(level + 1, verbose_level - 1);
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"after write_reps_csv" << endl;
			}
		}

		if (Control->f_write_data_files) {
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"before housekeeping f_write_files = true" << endl;
			}
			housekeeping(level + 1, f_write_files,
					Os.os_ticks(), verbose_level - 1);
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"after housekeeping" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"before housekeeping_no_data_file" << endl;
			}
			housekeeping_no_data_file(level + 1,
					Os.os_ticks(), verbose_level - 1);
			if (f_v) {
				cout << "poset_classification::compute_orbits "
						"after housekeeping_no_data_file" << endl;
			}
		}


		int nb_nodes;
		nb_nodes = nb_orbits_at_level(level + 1);
		if (nb_nodes == 0) {
			int j;
			for (j = level + 2; j <= to_level + 1; j++) {
				Poo->set_first_node_at_level(j,
						Poo->first_node_at_level(j - 1));
				}
			break;
		}

	} // next level


	if (f_v) {
		cout << "poset_classification::compute_orbits from "
				<< from_level << " to " << to_level << " done, "
						"last level with nodes is " << level << endl;
	}
	return level;
}

#if 0
void poset_classification::post_processing(int actual_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::post_processing problem_label_with_path=" << problem_label_with_path << " verbose_level=" << verbose_level << endl;
	}

	if (Control->f_write_tree) {
		Poo->print_tree();
		write_treefile(
				problem_label_with_path, depth,
				Control->draw_options,
				verbose_level - 1);

		//return 0;
	}
	if (Control->f_table_of_nodes) {
		if (f_v) {
			cout << "poset_classification::post_processing f_table_of_nodes" << endl;
		}
		Poo->make_tabe_of_nodes(verbose_level);
	}

	if (Control->f_list_all) {
		if (f_v) {
			cout << "poset_classification::post_processing f_list_all" << endl;
		}

		int d;

		for (d = 0; d <= depth; d++) {
			cout << "There are " << nb_orbits_at_level(d)
					<< " orbits on subsets of size " << d << ":" << endl;

#if 0
			if (d < Descr->orbits_on_subsets_size) {
				//continue;
			}
#endif

			list_all_orbits_at_level(d,
					false /* f_has_print_function */,
					NULL /* void (*print_function)(std::ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					Control->f_show_orbit_decomposition /* f_show_orbit_decomposition */,
					Control->f_show_stab /* f_show_stab */,
					Control->f_save_stab /* f_save_stab */,
					Control->f_show_whole_orbits /* f_show_whole_orbit*/);
		}
	}

	if (Control->f_list) {
		if (f_v) {
			cout << "poset_classification::post_processing f_list" << endl;
		}
#if 1
		//int f_show_orbit_decomposition = true;
		//int f_show_stab = true;
		//int f_save_stab = true;
		//int f_show_whole_orbit = false;

		if (f_v) {
			cout << "poset_classification::post_processing before "
					"list_all_orbits_at_level" << endl;
		}
		list_all_orbits_at_level(actual_size,
			false,
			NULL,
			this,
			Control->f_show_orbit_decomposition,
			Control->f_show_stab,
			Control->f_save_stab,
			Control->f_show_whole_orbits);

		if (f_v) {
			cout << "poset_classification::post_processing after "
					"list_all_orbits_at_level" << endl;
		}

#if 0
		int d;
		for (d = 0; d < 3; d++) {
			gen->print_schreier_vectors_at_depth(d, verbose_level);
		}
#endif
#endif
	}

	if (Control->f_level_summary_csv) {
		if (f_v) {
			cout << "poset_classification::post_processing preparing level spreadsheet" << endl;
		}
		{
			data_structures::spreadsheet *Sp;
			make_spreadsheet_of_level_info(Sp, actual_size, verbose_level);
			string fname_csv;

			fname_csv = problem_label_with_path + "_levels_" + std::to_string(actual_size) + ".csv";
			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
		if (f_v) {
			cout << "poset_classification::post_processing preparing level spreadsheet done" << endl;
		}
	}


	if (Control->f_orbit_reps_csv) {
		if (f_v) {
			cout << "poset_classification::post_processing preparing orbit spreadsheet" << endl;
		}
		{
			data_structures::spreadsheet *Sp;
			make_spreadsheet_of_orbit_reps(Sp, actual_size);
			string fname_csv;

			fname_csv = problem_label_with_path + "_orbits_at_level_" + std::to_string(actual_size) + ".csv";
			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
		if (f_v) {
			cout << "poset_classification::post_processing preparing orbit spreadsheet done" << endl;
		}
	}


	if (Control->f_draw_poset) {
		if (f_v) {
			cout << "poset_classification::post_processing before draw_poset" << endl;
		}
		if (!Control->f_draw_options) {
			cout << "poset_classification::post_processing Control->f_draw_poset && !Control->f_draw_options" << endl;
			exit(1);
		}
		draw_poset(get_problem_label_with_path(), actual_size,
			0 /* data1 */,
			Control->draw_options,
			verbose_level);
		if (f_v) {
			cout << "poset_classification::post_processing after draw_poset" << endl;
		}
	}

	if (Control->f_draw_full_poset) {
		if (f_v) {
			cout << "poset_classification::post_processing before draw_full_poset" << endl;
		}
		draw_poset_full(get_problem_label_with_path(), actual_size,
				0 /* data1 */,
				Control->draw_options,
				1 /* x_stretch */, verbose_level);
		if (f_v) {
			cout << "poset_classification::post_processing after draw_full_poset" << endl;
		}
	}
	if (Control->f_make_relations_with_flag_orbits) {
			string fname_prefix;


			fname_prefix = problem_label_with_path + "_flag_orbits";

			if (f_v) {
				cout << "poset_classification::post_processing before make_flag_orbits_on_relations" << endl;
			}
			make_flag_orbits_on_relations(
					depth, fname_prefix, verbose_level);
			if (f_v) {
				cout << "poset_classification::post_processing after make_flag_orbits_on_relations" << endl;
			}
	}
	if (Control->f_print_data_structure) {
		if (f_v) {
			cout << "poset_classification::post_processing f_print_data_structure" << endl;
		}
		print_data_structure_tex(actual_size, verbose_level);
	}


	if (Control->f_report) {

		if (f_v) {
			cout << "poset_classification::post_processing f_report" << endl;
		}

		report(Control->report_options, verbose_level);

	}
	if (Control->f_test_multi_edge_in_decomposition_matrix) {
		test_for_multi_edge_in_classification_graph(depth, verbose_level);
		}

	if (f_v) {
		cout << "poset_classification::post_processing done" << endl;
	}
}
#endif

void poset_classification::recognize(
		std::string &set_to_recognize,
		int h, int nb_to_recognize,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::recognize" << endl;
	}
	long int *recognize_set;
	int recognize_set_sz;
	int orb;
	long int *canonical_set;
	int *Elt_transporter;
	int *Elt_transporter_inv;

	cout << "recognize " << h << " / " << nb_to_recognize << endl;
	Lint_vec_scan(set_to_recognize, recognize_set, recognize_set_sz);
	cout << "input set = " << h << " / " << nb_to_recognize << " : ";
	Lint_vec_print(cout, recognize_set, recognize_set_sz);
	cout << endl;

	canonical_set = NEW_lint(recognize_set_sz);
	Elt_transporter = NEW_int(get_A()->elt_size_in_int);
	Elt_transporter_inv = NEW_int(get_A()->elt_size_in_int);


	data_structures_groups::set_and_stabilizer *SaS_original;
	data_structures_groups::set_and_stabilizer *SaS_canonical;
	int orbit_at_level;


	identify_and_get_stabilizer(
			recognize_set, recognize_set_sz, Elt_transporter,
			orbit_at_level,
			SaS_original,
			SaS_canonical,
			verbose_level);


	orb = trace_set(recognize_set,
		recognize_set_sz, recognize_set_sz /* level */,
		canonical_set, Elt_transporter,
		0 /*verbose_level */);

	cout << "recognize " << h << " / " << nb_to_recognize << endl;
	cout << "canonical set = ";
	Lint_vec_print(cout, canonical_set, recognize_set_sz);
	cout << endl;
	cout << "is orbit " << orb << endl;
	cout << "recognize " << h << " / " << nb_to_recognize << endl;
	cout << "transporter:" << endl;
	get_A()->Group_element->element_print_quick(Elt_transporter, cout);

	get_A()->Group_element->element_invert(Elt_transporter, Elt_transporter_inv, 0);
	cout << "recognize " << h << " / " << nb_to_recognize << endl;
	cout << "transporter inverse:" << endl;
	get_A()->Group_element->element_print_quick(Elt_transporter_inv, cout);

	cout << "Stabilizer of the given set:" << endl;
	SaS_original->print_generators_tex(cout);

	cout << "Stabilizer of the canonical set:" << endl;
	SaS_canonical->print_generators_tex(cout);

	FREE_lint(canonical_set);
	FREE_int(Elt_transporter);
	FREE_int(Elt_transporter_inv);
	FREE_lint(recognize_set);

	if (f_v) {
		cout << "poset_classification::recognize before FREE_OBJECT" << endl;
	}
	FREE_OBJECT(SaS_original);
	FREE_OBJECT(SaS_canonical);

	if (f_v) {
		cout << "poset_classification::recognize done" << endl;
	}
}

void poset_classification::extend_level(
		int size,
	int f_create_schreier_vector, 
	int f_use_invariant_subset_if_available, 
	int f_debug, 
	int f_write_candidate_file, 
	int verbose_level)
// calls downstep, upstep
{
	int f_v = (verbose_level >= 1);

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
		Poo->write_candidates_binary_using_sv(problem_label_with_path,
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

void poset_classification::compute_flag_orbits(
		int size,
	int f_create_schreier_vector,
	int f_use_invariant_subset_if_available, 
	int verbose_level)
// calls root[prev].downstep_subspace_action 
// or root[prev].downstep
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v4 = (verbose_level >= 4);
	int f, cur, l, prev, u;
	int f_print = f_v;
	double progress;

	f = Poo->first_node_at_level(size);
	cur = Poo->first_node_at_level(size + 1);
	l = cur - f;

	if (f_v) {
		cout << "################################################"
				"##################################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "poset_classification::compute_flag_orbits at level " << size
				<< " creating orbits at level " << size + 1
				<< " verbose_level=" << verbose_level << endl;
	}
	progress_last_time = 0;
	progress = 0;
	
	for (u = 0; u < l; u++) {
		

		
		prev = f + u;
		
		if (f_print) {
			print_level_info(size, prev);
			cout << " poset_classification::compute_flag_orbits "
					"level " << size << " node " << u << " / " << l
					<< " starting" << endl;
		}
			
		if (Poset->f_subspace_lattice) {
			if (f_v) {
				cout << "poset_classification::compute_flag_orbits "
						"level " << size << " before compute_flag_orbits_subspace_action" << endl;
			}
			Poo->get_node(prev)->compute_flag_orbits_subspace_action(this, size,
				f_create_schreier_vector,
				f_use_invariant_subset_if_available, 
				Control->f_lex,
				verbose_level - 1);
			if (f_v) {
				cout << "poset_classification::compute_flag_orbits "
						"level " << size << " after compute_flag_orbits_subspace_action" << endl;
			}
		}
		else {
			if (f_v4) {
				cout << "poset_classification::compute_flag_orbits "
						"level " << size << " before compute_flag_orbits" << endl;
			}
			Poo->get_node(prev)->compute_flag_orbits(this, size,
				f_create_schreier_vector,
				f_use_invariant_subset_if_available, 
				Control->f_lex,
				verbose_level - 1);
			if (f_v4) {
				cout << "poset_classification::compute_flag_orbits "
						"level " << size << " after compute_flag_orbits" << endl;
			}
		}
		if (f_print) {
			//cout << endl;
			print_level_info(size, prev);
			cout << " compute_flag_orbits level " << size << " node " << u << " / " << l
					<< " finished : ";
			if (Poo->get_node(prev)->has_Schreier_vector()) {
				//int nb = root[prev].sv[0];
				int nb = Poo->get_node(prev)->get_nb_of_live_points();
				cout << " found " << nb << " live points in "
					<< Poo->node_get_nb_of_extensions(prev) << " orbits : ";

				if (false) {
					int *live_points = Poo->get_node(prev)->live_points();
					cout << "The live points are : ";
					Int_vec_print(cout, live_points, nb);
				}
				cout << endl;
				}
			if (f_vv) {
				print_level_info(size, prev);
				cout << " compute_flag_orbits level " << size << " node " << u << " / " << l
						<< " the extensions are : ";
				if (false) {
					Poo->get_node(prev)->print_extensions(this);
				}
			}
			print_progress(progress);
			//cout << endl;
		}

		progress = (double) u / (double) l;

		if (f_v && ABS(progress - progress_last_time) > progress_epsilon) {
			f_print = true;
			progress_last_time = progress;
		}
		else {
			f_print = false;
		}
		
		
	}

	if (f_v) {
		cout << "poset_classification::compute_flag_orbits at "
				"level " << size << " done" << endl;
	}

}

void poset_classification::upstep(
		int size,
	int f_debug, 
	int verbose_level)
// calls extend_node
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f, cur, l, prev, u;
	int f_indicate_not_canonicals = false;
	int f_print = f_v;

	if (f_v) {
		cout << "poset_classification::upstep at level " << size << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	


	f = Poo->first_node_at_level(size);
	cur = Poo->first_node_at_level(size + 1);
	l = cur - f;

	progress_last_time = 0;

	if (f_v) {
		cout << "#################################################"
				"#################################################" << endl;
		print_problem_label();
		cout << endl;
		cout << "extension step at level " << size << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "f_indicate_not_canonicals="
				<< f_indicate_not_canonicals << endl;
	}
	count_extension_nodes_at_level(size);
	if (f_v) {
		cout << "with " << Poo->get_nb_extension_nodes_at_level_total(size)
			<< " extension nodes" << endl;
	}
	for (u = 0; u < l; u++) {

		if (f_vv) {
			cout << "poset_classification::upstep level " << size <<
					" case " << u << " / " << l << endl;
		}
		prev = f + u;
			
		if (f_print) {
			print_level_info(size, prev);
			cout << " Upstep : " << endl;
		}
		else {
		}

#if 0
		if (f_v4) {
			cout << "poset_classification::upstep "
					"before extend_node" << endl;
			print_extensions_at_level(cout, size);
		}
#endif

		if (f_vv) {
			cout << "poset_classification::upstep level " << size <<
					" case " << u << " / " << l << " before extend_node" << endl;
		}
		extend_node(size, prev, cur, 
			f_debug, 
			f_indicate_not_canonicals, 
			verbose_level - 2);

#if 1
		if (f_vv) {
			cout << "poset_classification::upstep level " << size <<
					" after extend_node, size="
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
			f_print = true;
			progress_last_time = progress;
		}
		else {
			f_print = false;
		}

	}

	Poo->set_first_node_at_level(size + 2, cur);
	Poo->set_nb_poset_orbit_nodes_used(cur);



	if (f_v) {
		cout << "poset_classification::upstep at level " << size << " done" << endl;
	}


}

void poset_classification::extend_node(
	int size, int prev, int &cur,
	int f_debug, 
	int f_indicate_not_canonicals, 
	int verbose_level)
// called by poset_classification::upstep
// Uses an upstep_work structure to handle the work.
// Calls upstep_work::handle_extension
{
	int nb_fuse_cur, nb_ext_cur, prev_ex;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "poset_classification::extend_node prev=" << prev
				<< " cur=" << cur << endl;
	}
	
	while (cur + Poo->node_get_nb_of_extensions(prev) + 10 >=
			Poo->get_nb_poset_orbit_nodes_allocated()) {
		print_level_info(size, prev);
		if (f_v) {
			cout << "poset_classification::extend_node "
					"running out of nodes" << endl;
			cout << "cur = " << cur << endl;
			cout << "allocated nodes = "
					<< Poo->get_nb_poset_orbit_nodes_allocated() << endl;
			cout << "reallocating" << endl;
		}
		Poo->reallocate();
		if (f_v) {
			cout << "allocated nodes = "
					<< Poo->get_nb_poset_orbit_nodes_allocated() << endl;
		}
	}
			
	nb_fuse_cur = 0;
	nb_ext_cur = 0;
			
	if (f_vv) {
		algebra::ring_theory::longinteger_object go;
		
		print_level_info(size, prev);
		//cout << "Level " << size << " Node " << cur << " : ";
		cout << " extending set ";

		print_set(prev);

		if (Poo->get_node(prev)->has_Schreier_vector()) {
			//int nb = root[prev].sv[0];
			int nb = Poo->get_node(prev)->get_nb_of_live_points();
			cout << " with " << nb << " live points" << endl;
		}

		cout << " with " << Poo->node_get_nb_of_extensions(prev)
				<< " extensions" << endl;
		cout << " verbose_level=" << verbose_level << endl;
		if (false) {
			cout << "poset_classification::extend_node prev=" << prev
					<< " cur=" << cur << " extensions:" << endl;
			Poo->get_node(prev)->print_extensions(this);
		}
	}



	int f_show_progress = false;
	if (Poo->node_get_nb_of_extensions(prev) > 10000) {
		f_show_progress = true;
	}
	int nb_flags_10;

	nb_flags_10 = Poo->node_get_nb_of_extensions(prev) / 10 + 1;

	for (prev_ex = 0; prev_ex < Poo->node_get_nb_of_extensions(prev); prev_ex++) {
		
		if (f_show_progress && (prev_ex % nb_flags_10) == 0) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << Poo->node_get_nb_of_extensions(prev)
					<< " : progress " << prev_ex / nb_flags_10 << " * 10 %" << endl;


		}

		if (false) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << Poo->node_get_nb_of_extensions(prev)
					<< ":" << endl;
			}
	

		{
		upstep_work Work;


#if 0
		if (false /*prev == 32 && prev_ex == 3*/) { 
			cout << "poset_classification::extend_node "
					"we are at node (32,3)" << endl;
			verbose_level_down = verbose_level + 20; 
		}
		else {
			verbose_level_down = verbose_level - 2;
		}
#endif
		//verbose_level_down = verbose_level - 4;

		if (f_vv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << Poo->node_get_nb_of_extensions(prev)
					<< ": before Work.init" << endl;
		}

		Work.init(this, size, prev, prev_ex, cur, 
			f_debug, 
			Control->f_lex,
			f_indicate_not_canonicals,
			verbose_level - 2);

		if (f_vv) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << Poo->node_get_nb_of_extensions(prev)
					<< ": after Work.init" << endl;
		}
		


		if (f_vvv) {
			if ((prev_ex % Work.mod_for_printing) == 0 && prev_ex) {
				print_progress_by_extension(size, cur,
						prev, prev_ex, nb_ext_cur, nb_fuse_cur);
			}
		}
		if (false) {
			print_level_info(size, prev);
			cout << "poset_classification::extend_node "
					"working on extension "
					<< prev_ex << " / " << Poo->node_get_nb_of_extensions(prev)
					<< ": before Work.handle_extension nb_ext_cur="
					<< nb_ext_cur << endl;
		}
		Work.handle_extension(nb_fuse_cur, nb_ext_cur, 
			verbose_level - 2);
		// in upstep_work.cpp

		if (false) {
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
					<< prev_ex << " / " << Poo->node_get_nb_of_extensions(prev)
					<< ":" << endl;
			cout << "poset_classification::extend_node "
					"after freeing Work" << endl;
		}

	}
			
			
	if (f_v) {
		print_progress(size, cur, prev, nb_ext_cur, nb_fuse_cur);
		//cout << "cur=" << cur << endl;
	}
	if (f_v) {
		cout << "poset_classification::extend_node prev=" << prev
				<< " cur=" << cur << " done" << endl;
	}
}

}}}



