/*
 * poset_classification_control.cpp
 *
 *  Created on: May 6, 2020
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {



static void poset_classification_control_early_test_function_cliques(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);



poset_classification_control::poset_classification_control()
{

	f_problem_label = FALSE;
	//problem_label = NULL;

	f_path = FALSE;
	//path = NULL;

	f_depth = FALSE;
	depth = 0;

	verbose_level = 0;
	verbose_level_group_theory = 0;

	f_recover = FALSE;
	//recover_fname = NULL;

	f_extend = FALSE;
	extend_from = 0;
	extend_to = 0;
	extend_r = 0;
	extend_m = 1;

	f_lex = FALSE;

	f_w = FALSE;
	f_W = FALSE;
	f_write_data_files = FALSE;
	f_t = FALSE;
	f_T = FALSE;



	f_draw_options = TRUE;
	draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);


#if 0
	f_write_tree = FALSE;

	f_find_node_by_stabilizer_order = FALSE;
	find_node_by_stabilizer_order = 0;


	f_draw_poset = FALSE;
	f_draw_full_poset = FALSE;
	f_plesken = FALSE;
	f_print_data_structure = FALSE;
	f_list = FALSE;
	f_list_all = FALSE;
	f_table_of_nodes = FALSE;
	f_make_relations_with_flag_orbits = FALSE;

	f_level_summary_csv = FALSE;
	f_orbit_reps_csv = FALSE;

	f_report = FALSE;
	report_options = NULL;

	f_node_label_is_group_order = FALSE;
	f_node_label_is_element = FALSE;

	f_show_orbit_decomposition = FALSE;
	f_show_stab = FALSE;
	f_save_stab = FALSE;
	f_show_whole_orbits = FALSE;

	f_export_schreier_trees = FALSE;
	f_draw_schreier_trees = FALSE;
	//schreier_tree_prefix[0] = 0;


	f_test_multi_edge_in_decomposition_matrix = FALSE;
#endif



	f_preferred_choice = FALSE;
	//std::vector<std::vector<int> > preferred_choice;

	f_clique_test = FALSE;
	//std::string clique_test_graph;
	clique_test_CG = NULL;


	f_has_invariant_subset_for_root_node = FALSE;
	invariant_subset_for_root_node = NULL;
	invariant_subset_for_root_node_size = 0;





	f_do_group_extension_in_upstep = TRUE;

	f_allowed_to_show_group_elements = FALSE;
	downstep_orbits_print_max_orbits = 25;
	downstep_orbits_print_max_points_per_orbit = 50;

}

poset_classification_control::~poset_classification_control()
{

}



int poset_classification_control::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "poset_classification_control::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = TRUE;
			problem_label.assign(argv[++i]);
			if (f_v) {
				cout << "-problem_label " << problem_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-path") == 0) {
			f_path = TRUE;
			path.assign(argv[++i]);
			if (f_v) {
				cout << "-path " << path << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-depth " << depth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-v") == 0) {
			i++;
			poset_classification_control::verbose_level = ST.strtoi(argv[i]);
			if (f_v) {
				cout << "-v " << poset_classification_control::verbose_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-gv") == 0) {
			i++;
			verbose_level_group_theory = ST.strtoi(argv[i]);
			if (f_v) {
				cout << "-gv " << verbose_level_group_theory << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recover") == 0) {
			f_recover = TRUE;
			recover_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-recover " << recover_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extend") == 0) {
			f_extend = TRUE;
			extend_from = ST.strtoi(argv[++i]);
			extend_to = ST.strtoi(argv[++i]);
			extend_r = ST.strtoi(argv[++i]);
			extend_m = ST.strtoi(argv[++i]);
			extend_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-extend from level " << extend_from
					<< " to level " << extend_to
					<< " cases congruent " << extend_r
					<< " mod " << extend_m
					<< " from file " << extend_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lex") == 0) {
			f_lex = TRUE;
			if (f_v) {
				cout << "-lex" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-w") == 0) {
			f_w = TRUE;
			if (f_v) {
				cout << "-w" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-W") == 0) {
			f_W = TRUE;
			if (f_v) {
				cout << "-W" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-write_data_files") == 0) {
			f_write_data_files = TRUE;
			if (f_v) {
				cout << "-write_data_files" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-t") == 0) {
			f_t = TRUE;
			if (f_v) {
				cout << "-t" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-T") == 0) {
			f_T = TRUE;
			if (f_v) {
				cout << "-T" << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = TRUE;

			draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);
			if (f_v) {
				cout << "-draw_options " << endl;
			}
			i += draw_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -draw_options " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			//cout << "-f_draw_options " << endl;
		}

#if 0
		else if (ST.stringcmp(argv[i], "-write_tree") == 0) {
			f_write_tree = TRUE;
			if (f_v) {
				cout << "-write_tree" << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-find_node_by_stabilizer_order") == 0) {
			f_find_node_by_stabilizer_order = TRUE;
			find_node_by_stabilizer_order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_node_by_stabilizer_order " << find_node_by_stabilizer_order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			if (f_v) {
				cout << "-draw_poset " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = TRUE;
			if (f_v) {
				cout << "-draw_full_poset " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-plesken") == 0) {
			f_plesken = TRUE;
			if (f_v) {
				cout << "-plesken " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_data_structure") == 0) {
			f_print_data_structure = TRUE;
			if (f_v) {
				cout << "-print_data_structure " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			if (f_v) {
				cout << "-list" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_all") == 0) {
			f_list_all = TRUE;
			if (f_v) {
				cout << "-list_all" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-table_of_nodes") == 0) {
			f_table_of_nodes = TRUE;
			if (f_v) {
				cout << "-table_of_nodes" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-make_relations_with_flag_orbits") == 0) {
			f_make_relations_with_flag_orbits = TRUE;
			if (f_v) {
				cout << "-make_relation_with_flag_orbits" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-level_summary_csv") == 0) {
			f_level_summary_csv = TRUE;
			if (f_v) {
				cout << "-level_summary_csv" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbit_reps_csv") == 0) {
			f_orbit_reps_csv = TRUE;
			if (f_v) {
				cout << "-orbit_reps_csv" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;

			report_options = NEW_OBJECT(poset_classification_report_options);
			if (f_v) {
				cout << "-report " << endl;
			}
			i += report_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -report " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}

			if (f_v) {
				cout << "-report" << endl;
				report_options->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-node_label_is_group_order") == 0) {
			f_node_label_is_group_order = TRUE;
			if (f_v) {
				cout << "-node_label_is_group_order" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-node_label_is_element") == 0) {
			f_node_label_is_element = TRUE;
			if (f_v) {
				cout << "-node_label_is_element" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_orbit_decomposition") == 0) {
			f_show_orbit_decomposition = TRUE;
			if (f_v) {
				cout << "-show_orbit_decomposition" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_stab") == 0) {
			f_show_stab = TRUE;
			if (f_v) {
				cout << "-show_stab" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_stab") == 0) {
			f_save_stab = TRUE;
			if (f_v) {
				cout << "-save_stab" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_whole_orbits") == 0) {
			f_show_whole_orbits = TRUE;
			if (f_v) {
				cout << "-show_whole_orbit" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_schreier_trees") == 0) {
			f_export_schreier_trees = TRUE;
			if (f_v) {
				cout << "-export_schreier_trees" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_schreier_trees") == 0) {
			f_draw_schreier_trees = TRUE;
			schreier_tree_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-draw_schreier_trees " << schreier_tree_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test_multi_edge_in_decomposition_matrix") == 0) {
			f_test_multi_edge_in_decomposition_matrix = TRUE;
			if (f_v) {
				cout << "-test_multi_edge_in_decomposition_matrix " << endl;
			}
		}
#endif


		else if (ST.stringcmp(argv[i], "-preferred_choice") == 0) {

			f_preferred_choice = TRUE;

			int node, pt, pt_pref;
			vector<int> v;

			node = ST.strtoi(argv[++i]);
			pt = ST.strtoi(argv[++i]);
			pt_pref = ST.strtoi(argv[++i]);
			v.push_back(node);
			v.push_back(pt);
			v.push_back(pt_pref);
			preferred_choice.push_back(v);

			if (f_v) {
				cout << "-preferred_choice " << node << " " << pt << " " << pt_pref << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-clique_test") == 0) {
			f_clique_test = TRUE;
			clique_test_graph.assign(argv[++i]);
			if (f_v) {
				cout << "-clique_test " << clique_test_graph << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "poset_classification_control::read_arguments "
					"unrecognized option '" << argv[i] << "'" << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "poset_classification_control::read_arguments done" << endl;
	}
	return i + 1;
}

void poset_classification_control::print()
{
	//cout << "poset_classification_control::print:" << endl;



	if (f_problem_label) {
		cout << "-problem_label " << problem_label << endl;
	}
	if (f_depth) {
		cout << "-depth " << depth << endl;
	}
	if (f_path) {
		cout << "-path" << path << endl;
	}
	cout << "v=" << verbose_level << endl;
	cout << "gv=" << verbose_level_group_theory << endl;

	if (f_recover) {
		cout << "-recover " << recover_fname << endl;
	}
	if (f_extend) {
		cout << "-extend from=" << extend_from << " to=" << extend_to
			<< " r=" << extend_r << " m=" << extend_m << " fname=" << extend_fname << endl;
	}

	if (f_lex) {
		cout << "-lex" << endl;
	}
	if (f_w) {
		cout << "-w" << endl;
	}
	if (f_W) {
		cout << "-W" << endl;
	}
	if (f_write_data_files) {
		cout << "-write_data_files" << endl;
	}
	if (f_T) {
		cout << "-T" << endl;
	}
	if (f_t) {
		cout << "-t" << endl;
	}

	if (f_draw_options) {
		cout << "-draw_options" << endl;
		draw_options->print();
	}
#if 0
	if (f_write_tree) {
		cout << "-write_tree" << endl;
	}
	if (f_find_node_by_stabilizer_order) {
		cout << "-find_node_by_stabilizer_order " << find_node_by_stabilizer_order << endl;
	}
	if (f_draw_poset) {
		cout << "-draw_poset" << endl;
	}
	if (f_draw_full_poset) {
		cout << "-draw_full_poset" << endl;
	}
	if (f_plesken) {
		cout << "-plesken" << endl;
	}
	if (f_print_data_structure) {
		cout << "-print_data_structure" << endl;
	}
	if (f_list) {
		cout << "-list" << endl;
	}
	if (f_list_all) {
		cout << "-list_all" << endl;
	}
	if (f_table_of_nodes) {
		cout << "-table_of_nodes" << endl;
	}
	if (f_make_relations_with_flag_orbits) {
		cout << "-make_relations_with_flag_orbits" << endl;
	}
	if (f_level_summary_csv) {
		cout << "-level_summary_csv" << endl;
	}
	if (f_orbit_reps_csv) {
		cout << "-orbit_reps_csv" << endl;
	}
	if (f_report) {
		cout << "-report ";
		report_options->print();
	}
	if (f_node_label_is_group_order) {
		cout << "-node_label_is_group_order" << endl;
	}
	if (f_node_label_is_element) {
		cout << "-node_label_is_element" << endl;
	}
	if (f_show_orbit_decomposition) {
		cout << "-show_orbit_decomposition" << endl;
	}
	if (f_show_stab) {
		cout << "-show_stab" << endl;
	}
	if (f_save_stab) {
		cout << "-save_stab" << endl;
	}
	if (f_show_whole_orbits) {
		cout << "-show_whole_orbits" << endl;
	}

	if (f_export_schreier_trees) {
		cout << "-export_schreier_trees" << endl;
	}
	if (f_node_label_is_group_order) {
		cout << "-node_label_is_group_order" << endl;
	}
	if (f_test_multi_edge_in_decomposition_matrix) {
		cout << "-test_multi_edge_in_decomposition_matrix" << endl;
	}
#endif


	if (f_preferred_choice) {
		for (int i = 0; i < preferred_choice.size(); i++) {
			cout << "-preferred_choice "
				<< preferred_choice[i][0]
				<< " " << preferred_choice[i][1]
				<< " " << preferred_choice[i][2] << endl;
		}
	}
	if (f_clique_test) {
		cout << "-clique_test " << clique_test_graph << endl;
	}
}


void poset_classification_control::prepare(poset_classification *PC, int verbose_level)
{
	int f_v = TRUE; //(verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_control::prepare" << endl;
	}
	if (f_clique_test) {
		if (f_v) {
			cout << "poset_classification_control::prepare -clique_test " << clique_test_graph << endl;
		}

		int idx;

		idx = orbiter_kernel_system::Orbiter->find_symbol(clique_test_graph);

		if (idx == -1) {
			cout << "poset_classification_control::prepare -clique_test cannot find symbol " << clique_test_graph << endl;
			exit(1);
		}

		clique_test_CG = (graph_theory::colored_graph *) orbiter_kernel_system::Orbiter->get_object(idx);
		if (f_v) {
			cout << "poset_classification_control::prepare -clique_test "
					"found a graph with " << clique_test_CG->nb_points << " vertices" << endl;
			cout << "poset_classification_control::prepare -clique_test "
					"PC->get_A2()->degree = " << PC->get_A2()->degree << endl;
		}

		if (PC->get_A2()->degree != clique_test_CG->nb_points) {
			cout << "poset_classification_control::prepare -clique_test "
					"found a graph with " << clique_test_CG->nb_points << " vertices" << endl;
			cout << "poset_classification_control::prepare -clique_test "
					"PC->get_A2()->degree = " << PC->get_A2()->degree << endl;
			cout << "poset_classification_control::prepare -clique_test degree of group does not match size of graph" << endl;
			exit(1);
		}

		PC->get_poset()->add_testing_without_group(
				poset_classification_control_early_test_function_cliques,
					this /* void *data */,
					verbose_level);

	}

	if (f_v) {
		cout << "poset_classification_control::prepare done" << endl;
	}
}


void poset_classification_control::early_test_func_for_clique_search(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_control::early_test_func_for_clique_search" << endl;
	}

	if (!f_clique_test) {
		cout << "poset_classification_control::early_test_func_for_clique_search !f_clique_test" << endl;
		exit(1);
	}
	if (clique_test_CG == NULL) {
		cout << "poset_classification_control::early_test_func_for_clique_search clique_test_CG == NULL" << endl;
		exit(1);
	}

	clique_test_CG->early_test_func_for_clique_search(
			S, len,
			candidates, nb_candidates,
			good_candidates, nb_good_candidates,
			verbose_level);

	if (f_v) {
		cout << "poset_classification_control::early_test_func_for_clique_search done" << endl;
	}
}



void poset_classification_control::init_root_node_invariant_subset(
	int *invariant_subset, int invariant_subset_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_control::init_root_node_invariant_subset" << endl;
	}
	f_has_invariant_subset_for_root_node = TRUE;
	invariant_subset_for_root_node = invariant_subset;
	invariant_subset_for_root_node_size = invariant_subset_size;
	if (f_v) {
		cout << "poset_classification_control::init_root_node_invariant_subset "
				"installed invariant subset of size "
				<< invariant_subset_size << endl;
	}
}



void poset_classification_control_preferred_choice_function(int pt, int &pt_pref,
		groups::schreier *Sch, void *data, int data2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_classification *PC = (poset_classification *) data;

	if (f_v) {
		cout << "poset_classification_control_preferred_choice_function data2=" << data2 << endl;
	}
	int i, l;
	int node1, pt1, pt_prev1;

	pt_pref = pt;

	l = PC->get_control()->preferred_choice.size();
	for (i = 0; i < l; i++) {
		node1 = PC->get_control()->preferred_choice[i][0];
		pt1 = PC->get_control()->preferred_choice[i][1];
		pt_prev1 = PC->get_control()->preferred_choice[i][2];
		if (node1 == data2 && pt == pt1) {
			if (f_v) {
				cout << "poset_classification_control_preferred_choice_function "
						"node=" << data2 << " pt=" << pt << " pt_pref=" << pt_pref << endl;
			}
			pt_pref = pt_prev1;
			break;
		}
	}

}


static void poset_classification_control_early_test_function_cliques(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	poset_classification_control *Control = (poset_classification_control *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_control_early_test_function_cliques for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}

	Control->early_test_func_for_clique_search(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);


	if (f_v) {
		cout << "poset_classification_control_early_test_function_cliques done" << endl;
	}
}



}}}

