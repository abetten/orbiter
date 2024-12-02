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



static void poset_classification_control_early_test_function_cliques(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);



poset_classification_control::poset_classification_control()
{
	Record_birth();

	f_problem_label = false;
	//problem_label = NULL;

	f_path = false;
	//path = NULL;

	f_depth = false;
	depth = 0;

	f_verbose_level = false;
	verbose_level = 0;

	f_verbose_level_group_theory = false;
	verbose_level_group_theory = 0;

	f_recover = false;
	//recover_fname = NULL;

	f_extend = false;
	extend_from = 0;
	extend_to = 0;
	extend_r = 0;
	extend_m = 1;

	f_lex = false;

	f_w = false;
	f_W = false;
	f_write_data_files = false;
	f_t = false;
	f_T = false;



	f_draw_options = true;
	std::string draw_options_label;
	//draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);




	f_preferred_choice = false;
	//std::vector<std::vector<int> > preferred_choice;

	f_clique_test = false;
	//std::string clique_test_graph;
	clique_test_CG = NULL;


	f_has_invariant_subset_for_root_node = false;
	invariant_subset_for_root_node = NULL;
	invariant_subset_for_root_node_size = 0;





	f_do_group_extension_in_upstep = true;

	f_allowed_to_show_group_elements = false;
	downstep_orbits_print_max_orbits = 25;
	downstep_orbits_print_max_points_per_orbit = 50;

}

poset_classification_control::~poset_classification_control()
{
	Record_death();

}



int poset_classification_control::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "poset_classification_control::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = true;
			problem_label.assign(argv[++i]);
			if (f_v) {
				cout << "-problem_label " << problem_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-path") == 0) {
			f_path = true;
			path.assign(argv[++i]);
			if (f_v) {
				cout << "-path " << path << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-depth") == 0) {
			f_depth = true;
			depth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-depth " << depth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-v") == 0) {
			i++;
			f_verbose_level = true;
			poset_classification_control::verbose_level = ST.strtoi(argv[i]);
			if (f_v) {
				cout << "-v " << poset_classification_control::verbose_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-gv") == 0) {
			i++;
			f_verbose_level_group_theory = true;
			verbose_level_group_theory = ST.strtoi(argv[i]);
			if (f_v) {
				cout << "-gv " << verbose_level_group_theory << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recover") == 0) {
			f_recover = true;
			recover_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-recover " << recover_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extend") == 0) {
			f_extend = true;
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
			f_lex = true;
			if (f_v) {
				cout << "-lex" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-w") == 0) {
			f_w = true;
			if (f_v) {
				cout << "-w" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-W") == 0) {
			f_W = true;
			if (f_v) {
				cout << "-W" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-write_data_files") == 0) {
			f_write_data_files = true;
			if (f_v) {
				cout << "-write_data_files" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-t") == 0) {
			f_t = true;
			if (f_v) {
				cout << "-t" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-T") == 0) {
			f_T = true;
			if (f_v) {
				cout << "-T" << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = true;
			draw_options_label.assign(argv[++i]);
			if (f_v) {
				cout << "-draw_options " << draw_options_label << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-preferred_choice") == 0) {

			f_preferred_choice = true;

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
			f_clique_test = true;
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
	if (f_verbose_level) {
		cout << "-v " << verbose_level << endl;
	}
	if (f_verbose_level_group_theory) {
		cout << "-vg " << verbose_level_group_theory << endl;
	}

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
		cout << "-draw_options " << draw_options_label << endl;
	}


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


void poset_classification_control::prepare(
		poset_classification *PC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_control::prepare" << endl;
	}
	if (f_clique_test) {
		if (f_v) {
			cout << "poset_classification_control::prepare -clique_test " << clique_test_graph << endl;
		}

		int idx;

		idx = other::orbiter_kernel_system::Orbiter->find_symbol(clique_test_graph);

		if (idx == -1) {
			cout << "poset_classification_control::prepare -clique_test cannot find symbol " << clique_test_graph << endl;
			exit(1);
		}

		clique_test_CG = (combinatorics::graph_theory::colored_graph *) other::orbiter_kernel_system::Orbiter->get_object(idx);
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

	clique_test_CG->Colored_graph_cliques->early_test_func_for_clique_search(
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
	f_has_invariant_subset_for_root_node = true;
	invariant_subset_for_root_node = invariant_subset;
	invariant_subset_for_root_node_size = invariant_subset_size;
	if (f_v) {
		cout << "poset_classification_control::init_root_node_invariant_subset "
				"installed invariant subset of size "
				<< invariant_subset_size << endl;
	}
}



void poset_classification_control_preferred_choice_function(
		int pt, int &pt_pref,
		groups::schreier *Sch,
		void *data, int data2,
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


static void poset_classification_control_early_test_function_cliques(
		long int *S, int len,
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

