/*
 * interface_combinatorics.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {



interface_combinatorics::interface_combinatorics()
{
	f_diophant = FALSE;
	Diophant_description = NULL;

	f_diophant_activity = FALSE;
	Diophant_activity_description = NULL;

	f_bent = FALSE;
	bent_n = 0;

	f_random_permutation = FALSE;
	random_permutation_degree = 0;
	//random_permutation_fname_csv = NULL;

	f_read_poset_file = FALSE;
	//read_poset_file_fname;

	f_grouping = FALSE;
	grouping_x_stretch = 0.7;

	f_list_parameters_of_SRG = FALSE;
	list_parameters_of_SRG_v_max = 0;

	f_conjugacy_classes_Sym_n = FALSE;
	conjugacy_classes_Sym_n_n = 0;

	f_tree_of_all_k_subsets = FALSE;
	tree_of_all_k_subsets_n = 0;
	tree_of_all_k_subsets_k = 0;

	f_Delandtsheer_Doyen = FALSE;
	Delandtsheer_Doyen_description = NULL;


	f_tdo_refinement = FALSE;
	Tdo_refinement_descr = NULL;

	f_tdo_print = FALSE;
	//tdo_print_fname;

	f_convert_stack_to_tdo = FALSE;
	//stack_fname;

	f_maximal_arc_parameters = FALSE;
	maximal_arc_parameters_q = 0;
	maximal_arc_parameters_r = 0;

	f_arc_parameters = FALSE;
	arc_parameters_q = arc_parameters_s = arc_parameters_r = 0;


	f_pentomino_puzzle = FALSE;

	f_regular_linear_space_classify = FALSE;
	Rls_descr = NULL;

	f_draw_layered_graph = FALSE;
	//draw_layered_graph_fname;
	Layered_graph_draw_options = NULL;

	f_read_solutions_and_tally = FALSE;
	//read_solutions_and_tally_fname
	read_solutions_and_tally_sz = 0;


	f_make_elementary_symmetric_functions = FALSE;
	make_elementary_symmetric_functions_n = 0;
	make_elementary_symmetric_functions_k_max = 0;

	f_Dedekind_numbers = FALSE;
	Dedekind_n_min = 0;
	Dedekind_n_max = 0;
	Dedekind_q_min = 0;
	Dedekind_q_max = 0;

	f_rank_k_subset = FALSE;
	rank_k_subset_n = 0;
	rank_k_subset_k = 0;
	//rank_k_subset_text;

	f_geometry_builder = FALSE;
	Geometry_builder_description = NULL;

}


void interface_combinatorics::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-diophant") == 0) {
		cout << "-diophant <description> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-diophant_activity") == 0) {
		cout << "-diophant_activity <description> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-bent") == 0) {
		cout << "-bent <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-random_permutation") == 0) {
		cout << "-random_permutation <ind : degree> <string : <fname_csv>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file") == 0) {
		cout << "-read_poset_file <string : file_name>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
		cout << "-read_poset_file_with_grouping <string : file_name> <double : x_stretch>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		cout << "-list_parameters_of_SRG <int : v_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		cout << "-conjugacy_classes_Sym_n <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		cout << "-tree_of_all_k_subsets <int : n> <int : k>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
			cout << "-Delandtsheer_Doyen <description>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-tdo_refinement") == 0) {
		cout << "-tdo_refinement <options>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-tdo_print") == 0) {
		cout << "-tdo_print <string : tdo-fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-convert_stack_to_tdo") == 0) {
		cout << "-convert_stack_to_tdo <string : stack_fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-maximal_arc_parameters") == 0) {
		cout << "-maximal_arc_parameters <int : q > < int : r >" << endl;
	}
	else if (ST.stringcmp(argv[i], "-arc_parameters") == 0) {
		cout << "-arc_parameters <int : q > <int : s > < int : r >" << endl;
	}
	else if (ST.stringcmp(argv[i], "-pentomino_puzzle") == 0) {
		cout << "-pentomino_puzzle" << endl;
	}
	else if (ST.stringcmp(argv[i], "-regular_linear_space_classify") == 0) {
		cout << "-regular_linear_space_classify <description>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_layered_graph") == 0) {
		cout << "-draw_layered_graph <string : fname> <layered_graph_options>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-read_solutions_and_tally") == 0) {
		cout << "-read_solutions_and_tally <string : fname> <int :read_solutions_and_tally_sz>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_elementary_symmetric_functions") == 0) {
		cout << "-make_elementary_symmetric_functions <int : n> <int :k_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-Dedekind_numbers") == 0) {
		cout << "-Dedekind_numbers <int : n_min> <int : n_max> <int : q_min> <int : q_max>  " << endl;
	}
	else if (ST.stringcmp(argv[i], "-rank_k_subset") == 0) {
		cout << "-rank_k_subset <int : n> <int : k> <string : text>  " << endl;
	}
	else if (ST.stringcmp(argv[i], "-geometry_builder") == 0) {
		cout << "-geometry_builder <description> -end" << endl;
	}
}

int interface_combinatorics::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;
	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-diophant") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-diophant_activity") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-bent") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-random_permutation") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-tdo_refinement") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-tdo_print") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-convert_stack_to_tdo") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-maximal_arc_parameters") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-arc_parameters") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-pentomino_puzzle") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-regular_linear_space_classify") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_layered_graph") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-read_solutions_and_tally") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-make_elementary_symmetric_functions") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-Dedekind_numbers") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-rank_k_subset") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-geometry_builder") == 0) {
		return true;
	}
	return false;
}

void interface_combinatorics::read_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_combinatorics::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_combinatorics::read_arguments the next argument is " << argv[i] << endl;
	}

	if (ST.stringcmp(argv[i], "-diophant") == 0) {
		f_diophant = TRUE;
		if (f_v) {
			cout << "-diophant " << endl;
		}
		Diophant_description = NEW_OBJECT(solvers::diophant_description);
		i += Diophant_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		if (f_v) {
			cout << "interface_combinatorics::read_arguments finished "
					"reading -diophant" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-diophant_activity") == 0) {
		f_diophant_activity = TRUE;
		if (f_v) {
			cout << "-diophant_activity " << endl;
		}
		Diophant_activity_description = NEW_OBJECT(solvers::diophant_activity_description);
		i += Diophant_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		if (f_v) {
			cout << "interface_combinatorics::read_arguments finished "
					"reading -diophant_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-bent") == 0) {
		f_bent = TRUE;
		bent_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-bent " << bent_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-random_permutation") == 0) {
		f_random_permutation = TRUE;
		random_permutation_degree = ST.strtoi(argv[++i]);
		random_permutation_fname_csv.assign(argv[++i]);
		if (f_v) {
			cout << "-random_permutation " << random_permutation_degree << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file") == 0) {
		f_read_poset_file = TRUE;
		f_grouping = FALSE;
		read_poset_file_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-read_poset_file " << read_poset_file_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
		f_read_poset_file = TRUE;
		f_grouping = TRUE;
		read_poset_file_fname.assign(argv[++i]);
		grouping_x_stretch = ST.strtof(argv[++i]);
		if (f_v) {
			cout << "-read_poset_file_with_grouping "
					<< read_poset_file_fname << " " << grouping_x_stretch << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		f_list_parameters_of_SRG = TRUE;
		list_parameters_of_SRG_v_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-list_parameters_of_SRG " << list_parameters_of_SRG_v_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		f_conjugacy_classes_Sym_n = TRUE;
		conjugacy_classes_Sym_n_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-conjugacy_classes_Sym_n " << conjugacy_classes_Sym_n_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		f_tree_of_all_k_subsets = TRUE;
		tree_of_all_k_subsets_n = ST.strtoi(argv[++i]);
		tree_of_all_k_subsets_k = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-tree_of_all_k_subsets " << tree_of_all_k_subsets_n << " " << tree_of_all_k_subsets_k << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
		f_Delandtsheer_Doyen = TRUE;
		Delandtsheer_Doyen_description = NEW_OBJECT(apps_combinatorics::delandtsheer_doyen_description);
		i += Delandtsheer_Doyen_description->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

		if (f_v) {
			cout << "-Delandtsheer_Doyen" << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-tdo_refinement") == 0) {
		f_tdo_refinement = TRUE;
		if (f_v) {
			cout << "-tdo_refinement " << endl;
		}
		Tdo_refinement_descr = NEW_OBJECT(combinatorics::tdo_refinement_description);
		i += Tdo_refinement_descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::read_arguments finished "
					"reading -tdo_refinement" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-tdo_print") == 0) {
		f_tdo_print = TRUE;
		tdo_print_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-tdo_print " << tdo_print_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-convert_stack_to_tdo") == 0) {
		f_convert_stack_to_tdo = TRUE;
		stack_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-convert_stack_to_tdo " << stack_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-maximal_arc_parameters") == 0) {
		f_maximal_arc_parameters = TRUE;
		maximal_arc_parameters_q = ST.strtoi(argv[++i]);
		maximal_arc_parameters_r = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-maximal_arc_parameters " << maximal_arc_parameters_q
				<< " " << maximal_arc_parameters_r << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-arc_parameters") == 0) {
		f_arc_parameters = TRUE;
		arc_parameters_q = ST.strtoi(argv[++i]);
		arc_parameters_s = ST.strtoi(argv[++i]);
		arc_parameters_r = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-arc_parameters " << arc_parameters_q
				<< " " << arc_parameters_s
				<< " " << arc_parameters_r
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-pentomino_puzzle") == 0) {
		f_pentomino_puzzle = TRUE;
		if (f_v) {
			cout << "-pentomino_puzzle " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-regular_linear_space_classify") == 0) {
		f_regular_linear_space_classify = TRUE;

		if (f_v) {
			cout << "-regular_linear_space_classify " << endl;
		}

		Rls_descr = NEW_OBJECT(apps_combinatorics::regular_linear_space_description);
		i += Rls_descr->read_arguments(argc - i - 1,
			argv + i + 1, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::read_arguments finished "
					"reading -regular_linear_space_classify" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}

			cout << "-regular_linear_space_classify " <<endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-draw_layered_graph") == 0) {
		f_draw_layered_graph = TRUE;
		draw_layered_graph_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-draw_layered_graph " << endl;
		}
		Layered_graph_draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);
		i += Layered_graph_draw_options->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::read_arguments "
					"finished reading -draw_layered_graph" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-read_solutions_and_tally") == 0) {
		f_read_solutions_and_tally = TRUE;
		read_solutions_and_tally_fname.assign(argv[++i]);
		read_solutions_and_tally_sz = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-read_solutions_and_tally " << read_solutions_and_tally_fname
				<< " " << read_solutions_and_tally_sz << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-make_elementary_symmetric_functions") == 0) {
		f_make_elementary_symmetric_functions = TRUE;
		make_elementary_symmetric_functions_n = ST.strtoi(argv[++i]);
		make_elementary_symmetric_functions_k_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-make_elementary_symmetric_functions " << make_elementary_symmetric_functions_n
				<< " " << make_elementary_symmetric_functions_k_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-Dedekind_numbers") == 0) {
		f_Dedekind_numbers = TRUE;
		Dedekind_n_min = ST.strtoi(argv[++i]);
		Dedekind_n_max = ST.strtoi(argv[++i]);
		Dedekind_q_min = ST.strtoi(argv[++i]);
		Dedekind_q_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-Dedekind_numbers " << Dedekind_n_min
				<< " " << Dedekind_n_max
				<< " " << Dedekind_q_min
				<< " " << Dedekind_q_max
				<< " " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-rank_k_subset") == 0) {
		f_rank_k_subset = TRUE;
		rank_k_subset_n = ST.strtoi(argv[++i]);
		rank_k_subset_k = ST.strtoi(argv[++i]);
		rank_k_subset_text.assign(argv[++i]);
		if (f_v) {
			cout << "-rank_k_subset " << rank_k_subset_n
				<< " " << rank_k_subset_k
				<< " " << rank_k_subset_text
				<< " " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-geometry_builder") == 0) {
		f_geometry_builder = TRUE;
		if (f_v) {
			cout << "-geometry_builder " << endl;
		}
		Geometry_builder_description = NEW_OBJECT(geometry_builder::geometry_builder_description);
		i += Geometry_builder_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		if (f_v) {
			cout << "interface_combinatorics::read_arguments finished "
					"reading -geometry_builder" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}

	if (f_v) {
		cout << "interface_combinatorics::read_arguments done" << endl;
	}
}


void interface_combinatorics::print()
{
	if (f_diophant) {
		cout << "-diophant " << endl;
		Diophant_description->print();
	}
	if (f_diophant_activity) {
		cout << "-diophant_activity " << endl;
		Diophant_activity_description->print();
	}
	if (f_bent) {
		cout << "-bent " << bent_n << endl;
	}
	if (f_random_permutation) {
		cout << "-random_permutation " << random_permutation_degree << endl;
	}
	if (f_read_poset_file) {
		cout << "-read_poset_file " << read_poset_file_fname << endl;
	}
	if (f_read_poset_file) {
		cout << "-read_poset_file_with_grouping "
				<< read_poset_file_fname << " " << grouping_x_stretch << endl;
	}
	if (f_list_parameters_of_SRG) {
		cout << "-list_parameters_of_SRG " << list_parameters_of_SRG_v_max << endl;
	}
	if (f_conjugacy_classes_Sym_n) {
		cout << "-conjugacy_classes_Sym_n " << conjugacy_classes_Sym_n_n << endl;
	}
	if (f_tree_of_all_k_subsets) {
		cout << "-tree_of_all_k_subsets " << tree_of_all_k_subsets_n << " " << tree_of_all_k_subsets_k << endl;
	}
	if (f_Delandtsheer_Doyen) {
		cout << "-Delandtsheer_Doyen" << endl;
		Delandtsheer_Doyen_description->print();
	}
	if (f_tdo_refinement) {
		cout << "-tdo_refinement " << endl;
		Tdo_refinement_descr->print();
	}
	if (f_tdo_print) {
		cout << "-tdo_print " << tdo_print_fname << endl;
	}
	if (f_convert_stack_to_tdo) {
		cout << "-convert_stack_to_tdo " << stack_fname << endl;
	}
	if (f_maximal_arc_parameters) {
		cout << "-maximal_arc_parameters " << maximal_arc_parameters_q
				<< " " << maximal_arc_parameters_r << endl;
	}
	if (f_arc_parameters) {
		cout << "-arc_parameters " << arc_parameters_q
				<< " " << arc_parameters_s
				<< " " << arc_parameters_r
				<< endl;
	}
	if (f_pentomino_puzzle) {
		cout << "-pentomino_puzzle " <<endl;
	}
	if (f_regular_linear_space_classify) {
		cout << "-regular_linear_space_classify " << endl;
		//Rls_descr->print();
	}
	if (f_draw_layered_graph) {
		cout << "-draw_layered_graph " << endl;
	}
	if (f_read_solutions_and_tally) {
		cout << "-read_solutions_and_tally " << read_solutions_and_tally_fname
				<< " " << read_solutions_and_tally_sz << endl;
	}

	if (f_make_elementary_symmetric_functions) {
		cout << "-make_elementary_symmetric_functions "
				<< make_elementary_symmetric_functions_n
				<< " " << make_elementary_symmetric_functions_k_max
				<< endl;
	}
	if (f_Dedekind_numbers) {
		cout << "-Dedekind_numbers " << Dedekind_n_min
				<< " " << Dedekind_n_max
				<< " " << Dedekind_q_min
				<< " " << Dedekind_q_max
				<< " " << endl;
	}
	if (f_rank_k_subset) {
		cout << "-rank_k_subset " << rank_k_subset_n
			<< " " << rank_k_subset_k
			<< " " << rank_k_subset_text
			<< " " << endl;
	}
	if (f_geometry_builder) {
		Geometry_builder_description->print();
	}
}


void interface_combinatorics::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::worker" << endl;
	}
	if (f_diophant) {
		do_diophant(Diophant_description, verbose_level);
	}
	else if (f_diophant_activity) {
		do_diophant_activity(Diophant_activity_description, verbose_level);
	}
	else if (f_bent) {
		do_bent(bent_n, verbose_level);
	}
	else if (f_random_permutation) {

		combinatorics::combinatorics_domain Combi;

		Combi.create_random_permutation(random_permutation_degree,
				random_permutation_fname_csv, verbose_level);
	}
	else if (f_read_poset_file) {

		combinatorics::combinatorics_domain Combi;

		Combi.do_read_poset_file(read_poset_file_fname,
				f_grouping, grouping_x_stretch,
				verbose_level);
	}
	else if (f_list_parameters_of_SRG) {

		graph_theory::graph_theory_domain G;

		G.list_parameters_of_SRG(list_parameters_of_SRG_v_max, verbose_level);
	}
	else if (f_conjugacy_classes_Sym_n) {

		do_conjugacy_classes_Sym_n(conjugacy_classes_Sym_n_n, verbose_level);
	}
	else if (f_tree_of_all_k_subsets) {

		combinatorics::combinatorics_domain Combi;

		Combi.do_make_tree_of_all_k_subsets(tree_of_all_k_subsets_n, tree_of_all_k_subsets_k, verbose_level);
	}
	else if (f_Delandtsheer_Doyen) {

		do_Delandtsheer_Doyen(Delandtsheer_Doyen_description, verbose_level);
	}
	else if (f_tdo_refinement) {

		combinatorics::combinatorics_domain Combi;

		Combi.do_tdo_refinement(Tdo_refinement_descr, verbose_level);
	}
	else if (f_tdo_print) {

		combinatorics::combinatorics_domain Combi;

		Combi.do_tdo_print(tdo_print_fname, verbose_level);
	}
	else if (f_convert_stack_to_tdo) {

		combinatorics::combinatorics_domain Combi;

		Combi.convert_stack_to_tdo(stack_fname, verbose_level);
	}
	else if (f_maximal_arc_parameters) {

		combinatorics::combinatorics_domain Combi;

		Combi.do_parameters_maximal_arc(maximal_arc_parameters_q,
				maximal_arc_parameters_r, verbose_level);
	}
	else if (f_arc_parameters) {

		combinatorics::combinatorics_domain Combi;

		Combi.do_parameters_arc(arc_parameters_q,
				arc_parameters_s, arc_parameters_r, verbose_level);
	}
	else if (f_pentomino_puzzle) {
		cout << "pentomino_puzzle " <<endl;

		combinatorics::pentomino_puzzle *P;

		P = NEW_OBJECT(combinatorics::pentomino_puzzle);

		P->main(verbose_level);

		FREE_OBJECT(P);

	}
	else if (f_regular_linear_space_classify) {

		apps_combinatorics::regular_ls_classify *Rls;

		Rls = NEW_OBJECT(apps_combinatorics::regular_ls_classify);

		if (f_v) {
			cout << "interface_combinatorics::worker before Rls->init_and_run" << endl;
		}
		Rls->init_and_run(Rls_descr, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::worker after Rls->init_and_run" << endl;
		}
		FREE_OBJECT(Rls);

	}
	else if (f_draw_layered_graph) {
		graphics::graphical_output GO;

		GO.draw_layered_graph_from_file(draw_layered_graph_fname,
				Layered_graph_draw_options,
				verbose_level);

	}
	else if (f_read_solutions_and_tally) {

		orbiter_kernel_system::file_io Fio;

		Fio.read_solutions_and_tally(read_solutions_and_tally_fname,
				read_solutions_and_tally_sz, verbose_level);

	}
	else if (f_make_elementary_symmetric_functions) {

		combinatorics::combinatorics_domain Combi;

		Combi.make_elementary_symmetric_functions(make_elementary_symmetric_functions_n,
				make_elementary_symmetric_functions_k_max, verbose_level);

	}
	else if (f_Dedekind_numbers) {

		combinatorics::combinatorics_domain Combi;

		Combi.Dedekind_numbers(
				Dedekind_n_min, Dedekind_n_max, Dedekind_q_min, Dedekind_q_max,
				verbose_level);

	}

	else if (f_rank_k_subset) {

		combinatorics::combinatorics_domain Combi;


		int *set;
		int sz;
		int i, j, r, N;
		int *Rk;

		Int_vec_scan(rank_k_subset_text, set, sz);

		N = (sz + rank_k_subset_k - 1) / rank_k_subset_k;
		Rk = NEW_int(N);
		i = 0;
		j = 0;
		while (i < sz) {


			r = Combi.rank_k_subset(set + i, rank_k_subset_n, rank_k_subset_k);

			cout << "The rank of ";
			Int_vec_print(cout, set + i, rank_k_subset_k);
			cout << " is " << r << endl;
			Rk[j] = r;

			i += rank_k_subset_k;
			j++;
		}

		cout << "the ranks of all subsets are: ";
		Int_vec_print(cout, Rk, N);
		cout << endl;

		data_structures::sorting Sorting;

		Sorting.int_vec_heapsort(Rk, N);

		cout << "the sorted ranks of all subsets are: ";
		Int_vec_print(cout, Rk, N);
		cout << endl;

	}
	else if (f_geometry_builder) {
		if (f_v) {
			cout << "interface_combinatorics::worker -geometry_builder" << endl;
		}

		geometry_builder::geometry_builder *GB;

		GB = NEW_OBJECT(geometry_builder::geometry_builder);

		GB->init_description(Geometry_builder_description, verbose_level);

		GB->gg->main2(verbose_level);

		FREE_OBJECT(GB);
	}


}


void interface_combinatorics::do_diophant(solvers::diophant_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_diophant" << endl;
	}

	solvers::diophant_create *DC;

	DC = NEW_OBJECT(solvers::diophant_create);

	DC->init(Descr, verbose_level);


	if (f_v) {
		cout << "interface_combinatorics::do_diophant done" << endl;
	}
}

void interface_combinatorics::do_diophant_activity(
		solvers::diophant_activity_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_diophant_activity" << endl;
	}

	solvers::diophant_activity *DA;

	DA = NEW_OBJECT(solvers::diophant_activity);

	DA->init_from_file(Descr, verbose_level);

	FREE_OBJECT(DA);

	if (f_v) {
		cout << "interface_combinatorics::do_diophant_activity done" << endl;
	}
}

void interface_combinatorics::do_bent(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_bent" << endl;
	}

	{
		combinatorics::boolean_function_domain *BF;

		BF = NEW_OBJECT(combinatorics::boolean_function_domain);

		if (f_v) {
			cout << "interface_combinatorics::do_bent before BF->init" << endl;
		}
		BF->init(n, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::do_bent after BF->init" << endl;
		}

		apps_combinatorics::boolean_function_classify *BFC;

		BFC = NEW_OBJECT(apps_combinatorics::boolean_function_classify);

		if (f_v) {
			cout << "interface_combinatorics::do_bent before BFC->init_group" << endl;
		}
		BFC->init_group(BF, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::do_bent after BFC->init_group" << endl;
		}

		if (f_v) {
			cout << "interface_combinatorics::do_bent before BFC->search_for_bent_functions" << endl;
		}
		BFC->search_for_bent_functions(verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::do_bent after BFC->search_for_bent_functions" << endl;
		}

		FREE_OBJECT(BFC);
		FREE_OBJECT(BF);
	}

	if (f_v) {
		cout << "interface_combinatorics::do_bent done" << endl;
	}
}


void interface_combinatorics::do_conjugacy_classes_Sym_n(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_conjugacy_classes_Sym_n" << endl;
	}

	int i;
	int cnt;
	ring_theory::longinteger_object class_size, S, F, A;
	ring_theory::longinteger_domain D;
	combinatorics::combinatorics_domain C;
	combinatorics::combinatorics_domain Combi;

	cnt = Combi.count_partitions(n);

	int *Parts;

	Parts = NEW_int(cnt * n);
	Combi.make_partitions(n, Parts, cnt);


	S.create(0, __FILE__, __LINE__);

	cout << "The conjugacy classes in Sym_" << n << " are:" << endl;
	for (i = 0; i < cnt; i++) {
		cout << i << " : ";
		Int_vec_print(cout, Parts + i * n, n);
		cout << " : ";

		C.size_of_conjugacy_class_in_sym_n(class_size, n, Parts + i * n);
		cout << class_size << " : ";
		cout << endl;

		D.add_in_place(S, class_size);
		}

	D.factorial(F, n);
	D.integral_division_exact(F, S, A);
	if (!A.is_one()) {
		cout << "the class sizes do not add up" << endl;
		exit(1);
		}
	cout << "The sum of the class sizes is n!" << endl;

	if (f_v) {
		cout << "interface_combinatorics::do_conjugacy_classes_Sym_n done" << endl;
	}
}


void interface_combinatorics::do_Delandtsheer_Doyen(apps_combinatorics::delandtsheer_doyen_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_Delandtsheer_Doyen" << endl;
	}

	apps_combinatorics::delandtsheer_doyen *DD;

	DD = NEW_OBJECT(apps_combinatorics::delandtsheer_doyen);

	DD->init(Descr, verbose_level);

	FREE_OBJECT(DD);


	if (f_v) {
		cout << "interface_combinatorics::do_Delandtsheer_Doyen done" << endl;
	}
}








}}
