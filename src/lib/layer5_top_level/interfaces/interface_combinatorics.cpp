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
namespace user_interface {



interface_combinatorics::interface_combinatorics()
{

	f_random_permutation = false;
	random_permutation_degree = 0;
	//random_permutation_fname_csv = NULL;

	f_create_random_k_subsets = false;
	create_random_k_subsets_n = 0;
	create_random_k_subsets_k = 0;
	create_random_k_subsets_nb = 0;

	f_read_poset_file = false;
	//read_poset_file_fname;

	f_grouping = false;
	grouping_x_stretch = 0.7;

	f_list_parameters_of_SRG = false;
	list_parameters_of_SRG_v_max = 0;

	f_conjugacy_classes_Sym_n = false;
	conjugacy_classes_Sym_n_n = 0;

	f_tree_of_all_k_subsets = false;
	tree_of_all_k_subsets_n = 0;
	tree_of_all_k_subsets_k = 0;

	f_Delandtsheer_Doyen = false;
	Delandtsheer_Doyen_description = NULL;


	f_tdo_refinement = false;
	Tdo_refinement_descr = NULL;

	f_tdo_print = false;
	//tdo_print_fname;

	f_convert_stack_to_tdo = false;
	//stack_fname;

	f_maximal_arc_parameters = false;
	maximal_arc_parameters_q = 0;
	maximal_arc_parameters_r = 0;

	f_arc_parameters = false;
	arc_parameters_q = arc_parameters_s = arc_parameters_r = 0;


	f_pentomino_puzzle = false;

	f_regular_linear_space_classify = false;
	Rls_descr = NULL;

	f_domino_portrait = false;
	domino_portrait_D = 0;
	domino_portrait_s = 0;
	//std::string domino_portrait_fname;
	domino_portrait_draw_options = NULL;

	f_read_solutions_and_tally = false;
	//read_solutions_and_tally_fname
	read_solutions_and_tally_sz = 0;


	f_make_elementary_symmetric_functions = false;
	make_elementary_symmetric_functions_n = 0;
	make_elementary_symmetric_functions_k_max = 0;

	f_Dedekind_numbers = false;
	Dedekind_n_min = 0;
	Dedekind_n_max = 0;
	Dedekind_q_min = 0;
	Dedekind_q_max = 0;

	f_rank_k_subset = false;
	rank_k_subset_n = 0;
	rank_k_subset_k = 0;
	//rank_k_subset_text;

	f_geometry_builder = false;
	Geometry_builder_description = NULL;

	f_union = false;
	//std::string union_set_of_sets_fname;
	//std::string union_input_fname;
	//std::string union_output_fname;

	f_dot_product_of_columns = false;
	//std::string dot_product_of_columns_fname;

	f_dot_product_of_rows = false;
	//std::string dot_product_of_rows_fname;

	f_matrix_multiply_over_Z = false;
	//std::string matrix_multiply_over_Z_label1;
	//std::string matrix_multiply_over_Z_label2;

	f_rowspan_over_R = false;
	//std::string rowspan_over_R_label;

}


void interface_combinatorics::print_help(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;


	if (ST.stringcmp(argv[i], "-random_permutation") == 0) {
		cout << "-random_permutation <int : degree> <string : <fname_csv>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-create_random_k_subsets") == 0) {
		cout << "-create_random_k_subsets <int : n> <int : k> <int : nb>" << endl;
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
	else if (ST.stringcmp(argv[i], "-domino_portrait") == 0) {
		cout << "-domino_portrait <string : fname> <int : D> <int : s> <layered_graph_options>" << endl;
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
	else if (ST.stringcmp(argv[i], "-union") == 0) {
		cout << "-union <fname : set_of_sets> <fname : input> <fname : output> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-dot_product_of_columns") == 0) {
		cout << "-dot_product_of_columns <label : matrix> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-dot_product_of_rows") == 0) {
		cout << "-dot_product_of_rows <label : matrix> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-matrix_multiply_over_Z") == 0) {
		cout << "-matrix_multiply_over_Z <label : matrix1> <label : matrix2> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-rowspan_over_R") == 0) {
		cout << "-rowspan_over_R <label : matrix> " << endl;
	}
}

int interface_combinatorics::recognize_keyword(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;
	if (i >= argc) {
		return false;
	}

	if (ST.stringcmp(argv[i], "-random_permutation") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-create_random_k_subsets") == 0) {
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
	else if (ST.stringcmp(argv[i], "-domino_portrait") == 0) {
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
	else if (ST.stringcmp(argv[i], "-union") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-dot_product_of_columns") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-dot_product_of_rows") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-matrix_multiply_over_Z") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-rowspan_over_R") == 0) {
		return true;
	}
	return false;
}

void interface_combinatorics::read_arguments(
		int argc,
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

	if (ST.stringcmp(argv[i], "-random_permutation") == 0) {
		f_random_permutation = true;
		random_permutation_degree = ST.strtoi(argv[++i]);
		random_permutation_fname_csv.assign(argv[++i]);
		if (f_v) {
			cout << "-random_permutation " << random_permutation_degree << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-create_random_k_subsets") == 0) {
		f_create_random_k_subsets = true;
		create_random_k_subsets_n = ST.strtoi(argv[++i]);
		create_random_k_subsets_k = ST.strtoi(argv[++i]);
		create_random_k_subsets_nb = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-create_random_k_subsets " << create_random_k_subsets_n << " " << create_random_k_subsets_k << " " << create_random_k_subsets_nb << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file") == 0) {
		f_read_poset_file = true;
		f_grouping = false;
		read_poset_file_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-read_poset_file " << read_poset_file_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
		f_read_poset_file = true;
		f_grouping = true;
		read_poset_file_fname.assign(argv[++i]);
		grouping_x_stretch = ST.strtof(argv[++i]);
		if (f_v) {
			cout << "-read_poset_file_with_grouping "
					<< read_poset_file_fname << " " << grouping_x_stretch << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		f_list_parameters_of_SRG = true;
		list_parameters_of_SRG_v_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-list_parameters_of_SRG " << list_parameters_of_SRG_v_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		f_conjugacy_classes_Sym_n = true;
		conjugacy_classes_Sym_n_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-conjugacy_classes_Sym_n " << conjugacy_classes_Sym_n_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		f_tree_of_all_k_subsets = true;
		tree_of_all_k_subsets_n = ST.strtoi(argv[++i]);
		tree_of_all_k_subsets_k = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-tree_of_all_k_subsets " << tree_of_all_k_subsets_n << " " << tree_of_all_k_subsets_k << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
		f_Delandtsheer_Doyen = true;
		Delandtsheer_Doyen_description = NEW_OBJECT(apps_combinatorics::delandtsheer_doyen_description);
		i += Delandtsheer_Doyen_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

		if (f_v) {
			cout << "-Delandtsheer_Doyen" << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-tdo_refinement") == 0) {
		f_tdo_refinement = true;
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
		f_tdo_print = true;
		tdo_print_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-tdo_print " << tdo_print_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-convert_stack_to_tdo") == 0) {
		f_convert_stack_to_tdo = true;
		stack_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-convert_stack_to_tdo " << stack_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-maximal_arc_parameters") == 0) {
		f_maximal_arc_parameters = true;
		maximal_arc_parameters_q = ST.strtoi(argv[++i]);
		maximal_arc_parameters_r = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-maximal_arc_parameters " << maximal_arc_parameters_q
				<< " " << maximal_arc_parameters_r << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-arc_parameters") == 0) {
		f_arc_parameters = true;
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
		f_pentomino_puzzle = true;
		if (f_v) {
			cout << "-pentomino_puzzle " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-regular_linear_space_classify") == 0) {
		f_regular_linear_space_classify = true;

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
	else if (ST.stringcmp(argv[i], "-domino_portrait") == 0) {
		f_domino_portrait = true;
		if (f_v) {
			cout << "-domino_portrait " << endl;
		}
		domino_portrait_D = ST.strtoi(argv[++i]);
		domino_portrait_s = ST.strtoi(argv[++i]);
		domino_portrait_fname.assign(argv[++i]);
		domino_portrait_draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);
		i += domino_portrait_draw_options->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::read_arguments "
					"finished reading -domino_portrait" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-read_solutions_and_tally") == 0) {
		f_read_solutions_and_tally = true;
		read_solutions_and_tally_fname.assign(argv[++i]);
		read_solutions_and_tally_sz = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-read_solutions_and_tally " << read_solutions_and_tally_fname
				<< " " << read_solutions_and_tally_sz << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-make_elementary_symmetric_functions") == 0) {
		f_make_elementary_symmetric_functions = true;
		make_elementary_symmetric_functions_n = ST.strtoi(argv[++i]);
		make_elementary_symmetric_functions_k_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-make_elementary_symmetric_functions " << make_elementary_symmetric_functions_n
				<< " " << make_elementary_symmetric_functions_k_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-Dedekind_numbers") == 0) {
		f_Dedekind_numbers = true;
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
		f_rank_k_subset = true;
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
		f_geometry_builder = true;
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
	else if (ST.stringcmp(argv[i], "-union") == 0) {
		f_union = true;
		union_set_of_sets_fname.assign(argv[++i]);
		union_input_fname.assign(argv[++i]);
		union_output_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-union " << union_set_of_sets_fname
				<< " " << union_input_fname
				<< " " << union_output_fname
				<< " " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-dot_product_of_columns") == 0) {
		f_dot_product_of_columns = true;
		dot_product_of_columns_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-dot_product_of_columns "
					<< dot_product_of_columns_fname
				<< " " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-dot_product_of_rows") == 0) {
		f_dot_product_of_rows = true;
		dot_product_of_rows_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-dot_product_of_rows "
					<< dot_product_of_rows_fname
				<< " " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-matrix_multiply_over_Z") == 0) {
		f_matrix_multiply_over_Z = true;
		matrix_multiply_over_Z_label1.assign(argv[++i]);
		matrix_multiply_over_Z_label2.assign(argv[++i]);
		if (f_v) {
			cout << "-matrix_multiply_over_Z "
					<< " " << matrix_multiply_over_Z_label1
					<< " " << matrix_multiply_over_Z_label2
				<< " " << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-rowspan_over_R") == 0) {
		f_rowspan_over_R = true;
		rowspan_over_R_label.assign(argv[++i]);
		if (f_v) {
			cout << "-rowspan_over_R "
					<< " " << rowspan_over_R_label
				<< " " << endl;
		}
	}



	if (f_v) {
		cout << "interface_combinatorics::read_arguments done" << endl;
	}
}


void interface_combinatorics::print()
{

	if (f_random_permutation) {
		cout << "-random_permutation " << random_permutation_degree << endl;
	}
	if (f_create_random_k_subsets) {
		cout << "-create_random_k_subsets " << create_random_k_subsets_n
				<< " " << create_random_k_subsets_k
				<< " " << create_random_k_subsets_nb
				<< endl;
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
	if (f_domino_portrait) {
		cout << "-draw_layered_graph " << domino_portrait_D
				<< " " << domino_portrait_s
				<< " " << domino_portrait_fname;
			cout << endl;
		domino_portrait_draw_options->print();
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
	if (f_union) {
			cout << "-union " << union_set_of_sets_fname
				<< " " << union_input_fname
				<< " " << union_output_fname
				<< " " << endl;
	}
	if (f_dot_product_of_columns) {
		cout << "-dot_product_of_columns "
				<< dot_product_of_columns_fname
			<< " " << endl;
	}
	if (f_dot_product_of_rows) {
		cout << "-dot_product_of_rows "
				<< dot_product_of_rows_fname
			<< " " << endl;
	}
	if (f_matrix_multiply_over_Z) {
		cout << "-matrix_multiply_over_Z "
				<< " " << matrix_multiply_over_Z_label1
				<< " " << matrix_multiply_over_Z_label2
			<< " " << endl;
	}
	if (f_rowspan_over_R) {
		cout << "-rowspan_over_R "
				<< " " << rowspan_over_R_label
			<< " " << endl;
	}

}


void interface_combinatorics::worker(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::worker" << endl;
	}


	if (f_random_permutation) {

		combinatorics::combinatorics_domain Combi;

		Combi.create_random_permutation(random_permutation_degree,
				random_permutation_fname_csv, verbose_level);
	}
	else if (f_create_random_k_subsets) {

		combinatorics::combinatorics_domain Combi;
		string fname;

		fname = "random_k_subsets_n" + std::to_string(create_random_k_subsets_n)+ "_k" + std::to_string(create_random_k_subsets_k)+ "_nb" + std::to_string(create_random_k_subsets_nb)+ ".csv";

		Combi.create_random_k_subsets(create_random_k_subsets_n,
				create_random_k_subsets_k,
				create_random_k_subsets_nb,
				fname, verbose_level);

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

		do_conjugacy_classes_Sym_n_file(conjugacy_classes_Sym_n_n, verbose_level);

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
	else if (f_domino_portrait) {
		graphics::graphical_output GO;

		GO.do_domino_portrait(
				domino_portrait_D,
				domino_portrait_s,
				domino_portrait_fname,
				domino_portrait_draw_options,
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

	else if (f_union) {
		if (f_v) {
			cout << "interface_combinatorics::worker -union" << endl;
		}

		data_structures::algorithms Algo;


		Algo.union_of_sets(union_set_of_sets_fname,
				union_input_fname, union_output_fname, verbose_level);
	}

	else if (f_dot_product_of_columns) {
		if (f_v) {
			cout << "interface_combinatorics::worker -dot_product_of_columns" << endl;
		}

		data_structures::algorithms Algo;



		Algo.dot_product_of_columns(dot_product_of_columns_fname, verbose_level);
	}

	else if (f_dot_product_of_rows) {
		if (f_v) {
			cout << "interface_combinatorics::worker -dot_product_of_rows" << endl;
		}

		data_structures::algorithms Algo;



		Algo.dot_product_of_rows(dot_product_of_rows_fname, verbose_level);
	}


	else if (f_matrix_multiply_over_Z) {
		if (f_v) {
			cout << "interface_combinatorics::worker -matrix_multiply_over_Z" << endl;
		}

		data_structures::algorithms Algo;



		Algo.matrix_multiply_over_Z(
				matrix_multiply_over_Z_label1,
				matrix_multiply_over_Z_label2,
				verbose_level);
	}

	else if (f_rowspan_over_R) {
		if (f_v) {
			cout << "interface_combinatorics::worker -rowspan_over_R" << endl;
		}

		data_structures::algorithms Algo;



		Algo.matrix_rowspan_over_R(rowspan_over_R_label, verbose_level);
	}



}


#if 0
void interface_combinatorics::do_bent(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_bent" << endl;
	}

	{

		field_theory::finite_field *F;

		F = NEW_OBJECT(field_theory::finite_field);
		F->finite_field_init(2, false /* f_without_tables */, 0);

		combinatorics::boolean_function_domain *BF;

		BF = NEW_OBJECT(combinatorics::boolean_function_domain);

		if (f_v) {
			cout << "interface_combinatorics::do_bent before BF->init" << endl;
		}
		BF->init(F, n, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::do_bent after BF->init" << endl;
		}

		apps_combinatorics::boolean_function_classify *BFC;

		BFC = NEW_OBJECT(apps_combinatorics::boolean_function_classify);

		if (f_v) {
			cout << "interface_combinatorics::do_bent before BFC->init_group" << endl;
		}
		BFC->init_group(BF, verbose_level);
		// creates group PGGL(n+1,q)
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
		FREE_OBJECT(F);
	}

	if (f_v) {
		cout << "interface_combinatorics::do_bent done" << endl;
	}
}
#endif

void interface_combinatorics::do_conjugacy_classes_Sym_n(
		int n, int verbose_level)
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


	S.create(0);

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

void interface_combinatorics::do_conjugacy_classes_Sym_n_file(
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_conjugacy_classes_Sym_n_file" << endl;
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


	S.create(0);

	string fname;

	fname = "classes_Sym_" + std::to_string(n) + ".csv";

	{
		ofstream fp(fname);

		fp << "ROW,CYCLETYPE,CLASSSIZE" << endl;
		//cout << "The conjugacy classes in Sym_" << n << " are:" << endl;
		for (i = 0; i < cnt; i++) {
			//cout << i << " : ";
			//Int_vec_print(cout, Parts + i * n, n);
			//cout << " : ";

			fp << i;

			std::string part;


			Int_vec_create_string_with_quotes(part, Parts + i * n, n);

			fp << "," << part;

			C.size_of_conjugacy_class_in_sym_n(class_size, n, Parts + i * n);
			fp << "," << class_size;
			fp << endl;

			D.add_in_place(S, class_size);
			}

		D.factorial(F, n);
		D.integral_division_exact(F, S, A);
		if (!A.is_one()) {
			cout << "the class sizes do not add up" << endl;
			exit(1);
			}
		cout << "The sum of the class sizes is n!" << endl;
		fp << "END" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "interface_combinatorics::do_conjugacy_classes_Sym_n_file done" << endl;
	}
}



void interface_combinatorics::do_Delandtsheer_Doyen(
		apps_combinatorics::delandtsheer_doyen_description *Descr,
		int verbose_level)
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


}}}

