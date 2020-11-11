/*
 * interface_combinatorics.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace interfaces {



interface_combinatorics::interface_combinatorics()
{
	f_create_combinatorial_object = FALSE;
	Combinatorial_object_description = NULL;

	f_diophant = FALSE;
	Diophant_description = NULL;

	f_diophant_activity = FALSE;
	Diophant_activity_description = NULL;

	f_save = FALSE;
	fname_prefix = FALSE;
	f_process_combinatorial_objects = FALSE;
	Job = NULL;
	f_bent = FALSE;
	bent_n = 0;
	f_random_permutation = FALSE;
	random_permutation_degree = 0;
	//random_permutation_fname_csv = NULL;
	f_create_graph = FALSE;
	CG = NULL;
	//char fname_graph[1000];
	Create_graph_description = NULL;
	f_read_poset_file = FALSE;
	read_poset_file_fname = NULL;
	f_grouping = FALSE;
	x_stretch = 0.7;
	f_graph_theoretic_activity_description = FALSE;
	Graph_theoretic_activity_description = NULL;
	f_list_parameters_of_SRG = FALSE;
	v_max = 0;
	f_conjugacy_classes_Sym_n = FALSE;
	n = 0;
	f_tree_of_all_k_subsets = FALSE;
	tree_n = 0;
	tree_k = 0;
	f_Delandtsheer_Doyen = FALSE;
	Delandtsheer_Doyen_description = NULL;
	f_graph_classify = FALSE;
	Graph_classify_description = NULL;
	f_tdo_refinement = FALSE;
	Tdo_refinement_descr = NULL;
	f_tdo_print = FALSE;
	tdo_print_fname = NULL;
	f_create_design = FALSE;
	Design_create_description = NULL;
	f_convert_stack_to_tdo = FALSE;
	stack_fname = NULL;
	f_maximal_arc_parameters = FALSE;
	maximal_arc_parameters_q = 0;
	maximal_arc_parameters_r = 0;
	f_pentomino_puzzle = FALSE;

	f_regular_linear_space_classify = FALSE;
	Rls_descr = NULL;

	f_create_files = FALSE;
	Create_file_description = NULL;

	f_draw_layered_graph = FALSE;
	//draw_layered_graph_fname;
	Layered_graph_draw_options = NULL;

	f_read_solutions_and_tally = FALSE;
	//read_solutions_and_tally_fname
	read_solutions_and_tally_sz = 0;
}


void interface_combinatorics::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
		cout << "-create_combinatorial_object " << endl;
	}
	else if (strcmp(argv[i], "-diophant") == 0) {
		cout << "-diophant <description> " << endl;
	}
	else if (strcmp(argv[i], "-diophant_activity") == 0) {
		cout << "-diophant_activity <description> " << endl;
	}
	else if (strcmp(argv[i], "-save") == 0) {
		cout << "-save <fname> " << endl;
	}
	else if (strcmp(argv[i], "-process_combinatorial_objects") == 0) {
		cout << "-process_combinatorial_objects " << endl;
	}
	else if (strcmp(argv[i], "-bent") == 0) {
		cout << "-bent <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-random_permutation") == 0) {
		cout << "-random_permutation <ind : degree> <string : <fname_csv>" << endl;
	}
	else if (strcmp(argv[i], "-create_graph") == 0) {
		cout << "-create_graph <description>" << endl;
	}
	else if (strcmp(argv[i], "-read_poset_file") == 0) {
		cout << "-read_poset_file <string : file_name>" << endl;
	}
	else if (strcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
		cout << "-read_poset_file_with_grouping <string : file_name> <double : x_stretch>" << endl;
	}
	else if (strcmp(argv[i], "-graph_theoretic_activity") == 0) {
		cout << "-graph_theoretic_activity <description>" << endl;
	}
	else if (strcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		cout << "-list_parameters_of_SRG <int : v_max>" << endl;
	}
	else if (strcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		cout << "-conjugacy_classes_Sym_n <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		cout << "-tree_of_all_k_subsets <int : n> <int : k>" << endl;
	}
	else if (strcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
			cout << "-Delandtsheer_Doyen <description>" << endl;
	}
	else if (strcmp(argv[i], "-graph_classify") == 0) {
		cout << "-graph_classify <description -end>" << endl;
	}
	else if (strcmp(argv[i], "-tdo_refinement") == 0) {
		cout << "-tdo_refinement <options>" << endl;
	}
	else if (strcmp(argv[i], "-tdo_print") == 0) {
		cout << "-tdo_print <string : tdo-fname>" << endl;
	}
	else if (strcmp(argv[i], "-create_design") == 0) {
		cout << "-create_design <options>" << endl;
	}
	else if (strcmp(argv[i], "-convert_stack_to_tdo") == 0) {
		cout << "-convert_stack_to_tdo <string : stack_fname>" << endl;
	}
	else if (strcmp(argv[i], "-maximal_arc_parameters") == 0) {
		cout << "-maximal_arc_parameters <int : q > < int : r >" << endl;
	}
	else if (strcmp(argv[i], "-pentomino_puzzle") == 0) {
		cout << "-pentomino_puzzle" << endl;
	}
	else if (strcmp(argv[i], "-regular_linear_space_classify") == 0) {
		cout << "-regular_linear_space_classify <description>" << endl;
	}
	else if (strcmp(argv[i], "-create_files") == 0) {
		cout << "-create_files <description>" << endl;
	}
	else if (strcmp(argv[i], "-draw_layered_graph") == 0) {
		cout << "-draw_layered_graph <string : fname> <layered_graph_options>" << endl;
	}
	else if (strcmp(argv[i], "-read_solutions_and_tally") == 0) {
		cout << "-read_solutions_and_tally <string : fname> <int :read_solutions_and_tally_sz>" << endl;
	}
}

int interface_combinatorics::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-diophant") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-diophant_activity") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-save") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-process_combinatorial_objects") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-bent") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-random_permutation") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_graph") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-read_poset_file") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-graph_theoretic_activity") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-graph_classify") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-tdo_refinement") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-tdo_print") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_design") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-convert_stack_to_tdo") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-maximal_arc_parameters") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-pentomino_puzzle") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-regular_linear_space_classify") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_files") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-draw_layered_graph") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-read_solutions_and_tally") == 0) {
		return true;
	}
	return false;
}

void interface_combinatorics::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_combinatorics::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
			f_create_combinatorial_object = TRUE;
			cout << "-create_combinatorial_object " << endl;
			Combinatorial_object_description = NEW_OBJECT(combinatorial_object_description);
			i += Combinatorial_object_description->read_arguments(argc - i - 1,
					argv + i + 1, verbose_level);
			cout << "interface_combinatorics::read_arguments finished "
					"reading -create_combinatorial_object" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-diophant") == 0) {
			f_diophant = TRUE;
			Diophant_description = NEW_OBJECT(diophant_description);
			i += Diophant_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-diophant" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-diophant_activity") == 0) {
			f_diophant_activity = TRUE;
			Diophant_activity_description = NEW_OBJECT(diophant_activity_description);
			i += Diophant_activity_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-diophant_activity" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			fname_prefix = argv[++i];
			cout << "-save " << fname_prefix << endl;
		}
		else if (strcmp(argv[i], "-process_combinatorial_objects") == 0) {
			f_process_combinatorial_objects = TRUE;

			cout << "-process_combinatorial_objects " << endl;

			Job = NEW_OBJECT(projective_space_job_description);
			i += Job->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level) + 1;
			cout << "interface_combinatorics::read_arguments finished "
					"reading -process_combinatorial_objects" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-bent") == 0) {
			f_bent = TRUE;
			bent_n = atoi(argv[++i]);
			cout << "-bent " << bent_n << endl;
		}
		else if (strcmp(argv[i], "-random_permutation") == 0) {
			f_random_permutation = TRUE;
			random_permutation_degree = atoi(argv[++i]);
			random_permutation_fname_csv.assign(argv[++i]);
			cout << "-random_permutation " << random_permutation_degree << endl;
		}
		else if (strcmp(argv[i], "-create_graph") == 0) {
			f_create_graph = TRUE;
			Create_graph_description = NEW_OBJECT(create_graph_description);
			i += Create_graph_description->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-create_graph" << endl;
		}
		else if (strcmp(argv[i], "-read_poset_file") == 0) {
			f_read_poset_file = TRUE;
			f_grouping = FALSE;
			read_poset_file_fname = argv[++i];
			cout << "-read_poset_file " << read_poset_file_fname << endl;
		}
		else if (strcmp(argv[i], "-read_poset_file_with_grouping") == 0) {
			f_read_poset_file = TRUE;
			f_grouping = TRUE;
			read_poset_file_fname = argv[++i];
			x_stretch = atof(argv[++i]);
			cout << "-read_poset_file_with_grouping "
					<< read_poset_file_fname << " " << x_stretch << endl;
		}
		else if (strcmp(argv[i], "-graph_theoretic_activity") == 0) {
			f_graph_theoretic_activity_description = TRUE;
			Graph_theoretic_activity_description = NEW_OBJECT(graph_theoretic_activity_description);
			i += Graph_theoretic_activity_description->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-graph_activity" << endl;
		}
		else if (strcmp(argv[i], "-list_parameters_of_SRG") == 0) {
			f_list_parameters_of_SRG = TRUE;
			v_max = atoi(argv[++i]);
			cout << "-list_parameters_of_SRG " << v_max << endl;
		}
		else if (strcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
			f_conjugacy_classes_Sym_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-conjugacy_classes_Sym_n " << n << endl;
		}
		else if (strcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
			f_tree_of_all_k_subsets = TRUE;
			tree_n = atoi(argv[++i]);
			tree_k = atoi(argv[++i]);
			cout << "-tree_of_all_k_subsets " << tree_n << " " << tree_k << endl;
		}
		else if (strcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
			f_Delandtsheer_Doyen = TRUE;
			Delandtsheer_Doyen_description = NEW_OBJECT(delandtsheer_doyen_description);
			i += Delandtsheer_Doyen_description->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-Delandtsheer_Doyen" << endl;
		}
		else if (strcmp(argv[i], "-graph_classify") == 0) {
			f_graph_classify = TRUE;

			cout << "-graph_classify " << endl;

			Graph_classify_description = NEW_OBJECT(graph_classify_description);
			i += Graph_classify_description->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level) + 1;
			cout << "interface_combinatorics::read_arguments finished reading -graph_classify" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-graph_classify " << endl;
		}
		else if (strcmp(argv[i], "-tdo_refinement") == 0) {
			f_tdo_refinement = TRUE;
			cout << "-tdo_refinement " << endl;
			Tdo_refinement_descr = NEW_OBJECT(tdo_refinement_description);
			i += Tdo_refinement_descr->read_arguments(argc - i - 1,
					argv + i + 1, verbose_level) - 1;
			cout << "interface_combinatorics::read_arguments finished "
					"reading -tdo_refinement" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-tdo_print") == 0) {
			f_tdo_print = TRUE;
			tdo_print_fname = argv[++i];
			cout << "-tdo_print " << tdo_print_fname << endl;
		}
		else if (strcmp(argv[i], "-create_design") == 0) {
			f_create_design = TRUE;
			Design_create_description = NEW_OBJECT(design_create_description);
			i += Design_create_description->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-create_design" << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-convert_stack_to_tdo") == 0) {
			f_convert_stack_to_tdo = TRUE;
			stack_fname = argv[++i];
			cout << "-convert_stack_to_tdo " << stack_fname << endl;
		}
		else if (strcmp(argv[i], "-maximal_arc_parameters") == 0) {
			f_maximal_arc_parameters = TRUE;
			maximal_arc_parameters_q = atoi(argv[++i]);
			maximal_arc_parameters_r = atoi(argv[++i]);
			cout << "-maximal_arc_parameters " << maximal_arc_parameters_q
					<< " " << maximal_arc_parameters_r << endl;
		}
		else if (strcmp(argv[i], "-pentomino_puzzle") == 0) {
			f_pentomino_puzzle = TRUE;
			cout << "-pentomino_puzzle " <<endl;
		}
		else if (strcmp(argv[i], "-regular_linear_space_classify") == 0) {
			f_regular_linear_space_classify = TRUE;

			cout << "-regular_linear_space_classify " << endl;

			Rls_descr = NEW_OBJECT(regular_linear_space_description);
			i += Rls_descr->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level) + 1;
			cout << "interface_combinatorics::read_arguments finished "
					"reading -regular_linear_space_classify" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}

			cout << "-regular_linear_space_classify " <<endl;
		}
		else if (strcmp(argv[i], "-create_files") == 0) {
			f_create_files = TRUE;

			cout << "-create_files " << endl;

			Create_file_description = NEW_OBJECT(create_file_description);
			i += Create_file_description->read_arguments(argc - i - 1,
				argv + i + 1, verbose_level) + 1;
			cout << "interface_combinatorics::read_arguments finished "
					"reading -create_files" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}

			cout << "-create_files " <<endl;
		}
		else if (strcmp(argv[i], "-draw_layered_graph") == 0) {
			f_draw_layered_graph = TRUE;
			draw_layered_graph_fname.assign(argv[++i]);
			cout << "-draw_layered_graph " << endl;
			Layered_graph_draw_options = NEW_OBJECT(layered_graph_draw_options);
			i += Layered_graph_draw_options->read_arguments(argc - i - 1,
					argv + i + 1, verbose_level);
			cout << "interface_combinatorics::read_arguments finished reading -draw_layered_graph" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-read_solutions_and_tally") == 0) {
			f_read_solutions_and_tally = TRUE;
			read_solutions_and_tally_fname.assign(argv[++i]);
			read_solutions_and_tally_sz = atoi(argv[++i]);
			cout << "-read_solutions_and_tally " << read_solutions_and_tally_fname << " " << read_solutions_and_tally_sz << endl;
		}
	}
	cout << "interface_combinatorics::read_arguments done" << endl;
}


void interface_combinatorics::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::worker" << endl;
	}
	if (f_create_combinatorial_object) {
		do_create_combinatorial_object(verbose_level);
	}
	else if (f_diophant) {
		do_diophant(Diophant_description, verbose_level);
	}
	else if (f_diophant_activity) {
		do_diophant_activity(Diophant_activity_description, verbose_level);
	}
	else if (f_process_combinatorial_objects) {
		do_process_combinatorial_object(verbose_level);
	}
	else if (f_bent) {
		do_bent(bent_n, verbose_level);
	}
	else if (f_random_permutation) {
		do_random_permutation(random_permutation_degree,
				random_permutation_fname_csv, verbose_level);
	}
	else if (f_create_graph || f_graph_theoretic_activity_description) {
		if (f_create_graph) {
			do_create_graph(Create_graph_description, verbose_level);
		}
		if (f_graph_theoretic_activity_description) {
			if (!f_create_graph) {
				cout << "-graph_activity needs -create_graph" << endl;
				exit(1);
			}
			do_graph_theoretic_activity(Graph_theoretic_activity_description, verbose_level);
		}
	}
	else if (f_read_poset_file) {

		do_read_poset_file(read_poset_file_fname, f_grouping, x_stretch, verbose_level);
	}
	else if (f_list_parameters_of_SRG) {

		graph_theory_domain G;

		G.list_parameters_of_SRG(v_max, verbose_level);
	}
	else if (f_conjugacy_classes_Sym_n) {

		do_conjugacy_classes_Sym_n(n, verbose_level);
	}
	else if (f_tree_of_all_k_subsets) {

		do_make_tree_of_all_k_subsets(tree_n, tree_k, verbose_level);
	}
	else if (f_Delandtsheer_Doyen) {

		do_Delandtsheer_Doyen(Delandtsheer_Doyen_description, verbose_level);
	}
	else if (f_graph_classify) {

		do_graph_classify(Graph_classify_description, verbose_level);
	}
	else if (f_tdo_refinement) {

		combinatorics_domain Combi;

		Combi.do_tdo_refinement(Tdo_refinement_descr, verbose_level);
	}
	else if (f_tdo_print) {

		combinatorics_domain Combi;

		Combi.do_tdo_print(tdo_print_fname, verbose_level);
	}
	else if (f_create_design) {

		do_create_design(Design_create_description, verbose_level);
	}
	else if (f_convert_stack_to_tdo) {

		convert_stack_to_tdo(stack_fname, verbose_level);
	}
	else if (f_maximal_arc_parameters) {

		do_parameters_maximal_arc(maximal_arc_parameters_q, maximal_arc_parameters_r, verbose_level);
	}
	else if (f_pentomino_puzzle) {
		cout << "pentomino_puzzle " <<endl;

		pentomino_puzzle *P;

		P = NEW_OBJECT(pentomino_puzzle);

		P->main(verbose_level);

		FREE_OBJECT(P);

	}
	else if (f_regular_linear_space_classify) {

		regular_ls_classify *Rls;

		Rls = NEW_OBJECT(regular_ls_classify);

		if (f_v) {
			cout << "interface_combinatorics::worker before Rls->init_and_run" << endl;
		}
		Rls->init_and_run(Rls_descr, verbose_level);
		if (f_v) {
			cout << "interface_combinatorics::worker after Rls->init_and_run" << endl;
		}
		FREE_OBJECT(Rls);

	}
	else if (f_create_files) {
		file_io Fio;

		Fio.create_file(Create_file_description, verbose_level);
	}
	else if (f_draw_layered_graph) {
		graphical_output GO;

		GO.draw_layered_graph_from_file(draw_layered_graph_fname,
				Layered_graph_draw_options,
				verbose_level);

	}
	else if (f_read_solutions_and_tally) {

		file_io Fio;
		int nb_solutions;
		int solution_size = read_solutions_and_tally_sz;
		int *Sol;
		int i, j;

		std::vector<std::vector<int> > Solutions;


		Fio.read_solutions_from_file_size_is_known(read_solutions_and_tally_fname,
				Solutions, solution_size,
				verbose_level);

		nb_solutions = Solutions.size();

		Sol = NEW_int(nb_solutions * solution_size);
		for (i = 0; i < nb_solutions; i++) {
			for (j = 0; j < solution_size; j++) {
				Sol[i * solution_size + j] = Solutions[i][j];
			}
		}


		cout << "nb_solutions = " << nb_solutions << endl;

		tally T;

		T.init(Sol, nb_solutions * solution_size, TRUE, 0);
		cout << "tally:" << endl;
		T.print(TRUE);
		cout << endl;


		int *Pts;
		int nb_pts;
		int multiplicity = 4;

		T.get_data_by_multiplicity(
				Pts, nb_pts, multiplicity, verbose_level);

		cout << "multiplicity " << multiplicity << " number of pts = " << nb_pts << endl;
		int_vec_print(cout, Pts, nb_pts);
		cout << endl;


		FREE_int(Sol);

	}
}

void interface_combinatorics::do_graph_theoretic_activity(
		graph_theoretic_activity_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_graph_theoretic_activity" << endl;
	}

	if (Descr->f_find_cliques) {
		cout << "before Clique_finder_control->all_cliques" << endl;
		Descr->Clique_finder_control->all_cliques(
				CG, fname_graph,
				verbose_level);
		cout << "after Clique_finder_control->all_cliques" << endl;

		cout << "nb_sol = " << Descr->Clique_finder_control->nb_sol << endl;
	}
	if (Descr->f_export_magma) {

		cout << "export_magma" << endl;

		string fname_magma;
		string fname_text;

		fname_magma.assign(fname_graph);
		//strcpy(fname_magma, fname_graph);

		fname_text.assign(fname_graph);
		//strcpy(fname_text, fname_graph);


		replace_extension_with(fname_magma, ".magma");
		replace_extension_with(fname_text, ".txt");

		cout << "exporting to magma as " << fname_magma << endl;


		CG->export_to_magma(fname_magma, verbose_level);

		CG->export_to_text(fname_text, verbose_level);

		cout << "export_magma done" << endl;
	}

	if (Descr->f_export_csv) {

		cout << "export_csv" << endl;

		string fname_csv;

		fname_csv.assign(fname_graph);


		replace_extension_with(fname_csv, ".csv");

		cout << "exporting to csv as " << fname_csv << endl;


		CG->export_to_csv(fname_csv, verbose_level);


		cout << "export_csv done" << endl;
	}


	if (Descr->f_export_maple) {

		cout << "export_maple" << endl;

		string fname_maple;

		fname_maple.assign(fname_graph);


		replace_extension_with(fname_maple, ".maple");

		cout << "exporting to maple as " << fname_maple << endl;


		CG->export_to_maple(fname_maple, verbose_level);

		cout << "export_maple done" << endl;
	}
	if (Descr->f_print) {
		CG->print();
	}
	if (Descr->f_sort_by_colors) {
		colored_graph *CG2;
		string fname2;

		fname2.assign(fname_graph);
		//strcpy(fname2, fname_graph);
		replace_extension_with(fname2, "_sorted.bin");
		CG2 = CG->sort_by_color_classes(verbose_level);
		CG2->save(fname2, verbose_level);
		FREE_OBJECT(CG2);
	}

#if 0
	if (Descr->f_split) {
		cout << "splitting by file " << Descr->split_file << endl;
		file_io Fio;
		long int *Split;
		string fname_out;
		char extension[1000];
		int m, n;
		int a, c;

		Fio.lint_matrix_read_csv(Descr->split_file, Split, m, n, verbose_level - 2);
		cout << "We found " << m << " cases for splitting" << endl;
		for (c = 0; c < m; c++) {

			cout << "splitting case " << c << " / " << m << ":" << endl;
			a = Split[2 * c + 0];

			colored_graph *Subgraph;
			fancy_set *color_subset;
			fancy_set *vertex_subset;

			Subgraph = CG->compute_neighborhood_subgraph(a,
					vertex_subset, color_subset, verbose_level);


			fname_out.assign(fname_graph);
			snprintf(extension, 1000, "_case_%03d.bin", c);
			replace_extension_with(fname_out, extension);


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}
#endif

	if (Descr->f_save) {
		cout << "before save fname_graph=" << fname_graph << endl;
		CG->save(fname_graph, verbose_level);
		cout << "after save" << endl;
	}



	if (f_v) {
		cout << "interface_combinatorics::do_graph_theoretic_activity done" << endl;
	}
}

void interface_combinatorics::do_create_graph(
		create_graph_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_create_graph" << endl;
	}

	create_graph *Gr;

	Gr = NEW_OBJECT(create_graph);

	if (f_v) {
		cout << "before Gr->init" << endl;
	}
	Gr->init(Descr, verbose_level);
	if (f_v) {
		cout << "after Gr->init" << endl;
	}
	if (f_v) {
		cout << "Gr->N=" << Gr->N << endl;
		cout << "Gr->label=" << Gr->label << endl;
		cout << "Adj:" << endl;
		int_matrix_print(Gr->Adj, Gr->N, Gr->N);
	}



	if (Gr->f_has_CG) {
		CG = Gr->CG;
		Gr->f_has_CG = FALSE;
		Gr->CG = NULL;
	}
	else {
		CG = NEW_OBJECT(colored_graph);
		CG->init_adjacency_no_colors(Gr->N, Gr->Adj, verbose_level);
	}

	fname_graph.assign(Gr->label);
	replace_extension_with(fname_graph, ".colored_graph");
	//snprintf(fname_graph, 2000, "%s.colored_graph", Gr->label);

	//CG->save(fname_graph, verbose_level);

	//FREE_OBJECT(CG);



	if (f_v) {
		cout << "interface_combinatorics::do_create_graph done" << endl;
	}
}

void interface_combinatorics::do_read_poset_file(const char *fname,
		int f_grouping, double x_stretch, int verbose_level)
// creates a layered graph file from a text file
// which was created by DISCRETA/sgls2.cpp
// for an example, see the bottom of this file.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_read_poset_file" << endl;
	}

	layered_graph *LG;

	LG = NEW_OBJECT(layered_graph);
	LG->init_poset_from_file(fname, f_grouping, x_stretch, verbose_level - 1);


	string fname_out;
	file_io Fio;

	fname_out.assign(fname);

	replace_extension_with(fname_out, ".layered_graph");


	LG->write_file(fname_out, 0 /*verbose_level*/);

	cout << "Written file " << fname_out << " of size "
			<< Fio.file_size(fname_out) << endl;

	FREE_OBJECT(LG);

	if (f_v) {
		cout << "interface_combinatorics::do_read_poset_file done" << endl;
	}
}

void interface_combinatorics::do_create_combinatorial_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "interface_combinatorics::do_create_combinatorial_object" << endl;
	}

	combinatorial_object_create *COC;

	COC = NEW_OBJECT(combinatorial_object_create);

	if (f_v) {
		cout << "before COC->init" << endl;
	}
	COC->init(Combinatorial_object_description, verbose_level);
	if (f_v) {
		cout << "after COC->init" << endl;
	}



	if (f_v) {
		cout << "we created a set of " << COC->nb_pts
				<< " points, called " << COC->fname << endl;

#if 0
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
#endif
	}





	if (f_save) {
		file_io Fio;
		string fname;

		fname.assign(fname_prefix);
		fname.append(COC->fname);

		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, COC->Pts, COC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	FREE_OBJECT(COC);

	if (f_v) {
		cout << "interface_combinatorics::do_create_combinatorial_object done" << endl;
	}
}


void interface_combinatorics::do_diophant(diophant_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_diophant" << endl;
	}

	diophant_create *DC;

	DC = NEW_OBJECT(diophant_create);

	DC->init(Descr, verbose_level);


	if (f_v) {
		cout << "interface_combinatorics::do_diophant done" << endl;
	}
}

void interface_combinatorics::do_diophant_activity(
		diophant_activity_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_diophant_activity" << endl;
	}

	diophant_activity *DA;

	DA = NEW_OBJECT(diophant_activity);

	DA->init(Descr, verbose_level);

	FREE_OBJECT(DA);

	if (f_v) {
		cout << "interface_combinatorics::do_diophant_activity done" << endl;
	}
}

void interface_combinatorics::do_process_combinatorial_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_process_combinatorial_object" << endl;
	}

	if (!Job->f_q) {
		cout << "please use option -q <q> within the job description" << endl;
		exit(1);
	}
	if (!Job->f_n) {
		cout << "please use option -n <n> to specify the projective "
				"dimension within the job description" << endl;
		exit(1);
	}
	if (!Job->f_fname_base_out) {
		cout << "please use option -fname_base_out <fname_base_out> "
				"within the job description" << endl;
		exit(1);
	}

	projective_space_job J;

	J.perform_job(Job, verbose_level);

	if (f_v) {
		cout << "interface_combinatorics::do_process_combinatorial_object done" << endl;
	}
}

void interface_combinatorics::do_bent(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_bent" << endl;
	}

	{
		boolean_function_domain *BF;

		BF = NEW_OBJECT(boolean_function_domain);

		BF->init(n, verbose_level);

		boolean_function_classify *BFC;

		BFC = NEW_OBJECT(boolean_function_classify);

		BFC->init_group(BF, verbose_level);

		BFC->search_for_bent_functions(verbose_level);

		FREE_OBJECT(BFC);
		FREE_OBJECT(BF);
	}

	if (f_v) {
		cout << "interface_combinatorics::do_bent done" << endl;
	}
}

void interface_combinatorics::do_random_permutation(int deg,
		std::string &fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_random_permutation" << endl;
	}

	{
		combinatorics_domain Combi;
		file_io Fio;


		int *P;

		P = NEW_int(deg);
		Combi.random_permutation(P, deg);

		Fio.int_vec_write_csv(P, deg, fname_csv, "perm");
	}

	if (f_v) {
		cout << "interface_combinatorics::do_random_permutation done" << endl;
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
	longinteger_object class_size, S, F, A;
	longinteger_domain D;
	combinatorics_domain C;
	combinatorics_domain Combi;

	cnt = Combi.count_partitions(n);

	int *Parts;

	Parts = NEW_int(cnt * n);
	Combi.make_partitions(n, Parts, cnt);


	S.create(0, __FILE__, __LINE__);

	cout << "The conjugacy classes in Sym_" << n << " are:" << endl;
	for (i = 0; i < cnt; i++) {
		cout << i << " : ";
		int_vec_print(cout, Parts + i * n, n);
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

void interface_combinatorics::do_make_tree_of_all_k_subsets(int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_make_tree_of_all_k_subsets" << endl;
	}

	combinatorics_domain Combi;
	int *set;
	int N;
	int h, i;
	char fname[1000];


	snprintf(fname, 1000, "all_k_subsets_%d_%d.tree", n, k);
	set = NEW_int(k);
	N = Combi.int_n_choose_k(n, k);


	{
		ofstream fp(fname);

		for (h = 0; h < N; h++) {
			Combi.unrank_k_subset(h, set, n, k);
			fp << k;
			for (i = 0; i < k; i++) {
				fp << " " << set[i];
				}
			fp << endl;
			}
		fp << "-1" << endl;
	}
	FREE_int(set);

	if (f_v) {
		cout << "interface_combinatorics::do_make_tree_of_all_k_subsets done" << endl;
	}
}

void interface_combinatorics::do_Delandtsheer_Doyen(delandtsheer_doyen_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_Delandtsheer_Doyen" << endl;
	}

	delandtsheer_doyen *DD;

	DD = NEW_OBJECT(delandtsheer_doyen);

	DD->init(Descr, verbose_level);

	FREE_OBJECT(DD);


	if (f_v) {
		cout << "interface_combinatorics::do_Delandtsheer_Doyen done" << endl;
	}
}


void interface_combinatorics::do_graph_classify(graph_classify_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_graph_classify" << endl;
	}
	{
	graph_classify Gen;
	int schreier_depth = 10000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int depth;
	//int f_embedded = TRUE;
	//int f_sideways = FALSE;

	os_interface Os;
	int t0 = Os.os_ticks();


	Gen.init(Descr, verbose_level);


	depth = Gen.gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);
	cout << "Gen.gen->main returns depth=" << depth << endl;

	if (Gen.Descr->f_tournament) {
		Gen.print_score_sequences(depth, verbose_level);
		}

	//Gen.gen->draw_poset(Gen.gen->fname_base, depth,
	//Gen.n /* data1 */, f_embedded, Gen.gen->verbose_level);


#if 0
	if (Gen.Descr->Control->f_draw_poset) {
		Gen.gen->draw_poset(Gen.gen->get_problem_label_with_path(), depth,
			Gen.Descr->n /* data1 */, f_embedded, f_sideways, 100 /* rad */,
			verbose_level);
		}


	if (Gen.Descr->Control->f_draw_full_poset) {
		//double x_stretch = 0.4;
		cout << "Gen.f_draw_full_poset" << endl;
		Gen.gen->draw_poset_full(Gen.gen->get_problem_label_with_path(), depth,
			Gen.Descr->n /* data1 */, f_embedded, f_sideways, 100 /* rad */,
			Gen.Descr->x_stretch, verbose_level);

		const char *fname_prefix = "flag_orbits";

		Gen.gen->make_flag_orbits_on_relations(
				depth, fname_prefix, verbose_level);
		}

	//Gen.gen->print_data_structure_tex(depth, Gen.gen->verbose_level);

	if (Gen.Descr->Control->f_plesken) {
		latex_interface L;
		int *P;
		int N;
		Gen.gen->Plesken_matrix_up(depth, P, N, verbose_level);
		cout << "Plesken matrix up:" << endl;
		L.int_matrix_print_tex(cout, P, N, N);

		FREE_int(P);
		Gen.gen->Plesken_matrix_down(depth, P, N, verbose_level);
		cout << "Plesken matrix down:" << endl;
		L.int_matrix_print_tex(cout, P, N, N);

		FREE_int(P);
		}

	if (Gen.Descr->Control->f_list) {
		int f_show_orbit_decomposition = FALSE;
		int f_show_stab = FALSE;
		int f_save_stab = FALSE;
		int f_show_whole_orbit = FALSE;

		Gen.gen->list_all_orbits_at_level(Gen.Descr->Control->depth,
			FALSE, NULL, NULL,
			f_show_orbit_decomposition,
			f_show_stab, f_save_stab, f_show_whole_orbit);
		}

	if (Gen.Descr->Control->f_list_all) {
		int f_show_orbit_decomposition = FALSE;
		int f_show_stab = FALSE;
		int f_save_stab = FALSE;
		int f_show_whole_orbit = FALSE;
		int j;

		for (j = 0; j <= Gen.Descr->Control->depth; j++) {
			Gen.gen->list_all_orbits_at_level(j,
				FALSE, NULL, NULL,
				f_show_orbit_decomposition,
				f_show_stab, f_save_stab, f_show_whole_orbit);
			}
		}
#endif

	if (Gen.Descr->f_draw_graphs) {
#if 0
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax = 1000000;
		int ymax = 1000000;
#endif
		int level;

		for (level = 0; level <= Gen.Descr->Control->depth; level++) {
			Gen.draw_graphs(level, //Gen.Descr->Control->scale,
					Descr->draw_options,
					//xmax_in, ymax_in, xmax, ymax,
					//Gen.Descr->Control->f_embedded, Gen.Descr->Control->f_sideways,
					verbose_level);
			}
		}

	if (Gen.Descr->f_draw_graphs_at_level) {
#if 0
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax = 1000000;
		int ymax = 1000000;
#endif

		cout << "before Gen.draw_graphs" << endl;
		Gen.draw_graphs(Gen.Descr->level,
				Descr->draw_options,
				//Gen.Descr->Control->scale,
				//xmax_in, ymax_in, xmax, ymax,
				//Gen.Descr->Control->f_embedded, Gen.Descr->Control->f_sideways,
				verbose_level);
		cout << "after Gen.draw_graphs" << endl;
		}

	if (Gen.Descr->f_draw_level_graph) {

		if (!Descr->f_draw_options) {
			cout << "please use option -draw_options" << endl;
			exit(1);
		}

		Gen.gen->draw_level_graph(Gen.gen->get_problem_label_with_path(),
				Gen.Descr->Control->depth, Gen.Descr->n /* data1 */,
				Gen.Descr->level_graph_level,
				Descr->draw_options,
				//f_embedded, f_sideways,
				verbose_level - 3);
		}

	if (Gen.Descr->f_test_multi_edge) {
		Gen.gen->test_for_multi_edge_in_classification_graph(
				depth, verbose_level);
		}
	if (Gen.Descr->f_identify) {
		int *transporter;
		int orbit_at_level;

		transporter = NEW_int(Gen.gen->get_A()->elt_size_in_int);

		Gen.gen->identify(Gen.Descr->identify_data, Gen.Descr->identify_data_sz,
				transporter, orbit_at_level, verbose_level);

		FREE_int(transporter);
		}

	int N, F, level;

	N = 0;
	F = 0;
	for (level = 0; level <= Gen.Descr->Control->depth; level++) {
		N += Gen.gen->nb_orbits_at_level(level);
		}
	for (level = 0; level < Gen.Descr->Control->depth; level++) {
		F += Gen.gen->nb_flag_orbits_up_at_level(level);
		}
	cout << "N=" << N << endl;
	cout << "F=" << F << endl;
	} // clean up graph_generator
	if (f_v) {
		cout << "interface_combinatorics::do_graph_classify done" << endl;
	}

}


void interface_combinatorics::do_create_design(design_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_create_design" << endl;
	}

	design_create *DC;
	latex_interface L;
	//int j;

	DC = NEW_OBJECT(design_create);

	if (f_v) {
		cout << "before DC->init" << endl;
	}
	DC->init(Descr, verbose_level);
	if (f_v) {
		cout << "after DC->init" << endl;
	}



	//action *A;
	//int *Elt1;
	//int *Elt2;

	//A = DC->A;

	//Elt2 = NEW_int(A->elt_size_in_int);




	if (f_v) {
		cout << "We have created the following design:" << endl;
		cout << "$$" << endl;
		L.lint_set_print_tex(cout, DC->set, DC->sz);
		cout << endl;
		cout << "$$" << endl;
	}

	if (DC->f_has_group) {
		if (f_v) {
			cout << "The stabilizer is generated by:" << endl;
			DC->Sg->print_generators_tex(cout);
		}
	}


	{
		int nb_pts = DC->P->N_points;
		int nb_blocks = DC->sz;
		int *Incma;
		int h, i, j; //, a;
		int pts_per_element = DC->k;


		Incma = NEW_int(nb_pts * nb_blocks);
		int_vec_zero(Incma, nb_pts * nb_blocks);

		for (j = 0; j < nb_blocks; j++) {
			//cout << "j=" << j << " / " << set_size
			//<< " the_set[j]=" << the_set[j] << endl;
			//Grass->unrank_int(the_set[j], 0/*verbose_level - 4*/);

			//a = DC->set[j];
			DC->unrank_block_in_PG_2_q(DC->block,
					DC->set[j], 0 /* verbose_level*/);
			for (h = 0; h < pts_per_element; h++) {
				//PG_element_unrank_modified(*F, v, 1, k, h);
				//F->mult_vector_from_the_left(v, Grass->M, w, k, n);
				//PG_element_rank_modified(*F, w, 1, n, i);
				i = DC->block[h];
				Incma[i * nb_blocks + j] = 1;
				}
			}

		if (f_v) {
			cout << "Computing incidence matrix done" << endl;
		}



		incidence_structure *Inc;
		partitionstack *Stack;


		if (f_v) {
			cout << "Opening incidence data structure:" << endl;
		}

		Inc = NEW_OBJECT(incidence_structure);
		Inc->init_by_matrix(nb_pts, nb_blocks, Incma, 0 /* verbose_level */);
		Stack = NEW_OBJECT(partitionstack);
		Stack->allocate(nb_pts + nb_blocks, 0 /* verbose_level */);
		Stack->subset_continguous(nb_pts, nb_blocks);
		Stack->split_cell(0 /* verbose_level */);
		Stack->sort_cells();


		if (f_v) {
			cout << "Initial scheme:" << endl;
			Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
			Stack->print_classes_points_and_lines(cout);
			//print_decomposition(Grass, Stack, the_set);
		}
		Inc->refine_row_partition_safe(*Stack, 0/*verbose_level - 3*/);
		if (f_v) {
			cout << "Row-scheme:" << endl;
			Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
			Stack->print_classes_points_and_lines(cout);
			//print_decomposition(Grass, Stack, the_set);
		}
		Inc->refine_column_partition_safe(*Stack, 0/*verbose_level - 3*/);
		if (f_v) {
			cout << "Column-scheme:" << endl;
			Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
			Stack->print_classes_points_and_lines(cout);
			//print_decomposition(Grass, Stack, the_set);
		}
		Inc->refine_row_partition_safe(*Stack, 0/*verbose_level - 3*/);
		if (f_v) {
			cout << "Row-scheme:" << endl;
			Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
			Stack->print_classes_points_and_lines(cout);
			//print_decomposition(Grass, Stack, the_set);
		}
	}



	if (f_v) {
		cout << "interface_combinatorics::do_create_design done" << endl;
	}

}


void interface_combinatorics::convert_stack_to_tdo(const char *stack_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	char str[1000];
	char ext[1000];
	char fname_out[2000];
	char label[2000];

	if (f_v) {
		cout << "interface_combinatorics::convert_stack_to_tdo" << endl;
	}
	strcpy(str, stack_fname);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);
	snprintf(fname_out, 2000, "%s.tdo", str);

	if (f_v) {
		cout << "reading stack file " << stack_fname << endl;
	}
	{
		geo_parameter GP;
		tdo_scheme G;
		ifstream f(stack_fname);
		ofstream g(fname_out);
		for (i = 0; ; i++) {
			if (f.eof()) {
				if (f_v) {
					cout << "end of file reached" << endl;
				}
				break;
				}
			if (!GP.input(f)) {
				if (f_v) {
					cout << "GP.input returns false" << endl;
				}
				break;
				}
			if (f_v) {
				cout << "read decomposition " << i
							<< " v=" << GP.v << " b=" << GP.b << endl;
			}
			GP.convert_single_to_stack(verbose_level - 1);
			if (f_v) {
				cout << "after convert_single_to_stack" << endl;
			}
			if (strlen(GP.label)) {
				snprintf(label, 2000, "%s", GP.label);
			}
			else {
				snprintf(label, 2000, "%d", i);
			}
			GP.write(g, label);
			if (f_v) {
				cout << "after write" << endl;
			}
			GP.init_tdo_scheme(G, verbose_level - 1);
			if (f_v) {
				cout << "after init_tdo_scheme" << endl;
			}
			if (f_vv) {
				GP.print_schemes(G);
			}
		}
		g << "-1 " << i << endl;
	}
	if (f_v) {
		file_io Fio;
		cout << "written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;
		cout << "interface_combinatorics::convert_stack_to_tdo done" << endl;
	}
}

void interface_combinatorics::do_parameters_maximal_arc(int q, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 2;
	int v[2], b[2], aij[4];
	int Q;
	char fname[1000];
	file_io Fio;

	if (f_v) {
		cout << "interface_combinatorics::do_parameters_maximal_arc q=" << q << " r=" << r << endl;
	}

	Q = q * q;
	v[0] = q * (r - 1) + r;
	v[1] = Q + q * (2 - r) - r + 1;
	b[0] = Q - Q / r + q * 2 - q / r + 1;
	b[1] = Q / r + q / r - q;
	aij[0] = q + 1;
	aij[1] = 0;
	aij[2] = q - q / r + 1;
	aij[3] = q / r;
	snprintf(fname, 1000, "max_arc_q%d_r%d.stack", q, r);

	Fio.write_decomposition_stack(fname, m, n, v, b, aij, verbose_level - 1);
}







}}
