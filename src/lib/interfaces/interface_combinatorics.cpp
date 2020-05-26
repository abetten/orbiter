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
	argc = 0;
	argv = NULL;

	f_create_combinatorial_object = FALSE;
	Combinatorial_object_description = NULL;
	f_save = FALSE;
	fname_prefix = FALSE;
	f_process_combinatorial_objects = FALSE;
	Job = NULL;
	f_bent = FALSE;
	bent_n = 0;
	f_random_permutation = FALSE;
	random_permutation_degree = 0;
	random_permutation_fname_csv = NULL;
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
	f_Kramer_Mesner = FALSE;
	f_tree_of_all_k_subsets = FALSE;
	tree_n = 0;
	tree_k = 0;
	f_Delandtsheer_Doyen = FALSE;
	Delandtsheer_Doyen_description = NULL;
	f_graph_classify = FALSE;
	f_tdo_refinement = FALSE;
	Tdo_refinement_descr = NULL;
	f_create_design = FALSE;
	Design_create_description = NULL;
}


void interface_combinatorics::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
		cout << "-create_combinatorial_object " << endl;
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
	else if (strcmp(argv[i], "-graph_activity") == 0) {
		cout << "-graph_activity <description>" << endl;
	}
	else if (strcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		cout << "-list_parameters_of_SRG <int : v_max>" << endl;
	}
	else if (strcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		cout << "-conjugacy_classes_Sym_n <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-Kramer_Mesner") == 0) {
		cout << "-Kramer_Mesner <description>" << endl;
	}
	else if (strcmp(argv[i], "-tree_of_all_k_subsets") == 0) {
		cout << "-tree_of_all_k_subsets <int : n> <int : k>" << endl;
	}
	else if (strcmp(argv[i], "-Delandtsheer_Doyen") == 0) {
			cout << "-Delandtsheer_Doyen <description>" << endl;
	}
	else if (strcmp(argv[i], "-graph_classify") == 0) {
		cout << "-graph_classify <options>" << endl;
	}
	else if (strcmp(argv[i], "-tdo_refinement") == 0) {
		cout << "-tdo_refinement <options>" << endl;
	}
	else if (strcmp(argv[i], "-create_design") == 0) {
		cout << "-create_design <options>" << endl;
	}
}

int interface_combinatorics::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
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
	else if (strcmp(argv[i], "-graph_activity") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-list_parameters_of_SRG") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-conjugacy_classes_Sym_n") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-Kramer_Mesner") == 0) {
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
	else if (strcmp(argv[i], "-create_design") == 0) {
		return true;
	}
	return false;
}

void interface_combinatorics::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_combinatorics::read_arguments" << endl;

	interface_combinatorics::argc = argc;
	interface_combinatorics::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
			f_create_combinatorial_object = TRUE;
			cout << "-create_combinatorial_object " << endl;
			Combinatorial_object_description = NEW_OBJECT(combinatorial_object_description);
			i += Combinatorial_object_description->read_arguments(argc - i - 1,
					argv + i + 1, verbose_level) - 1;
			cout << "interface_combinatorics::read_arguments finished reading -create_combinatorial_object" << endl;
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
			cout << "interface_combinatorics::read_arguments finished reading -process_combinatorial_objects" << endl;
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
			random_permutation_fname_csv = argv[++i];
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
			cout << "-read_poset_file_with_grouping " << read_poset_file_fname << " " << x_stretch << endl;
		}
		else if (strcmp(argv[i], "-graph_activity") == 0) {
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
		else if (strcmp(argv[i], "-Kramer_Mesner") == 0) {
			f_Kramer_Mesner = TRUE;
			cout << "-Kramer_Mesner " << endl;
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
		}
		if (strcmp(argv[i], "-tdo_refinement") == 0) {
			f_tdo_refinement = TRUE;
			cout << "-tdo_refinement " << endl;
			Tdo_refinement_descr = NEW_OBJECT(tdo_refinement_description);
			i += Tdo_refinement_descr->read_arguments(argc - i - 1,
					argv + i + 1, verbose_level) - 1;
			cout << "interface_combinatorics::read_arguments finished reading -tdo_refinement" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
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
	}
	cout << "interface_combinatorics::read_arguments done" << endl;
}


void interface_combinatorics::worker(int verbose_level)
{
	if (f_create_combinatorial_object) {
		do_create_combinatorial_object(verbose_level);
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
	else if (f_Kramer_Mesner) {

		do_Kramer_Mesner(verbose_level);
	}
	else if (f_tree_of_all_k_subsets) {

		do_make_tree_of_all_k_subsets(tree_n, tree_k, verbose_level);
	}
	else if (f_Delandtsheer_Doyen) {

		do_Delandtsheer_Doyen(Delandtsheer_Doyen_description, verbose_level);
	}
	else if (f_graph_classify) {

		do_graph_classify(verbose_level);
	}
	else if (f_tdo_refinement) {

		do_tdo_refinement(Tdo_refinement_descr, verbose_level);
	}
	else if (f_create_design) {

		do_create_design(Design_create_description, verbose_level);
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
	else if (Descr->f_export_magma) {

		cout << "export_magma" << endl;

		char fname_magma[1000];
		char fname_text[1000];

		strcpy(fname_magma, fname_graph);

		strcpy(fname_text, fname_graph);


		replace_extension_with(fname_magma, ".magma");
		replace_extension_with(fname_text, ".txt");

		cout << "exporting to magma as " << fname_magma << endl;


		CG->export_to_magma(fname_magma, verbose_level);

		CG->export_to_text(fname_text, verbose_level);

		cout << "export_magma done" << endl;
	}

	else if (Descr->f_export_maple) {

		cout << "export_maple" << endl;

		char fname_maple[1000];

		strcpy(fname_maple, fname_graph);


		replace_extension_with(fname_maple, ".maple");

		cout << "exporting to maple as " << fname_maple << endl;


		CG->export_to_maple(fname_maple, verbose_level);

		cout << "export_maple done" << endl;
	}
	else if (Descr->f_print) {
		CG->print();
	}
	else if (Descr->f_sort_by_colors) {
		colored_graph *CG2;
		char fname2[1000];

		strcpy(fname2, fname_graph);
		replace_extension_with(fname2, "_sorted.bin");
		CG2 = CG->sort_by_color_classes(verbose_level);
		CG2->save(fname2, verbose_level);
		delete CG2;
	}

	else if (Descr->f_split) {
		cout << "splitting by file " << Descr->split_file << endl;
		file_io Fio;
		long int *Split;
		char fname_out[1000];
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

			sprintf(fname_out, "%s", fname_graph);
			sprintf(extension, "_case_%03d.bin", c);
			replace_extension_with(fname_out, extension);


			Subgraph->save(fname_out, verbose_level - 2);
		}
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

	sprintf(fname_graph, "%s.colored_graph", Gr->label);

	CG->save(fname_graph, verbose_level);

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


	char fname_out[1000];
	file_io Fio;

	sprintf(fname_out, "%s", fname);

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
	int i;

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
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
	}





	if (f_save) {
		file_io Fio;
		char fname[1000];

		sprintf(fname, "%s%s", fname_prefix, COC->fname);

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
		cout << "please use option -n <n> to specify the projective dimension within the job description" << endl;
		exit(1);
	}
	if (!Job->f_fname_base_out) {
		cout << "please use option -fname_base_out <fname_base_out> within the job description" << endl;
		exit(1);
	}

	Job->perform_job(verbose_level);

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
		boolean_function *BF;

		BF = NEW_OBJECT(boolean_function);

		BF->init(n, verbose_level);

		BF->search_for_bent_functions(verbose_level);

		FREE_OBJECT(BF);
	}

	if (f_v) {
		cout << "interface_combinatorics::do_bent done" << endl;
	}
}

void interface_combinatorics::do_random_permutation(int deg,
		const char *fname_csv, int verbose_level)
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

		D.size_of_conjugacy_class_in_sym_n(class_size, n, Parts + i * n);
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

void interface_combinatorics::do_Kramer_Mesner(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_Kramer_Mesner" << endl;
	}

	{
	kramer_mesner KM;

	cout << "km.cpp: before read_arguments" << endl;
	KM.read_arguments(argc, argv, verbose_level);

	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	sims *S;

	cout << "interface_combinatorics::do_Kramer_Mesner before init_group" << endl;
	KM.init_group(S, verbose_level);


	cout << "interface_combinatorics::do_Kramer_Mesner before orbits" << endl;
	KM.orbits(S, verbose_level);

	delete S;
	}

	if (f_v) {
		cout << "interface_combinatorics::do_Kramer_Mesner done" << endl;
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


	sprintf(fname, "all_k_subsets_%d_%d.tree", n, k);
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


void interface_combinatorics::do_graph_classify(int verbose_level)
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
	int f_embedded = TRUE;
	int f_sideways = FALSE;

	os_interface Os;
	int t0 = Os.os_ticks();


	Gen.init(argc, argv, verbose_level);

	int verbose_level = Gen.gen->Control->verbose_level;

	depth = Gen.gen->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		Gen.gen->Control->verbose_level);
	cout << "Gen.gen->main returns depth=" << depth << endl;

	if (Gen.f_tournament) {
		Gen.print_score_sequences(depth, verbose_level);
		}

	//Gen.gen->draw_poset(Gen.gen->fname_base, depth,
	//Gen.n /* data1 */, f_embedded, Gen.gen->verbose_level);


	if (Gen.Control->f_draw_poset) {
		Gen.gen->draw_poset(Gen.gen->fname_base, depth,
			Gen.n /* data1 */, f_embedded, f_sideways,
			verbose_level);
		}


	if (Gen.Control->f_draw_full_poset) {
		//double x_stretch = 0.4;
		cout << "Gen.f_draw_full_poset" << endl;
		Gen.gen->draw_poset_full(Gen.gen->fname_base, depth,
			Gen.n /* data1 */, f_embedded, f_sideways,
			Gen.x_stretch, verbose_level);

		const char *fname_prefix = "flag_orbits";

		Gen.gen->make_flag_orbits_on_relations(
				depth, fname_prefix, verbose_level);
		}

	//Gen.gen->print_data_structure_tex(depth, Gen.gen->verbose_level);

	if (Gen.Control->f_plesken) {
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

	if (Gen.Control->f_list) {
		int f_show_orbit_decomposition = FALSE;
		int f_show_stab = FALSE;
		int f_save_stab = FALSE;
		int f_show_whole_orbit = FALSE;

		Gen.gen->list_all_orbits_at_level(Gen.gen->depth,
			FALSE, NULL, NULL,
			f_show_orbit_decomposition,
			f_show_stab, f_save_stab, f_show_whole_orbit);
		}

	if (Gen.Control->f_list_all) {
		int f_show_orbit_decomposition = FALSE;
		int f_show_stab = FALSE;
		int f_save_stab = FALSE;
		int f_show_whole_orbit = FALSE;
		int j;

		for (j = 0; j <= Gen.gen->depth; j++) {
			Gen.gen->list_all_orbits_at_level(j,
				FALSE, NULL, NULL,
				f_show_orbit_decomposition,
				f_show_stab, f_save_stab, f_show_whole_orbit);
			}
		}

	if (Gen.f_draw_graphs) {
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax = 1000000;
		int ymax = 1000000;
		int level;

		for (level = 0; level <= Gen.gen->depth; level++) {
			Gen.draw_graphs(level, Gen.Control->scale,
					xmax_in, ymax_in, xmax, ymax,
					Gen.Control->f_embedded, Gen.Control->f_sideways,
					verbose_level);
			}
		}

	if (Gen.f_draw_graphs_at_level) {
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax = 1000000;
		int ymax = 1000000;

		cout << "before Gen.draw_graphs" << endl;
		Gen.draw_graphs(Gen.level, Gen.Control->scale,
				xmax_in, ymax_in, xmax, ymax,
				Gen.Control->f_embedded, Gen.Control->f_sideways,
				verbose_level);
		cout << "after Gen.draw_graphs" << endl;
		}

	if (Gen.f_draw_level_graph) {
		Gen.gen->draw_level_graph(Gen.gen->fname_base,
				Gen.gen->depth, Gen.n /* data1 */,
				Gen.level_graph_level,
				f_embedded, f_sideways,
				verbose_level - 3);
		}

	if (Gen.f_test_multi_edge) {
		Gen.gen->test_for_multi_edge_in_classification_graph(
				depth, verbose_level);
		}
	if (Gen.f_identify) {
		int *transporter;
		int orbit_at_level;

		transporter = NEW_int(Gen.gen->Poset->A->elt_size_in_int);

		Gen.gen->identify(Gen.identify_data, Gen.identify_data_sz,
				transporter, orbit_at_level, verbose_level);

		FREE_int(transporter);
		}

	int N, F, level;

	N = 0;
	F = 0;
	for (level = 0; level <= Gen.gen->depth; level++) {
		N += Gen.gen->nb_orbits_at_level(level);
		}
	for (level = 0; level < Gen.gen->depth; level++) {
		F += Gen.gen->nb_flag_orbits_up_at_level(level);
		}
	cout << "N=" << N << endl;
	cout << "F=" << F << endl;
	} // clean up graph_generator
	if (f_v) {
		cout << "interface_combinatorics::do_graph_classify done" << endl;
	}

}


void interface_combinatorics::do_tdo_refinement(tdo_refinement_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_tdo_refinement" << endl;
	}

	tdo_refinement *R;

	R = NEW_OBJECT(tdo_refinement);

	R->init(Descr, verbose_level);
	R->main_loop(verbose_level);

	FREE_OBJECT(R);

	if (f_v) {
		cout << "interface_combinatorics::do_tdo_refinement done" << endl;
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

	cout << "before DC->init" << endl;
	DC->init(Descr, verbose_level);
	cout << "after DC->init" << endl;



	action *A;
	//int *Elt1;
	int *Elt2;

	A = DC->A;

	Elt2 = NEW_int(A->elt_size_in_int);




	cout << "We have created the following design:" << endl;
	cout << "$$" << endl;
	L.lint_set_print_tex(cout, DC->set, DC->sz);
	cout << endl;
	cout << "$$" << endl;

	if (DC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		DC->Sg->print_generators_tex(cout);
	}


	{
		int nb_pts = DC->P->N_points;
		int nb_blocks = DC->sz;
		int *Incma;
		int h, i, j, a;
		int pts_per_element = DC->k;


		Incma = NEW_int(nb_pts * nb_blocks);
		int_vec_zero(Incma, nb_pts * nb_blocks);

		for (j = 0; j < nb_blocks; j++) {
			//cout << "j=" << j << " / " << set_size
			//<< " the_set[j]=" << the_set[j] << endl;
			//Grass->unrank_int(the_set[j], 0/*verbose_level - 4*/);

			a = DC->set[j];
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

		cout << "Computing incidence matrix done" << endl;



		incidence_structure *Inc;
		partitionstack *Stack;


		cout << "Opening incidence data structure:" << endl;

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



}}
