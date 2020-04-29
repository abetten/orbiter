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
	Descr = NULL;
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
			Descr = NEW_OBJECT(combinatorial_object_description);
			i += Descr->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;
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

			i += Job->read_arguments(argc - i,
				argv + i + 1, verbose_level) + 1;
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
	COC->init(Descr, verbose_level);
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
		cout << "please use option -n <n> to specify the projective dimension  within the job description" << endl;
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
	KM.orbits(argc, argv, S, verbose_level);

	delete S;
	}

	if (f_v) {
		cout << "interface_combinatorics::do_Kramer_Mesner done" << endl;
	}
}


}}
