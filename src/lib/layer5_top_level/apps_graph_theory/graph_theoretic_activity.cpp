/*
 * graph_theoretic_activity.cpp
 *
 *  Created on: Mar 23, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


graph_theoretic_activity::graph_theoretic_activity()
{
	Record_birth();
	Descr = NULL;
	nb = 0;
	CG = NULL;
}

graph_theoretic_activity::~graph_theoretic_activity()
{
	Record_death();
}


void graph_theoretic_activity::init(
		graph_theoretic_activity_description *Descr,
		int nb,
		combinatorics::graph_theory::colored_graph **CG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::init" << endl;
	}

	graph_theoretic_activity::Descr = Descr;
	graph_theoretic_activity::nb = nb;
	graph_theoretic_activity::CG = CG;

	int i;

	if (f_v) {
		cout << "graph_theoretic_activity::init, "
				"labels = ";
		for (i = 0; i < nb; i++) {
			cout << graph_theoretic_activity::CG[i]->label;
			if (i < nb - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}


	if (f_v) {
		cout << "graph_theoretic_activity::init done" << endl;
	}
}

void graph_theoretic_activity::feedback_headings(
		graph_theoretic_activity_description *Descr,
		std::string &headings,
		int &nb_cols,
		int verbose_level)
// CG is not available
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::feedback_headings" << endl;
	}

	graph_theoretic_activity::Descr = Descr;

	nb_cols = 0;

	if (Descr->f_find_cliques) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_find_cliques" << endl;
		}

		headings = "NB_CLQ_" + std::to_string(Descr->Clique_finder_control->target_size);
		nb_cols = 1;

	}
	else if (Descr->f_test_SRG_property) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_test_SRG_property" << endl;
		}

		headings = "is_SRG,lambda,mu";
		nb_cols = 3;


	}
	else if (Descr->f_test_Neumaier_property) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_test_Neumaier_property" << endl;
		}

		headings = "";
		nb_cols = 0;

	}
	else if (Descr->f_find_subgraph) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_find_subgraph " << Descr->find_subgraph_label << endl;
		}


	}
	else if (Descr->f_test_automorphism_property_of_group) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_test_automorphism_property_of_group "
					<< Descr->test_automorphism_property_of_group_label << endl;
		}


	}


	else if (Descr->f_export_magma) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_export_magma" << endl;
		}

	}
	else if (Descr->f_export_maple) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_export_maple" << endl;
		}


	}
	else if (Descr->f_export_csv) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_export_csv" << endl;
		}


	}

	else if (Descr->f_export_graphviz) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_export_graphviz" << endl;
		}

	}

	else if (Descr->f_print) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_print" << endl;
		}

	}
	else if (Descr->f_sort_by_colors) {
		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_sort_by_colors" << endl;
		}

	}

	else if (Descr->f_split) {
		if (f_v) {
			cout << "splitting by file " << Descr->split_by_file << endl;
		}

	}

	else if (Descr->f_split_by_starters) {
		if (f_v) {
			cout << "splitting by file " << Descr->split_by_starters_fname_reps
					<< " column " << Descr->split_by_starters_col_label << endl;
		}
	}


	else if (Descr->f_combine_by_starters) {
		if (f_v) {
			cout << "combining by file " << Descr->combine_by_starters_fname_reps
					<< " column " << Descr->combine_by_starters_col_label << endl;
		}

	}

	else if (Descr->f_split_by_clique) {
		if (f_v) {
			cout << "splitting by clique " << Descr->split_by_clique_label
					<< " clique " << Descr->split_by_clique_set << endl;
		}


	}

	else if (Descr->f_save) {

	}

	else if (Descr->f_automorphism_group) {

		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_automorphism_group" << endl;
		}


	}
	else if (Descr->f_properties) {

		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_properties" << endl;
		}

	}
	else if (Descr->f_eigenvalues) {

		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_properties" << endl;
		}


	}
	else if (Descr->f_draw) {

		if (f_v) {
			cout << "graph_theoretic_activity::feedback_headings "
					"f_draw" << endl;
		}
	}

}

void graph_theoretic_activity::get_label(
		graph_theoretic_activity_description *Descr,
		std::string &description_txt,
		int verbose_level)
// CG is not available
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::get_label" << endl;
	}

	graph_theoretic_activity::Descr = Descr;


	if (Descr->f_find_cliques) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_find_cliques" << endl;
		}

		description_txt = "cliques_sz" + std::to_string(Descr->Clique_finder_control->target_size);

	}
	else if (Descr->f_test_SRG_property) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_test_SRG_property" << endl;
		}

		description_txt = "test_SRG";


	}
	else if (Descr->f_test_Neumaier_property) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_test_Neumaier_property" << endl;
		}

		description_txt = "test_Neumaier";


	}
	else if (Descr->f_find_subgraph) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_find_subgraph " << Descr->find_subgraph_label << endl;
		}



	}
	else if (Descr->f_test_automorphism_property_of_group) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_test_automorphism_property_of_group "
					<< Descr->test_automorphism_property_of_group_label << endl;
		}


	}
	else if (Descr->f_export_magma) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_export_magma" << endl;
		}

	}
	else if (Descr->f_export_maple) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_export_maple" << endl;
		}


	}
	else if (Descr->f_export_csv) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_export_csv" << endl;
		}


	}

	else if (Descr->f_export_graphviz) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_export_graphviz" << endl;
		}

	}

	else if (Descr->f_print) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_print" << endl;
		}

	}
	else if (Descr->f_sort_by_colors) {
		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_sort_by_colors" << endl;
		}

	}

	else if (Descr->f_split) {
		if (f_v) {
			cout << "splitting by file " << Descr->split_by_file << endl;
		}

	}

	else if (Descr->f_split_by_starters) {
		if (f_v) {
			cout << "splitting by file " << Descr->split_by_starters_fname_reps
					<< " column " << Descr->split_by_starters_col_label << endl;
		}
	}


	else if (Descr->f_combine_by_starters) {
		if (f_v) {
			cout << "combining by file " << Descr->combine_by_starters_fname_reps
					<< " column " << Descr->combine_by_starters_col_label << endl;
		}

	}

	else if (Descr->f_split_by_clique) {
		if (f_v) {
			cout << "splitting by clique " << Descr->split_by_clique_label
					<< " clique " << Descr->split_by_clique_set << endl;
		}


	}

	else if (Descr->f_save) {

	}

	else if (Descr->f_automorphism_group) {

		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_automorphism_group" << endl;
		}


	}
	else if (Descr->f_properties) {

		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_properties" << endl;
		}

	}
	else if (Descr->f_eigenvalues) {

		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_properties" << endl;
		}


	}
	else if (Descr->f_draw) {

		if (f_v) {
			cout << "graph_theoretic_activity::get_label "
					"f_draw" << endl;
		}
	}

}


void graph_theoretic_activity::perform_activity(
		std::vector<std::string> &feedback,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::perform_activity" << endl;
	}
	other::data_structures::string_tools ST;

	if (Descr->f_find_cliques) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_find_cliques" << endl;
		}

		combinatorics::graph_theory::clique_finder *CF;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before CG->Colored_graph_cliques->all_cliques" << endl;
		}
		CG[0]->Colored_graph_cliques->all_cliques(
				Descr->Clique_finder_control,
				CG[0]->label,
				feedback,
				CF,
				verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after CG->Colored_graph_cliques->all_cliques" << endl;
		}

		if (false) {
			cout << "graph_theoretic_activity::perform_activity "
					"CG[0]->label=" << CG[0]->label
					<< " nb_sol = " << Descr->Clique_finder_control->nb_sol << endl;
		}

#if 0
		string fname_solution;

		fname_solution = CG[0]->label + "_solutions.csv";

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before CF->write_solutions" << endl;
		}
		CF->write_solutions(
				fname_solution,
					verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after CF->write_solutions" << endl;
		}


		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"CG[0]->label=" << CG[0]->label
					<< " nb_sol = " << Descr->Clique_finder_control->nb_sol << endl;
		}
#endif

		//FREE_OBJECT(CF);

	}
	else if (Descr->f_test_SRG_property) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_test_SRG_property" << endl;
		}

		int ret;
		int lambda, mu;
		string s1, s2, s3;

		ret = CG[0]->test_SRG_property(
				lambda,
				mu,
				verbose_level);
		s1 = "\"" + std::to_string(ret) + "\"";
		s2 = "\"" + std::to_string(lambda) + "\"";
		s3 = "\"" + std::to_string(mu) + "\"";
		feedback.push_back(s1);
		feedback.push_back(s2);
		feedback.push_back(s3);


	}

	else if (Descr->f_test_Neumaier_property) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_test_Neumaier_property" << endl;
		}

		int regularity;
		int lambda_value;
		int nexus;
		int ret;

		ret = CG[0]->Colored_graph_cliques->test_Neumaier_property(
				regularity,
				lambda_value,
				Descr->test_Neumaier_property_clique_size,
				nexus,
				verbose_level);
		if (ret) {
			cout << "The graph is Neumaier, "
					"the parameters are NG(" << CG[0]->nb_points << ","
					<< regularity << ","
					<< lambda_value << ";"
					<< nexus << ","
					<< Descr->test_Neumaier_property_clique_size
					<< ")" << endl;
		}
		else {
			cout << "The graph is not Neumaier" << endl;
		}

	}
	else if (Descr->f_find_subgraph) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_find_subgraph " << Descr->find_subgraph_label << endl;
		}


		combinatorics::graph_theory::graph_theory_domain Graph_Domain;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before Graph_Domain.find_subgraph" << endl;
		}
		Graph_Domain.find_subgraph(
				nb, CG,
				Descr->find_subgraph_label,
				verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after Graph_Domain.find_subgraph" << endl;
		}



		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"CG[0]->label=" << CG[0]->label << endl;
		}

	}


	else if (Descr->f_test_automorphism_property_of_group) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_test_automorphism_property_of_group "
					<< Descr->test_automorphism_property_of_group_label << endl;
		}

		apps_graph_theory::graph_theory_apps Graph_theory_apps;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before Graph_theory_apps.test_automorphism_property_of_group" << endl;
		}
		Graph_theory_apps.test_automorphism_property_of_group(
				nb, CG,
				Descr->test_automorphism_property_of_group_label,
				verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after Graph_theory_apps.test_automorphism_property_of_group" << endl;
		}



	}
	else if (Descr->f_common_neighbors) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"-common_neighbors "
					<< Descr->common_neighbors_set << endl;
		}

		apps_graph_theory::graph_theory_apps Graph_theory_apps;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before Graph_theory_apps.common_neighbors" << endl;
		}
		Graph_theory_apps.common_neighbors(
				nb, CG,
				Descr->common_neighbors_set,
				verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after Graph_theory_apps.common_neighbors" << endl;
		}


	}


	else if (Descr->f_export_magma) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_export_magma" << endl;
		}

		string fname_magma;
		string fname_text;

		fname_magma.assign(CG[0]->label);

		fname_text.assign(CG[0]->label);


		ST.replace_extension_with(fname_magma, ".magma");
		ST.replace_extension_with(fname_text, ".txt");

		if (f_v) {
			cout << "exporting to magma as " << fname_magma << endl;
		}


		CG[0]->export_to_magma(fname_magma, verbose_level);

		CG[0]->export_to_text(fname_text, verbose_level);

		if (f_v) {
			cout << "export_magma done" << endl;
		}
	}
	else if (Descr->f_export_maple) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_export_maple" << endl;
		}


		string fname_maple;

		fname_maple.assign(CG[0]->label);


		ST.replace_extension_with(fname_maple, ".maple");

		if (f_v) {
			cout << "exporting to maple as " << fname_maple << endl;
		}


		CG[0]->export_to_maple(fname_maple, verbose_level);

		if (f_v) {
			cout << "export_maple done" << endl;
		}

	}
	else if (Descr->f_export_csv) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_export_csv" << endl;
		}


		string fname_csv;

		fname_csv.assign(CG[0]->label);


		ST.replace_extension_with(fname_csv, ".csv");

		cout << "exporting to csv as " << fname_csv << endl;


		CG[0]->export_to_csv(fname_csv, verbose_level);

	}

	else if (Descr->f_export_graphviz) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_export_graphviz" << endl;
		}


		string fname_csv;

		fname_csv.assign(CG[0]->label);


		ST.replace_extension_with(fname_csv, ".gv");

		if (f_v) {
			cout << "exporting to gv as " << fname_csv << endl;
		}


		CG[0]->export_to_graphviz(fname_csv, verbose_level);

	}

	else if (Descr->f_print) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_print" << endl;
		}
		CG[0]->print();

	}
	else if (Descr->f_sort_by_colors) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_sort_by_colors" << endl;
		}
		combinatorics::graph_theory::colored_graph *CG2;
		string fname2;

		fname2.assign(CG[0]->label);
		//strcpy(fname2, fname_graph);
		ST.replace_extension_with(fname2, "_sorted.bin");
		CG2 = CG[0]->sort_by_color_classes(verbose_level);
		CG2->save(fname2, verbose_level);
		FREE_OBJECT(CG2);

	}

	else if (Descr->f_split) {
		if (f_v) {
			cout << "splitting by file " << Descr->split_by_file << endl;
		}

		other::orbiter_kernel_system::file_io Fio;
		long int *Split;
		int m, n;
		int a, c;
		other::data_structures::string_tools ST;


		Fio.Csv_file_support->lint_matrix_read_csv(
				Descr->split_by_file, Split, m, n, verbose_level - 2);
		if (f_v) {
			cout << "We found " << m << " cases for splitting" << endl;
		}
		for (c = 0; c < m; c++) {

			if (f_v) {
				cout << "splitting case " << c << " / " << m << ":" << endl;
			}
			a = Split[2 * c + 0];

			combinatorics::graph_theory::colored_graph *Subgraph;
			other::data_structures::fancy_set *color_subset;
			other::data_structures::fancy_set *vertex_subset;

			Subgraph = CG[0]->compute_neighborhood_subgraph(a,
					vertex_subset, color_subset,
					verbose_level - 2);

			string fname_out;

			fname_out = Descr->split_input_fname;
			ST.chop_off_extension(fname_out);

			fname_out += "_case_" + std::to_string(c) + ".bin";


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}

	else if (Descr->f_split_by_starters) {
		if (f_v) {
			cout << "splitting by file " << Descr->split_by_starters_fname_reps
					<< " column " << Descr->split_by_starters_col_label << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		other::data_structures::set_of_sets *Reps;
		//string_tools ST;
		int c;


		Fio.Csv_file_support->read_column_and_parse(
				Descr->split_by_starters_fname_reps,
				Descr->split_by_starters_col_label,
				Reps, verbose_level);


		if (f_v) {
			cout << "We found " << Reps->nb_sets << " cases for splitting" << endl;
		}

		for (c = 0; c < Reps->nb_sets; c++) {

			if (f_v) {
				cout << "splitting case " << c << " / " << Reps->nb_sets << ":" << endl;
			}

			combinatorics::graph_theory::colored_graph *Subgraph;
			other::data_structures::fancy_set *color_subset;
			other::data_structures::fancy_set *vertex_subset;


			Subgraph = CG[0]->compute_neighborhood_subgraph_based_on_subset(
					Reps->Sets[c], Reps->Set_size[c],
					vertex_subset, color_subset,
					verbose_level - 2);

			string fname_out;

			fname_out = CG[0]->label + "_case_" + std::to_string(c) + ".bin";


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}


	else if (Descr->f_combine_by_starters) {
		if (f_v) {
			cout << "combining by file " << Descr->combine_by_starters_fname_reps
					<< " column " << Descr->combine_by_starters_col_label << endl;
		}
		other::orbiter_kernel_system::file_io Fio;
		other::data_structures::set_of_sets *Reps;
		//string_tools ST;
		int c;


		Fio.Csv_file_support->read_column_and_parse(
				Descr->combine_by_starters_fname_reps,
				Descr->combine_by_starters_col_label,
				Reps, verbose_level);


		if (f_v) {
			cout << "We found " << Reps->nb_sets << " cases for splitting" << endl;
		}

		if (Reps->nb_sets == 0) {
			cout << "Reps->nb_sets == 0" << endl;
			exit(1);
		}

		int starter_size;

		starter_size = Reps->Set_size[0];
		if (f_v) {
			cout << "starter_size = " << starter_size << endl;
		}

		int nb_sol_total, sz;

		nb_sol_total = 0;

		for (c = 0; c < Reps->nb_sets; c++) {

			if (false) {
				cout << "combining solutions from case " << c << " / " << Reps->nb_sets << ":" << endl;
				cout << "starter set = ";
				Lint_vec_print(cout, Reps->Sets[c], Reps->Set_size[c]);
				cout << endl;
			}

			combinatorics::graph_theory::colored_graph *Subgraph;
			other::data_structures::fancy_set *color_subset;
			other::data_structures::fancy_set *vertex_subset;


			Subgraph = CG[0]->compute_neighborhood_subgraph_based_on_subset(
					Reps->Sets[c], Reps->Set_size[c],
					vertex_subset, color_subset,
					verbose_level - 2);


			string fname_sol;


			fname_sol = CG[0]->label + "_case_" + std::to_string(c) + "_sol.csv";

			int *M;
			int nb_sol, width;
			int sz1;

			Fio.Csv_file_support->int_matrix_read_csv(
					fname_sol, M,
					nb_sol, width, verbose_level - 2);


			if (f_v) {
				cout << "combining solutions from case " << c << " / " << Reps->nb_sets << " with " << nb_sol << " solutions" << endl;
				cout << "starter set = ";
				Lint_vec_print(cout, Reps->Sets[c], Reps->Set_size[c]);
				cout << endl;
			}

			nb_sol_total += nb_sol;

			sz1 = Reps->Set_size[c] + width;

			if (c == 0) {
				sz = sz1;
			}
			else if (sz1 != sz) {
				cout << "sz1 != sz" << endl;
				exit(1);
			}

			FREE_int(M);
			FREE_OBJECT(Subgraph);
			FREE_OBJECT(color_subset);
			FREE_OBJECT(vertex_subset);


		}

		if (f_v) {
			cout << "combining solutions nb_sol_total = " << nb_sol_total << endl;
			cout << "combining solutions sz = " << sz << endl;
		}

		int *Sol;
		int cur;

		Sol = NEW_int(nb_sol_total * sz);

		cur = 0;

		for (c = 0; c < Reps->nb_sets; c++) {

			if (false) {
				cout << "combining solutions from case " << c << " / " << Reps->nb_sets << ":" << endl;
			}

			combinatorics::graph_theory::colored_graph *Subgraph;
			other::data_structures::fancy_set *color_subset;
			other::data_structures::fancy_set *vertex_subset;


			Subgraph = CG[0]->compute_neighborhood_subgraph_based_on_subset(
					Reps->Sets[c], Reps->Set_size[c],
					vertex_subset, color_subset,
					verbose_level - 2);


			string fname_sol;


			fname_sol = CG[0]->label + "_case_" + std::to_string(c) + "_sol.csv";

			int *M;
			int nb_sol, width;
			int a;

			Fio.Csv_file_support->int_matrix_read_csv(
					fname_sol, M,
					nb_sol, width, verbose_level - 2);


			if (f_v) {
				cout << "combining solutions from case " << c << " / " << Reps->nb_sets << " with " << nb_sol << " solutions" << endl;
				cout << "starter set = ";
				Lint_vec_print(cout, Reps->Sets[c], Reps->Set_size[c]);
				cout << endl;
			}



			int i, j;

			for (i = 0; i < nb_sol; i++, cur++) {
				for (j = 0; j < Reps->Set_size[c]; j++) {
					Sol[cur * sz + j] = Reps->Sets[c][j];
				}
				for (j = 0; j < width; j++) {
					a = M[i * width + j];
					Sol[cur * sz + Reps->Set_size[c] + j] = a;
				}

			}
			FREE_int(M);
			FREE_OBJECT(Subgraph);
			FREE_OBJECT(color_subset);
			FREE_OBJECT(vertex_subset);
		}
		if (cur != nb_sol_total) {
			cout << "cur != nb_sol_total" << endl;
			cout << "cur = " << cur << endl;
			cout << "nb_sol_total = " << nb_sol_total << endl;
			exit(1);
		}

		string fname_out;

		fname_out = CG[0]->label + "_split_" + std::to_string(starter_size) + "_sol.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname_out, Sol,
				nb_sol_total, sz);

		if (f_v) {
			cout << "Written file " << fname_out << " of size "
					<< Fio.file_size(fname_out) << endl;
		}

		FREE_int(Sol);

	}

	else if (Descr->f_split_by_clique) {
		if (f_v) {
			cout << "splitting by clique " << Descr->split_by_clique_label
					<< " clique " << Descr->split_by_clique_set << endl;
		}

		long int *set;
		int sz;

		Lint_vec_scan(Descr->split_by_clique_set, set, sz);

		combinatorics::graph_theory::colored_graph *Subgraph;
		other::data_structures::fancy_set *color_subset;
		other::data_structures::fancy_set *vertex_subset;


		Subgraph = CG[0]->compute_neighborhood_subgraph_based_on_subset(
				set, sz,
				vertex_subset, color_subset,
				verbose_level);



		string fname_base, fname_out, fname_subset;


		fname_base = CG[0]->label + "_" + Descr->split_by_clique_label;

		fname_out = fname_base + ".graph";


		Subgraph->save(fname_out, verbose_level - 2);

		fname_subset = fname_base + "_subset.txt";

		vertex_subset->save(fname_subset, verbose_level);

		FREE_OBJECT(Subgraph);

	}

	else if (Descr->f_save) {

		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = CG[0]->label + ".colored_graph";

		cout << "before save fname_graph=" << fname << endl;
		CG[0]->save(fname, verbose_level);
		cout << "after save" << endl;


#if 0
		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, COC->Pts, COC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
#endif

	}

	else if (Descr->f_automorphism_group_colored_graph) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_automorphism_group" << endl;
		}

		graph_theory_apps GTA;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before GTA.automorphism_group" << endl;
		}
		GTA.automorphism_group(CG[0], feedback, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after GTA.automorphism_group" << endl;
		}

	}

	else if (Descr->f_automorphism_group) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_automorphism_group" << endl;
		}

		graph_theory_apps GTA;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before GTA.automorphism_group_bw" << endl;
		}
		GTA.automorphism_group_bw(CG[0], feedback, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after GTA.automorphism_group_bw" << endl;
		}

	}

	else if (Descr->f_properties) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_properties" << endl;
		}

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before CG->properties" << endl;
		}
		CG[0]->properties(verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after CG->properties" << endl;
		}
	}
	else if (Descr->f_eigenvalues) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_properties" << endl;
		}

		combinatorics::graph_theory::graph_theory_domain GT;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before GT.eigenvalues" << endl;
		}
		GT.eigenvalues(CG[0], verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after GT.eigenvalues" << endl;
		}


	}
	else if (Descr->f_draw) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_draw" << endl;
		}

		string fname;

		fname = CG[0]->label + "_draw.mp";

		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->draw_options);

		CG[0]->draw_on_circle(
				fname,
				Draw_options,
				verbose_level);
	}


	if (f_v) {
		int i;

		cout << "feedback:";
		for (i = 0; i < feedback.size(); i++) {
			cout << feedback[i];
			if (i < feedback.size() - 1) {
				cout << ",";
			}
		}
		cout << endl;
	}


	if (f_v) {
		cout << "graph_theoretic_activity::perform_activity done" << endl;
	}
}

}}}



