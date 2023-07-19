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
	Descr = NULL;
	CG = NULL;
}

graph_theoretic_activity::~graph_theoretic_activity()
{
}


void graph_theoretic_activity::init(graph_theoretic_activity_description *Descr,
		graph_theory::colored_graph *CG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::init" << endl;
	}

	graph_theoretic_activity::Descr = Descr;
	graph_theoretic_activity::CG = CG;

	if (f_v) {
		cout << "graph_theoretic_activity::init, "
				"label = " << graph_theoretic_activity::CG->label << endl;
	}


	if (f_v) {
		cout << "graph_theoretic_activity::init done" << endl;
	}
}

void graph_theoretic_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::perform_activity, CG->label=" << CG->label << endl;
	}
	data_structures::string_tools ST;

	if (Descr->f_find_cliques) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_find_cliques" << endl;
		}

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity before CG->all_cliques" << endl;
		}
		CG->all_cliques(
				Descr->Clique_finder_control,
				CG->label, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity after CG->all_cliques" << endl;
		}




		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity Gr->label=" << CG->label << " nb_sol = " << Descr->Clique_finder_control->nb_sol << endl;
		}

	}
	else if (Descr->f_find_subgraph) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"f_find_subgraph " << Descr->find_subgraph_label << endl;
		}

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity before CG->find_subgraph" << endl;
		}
		CG->find_subgraph(
				Descr->find_subgraph_label,
				verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity after CG->find_subgraph" << endl;
		}




		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity Gr->label=" << CG->label << endl;
		}

	}
	else if (Descr->f_export_magma) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_export_magma" << endl;
		}

		string fname_magma;
		string fname_text;

		fname_magma.assign(CG->label);

		fname_text.assign(CG->label);


		ST.replace_extension_with(fname_magma, ".magma");
		ST.replace_extension_with(fname_text, ".txt");

		if (f_v) {
			cout << "exporting to magma as " << fname_magma << endl;
		}


		CG->export_to_magma(fname_magma, verbose_level);

		CG->export_to_text(fname_text, verbose_level);

		if (f_v) {
			cout << "export_magma done" << endl;
		}
	}
	else if (Descr->f_export_maple) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_export_maple" << endl;
		}


		string fname_maple;

		fname_maple.assign(CG->label);


		ST.replace_extension_with(fname_maple, ".maple");

		if (f_v) {
			cout << "exporting to maple as " << fname_maple << endl;
		}


		CG->export_to_maple(fname_maple, verbose_level);

		if (f_v) {
			cout << "export_maple done" << endl;
		}

	}
	else if (Descr->f_export_csv) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_export_csv" << endl;
		}


		string fname_csv;

		fname_csv.assign(CG->label);


		ST.replace_extension_with(fname_csv, ".csv");

		cout << "exporting to csv as " << fname_csv << endl;


		CG->export_to_csv(fname_csv, verbose_level);

	}

	else if (Descr->f_export_graphviz) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_export_graphviz" << endl;
		}


		string fname_csv;

		fname_csv.assign(CG->label);


		ST.replace_extension_with(fname_csv, ".gv");

		cout << "exporting to gv as " << fname_csv << endl;


		CG->export_to_graphviz(fname_csv, verbose_level);

	}

	else if (Descr->f_print) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_print" << endl;
		}
		CG->print();

	}
	else if (Descr->f_sort_by_colors) {
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_sort_by_colors" << endl;
		}
		graph_theory::colored_graph *CG2;
		string fname2;

		fname2.assign(CG->label);
		//strcpy(fname2, fname_graph);
		ST.replace_extension_with(fname2, "_sorted.bin");
		CG2 = CG->sort_by_color_classes(verbose_level);
		CG2->save(fname2, verbose_level);
		FREE_OBJECT(CG2);

	}

	else if (Descr->f_split) {
		cout << "splitting by file " << Descr->split_by_file << endl;
		orbiter_kernel_system::file_io Fio;
		long int *Split;
		int m, n;
		int a, c;
		data_structures::string_tools ST;


		Fio.Csv_file_support->lint_matrix_read_csv(
				Descr->split_by_file, Split, m, n, verbose_level - 2);
		cout << "We found " << m << " cases for splitting" << endl;
		for (c = 0; c < m; c++) {

			cout << "splitting case " << c << " / " << m << ":" << endl;
			a = Split[2 * c + 0];

			graph_theory::colored_graph *Subgraph;
			data_structures::fancy_set *color_subset;
			data_structures::fancy_set *vertex_subset;

			Subgraph = CG->compute_neighborhood_subgraph(a,
					vertex_subset, color_subset,
					verbose_level);

			string fname_out;

			fname_out = Descr->split_input_fname;
			ST.chop_off_extension(fname_out);

			fname_out += "_case_" + std::to_string(c) + ".bin";


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}

	else if (Descr->f_split_by_starters) {
		cout << "splitting by file " << Descr->split_by_starters_fname_reps
				<< " column " << Descr->split_by_starters_col_label << endl;
		orbiter_kernel_system::file_io Fio;
		data_structures::set_of_sets *Reps;
		//string_tools ST;
		int c;


		Fio.read_column_and_parse(Descr->split_by_starters_fname_reps,
				Descr->split_by_starters_col_label,
				Reps, verbose_level);


		cout << "We found " << Reps->nb_sets << " cases for splitting" << endl;

		for (c = 0; c < Reps->nb_sets; c++) {

			cout << "splitting case " << c << " / " << Reps->nb_sets << ":" << endl;

			graph_theory::colored_graph *Subgraph;
			data_structures::fancy_set *color_subset;
			data_structures::fancy_set *vertex_subset;


			Subgraph = CG->compute_neighborhood_subgraph_based_on_subset(
					Reps->Sets[c], Reps->Set_size[c],
					vertex_subset, color_subset,
					verbose_level);

			string fname_out;

			fname_out.assign(CG->label);
			//ST.chop_off_extension(fname_out);

			fname_out += "_case_" + std::to_string(c) + ".bin";


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}
	else if (Descr->f_split_by_clique) {
		cout << "splitting by clique " << Descr->split_by_clique_label
				<< " clique " << Descr->split_by_clique_set << endl;

		long int *set;
		int sz;

		Lint_vec_scan(Descr->split_by_clique_set, set, sz);

		graph_theory::colored_graph *Subgraph;
		data_structures::fancy_set *color_subset;
		data_structures::fancy_set *vertex_subset;


		Subgraph = CG->compute_neighborhood_subgraph_based_on_subset(
				set, sz,
				vertex_subset, color_subset,
				verbose_level);



		string fname_base, fname_out, fname_subset;

		fname_base = CG->label;

		fname_base = "_" + Descr->split_by_clique_label;

		fname_out = fname_base + ".graph";


		Subgraph->save(fname_out, verbose_level - 2);

		fname_subset = fname_base + "_subset.txt";

		vertex_subset->save(fname_subset, verbose_level);

		FREE_OBJECT(Subgraph);

	}

	else if (Descr->f_save) {

		orbiter_kernel_system::file_io Fio;
		string fname;

		fname = CG->label + ".colored_graph";

		cout << "before save fname_graph=" << fname << endl;
		CG->save(fname, verbose_level);
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

	else if (Descr->f_automorphism_group) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_automorphism_group" << endl;
		}

		graph_theory_apps GTA;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before GTA.automorphism_group" << endl;
		}
		GTA.automorphism_group(CG, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after GTA.automorphism_group" << endl;
		}

	}
	else if (Descr->f_properties) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_properties" << endl;
		}

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before CG->properties" << endl;
		}
		CG->properties(verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after CG->properties" << endl;
		}
	}
	else if (Descr->f_eigenvalues) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_properties" << endl;
		}

		graph_theory::graph_theory_domain GT;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"before GT.eigenvalues" << endl;
		}
		GT.eigenvalues(CG, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity "
					"after GT.eigenvalues" << endl;
		}


	}
	else if (Descr->f_draw) {

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity f_draw" << endl;
		}

		string fname;

		fname = CG->label + "_draw.mp";

		CG->draw_on_circle(
				fname,
				orbiter_kernel_system::Orbiter->draw_options,
				verbose_level);
	}




	if (f_v) {
		cout << "graph_theoretic_activity::perform_activity done" << endl;
	}
}

}}}



