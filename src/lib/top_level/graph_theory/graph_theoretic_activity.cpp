/*
 * graph_theoretic_activity.cpp
 *
 *  Created on: Mar 23, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


graph_theoretic_activity::graph_theoretic_activity()
{
	Descr = NULL;
	CG = NULL;
}

graph_theoretic_activity::~graph_theoretic_activity()
{
}


void graph_theoretic_activity::init(graph_theoretic_activity_description *Descr,
		colored_graph *CG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::init" << endl;
	}

	graph_theoretic_activity::Descr = Descr;
	graph_theoretic_activity::CG = CG;

	if (f_v) {
		cout << "graph_theoretic_activity::init, label = " << graph_theoretic_activity::CG->label << endl;
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
	string_tools ST;

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
		colored_graph *CG2;
		string fname2;

		fname2.assign(CG->label);
		//strcpy(fname2, fname_graph);
		ST.replace_extension_with(fname2, "_sorted.bin");
		CG2 = CG->sort_by_color_classes(verbose_level);
		CG2->save(fname2, verbose_level);
		FREE_OBJECT(CG2);

	}

	if (Descr->f_split) {
		cout << "splitting by file " << Descr->split_by_file << endl;
		file_io Fio;
		long int *Split;
		int m, n;
		int a, c;
		string_tools ST;


		Fio.lint_matrix_read_csv(Descr->split_by_file, Split, m, n, verbose_level - 2);
		cout << "We found " << m << " cases for splitting" << endl;
		for (c = 0; c < m; c++) {

			cout << "splitting case " << c << " / " << m << ":" << endl;
			a = Split[2 * c + 0];

			colored_graph *Subgraph;
			fancy_set *color_subset;
			fancy_set *vertex_subset;

			Subgraph = CG->compute_neighborhood_subgraph(a,
					vertex_subset, color_subset,
					verbose_level);

			string fname_out;

			fname_out.assign(Descr->split_input_fname);
			ST.chop_off_extension(fname_out);

			char str[1000];
			sprintf(str, "_case_%03d.bin", c);
			fname_out.append(str);


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}

	else if (Descr->f_save) {

		file_io Fio;
		string fname;

		fname.assign(CG->label);
		fname.append(".colored_graph");

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

		file_io Fio;
		string fname;

		fname.assign(CG->label);
		fname.append(".colored_graph");


		nauty_interface_with_group Nauty;
		action *Aut;

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity before Nauty.create_automorphism_group_of_colored_graph_object" << endl;
		}
		Aut = Nauty.create_automorphism_group_of_colored_graph_object(CG, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity after Nauty.create_automorphism_group_of_colored_graph_object" << endl;
		}

		string fname_report;

		fname_report.assign(CG->label);
		fname_report.append("_report.tex");


		{
			char title[1000];
			char author[1000];

			snprintf(title, 1000, "Automorphism group of %s", CG->label_tex.c_str());
			//strcpy(author, "");
			author[0] = 0;


			{
				ofstream ost(fname_report);
				latex_interface L;

				L.head(ost,
						FALSE /* f_book*/,
						TRUE /* f_title */,
						title, author,
						FALSE /* f_toc */,
						FALSE /* f_landscape */,
						TRUE /* f_12pt */,
						TRUE /* f_enlarged_page */,
						TRUE /* f_pagenumbers */,
						NULL /* extra_praeamble */);


				longinteger_object go;

				Aut->Strong_gens->group_order(go);

				ost << "\\noindent The automorphism group of $" << CG->label_tex << "$ "
						"has order " << go << " and is generated by:\\\\" << endl;
				Aut->Strong_gens->print_generators_tex(ost);


				if (f_v) {
					cout << "graph_theoretic_activity::perform_activity after report" << endl;
				}


				L.foot(ost);

			}
			file_io Fio;

			cout << "written file " << fname_report << " of size "
					<< Fio.file_size(fname_report) << endl;
		}

		string fname_group;

		fname_group.assign(CG->label);
		fname_group.append("_group.makefile");

		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity before Aut->export_to_orbiter label = " << CG->label << endl;
		}
		Aut->degree--;
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity before Aut->export_to_orbiter degree = " << Aut->degree << endl;
		}
		Aut->export_to_orbiter(fname_group, CG->label, Aut->Strong_gens, verbose_level);
		if (f_v) {
			cout << "graph_theoretic_activity::perform_activity after Aut->export_to_orbiter" << endl;
		}
		//file_io Fio;

		cout << "written file " << fname_group << " of size "
				<< Fio.file_size(fname_group) << endl;

	}



	if (f_v) {
		cout << "graph_theoretic_activity::perform_activity done" << endl;
	}
}

}}



