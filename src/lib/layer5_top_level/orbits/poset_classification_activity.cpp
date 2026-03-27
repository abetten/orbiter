/*
 * poset_classification_activity.cpp
 *
 *  Created on: Feb 19, 2023
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


poset_classification_activity::poset_classification_activity()
{
	Record_birth();
	Descr = NULL;
	PC = NULL;
	actual_size = 0;
}

poset_classification_activity::~poset_classification_activity()
{
	Record_death();
}

void poset_classification_activity::init(
		poset_classification_activity_description *Descr,
		poset_classification::poset_classification *PC,
		int actual_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_activity::init" << endl;
	}
	poset_classification_activity::Descr = Descr;
	poset_classification_activity::PC = PC;
	poset_classification_activity::actual_size = actual_size;
	if (f_v) {
		cout << "poset_classification_activity::init done" << endl;
	}

}


void poset_classification_activity::perform_work(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_activity::perform_work" << endl;
	}

	if (Descr->f_report) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_report" << endl;
		}

		poset_classification::pc_latex_interface Pc_latex_interface;

		Pc_latex_interface.init(
				PC,
				PC->get_depth(),
				Descr->report_options /*poset_classification_report_options *Opt*/,
				verbose_level);

		Pc_latex_interface.report(verbose_level);

		//PC->report(Descr->report_options, verbose_level);

	}

	if (Descr->f_export_level_to_cpp) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_export_level_to_cpp" << endl;
		}

		poset_classification::poset_classification_global PCG;

		PCG.init(
				PC,
				verbose_level);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before PCG.generate_source_code" << endl;
		}
		PCG.generate_source_code(
				Descr->export_level_to_cpp_level, verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after PCG.generate_source_code" << endl;
		}

	}

	if (Descr->f_export_history_to_cpp) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_export_history_to_cpp" << endl;
		}

		poset_classification::poset_classification_global PCG;

		PCG.init(
				PC,
				verbose_level);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before PCG.generate_history" << endl;
		}
		PCG.generate_history(
				Descr->export_history_to_cpp_level, verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after PCG.generate_history" << endl;
		}

	}


	if (f_v) {
		cout << "poset_classification_activity::perform_work "
				"problem_label_with_path="
				<< PC->get_problem_label_with_path()
				<< " verbose_level=" << verbose_level << endl;
	}

	if (Descr->f_write_tree) {


		other::graphics::draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->write_tree_draw_options);


		poset_classification::pc_tree_interface Pc_tree_interface;

		Pc_tree_interface.init(
				PC,
				PC->get_depth(),
				Draw_options,
				verbose_level - 1);

		//PC->get_Poo()->print_tree();

		Pc_tree_interface.write_treefile(
				PC->get_depth(),
				verbose_level - 1);

		//return 0;
	}
	if (Descr->f_table_of_nodes) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_table_of_nodes" << endl;
		}
		PC->get_Poo()->make_table_of_nodes(verbose_level);
	}

	if (Descr->f_list_all) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_list_all" << endl;
		}

		int d;

		for (d = 0; d <= PC->get_depth(); d++) {
			cout << "There are " << PC->get_Poo()->nb_orbits_at_level(d)
					<< " orbits on subsets of size " << d << ":" << endl;

#if 0
			if (d < Descr->orbits_on_subsets_size) {
				//continue;
			}
#endif

			poset_classification::poset_classification_global PCG;

			PCG.init(
					PC,
					verbose_level);

			PCG.list_all_orbits_at_level(
					d,
					false /* f_has_print_function */,
					NULL /* void (*print_function)(std::ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					Descr->f_show_orbit_decomposition /* f_show_orbit_decomposition */,
					Descr->f_show_stab /* f_show_stab */,
					Descr->f_save_stab /* f_save_stab */,
					Descr->f_show_whole_orbits /* f_show_whole_orbit*/);
		}
	}

	if (Descr->f_list) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work f_list" << endl;
		}
#if 1
		//int f_show_orbit_decomposition = true;
		//int f_show_stab = true;
		//int f_save_stab = true;
		//int f_show_whole_orbit = false;

		poset_classification::poset_classification_global PCG;

		PCG.init(
				PC,
				verbose_level);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before list_all_orbits_at_level" << endl;
		}
		PCG.list_all_orbits_at_level(
				actual_size,
			false,
			NULL,
			this,
			Descr->f_show_orbit_decomposition,
			Descr->f_show_stab,
			Descr->f_save_stab,
			Descr->f_show_whole_orbits);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after list_all_orbits_at_level" << endl;
		}

#if 0
		int d;
		for (d = 0; d < 3; d++) {
			gen->print_schreier_vectors_at_depth(d, verbose_level);
		}
#endif
#endif
	}

	if (Descr->f_level_summary_csv) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing level spreadsheet" << endl;
		}

		poset_classification::pc_convert_data_structure Pc_convert_data_structure;

		Pc_convert_data_structure.init(
				PC,
				0 /* verbose_level*/);

		{
			other::data_structures::spreadsheet *Sp;

			Pc_convert_data_structure.make_spreadsheet_of_level_info(
					Sp, actual_size, verbose_level);


			string fname_csv;

			fname_csv = PC->get_problem_label_with_path()
					+ "_levels_"
					+ std::to_string(actual_size)
					+ ".csv";

			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing level spreadsheet done" << endl;
		}
	}


	if (Descr->f_orbit_reps_csv) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"-orbit_reps_csv" << endl;
		}

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before PC->get_Poo()->save_representatives_up_to_a_given_level_to_csv" << endl;
		}
		PC->get_Poo()->save_representatives_up_to_a_given_level_to_csv(
				actual_size, verbose_level - 2);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after PC->get_Poo()->save_representatives_up_to_a_given_level_to_csv" << endl;
		}

#if 0
		{
			//other::data_structures::spreadsheet *Sp;
			std::string *Headings;
			std::string *Table;
			int nb_rows, nb_cols;

			PC->get_Poo()->make_table_of_orbit_reps(
					Headings,
					Table,
					nb_rows, nb_cols,
					0 /* level_min */, actual_size /* level_max */,
					0 /*verbose_level*/);

			other::orbiter_kernel_system::file_io Fio;
			string fname_csv;

			fname_csv = PC->get_problem_label_with_path()
					+ "_orbits_at_level_"
					+ std::to_string(actual_size)
					+ ".csv";

			Fio.Csv_file_support->write_table_of_strings_with_col_headings(
					fname_csv,
					nb_rows, nb_cols, Table,
					Headings,
					verbose_level);

			delete [] Table;
			delete [] Headings;

#if 0
			PC->make_spreadsheet_of_orbit_reps(
					Sp, actual_size);
#endif
			//Sp->save(fname_csv, verbose_level);
			//FREE_OBJECT(Sp);
		}
#endif
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing orbit spreadsheet done" << endl;
		}
	}


	if (Descr->f_draw_poset) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before draw_poset" << endl;
		}
#if 0
		if (!Descr->f_draw_options) {
			cout << "poset_classification_activity::perform_work "
					"Descr->f_draw_poset && !Control->f_draw_options" << endl;
			exit(1);
		}
#endif

		other::graphics::draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->draw_poset_draw_options);

		PC->draw_poset(
			PC->get_problem_label_with_path(), actual_size,
			0 /* data1 */,
			Draw_options,
			verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after draw_poset" << endl;
		}
	}

	if (Descr->f_draw_full_poset) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before draw_full_poset" << endl;
		}

		other::graphics::draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->draw_full_poset_draw_options);

		PC->draw_poset_full(
				PC->get_problem_label_with_path(), actual_size,
				0 /* data1 */,
				Draw_options,
				//1 /* x_stretch */,
				verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after draw_full_poset" << endl;
		}
	}

	if (Descr->f_plesken_ring) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_plesken_ring" << endl;
		}

		poset_classification::pc_combinatorics Pc_combinatorics;

		Pc_combinatorics.init(
				PC,
				0 /* verbose_level*/);

#if 0
		int *Pup;
		int *Pdown;
		int N;


		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before Pc_combinatorics.Plesken_matrices" << endl;
		}
		Pc_combinatorics.Plesken_matrices(
				Pup,
				Pdown,
				N,
				verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after Pc_combinatorics.Plesken_matrices" << endl;
		}
#endif

		layer5_applications::apps_combinatorics::plesken_ring *Plesken_ring;

		Plesken_ring = NEW_OBJECT(layer5_applications::apps_combinatorics::plesken_ring);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before Plesken_ring->init" << endl;
		}
		Plesken_ring->init(
				PC,
				verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after Plesken_ring->init" << endl;
		}


		other::orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);
		Symb->init_plesken_ring(
				PC->get_problem_label(), Plesken_ring, verbose_level);
		if (f_v) {
			cout << "symbol_definition::definition_of_orthogonal_space "
					"before add_symbol_table_entry" << endl;
		}
		//Sym->Orbiter_top_level_session->add_symbol_table_entry(
		//		define_label, Symb, verbose_level);

		nb_output = 1;
		Output = Symb;


		//FREE_int(Pup);
		//FREE_int(Pdown);

	}
	if (Descr->f_make_relations_with_flag_orbits) {


		poset_classification::pc_convert_data_structure Pc_convert_data_structure;

		Pc_convert_data_structure.init(
				PC,
				0 /* verbose_level*/);



		string fname_prefix;


		fname_prefix = PC->get_problem_label_with_path() + "_flag_orbits";

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before make_flag_orbits_on_relations" << endl;
		}
		Pc_convert_data_structure.make_flag_orbits_on_relations(
				PC->get_depth(), fname_prefix,
				verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after make_flag_orbits_on_relations" << endl;
		}
	}
	if (Descr->f_print_data_structure) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_print_data_structure" << endl;
		}
		poset_classification::pc_latex_interface Pc_latex_interface;

		Pc_latex_interface.init(
				PC,
				PC->get_depth(),
				NULL /*poset_classification_report_options *Opt*/,
				verbose_level);

		Pc_latex_interface.print_data_structure_tex(
				actual_size, verbose_level);
	}


	if (Descr->f_test_multi_edge_in_decomposition_matrix) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_test_multi_edge_in_decomposition_matrix" << endl;
		}

		poset_classification::pc_combinatorics Pc_combinatorics;

		Pc_combinatorics.init(
				PC,
				verbose_level);


		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before Pc_combinatorics.test_for_multi_edge_in_classification_graph" << endl;
		}
		Pc_combinatorics.test_for_multi_edge_in_classification_graph(
				PC->get_depth(), verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after Pc_combinatorics.test_for_multi_edge_in_classification_graph" << endl;
		}
	}


	if (Descr->recognize.size()) {
		int h;

		layer4_classification::poset_classification::poset_classification_global PCG;

		PCG.init(
				PC,
				verbose_level);

		for (h = 0; h < Descr->recognize.size(); h++) {

			PCG.recognize(
					Descr->recognize[h],
					h,
					Descr->recognize.size(),
					verbose_level);
		}
	}

	if (Descr->f_pair_relations_within_orbit) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_pair_relations_within_orbit orbit_idx = "
					<< Descr->pair_relations_within_orbit_idx << endl;
		}

		poset_classification::pc_combinatorics Pc_combinatorics;

		Pc_combinatorics.init(
				PC,
				verbose_level);


		int level, po;
		int *M;
		int ol;

		PC->get_Poo()->node_to_lvl_po(
				Descr->pair_relations_within_orbit_idx, level, po);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_pair_relations_within_orbit level = "
					<< level << " po = " << po << endl;
		}

		int f_do_element_idx = true;
		long int *Element_idx = NULL;

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before Pc_combinatorics.pairwise_join_and_identify" << endl;
		}

		Pc_combinatorics.pairwise_join_and_identify(
				level, po, M, ol,
				f_do_element_idx, Element_idx,
				verbose_level);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after Pc_combinatorics.pairwise_join_and_identify" << endl;
		}


		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"color matrix of size " << ol << endl;
			Int_matrix_print(M, ol, ol);
		}

		other::orbiter_kernel_system::file_io Fio;

		std::string fname;

		fname = PC->get_problem_label() + "_pairs_on_orbit_"
				+ std::to_string(Descr->pair_relations_within_orbit_idx) + ".csv";
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, M, ol, ol);
		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}


		if (f_do_element_idx) {

			fname = PC->get_problem_label() + "_pairs_on_orbit_"
					+ std::to_string(Descr->pair_relations_within_orbit_idx) + "_element_idx.csv";
			Fio.Csv_file_support->lint_matrix_write_csv(
					fname, Element_idx, ol, ol);
			if (f_v) {
				cout << "Written file " << fname
						<< " of size " << Fio.file_size(fname) << endl;
			}

		}

		if (M) {
			FREE_int(M);
		}
		if (Element_idx) {
			FREE_lint(Element_idx);
		}


	}
	else if (Descr->f_export_orbits_long) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_export_orbits_long" << endl;
		}

		other::data_structures::set_of_sets *All_orbits;

		{
			int *Nb_orbits;
			int *Orbit_first;
			int nb_orbits_total;
			int nb_sets_total;

			std::string *Table;
			std::string *Col_headings;
			int nb_rows;
			int nb_cols;

			if (f_v) {
				cout << "poset_classification_activity::perform_work "
						"before get_all_orbits_expanded_table" << endl;
			}
			PC->get_Poo()->get_all_orbits_expanded_table(
					All_orbits,
					Nb_orbits,
					Orbit_first,
					nb_orbits_total,
					nb_sets_total,
					Table,
					Col_headings,
					nb_rows, nb_cols,
					verbose_level);

			if (f_v) {
				cout << "poset_classification_activity::perform_work "
						"after get_all_orbits_expanded_table" << endl;
			}

			other::orbiter_kernel_system::file_io Fio;

			std::string fname;

			fname = PC->get_problem_label() + "_all_orbits_expanded.csv";

			Fio.Csv_file_support->write_table_of_strings_with_col_headings(
					fname,
					nb_rows, nb_cols, Table,
					Col_headings,
					verbose_level);

			delete [] Table;
			delete [] Col_headings;


			if (f_v) {
				cout << "vector_ge::save_csv Written file " << fname
						<< " of size " << Fio.file_size(fname) << endl;
			}



			FREE_int(Nb_orbits);
			FREE_int(Orbit_first);
		}

		std::string fname;

		fname = PC->get_problem_label() + "_all_orbits_long.csv";

		All_orbits->save_csv(
				fname,
				verbose_level);

		other::orbiter_kernel_system::file_io Fio;



		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}


		if (f_v) {
			cout << "poset_classification_activity::perform_work All_orbits=" << endl;
			All_orbits->print_table();
		}
		FREE_OBJECT(All_orbits);

	}

	if (f_v) {
		cout << "poset_classification_activity::perform_work done" << endl;
	}
}







}}}



