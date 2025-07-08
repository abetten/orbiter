/*
 * poset_classification_activity.cpp
 *
 *  Created on: Feb 19, 2023
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
		poset_classification *PC,
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

		PC->report(Descr->report_options, verbose_level);

	}

	if (Descr->f_export_level_to_cpp) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_export_level_to_cpp" << endl;
		}

		poset_classification_global PCG;

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

		poset_classification_global PCG;

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


		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->write_tree_draw_options);


		PC->get_Poo()->print_tree();
		PC->write_treefile(
				PC->get_problem_label_with_path(),
				PC->get_depth(),
				Draw_options,
				verbose_level - 1);

		//return 0;
	}
	if (Descr->f_table_of_nodes) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_table_of_nodes" << endl;
		}
		PC->get_Poo()->make_tabe_of_nodes(verbose_level);
	}

	if (Descr->f_list_all) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_list_all" << endl;
		}

		int d;

		for (d = 0; d <= PC->get_depth(); d++) {
			cout << "There are " << PC->nb_orbits_at_level(d)
					<< " orbits on subsets of size " << d << ":" << endl;

#if 0
			if (d < Descr->orbits_on_subsets_size) {
				//continue;
			}
#endif

			poset_classification_global PCG;

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

		poset_classification_global PCG;

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
		{
			other::data_structures::spreadsheet *Sp;
			PC->make_spreadsheet_of_level_info(
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

		other::graphics::layered_graph_draw_options *Draw_options;

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

		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->draw_full_poset_draw_options);

		PC->draw_poset_full(
				PC->get_problem_label_with_path(), actual_size,
				0 /* data1 */,
				Draw_options,
				1 /* x_stretch */,
				verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after draw_full_poset" << endl;
		}
	}
	if (Descr->f_make_relations_with_flag_orbits) {
			string fname_prefix;


			fname_prefix = PC->get_problem_label_with_path() + "_flag_orbits";

			if (f_v) {
				cout << "poset_classification_activity::perform_work "
						"before make_flag_orbits_on_relations" << endl;
			}
			PC->make_flag_orbits_on_relations(
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
		PC->print_data_structure_tex(
				actual_size, verbose_level);
	}


	if (Descr->f_test_multi_edge_in_decomposition_matrix) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_test_multi_edge_in_decomposition_matrix" << endl;
		}

		poset_classification_global PCG;

		PCG.init(
				PC,
				verbose_level);


		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before PCG.test_for_multi_edge_in_classification_graph" << endl;
		}
		PCG.test_for_multi_edge_in_classification_graph(
				PC->get_depth(), verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after PCG.test_for_multi_edge_in_classification_graph" << endl;
		}
	}


	if (Descr->recognize.size()) {
		int h;

		for (h = 0; h < Descr->recognize.size(); h++) {

			PC->recognize(
					Descr->recognize[h],
					h, Descr->recognize.size(),
					verbose_level);
		}
	}

	if (f_v) {
		cout << "poset_classification_activity::perform_work done" << endl;
	}
}





}}}


