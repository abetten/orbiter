/*
 * design_activity_description.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_activity_description::design_activity_description()
{

	f_load_table = false;
	//std::string load_table_label;
	//std::string load_table_group;

	//std::string load_table_H_label;
	//std::string load_table_H_group_order;
	//std::string load_table_H_gens;
	load_table_selected_orbit_length = 0;

	f_canonical_form = false;
	Canonical_form_Descr = NULL;


	f_extract_solutions_by_index_csv = false;
	f_extract_solutions_by_index_txt = false;
	//std::string extract_solutions_by_index_label;
	//std::string extract_solutions_by_index_group;
	//std::string extract_solutions_by_index_fname_solutions_in;
	//std::string extract_solutions_by_index_col_label;
	//std::string extract_solutions_by_index_fname_solutions_out;
	//std::string extract_solutions_by_index_prefix;

	f_export_inc = false;
	f_export_incidence_matrix = false;
	f_export_incidence_matrix_latex = false;
	std::string export_incidence_matrix_latex_draw_options;

	f_intersection_matrix = false;
	f_save = false;
	f_export_blocks = false;
	f_row_sums = false;
	f_tactical_decomposition = false;

	f_orbits_on_blocks = false;
	orbits_on_blocks_sz = -1;
	//std::string orbits_on_blocks_control;

	f_one_point_extension = false;
	one_point_extension_pair_orbit_idx = -1;
	//std::string one_point_extension_control;

}

design_activity_description::~design_activity_description()
{

}


int design_activity_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "design_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-load_table") == 0) {
			f_load_table = true;
			load_table_label.assign(argv[++i]);
			load_table_group.assign(argv[++i]);
			load_table_H_label.assign(argv[++i]);
			load_table_H_group_order.assign(argv[++i]);
			load_table_H_gens.assign(argv[++i]);
			load_table_selected_orbit_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-load_table " << load_table_label
						<< " " << load_table_group
						<< " " << load_table_H_label
						<< " " << load_table_H_group_order
						<< " " << load_table_H_gens
						<< " " << load_table_selected_orbit_length
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = true;
			if (f_v) {
				cout << "-canonical_form, reading extra arguments" << endl;
			}

			Canonical_form_Descr = NEW_OBJECT(canonical_form_classification::classification_of_objects_description);

			i += Canonical_form_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -f_canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_solutions_by_index_csv") == 0) {
			f_extract_solutions_by_index_csv = true;
			extract_solutions_by_index_label.assign(argv[++i]);
			extract_solutions_by_index_group.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_in.assign(argv[++i]);
			extract_solutions_by_index_col_label.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_out.assign(argv[++i]);
			extract_solutions_by_index_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_solutions_by_index_csv "
						<< extract_solutions_by_index_label << " "
						<< extract_solutions_by_index_group << " "
						<< extract_solutions_by_index_fname_solutions_in << " "
						<< extract_solutions_by_index_col_label << " "
						<< extract_solutions_by_index_fname_solutions_out << " "
						<< extract_solutions_by_index_prefix << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_solutions_by_index_txt") == 0) {
			f_extract_solutions_by_index_txt = true;
			extract_solutions_by_index_label.assign(argv[++i]);
			extract_solutions_by_index_group.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_in.assign(argv[++i]);
			extract_solutions_by_index_col_label.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_out.assign(argv[++i]);
			extract_solutions_by_index_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_solutions_by_index_txt "
						<< extract_solutions_by_index_label << " "
						<< extract_solutions_by_index_group << " "
						<< extract_solutions_by_index_fname_solutions_in << " "
						<< extract_solutions_by_index_col_label << " "
						<< extract_solutions_by_index_fname_solutions_out << " "
						<< extract_solutions_by_index_prefix << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_inc") == 0) {
			f_export_inc = true;
			if (f_v) {
				cout << "-export_inc " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_incidence_matrix") == 0) {
			f_export_incidence_matrix = true;
			if (f_v) {
				cout << "-export_incidence_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_incidence_matrix_latex") == 0) {
			f_export_incidence_matrix_latex = true;
			export_incidence_matrix_latex_draw_options.assign(argv[++i]);
			if (f_v) {
				cout << "-export_incidence_matrix_latex " << export_incidence_matrix_latex_draw_options << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_incidence_matrix_latex") == 0) {
			f_export_incidence_matrix = true;
			if (f_v) {
				cout << "-export_incidence_matrix_latex " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-intersection_matrix") == 0) {
			f_intersection_matrix = true;
			if (f_v) {
				cout << "-intersection_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save") == 0) {
			f_save = true;
			if (f_v) {
				cout << "-save " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-export_blocks") == 0) {
			f_export_blocks = true;
			if (f_v) {
				cout << "-export_blocks " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-row_sums") == 0) {
			f_row_sums = true;
			if (f_v) {
				cout << "-row_sums " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-tactical_decomposition") == 0) {
			f_tactical_decomposition = true;
			if (f_v) {
				cout << "-tactical_decomposition " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbits_on_blocks") == 0) {
			f_orbits_on_blocks = true;
			orbits_on_blocks_sz = ST.strtoi(argv[++i]);
			orbits_on_blocks_control.assign(argv[++i]);
			if (f_v) {
				cout << "-orbits_on_blocks "
						<< " " << orbits_on_blocks_sz
						<< " " << orbits_on_blocks_control
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-one_point_extension") == 0) {
			f_one_point_extension = true;
			one_point_extension_pair_orbit_idx = ST.strtoi(argv[++i]);
			one_point_extension_control.assign(argv[++i]);
			if (f_v) {
				cout << "-one_point_extension "
						<< " " << one_point_extension_pair_orbit_idx
						<< " " << one_point_extension_control
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "design_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "design_activity_description::read_arguments done" << endl;
	return i + 1;
}

void design_activity_description::print()
{
	if (f_canonical_form) {
		cout << "-canonical_form " << endl;
		Canonical_form_Descr->print();
	}
	if (f_load_table) {
		cout << "-load_table " << load_table_label
				<< " " << load_table_group
				<< " " << load_table_H_label
				<< " " << load_table_H_group_order
				<< " " << load_table_H_gens
				<< " " << load_table_selected_orbit_length
				<< endl;
	}
	if (f_extract_solutions_by_index_csv) {
		cout << "-extract_solutions_by_index_csv "
				<< extract_solutions_by_index_label << " "
				<< extract_solutions_by_index_group << " "
				<< extract_solutions_by_index_fname_solutions_in << " "
				<< extract_solutions_by_index_col_label << " "
				<< extract_solutions_by_index_fname_solutions_out << " "
				<< extract_solutions_by_index_prefix << " "
				<< endl;
	}
	if (f_extract_solutions_by_index_txt) {
		cout << "-extract_solutions_by_index_txt "
				<< extract_solutions_by_index_label << " "
				<< extract_solutions_by_index_group << " "
				<< extract_solutions_by_index_fname_solutions_in << " "
				<< extract_solutions_by_index_col_label << " "
				<< extract_solutions_by_index_fname_solutions_out << " "
				<< extract_solutions_by_index_prefix << " "
				<< endl;
	}
	if (f_export_inc) {
		cout << "-export_inc " << endl;
	}
	if (f_export_incidence_matrix) {
		cout << "-export_incidence_matrix " << endl;
	}
	if (f_export_incidence_matrix_latex) {
		cout << "-export_incidence_matrix_latex " << export_incidence_matrix_latex_draw_options << endl;
	}
	if (f_intersection_matrix) {
		cout << "-intersection_matrix " << endl;
	}
	if (f_save) {
		cout << "-save " << endl;
	}
	if (f_export_blocks) {
		cout << "-export_blocks " << endl;
	}
	if (f_row_sums) {
		cout << "-row_sums " << endl;
	}
	if (f_tactical_decomposition) {
		cout << "-tactical_decomposition " << endl;
	}
	if (f_orbits_on_blocks) {
		cout << "-orbits_on_blocks "
				<< " " << orbits_on_blocks_sz
				<< " " << orbits_on_blocks_control
				<< endl;
	}
	if (f_one_point_extension) {
		cout << "-one_point_extension "
				<< " " << one_point_extension_pair_orbit_idx
				<< " " << one_point_extension_control
				<< endl;
	}
}



}}}


