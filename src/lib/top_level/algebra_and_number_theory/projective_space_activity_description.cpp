/*
 * projective_space_activity_description.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


projective_space_activity_description::projective_space_activity_description()
{

	f_input = FALSE;
	Data = NULL;

	f_canonical_form_PG = FALSE;
	//canonical_form_PG_n = 0;
	Canonical_form_PG_Descr = NULL;

	f_table_of_cubic_surfaces_compute_properties = FALSE;
	//std::string _table_of_cubic_surfaces_compute_fname_csv;
	table_of_cubic_surfaces_compute_defining_q = 0;
	table_of_cubic_surfaces_compute_column_offset = 0;

	f_cubic_surface_properties_analyze = FALSE;
	//std::string cubic_surface_properties_fname_csv;
	cubic_surface_properties_defining_q = 0;

	f_canonical_form_of_code = FALSE;
	//canonical_form_of_code_label;
	canonical_form_of_code_m = 0;
	canonical_form_of_code_n = 0;
	//canonical_form_of_code_text;

	f_map = FALSE;
	//std::string map_label;
	//std::string map_parameters;

	f_analyze_del_Pezzo_surface = FALSE;
	//analyze_del_Pezzo_surface_label;
	//analyze_del_Pezzo_surface_parameters;

	f_cheat_sheet_for_decomposition_by_element_PG = FALSE;
	decomposition_by_element_power = 0;
	//std::string decomposition_by_element_data;
	//std::string decomposition_by_element_fname;


	f_define_surface = FALSE;
	//std::string define_surface_label
	Surface_Descr = NULL;
}

projective_space_activity_description::~projective_space_activity_description()
{

}


int projective_space_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "projective_space_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			cout << "-input" << endl;
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "projective_space_activity_description::read_arguments finished reading -input" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			//canonical_form_PG_n = strtoi(argv[++i]);
			cout << "-canonical_form_PG, reading extra arguments" << endl;

			Canonical_form_PG_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_PG_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			cout << "done reading -canonical_form_PG " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-table_of_cubic_surfaces_compute_properties") == 0) {
			f_table_of_cubic_surfaces_compute_properties = TRUE;
			cout << "-table_of_cubic_surfaces_compute_properties next argument is " << argv[i + 1] << endl;
			table_of_cubic_surfaces_compute_fname_csv.assign(argv[++i]);
			table_of_cubic_surfaces_compute_defining_q = strtoi(argv[++i]);
			table_of_cubic_surfaces_compute_column_offset = strtoi(argv[++i]);
			cout << "-table_of_cubic_surfaces_compute_properties "
					<< table_of_cubic_surfaces_compute_fname_csv << " "
					<< table_of_cubic_surfaces_compute_defining_q << " "
					<< table_of_cubic_surfaces_compute_column_offset << " "
					<< endl;
		}
		else if (stringcmp(argv[i], "-cubic_surface_properties_analyze") == 0) {
			f_cubic_surface_properties_analyze = TRUE;
			cubic_surface_properties_fname_csv.assign(argv[++i]);
			cubic_surface_properties_defining_q = strtoi(argv[++i]);
			cout << "-cubic_surface_properties " << cubic_surface_properties_fname_csv
					<< " " << cubic_surface_properties_defining_q << endl;
		}
		else if (stringcmp(argv[i], "-canonical_form_of_code") == 0) {
			f_canonical_form_of_code = TRUE;
			canonical_form_of_code_label.assign(argv[++i]);
			canonical_form_of_code_m = strtoi(argv[++i]);
			canonical_form_of_code_n = strtoi(argv[++i]);
			canonical_form_of_code_text.assign(argv[++i]);
			cout << "-canonical_form_of_code "
					<< canonical_form_of_code_label << " "
					<< canonical_form_of_code_m << " "
					<< canonical_form_of_code_n << " "
					<< canonical_form_of_code_text << " "
					<< endl;
		}
		else if (stringcmp(argv[i], "-map") == 0) {
			f_map = TRUE;
			map_label.assign(argv[++i]);
			map_parameters.assign(argv[++i]);
			cout << "-map "
					<< map_label << " "
					<< map_parameters << " "
					<< endl;
		}
		else if (stringcmp(argv[i], "-analyze_del_Pezzo_surface") == 0) {
			f_analyze_del_Pezzo_surface = TRUE;
			analyze_del_Pezzo_surface_label.assign(argv[++i]);
			analyze_del_Pezzo_surface_parameters.assign(argv[++i]);
			cout << "-analyze_del_Pezzo_surface "
					<< analyze_del_Pezzo_surface_label << " "
					<< analyze_del_Pezzo_surface_parameters << " "
					<< endl;
		}
		else if (stringcmp(argv[i], "-cheat_sheet_for_decomposition_by_element_PG") == 0) {
			f_cheat_sheet_for_decomposition_by_element_PG = TRUE;
			decomposition_by_element_power = strtoi(argv[++i]);
			decomposition_by_element_data.assign(argv[++i]);
			decomposition_by_element_fname.assign(argv[++i]);
			cout << "-cheat_sheet_for_decomposition_by_element_PG "
					<< decomposition_by_element_power
					<< " " << decomposition_by_element_data
					<< " " << decomposition_by_element_fname
					<< endl;
		}
		else if (stringcmp(argv[i], "-define_surface") == 0) {
			f_define_surface = TRUE;
			cout << "-define_surface, reading extra arguments" << endl;

			define_surface_label.assign(argv[++i]);
			Surface_Descr = NEW_OBJECT(surface_create_description);

			i += Surface_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			cout << "done reading -define_surface " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-define_surface " << define_surface_label << endl;
		}

		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "projective_space_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "projective_space_activity_description::read_arguments looping, i=" << i << endl;
	} // next i

	cout << "projective_space_activity_description::read_arguments done" << endl;
	return i + 1;
}


}}
