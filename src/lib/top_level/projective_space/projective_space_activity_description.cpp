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


	f_classify_surfaces_with_double_sixes = FALSE;
	//std::string classify_surfaces_with_double_sixes_label;
	classify_surfaces_with_double_sixes_control = NULL;



	f_classify_surfaces_through_arcs_and_two_lines = FALSE;
	f_test_nb_Eckardt_points = FALSE;
	nb_E = 0;
	f_classify_surfaces_through_arcs_and_trihedral_pairs = FALSE;
	f_trihedra1_control = FALSE;
	Trihedra1_control = NULL;
	f_trihedra2_control = FALSE;
	Trihedra2_control = NULL;
	f_control_six_arcs = FALSE;
			Control_six_arcs = NULL;
	f_create_surface = FALSE;
	surface_description = NULL;

	f_sweep = FALSE;
	//std::string sweep_fname;

	f_sweep_4 = FALSE;
	//std::string sweep_4_fname;
	sweep_4_surface_description = NULL;

	f_sweep_4_27 = FALSE;
	//std::string sweep_4_27_fname;
	sweep_4_27_surface_description = NULL;

	f_six_arcs = FALSE;
	f_filter_by_nb_Eckardt_points = FALSE;
	nb_Eckardt_points = 0;


	f_surface_quartic = FALSE;
	f_surface_clebsch = FALSE;
	f_surface_codes = FALSE;

	f_make_gilbert_varshamov_code = FALSE;
	make_gilbert_varshamov_code_n = 0;
	make_gilbert_varshamov_code_d = 0;

#if 0
	f_spread_table_init = FALSE;
	dimension_of_spread_elements = 0;
	//spread_selection_text = NULL;
	//spread_tables_prefix = NULL;
#endif

	f_spread_classify = FALSE;
	spread_classify_k = 0;
	spread_classify_Control = NULL;

	f_classify_semifields = FALSE;
	Semifield_classify_description = NULL;
	Semifield_classify_Control = NULL;

	f_cheat_sheet = FALSE;

	f_classify_quartic_curves_nauty = FALSE;
	//std::string classify_quartic_curves_nauty_fname_mask;
	classify_quartic_curves_nauty_nb = 0;

	f_classify_quartic_curves_with_substructure = FALSE;
	//std::string classify_quartic_curves_with_substructure_fname_mask;
	classify_quartic_curves_with_substructure_nb = 0;
	classify_quartic_curves_with_substructure_size = 0;
	//std::string classify_quartic_curves_with_substructure_fname_classification;

	f_set_stabilizer = FALSE;
	set_stabilizer_intermediate_set_size = 0;
	//std::string set_stabilizer_fname_mask;
	set_stabilizer_nb = 0;

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

		// cubic surfaces:
		else if (stringcmp(argv[i], "-classify_surfaces_with_double_sixes") == 0) {
			f_classify_surfaces_with_double_sixes = TRUE;
			classify_surfaces_with_double_sixes_label.assign(argv[++i]);
			classify_surfaces_with_double_sixes_control = NEW_OBJECT(poset_classification_control);
			cout << "-classify_surfaces_with_double_sixes " << endl;
			i += classify_surfaces_with_double_sixes_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -poset_classification_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-classify_surfaces_with_double_sixes " << classify_surfaces_with_double_sixes_label << endl;
			classify_surfaces_with_double_sixes_control->print();
		}

		else if (stringcmp(argv[i], "-classify_surfaces_through_arcs_and_two_lines") == 0) {
			f_classify_surfaces_through_arcs_and_two_lines = TRUE;
			cout << "-classify_surfaces_through_arcs_and_two_lines " << endl;
		}

		else if (stringcmp(argv[i], "-test_nb_Eckardt_points") == 0) {
			f_test_nb_Eckardt_points = TRUE;
			nb_E = strtoi(argv[++i]);
			cout << "-test_nb_Eckardt_points " << nb_E << endl;
		}
		else if (stringcmp(argv[i], "-classify_surfaces_through_arcs_and_trihedral_pairs") == 0) {
			f_classify_surfaces_through_arcs_and_trihedral_pairs = TRUE;
			cout << "-classify_surfaces_through_arcs_and_trihedral_pairs " << endl;
		}
		else if (stringcmp(argv[i], "-create_surface") == 0) {
			f_create_surface = TRUE;
			surface_description = NEW_OBJECT(surface_create_description);
			cout << "-create_surface" << endl;
			i += surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			cout << "done with -create_surface" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		else if (stringcmp(argv[i], "-sweep") == 0) {
			f_sweep = TRUE;
			sweep_fname.assign(argv[++i]);
			cout << "-sweep " << sweep_fname << endl;
		}

		else if (stringcmp(argv[i], "-sweep_4") == 0) {
			f_sweep_4 = TRUE;
			sweep_4_fname.assign(argv[++i]);
			sweep_4_surface_description = NEW_OBJECT(surface_create_description);
			cout << "-sweep_4" << endl;
			i += sweep_4_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			cout << "done with -sweep_4" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-sweep_4 " << sweep_4_fname << endl;
		}

		else if (stringcmp(argv[i], "-sweep_4_27") == 0) {
			f_sweep_4_27 = TRUE;
			sweep_4_27_fname.assign(argv[++i]);
			sweep_4_27_surface_description = NEW_OBJECT(surface_create_description);
			cout << "-sweep_4_27" << endl;
			i += sweep_4_27_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			cout << "done with -sweep_4_27" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-sweep_4_27 " << sweep_4_27_fname << endl;
		}

		else if (stringcmp(argv[i], "-six_arcs") == 0) {
			f_six_arcs = TRUE;
			cout << "-six_arcs" << endl;
		}
		else if (stringcmp(argv[i], "-filter_by_nb_Eckardt_points") == 0) {
			f_filter_by_nb_Eckardt_points = TRUE;
			nb_Eckardt_points = strtoi(argv[++i]);
			cout << "-filter_by_nb_Eckardt_points " << nb_Eckardt_points << endl;
		}
		else if (stringcmp(argv[i], "-surface_quartic") == 0) {
			f_surface_quartic = TRUE;
			cout << "-surface_quartic" << endl;
		}
		else if (stringcmp(argv[i], "-surface_clebsch") == 0) {
			f_surface_clebsch = TRUE;
			cout << "-surface_clebsch" << endl;
		}
		else if (stringcmp(argv[i], "-surface_codes") == 0) {
			f_surface_codes = TRUE;
			cout << "-surface_codes" << endl;
		}
		else if (stringcmp(argv[i], "-trihedra1_control") == 0) {
			f_trihedra1_control = TRUE;
			Trihedra1_control = NEW_OBJECT(poset_classification_control);
			i += Trihedra1_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -trihedra1_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-trihedra2_control") == 0) {
			f_trihedra2_control = TRUE;
			Trihedra2_control = NEW_OBJECT(poset_classification_control);
			i += Trihedra2_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -trihedra2_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-control_six_arcs") == 0) {
			f_control_six_arcs = TRUE;
			Control_six_arcs = NEW_OBJECT(poset_classification_control);
			i += Control_six_arcs->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -control_six_arcs " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-make_gilbert_varshamov_code") == 0) {
			f_make_gilbert_varshamov_code = TRUE;
			make_gilbert_varshamov_code_n = strtoi(argv[++i]);
			make_gilbert_varshamov_code_d = strtoi(argv[++i]);
			cout << "-make_gilbert_varshamov_code" << make_gilbert_varshamov_code_n
					<< " " << make_gilbert_varshamov_code_d << endl;
		}

		else if (stringcmp(argv[i], "-spread_classify") == 0) {
			f_spread_classify = TRUE;
			spread_classify_k = strtoi(argv[++i]);
			spread_classify_Control = NEW_OBJECT(poset_classification_control);
			cout << "-spread_classify " << endl;
			i += spread_classify_Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -spread_classify " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-spread_classify " << spread_classify_k << endl;
			spread_classify_Control->print();
		}
		// semifields
		else if (stringcmp(argv[i], "-classify_semifields") == 0) {
			f_classify_semifields = TRUE;
			Semifield_classify_description = NEW_OBJECT(semifield_classify_description);
			cout << "-classify_semifields" << endl;
			i += Semifield_classify_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -classify_semifields " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			Semifield_classify_Control = NEW_OBJECT(poset_classification_control);
			cout << "reading control " << endl;
			i += Semifield_classify_Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading control " << endl;
			cout << "-classify_semifields " << endl;
		}
		else if (stringcmp(argv[i], "-cheat_sheet") == 0) {
			f_cheat_sheet = TRUE;
			cout << "-cheat_sheet " << endl;
		}
		else if (stringcmp(argv[i], "-classify_quartic_curves_nauty") == 0) {
			f_classify_quartic_curves_nauty = TRUE;
			classify_quartic_curves_nauty_fname_mask.assign(argv[++i]);
			classify_quartic_curves_nauty_nb = strtoi(argv[++i]);
			cout << "-classify_quartic_curves_nauty "
					<< classify_quartic_curves_nauty_fname_mask
					<< " " << classify_quartic_curves_nauty_nb << endl;
		}
		else if (stringcmp(argv[i], "-classify_quartic_curves_with_substructure") == 0) {
			f_classify_quartic_curves_with_substructure = TRUE;
			classify_quartic_curves_with_substructure_fname_mask.assign(argv[++i]);
			classify_quartic_curves_with_substructure_nb = strtoi(argv[++i]);
			classify_quartic_curves_with_substructure_size = strtoi(argv[++i]);
			classify_quartic_curves_with_substructure_fname_classification.assign(argv[++i]);
			cout << "-classify_quartic_curves_with_substructure "
					<< classify_quartic_curves_with_substructure_fname_mask
					<< " " << classify_quartic_curves_with_substructure_nb
					<< " " << classify_quartic_curves_with_substructure_size
					<< " " << classify_quartic_curves_with_substructure_fname_classification
					<< endl;
		}
		else if (stringcmp(argv[i], "-set_stabilizer") == 0) {
			f_set_stabilizer = TRUE;
			set_stabilizer_intermediate_set_size = strtoi(argv[++i]);
			set_stabilizer_fname_mask.assign(argv[++i]);
			set_stabilizer_nb = strtoi(argv[++i]);
			cout << "-set_stabilizer "
					<< set_stabilizer_intermediate_set_size << " "
					<< set_stabilizer_fname_mask << " "
					<< set_stabilizer_nb << endl;
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
