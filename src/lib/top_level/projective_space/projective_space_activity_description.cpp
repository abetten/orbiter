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

	//f_input = FALSE;
	//Data = NULL;

#if 0
	f_canonical_form_PG = FALSE;
	//canonical_form_PG_n = 0;
	Canonical_form_PG_Descr = NULL;
#endif

	f_export_point_line_incidence_matrix = FALSE;

	f_table_of_cubic_surfaces_compute_properties = FALSE;
	//std::string _table_of_cubic_surfaces_compute_fname_csv;
	table_of_cubic_surfaces_compute_defining_q = 0;
	table_of_cubic_surfaces_compute_column_offset = 0;

	f_cubic_surface_properties_analyze = FALSE;
	//std::string cubic_surface_properties_fname_csv;
	cubic_surface_properties_defining_q = 0;

	f_canonical_form_of_code = FALSE;
	//canonical_form_of_code_label;
	//canonical_form_of_code_generator_matrix;
	Canonical_form_codes_Descr = NULL;

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


	f_decomposition_by_subgroup = FALSE;
	//std::string decomposition_by_subgroup_label;
	decomposition_by_subgroup_Descr = NULL;


	f_define_object = FALSE;
	//std::string define_object_label;
	Object_Descr = NULL;


	f_define_surface = FALSE;
	//std::string define_surface_label
	Surface_Descr = NULL;

	f_table_of_quartic_curves = FALSE;

	f_table_of_cubic_surfaces = FALSE;


	f_define_quartic_curve = FALSE;
	//std::string define_quartic_curve_label;
	Quartic_curve_descr = NULL;

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
	//f_create_surface = FALSE;
	//surface_description = NULL;

	f_sweep = FALSE;
	//std::string sweep_fname;

	f_sweep_4 = FALSE;
	//std::string sweep_4_fname;
	sweep_4_surface_description = NULL;

	f_sweep_4_27 = FALSE;
	//std::string sweep_4_27_fname;
	sweep_4_27_surface_description = NULL;

	f_six_arcs_not_on_conic = FALSE;
	f_filter_by_nb_Eckardt_points = FALSE;
	nb_Eckardt_points = 0;


	f_surface_quartic = FALSE;
	f_surface_clebsch = FALSE;
	f_surface_codes = FALSE;

	f_make_gilbert_varshamov_code = FALSE;
	make_gilbert_varshamov_code_n = 0;
	make_gilbert_varshamov_code_d = 0;

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
	//std::string classify_quartic_curves_nauty_fname_classification;

	f_classify_quartic_curves_with_substructure = FALSE;
	//std::string classify_quartic_curves_with_substructure_fname_mask;
	classify_quartic_curves_with_substructure_nb = 0;
	classify_quartic_curves_with_substructure_size = 0;
	classify_quartic_curves_with_substructure_degree = 0;
	//std::string classify_quartic_curves_with_substructure_fname_classification;

	f_set_stabilizer = FALSE;
	set_stabilizer_intermediate_set_size = 0;
	//std::string set_stabilizer_fname_mask;
	set_stabilizer_nb = 0;
	//std::string set_stabilizer_column_label;
	//std::string set_stabilizer_fname_out;

	f_conic_type = FALSE;
	conic_type_threshold = 0;
	//std::string conic_type_set_text;

	f_lift_skew_hexagon = FALSE;
	//lift_skew_hexagon_text

	f_lift_skew_hexagon_with_polarity = FALSE;
	//std::string lift_skew_hexagon_with_polarity_polarity;

	f_arc_with_given_set_as_s_lines_after_dualizing = FALSE;
	arc_size = 0;
	arc_d = 0;
	arc_d_low = 0;
	arc_s = 0;
	//std::string arc_input_set;
	//std::string arc_label;

	f_arc_with_two_given_sets_of_lines_after_dualizing = FALSE;
	//int arc_size;
	//int arc_d;
	arc_t = 0;
	//t_lines_string;

	f_arc_with_three_given_sets_of_lines_after_dualizing = FALSE;
	arc_u = 0;
	//u_lines_string;

	f_dualize_hyperplanes_to_points = FALSE;
	f_dualize_points_to_hyperplanes = FALSE;
	//std::string dualize_input_set;

	f_dualize_rank_k_subspaces = FALSE;
	dualize_rank_k_subspaces_k = 0;

	f_classify_arcs = FALSE;
	Arc_generator_description = NULL;

	f_classify_cubic_curves = FALSE;

	f_latex_homogeneous_equation = FALSE;
	latex_homogeneous_equation_degree = 0;
	//std::string latex_homogeneous_equation_symbol_txt
	//std::string latex_homogeneous_equation_symbol_tex
	//std::string latex_homogeneous_equation_text;

	f_lines_on_point_but_within_a_plane = FALSE;
	lines_on_point_but_within_a_plane_point_rk = 0;
	lines_on_point_but_within_a_plane_plane_rk = 0;

}

projective_space_activity_description::~projective_space_activity_description()
{

}


int projective_space_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "projective_space_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

#if 0
		if (stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			if (f_v) {
				cout << "-input" << endl;
			}
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			if (f_v) {
				cout << "projective_space_activity_description::read_arguments finished reading -input" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
#endif

#if 0
		if (stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			if (f_v) {
				cout << "-canonical_form_PG, reading extra arguments" << endl;
			}

			Canonical_form_PG_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_PG_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -canonical_form_PG " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
#endif

		if (ST.stringcmp(argv[i], "-export_point_line_incidence_matrix") == 0) {
			f_export_point_line_incidence_matrix = TRUE;
			if (f_v) {
				cout << "-export_point_line_incidence_matrix " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-table_of_cubic_surfaces_compute_properties") == 0) {
			f_table_of_cubic_surfaces_compute_properties = TRUE;
			if (f_v) {
				cout << "-table_of_cubic_surfaces_compute_properties next argument is " << argv[i + 1] << endl;
				table_of_cubic_surfaces_compute_fname_csv.assign(argv[++i]);
				table_of_cubic_surfaces_compute_defining_q = ST.strtoi(argv[++i]);
				table_of_cubic_surfaces_compute_column_offset = ST.strtoi(argv[++i]);
				cout << "-table_of_cubic_surfaces_compute_properties "
						<< table_of_cubic_surfaces_compute_fname_csv << " "
						<< table_of_cubic_surfaces_compute_defining_q << " "
						<< table_of_cubic_surfaces_compute_column_offset << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-cubic_surface_properties_analyze") == 0) {
			f_cubic_surface_properties_analyze = TRUE;
			cubic_surface_properties_fname_csv.assign(argv[++i]);
			cubic_surface_properties_defining_q = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cubic_surface_properties "
						<< cubic_surface_properties_fname_csv
						<< " " << cubic_surface_properties_defining_q << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-canonical_form_of_code") == 0) {
			f_canonical_form_of_code = TRUE;
			canonical_form_of_code_label.assign(argv[++i]);
			canonical_form_of_code_generator_matrix.assign(argv[++i]);

			Canonical_form_codes_Descr = NEW_OBJECT(combinatorics::classification_of_objects_description);

			i += Canonical_form_codes_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -canonical_form_of_code " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}


			if (f_v) {
				cout << "-canonical_form_of_code "
						<< canonical_form_of_code_label << " "
						<< canonical_form_of_code_generator_matrix << " "
						<< endl;
				Canonical_form_codes_Descr->print();
			}
		}

		else if (ST.stringcmp(argv[i], "-map") == 0) {
			f_map = TRUE;
			map_label.assign(argv[++i]);
			map_parameters.assign(argv[++i]);
			if (f_v) {
				cout << "-map "
						<< map_label << " "
						<< map_parameters << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-analyze_del_Pezzo_surface") == 0) {
			f_analyze_del_Pezzo_surface = TRUE;
			analyze_del_Pezzo_surface_label.assign(argv[++i]);
			analyze_del_Pezzo_surface_parameters.assign(argv[++i]);
			if (f_v) {
				cout << "-analyze_del_Pezzo_surface "
						<< analyze_del_Pezzo_surface_label << " "
						<< analyze_del_Pezzo_surface_parameters << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-cheat_sheet_for_decomposition_by_element_PG") == 0) {
			f_cheat_sheet_for_decomposition_by_element_PG = TRUE;
			decomposition_by_element_power = ST.strtoi(argv[++i]);
			decomposition_by_element_data.assign(argv[++i]);
			decomposition_by_element_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_for_decomposition_by_element_PG "
						<< decomposition_by_element_power
						<< " " << decomposition_by_element_data
						<< " " << decomposition_by_element_fname
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-decomposition_by_subgroup") == 0) {
			f_decomposition_by_subgroup = TRUE;
			decomposition_by_subgroup_label.assign(argv[++i]);
			decomposition_by_subgroup_Descr = NEW_OBJECT(groups::linear_group_description);
			i += decomposition_by_subgroup_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -H" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			if (f_v) {
				cout << "-decomposition_by_subgroup "
						<< decomposition_by_subgroup_label
						<< " " << decomposition_by_element_data
						<< " " << decomposition_by_element_fname
						<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-define_object") == 0) {
			f_define_object = TRUE;
			if (f_v) {
				cout << "-define_object, reading extra arguments" << endl;
			}

			define_object_label.assign(argv[++i]);
			Object_Descr = NEW_OBJECT(geometric_object_description);

			i += Object_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -define_object " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-define_object " << define_object_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-define_surface") == 0) {
			f_define_surface = TRUE;
			if (f_v) {
				cout << "-define_surface, reading extra arguments" << endl;
			}

			define_surface_label.assign(argv[++i]);
			Surface_Descr = NEW_OBJECT(surface_create_description);

			i += Surface_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -define_surface " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-define_surface " << define_surface_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-table_of_quartic_curves") == 0) {
			f_table_of_quartic_curves = TRUE;
			if (f_v) {
				cout << "-table_of_quartic_curves " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-table_of_cubic_surfaces") == 0) {
			f_table_of_cubic_surfaces = TRUE;
			if (f_v) {
				cout << "-table_of_cubic_surfaces " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-define_quartic_curve") == 0) {
			f_define_quartic_curve = TRUE;
			if (f_v) {
				cout << "-define_quartic_curve, reading extra arguments" << endl;
			}

			define_quartic_curve_label.assign(argv[++i]);
			Quartic_curve_descr = NEW_OBJECT(quartic_curve_create_description);

			i += Quartic_curve_descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -define_quartic_curve " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-define_quartic_curve " << define_quartic_curve_label << endl;
			}
		}


		// cubic surfaces:
		else if (ST.stringcmp(argv[i], "-classify_surfaces_with_double_sixes") == 0) {
			f_classify_surfaces_with_double_sixes = TRUE;
			classify_surfaces_with_double_sixes_label.assign(argv[++i]);
			classify_surfaces_with_double_sixes_control = NEW_OBJECT(poset_classification_control);
			if (f_v) {
				cout << "-classify_surfaces_with_double_sixes " << endl;
			}
			i += classify_surfaces_with_double_sixes_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -poset_classification_control " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-classify_surfaces_with_double_sixes " << classify_surfaces_with_double_sixes_label << endl;
				classify_surfaces_with_double_sixes_control->print();
			}
		}

		else if (ST.stringcmp(argv[i], "-classify_surfaces_through_arcs_and_two_lines") == 0) {
			f_classify_surfaces_through_arcs_and_two_lines = TRUE;
			if (f_v) {
				cout << "-classify_surfaces_through_arcs_and_two_lines " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-test_nb_Eckardt_points") == 0) {
			f_test_nb_Eckardt_points = TRUE;
			nb_E = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_nb_Eckardt_points " << nb_E << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-classify_surfaces_through_arcs_and_trihedral_pairs") == 0) {
			f_classify_surfaces_through_arcs_and_trihedral_pairs = TRUE;
			if (f_v) {
				cout << "-classify_surfaces_through_arcs_and_trihedral_pairs " << endl;
			}
		}
#if 0
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
#endif

		else if (ST.stringcmp(argv[i], "-sweep") == 0) {
			f_sweep = TRUE;
			sweep_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-sweep " << sweep_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-sweep_4") == 0) {
			f_sweep_4 = TRUE;
			sweep_4_fname.assign(argv[++i]);
			sweep_4_surface_description = NEW_OBJECT(surface_create_description);
			if (f_v) {
				cout << "-sweep_4" << endl;
			}
			i += sweep_4_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -sweep_4" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-sweep_4 " << sweep_4_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-sweep_4_27") == 0) {
			f_sweep_4_27 = TRUE;
			sweep_4_27_fname.assign(argv[++i]);
			sweep_4_27_surface_description = NEW_OBJECT(surface_create_description);
			if (f_v) {
				cout << "-sweep_4_27" << endl;
			}
			i += sweep_4_27_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -sweep_4_27" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-sweep_4_27 " << sweep_4_27_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-six_arcs_not_on_conic") == 0) {
			f_six_arcs_not_on_conic = TRUE;
			if (f_v) {
				cout << "-six_arcs_not_on_conic" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-filter_by_nb_Eckardt_points") == 0) {
			f_filter_by_nb_Eckardt_points = TRUE;
			nb_Eckardt_points = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-filter_by_nb_Eckardt_points " << nb_Eckardt_points << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-surface_quartic") == 0) {
			f_surface_quartic = TRUE;
			if (f_v) {
				cout << "-surface_quartic" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-surface_clebsch") == 0) {
			f_surface_clebsch = TRUE;
			if (f_v) {
				cout << "-surface_clebsch" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-surface_codes") == 0) {
			f_surface_codes = TRUE;
			if (f_v) {
				cout << "-surface_codes" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-trihedra1_control") == 0) {
			f_trihedra1_control = TRUE;
			Trihedra1_control = NEW_OBJECT(poset_classification_control);
			i += Trihedra1_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -trihedra1_control " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

		else if (ST.stringcmp(argv[i], "-trihedra2_control") == 0) {
			f_trihedra2_control = TRUE;
			Trihedra2_control = NEW_OBJECT(poset_classification_control);
			i += Trihedra2_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -trihedra2_control " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

		else if (ST.stringcmp(argv[i], "-control_six_arcs") == 0) {
			f_control_six_arcs = TRUE;
			Control_six_arcs = NEW_OBJECT(poset_classification_control);
			i += Control_six_arcs->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -control_six_arcs " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

		else if (ST.stringcmp(argv[i], "-make_gilbert_varshamov_code") == 0) {
			f_make_gilbert_varshamov_code = TRUE;
			make_gilbert_varshamov_code_n = ST.strtoi(argv[++i]);
			make_gilbert_varshamov_code_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-make_gilbert_varshamov_code" << make_gilbert_varshamov_code_n
						<< " " << make_gilbert_varshamov_code_d << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-spread_classify") == 0) {
			f_spread_classify = TRUE;
			spread_classify_k = ST.strtoi(argv[++i]);
			spread_classify_Control = NEW_OBJECT(poset_classification_control);
			if (f_v) {
				cout << "-spread_classify " << endl;
			}
			i += spread_classify_Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -spread_classify " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-spread_classify " << spread_classify_k << endl;
				spread_classify_Control->print();
			}
		}

		// semifields
		else if (ST.stringcmp(argv[i], "-classify_semifields") == 0) {
			f_classify_semifields = TRUE;
			Semifield_classify_description = NEW_OBJECT(semifield_classify_description);
			if (f_v) {
				cout << "-classify_semifields" << endl;
			}
			i += Semifield_classify_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classify_semifields " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			Semifield_classify_Control = NEW_OBJECT(poset_classification_control);
			if (f_v) {
				cout << "reading control " << endl;
			}
			i += Semifield_classify_Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading control " << endl;
				cout << "-classify_semifields " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-cheat_sheet") == 0) {
			f_cheat_sheet = TRUE;
			if (f_v) {
				cout << "-cheat_sheet " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-classify_quartic_curves_nauty") == 0) {
			f_classify_quartic_curves_nauty = TRUE;
			classify_quartic_curves_nauty_fname_mask.assign(argv[++i]);
			classify_quartic_curves_nauty_nb = ST.strtoi(argv[++i]);
			classify_quartic_curves_nauty_fname_classification.assign(argv[++i]);
			if (f_v) {
				cout << "-classify_quartic_curves_nauty "
						<< classify_quartic_curves_nauty_fname_mask
						<< " " << classify_quartic_curves_nauty_nb
						<< " " << classify_quartic_curves_nauty_fname_classification
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-classify_quartic_curves_with_substructure") == 0) {
			f_classify_quartic_curves_with_substructure = TRUE;
			classify_quartic_curves_with_substructure_fname_mask.assign(argv[++i]);
			classify_quartic_curves_with_substructure_nb = ST.strtoi(argv[++i]);
			classify_quartic_curves_with_substructure_size = ST.strtoi(argv[++i]);
			classify_quartic_curves_with_substructure_degree = ST.strtoi(argv[++i]);
			classify_quartic_curves_with_substructure_fname_classification.assign(argv[++i]);
			if (f_v) {
				cout << "-classify_quartic_curves_with_substructure "
						<< classify_quartic_curves_with_substructure_fname_mask
						<< " " << classify_quartic_curves_with_substructure_nb
						<< " " << classify_quartic_curves_with_substructure_size
						<< " " << classify_quartic_curves_with_substructure_degree
						<< " " << classify_quartic_curves_with_substructure_fname_classification
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-set_stabilizer") == 0) {
			f_set_stabilizer = TRUE;
			set_stabilizer_intermediate_set_size = ST.strtoi(argv[++i]);
			set_stabilizer_fname_mask.assign(argv[++i]);
			set_stabilizer_nb = ST.strtoi(argv[++i]);
			set_stabilizer_column_label.assign(argv[++i]);
			set_stabilizer_fname_out.assign(argv[++i]);
			if (f_v) {
				cout << "-set_stabilizer "
						<< set_stabilizer_intermediate_set_size << " "
						<< set_stabilizer_fname_mask << " "
						<< set_stabilizer_nb << " "
						<< set_stabilizer_column_label << " "
						<< set_stabilizer_fname_out << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			conic_type_threshold = ST.strtoi(argv[++i]);
			conic_type_set_text.assign(argv[++i]);
			if (f_v) {
				cout << "-conic_type "
						<< " " << conic_type_threshold
						<< " " << conic_type_set_text  << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-lift_skew_hexagon") == 0) {
			f_lift_skew_hexagon = TRUE;
			lift_skew_hexagon_text.assign(argv[++i]);
			if (f_v) {
				cout << "-lift_skew_hexagon "
						<< lift_skew_hexagon_text << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-lift_skew_hexagon_with_polarity") == 0) {
			f_lift_skew_hexagon_with_polarity = TRUE;
			lift_skew_hexagon_with_polarity_polarity.assign(argv[++i]);
			if (f_v) {
				cout << "-lift_skew_hexagon_with_polarity "
						<< " " << lift_skew_hexagon_with_polarity_polarity
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-arc_with_given_set_as_s_lines_after_dualizing") == 0) {
			f_arc_with_given_set_as_s_lines_after_dualizing = TRUE;
			arc_size = ST.strtoi(argv[++i]);
			arc_d = ST.strtoi(argv[++i]);
			arc_d_low = ST.strtoi(argv[++i]);
			arc_s = ST.strtoi(argv[++i]);
			arc_input_set.assign(argv[++i]);
			arc_label.assign(argv[++i]);
			if (f_v) {
				cout << "-arc_with_given_set_as_s_lines_after_dualizing "
						<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " " << arc_input_set << " " << arc_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-arc_with_two_given_sets_of_lines_after_dualizing") == 0) {
			f_arc_with_two_given_sets_of_lines_after_dualizing = TRUE;
			arc_size = ST.strtoi(argv[++i]);
			arc_d = ST.strtoi(argv[++i]);
			arc_d_low = ST.strtoi(argv[++i]);
			arc_s = ST.strtoi(argv[++i]);
			arc_t = ST.strtoi(argv[++i]);
			t_lines_string.assign(argv[++i]);
			arc_input_set.assign(argv[++i]);
			arc_label.assign(argv[++i]);
			if (f_v) {
				cout << "-arc_with_two_given_sets_of_lines_after_dualizing src_size="
						<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " t=" << arc_t << " " << t_lines_string << " " << arc_input_set << " " << arc_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-arc_with_three_given_sets_of_lines_after_dualizing") == 0) {
			f_arc_with_three_given_sets_of_lines_after_dualizing = TRUE;
			arc_size = ST.strtoi(argv[++i]);
			arc_d = ST.strtoi(argv[++i]);
			arc_d_low = ST.strtoi(argv[++i]);
			arc_s = ST.strtoi(argv[++i]);
			arc_t = ST.strtoi(argv[++i]);
			t_lines_string.assign(argv[++i]);
			arc_u = ST.strtoi(argv[++i]);
			u_lines_string.assign(argv[++i]);
			arc_input_set.assign(argv[++i]);
			arc_label.assign(argv[++i]);
			if (f_v) {
				cout << "-arc_with_three_given_sets_of_lines_after_dualizing "
						<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " " << arc_input_set << " " << arc_label << endl;
				cout << "arc_t = " << arc_t << " t_lines_string = " << t_lines_string << endl;
				cout << "arc_u = " << arc_u << " u_lines_string = " << u_lines_string << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-dualize_hyperplanes_to_points") == 0) {
			f_dualize_hyperplanes_to_points = TRUE;
			dualize_input_set.assign(argv[++i]);
			if (f_v) {
				cout << "-dualize_hyperplanes_to_points " << dualize_input_set << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-dualize_points_to_hyperplanes") == 0) {
			f_dualize_points_to_hyperplanes = TRUE;
			dualize_input_set.assign(argv[++i]);
			if (f_v) {
				cout << "-dualize_points_to_hyperplanes " << dualize_input_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dualize_rank_k_subspaces") == 0) {
			f_dualize_rank_k_subspaces = TRUE;
			dualize_rank_k_subspaces_k = ST.strtoi(argv[++i]);
			dualize_input_set.assign(argv[++i]);
			if (f_v) {
				cout << "-dualize_rank_k_subspaces " << dualize_rank_k_subspaces_k << " " << dualize_input_set << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-classify_arcs") == 0) {
			f_classify_arcs = TRUE;
			Arc_generator_description = NEW_OBJECT(arc_generator_description);
			if (f_v) {
				cout << "-classify_arcs" << endl;
			}
			i += Arc_generator_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classify_arcs " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		// cubic curves
		else if (ST.stringcmp(argv[i], "-classify_cubic_curves") == 0) {
			f_classify_cubic_curves = TRUE;
			Arc_generator_description = NEW_OBJECT(arc_generator_description);
			if (f_v) {
				cout << "-classify_cubic_curves" << endl;
			}
			i += Arc_generator_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classify_cubic_curves " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-classify_cubic_curves " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-latex_homogeneous_equation") == 0) {
			f_latex_homogeneous_equation = TRUE;
			latex_homogeneous_equation_degree = ST.strtoi(argv[++i]);
			latex_homogeneous_equation_symbol_txt.assign(argv[++i]);
			latex_homogeneous_equation_symbol_tex.assign(argv[++i]);
			latex_homogeneous_equation_text.assign(argv[++i]);
			if (f_v) {
				cout << "-latex_homogeneous_equation " << latex_homogeneous_equation_degree
						<< " " << latex_homogeneous_equation_symbol_txt
						<< " " << latex_homogeneous_equation_symbol_tex
						<< " " << latex_homogeneous_equation_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lines_on_point_but_within_a_plane") == 0) {
			f_lines_on_point_but_within_a_plane = TRUE;
			lines_on_point_but_within_a_plane_point_rk = ST.strtoi(argv[++i]);
			lines_on_point_but_within_a_plane_plane_rk = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-lines_on_point_but_within_a_plane "
						<< " " << lines_on_point_but_within_a_plane_point_rk
						<< " " << lines_on_point_but_within_a_plane_plane_rk
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}

		else {
			cout << "projective_space_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

		if (f_v) {
			cout << "projective_space_activity_description::read_arguments looping, i=" << i << endl;
		}
	} // next i

	if (f_v) {
		cout << "projective_space_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void projective_space_activity_description::print()
{
	if (f_table_of_cubic_surfaces_compute_properties) {
		cout << "-table_of_cubic_surfaces_compute_properties "
				<< table_of_cubic_surfaces_compute_fname_csv << " "
				<< table_of_cubic_surfaces_compute_defining_q << " "
				<< table_of_cubic_surfaces_compute_column_offset << " "
				<< endl;
	}
	if (f_cubic_surface_properties_analyze) {
		cout << "-cubic_surface_properties " << cubic_surface_properties_fname_csv
				<< " " << cubic_surface_properties_defining_q << endl;
	}
	if (f_canonical_form_of_code) {
		cout << "-canonical_form_of_code "
				<< canonical_form_of_code_label << " "
				<< canonical_form_of_code_generator_matrix << " "
				<< endl;
		Canonical_form_codes_Descr->print();
	}

	if (f_map) {
		cout << "-map "
				<< map_label << " "
				<< map_parameters << " "
				<< endl;
	}
	if (f_analyze_del_Pezzo_surface) {
		cout << "-analyze_del_Pezzo_surface "
				<< analyze_del_Pezzo_surface_label << " "
				<< analyze_del_Pezzo_surface_parameters << " "
				<< endl;
	}
	if (f_cheat_sheet_for_decomposition_by_element_PG) {
		cout << "-cheat_sheet_for_decomposition_by_element_PG "
				<< decomposition_by_element_power
				<< " " << decomposition_by_element_data
				<< " " << decomposition_by_element_fname
				<< endl;
	}


	if (f_decomposition_by_subgroup) {
		cout << "-decomposition_by_subgroup "
					<< decomposition_by_subgroup_label
					<< " ";
		decomposition_by_subgroup_Descr->print();
	}

	if (f_define_object) {
		cout << "-define_object " << define_object_label << endl;
		Object_Descr->print();
	}
	if (f_define_surface) {
		cout << "-define_surface " << define_surface_label << endl;
		Surface_Descr->print();
	}
	if (f_table_of_quartic_curves) {
		cout << "-table_of_quartic_curves " << endl;
	}
	if (f_table_of_cubic_surfaces) {
		cout << "-table_of_cubic_surfaces " << endl;
	}
	if (f_define_quartic_curve) {
		cout << "-define_quartic_curve " << define_quartic_curve_label << endl;
		Quartic_curve_descr->print();
	}


	// cubic surfaces:
	if (f_classify_surfaces_with_double_sixes) {
		cout << "-classify_surfaces_with_double_sixes " << classify_surfaces_with_double_sixes_label << endl;
		classify_surfaces_with_double_sixes_control->print();
	}

	if (f_classify_surfaces_through_arcs_and_two_lines) {
		cout << "-classify_surfaces_through_arcs_and_two_lines " << endl;
	}

	if (f_test_nb_Eckardt_points) {
		cout << "-test_nb_Eckardt_points " << nb_E << endl;
	}
	if (f_classify_surfaces_through_arcs_and_trihedral_pairs) {
		cout << "-classify_surfaces_through_arcs_and_trihedral_pairs " << endl;
	}

	if (f_sweep) {
		cout << "-sweep " << sweep_fname << endl;
	}

	if (f_sweep_4) {
		cout << "-sweep_4 " << sweep_4_fname << endl;
	}

	if (f_sweep_4_27) {
		cout << "-sweep_4_27 " << sweep_4_27_fname << endl;
	}

	if (f_six_arcs_not_on_conic) {
		cout << "-six_arcs_not_on_conic" << endl;
	}
	if (f_filter_by_nb_Eckardt_points) {
		cout << "-filter_by_nb_Eckardt_points " << nb_Eckardt_points << endl;
	}
	if (f_surface_quartic) {
		cout << "-surface_quartic" << endl;
	}
	if (f_surface_quartic) {
		cout << "-surface_clebsch" << endl;
	}
	if (f_surface_codes) {
		cout << "-surface_codes" << endl;
	}
	if (f_trihedra1_control) {
		cout << "-trihedra1_control " << endl;
		Trihedra1_control->print();
	}
	if (f_trihedra2_control) {
		cout << "-trihedra2_control " << endl;
		Trihedra2_control->print();
	}
	if (f_control_six_arcs) {
		cout << "-control_six_arcs " << endl;
		Control_six_arcs->print();
	}
	if (f_make_gilbert_varshamov_code) {
		cout << "-make_gilbert_varshamov_code" << make_gilbert_varshamov_code_n
				<< " " << make_gilbert_varshamov_code_d << endl;
	}

	if (f_spread_classify) {
		f_spread_classify = TRUE;
		cout << "-spread_classify " << spread_classify_k << endl;
		spread_classify_Control->print();
	}

	// semifields
	if (f_classify_semifields) {
		cout << "-classify_semifields " << endl;
		Semifield_classify_Control->print();
	}
	if (f_cheat_sheet) {
		cout << "-cheat_sheet " << endl;
	}
	if (f_classify_quartic_curves_nauty) {
		cout << "-classify_quartic_curves_nauty "
				<< classify_quartic_curves_nauty_fname_mask
				<< " " << classify_quartic_curves_nauty_nb
				<< " " << classify_quartic_curves_nauty_fname_classification
				<< endl;
	}
	if (f_classify_quartic_curves_with_substructure) {
		cout << "-classify_quartic_curves_with_substructure "
				<< classify_quartic_curves_with_substructure_fname_mask
				<< " " << classify_quartic_curves_with_substructure_nb
				<< " " << classify_quartic_curves_with_substructure_size
				<< " " << classify_quartic_curves_with_substructure_degree
				<< " " << classify_quartic_curves_with_substructure_fname_classification
				<< endl;
	}
	if (f_set_stabilizer) {
		cout << "-set_stabilizer "
				<< set_stabilizer_intermediate_set_size << " "
				<< set_stabilizer_fname_mask << " "
				<< set_stabilizer_nb << " "
				<< set_stabilizer_column_label << " "
				<< set_stabilizer_fname_out << " "
				<< endl;
	}
	if (f_conic_type) {
		cout << "-conic_type "
				<< conic_type_set_text << endl;
	}

	if (f_lift_skew_hexagon) {
		cout << "-lift_skew_hexagon "
				<< lift_skew_hexagon_text << endl;
	}

	if (f_lift_skew_hexagon_with_polarity) {
		cout << "-lift_skew_hexagon_with_polarity "
				<< " " << lift_skew_hexagon_with_polarity_polarity
				<< endl;
	}
	if (f_arc_with_given_set_as_s_lines_after_dualizing) {
		cout << "-arc_with_given_set_as_s_lines_after_dualizing "
				<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " " << arc_input_set << " " << arc_label << endl;
	}
	if (f_arc_with_two_given_sets_of_lines_after_dualizing) {
		cout << "-arc_with_two_given_sets_of_lines_after_dualizing src_size="
				<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " t=" << arc_t << " " << t_lines_string << " " << arc_input_set << " " << arc_label << endl;
	}
	if (f_arc_with_three_given_sets_of_lines_after_dualizing) {
		cout << "-arc_with_three_given_sets_of_lines_after_dualizing "
				<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " " << arc_input_set << " " << arc_label << endl;
		cout << "arc_t = " << arc_t << " t_lines_string = " << t_lines_string << endl;
		cout << "arc_u = " << arc_u << " u_lines_string = " << u_lines_string << endl;
	}
	if (f_dualize_hyperplanes_to_points) {
		cout << "-dualize_hyperplanes_to_points" << " " << dualize_input_set << endl;
	}
	if (f_dualize_points_to_hyperplanes) {
		cout << "-dualize_points_to_hyperplanes" << " " << dualize_input_set << endl;
	}
	if (f_dualize_rank_k_subspaces) {
		cout << "-dualize_rank_k_subspaces " << dualize_rank_k_subspaces_k << " " << dualize_input_set << endl;
	}
	if (f_classify_arcs) {
		cout << "-classify_arcs " << endl;
		Arc_generator_description->print();
	}
	// cubic curves
	if (f_classify_cubic_curves) {
		cout << "-classify_cubic_curves" << endl;
		Arc_generator_description->print();
	}
	if (f_latex_homogeneous_equation) {
		cout << "-latex_homogeneous_equation " << latex_homogeneous_equation_degree
				<< " " << latex_homogeneous_equation_symbol_txt
				<< " " << latex_homogeneous_equation_symbol_tex
				<< " " << latex_homogeneous_equation_text << endl;
	}
	if (f_lines_on_point_but_within_a_plane) {
		cout << "-lines_on_point_but_within_a_plane "
				<< " " << lines_on_point_but_within_a_plane_point_rk
				<< " " << lines_on_point_but_within_a_plane_plane_rk
				<< endl;
	}

}



}}
