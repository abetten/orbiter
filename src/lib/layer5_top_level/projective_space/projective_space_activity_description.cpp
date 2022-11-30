/*
 * projective_space_activity_description.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {


projective_space_activity_description::projective_space_activity_description()
{


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
	//std::string map_ring_label;
	//std::string map_formula_label;
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

	f_table_of_quartic_curves = FALSE;

	f_table_of_cubic_surfaces = FALSE;

	f_classify_surfaces_with_double_sixes = FALSE;
	//std::string classify_surfaces_with_double_sixes_label;
	//std::string classify_surfaces_with_double_sixes_control_label;



	f_classify_surfaces_through_arcs_and_two_lines = FALSE;
	f_test_nb_Eckardt_points = FALSE;
	nb_E = 0;
	f_classify_surfaces_through_arcs_and_trihedral_pairs = FALSE;
	f_trihedra1_control = FALSE;
	Trihedra1_control = NULL;
	f_trihedra2_control = FALSE;
	Trihedra2_control = NULL;
	f_control_six_arcs = FALSE;
	//std::string &Control_six_arcs_label;

	//f_create_surface = FALSE;
	//surface_description = NULL;

	f_sweep = FALSE;
	//std::string sweep_fname;

	f_sweep_4_15_lines = FALSE;
	//std::string sweep_4_15_lines_fname;
	sweep_4_15_lines_surface_description = NULL;

	f_sweep_F_beta_9_lines = FALSE;
	//std::string sweep_F_beta_9_lines_fname;
	sweep_F_beta_9_lines_surface_description = NULL;

	f_sweep_6_9_lines = FALSE;
	//std::string sweep_6_9_lines_fname;
	sweep_6_9_lines_surface_description = NULL;

	f_sweep_4_27 = FALSE;
	//std::string sweep_4_27_fname;
	sweep_4_27_surface_description = NULL;

	f_sweep_4_L9_E4 = FALSE;
	//std::string sweep_4_L9_E4_fname;
	sweep_4_L9_E4_surface_description = NULL;

	f_six_arcs_not_on_conic = FALSE;
	f_filter_by_nb_Eckardt_points = FALSE;
	nb_Eckardt_points = 0;


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

	f_lines_on_point_but_within_a_plane = FALSE;
	lines_on_point_but_within_a_plane_point_rk = 0;
	lines_on_point_but_within_a_plane_plane_rk = 0;

	f_rank_lines_in_PG = FALSE;
	//rank_lines_in_PG_label;

	f_unrank_lines_in_PG = FALSE;
	//std::string unrank_lines_in_PG_text;

	f_move_two_lines_in_hyperplane_stabilizer = FALSE;
	line1_from = 0;
	line2_from = 0;
	line1_to = 0;
	line2_to = 0;

	f_move_two_lines_in_hyperplane_stabilizer_text = FALSE;
	//std:string line1_from_text;
	//std:string line2_from_text;
	//std:string line1_to_text;
	//std:string line2_to_text;

	f_planes_through_line = FALSE;
	//std::string planes_through_line_rank;

	f_restricted_incidence_matrix = FALSE;
	restricted_incidence_matrix_type_row_objects = 0;
	restricted_incidence_matrix_type_col_objects = 0;
	std::string restricted_incidence_matrix_row_objects;
	std::string restricted_incidence_matrix_col_objects;
	//std::string restricted_incidence_matrix_file_name;

	f_make_relation = FALSE;
	make_relation_plane_rk = 0;

	f_plane_intersection_type_of_klein_image = FALSE;
	plane_intersection_type_of_klein_image_threshold = 0;
	//std::string plane_intersection_type_of_klein_image_input;

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
			map_ring_label.assign(argv[++i]);
			map_formula_label.assign(argv[++i]);
			map_parameters.assign(argv[++i]);
			if (f_v) {
				cout << "-map "
						<< map_ring_label << " "
						<< map_formula_label << " "
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

		// cubic surfaces:
		else if (ST.stringcmp(argv[i], "-classify_surfaces_with_double_sixes") == 0) {
			f_classify_surfaces_with_double_sixes = TRUE;
			classify_surfaces_with_double_sixes_label.assign(argv[++i]);
			classify_surfaces_with_double_sixes_control_label.assign(argv[++i]);

			if (f_v) {
				cout << "-classify_surfaces_with_double_sixes "
						<< classify_surfaces_with_double_sixes_label
						<< " " << classify_surfaces_with_double_sixes_control_label << endl;
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

		else if (ST.stringcmp(argv[i], "-sweep") == 0) {
			f_sweep = TRUE;
			sweep_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-sweep " << sweep_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-sweep_4_15_lines") == 0) {
			f_sweep_4_15_lines = TRUE;
			sweep_4_15_lines_fname.assign(argv[++i]);
			sweep_4_15_lines_surface_description = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
			if (f_v) {
				cout << "-sweep_4_15_lines" << endl;
			}
			i += sweep_4_15_lines_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -sweep_4_15_lines" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-sweep_4_15_lines " << sweep_4_15_lines_fname << endl;
				sweep_4_15_lines_surface_description->print();
			}
		}

		else if (ST.stringcmp(argv[i], "-sweep_F_beta_9_lines") == 0) {
			f_sweep_F_beta_9_lines = TRUE;
			sweep_F_beta_9_lines_fname.assign(argv[++i]);
			sweep_F_beta_9_lines_surface_description = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
			if (f_v) {
				cout << "-sweep_F_beta_9_lines" << endl;
			}
			i += sweep_F_beta_9_lines_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -sweep_F_beta_9_lines" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-sweep_F_beta_9_lines " << sweep_F_beta_9_lines_fname << endl;
				sweep_F_beta_9_lines_surface_description->print();
			}
		}


		else if (ST.stringcmp(argv[i], "-sweep_6_9_lines") == 0) {
			f_sweep_6_9_lines = TRUE;
			sweep_6_9_lines_fname.assign(argv[++i]);
			sweep_6_9_lines_surface_description = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
			if (f_v) {
				cout << "-sweep_6_9_lines" << endl;
			}
			i += sweep_6_9_lines_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -sweep_6_9_lines" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-sweep_6_9_lines " << sweep_6_9_lines_fname << endl;
				sweep_6_9_lines_surface_description->print();
			}
		}

		else if (ST.stringcmp(argv[i], "-sweep_4_27") == 0) {
			f_sweep_4_27 = TRUE;
			sweep_4_27_fname.assign(argv[++i]);
			sweep_4_27_surface_description = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
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

		else if (ST.stringcmp(argv[i], "-sweep_4_L9_E4") == 0) {
			f_sweep_4_L9_E4 = TRUE;
			sweep_4_L9_E4_fname.assign(argv[++i]);
			sweep_4_L9_E4_surface_description = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description);
			if (f_v) {
				cout << "-sweep_4_L9_E4" << endl;
			}
			i += sweep_4_L9_E4_surface_description->read_arguments(
					argc - (i + 1), argv + i + 1,
					verbose_level);
			if (f_v) {
				cout << "done with -sweep_4_L9_E4" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-sweep_4_L9_E4 " << sweep_4_L9_E4_fname << endl;
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

		else if (ST.stringcmp(argv[i], "-trihedra1_control") == 0) {
			f_trihedra1_control = TRUE;
			Trihedra1_control = NEW_OBJECT(poset_classification::poset_classification_control);
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
			Trihedra2_control = NEW_OBJECT(poset_classification::poset_classification_control);
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
			Control_six_arcs_label.assign(argv[++i]);

			if (f_v) {
				cout << "-control_six_arcs " << Control_six_arcs_label << endl;
			}
		}

		// semifields
		else if (ST.stringcmp(argv[i], "-classify_semifields") == 0) {
			f_classify_semifields = TRUE;
			Semifield_classify_description = NEW_OBJECT(semifields::semifield_classify_description);
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
			Semifield_classify_Control = NEW_OBJECT(poset_classification::poset_classification_control);
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
			Arc_generator_description = NEW_OBJECT(apps_geometry::arc_generator_description);
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
			Arc_generator_description = NEW_OBJECT(apps_geometry::arc_generator_description);
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
		else if (ST.stringcmp(argv[i], "-rank_lines_in_PG") == 0) {
			f_rank_lines_in_PG = TRUE;
			rank_lines_in_PG_label.assign(argv[++i]);
			if (f_v) {
				cout << "-rank_lines_in_PG " << rank_lines_in_PG_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-unrank_lines_in_PG") == 0) {
			f_unrank_lines_in_PG = TRUE;
			unrank_lines_in_PG_text.assign(argv[++i]);
			if (f_v) {
				cout << "-unrank_lines_in_PG " << unrank_lines_in_PG_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer") == 0) {
			f_move_two_lines_in_hyperplane_stabilizer = TRUE;
			line1_from = ST.strtoi(argv[++i]);
			line2_from = ST.strtoi(argv[++i]);
			line1_to = ST.strtoi(argv[++i]);
			line2_to = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-move_two_lines_in_hyperplane_stabilizer"
					<< " " << line1_from
					<< " " << line1_from
					<< " " << line1_to
					<< " " << line2_to
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer_text") == 0) {
			f_move_two_lines_in_hyperplane_stabilizer_text = TRUE;
			line1_from_text.assign(argv[++i]);
			line2_from_text.assign(argv[++i]);
			line1_to_text.assign(argv[++i]);
			line2_to_text.assign(argv[++i]);
			if (f_v) {
				cout << "-move_two_lines_in_hyperplane_stabilizer_text"
					<< " " << line1_from_text
					<< " " << line2_from_text
					<< " " << line1_to_text
					<< " " << line2_to_text
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-planes_through_line") == 0) {
			f_planes_through_line = TRUE;
			planes_through_line_rank.assign(argv[++i]);
			if (f_v) {
				cout << "-planes_through_line"
					<< " " << planes_through_line_rank
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-restricted_incidence_matrix") == 0) {
			f_restricted_incidence_matrix = TRUE;
			restricted_incidence_matrix_type_row_objects = ST.strtoi(argv[++i]);
			restricted_incidence_matrix_type_col_objects = ST.strtoi(argv[++i]);
			restricted_incidence_matrix_row_objects.assign(argv[++i]);
			restricted_incidence_matrix_col_objects.assign(argv[++i]);
			restricted_incidence_matrix_file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-restricted_incidence_matrix"
						<< " " << restricted_incidence_matrix_type_row_objects
						<< " " << restricted_incidence_matrix_type_col_objects
						<< " " << restricted_incidence_matrix_row_objects
						<< " " << restricted_incidence_matrix_col_objects
						<< " " << restricted_incidence_matrix_file_name
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-make_relation") == 0) {
			f_make_relation = TRUE;
			make_relation_plane_rk = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-make_relation " << make_relation_plane_rk << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-plane_intersection_type_of_klein_image") == 0) {
			f_plane_intersection_type_of_klein_image = TRUE;
			plane_intersection_type_of_klein_image_threshold = ST.strtoi(argv[++i]);
			plane_intersection_type_of_klein_image_input.assign(argv[++i]);
			if (f_v) {
				cout << "-plane_intersection_type_of_klein_image " << plane_intersection_type_of_klein_image_threshold << " " << plane_intersection_type_of_klein_image_input << endl;
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
				<< map_ring_label << " "
				<< map_formula_label << " "
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


	if (f_table_of_quartic_curves) {
		cout << "-table_of_quartic_curves " << endl;
	}
	if (f_table_of_cubic_surfaces) {
		cout << "-table_of_cubic_surfaces " << endl;
	}

	// cubic surfaces:
	if (f_classify_surfaces_with_double_sixes) {
		cout << "-classify_surfaces_with_double_sixes "
				<< classify_surfaces_with_double_sixes_label
				<< " " << classify_surfaces_with_double_sixes_control_label << endl;
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

	if (f_sweep_4_15_lines) {
		cout << "-sweep_4_15_lines " << sweep_4_15_lines_fname << endl;
		sweep_4_15_lines_surface_description->print();
	}

	if (f_sweep_F_beta_9_lines) {
		cout << "-sweep_F_beta_9_lines " << sweep_F_beta_9_lines_fname << endl;
		sweep_F_beta_9_lines_surface_description->print();
	}

	if (f_sweep_6_9_lines) {
		cout << "-sweep_6_9_lines " << sweep_6_9_lines_fname << endl;
		sweep_6_9_lines_surface_description->print();
	}

	if (f_sweep_4_27) {
		cout << "-sweep_4_27 " << sweep_4_27_fname << endl;
	}

	if (f_sweep_4_L9_E4) {
		cout << "-sweep_4_L9_E4 " << sweep_4_L9_E4_fname << endl;
	}

	if (f_six_arcs_not_on_conic) {
		cout << "-six_arcs_not_on_conic" << endl;
	}
	if (f_filter_by_nb_Eckardt_points) {
		cout << "-filter_by_nb_Eckardt_points " << nb_Eckardt_points << endl;
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
		cout << "-control_six_arcs " << Control_six_arcs_label << endl;
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
	if (f_lines_on_point_but_within_a_plane) {
		cout << "-lines_on_point_but_within_a_plane "
				<< " " << lines_on_point_but_within_a_plane_point_rk
				<< " " << lines_on_point_but_within_a_plane_plane_rk
				<< endl;
	}
	if (f_rank_lines_in_PG) {
		cout << "-rank_lines_in_PG " << rank_lines_in_PG_label << endl;
	}

	if (f_unrank_lines_in_PG) {
		cout << "-unrank_lines_in_PG " << unrank_lines_in_PG_text << endl;
	}
	if (f_move_two_lines_in_hyperplane_stabilizer) {
		cout << "-move_two_lines_in_hyperplane_stabilizer"
				<< " " << line1_from
				<< " " << line1_from
				<< " " << line1_to
				<< " " << line2_to
				<< endl;
	}
	if (f_move_two_lines_in_hyperplane_stabilizer_text) {
		cout << "-move_two_lines_in_hyperplane_stabilizer_text"
				<< " " << line1_from_text
				<< " " << line2_from_text
				<< " " << line1_to_text
				<< " " << line2_to_text
				<< endl;
	}
	if (f_planes_through_line) {
		cout << "-planes_through_line"
			<< " " << planes_through_line_rank
			<< endl;
	}
	if (f_restricted_incidence_matrix) {
		cout << "-restricted_incidence_matrix"
				<< " " << restricted_incidence_matrix_type_row_objects
				<< " " << restricted_incidence_matrix_type_col_objects
				<< " " << restricted_incidence_matrix_row_objects
				<< " " << restricted_incidence_matrix_col_objects
				<< " " << restricted_incidence_matrix_file_name
			<< endl;
	}
	if (f_make_relation) {
		cout << "-make_relation " << make_relation_plane_rk << endl;
	}
	if (f_plane_intersection_type_of_klein_image) {
		cout << "-plane_intersection_type_of_klein_image " << plane_intersection_type_of_klein_image_threshold << " " << plane_intersection_type_of_klein_image_input << endl;
	}


}



}}}

