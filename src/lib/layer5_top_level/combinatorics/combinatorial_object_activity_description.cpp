/*
 * combinatorial_object_activity_description.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combinatorial_object_activity_description::combinatorial_object_activity_description()
{
	f_save = FALSE;

	f_save_as = FALSE;
	//std::string save_as_fname;

	f_extract_subset = FALSE;
	//std::string extract_subset_set;
	//std::string extract_subset_fname;

	f_line_type = FALSE;
	//std::string line_type_projective_space_label;
	//std::string line_type_prefix;

	f_conic_type = FALSE;
	conic_type_threshold = 0;

	f_non_conical_type = FALSE;

	f_ideal = FALSE;
	//std::string ideal_ring_label;
	//ideal_degree = 0;


	f_canonical_form_PG = FALSE;
	//std::string &canonical_form_PG_PG_label;
	f_canonical_form_PG_has_PA = FALSE;
	Canonical_form_PG_PA = NULL;
	Canonical_form_PG_Descr = NULL;

	f_canonical_form = FALSE;
	Canonical_form_Descr = NULL;

	f_report = FALSE;
	Classification_of_objects_report_options = NULL;

	f_draw_incidence_matrices = FALSE;
	//std::string draw_incidence_matrices_prefix;

	f_test_distinguishing_property = FALSE;
	//test_distinguishing_property_graph

	f_unpack_from_restricted_action = FALSE;
	//std::string unpack_from_restricted_action_prefix;
	//std::string unpack_from_restricted_action_group_label;

	f_line_covering_type = FALSE;
	//std::string line_covering_type_prefix;
	//std::string line_covering_type_projective_space;
	//std::string line_covering_type_lines;


}

combinatorial_object_activity_description::~combinatorial_object_activity_description()
{
}


int combinatorial_object_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "combinatorial_object_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		// activities for COC:


		if (ST.stringcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			if (f_v) {
				cout << "-save " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_as") == 0) {
			f_save_as = TRUE;
			save_as_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-save_as " << save_as_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_subset") == 0) {
			f_extract_subset = TRUE;
			extract_subset_set.assign(argv[++i]);
			extract_subset_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_subset " << extract_subset_set << " " << extract_subset_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-line_type") == 0) {
			f_line_type = TRUE;
			line_type_projective_space_label.assign(argv[++i]);
			line_type_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-line_type " << line_type_projective_space_label << " " << line_type_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			conic_type_threshold = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-conic_type " << conic_type_threshold << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-non_conical_type") == 0) {
			f_non_conical_type = TRUE;
			if (f_v) {
				cout << "-non_conical_type " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ideal") == 0) {
			f_ideal = TRUE;
			ideal_ring_label.assign(argv[++i]);
			//ideal_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-ideal " << ideal_ring_label << endl;
			}
		}

		// activities for IS:

		else if (ST.stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			if (f_v) {
				cout << "-canonical_form_PG, reading extra arguments" << endl;
			}

			canonical_form_PG_PG_label.assign(argv[++i]);

			Canonical_form_PG_Descr = NEW_OBJECT(combinatorics::classification_of_objects_description);

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

		else if (ST.stringcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			if (f_v) {
				cout << "-canonical_form, reading extra arguments" << endl;
			}

			Canonical_form_Descr = NEW_OBJECT(combinatorics::classification_of_objects_description);

			i += Canonical_form_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;

			Classification_of_objects_report_options = NEW_OBJECT(combinatorics::classification_of_objects_report_options);
			i += Classification_of_objects_report_options->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
		}
		else if (ST.stringcmp(argv[i], "-draw_incidence_matrices") == 0) {
			f_draw_incidence_matrices = TRUE;
			draw_incidence_matrices_prefix.assign(argv[++i]);

		}
		else if (ST.stringcmp(argv[i], "-test_distinguishing_property") == 0) {
			f_test_distinguishing_property = TRUE;
			test_distinguishing_property_graph.assign(argv[++i]);
			cout << "-test_distinguishing_property " << test_distinguishing_property_graph << endl;
		}
		else if (ST.stringcmp(argv[i], "-unpack_from_restricted_action") == 0) {
			f_unpack_from_restricted_action = TRUE;
			unpack_from_restricted_action_prefix.assign(argv[++i]);
			unpack_from_restricted_action_group_label.assign(argv[++i]);
			cout << "-unpack_from_restricted_action " << unpack_from_restricted_action_prefix
					<< " " << unpack_from_restricted_action_group_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-line_covering_type") == 0) {
			f_line_covering_type = TRUE;
			line_covering_type_prefix.assign(argv[++i]);
			line_covering_type_projective_space.assign(argv[++i]);
			line_covering_type_lines.assign(argv[++i]);
			cout << "-line_covering_type " << line_covering_type_prefix
					<< " " << line_covering_type_projective_space
					<< " " << line_covering_type_lines
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "combinatorial_object_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "combinatorial_object_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void combinatorial_object_activity_description::print()
{
	if (f_save) {
		cout << "-save " << endl;
	}
	if (f_save_as) {
		cout << "-save_as " << save_as_fname << endl;
	}
	if (f_extract_subset) {
		cout << "-extract_subset " << extract_subset_set << " " << extract_subset_fname << endl;
	}
	if (f_line_type) {
		cout << "-line_type " << line_type_projective_space_label << " " << line_type_prefix << endl;
	}
	if (f_conic_type) {
		cout << "-conic_type " << conic_type_threshold << endl;
	}
	if (f_non_conical_type) {
		cout << "-f_non_conical_type " << endl;
	}
	if (f_ideal) {
		cout << "-ideal " << ideal_ring_label << endl;
	}
	if (f_canonical_form_PG) {
		cout << "-canonical_form_PG " << canonical_form_PG_PG_label << endl;
		Canonical_form_PG_Descr->print();
	}
	if (f_canonical_form) {
		cout << "-canonical_form " << endl;
		Canonical_form_Descr->print();
	}
	if (f_report) {
		cout << "-report " << endl;
		Classification_of_objects_report_options->print();
	}
	if (f_draw_incidence_matrices) {
		cout << "-draw_incidence_matrices " << draw_incidence_matrices_prefix << endl;
	}
	if (f_test_distinguishing_property) {
		cout << "-test_distinguishing_property " << test_distinguishing_property_graph << endl;
	}
	if (f_unpack_from_restricted_action) {
		cout << "-unpack_from_restricted_action " << unpack_from_restricted_action_prefix
				<< " " << unpack_from_restricted_action_group_label << endl;
	}
	if (f_line_covering_type) {
		cout << "-line_covering_type " << line_covering_type_prefix
				<< " " << line_covering_type_projective_space
				<< " " << line_covering_type_lines
				<< endl;
	}
}



}}}

