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
	f_save = false;

	f_save_as = false;
	//std::string save_as_fname;

	f_extract_subset = false;
	//std::string extract_subset_set;
	//std::string extract_subset_fname;

	f_line_type_old = false;

	f_line_type = false;

	f_conic_type = false;
	conic_type_threshold = 0;

	f_non_conical_type = false;

	f_ideal = false;
	//std::string ideal_ring_label;
	//ideal_degree = 0;


	f_canonical_form_PG = false;
	//std::string &canonical_form_PG_PG_label;
	f_canonical_form_PG_has_PA = false;
	Canonical_form_PG_PA = NULL;
	Canonical_form_PG_Descr = NULL;

	f_canonical_form = false;
	Canonical_form_Descr = NULL;

	f_report = false;
	Objects_report_options = NULL;

	f_draw_incidence_matrices = false;
	//std::string draw_incidence_matrices_prefix;

	f_test_distinguishing_property = false;
	//test_distinguishing_property_graph

	f_covering_type = 0;
	//std::string covering_type_orbits;
	covering_type_size = 0;

	f_filter_by_Steiner_property = false;

	f_compute_frequency = false;
	//std::string compute_frequency_graph;

	f_unpack_from_restricted_action = false;
	//std::string unpack_from_restricted_action_prefix;
	//std::string unpack_from_restricted_action_group_label;

	f_line_covering_type = false;
	//std::string line_covering_type_prefix;
	//std::string line_covering_type_projective_space;
	//std::string line_covering_type_lines;

	f_activity = false;
	Activity_description = NULL;

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
			f_save = true;
			if (f_v) {
				cout << "-save " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_as") == 0) {
			f_save_as = true;
			save_as_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-save_as " << save_as_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_subset") == 0) {
			f_extract_subset = true;
			extract_subset_set.assign(argv[++i]);
			extract_subset_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_subset " << extract_subset_set << " " << extract_subset_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-line_type_old") == 0) {
			f_line_type_old = true;
			if (f_v) {
				cout << "-line_type_old " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-line_type") == 0) {
			f_line_type = true;
			if (f_v) {
				cout << "-line_type " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = true;
			conic_type_threshold = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-conic_type " << conic_type_threshold << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-non_conical_type") == 0) {
			f_non_conical_type = true;
			if (f_v) {
				cout << "-non_conical_type " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ideal") == 0) {
			f_ideal = true;
			ideal_ring_label.assign(argv[++i]);
			//ideal_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-ideal " << ideal_ring_label << endl;
			}
		}

		// activities for IS:

		else if (ST.stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = true;
			if (f_v) {
				cout << "-canonical_form_PG, reading extra arguments" << endl;
			}

			canonical_form_PG_PG_label.assign(argv[++i]);

			Canonical_form_PG_Descr = NEW_OBJECT(canonical_form_classification::classification_of_objects_description);

			i += Canonical_form_PG_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -canonical_form_PG " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				Canonical_form_PG_Descr->print();
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
				cout << "done reading -canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;

			Objects_report_options = NEW_OBJECT(canonical_form_classification::objects_report_options);
			i += Objects_report_options->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
		}

		else if (ST.stringcmp(argv[i], "-draw_incidence_matrices") == 0) {
			f_draw_incidence_matrices = true;
			draw_incidence_matrices_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-draw_incidence_matrices " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-test_distinguishing_property") == 0) {
			f_test_distinguishing_property = true;
			test_distinguishing_property_graph.assign(argv[++i]);
			if (f_v) {
				cout << "-test_distinguishing_property " << test_distinguishing_property_graph << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-covering_type") == 0) {
			f_covering_type = true;
			covering_type_orbits.assign(argv[++i]);
			covering_type_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-covering_type " << covering_type_orbits << " " << covering_type_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-filter_by_Steiner_property") == 0) {
			f_filter_by_Steiner_property = true;
			if (f_v) {
				cout << "-filter_by_Steiner_property " << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-compute_frequency") == 0) {
			f_compute_frequency = true;
			compute_frequency_graph.assign(argv[++i]);
			if (f_v) {
				cout << "-compute_frequency " << compute_frequency_graph << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-unpack_from_restricted_action") == 0) {
			f_unpack_from_restricted_action = true;
			unpack_from_restricted_action_prefix.assign(argv[++i]);
			unpack_from_restricted_action_group_label.assign(argv[++i]);
			if (f_v) {
				cout << "-unpack_from_restricted_action " << unpack_from_restricted_action_prefix
						<< " " << unpack_from_restricted_action_group_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-line_covering_type") == 0) {
			f_line_covering_type = true;
			line_covering_type_prefix.assign(argv[++i]);
			line_covering_type_projective_space.assign(argv[++i]);
			line_covering_type_lines.assign(argv[++i]);
			if (f_v) {
				cout << "-line_covering_type " << line_covering_type_prefix
						<< " " << line_covering_type_projective_space
						<< " " << line_covering_type_lines
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-activity") == 0) {
			i++;

			f_activity = true;
			Activity_description = NEW_OBJECT(user_interface::activity_description);

			Activity_description->read_arguments(argc, argv, i, verbose_level);
			break;
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
	if (f_line_type_old) {
		cout << "-line_type_old " << endl;
	}
	if (f_line_type) {
		cout << "-line_type " << endl;
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
		Objects_report_options->print();
	}
	if (f_draw_incidence_matrices) {
		cout << "-draw_incidence_matrices " << draw_incidence_matrices_prefix << endl;
	}
	if (f_test_distinguishing_property) {
		cout << "-test_distinguishing_property " << test_distinguishing_property_graph << endl;
	}
	if (f_covering_type) {
		cout << "-covering_type " << covering_type_orbits << " " << covering_type_size << endl;
	}
	if (f_filter_by_Steiner_property) {
		cout << "-filter_by_Steiner_property " << endl;
	}
	if (f_compute_frequency) {
		cout << "-compute_frequency " << compute_frequency_graph << endl;
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

