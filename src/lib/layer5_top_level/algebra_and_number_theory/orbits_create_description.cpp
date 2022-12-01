/*
 * orbits_create_description.cpp
 *
 *  Created on: Nov 5, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbits_create_description::orbits_create_description()
{
	f_group = FALSE;
	//std::string group_label;

	f_on_points = FALSE;

	f_on_subsets = FALSE;
	on_subsets_size = 0;
	//std::string on_subsets_poset_classification_control_label;

	f_on_subspaces = FALSE;
	on_subspaces_dimension = 0;
	//std::string on_subspaces_poset_classification_control_label;

	f_on_tensors = FALSE;
	on_tensors_dimension = 0;
	//std::string on_tensors_poset_classification_control_label;

	f_on_partition = FALSE;
	on_partition_k = 0;
	//std::string on_partition_poset_classification_control_label;

	f_on_polynomials = FALSE;
	on_polynomials_degree = 0;

#if 0
	f_draw_tree = FALSE;
	draw_tree_idx = 0;

	f_recognize = FALSE;
	//std::string recognize_text;
#endif
}

orbits_create_description::~orbits_create_description()
{
}

int orbits_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	cout << "design_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-group") == 0) {
			f_group = TRUE;
			group_label.assign(argv[++i]);
			if (f_v) {
				cout << "-group " << group_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_points") == 0) {
			f_on_points = TRUE;
			if (f_v) {
				cout << "-on_points" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_subsets") == 0) {
			f_on_subsets = TRUE;
			on_subsets_size = ST.strtoi(argv[++i]);
			on_subsets_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_subsets " << on_subsets_size << " " << on_subsets_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_subspaces") == 0) {
			f_on_subspaces = TRUE;
			on_subspaces_dimension = ST.strtoi(argv[++i]);
			on_subspaces_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_subspaces " << on_subspaces_dimension << " "
						<< on_subspaces_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_tensors") == 0) {
			f_on_tensors = TRUE;
			on_tensors_dimension = ST.strtoi(argv[++i]);
			on_tensors_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_tensors " << on_tensors_dimension << " "
						<< on_tensors_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_partition") == 0) {
			f_on_partition = TRUE;
			on_partition_k = ST.strtoi(argv[++i]);
			on_partition_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_partition " << on_partition_k << " "
						<< on_partition_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_polynomials") == 0) {
			f_on_polynomials = TRUE;
			on_polynomials_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-on_polynomials " << on_polynomials_degree << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-draw_tree") == 0) {
			f_draw_tree = TRUE;
			draw_tree_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-draw_tree " << draw_tree_idx << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-recognize") == 0) {
			f_recognize = TRUE;
			recognize_text = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-recognize " << recognize_text << endl;
			}
		}
#endif

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "orbits_create_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "orbits_create_description::read_arguments done" << endl;
	return i + 1;
}


void orbits_create_description::print()
{
	if (f_group) {
		cout << "-group " << group_label << endl;
	}
	if (f_on_points) {
		cout << "-on_points" << endl;
	}
	if (f_on_subsets) {
		cout << "-on_subsets " << on_subsets_size << " "
				<< on_subsets_poset_classification_control_label << endl;
	}
	if (f_on_subspaces) {
		cout << "-on_subspaces " << on_subspaces_dimension << " "
				<< on_subspaces_poset_classification_control_label << endl;
	}
	if (f_on_tensors) {
		cout << "-on_tensors " << on_tensors_dimension << " "
				<< on_tensors_poset_classification_control_label << endl;
	}
	if (f_on_partition) {
		cout << "-on_partition " << on_partition_k << " "
				<< on_partition_poset_classification_control_label << endl;
	}
	if (f_on_polynomials) {
		cout << "-on_polynomials " << on_polynomials_degree << endl;
	}
#if 0
	if (f_recognize) {
		cout << "-recognize " << recognize_text << endl;
	}
#endif
}


}}}




