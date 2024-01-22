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
	f_group = false;
	//std::string group_label;

	f_on_points = false;

	f_on_points_with_generators = false;
	//std::string on_points_with_generators_gens_label;

	f_on_subsets = false;
	on_subsets_size = 0;
	//std::string on_subsets_poset_classification_control_label;

	f_of_one_subset = false;
	//std::string of_one_subset_label;

	f_on_subspaces = false;
	on_subspaces_dimension = 0;
	//std::string on_subspaces_poset_classification_control_label;

	f_on_tensors = false;
	on_tensors_dimension = 0;
	//std::string on_tensors_poset_classification_control_label;

	f_on_partition = false;
	on_partition_k = 0;
	//std::string on_partition_poset_classification_control_label;

	f_on_polynomials = false;
	//std::string on_polynomials_ring;

	f_of_one_polynomial = false;
	//std::string of_one_polynomial_ring;
	//std::string of_one_polynomial_equation;

	f_on_cubic_curves = false;
	//std::string on_cubic_curves_control;

	f_on_cubic_surfaces = false;
	//std::string on_cubic_surfaces_PA;
	//std::string on_cubic_surfaces_control;

	f_classification_by_canonical_form = false;
	Canonical_form_classifier_description = NULL;

	f_override_generators = false;
	//std::string override_generators_label;

}

orbits_create_description::~orbits_create_description()
{
}

int orbits_create_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	cout << "design_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-group") == 0) {
			f_group = true;
			group_label.assign(argv[++i]);
			if (f_v) {
				cout << "-group " << group_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_points") == 0) {
			f_on_points = true;
			if (f_v) {
				cout << "-on_points" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_points_with_generators") == 0) {
			f_on_points_with_generators = true;
			on_points_with_generators_gens_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_points_with_generators " << on_points_with_generators_gens_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-on_subsets") == 0) {
			f_on_subsets = true;
			on_subsets_size = ST.strtoi(argv[++i]);
			on_subsets_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_subsets " << on_subsets_size << " "
						<< on_subsets_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-of_one_subset") == 0) {
			f_of_one_subset = true;
			of_one_subset_label.assign(argv[++i]);
			if (f_v) {
				cout << "-of_one_subset " << of_one_subset_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-on_subspaces") == 0) {
			f_on_subspaces = true;
			on_subspaces_dimension = ST.strtoi(argv[++i]);
			on_subspaces_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_subspaces " << on_subspaces_dimension << " "
						<< on_subspaces_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_tensors") == 0) {
			f_on_tensors = true;
			on_tensors_dimension = ST.strtoi(argv[++i]);
			on_tensors_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_tensors " << on_tensors_dimension << " "
						<< on_tensors_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_partition") == 0) {
			f_on_partition = true;
			on_partition_k = ST.strtoi(argv[++i]);
			on_partition_poset_classification_control_label.assign(argv[++i]);
			if (f_v) {
				cout << "-on_partition " << on_partition_k << " "
						<< on_partition_poset_classification_control_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_polynomials") == 0) {
			f_on_polynomials = true;
			on_polynomials_ring.assign(argv[++i]);
			if (f_v) {
				cout << "-on_polynomials " << on_polynomials_ring << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-of_one_polynomial") == 0) {
			f_of_one_polynomial = true;
			of_one_polynomial_ring.assign(argv[++i]);
			of_one_polynomial_equation.assign(argv[++i]);
			if (f_v) {
				cout << "-of_one_polynomial " << of_one_polynomial_ring
						<< " " << of_one_polynomial_equation << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_cubic_curves") == 0) {
			f_on_cubic_curves = true;
			on_cubic_curves_control.assign(argv[++i]);
			if (f_v) {
				cout << "-on_cubic_curves " << on_cubic_curves_control << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_cubic_surfaces") == 0) {
			f_on_cubic_surfaces = true;
			on_cubic_surfaces_PA.assign(argv[++i]);
			on_cubic_surfaces_control.assign(argv[++i]);
			if (f_v) {
				cout << "-on_cubic_surfaces " << on_cubic_surfaces_PA
						<< " " << on_cubic_surfaces_control << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-classification_by_canonical_form") == 0) {
			f_classification_by_canonical_form = true;
			Canonical_form_classifier_description = NEW_OBJECT(canonical_form::canonical_form_classifier_description);
			if (f_v) {
				cout << "-classification_by_canonical_form" << endl;
			}
			i += Canonical_form_classifier_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -classification_by_canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			if (f_v) {
				cout << "-classification_by_canonical_form " << endl;
				Canonical_form_classifier_description->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-override_generators") == 0) {
			f_override_generators = true;
			override_generators_label = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-override_generators " << override_generators_label << endl;
			}
		}


	#if 0
		else if (ST.stringcmp(argv[i], "-draw_tree") == 0) {
			f_draw_tree = true;
			draw_tree_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-draw_tree " << draw_tree_idx << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-recognize") == 0) {
			f_recognize = true;
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
	if (f_of_one_subset) {
		cout << "-of_one_subset " << of_one_subset_label << endl;
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
		cout << "-on_polynomials " << on_polynomials_ring << endl;
	}
	if (f_of_one_polynomial) {
		cout << "-of_one_polynomial " << of_one_polynomial_ring
				<< " " << of_one_polynomial_equation << endl;
	}
	if (f_on_cubic_curves) {
		cout << "-on_cubic_curves " << on_cubic_curves_control << endl;
	}
	if (f_on_cubic_surfaces) {
		cout << "-on_cubic_surfaces " << on_cubic_surfaces_PA
				<< " " << on_cubic_surfaces_control << endl;
	}
	if (f_classification_by_canonical_form) {
		cout << "-classification_by_canonical_form " << endl;
		Canonical_form_classifier_description->print();
	}
	if (f_override_generators) {
		cout << "-override_generators " << override_generators_label << endl;
	}
#if 0
	if (f_recognize) {
		cout << "-recognize " << recognize_text << endl;
	}
#endif
}


}}}




