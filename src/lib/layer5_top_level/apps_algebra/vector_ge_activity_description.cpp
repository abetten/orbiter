/*
 * vector_ge_activity_description.cpp
 *
 *  Created on: Dec 24, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {





vector_ge_activity_description::vector_ge_activity_description()
{
	Record_birth();

	f_report = false;

	f_report_with_options = false;
	//std::string report_options;

	f_report_elements_coded = false;

	f_export_GAP = false;

	f_transform_variety = false;
	//std::string transform_variety_label;

	f_multiply = false;

	f_conjugate = false;

	f_conjugate_inverse = false;

	f_select_subset = false;
	//std::string select_subset_vector_label;

	f_field_reduction = false;
	field_reduction_subfield_index = 0;

	f_rational_canonical_form = false;

	f_products_of_pairs = false;

	f_order_of_products_of_pairs = false;

	f_apply_isomorphism_wedge_product_4to6 = false;

}


vector_ge_activity_description::~vector_ge_activity_description()
{
	Record_death();
}


int vector_ge_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "vector_ge_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_with_options") == 0) {
			f_report_with_options = true;
			report_options.assign(argv[++i]);
			if (f_v) {
				cout << "-report " << report_options << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_elements_coded") == 0) {
			f_report_elements_coded = true;
			if (f_v) {
				cout << "-report_elements_coded " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_GAP") == 0) {
			f_export_GAP = true;
			if (f_v) {
				cout << "-export_GAP " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-transform_variety") == 0) {
			f_transform_variety = true;
			transform_variety_label.assign(argv[++i]);
			if (f_v) {
				cout << "-transform_variety " << transform_variety_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply") == 0) {
			f_multiply = true;
			if (f_v) {
				cout << "-multiply " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conjugate") == 0) {
			f_conjugate = true;
			if (f_v) {
				cout << "-conjugate " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conjugate_inverse") == 0) {
			f_conjugate_inverse = true;
			if (f_v) {
				cout << "-conjugate_inverse " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select_subset") == 0) {
			f_select_subset = true;
			select_subset_vector_label.assign(argv[++i]);
			if (f_v) {
				cout << "-select_subset " << select_subset_vector_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-field_reduction") == 0) {
			f_field_reduction = true;
			field_reduction_subfield_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-field_reduction " << field_reduction_subfield_index << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rational_canonical_form") == 0) {
			f_rational_canonical_form = true;
			if (f_v) {
				cout << "-rational_canonical_form " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-products_of_pairs") == 0) {
			f_products_of_pairs = true;
			if (f_v) {
				cout << "-products_of_pairs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-order_of_products_of_pairs") == 0) {
			f_order_of_products_of_pairs = true;
			if (f_v) {
				cout << "-order_of_products_of_pairs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-apply_isomorphism_wedge_product_4to6") == 0) {
			f_apply_isomorphism_wedge_product_4to6 = true;
			if (f_v) {
				cout << "-apply_isomorphism_wedge_product_4to6 " << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "vector_ge_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "vector_ge_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void vector_ge_activity_description::print()
{

	if (f_report) {
		cout << "-report " << report_options << endl;
	}
	if (f_report_with_options) {
		cout << "-report " << report_options << endl;
	}
	if (f_report_elements_coded) {
		cout << "-report_elements_coded " << endl;
	}
	if (f_export_GAP) {
		cout << "-export_GAP " << endl;
	}
	if (f_transform_variety) {
		cout << "-transform_variety " << transform_variety_label << endl;
	}
	if (f_multiply) {
		cout << "-multiply " << endl;
	}
	if (f_conjugate) {
		cout << "-conjugate " << endl;
	}
	if (f_conjugate_inverse) {
		cout << "-conjugate_inverse " << endl;
	}
	if (f_select_subset) {
		cout << "-select_subset " << select_subset_vector_label << endl;
	}
	if (f_field_reduction) {
		cout << "-field_reduction " << field_reduction_subfield_index << endl;
	}
	if (f_rational_canonical_form) {
		cout << "-rational_canonical_form " << endl;
	}
	if (f_products_of_pairs) {
		cout << "-products_of_pairs " << endl;
	}
	if (f_order_of_products_of_pairs) {
		cout << "-order_of_products_of_pairs " << endl;
	}
	if (f_apply_isomorphism_wedge_product_4to6) {
		cout << "-apply_isomorphism_wedge_product_4to6 " << endl;
	}

}





}}}

