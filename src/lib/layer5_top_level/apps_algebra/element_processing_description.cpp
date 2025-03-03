/*
 * element_processing_description.cpp
 *
 *  Created on: Dec 23, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


element_processing_description::element_processing_description()
{
	Record_birth();
	f_input = false;
	//std::string input_label;

	f_print = false;

	f_apply_isomorphism_wedge_product_4to6 = false;

	f_with_permutation = false;

	f_with_fix_structure = false;

	f_order_of_products_of_pairs = false;

	f_products_of_pairs = false;

	f_conjugate = false;
	//std::string conjugate_data;

	f_print_action_on_surface = false;
	//std::string print_action_on_surface_label;


}

element_processing_description::~element_processing_description()
{
	Record_death();
}

int element_processing_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "element_processing_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-input") == 0) {
			f_input = true;
			input_label.assign(argv[++i]);
			if (f_v) {
				cout << "-input " << input_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print") == 0) {
			f_print = true;
			if (f_v) {
				cout << "-print " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-apply_isomorphism_wedge_product_4to6") == 0) {
			f_apply_isomorphism_wedge_product_4to6 = true;
			if (f_v) {
				cout << "-apply_isomorphism_wedge_product_4to6 " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-with_permutation") == 0) {
			f_with_permutation = true;
			if (f_v) {
				cout << "-with_permutation " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-with_fix_structure") == 0) {
			f_with_fix_structure = true;
			if (f_v) {
				cout << "-with_fix_structure " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-order_of_products_of_pairs") == 0) {
			f_order_of_products_of_pairs = true;
			if (f_v) {
				cout << "-order_of_products_of_pairs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-products_of_pairs") == 0) {
			f_products_of_pairs = true;
			if (f_v) {
				cout << "-products_of_pairs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-conjugate") == 0) {
			f_conjugate = true;
			conjugate_data.assign(argv[++i]);
			if (f_v) {
				cout << "-conjugate " << conjugate_data << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_action_on_surface") == 0) {
			f_print_action_on_surface = true;
			print_action_on_surface_label.assign(argv[++i]);
			if (f_v) {
				cout << "-print_action_on_surface " << print_action_on_surface_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "element_processing_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "element_processing_description::read_arguments done" << endl;
	}
	return i + 1;
}


void element_processing_description::print()
{
	if (f_input) {
		cout << "-input " << input_label << endl;
	}
	if (f_print) {
		cout << "-print " << endl;
	}
	if (f_apply_isomorphism_wedge_product_4to6) {
		cout << "-apply_isomorphism_wedge_product_4to6 " << endl;
	}
	if (f_with_permutation) {
		cout << "-with_permutation " << endl;
	}
	if (f_with_fix_structure) {
		cout << "-with_fix_structure " << endl;
	}
	if (f_order_of_products_of_pairs) {
		cout << "-order_of_products_of_pairs " << endl;
	}
	if (f_products_of_pairs) {
		cout << "-products_of_pairs " << endl;
	}
	if (f_conjugate) {
		cout << "-conjugate " << conjugate_data << endl;
	}
	if (f_print_action_on_surface) {
		cout << "-print_action_on_surface " << print_action_on_surface_label << endl;
	}
}




}}}

