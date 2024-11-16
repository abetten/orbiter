/*
 * group_modification_description.cpp
 *
 *  Created on: Dec 1, 2021
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {


group_modification_description::group_modification_description()
{
	f_restricted_action = false;
	//std::string restricted_action_set_text;
	//std::string restricted_action_set_text_tex;

	f_on_k_subspaces = false;
	on_k_subspaces_k = 0;

	f_on_k_subsets = false;
	on_k_subsets_k = 0;

	f_on_wedge_product = false;

	f_on_cosets_of_subgroup = false;
	//std::string on_cosets_of_subgroup_subgroup;

	f_create_special_subgroup = false;

	f_create_even_subgroup = false;

	f_derived_subgroup = false;

	f_point_stabilizer = false;
	point_stabilizer_point = 0;

	f_set_stabilizer = false;
	//std::string set_stabilizer_the_set;
	//std::string set_stabilizer_control;

	f_projectivity_subgroup = false;

	f_subfield_subgroup = false;
	subfield_subgroup_index = 0;

	f_action_on_self_by_right_multiplication = false;

	f_direct_product = false;
	//std::string direct_product_input;
	//std::string direct_product_subgroup_order;
	//std::string direct_product_subgroup_gens;

	f_polarity_extension = false;
	//std::string polarity_extension_input;
	//std::string polarity_extension_PA;

	f_on_middle_layer_grassmannian = false;

	f_on_points_and_hyperplanes = false;

	f_holomorph = false;

	f_automorphism_group = false;

	f_subgroup_by_lattice = false;
	subgroup_by_lattice_orbit_index = -1;

	f_stabilizer_of_variety = false;
	//std::string stabilizer_of_variety_label;

	f_import = false;

	//std::vector<std::string> from;

}


group_modification_description::~group_modification_description()
{
}


int group_modification_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "group_modification_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-restricted_action") == 0) {
			f_restricted_action = true;
			restricted_action_set_text.assign(argv[++i]);
			restricted_action_set_text_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-restricted_action "
						<< " " << restricted_action_set_text
						<< " " << restricted_action_set_text_tex
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_k_subspaces") == 0) {
			f_on_k_subspaces = true;
			on_k_subspaces_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_k_subsets") == 0) {
			f_on_k_subsets = true;
			on_k_subsets_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-on_k_subsets " << on_k_subsets_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_wedge_product") == 0) {
			f_on_wedge_product = true;
			if (f_v) {
				cout << "-on_wedge_product " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_cosets_of_subgroup") == 0) {
			f_on_cosets_of_subgroup = true;
			on_cosets_of_subgroup_subgroup.assign(argv[++i]);
			if (f_v) {
				cout << "-on_cosets_of_subgroup " << on_cosets_of_subgroup_subgroup << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_special_subgroup") == 0) {
			f_create_special_subgroup = true;
			if (f_v) {
				cout << "-create_special_subgroup " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_even_subgroup") == 0) {
			f_create_even_subgroup = true;
			if (f_v) {
				cout << "-create_even_subgroup " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-derived_subgroup") == 0) {
			f_derived_subgroup = true;
			if (f_v) {
				cout << "-derived_subgroup " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-point_stabilizer") == 0) {
			f_point_stabilizer = true;
			point_stabilizer_point = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-point_stabilizer " << point_stabilizer_point << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-set_stabilizer") == 0) {
			f_set_stabilizer = true;
			set_stabilizer_the_set.assign(argv[++i]);
			set_stabilizer_control.assign(argv[++i]);
			if (f_v) {
				cout << "-set_stabilizer " << set_stabilizer_the_set
						<< " " << set_stabilizer_control << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-projectivity_subgroup") == 0) {
			f_projectivity_subgroup = true;
			if (f_v) {
				cout << "-projectivity_subgroup " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subfield_subgroup") == 0) {
			f_subfield_subgroup = true;
			subfield_subgroup_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subfield_subgroup " << subfield_subgroup_index << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-action_on_self_by_right_multiplication") == 0) {
			f_action_on_self_by_right_multiplication = true;
			if (f_v) {
				cout << "-action_on_self_by_right_multiplication" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-direct_product") == 0) {
			f_direct_product = true;
			direct_product_input.assign(argv[++i]);
			direct_product_subgroup_order.assign(argv[++i]);
			direct_product_subgroup_gens.assign(argv[++i]);
			if (f_v) {
				cout << "-direct_product "
						<< direct_product_input
						<< " " << direct_product_subgroup_order
						<< " " << direct_product_subgroup_gens
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polarity_extension") == 0) {
			f_polarity_extension = true;
			polarity_extension_input.assign(argv[++i]);
			polarity_extension_PA.assign(argv[++i]);
			if (f_v) {
				cout << "-polarity_extension "
						<< polarity_extension_input
						<< " " << polarity_extension_PA
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_middle_layer_grassmannian") == 0) {
			f_on_middle_layer_grassmannian = true;
			if (f_v) {
				cout << "-on_middle_layer_grassmannian" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_points_and_hyperplanes") == 0) {
			f_on_points_and_hyperplanes = true;
			if (f_v) {
				cout << "-on_points_and_hyperplanes" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-holomorph") == 0) {
			f_holomorph = true;
			if (f_v) {
				cout << "-holomorph" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-automorphism_group") == 0) {
			f_automorphism_group = true;
			if (f_v) {
				cout << "-automorphism_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_by_lattice") == 0) {
			f_subgroup_by_lattice = true;
			subgroup_by_lattice_orbit_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subgroup_by_lattice " << subgroup_by_lattice_orbit_index << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-stabilizer_of_variety") == 0) {
			f_stabilizer_of_variety = true;
			stabilizer_of_variety_label.assign(argv[++i]);
			if (f_v) {
				cout << "-stabilizer_of_variety " << stabilizer_of_variety_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-import") == 0) {
			f_import = true;
			if (f_v) {
				cout << "-import " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-from") == 0) {
			std::string from_text;
			from_text.assign(argv[++i]);
			from.push_back(from_text);
			if (f_v) {
				cout << "-from " << from_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "group_modification_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "group_modification_description::read_arguments done" << endl;
	}
	return i + 1;
}


void group_modification_description::print()
{
	if (f_restricted_action) {
		cout << "-restricted_action "
				<< " " << restricted_action_set_text
				<< " " << restricted_action_set_text_tex
				<< endl;
	}
	if (f_on_k_subspaces) {
		cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
	}
	if (f_on_k_subsets) {
		cout << "-on_k_subsets " << on_k_subsets_k << endl;
	}
	if (f_on_wedge_product) {
		cout << "-on_wedge_product " << endl;
	}
	if (f_on_cosets_of_subgroup) {
		cout << "-on_cosets_of_subgroup " << on_cosets_of_subgroup_subgroup << endl;
	}
	if (f_create_special_subgroup) {
		cout << "-create_special_subgroup " << endl;
	}
	if (f_create_even_subgroup) {
		cout << "-create_even_subgroup " << endl;
	}
	if (f_derived_subgroup) {
		cout << "-derived_subgroup " << endl;
	}
	if (f_point_stabilizer) {
		cout << "-point_stabilizer " << point_stabilizer_point << endl;
	}
	if (f_set_stabilizer) {
		cout << "-set_stabilizer " << set_stabilizer_the_set
				<< " " << set_stabilizer_control << endl;
	}
	if (f_projectivity_subgroup) {
		cout << "-projectivity_subgroup " << endl;
	}
	if (f_subfield_subgroup) {
		cout << "-subfield_subgroup " << subfield_subgroup_index << endl;
	}
	if (f_action_on_self_by_right_multiplication) {
		cout << "-action_on_self_by_right_multiplication" << endl;
	}
	if (f_direct_product) {
		cout << "-direct_product "
				<< direct_product_input
				<< " " << direct_product_subgroup_order
				<< " " << direct_product_subgroup_gens
				<< endl;
	}
	if (f_polarity_extension) {
		cout << "-polarity_extension "
				<< polarity_extension_input
				<< " " << polarity_extension_PA
				<< endl;
	}
	if (f_on_middle_layer_grassmannian) {
		cout << "-on_middle_layer_grassmannian" << endl;
	}
	if (f_on_middle_layer_grassmannian) {
		cout << "-on_points_and_hyperplanes" << endl;
	}
	if (f_holomorph) {
		cout << "-holomorph" << endl;
	}
	if (f_automorphism_group) {
		cout << "-automorphism_group" << endl;
	}
	if (f_subgroup_by_lattice) {
		cout << "-subgroup_by_lattice " << subgroup_by_lattice_orbit_index << endl;
	}
	if (f_stabilizer_of_variety) {
		cout << "-stabilizer_of_variety " << stabilizer_of_variety_label << endl;
	}
	if (f_import) {
		cout << "-import " << endl;
	}


	if (from.size()) {
		int i;
		for (i = 0; i < from.size(); i++) {
			cout << "-from " << from[i] << endl;
		}
	}
}



}}}

