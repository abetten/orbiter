/*
 * permutation_group_description.cpp
 *
 *  Created on: Sep 26, 2021
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {



permutation_group_description::permutation_group_description()
{
	degree = 0;
	type = unknown_permutation_group_t;

	f_bsgs = false;
	//std::string bsgs_label;
	//std::string bsgs_label_tex;
	//std::string bsgs_order_text;
	//std::string bsgs_base;
	bsgs_nb_generators = 0;
	//std::string bsgs_generators;

	f_subgroup_by_generators = false;
	//std::string subgroup_label;
	//std::string subgroup_order_text;
	nb_subgroup_generators = 0;
	//std::string subgroup_generators_label;

}


permutation_group_description::~permutation_group_description()
{
}


int permutation_group_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "permutation_group_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-symmetric_group") == 0) {
			degree = ST.strtoi(argv[++i]);
			type = symmetric_group_t;
			if (f_v) {
				cout << "-symmetric_group " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-cyclic_group") == 0) {
			degree = ST.strtoi(argv[++i]);
			type = cyclic_group_t;
			if (f_v) {
				cout << "-cyclic_group " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-elementary_abelian_group") == 0) {
			degree = ST.strtoi(argv[++i]);
			type = elementary_abelian_group_t;
			if (f_v) {
				cout << "-elementary_abelian_group " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identity_group") == 0) {
			degree = ST.strtoi(argv[++i]);
			type = identity_group_t;
			if (f_v) {
				cout << "-identity_group " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dihedral_group") == 0) {
			degree = ST.strtoi(argv[++i]);
			type = dihedral_group_t;
			if (f_v) {
				cout << "-dihedral_group " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-bsgs") == 0) {
			f_bsgs = true;
			bsgs_label.assign(argv[++i]);
			bsgs_label_tex.assign(argv[++i]);
			degree = ST.strtoi(argv[++i]);
			bsgs_order_text.assign(argv[++i]);
			bsgs_base.assign(argv[++i]);
			bsgs_nb_generators = ST.strtoi(argv[++i]);
			bsgs_generators.assign(argv[++i]);
			type = bsgs_t;

			if (f_v) {
				cout << "-bsgs"
						<< " " << bsgs_label
						<< " " << bsgs_label_tex
						<< " " << degree
						<< " " << bsgs_order_text
						<< " " << bsgs_base
						<< " " << bsgs_nb_generators
						<< " " << bsgs_generators
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_by_generators") == 0) {
			f_subgroup_by_generators = true;
			subgroup_label.assign(argv[++i]);
			subgroup_order_text.assign(argv[++i]);
			nb_subgroup_generators = ST.strtoi(argv[++i]);
			subgroup_generators_label.assign(argv[++i]);

			if (f_v) {
				cout << "-subgroup_by_generators " << subgroup_label
						<< " " << subgroup_order_text
						<< " " << nb_subgroup_generators
						<< " " << subgroup_generators_label
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
			cout << "permutation_group_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	if (f_v) {
		cout << "permutation_group_description::read_arguments done" << endl;
	}
	return i + 1;
}


void permutation_group_description::print()
{
	if (type == symmetric_group_t) {
		cout << "-symmetric_group " << degree << endl;
	}
	if (type == cyclic_group_t) {
		cout << "-cyclic_group " << degree << endl;
	}
	if (type == elementary_abelian_group_t) {
		cout << "-elementary_abelian_group " << degree << endl;
	}
	if (type == dihedral_group_t) {
		cout << "-dihedral_group " << degree << endl;
	}
	if (f_bsgs) {
		cout << "-bsgs"
				<< " " << bsgs_label
				<< " " << bsgs_label_tex
				<< " " << degree
				<< " " << bsgs_order_text
				<< " " << bsgs_base
				<< " " << bsgs_nb_generators
				<< " " << bsgs_generators
				<< endl;
	}
	if (f_subgroup_by_generators) {
		cout << "-subgroup_by_generators " << subgroup_label
				<< " " << nb_subgroup_generators
				<< " " << subgroup_generators_label
				<< endl;
	}
}



}}}

