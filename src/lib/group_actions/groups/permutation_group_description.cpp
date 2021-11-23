/*
 * permutation_group_description.cpp
 *
 *  Created on: Sep 26, 2021
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace group_actions {

permutation_group_description::permutation_group_description()
{
	degree = 0;
	type = unknown_permutation_group_t;

	f_bsgs = FALSE;
	//std::string bsgs_label;
	//std::string bsgs_label_tex;
	//std::string bsgs_order_text;
	//std::string bsgs_base;
	bsgs_nb_generators = 0;
	bsgs_generators = NULL;

	f_subgroup_by_generators = FALSE;
	//std::string subgroup_label;
	//std::string subgroup_order_text;
	nb_subgroup_generators = 0;
	subgroup_generators_as_string = NULL;

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

	if (f_v) {
		cout << "permutation_group_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


		if (stringcmp(argv[i], "-symmetric_group") == 0) {
			degree = strtoi(argv[++i]);
			type = symmetric_group_t;
			if (f_v) {
				cout << "-symmetric_group " << degree << endl;
			}
		}
		else if (stringcmp(argv[i], "-bsgs") == 0) {
			f_bsgs = TRUE;
			bsgs_label.assign(argv[++i]);
			bsgs_label_tex.assign(argv[++i]);
			degree = strtoi(argv[++i]);
			bsgs_order_text.assign(argv[++i]);
			bsgs_base.assign(argv[++i]);
			bsgs_nb_generators = strtoi(argv[++i]);
			bsgs_generators = new std::string [bsgs_nb_generators];
			type = bsgs_t;

			os_interface Os;

			i++;
			for (int h = 0; h < bsgs_nb_generators; h++) {

				Os.get_string_from_command_line(bsgs_generators[h], argc, argv, i, verbose_level);
			}
			i--;

			if (f_v) {
				cout << "-bsgs"
						<< " " << bsgs_label
						<< " " << bsgs_label_tex
						<< " " << degree
						<< " " << bsgs_order_text
						<< " " << bsgs_base
						<< " " << bsgs_nb_generators;
				for (int h = 0; h < bsgs_nb_generators; h++) {
					cout << " " << bsgs_generators[h] << endl;
				}
				cout << endl;
			}
		}
		else if (stringcmp(argv[i], "-subgroup_by_generators") == 0) {
			f_subgroup_by_generators = TRUE;
			subgroup_label.assign(argv[++i]);
			subgroup_order_text.assign(argv[++i]);
			nb_subgroup_generators = strtoi(argv[++i]);
			subgroup_generators_as_string = new std::string [nb_subgroup_generators];

			os_interface Os;

			i++;
			for (int h = 0; h < nb_subgroup_generators; h++) {

				Os.get_string_from_command_line(subgroup_generators_as_string[h], argc, argv, i, verbose_level);
			}
			i--;
			if (f_v) {
				cout << "-subgroup_by_generators " << subgroup_label
						<< " " << nb_subgroup_generators << endl;
				for (int h = 0; h < nb_subgroup_generators; h++) {
					cout << " " << subgroup_generators_as_string[h] << endl;
				}
				cout << endl;
			}
		}
		else if (stringcmp(argv[i], "-end") == 0) {
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
	if (f_bsgs) {
		cout << "-bsgs"
				<< " " << bsgs_label
				<< " " << bsgs_label_tex
				<< " " << degree
				<< " " << bsgs_order_text
				<< " " << bsgs_base
				<< " " << bsgs_nb_generators;
		for (int h = 0; h < bsgs_nb_generators; h++) {
			cout << " " << bsgs_generators[h] << endl;
		}
		cout << endl;
	}
	if (f_subgroup_by_generators) {
		cout << "-subgroup_by_generators \""
				<< subgroup_label << "\" \""
				<< subgroup_order_text << "\" "
				<< nb_subgroup_generators << endl;
		for (int h = 0; h < nb_subgroup_generators; h++) {
			cout << " \"" << subgroup_generators_as_string[h] << "\"" << endl;
		}
		cout << endl;
	}
}



}}

