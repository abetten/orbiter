/*
 * group_modification_description.cpp
 *
 *  Created on: Dec 1, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::foundations;

namespace orbiter {
namespace top_level {

group_modification_description::group_modification_description()
{
	f_restricted_action = FALSE;
	//std::string restricted_action_set_text;
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
	string_tools ST;

	if (f_v) {
		cout << "group_modification_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-restricted_action") == 0) {
			f_restricted_action = TRUE;
			restricted_action_set_text.assign(argv[++i]);
			if (f_v) {
				cout << "-restricted_action " << restricted_action_set_text << endl;
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
		cout << "-restricted_action " << restricted_action_set_text << endl;
	}
	if (from.size()) {
		int i;
		for (i = 0; i < from.size(); i++) {
			cout << "-from " << from[i] << endl;
		}
	}
}



}}

