/*
 * large_set_activity_description.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


large_set_activity_description::large_set_activity_description()
{
}

large_set_activity_description::~large_set_activity_description()
{

}


int large_set_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "large_set_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

#if 0
		if (stringcmp(argv[i], "-create_table") == 0) {
			f_create_table = TRUE;
			create_table_label.assign(argv[++i]);
			create_table_group_order.assign(argv[++i]);
			create_table_gens.assign(argv[++i]);
			cout << "-create_table " << create_table_label
					<< " " << create_table_group_order
					<< " " << create_table_gens
					<< endl;
		}
		else if (stringcmp(argv[i], "-load_table") == 0) {
			f_load_table = TRUE;
			create_table_label.assign(argv[++i]);
			create_table_group_order.assign(argv[++i]);
			create_table_gens.assign(argv[++i]);
			load_table_H_label.assign(argv[++i]);
			load_table_H_group_order.assign(argv[++i]);
			load_table_H_gens.assign(argv[++i]);
			load_table_selected_orbit_length = strtoi(argv[++i]);
			cout << "-load_table " << create_table_label
					<< " " << create_table_group_order
					<< " " << create_table_gens
					<< " " << load_table_H_label
					<< " " << load_table_H_group_order
					<< " " << load_table_H_gens
					<< " " << load_table_selected_orbit_length
					<< endl;
		}
#endif
		if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "large_set_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "large_set_activity_description::read_arguments done" << endl;
	return i + 1;
}




}}}


