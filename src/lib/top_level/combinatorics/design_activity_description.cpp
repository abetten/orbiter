/*
 * design_activity_description.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


design_activity_description::design_activity_description()
{
	f_create_table = FALSE;
	//std::string create_table_label;
	//std::string create_table_group_order;
	//std::string create_table_gens;

	f_load_table = FALSE;

	//std::string load_table_H_label;
	//std::string load_table_H_group_order;
	//std::string load_table_H_gens;
	load_table_selected_orbit_length = 0;

	f_canonical_form = FALSE;
	Canonical_form_Descr = NULL;


	f_extract_solutions_by_index = FALSE;
	//std::string extract_solutions_by_index_fname_solutions_in;
	//std::string extract_solutions_by_index_fname_solutions_out;

}

design_activity_description::~design_activity_description()
{

}


int design_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "design_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-create_table") == 0) {
			f_create_table = TRUE;
			create_table_label.assign(argv[++i]);
			create_table_group_order.assign(argv[++i]);
			create_table_gens.assign(argv[++i]);
			if (f_v) {
				cout << "-create_table " << create_table_label
						<< " " << create_table_group_order
						<< " " << create_table_gens
						<< endl;
			}
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
			if (f_v) {
				cout << "-load_table " << create_table_label
						<< " " << create_table_group_order
						<< " " << create_table_gens
						<< " " << load_table_H_label
						<< " " << load_table_H_group_order
						<< " " << load_table_H_gens
						<< " " << load_table_selected_orbit_length
						<< endl;
			}
		}
		else if (stringcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			if (f_v) {
				cout << "-canonical_form, reading extra arguments" << endl;
			}

			Canonical_form_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -f_canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (stringcmp(argv[i], "-extract_solutions_by_index") == 0) {
			f_extract_solutions_by_index = TRUE;
			create_table_label.assign(argv[++i]);
			create_table_group_order.assign(argv[++i]);
			create_table_gens.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_in.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_out.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_solutions_by_index "
						<< create_table_label << " "
						<< create_table_group_order << " "
						<< create_table_gens << " "
						<< extract_solutions_by_index_fname_solutions_in << " "
						<< extract_solutions_by_index_fname_solutions_out << " "
						<< endl;
			}
		}

		else if (stringcmp(argv[i], "-end") == 0) {
			break;
		}
	} // next i
	cout << "design_activity_description::read_arguments done" << endl;
	return i + 1;
}

void design_activity_description::print()
{
	if (f_canonical_form) {
		cout << "-canonical_form " << endl;
		Canonical_form_Descr->print();
	}
	if (f_create_table) {
		cout << "-create_table " << create_table_label
				<< " " << create_table_group_order
				<< " " << create_table_gens
				<< endl;
	}
	if (f_load_table) {
		cout << "-load_table "
				<< create_table_label
				<< " " << create_table_group_order
				<< " " << create_table_gens
				<< " " << load_table_H_label
				<< " " << load_table_H_group_order
				<< " " << load_table_H_gens
				<< " " << load_table_selected_orbit_length
				<< endl;
	}
	if (f_extract_solutions_by_index) {
		cout << "-extract_solutions_by_index "
				<< create_table_label << " "
				<< create_table_group_order << " "
				<< create_table_gens << " "
				<< extract_solutions_by_index_fname_solutions_in << " "
				<< extract_solutions_by_index_fname_solutions_out << " "
				<< endl;
	}
}



}}


