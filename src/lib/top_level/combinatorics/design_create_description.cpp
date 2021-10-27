/*
 * design_create_description.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


design_create_description::design_create_description()
{
	f_q = FALSE;
	q = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_family = FALSE;
	//family_name;

	f_list_of_blocks = FALSE;
	list_of_blocks_v = 0;
	list_of_blocks_k = 0;
	//std::string list_of_blocks_text;

	f_list_of_blocks_from_file = FALSE;
	//std::string list_of_blocks_from_file_fname;

}

design_create_description::~design_create_description()
{
	freeself();
}

void design_create_description::null()
{
}

void design_create_description::freeself()
{
	null();
}

int design_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	cout << "design_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = strtoi(argv[++i]);
			if (f_v) {
				cout << "-q " << q << endl;
			}
		}
		else if (stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = strtoi(argv[++i]);
			if (f_v) {
				cout << "-catalogue " << iso << endl;
			}
		}
		else if (stringcmp(argv[i], "-family") == 0) {
			f_family = TRUE;
			family_name.assign(argv[++i]);
			if (f_v) {
				cout << "-family " << family_name << endl;
			}
		}
		else if (stringcmp(argv[i], "-list_of_blocks") == 0) {
			f_list_of_blocks = TRUE;
			list_of_blocks_v = strtoi(argv[++i]);
			list_of_blocks_k = strtoi(argv[++i]);
			list_of_blocks_text.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_blocks " << list_of_blocks_v
						<< " " << list_of_blocks_k
						<< " " << list_of_blocks_text
						<< endl;
			}
		}
		else if (stringcmp(argv[i], "-list_of_blocks_from_file") == 0) {
			f_list_of_blocks_from_file = TRUE;
			list_of_blocks_v = strtoi(argv[++i]);
			list_of_blocks_k = strtoi(argv[++i]);
			list_of_blocks_from_file_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-list_of_blocks_from_file " << list_of_blocks_v
						<< " " << list_of_blocks_k
						<< " " << list_of_blocks_from_file_fname
						<< endl;
			}
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			break;
		}
	} // next i
	cout << "design_create_description::read_arguments done" << endl;
	return i + 1;
}


int design_create_description::get_q()
{
	if (!f_q) {
		cout << "design_create_description::get_q "
				"q has not been set yet" << endl;
		exit(1);
	}
	return q;
}

void design_create_description::print()
{
	if (f_q) {
		cout << "-q " << q << endl;
	}
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_family) {
		cout << "-family " << family_name << endl;
	}
	if (f_list_of_blocks) {
		cout << "-list_of_blocks " << list_of_blocks_v
				<< " " << list_of_blocks_k
				<< " " << list_of_blocks_text
				<< endl;
	}
	if (f_list_of_blocks_from_file) {
		cout << "-list_of_blocks_from_file " << list_of_blocks_v
				<< " " << list_of_blocks_k
				<< " " << list_of_blocks_from_file_fname
				<< endl;
	}
}


}}




