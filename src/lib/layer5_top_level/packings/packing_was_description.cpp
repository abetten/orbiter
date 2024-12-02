/*
 * packing_was_description.cpp
 *
 *  Created on: Jun 10, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


packing_was_description::packing_was_description()
{
	Record_birth();

	f_process_long_orbits = false;
	Long_Orbits_Descr = NULL;

	f_fixp_clique_types_save_individually = false;

	f_spread_tables_prefix = false;
	//spread_tables_prefix = "";

	f_exact_cover = false;
	ECA = NULL;

	f_isomorph = false;
	IA = NULL;

	f_H = false;
	//std::string H_label;
	H_Descr = NULL;

	f_N = false;
	//std::string N_label;
	N_Descr = NULL;

	f_report = false;
	//clique_size = 0;

	f_regular_packing = false;
}

packing_was_description::~packing_was_description()
{
	Record_death();
}

int packing_was_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	other::data_structures::string_tools ST;



	cout << "packing_was_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-H") == 0) {
			f_H = true;
			cout << "reading -H" << endl;
			H_label.assign(argv[++i]);
			H_Descr = NEW_OBJECT(group_constructions::linear_group_description);
			i += H_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			//i++;
			cout << "done reading -H" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-N") == 0) {
			f_N = true;
			cout << "reading -N" << endl;
			N_label.assign(argv[++i]);
			N_Descr = NEW_OBJECT(group_constructions::linear_group_description);
			i += N_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			//i++;
			cout << "done reading -N" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-fixp_clique_types_save_individually") == 0) {
			f_fixp_clique_types_save_individually = true;
			cout << "-fixp_clique_types_save_individually " << endl;
		}


		else if (ST.stringcmp(argv[i], "-process_long_orbits") == 0) {
			f_process_long_orbits = true;
			Long_Orbits_Descr = NEW_OBJECT(packing_long_orbits_description);
			cout << "-process_long_orbits " << endl;
			i += Long_Orbits_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -process_long_orbits " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}




		else if (ST.stringcmp(argv[i], "-spread_tables_prefix") == 0) {
			f_spread_tables_prefix = true;
			spread_tables_prefix.assign(argv[++i]);
			cout << "-spread_tables_prefix "
				<< spread_tables_prefix << endl;
		}

		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			cout << "-report " << endl;
		}

		else if (ST.stringcmp(argv[i], "-exact_cover") == 0) {
			f_exact_cover = true;
			ECA = NEW_OBJECT(solvers_package::exact_cover_arguments);
			i += ECA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -exact_cover " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-exact_cover " << endl;
		}

		else if (ST.stringcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = true;
			IA = NEW_OBJECT(isomorph::isomorph_arguments);
			i += IA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -isomorph " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-isomorph " << endl;
		}

		else if (ST.stringcmp(argv[i], "-regular_packing") == 0) {
			f_regular_packing = true;
			cout << "-regular_packing " << endl;
		}



		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "packing_was_description::read_arguments unknown argument " << argv[i] << endl;
			exit(1);
		}
	} // next i

	cout << "packing_was_description::read_arguments done" << endl;
	return i + 1;
}


void packing_was_description::print()
{
	if (f_H) {
		cout << "-H" << " " << H_label << endl;
		H_Descr->print();
	}

	if (f_N) {
		cout << "-N" << " " << N_label << endl;
		N_Descr->print();
	}

	if (f_fixp_clique_types_save_individually) {
		cout << "-fixp_clique_types_save_individually " << endl;
	}


	if (f_process_long_orbits) {
		cout << "-process_long_orbits " << endl;
		Long_Orbits_Descr->print();
	}




	if (f_spread_tables_prefix) {
		cout << "-spread_tables_prefix " << spread_tables_prefix << endl;
	}

	if (f_report) {
		cout << "-report " << endl;
	}

	if (f_exact_cover) {
		cout << "-exact_cover " << endl;
	}

	if (f_isomorph) {
		cout << "-isomorph " << endl;
	}

	if (f_regular_packing) {
		cout << "-regular_packing " << endl;
	}
}



}}}

