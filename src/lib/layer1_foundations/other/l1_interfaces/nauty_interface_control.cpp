/*
 * nauty_interface_control.cpp
 *
 *  Created on: Dec 14, 2024
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace l1_interfaces {


nauty_interface_control::nauty_interface_control()
{
	Record_birth();

	f_save_nauty_input_graphs = false;
	//std::string save_nauty_input_graphs_prefix;

	f_save_orbit_of_equations = false;
	//std::string save_orbit_of_equations_prefix;

	f_partition = false;
	//std::string partition_text;

	f_show_canonical_form = false;

}

nauty_interface_control::~nauty_interface_control()
{
	Record_death();

}

void nauty_interface_control::init(
		int f_save_nauty_input_graphs,
		std::string &save_nauty_input_graphs_prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_interface_control::init" << endl;
	}

	nauty_interface_control::f_save_nauty_input_graphs = f_save_nauty_input_graphs;
	nauty_interface_control::save_nauty_input_graphs_prefix = save_nauty_input_graphs_prefix;

	if (f_v) {
		cout << "nauty_interface_control::init f_save_nauty_input_graphs = " << f_save_nauty_input_graphs << endl;
		cout << "nauty_interface_control::init prefix = " << save_nauty_input_graphs_prefix << endl;
	}

	if (f_v) {
		cout << "nauty_interface_control::init done" << endl;
	}
}

int nauty_interface_control::parse_arguments(
		int argc, std::string *argv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "nauty_interface_control::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-save_nauty_input_graphs") == 0) {
			f_save_nauty_input_graphs = true;
			save_nauty_input_graphs_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-save_nauty_input_graphs " << save_nauty_input_graphs_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_orbit_of_equations") == 0) {
			f_save_orbit_of_equations = true;
			save_orbit_of_equations_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-save_orbit_of_equations " << save_orbit_of_equations_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-partition") == 0) {
			f_partition = true;
			partition_text.assign(argv[++i]);
			if (f_v) {
				cout << "-partition " << partition_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_canonical_form") == 0) {
			f_show_canonical_form = true;
			if (f_v) {
				cout << "-show_canonical_form" << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			return i + 1;
		}
		else {
			cout << "nauty_interface_control::parse_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	}
	cout << "nauty_interface_control::parse_arguments "
			"did not see -end option" << endl;
	exit(1);
}


void nauty_interface_control::print()
{
	if (f_save_nauty_input_graphs) {
		cout << "-save_nauty_input_graphs " << save_nauty_input_graphs_prefix << endl;
	}
	if (f_save_orbit_of_equations) {
		cout << "-save_orbit_of_equations " << save_orbit_of_equations_prefix << endl;
	}
	if (f_partition) {
		cout << "-partition " << partition_text << endl;
	}
	if (f_show_canonical_form) {
		cout << "-show_canonical_form" << endl;
	}
}





}}}}



