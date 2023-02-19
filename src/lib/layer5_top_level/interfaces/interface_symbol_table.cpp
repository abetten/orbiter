/*
 * interface_symbol_table.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {




interface_symbol_table::interface_symbol_table()
{

	Orbiter_top_level_session = NULL;

	f_define = FALSE;
	Symbol_definition = NULL;

	f_print_symbols = FALSE;

	f_with = FALSE;
	//std::vector<std::string> with_labels;

	f_activity = FALSE;
	Activity_description = NULL;

}

void interface_symbol_table::init(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::init" << endl;
	}
	interface_symbol_table::Orbiter_top_level_session =
			Orbiter_top_level_session;
	if (f_v) {
		cout << "interface_symbol_table::done" << endl;
	}
}

void interface_symbol_table::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-define") == 0) {
		cout << "-define <string : label> description -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-print_symbols") == 0) {
		cout << "-print_symbols" << endl;
	}
	else if (ST.stringcmp(argv[i], "-with") == 0) {
		cout << "-with <string : label> *[ -and <string : label> ] -do ... -end" << endl;
	}
}

int interface_symbol_table::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-define") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-print_symbols") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-with") == 0) {
		return true;
	}
	return false;
}

void interface_symbol_table::read_arguments(
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_symbol_table::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_symbol_table::read_arguments "
				"the next argument is " << argv[i] << endl;
	}

	if (ST.stringcmp(argv[i], "-define") == 0) {

		f_define = TRUE;
		Symbol_definition = NEW_OBJECT(symbol_definition);


		if (f_v) {
			cout << "interface_symbol_table::read_arguments "
					"before Symbol_definition->read_definition" << endl;
		}
		Symbol_definition->read_definition(this, argc, argv, i, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_arguments "
					"after Symbol_definition->read_definition" << endl;
		}

#if 0
		if (f_v) {
			cout << "interface_symbol_table::read_arguments "
					"before Symbol_definition->perform_definition" << endl;
		}
		Symbol_definition->perform_definition(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::read_arguments "
					"after Symbol_definition->perform_definition" << endl;
		}
#endif
	}

	else if (ST.stringcmp(argv[i], "-print_symbols") == 0) {
		f_print_symbols = TRUE;
		if (f_v) {
			cout << "-print_symbols" << endl;
		}
		i++;
	}

	else if (ST.stringcmp(argv[i], "-with") == 0) {
		read_with(argc, argv, i, verbose_level);
	}

	if (f_v) {
		cout << "interface_symbol_table::read_arguments done" << endl;
	}
	//return i;
}






void interface_symbol_table::read_with(
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_with" << endl;
	}

	f_with = TRUE;
	string s;
	data_structures::string_tools ST;

	s.assign(argv[++i]);
	with_labels.push_back(s);

	while (TRUE) {
		i++;
		if (ST.stringcmp(argv[i], "-and") == 0) {
			string s;

			s.assign(argv[++i]);
			with_labels.push_back(s);
		}
		else if (ST.stringcmp(argv[i], "-do") == 0) {
			i++;

			f_activity = TRUE;
			Activity_description = NEW_OBJECT(activity_description);

			Activity_description->read_arguments(this, argc, argv, i, verbose_level);
			break;
		}
		else {
			cout << "syntax error after -with, seeing " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "interface_symbol_table::read_with done" << endl;
	}

}


void interface_symbol_table::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::worker" << endl;
	}
	if (f_define) {
		Symbol_definition->perform_definition(verbose_level);
	}
	else if (f_print_symbols) {
		Orbiter_top_level_session->print_symbol_table();
	}
	else if (f_activity) {
		Activity_description->worker(verbose_level);
	}
	if (f_v) {
		cout << "interface_symbol_table::worker done" << endl;
	}
}

void interface_symbol_table::print()
{
	if (f_define) {
		Symbol_definition->print();
	}
	if (f_print_symbols) {
		cout << "print_symbol_table" << endl;
	}
	if (f_activity) {
		Activity_description->print();
	}
}

void interface_symbol_table::print_with()
{
	int i;

	for (i = 0; i < with_labels.size(); i++) {
		cout << with_labels[i];
		if (i < with_labels.size() - 1) {
			cout << ", ";
		}
	}
	cout << endl;

}

}}}

