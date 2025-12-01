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
	Record_birth();

	Orbiter_top_level_session = NULL;

	f_define = false;
	Symbol_definition = NULL;

	f_assign = false;
	//std::vector<std::string> assign_labels;

	f_print_symbols = false;

	f_with = false;
	//std::vector<std::string> with_labels;

	f_activity = false;
	Activity_description = NULL;

}

interface_symbol_table::~interface_symbol_table()
{
	Record_death();

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

void interface_symbol_table::print_help(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	other::data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-define") == 0) {
		cout << "-define <string : label> description -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-assign") == 0) {
		cout << "-assign <string : label>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-print_symbols") == 0) {
		cout << "-print_symbols" << endl;
	}
	else if (ST.stringcmp(argv[i], "-with") == 0) {
		cout << "-with <string : label> *[ -and <string : label> ] -do ... -end" << endl;
	}
}

int interface_symbol_table::recognize_keyword(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	other::data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-define") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-assign") == 0) {
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

	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_symbol_table::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_symbol_table::read_arguments "
				"the next argument is " << argv[i] << endl;
	}

	if (ST.stringcmp(argv[i], "-define") == 0) {

		f_define = true;
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

	}

	else if (ST.stringcmp(argv[i], "-assign") == 0) {

		f_assign = true;

		string s;
		other::data_structures::string_tools ST;

		s.assign(argv[++i]);
		assign_labels.push_back(s);

		while (true) {
			i++;
			if (ST.stringcmp(argv[i], "-and") == 0) {
				string s;

				s.assign(argv[++i]);
				assign_labels.push_back(s);
			}
			else {
				break;
			}
		}


		if (ST.stringcmp(argv[i], "-from") != 0) {
			cout << "after assign, we need -from" << endl;
			cout << "but we have " << argv[i] << endl;
			exit(1);
		}
		i++;
#if 1
		if (ST.stringcmp(argv[i], "-with") != 0) {
			cout << "after assign, we need -from and -with" << endl;
			cout << "but we have " << argv[i] << endl;
			exit(1);
		}
		read_with(argc, argv, i, verbose_level);
#endif

	}

	else if (ST.stringcmp(argv[i], "-print_symbols") == 0) {
		f_print_symbols = true;
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

	f_with = true;
	string s;
	other::data_structures::string_tools ST;

	s.assign(argv[++i]);
	with_labels.push_back(s);

	while (true) {
		i++;
		if (ST.stringcmp(argv[i], "-and") == 0) {
			string s;

			s.assign(argv[++i]);
			with_labels.push_back(s);
		}
		else if (ST.stringcmp(argv[i], "-do") == 0) {
			i++;

			f_activity = true;
			Activity_description = NEW_OBJECT(activity_description);

			Activity_description->Sym = this;

			Activity_description->read_arguments(argc, argv, i, verbose_level);
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

void interface_symbol_table::read_from(
		int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_from" << endl;
	}

	f_with = true;
	string s;
	other::data_structures::string_tools ST;

	s.assign(argv[++i]);
	with_labels.push_back(s);

	while (true) {
		i++;
		if (ST.stringcmp(argv[i], "-and") == 0) {
			string s;

			s.assign(argv[++i]);
			with_labels.push_back(s);
		}
		else if (ST.stringcmp(argv[i], "-do") == 0) {
			i++;

			f_activity = true;
			Activity_description = NEW_OBJECT(activity_description);

			Activity_description->Sym = this;

			Activity_description->read_arguments(argc, argv, i, verbose_level);
			break;
		}
		else {
			cout << "syntax error after -from, seeing " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "interface_symbol_table::read_from done" << endl;
	}

}


void interface_symbol_table::worker(
		int verbose_level)
// called from orbiter_command::execute
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::worker" << endl;
	}
	if (f_define) {
		if (f_v) {
			cout << "interface_symbol_table::worker f_define" << endl;
		}
		Symbol_definition->perform_definition(verbose_level);
	}
	else if (f_print_symbols) {
		if (f_v) {
			cout << "interface_symbol_table::worker f_print_symbols" << endl;
		}
		Orbiter_top_level_session->print_symbol_table();
	}
	else if (f_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_activity" << endl;
		}

		int nb_output;
		other::orbiter_kernel_system::orbiter_symbol_table_entry *Output;

		if (f_v) {
			cout << "interface_symbol_table::worker "
					"before Activity_description->worker" << endl;
		}
		Activity_description->worker(
				with_labels,
				nb_output, Output,
				verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::worker "
					"after Activity_description->worker" << endl;
		}

		if (f_v) {
			cout << "interface_symbol_table::worker "
					"nb_output = " << nb_output << endl;
		}

		if (f_assign) {

			if (f_v) {
				cout << "interface_symbol_table::worker "
						"performing the assignment" << endl;
			}

			if (f_v) {
				cout << "interface_symbol_table::worker "
						"before do_assignment" << endl;
			}
			do_assignment(
					nb_output,
					Output,
					verbose_level);
			if (f_v) {
				cout << "interface_symbol_table::worker "
						"after do_assignment" << endl;
			}
#if 0
			int f_assign;
			std::vector<std::string> assign_labels;

			int f_print_symbols;

			int f_with;
			std::vector<std::string> with_labels;
#endif

			if (f_v) {
				cout << "interface_symbol_table::worker "
						"performing the assignment done" << endl;
			}
		}

	}
	if (f_v) {
		cout << "interface_symbol_table::worker done" << endl;
	}
}

void interface_symbol_table::do_assignment(
		int &nb_output,
		other::orbiter_kernel_system::orbiter_symbol_table_entry *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::do_assignment" << endl;
	}

	if (assign_labels.size() > nb_output) {
		cout << "interface_symbol_table::do_assignment "
				"assign_labels.size() > nb_output" << endl;
		exit(1);
	}

	int i;

	for (i = 0; i < assign_labels.size(); i++) {
		if (f_v) {
			cout << "interface_symbol_table::do_assignment "
					"assigning " << assign_labels[i] << endl;
		}

		Output[i].label = assign_labels[i];


		if (f_v) {
			cout << "interface_symbol_table::do_assignment "
					"before add_symbol_table_entry" << endl;
		}
		Orbiter_top_level_session->add_symbol_table_entry(
				assign_labels[i], Output + i, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_assignment "
					"after add_symbol_table_entry" << endl;
		}


	}

	if (f_v) {
		cout << "interface_symbol_table::do_assignment done" << endl;
	}
}

void interface_symbol_table::print()
{
	//cout << "interface_symbol_table::print" << endl;
	if (f_define) {
		Symbol_definition->print();
	}
	if (f_assign) {

		cout << "-assign ";

		int i;
		for (i = 0; i < assign_labels.size(); i++) {
			cout << assign_labels[i] << " ";
			if (i < assign_labels.size() - 1) {
				cout << "-and ";
			}
		}
		cout << "-from ";
		Activity_description->print();
	}
	if (f_print_symbols) {
		cout << "print_symbol_table" << endl;
	}
	if (f_activity && ! f_assign) {
		Activity_description->print();
	}
	//cout << "interface_symbol_table::print done" << endl;
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

