/*
 * orbiter_top_level_session.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


orbiter_top_level_session *The_Orbiter_top_level_session; // global top level Orbiter session



orbiter_top_level_session::orbiter_top_level_session()
{
	Orbiter_session = NULL;
}

orbiter_top_level_session::~orbiter_top_level_session()
{
	if (Orbiter_session) {
		delete Orbiter_session;
	}
}

int orbiter_top_level_session::startup_and_read_arguments(int argc,
		std::string *argv, int i0)
{
	int i;

	//cout << "orbiter_top_level_session::startup_and_read_arguments" << endl;

	Orbiter_session = new orbiter_session;

	i = Orbiter_session->read_arguments(argc, argv, i0);



	//cout << "orbiter_top_level_session::startup_and_read_arguments done" << endl;
	return i;
}

void orbiter_top_level_session::handle_everything(int argc, std::string *Argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (FALSE) {
		cout << "orbiter_top_level_session::handle_everything" << endl;
	}
	if (Orbiter_session->f_list_arguments) {
		int j;

		cout << "argument list:" << endl;
		for (j = 0; j < argc; j++) {
			cout << j << " : " << Argv[j] << endl;
		}
#if 0
		string cmd;

		cmd.assign(Session.orbiter_path);
		cmd.append("orbiter.out");
		for (j = 1; j < argc; j++) {
			cmd.append(" \"");
			cmd.append(argv[j]);
			cmd.append("\" ");
		}
		cout << "system: " << cmd << endl;
		system(cmd.c_str());
		exit(1);
#endif
	}


	if (Orbiter_session->f_fork) {
		if (f_v) {
			cout << "before Top_level_session.Orbiter_session->fork" << endl;
		}
		Orbiter_session->fork(argc, Argv, verbose_level);
		if (f_v) {
			cout << "after Session.fork" << endl;
		}
	}
	else {
		if (Orbiter_session->f_seed) {
			os_interface Os;

			if (f_v) {
				cout << "seeding random number generator with " << Orbiter_session->the_seed << endl;
			}
			srand(Orbiter_session->the_seed);
			Os.random_integer(1000);
		}

		// main dispatch:

		parse_and_execute(argc, Argv, i, verbose_level);


		// finish:

		if (f_memory_debug) {
			global_mem_object_registry.dump();
		}
	}
	if (f_v) {
		cout << "orbiter_top_level_session::handle_everything done" << endl;
	}

}

void orbiter_top_level_session::parse_and_execute(int argc, std::string *Argv, int i, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;

	if (FALSE) {
		cout << "orbiter_top_level_session::parse_and_execute, parsing the orbiter dash code" << endl;
	}


	vector<void *> program;

	if (f_vv) {
		cout << "orbiter_top_level_session::parse_and_execute before parse" << endl;
	}
	parse(argc, Argv, i, program, 0 /* verbose_level */);
	if (f_vv) {
		cout << "orbiter_top_level_session::parse_and_execute after parse" << endl;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute, we parsed the following orbiter dash code program:" << endl;
	}
	for (i = 0; i < program.size(); i++) {

		orbiter_command *OC;

		OC = (orbiter_command *) program[i];

		cout << "Command " << i << ":" << endl;
		OC->print();
	}

	if (f_v) {
		cout << "################################################################################################" << endl;
	}
	if (f_v) {
		cout << "Executing commands:" << endl;
	}

	for (i = 0; i < program.size(); i++) {

		orbiter_command *OC;

		OC = (orbiter_command *) program[i];

		if (f_v) {
			cout << "################################################################################################" << endl;
			cout << "Executing command " << i << ":" << endl;
			OC->print();
			cout << "################################################################################################" << endl;
		}

		OC->execute(verbose_level);

	}


	if (f_v) {
		cout << "Executing commands done" << endl;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute done" << endl;
	}
}

void orbiter_top_level_session::parse(int argc, std::string *Argv, int &i, std::vector<void *> &program, int verbose_level)
{
	int cnt = 0;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	int i_prev = -1;

	if (f_v) {
		cout << "orbiter_top_level_session::parse, parsing the orbiter dash code" << endl;
	}

	while (i < argc) {
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"cnt = " << cnt << ", i = " << i << endl;
			if (i < argc) {
				if (f_vv) {
					cout << "orbiter_top_level_session::parse i=" << i << ", next argument is " << Argv[i] << endl;
				}
			}
		}
		if (i_prev == i) {
			cout << "orbiter_top_level_session::parse we seem to be stuck in a look" << endl;
			exit(1);
		}
		i_prev = i;
		if (f_v) {
			cout << "orbiter_top_level_session::parse before Interface_symbol_table, i = " << i << endl;
		}

		orbiter_command *OC;

		OC = NEW_OBJECT(orbiter_command);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse before OC->parse" << endl;
		}
		OC->parse(this, argc, Argv, i, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse after OC->parse" << endl;
		}

		program.push_back(OC);

#if 0
		if (f_v) {
			cout << "orbiter_top_level_session::parse before OC->execute" << endl;
		}
		OC->execute(verbose_level);
		if (f_v) {
			cout << "orbiter_top_level_session::parse after OC->execute" << endl;
		}
#endif




		//cout << "Command is unrecognized " << Argv[i] << endl;
		//exit(1);
		cnt++;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse, parsing the orbiter dash code done" << endl;
	}
}

void *orbiter_top_level_session::get_object(int idx)
{
	return Orbiter_session->get_object(idx);
}

int orbiter_top_level_session::find_symbol(std::string &label)
{
	return Orbiter_session->find_symbol(label);
}

void orbiter_top_level_session::find_symbols(std::vector<std::string> &Labels, int *&Idx)
{

	Orbiter_session->find_symbols(Labels, Idx);
}

void orbiter_top_level_session::print_symbol_table()
{
	Orbiter_session->print_symbol_table();
}

void orbiter_top_level_session::add_symbol_table_entry(std::string &label,
		orbiter_symbol_table_entry *Symb, int verbose_level)
{
	Orbiter_session->add_symbol_table_entry(label, Symb, verbose_level);
}


}}

