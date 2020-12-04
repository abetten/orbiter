/*
 * orbiter_top_level_session.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {




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

	if (f_v) {
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

	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute" << endl;
	}


	while (i < argc) {
		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_symbol_table, i = " << i << endl;
		}
		{

			interface_symbol_table Interface_symbol_table;
			if (Interface_symbol_table.recognize_keyword(argc, Argv, i, verbose_level)) {
				if (f_v) {
					cout << "recognizing keyword from Interface_symbol_table" << endl;
				}
				i = Interface_symbol_table.read_arguments(argc, Argv, i, verbose_level);
				Interface_symbol_table.worker(this, verbose_level);
			}
		}

		if (i < argc) {
			cout << "next argument is " << Argv[i] << endl;
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_algebra, i = " << i << endl;
		}
		{

			interface_algebra Interface_algebra;
			if (Interface_algebra.recognize_keyword(argc, Argv, i, verbose_level)) {
				if (f_v) {
					cout << "recognizing keyword from Interface_algebra" << endl;
				}
				i = Interface_algebra.read_arguments(argc, Argv, i, verbose_level);
				Interface_algebra.worker(verbose_level);
			}
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_cryptography, i = " << i << endl;
		}
		{

			interface_cryptography Interface_cryptography;
			if (Interface_cryptography.recognize_keyword(argc, Argv, i, verbose_level)) {
				if (f_v) {
					cout << "recognizing keyword from Interface_cryptography" << endl;
				}
				i = Interface_cryptography.read_arguments(argc, Argv, i, verbose_level);
				Interface_cryptography.worker(verbose_level);
			}
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_combinatorics, i = " << i << endl;
		}
		{

			interface_combinatorics Interface_combinatorics;
			if (Interface_combinatorics.recognize_keyword(argc, Argv, i, verbose_level)) {
				if (f_v) {
					cout << "recognizing keyword from Interface_combinatorics" << endl;
				}
				i = Interface_combinatorics.read_arguments(argc, Argv, i, verbose_level);
				Interface_combinatorics.worker(verbose_level);
			}
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_coding_theory, i = " << i << endl;
		}
		{

			interface_coding_theory Interface_coding_theory;
			if (Interface_coding_theory.recognize_keyword(argc, Argv, i, verbose_level)) {
				if (f_v) {
					cout << "recognizing keyword from Interface_coding_theory" << endl;
				}
				i = Interface_coding_theory.read_arguments(argc, Argv, i, verbose_level);
				Interface_coding_theory.worker(verbose_level);
			}
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_povray, i = " << i << endl;
		}
		{

			interface_povray Interface_povray;
			if (Interface_povray.recognize_keyword(argc, Argv, i, verbose_level)) {
				if (f_v) {
					cout << "recognizing keyword from Interface_povray" << endl;
				}
				i = Interface_povray.read_arguments(argc, Argv, i, verbose_level);
				Interface_povray.worker(verbose_level);
			}
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_projective, i = " << i << endl;
		}
		{

			interface_projective Interface_projective;
			if (Interface_projective.recognize_keyword(argc, Argv, i, verbose_level)) {
				i = Interface_projective.read_arguments(argc, Argv, i, verbose_level);
				Interface_projective.worker(verbose_level);
			}
		}

		if (f_v) {
			cout << "orbiter_top_level_session::parse_and_execute before Interface_toolkit, i = " << i << endl;
		}
		{

			interface_toolkit Interface_toolkit;
			if (Interface_toolkit.recognize_keyword(argc, Argv, i, verbose_level)) {
				i = Interface_toolkit.read_arguments(argc, Argv, i, verbose_level);
				Interface_toolkit.worker(verbose_level);
			}
		}
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute done" << endl;
	}
}


void *orbiter_top_level_session::get_object(int idx)
{
	return Orbiter_session->Orbiter_symbol_table->get_object(idx);
}

int orbiter_top_level_session::find_symbol(std::string &label)
{
	return Orbiter_session->Orbiter_symbol_table->find_symbol(label);
}

void orbiter_top_level_session::find_symbols(std::vector<std::string> &Labels, int *&Idx)
{
	int i, idx;

	Idx = NEW_int(Labels.size());

	for (i = 0; i < Labels.size(); i++) {
		idx = find_symbol(Labels[i]);
		if (idx == -1) {
			cout << "cannot find symbol " << Labels[i] << endl;
			exit(1);
		}
		Idx[i] = idx;
	}
}

void orbiter_top_level_session::print_symbol_table()
{
	Orbiter_session->Orbiter_symbol_table->print_symbol_table();
}

void orbiter_top_level_session::add_symbol_table_entry(std::string &label,
		orbiter_symbol_table_entry *Symb, int verbose_level)
{
	Orbiter_session->Orbiter_symbol_table->add_symbol_table_entry(label, Symb, verbose_level);
}


}}

