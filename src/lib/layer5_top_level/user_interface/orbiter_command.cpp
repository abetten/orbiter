/*
 * orbiter_command.cpp
 *
 *  Created on: Jun 20, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {





orbiter_command::orbiter_command()
{
	Record_birth();
	Orbiter_top_level_session = NULL;

	f_algebra = false;
	Algebra = NULL;

	f_coding_theory = false;
	Coding_theory = NULL;

	f_combinatorics = false;
	Combinatorics = NULL;

	f_cryptography = false;
	Cryptography = NULL;

	f_povray = false;
	Povray = NULL;

	f_projective = false;
	Projective = NULL;

	f_symbol_table = false;
	Symbol_table = NULL;

	f_toolkit = false;
	Toolkit = NULL;
}

orbiter_command::~orbiter_command()
{
	Record_death();
}


void orbiter_command::parse(
		orbiter_top_level_session *Orbiter_top_level_session,
		int argc, std::string *Argv, int &i, int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_command::parse" << endl;
	}

	orbiter_command::Orbiter_top_level_session = Orbiter_top_level_session;

	{

		interface_symbol_table *Interface_symbol_table;

		Interface_symbol_table = NEW_OBJECT(interface_symbol_table);
		Interface_symbol_table->init(Orbiter_top_level_session, verbose_level);

		if (Interface_symbol_table->recognize_keyword(argc, Argv, i, verbose_level)) {
			if (f_v) {
				cout << "orbiter_command::parse recognizing "
						"keyword from Interface_symbol_table" << endl;
			}
			Interface_symbol_table->read_arguments(argc, Argv, i, verbose_level);
			if (f_v) {
				cout << "orbiter_command::parse after "
						"Interface_symbol_table.read_arguments, i=" << i << endl;
			}
			//Interface_symbol_table.worker(verbose_level);

			f_symbol_table = true;
			Symbol_table = Interface_symbol_table;
			return;
		}
		else {
			FREE_OBJECT(Interface_symbol_table);
		}
	}


	if (f_v) {
		cout << "orbiter_command::parse before Interface_algebra, "
				"i = " << i << " " << Argv[i] << endl;
	}
	{

		interface_algebra *Interface_algebra;
		Interface_algebra = NEW_OBJECT(interface_algebra);
		if (Interface_algebra->recognize_keyword(argc, Argv, i, verbose_level)) {
			if (f_v) {
				cout << "orbiter_command::parse recognizing "
						"keyword from Interface_algebra" << endl;
			}
			Interface_algebra->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_algebra.worker(verbose_level);
			f_algebra = true;
			Algebra = Interface_algebra;
			return;
		}
		else {
			FREE_OBJECT(Interface_algebra);
		}
	}

	if (f_v) {
		cout << "orbiter_command::parse before Interface_cryptography, "
				"i = " << i << " " << Argv[i] << endl;
	}
	{

		interface_cryptography *Interface_cryptography;
		Interface_cryptography = NEW_OBJECT(interface_cryptography);
		if (Interface_cryptography->recognize_keyword(argc, Argv, i, verbose_level)) {
			if (f_v) {
				cout << "orbiter_command::parse recognizing keyword "
						"from Interface_cryptography" << endl;
			}
			Interface_cryptography->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_cryptography.worker(verbose_level);
			f_cryptography = true;
			Cryptography = Interface_cryptography;
			return;
		}
		else {
			FREE_OBJECT(Interface_cryptography);
		}
	}

	if (f_v) {
		cout << "orbiter_command::parse before Interface_combinatorics, "
				"i = " << i << " " << Argv[i] << endl;
	}
	{

		interface_combinatorics *Interface_combinatorics;
		Interface_combinatorics = NEW_OBJECT(interface_combinatorics);
		if (Interface_combinatorics->recognize_keyword(argc, Argv, i, verbose_level)) {
			if (f_v) {
				cout << "orbiter_command::parse recognizing keyword "
						"from Interface_combinatorics" << endl;
			}
			Interface_combinatorics->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_combinatorics.worker(verbose_level);
			f_combinatorics = true;
			Combinatorics = Interface_combinatorics;
			return;
		}
		else {
			FREE_OBJECT(Interface_combinatorics);
		}
	}

	if (f_v) {
		cout << "orbiter_command::parse before Interface_coding_theory, "
				"i = " << i << " " << Argv[i] << endl;
	}
	{

		interface_coding_theory *Interface_coding_theory;
		Interface_coding_theory = NEW_OBJECT(interface_coding_theory);
		if (Interface_coding_theory->recognize_keyword(argc, Argv, i, verbose_level)) {
			if (f_v) {
				cout << "orbiter_command::parse recognizing keyword "
						"from Interface_coding_theory" << endl;
			}
			Interface_coding_theory->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_coding_theory.worker(verbose_level);
			f_coding_theory = true;
			Coding_theory = Interface_coding_theory;
			return;
		}
		else {
			FREE_OBJECT(Interface_coding_theory);
		}
	}

	if (f_v) {
		cout << "orbiter_command::parse before Interface_povray, "
				"i = " << i << " " << Argv[i] << endl;
	}
	{

		interface_povray *Interface_povray;
		Interface_povray = NEW_OBJECT(interface_povray);
		if (Interface_povray->recognize_keyword(argc, Argv, i, verbose_level)) {
			if (f_v) {
				cout << "orbiter_command::parse recognizing "
						"keyword from Interface_povray" << endl;
			}
			Interface_povray->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_povray.worker(verbose_level);
			f_povray = true;
			Povray = Interface_povray;
			return;
		}
		else {
			FREE_OBJECT(Interface_povray);
		}
	}

	if (f_v) {
		cout << "orbiter_command::parse before Interface_projective, "
				"i = " << i << " " << Argv[i] << endl;
	}
	{

		interface_projective *Interface_projective;
		Interface_projective = NEW_OBJECT(interface_projective);
		if (Interface_projective->recognize_keyword(argc, Argv, i, verbose_level)) {
			Interface_projective->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_projective.worker(verbose_level);
			f_projective = true;
			Projective = Interface_projective;
			return;
		}
		else {
			FREE_OBJECT(Interface_projective);
		}
	}

	if (f_v) {
		cout << "orbiter_command::parse before Interface_toolkit, "
				"i = " << i << endl;
	}
	{

		interface_toolkit *Interface_toolkit;
		Interface_toolkit = NEW_OBJECT(interface_toolkit);
		if (Interface_toolkit->recognize_keyword(argc, Argv, i, verbose_level)) {
			Interface_toolkit->read_arguments(argc, Argv, i, verbose_level);
			i++;
			//Interface_toolkit.worker(verbose_level);
			f_toolkit = true;
			Toolkit = Interface_toolkit;
			return;
		}
		else {
			FREE_OBJECT(Interface_toolkit);
		}
	}
	cout << "orbiter_command::parse command " << Argv[i] << " at position "
			<< i << " is unrecognized" << endl;

#if 0
	for (int j = 0; j <= i; j++) {
		cout << Argv[j] << endl;
	}
#endif
	exit(1);

}

void orbiter_command::execute(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_command::execute" << endl;
	}

	if (f_symbol_table) {
		Symbol_table->worker(verbose_level);
	}
	else if (f_algebra) {
		Algebra->worker(verbose_level);
	}
	else if (f_cryptography) {
		Cryptography->worker(verbose_level);
	}
	else if (f_combinatorics) {
		Combinatorics->worker(verbose_level);
	}
	else if (f_coding_theory) {
		Coding_theory->worker(verbose_level);
	}
	else if (f_povray) {
		Povray->worker(verbose_level);
	}
	else if (f_projective) {
		Projective->worker(verbose_level);
	}
	else if (f_toolkit) {
		Toolkit->worker(verbose_level);
	}
	else {
		cout << "orbiter_command::execute unknown type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbiter_command::execute done" << endl;
	}

}


void orbiter_command::print()
{
	if (f_symbol_table) {
		Symbol_table->print();
	}
	else if (f_algebra) {
		Algebra->print();
	}
	else if (f_cryptography) {
		Cryptography->print();
	}
	else if (f_combinatorics) {
		Combinatorics->print();
	}
	else if (f_coding_theory) {
		Coding_theory->print();
	}
	else if (f_povray) {
		Povray->print();
	}
	else if (f_projective) {
		Projective->print();
	}
	else if (f_toolkit) {
		Toolkit->print();
	}
	else {
		cout << "orbiter_command::print unknown type" << endl;
		exit(1);
	}

}

}}}



