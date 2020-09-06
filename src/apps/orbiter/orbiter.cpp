// orbiter.cpp
//
// by Anton Betten
//
// started: 4/3/2020
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::interfaces;

int build_number =
#include "../../../build_number"
;

void work(orbiter_session *Session, int argc, const char **argv, int i, int verbose_level);


int main(int argc, const char **argv)
{

	//cout << "orbiter.out main" << endl;

	orbiter_session Session;
	int i;


	// setup:


	cout << "Welcome to Orbiter!  Your build number is " << build_number << "." << endl;

	i = Session.read_arguments(argc, argv, 1);


	int verbose_level;

	verbose_level = Session.verbose_level;

	int f_v = (Session.verbose_level > 1);


	if (Session.f_list_arguments) {
		int j;

		cout << "argument list:" << endl;
		for (j = 0; j < argc; j++) {
			cout << j << " : " << argv[j] << endl;
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
	if (Session.f_fork) {
		if (f_v) {
			cout << "before Session.fork" << endl;
		}
		Session.fork(argc, argv, verbose_level);
		if (f_v) {
			cout << "after Session.fork" << endl;
		}
	}
	else {
		if (Session.f_seed) {
			os_interface Os;

			if (f_v) {
				cout << "seeding random number generator with " << Session.the_seed << endl;
			}
			srand(Session.the_seed);
			Os.random_integer(1000);
		}

		// main dispatch:

		work(&Session, argc, argv, i, verbose_level);


		// finish:

		if (f_memory_debug) {
			global_mem_object_registry.dump();
		}
	}

	cout << "Orbiter session finished." << endl;
	cout << "User time: ";
	the_end(Session.t0);

}


void work(orbiter_session *Session, int argc, const char **argv, int i, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "work" << endl;
	}



	if (f_v) {
		cout << "work before Interface_algebra" << endl;
	}
	{

		interface_algebra Interface_algebra;
		if (Interface_algebra.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_algebra.read_arguments(argc, argv, i, verbose_level);
			Interface_algebra.worker(Session, verbose_level);
		}
	}

	if (f_v) {
		cout << "work before Interface_cryptography" << endl;
	}
	{

		interface_cryptography Interface_cryptography;
		if (Interface_cryptography.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_cryptography.read_arguments(argc, argv, i, verbose_level);
			Interface_cryptography.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "work before Interface_combinatorics" << endl;
	}
	{

		interface_combinatorics Interface_combinatorics;
		if (Interface_combinatorics.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_combinatorics.read_arguments(argc, argv, i, verbose_level);
			Interface_combinatorics.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "work before Interface_coding_theory" << endl;
	}
	{

		interface_coding_theory Interface_coding_theory;
		if (Interface_coding_theory.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_coding_theory.read_arguments(argc, argv, i, verbose_level);
			Interface_coding_theory.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "work before Interface_povray" << endl;
	}
	{

		interface_povray Interface_povray;
		if (Interface_povray.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_povray.read_arguments(argc, argv, i, verbose_level);
			Interface_povray.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "work before Interface_projective" << endl;
	}
	{

		interface_projective Interface_projective;
		if (Interface_projective.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_projective.read_arguments(argc, argv, i, verbose_level);
			Interface_projective.worker(Session, verbose_level);
		}
	}

	if (f_v) {
		cout << "work done" << endl;
	}
}

