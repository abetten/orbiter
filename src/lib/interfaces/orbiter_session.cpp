/*
 * orbiter_session.cpp
 *
 *  Created on: May 26, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {


orbiter_session *Orbiter_session = NULL;


orbiter_session::orbiter_session()
{

	verbose_level = 0;

	t0 = 0;

	f_seed = FALSE;
	the_seed = TRUE;

	f_memory_debug = FALSE;
	memory_debug_verbose_level = 0;

	f_override_polynomial = FALSE;
	//override_polynomial = NULL;

	f_orbiter_path = FALSE;
	//orbiter_path;
}


orbiter_session::~orbiter_session()
{

}


void orbiter_session::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-v") == 0) {
		cout << "-v <int : verbosity>" << endl;
	}
	else if (strcmp(argv[i], "-seed") == 0) {
		cout << "-seed <int : seed>" << endl;
	}
	else if (strcmp(argv[i], "-memory_debug") == 0) {
		cout << "-memory_debug <int : memory_debug_verbose_level>" << endl;
	}
	else if (strcmp(argv[i], "-override_polynomial") == 0) {
		cout << "-override_polynomial <string : polynomial in decimal>" << endl;
	}
	else if (strcmp(argv[i], "-orbiter_path") == 0) {
		cout << "-orbiter_path <string : path>" << endl;
	}
}

int orbiter_session::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-v") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-seed") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-memory_debug") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-override_polynomial") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-orbiter_path") == 0) {
		return true;
	}
	return false;
}

int orbiter_session::read_arguments(int argc,
		const char **argv, int i0)
{
	int i;

	//cout << "orbiter_session::read_arguments" << endl;

	os_interface Os;

	t0 = Os.os_ticks();

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-seed") == 0) {
			f_seed = TRUE;
			the_seed = atoi(argv[++i]);
			cout << "-seed " << the_seed << endl;
		}
		else if (strcmp(argv[i], "-memory_debug") == 0) {
			f_memory_debug = TRUE;
			memory_debug_verbose_level = atoi(argv[++i]);
			cout << "-memory_debug " << memory_debug_verbose_level << endl;
		}
		else if (strcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial.assign(argv[++i]);
			cout << "-override_polynomial " << override_polynomial << endl;
		}
		else if (strcmp(argv[i], "-orbiter_path") == 0) {
			f_orbiter_path = TRUE;
			orbiter_path.assign(argv[++i]);
			cout << "-orbiter_path " << orbiter_path << endl;
		}
		else {
			break;
		}
	}
	//cout << "orbiter_session::read_arguments done" << endl;
	return i;
}

void orbiter_session::work(int argc, const char **argv, int i, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "before Interface_algebra" << endl;
	}
	{

		if (f_v) {
			cout << "before Interface_algebra.recognize_keyword" << endl;
		}
		if (Interface_algebra.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_algebra.read_arguments(argc, argv, i, verbose_level);
			Interface_algebra.worker(this, verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_cryptography" << endl;
	}
	{

		if (f_v) {
			cout << "before Interface_cryptography.recognize_keyword" << endl;
		}
		if (Interface_cryptography.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_cryptography.read_arguments(argc, argv, i, verbose_level);
			Interface_cryptography.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_combinatorics" << endl;
	}
	{

		if (Interface_combinatorics.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_combinatorics.read_arguments(argc, argv, i, verbose_level);
			Interface_combinatorics.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_coding_theory" << endl;
	}
	{

		if (Interface_coding_theory.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_coding_theory.read_arguments(argc, argv, i, verbose_level);
			Interface_coding_theory.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_povray" << endl;
	}
	{

		if (Interface_povray.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_povray.read_arguments(argc, argv, i, verbose_level);
			Interface_povray.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_projective" << endl;
	}
	{

		if (Interface_projective.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_projective.read_arguments(argc, argv, i, verbose_level);
			Interface_projective.worker(this, verbose_level);
		}
	}
}


}}

