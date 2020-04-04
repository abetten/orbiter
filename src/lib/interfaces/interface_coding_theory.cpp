/*
 * interface_coding_theory.cpp
 *
 *  Created on: Apr 4, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::interfaces;


interface_coding_theory::interface_coding_theory()
{
	f_make_macwilliams_system = FALSE;
	q = 0;
	n = 0;
	k = 0;
}


void interface_coding_theory::print_help(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		cout << "-make_macwilliams_system <int : q> <int : n> <int k>" << endl;
	}
}

int interface_coding_theory::recognize_keyword(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		return true;
	}
	return false;
}

void interface_coding_theory::read_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_coding_theory::read_arguments" << endl;
	//return 0;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
			f_make_macwilliams_system = TRUE;
			q = atoi(argv[++i]);
			n = atoi(argv[++i]);
			k = atoi(argv[++i]);
			cout << "-make_macwilliams_system " << q << " " << n << " " << k << endl;
		}
	}
}


void interface_coding_theory::worker(int verbose_level)
{
	if (f_make_macwilliams_system) {
		do_make_macwilliams_system(q, n, k, verbose_level);
	}
}


void interface_coding_theory::do_make_macwilliams_system(int q, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object *M;

	if (f_v) {
		cout << "interface_coding_theory::do_make_macwilliams_system" << endl;
	}

	D.make_mac_williams_equations(M, n, k, q, verbose_level);



	if (f_v) {
		cout << "interface_coding_theory::do_make_macwilliams_system done" << endl;
	}
}
