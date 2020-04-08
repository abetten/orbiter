/*
 * interface_coding_theory.cpp
 *
 *  Created on: Apr 4, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace interfaces {


interface_coding_theory::interface_coding_theory()
{
	argc = 0;
	argv = NULL;

	f_make_macwilliams_system = FALSE;
	q = 0;
	n = 0;
	k = 0;
	f_codes_classify = FALSE;
}


void interface_coding_theory::print_help(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		cout << "-make_macwilliams_system <int : q> <int : n> <int k>" << endl;
	}
	else if (strcmp(argv[i], "-code_classify") == 0) {
		cout << "-code_classify" << endl;
	}
}

int interface_coding_theory::recognize_keyword(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-codes_classify") == 0) {
		return true;
	}
	return false;
}

void interface_coding_theory::read_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_coding_theory::read_arguments" << endl;
	//return 0;

	interface_coding_theory::argc = argc;
	interface_coding_theory::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
			f_make_macwilliams_system = TRUE;
			q = atoi(argv[++i]);
			n = atoi(argv[++i]);
			k = atoi(argv[++i]);
			cout << "-make_macwilliams_system " << q << " " << n << " " << k << endl;
		}
		else if (strcmp(argv[i], "-codes_classify") == 0) {
			f_codes_classify = TRUE;
			cout << "-codes_classify " << endl;
		}
	}
}


void interface_coding_theory::worker(int verbose_level)
{
	if (f_make_macwilliams_system) {
		do_make_macwilliams_system(q, n, k, verbose_level);
	}
	else if (f_codes_classify) {
		do_codes_classify(verbose_level);
	}
}

void interface_coding_theory::do_make_macwilliams_system(int q, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object *M;
	int i, j;

	if (f_v) {
		cout << "interface_coding_theory::do_make_macwilliams_system" << endl;
	}

	D.make_mac_williams_equations(M, n, k, q, verbose_level);

	cout << "\\begin{array}{r|*{" << n << "}{r}}" << endl;
	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			cout << M[i * (n + 1) + j];
			if (j < n) {
				cout << " & ";
			}
		}
		cout << "\\\\" << endl;
	}
	cout << "\\end{array}" << endl;

	cout << "[";
	for (i = 0; i <= n; i++) {
		cout << "[";
		for (j = 0; j <= n; j++) {
			cout << M[i * (n + 1) + j];
			if (j < n) {
				cout << ",";
			}
		}
		cout << "]";
		if (i < n) {
			cout << ",";
		}
	}
	cout << "]" << endl;


	if (f_v) {
		cout << "interface_coding_theory::do_make_macwilliams_system done" << endl;
	}
}

void interface_coding_theory::do_codes_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_coding_theory::do_codes_classify" << endl;
	}

	{
		code_classify cg;

		//cout << argv[0] << endl;
		if (f_v) {
			cout << "interface_coding_theory::do_codes_classify before init" << endl;
		}
		cg.init(argc, argv);
		if (f_v) {
			cout << "interface_coding_theory::do_codes_classify after init" << endl;
		}

		if (f_v) {
			cout << "interface_coding_theory::do_codes_classify before main" << endl;
		}
		cg.main(cg.verbose_level);
		if (f_v) {
			cout << "interface_coding_theory::do_codes_classify after main" << endl;
		}
		cg.F->print_call_stats(cout);


	}

	if (f_v) {
		cout << "interface_coding_theory::do_codes_classify done" << endl;
	}
}

}}

