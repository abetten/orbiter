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
	f_BCH = FALSE;
	f_BCH_dual = FALSE;
	BCH_t = 0;
	//BCH_b = 0;
}


void interface_coding_theory::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		cout << "-make_macwilliams_system <int : q> <int : n> <int k>" << endl;
	}
	else if (strcmp(argv[i], "-code_classify") == 0) {
		cout << "-code_classify" << endl;
	}
	else if (strcmp(argv[i], "-BCH") == 0) {
		cout << "-BCH <int : n> <int : q> <int t>" << endl;
	}
	else if (strcmp(argv[i], "-BCH_dual") == 0) {
		cout << "-BCH_dual <int : n> <int : q> <int t>" << endl;
	}
}

int interface_coding_theory::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-codes_classify") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-BCH") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-BCH_dual") == 0) {
		return true;
	}
	return false;
}

void interface_coding_theory::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
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
		else if (strcmp(argv[i], "-BCH") == 0) {
			f_BCH = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			BCH_t = atoi(argv[++i]);
			//BCH_b = atoi(argv[++i]);
			cout << "-BCH " << n << " " << q << " " << BCH_t << endl;
		}
		else if (strcmp(argv[i], "-BCH_dual") == 0) {
			f_BCH_dual = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			BCH_t = atoi(argv[++i]);
			//BCH_b = atoi(argv[++i]);
			cout << "-BCH " << n << " " << q << " " << BCH_t << endl;
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
	else if (f_BCH) {
		make_BCH_codes(n, q, BCH_t, 1, FALSE, verbose_level);
	}
	else if (f_BCH_dual) {
		make_BCH_codes(n, q, BCH_t, 1, TRUE, verbose_level);
	}
}

void interface_coding_theory::do_make_macwilliams_system(
		int q, int n, int k, int verbose_level)
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

		poset_classification_control *Control = NULL;

		Control = NEW_OBJECT(poset_classification_control);

		cg.init(Control);
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


void interface_coding_theory::make_BCH_codes(int n, int q, int t, int b, int f_dual, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_coding_theory::make_BCH_codes" << endl;
	}

	char fname[1000];
	number_theory_domain NT;
	int *roots;
	int nb_roots;
	int i, j;

	roots = NEW_int(t - 1);
	nb_roots = t - 1;
	for (i = 0; i < t - 1; i++) {
		j = NT.mod(b + i, n);
		roots[i] = j;
		}
	sprintf(fname, "BCH_%d_%d.txt", n, t);

	cout << "roots: ";
	int_vec_print(cout, roots, nb_roots);
	cout << endl;

	coding_theory_domain Codes;


	Codes.make_cyclic_code(n, q, t, roots, nb_roots,
			FALSE /*f_poly*/, NULL /*poly*/, f_dual,
			fname, verbose_level);

	FREE_int(roots);

	if (f_v) {
		cout << "interface_coding_theory::make_BCH_codes done" << endl;
	}
}



}}

