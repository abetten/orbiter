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
	//argc = 0;
	//argv = NULL;

	f_make_macwilliams_system = FALSE;
	q = 0;
	n = 0;
	k = 0;
	f_BCH = FALSE;
	f_BCH_dual = FALSE;
	BCH_t = 0;
	//BCH_b = 0;
	f_Hamming_graph = FALSE;
	f_NTT = FALSE;
	ntt_fname_code = NULL;
	f_draw_matrix = FALSE;
	bit_depth = 8;
	//fname = NULL;
	box_width = 0;
}


void interface_coding_theory::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
		cout << "-make_macwilliams_system <int : q> <int : n> <int k>" << endl;
	}
	else if (strcmp(argv[i], "-BCH") == 0) {
		cout << "-BCH <int : n> <int : q> <int t>" << endl;
	}
	else if (strcmp(argv[i], "-BCH_dual") == 0) {
		cout << "-BCH_dual <int : n> <int : q> <int t>" << endl;
	}
	else if (strcmp(argv[i], "-Hamming_graph") == 0) {
		cout << "-Hamming_graph <int : n> <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-NTT") == 0) {
		cout << "-NTT <int : n> <int : q> <string : fname_code> " << endl;
	}
	else if (strcmp(argv[i], "-draw_matrix") == 0) {
		cout << "-draw_matrix <string : fname> <int : box_width> <int : bit_depth>" << endl;
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
	else if (strcmp(argv[i], "-BCH") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-BCH_dual") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-Hamming_graph") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-NTT") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-draw_matrix") == 0) {
		return true;
	}
	return false;
}

void interface_coding_theory::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_coding_theory::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-make_macwilliams_system") == 0) {
			f_make_macwilliams_system = TRUE;
			q = atoi(argv[++i]);
			n = atoi(argv[++i]);
			k = atoi(argv[++i]);
			cout << "-make_macwilliams_system " << q << " " << n << " " << k << endl;
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
		else if (strcmp(argv[i], "-Hamming_graph") == 0) {
			f_Hamming_graph = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-Hamming_graph " << n << " " << q << endl;
		}
		else if (strcmp(argv[i], "-NTT") == 0) {
			f_NTT = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			ntt_fname_code = argv[++i];
			cout << "-NTT " << n << " " << q << " " << ntt_fname_code << endl;
		}
		else if (strcmp(argv[i], "-draw_matrix") == 0) {
			f_draw_matrix = TRUE;
			fname.assign(argv[++i]);
			box_width = atoi(argv[++i]);
			bit_depth = atoi(argv[++i]);
			cout << "-draw_matrix " << fname << " " << box_width << " " << bit_depth << endl;
		}
	}
}


void interface_coding_theory::worker(int verbose_level)
{
	if (f_make_macwilliams_system) {
		do_make_macwilliams_system(q, n, k, verbose_level);
	}
	else if (f_BCH) {
		make_BCH_codes(n, q, BCH_t, 1, FALSE, verbose_level);
	}
	else if (f_BCH_dual) {
		make_BCH_codes(n, q, BCH_t, 1, TRUE, verbose_level);
	}
	else if (f_Hamming_graph) {

		algebra_global Algebra;

		Algebra.make_Hamming_graph_and_write_file(n, q,
				FALSE /* f_projective*/, verbose_level);
	}
	else if (f_NTT) {
		number_theoretic_transform NTT;

		NTT.init(ntt_fname_code, n, q, verbose_level);
	}
	else if (f_draw_matrix) {
		file_io Fio;
		int *M;
		int m, n;

		Fio.int_matrix_read_csv(fname.c_str(), M, m, n, verbose_level);
		draw_bitmap(fname, M, m, n,
				FALSE, 0, // int f_partition, int part_width,
				0, NULL, 0, NULL, // int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
				TRUE /* f_box_width */, box_width,
				FALSE /* f_invert_colors */, bit_depth,
				verbose_level);
		FREE_int(M);
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
	snprintf(fname, 1000, "BCH_%d_%d.txt", n, t);

	cout << "roots: ";
	int_vec_print(cout, roots, nb_roots);
	cout << endl;

	coding_theory_domain Codes;

	string dummy;

	dummy.assign("");

	Codes.make_cyclic_code(n, q, t, roots, nb_roots,
			FALSE /*f_poly*/, dummy /*poly*/, f_dual,
			fname, verbose_level);

	FREE_int(roots);

	if (f_v) {
		cout << "interface_coding_theory::make_BCH_codes done" << endl;
	}
}



}}

