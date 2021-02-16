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
	f_make_macwilliams_system = FALSE;
	q = 0;
	n = 0;
	k = 0;

	f_upper_bound_for_d = FALSE;

	f_BCH = FALSE;
	f_BCH_dual = FALSE;
	BCH_t = 0;
	//BCH_b = 0;
	f_Hamming_graph = FALSE;
	f_NTT = FALSE;
	//ntt_fname_code = NULL;

	f_general_code_binary = FALSE;
	general_code_binary_n = 0;
	//std::string general_code_binary_text;

	f_linear_code_through_basis = FALSE;
	linear_code_through_basis_n = 0;
	//std::string linear_code_through_basis_text;

	f_long_code = FALSE;
	long_code_n = 0;
	//long_code_generators;

}


void interface_coding_theory::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-make_macwilliams_system") == 0) {
		cout << "-make_macwilliams_system <int : q> <int : n> <int k>" << endl;
	}
	else if (stringcmp(argv[i], "-upper_bound_for_d") == 0) {
		cout << "-upper_bound_for_d <int : n> <int k> <int : q> " << endl;
	}
	else if (stringcmp(argv[i], "-BCH") == 0) {
		cout << "-BCH <int : n> <int : q> <int t>" << endl;
	}
	else if (stringcmp(argv[i], "-BCH_dual") == 0) {
		cout << "-BCH_dual <int : n> <int : q> <int t>" << endl;
	}
	else if (stringcmp(argv[i], "-Hamming_graph") == 0) {
		cout << "-Hamming_graph <int : n> <int : q>" << endl;
	}
	else if (stringcmp(argv[i], "-NTT") == 0) {
		cout << "-NTT <int : n> <int : q> <string : fname_code> " << endl;
	}
	else if (stringcmp(argv[i], "-general_code_binary") == 0) {
		cout << "-general_code_binary <int : n> <string : set> " << endl;
	}
	else if (stringcmp(argv[i], "-linear_code_through_basis") == 0) {
		cout << "-linear_code_through_basis <int : n> <string : set> " << endl;
	}
	else if (stringcmp(argv[i], "-long_code") == 0) {
		cout << "-long_code <int : n> <int : nb_generators=k> <string : generator1> .. <string : generatork>" << endl;
	}
}

int interface_coding_theory::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (i >= argc) {
		return false;
	}
	if (f_v) {
		cout << "interface_coding_theory::recognize_keyword argv[i]=" << argv[i] << " i=" << i << " argc=" << argc << endl;
	}
	if (stringcmp(argv[i], "-make_macwilliams_system") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-upper_bound_for_d") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-BCH") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-BCH_dual") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-Hamming_graph") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-NTT") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-general_code_binary") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-linear_code_through_basis") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-long_code") == 0) {
		return true;
	}
	if (f_v) {
		cout << "interface_coding_theory::recognize_keyword unrecognized" << endl;
	}
	return false;
}

int interface_coding_theory::read_arguments(int argc,
		std::string *argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_coding_theory::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (stringcmp(argv[i], "-make_macwilliams_system") == 0) {
			f_make_macwilliams_system = TRUE;
			q = strtoi(argv[++i]);
			n = strtoi(argv[++i]);
			k = strtoi(argv[++i]);
			cout << "-make_macwilliams_system " << q << " " << n << " " << k << endl;
		}
		else if (stringcmp(argv[i], "-upper_bound_for_d") == 0) {
			f_upper_bound_for_d = TRUE;
			n = strtoi(argv[++i]);
			k = strtoi(argv[++i]);
			q = strtoi(argv[++i]);
			cout << "-upper_bound_for_d " << n << " " << k << " " << q << endl;
		}
		else if (stringcmp(argv[i], "-BCH") == 0) {
			f_BCH = TRUE;
			n = strtoi(argv[++i]);
			q = strtoi(argv[++i]);
			BCH_t = strtoi(argv[++i]);
			//BCH_b = atoi(argv[++i]);
			cout << "-BCH " << n << " " << q << " " << BCH_t << endl;
		}
		else if (stringcmp(argv[i], "-BCH_dual") == 0) {
			f_BCH_dual = TRUE;
			n = strtoi(argv[++i]);
			q = strtoi(argv[++i]);
			BCH_t = strtoi(argv[++i]);
			//BCH_b = atoi(argv[++i]);
			cout << "-BCH " << n << " " << q << " " << BCH_t << endl;
		}
		else if (stringcmp(argv[i], "-Hamming_graph") == 0) {
			f_Hamming_graph = TRUE;
			n = strtoi(argv[++i]);
			q = strtoi(argv[++i]);
			cout << "-Hamming_graph " << n << " " << q << endl;
		}
		else if (stringcmp(argv[i], "-NTT") == 0) {
			f_NTT = TRUE;
			n = strtoi(argv[++i]);
			q = strtoi(argv[++i]);
			ntt_fname_code.assign(argv[++i]);
			cout << "-NTT " << n << " " << q << " " << ntt_fname_code << endl;
		}
		else if (stringcmp(argv[i], "-general_code_binary") == 0) {
			f_general_code_binary = TRUE;
			general_code_binary_n = strtoi(argv[++i]);
			general_code_binary_text.assign(argv[++i]);
			cout << "-general_code_binary " << general_code_binary_n << " "
					<< general_code_binary_text << endl;
		}
		else if (stringcmp(argv[i], "-linear_code_through_basis") == 0) {
			f_linear_code_through_basis = TRUE;
			linear_code_through_basis_n = strtoi(argv[++i]);
			linear_code_through_basis_text.assign(argv[++i]);
			cout << "-linear_code_through_basis " << linear_code_through_basis_n
					<< " " << linear_code_through_basis_text << endl;
		}
		else if (stringcmp(argv[i], "-long_code") == 0) {
			f_long_code = TRUE;
			long_code_n = strtoi(argv[++i]);

			int n, h;
			n = strtoi(argv[++i]);
			for (h = 0; h < n; h++) {
				string s;

				s.assign(argv[++i]);
				if (stringcmp(s, "-set_builder") == 0) {
					set_builder_description Descr;

					cout << "reading -set_builder" << endl;
					i += Descr.read_arguments(argc - (i + 1),
						argv + i + 1, verbose_level);

					cout << "-set_builder" << endl;
					cout << "i = " << i << endl;
					cout << "argc = " << argc << endl;
					if (i < argc) {
						cout << "next argument is " << argv[i] << endl;
					}
					set_builder S;

					S.init(&Descr, verbose_level);

					cout << "set_builder found the following set of size " << S.sz << endl;
					lint_vec_print(cout, S.set, S.sz);
					cout << endl;

					s.assign("");
					int j;
					char str[1000];

					for (j = 0; j < S.sz; j++) {
						if (j) {
							s.append(",");
						}
						sprintf(str, "%ld", S.set[j]);
						s.append(str);
					}
					cout << "as string: " << s << endl;

				}
				long_code_generators.push_back(s);
			}
			cout << "-long_code " << long_code_n << endl;
			for (int h = 0; h < n; h++) {
				cout << " " << long_code_generators[h] << endl;
			}
		}
		else {
			break;
		}
	}
	cout << "interface_coding_theory::read_arguments done" << endl;
	return i;
}


void interface_coding_theory::worker(int verbose_level)
{
	if (f_make_macwilliams_system) {

		coding_theory_domain Coding;

		Coding.do_make_macwilliams_system(q, n, k, verbose_level);
	}
	else if (f_upper_bound_for_d) {

		coding_theory_domain Coding;
		int d_singleton;
		int d_hamming;
		int d_plotkin;
		int d_griesmer;

		d_singleton = Coding.singleton_bound_for_d(n, k, q, verbose_level);
		d_hamming = Coding.hamming_bound_for_d(n, k, q, verbose_level);
		d_plotkin = Coding.plotkin_bound_for_d(n, k, q, verbose_level);
		d_griesmer = Coding.griesmer_bound_for_d(n, k, q, verbose_level);

		cout << "n = " << n << " k=" << k << " q=" << q << endl;

		cout << "d_singleton = " << d_singleton << endl;
		cout << "d_hamming = " << d_hamming << endl;
		cout << "d_plotkin = " << d_plotkin << endl;
		cout << "d_griesmer = " << d_griesmer << endl;

	}
	else if (f_BCH) {

		coding_theory_domain Coding;

		Coding.make_BCH_codes(n, q, BCH_t, 1, FALSE, verbose_level);
	}
	else if (f_BCH_dual) {

		coding_theory_domain Coding;

		Coding.make_BCH_codes(n, q, BCH_t, 1, TRUE, verbose_level);
	}
	else if (f_Hamming_graph) {

		coding_theory_domain Coding;

		Coding.make_Hamming_graph_and_write_file(n, q,
				FALSE /* f_projective*/, verbose_level);
	}
	else if (f_NTT) {
		number_theoretic_transform NTT;

		NTT.init(ntt_fname_code, n, q, verbose_level);
	}
	else if (f_general_code_binary) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory_domain Codes;


			lint_vec_scan(general_code_binary_text, set, sz);

			Codes.investigate_code(set, sz, general_code_binary_n, f_embellish, verbose_level);

			FREE_lint(set);

	}
	else if (f_linear_code_through_basis) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory_domain Codes;


			lint_vec_scan(linear_code_through_basis_text, set, sz);

			Codes.do_linear_code_through_basis(
					linear_code_through_basis_n,
					set, sz /*k*/,
					f_embellish,
					verbose_level);

			FREE_lint(set);

	}
	else if (f_long_code) {
			coding_theory_domain Codes;
			string dummy;

			Codes.do_long_code(
					long_code_n,
					long_code_generators,
					FALSE /* f_nearest_codeword */,
					dummy /* const char *nearest_codeword_text */,
					verbose_level);

	}
}





}}

