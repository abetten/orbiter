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
	f_BCH = FALSE;
	f_BCH_dual = FALSE;
	BCH_t = 0;
	//BCH_b = 0;
	f_Hamming_graph = FALSE;
	f_NTT = FALSE;
	//ntt_fname_code = NULL;
	f_draw_matrix = FALSE;
	bit_depth = 8;
	//fname = NULL;
	box_width = 0;
	f_draw_matrix_partition = FALSE;
	draw_matrix_partition_width = 0;
	//std::string draw_matrix_partition_rows;
	//std::string draw_matrix_partition_cols;

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
	else if (stringcmp(argv[i], "-draw_matrix") == 0) {
		cout << "-draw_matrix <string : fname> <int : box_width> <int : bit_depth>" << endl;
	}
	else if (stringcmp(argv[i], "-draw_matrix_partition") == 0) {
		cout << "-draw_matrix_partition <int : width> "
				"<string : row partition> <string : col partition> " << endl;
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

	if (f_v) {
		cout << "interface_coding_theory::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-make_macwilliams_system") == 0) {
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
	else if (stringcmp(argv[i], "-draw_matrix") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-draw_matrix_partition") == 0) {
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
		else if (stringcmp(argv[i], "-draw_matrix") == 0) {
			f_draw_matrix = TRUE;
			fname.assign(argv[++i]);
			box_width = strtoi(argv[++i]);
			bit_depth = strtoi(argv[++i]);
			cout << "-draw_matrix " << fname << " " << box_width << " " << bit_depth << endl;
		}
		else if (stringcmp(argv[i], "-draw_matrix_partition") == 0) {
			f_draw_matrix_partition = TRUE;
			draw_matrix_partition_width = strtoi(argv[++i]);
			draw_matrix_partition_rows.assign(argv[++i]);
			draw_matrix_partition_cols.assign(argv[++i]);
			cout << "-draw_matrix_partition " << draw_matrix_partition_rows
					<< " " << draw_matrix_partition_cols << endl;
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
	return i;
}


void interface_coding_theory::worker(int verbose_level)
{
	if (f_make_macwilliams_system) {

		coding_theory_domain Coding;

		Coding.do_make_macwilliams_system(q, n, k, verbose_level);
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
	else if (f_draw_matrix) {
		file_io Fio;
		int *M;
		int m, n;

		Fio.int_matrix_read_csv(fname, M, m, n, verbose_level);

		if (f_draw_matrix_partition) {
			int *row_parts;
			int *col_parts;
			int nb_row_parts;
			int nb_col_parts;

			int_vec_scan(draw_matrix_partition_rows, row_parts, nb_row_parts);
			int_vec_scan(draw_matrix_partition_cols, col_parts, nb_col_parts);
			draw_bitmap(fname, M, m, n,
					TRUE, draw_matrix_partition_width, // int f_partition, int part_width,
					nb_row_parts, row_parts, nb_col_parts, col_parts, // int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
					TRUE /* f_box_width */, box_width,
					FALSE /* f_invert_colors */, bit_depth,
					verbose_level);
			FREE_int(row_parts);
			FREE_int(col_parts);
		}
		else {
			draw_bitmap(fname, M, m, n,
					FALSE, 0, // int f_partition, int part_width,
					0, NULL, 0, NULL, // int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
					TRUE /* f_box_width */, box_width,
					FALSE /* f_invert_colors */, bit_depth,
					verbose_level);
		}
		FREE_int(M);
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

