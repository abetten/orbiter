/*
 * interface_coding_theory.cpp
 *
 *  Created on: Apr 4, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace user_interface {


interface_coding_theory::interface_coding_theory()
{
	f_make_macwilliams_system = FALSE;
	make_macwilliams_system_q = 0;
	make_macwilliams_system_n = 0;
	make_macwilliams_system_k = 0;

	f_table_of_bounds = FALSE;
	table_of_bounds_n_max = 0;
	table_of_bounds_q = 0;

	f_make_bounds_for_d_given_n_and_k_and_q = FALSE;
	make_bounds_n = 0;
	make_bounds_k = 0;
	make_bounds_q = 0;

	f_BCH = FALSE;
	f_BCH_dual = FALSE;
	BCH_n = 0;
	BCH_q = 0;
	BCH_t = 0;
	//BCH_b = 0;

	f_Hamming_space_distance_matrix = FALSE;
	Hamming_space_n = 0;
	Hamming_space_q = 0;

	f_general_code_binary = FALSE;
	general_code_binary_n = 0;
	//std::string general_code_binary_text;

	f_code_diagram = FALSE;
	//std::string code_diagram_label;
	//std::string code_diagram_codewords_text;
	code_diagram_n = 0;

	f_code_diagram_from_file = FALSE;
	//std::string code_diagram_from_file_codewords_fname;

	f_enhance = FALSE;
	enhance_radius = 0;

	f_metric_balls = FALSE;
	metric_ball_radius = 0;

	f_linear_code_through_basis = FALSE;
	linear_code_through_basis_n = 0;
	//std::string linear_code_through_basis_text;

	f_linear_code_through_columns_of_parity_check_projectively = FALSE;
	f_linear_code_through_columns_of_parity_check = FALSE;
	linear_code_through_columns_of_parity_check_k = 0;
	//std::string linear_code_through_columns_of_parity_check_text;

	f_long_code = FALSE;
	long_code_n = 0;
	//long_code_generators;

	f_encode_text_5bits = FALSE;
	//encode_text_5bits_input;
	//encode_text_5bits_fname;

	f_field_induction = FALSE;
	//std::string field_induction_fname_in;
	//std::string field_induction_fname_out;
	field_induction_nb_bits = 0;

	f_crc32 = FALSE;
	//std::string crc32_text;

	f_crc32_hexdata = FALSE;
	//std::string crc32_hexdata_text;

	f_crc32_test = FALSE;
	crc32_test_block_length = 0;

	f_crc32_remainders = FALSE;
	crc32_remainders_message_length = 0;

	f_crc256_test = FALSE;
	crc256_test_message_length = 0;
	crc256_test_R = 0;
	crc256_test_k = 0;

	f_crc32_file_based = FALSE;
	//std::string crc32_file_based_fname;
	crc32_file_based_block_length = 0;

	f_crc_new_file_based = FALSE;
	//std::string crc_new_file_based_fname;

}


void interface_coding_theory::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-make_macwilliams_system") == 0) {
		cout << "-make_macwilliams_system <int : q> <int : n> <int k>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-table_of_bounds") == 0) {
		cout << "-table_of_bounds <int : n_max> <int : q> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_bounds_for_d_given_n_and_k_and_q") == 0) {
		cout << "-make_bounds_for_d_given_n_and_k_and_q <int : n> <int k> <int : q> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-BCH") == 0) {
		cout << "-BCH <int : n> <int : q> <int t>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-BCH_dual") == 0) {
		cout << "-BCH_dual <int : n> <int : q> <int t>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-Hamming_space_distance_matrix") == 0) {
		cout << "-Hamming_space_distance_matrix <int : n> <int : q>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-general_code_binary") == 0) {
		cout << "-general_code_binary <int : n> <string : set> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-code_diagram") == 0) {
		cout << "-code_diagram <string : label> <string : codewords> <int : n> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-code_diagram_from_file") == 0) {
		cout << "-code_diagram_from_file <string : label> <string : fname_codewords> <int : n> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-enhance") == 0) {
		cout << "-enhance <int : radius>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-metric_balls") == 0) {
		cout << "-metric_balls <int : radius_of_metric_ball> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-linear_code_through_basis") == 0) {
		cout << "-linear_code_through_basis <int : n> <string : set> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-linear_code_through_columns_of_parity_check_projectively") == 0) {
		cout << "-linear_code_through_columns_of_parity_check <int : k> <string : set> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-linear_code_through_columns_of_parity_check") == 0) {
		cout << "-linear_code_through_columns_of_parity_check <int : k> <string : set> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-long_code") == 0) {
		cout << "-long_code <int : n> <int : nb_generators=k> <string : generator_1> .. <string : generator_k>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-encode_text_5bits") == 0) {
		cout << "-encode_text_5bits <string : text> <string : fname>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-field_induction") == 0) {
		cout << "-field_induction <string : fname_in> <string : fname_out> <int : nb_bits>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc32") == 0) {
		cout << "-crc32 <string : text>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc32_hexdata") == 0) {
		cout << "-crc32_hexdata <string : text>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc32_test") == 0) {
		cout << "-crc32_test <int : block_length>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc256_test") == 0) {
		cout << "-crc256_test <int : message_length> <int : R> <int : k>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc32_remainders") == 0) {
		cout << "-crc32_remainders <int : message_length>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc32_file_based") == 0) {
		cout << "-crc32_file_based <string : fname_in> <int : block_length>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-crc_new_file_based") == 0) {
		cout << "-crc_new_file_based <string : fname_in>" << endl;
	}
}

int interface_coding_theory::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (f_v) {
		cout << "interface_coding_theory::recognize_keyword argv[i]="
				<< argv[i] << " i=" << i << " argc=" << argc << endl;
	}
	if (ST.stringcmp(argv[i], "-make_macwilliams_system") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-table_of_bounds") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-make_bounds_for_d_given_n_and_k_and_q") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-BCH") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-BCH_dual") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-Hamming_space_distance_matrix") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-general_code_binary") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-code_diagram") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-code_diagram_from_file") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-enhance") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-metric_balls") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-linear_code_through_basis") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-linear_code_through_columns_of_parity_check_projectively") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-linear_code_through_columns_of_parity_check") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-long_code") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-encode_text_5bits") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-field_induction") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc32") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc32_hexdata") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc32_test") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc256_test") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc32_remainders") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc32_file_based") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-crc_new_file_based") == 0) {
		return true;
	}
	if (f_v) {
		cout << "interface_coding_theory::recognize_keyword not recognizing" << endl;
	}
	return false;
}

void interface_coding_theory::read_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_coding_theory::read_arguments" << endl;
	}



	if (f_v) {
		cout << "interface_coding_theory::read_arguments "
				"the next argument is " << argv[i] << endl;
	}


	if (ST.stringcmp(argv[i], "-make_macwilliams_system") == 0) {
		f_make_macwilliams_system = TRUE;
		make_macwilliams_system_n = ST.strtoi(argv[++i]);
		make_macwilliams_system_k = ST.strtoi(argv[++i]);
		make_macwilliams_system_q = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-make_macwilliams_system " << make_macwilliams_system_n << " " << make_macwilliams_system_k << " " << make_macwilliams_system_q << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-table_of_bounds") == 0) {
		f_table_of_bounds = TRUE;
		table_of_bounds_n_max = ST.strtoi(argv[++i]);
		table_of_bounds_q = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-table_of_bounds " << table_of_bounds_n_max
					<< " " << table_of_bounds_q << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-make_bounds_for_d_given_n_and_k_and_q") == 0) {
		f_make_bounds_for_d_given_n_and_k_and_q = TRUE;
		make_bounds_n = ST.strtoi(argv[++i]);
		make_bounds_k = ST.strtoi(argv[++i]);
		make_bounds_q = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-make_bounds_for_d_given_n_and_k_and_q "
					<< make_bounds_n << " " << make_bounds_k << " " << make_bounds_q << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-BCH") == 0) {
		f_BCH = TRUE;
		BCH_n = ST.strtoi(argv[++i]);
		BCH_q = ST.strtoi(argv[++i]);
		BCH_t = ST.strtoi(argv[++i]);
		//BCH_b = atoi(argv[++i]);
		if (f_v) {
			cout << "-BCH " << BCH_n << " " << BCH_q << " " << BCH_t << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-BCH_dual") == 0) {
		f_BCH_dual = TRUE;
		BCH_n = ST.strtoi(argv[++i]);
		BCH_q = ST.strtoi(argv[++i]);
		BCH_t = ST.strtoi(argv[++i]);
		//BCH_b = atoi(argv[++i]);
		if (f_v) {
			cout << "-BCH " << BCH_n << " " << BCH_q << " " << BCH_t << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-Hamming_space_distance_matrix") == 0) {
		f_Hamming_space_distance_matrix = TRUE;
		Hamming_space_n = ST.strtoi(argv[++i]);
		Hamming_space_q = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-Hamming_space_distance_matrix " << Hamming_space_n << " " << Hamming_space_q << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-general_code_binary") == 0) {
		f_general_code_binary = TRUE;
		general_code_binary_n = ST.strtoi(argv[++i]);
		general_code_binary_text.assign(argv[++i]);
		if (f_v) {
			cout << "-general_code_binary " << general_code_binary_n << " "
					<< general_code_binary_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-code_diagram") == 0) {
		f_code_diagram = TRUE;
		code_diagram_label.assign(argv[++i]);
		code_diagram_codewords_text.assign(argv[++i]);
		code_diagram_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-code_diagram " << code_diagram_label
					<< " " << code_diagram_codewords_text
					<< " " << code_diagram_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-code_diagram_from_file") == 0) {
		f_code_diagram_from_file = TRUE;
		code_diagram_label.assign(argv[++i]);
		code_diagram_from_file_codewords_fname.assign(argv[++i]);
		code_diagram_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-code_diagram_from_file " << code_diagram_label
					<< " " << code_diagram_from_file_codewords_fname
					<< " " << code_diagram_n << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-enhance") == 0) {
		f_enhance = TRUE;
		enhance_radius = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-enhance " << enhance_radius << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-metric_balls") == 0) {
		f_metric_balls = TRUE;
		metric_ball_radius = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-metric_balls " << metric_ball_radius << endl;
		}
	}



	else if (ST.stringcmp(argv[i], "-linear_code_through_basis") == 0) {
		f_linear_code_through_basis = TRUE;
		linear_code_through_basis_n = ST.strtoi(argv[++i]);
		linear_code_through_basis_text.assign(argv[++i]);
		if (f_v) {
			cout << "-linear_code_through_basis " << linear_code_through_basis_n
					<< " " << linear_code_through_basis_text << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-linear_code_through_columns_of_parity_check_projectively") == 0) {
		f_linear_code_through_columns_of_parity_check_projectively = TRUE;
		linear_code_through_columns_of_parity_check_k = ST.strtoi(argv[++i]);
		linear_code_through_columns_of_parity_check_text.assign(argv[++i]);
		if (f_v) {
			cout << "-linear_code_through_columns_of_parity_check_projectively "
				<< linear_code_through_columns_of_parity_check_k
				<< " " << linear_code_through_columns_of_parity_check_text << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-linear_code_through_columns_of_parity_check") == 0) {
		f_linear_code_through_columns_of_parity_check = TRUE;
		linear_code_through_columns_of_parity_check_k = ST.strtoi(argv[++i]);
		linear_code_through_columns_of_parity_check_text.assign(argv[++i]);
		if (f_v) {
			cout << "-linear_code_through_columns_of_parity_check "
				<< linear_code_through_columns_of_parity_check_k
				<< " " << linear_code_through_columns_of_parity_check_text << endl;
		}
	}

	else if (ST.stringcmp(argv[i], "-long_code") == 0) {
		f_long_code = TRUE;
		long_code_n = ST.strtoi(argv[++i]);

		int n, h;
		n = ST.strtoi(argv[++i]);
		for (h = 0; h < n; h++) {
			string s;

			s.assign(argv[++i]);
#if 0
			if (stringcmp(s, "-set_builder") == 0) {
				set_builder_description Descr;

				if (f_v) {
					cout << "reading -set_builder" << endl;
				}
				i += Descr.read_arguments(argc - (i + 1),
					argv + i + 1, verbose_level);

				if (f_v) {
					cout << "-set_builder" << endl;
					cout << "i = " << i << endl;
					cout << "argc = " << argc << endl;
					if (i < argc) {
						cout << "next argument is " << argv[i] << endl;
					}
				}

				set_builder S;

				S.init(&Descr, verbose_level);

				if (f_v) {
					cout << "set_builder found the following set of size " << S.sz << endl;
					Orbiter->Lint_vec.print(cout, S.set, S.sz);
					cout << endl;
				}

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
				if (f_v) {
					cout << "as string: " << s << endl;
				}

			}
#endif
			long_code_generators.push_back(s);
		}
		if (f_v) {
			cout << "-long_code " << long_code_n << endl;
			for (int h = 0; h < n; h++) {
				cout << " " << long_code_generators[h] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-encode_text_5bits") == 0) {
		f_encode_text_5bits = TRUE;
		encode_text_5bits_input.assign(argv[++i]);
		encode_text_5bits_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-encode_text_5bits " << encode_text_5bits_input << " "
					<< encode_text_5bits_fname << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-field_induction") == 0) {
		f_field_induction = TRUE;
		field_induction_fname_in.assign(argv[++i]);
		field_induction_fname_out.assign(argv[++i]);
		field_induction_nb_bits = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-field_induction " << field_induction_fname_in
					<< " " << field_induction_fname_out
					<< " " << field_induction_nb_bits
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc32") == 0) {
		f_crc32 = TRUE;
		crc32_text.assign(argv[++i]);
		if (f_v) {
			cout << "-crc32 " << crc32_text
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc32_hexdata") == 0) {
		f_crc32_hexdata = TRUE;
		crc32_hexdata_text.assign(argv[++i]);
		if (f_v) {
			cout << "-crc32_hexdata " << crc32_hexdata_text
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc32_test") == 0) {
		f_crc32_test = TRUE;
		crc32_test_block_length = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-crc32_test " << crc32_test_block_length << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc256_test") == 0) {
		f_crc256_test = TRUE;
		crc256_test_message_length = ST.strtoi(argv[++i]);
		crc256_test_R = ST.strtoi(argv[++i]);
		crc256_test_k = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-crc256_test " << crc256_test_message_length << " " << crc256_test_R << " " << crc256_test_k << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc32_remainders") == 0) {
		f_crc32_remainders = TRUE;
		crc32_remainders_message_length = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-crc32_remainders " << crc32_remainders_message_length << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc32_file_based") == 0) {
		f_crc32_file_based = TRUE;
		crc32_file_based_fname.assign(argv[++i]);
		crc32_file_based_block_length = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-crc32_file_based " << crc32_file_based_fname
					<< " " << crc32_file_based_block_length << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-crc_new_file_based") == 0) {
		f_crc_new_file_based = TRUE;
		crc_new_file_based_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-crc_new_file_based "
					<< crc_new_file_based_fname
					<< endl;
		}
	}

	if (f_v) {
		cout << "interface_coding_theory::read_arguments done" << endl;
	}
}


void interface_coding_theory::print()
{
	if (f_make_macwilliams_system) {
		cout << "-make_macwilliams_system " << make_macwilliams_system_n << " " << make_macwilliams_system_k << " " << make_macwilliams_system_q << endl;
	}
	if (f_table_of_bounds) {
		cout << "-table_of_bounds " << table_of_bounds_n_max << " " << table_of_bounds_q << endl;
	}
	if (f_make_bounds_for_d_given_n_and_k_and_q) {
		cout << "-make_bounds_for_d_given_n_and_k_and_q " << make_bounds_n << " " << make_bounds_k << " " << make_bounds_q << endl;
	}
	if (f_BCH) {
		cout << "-BCH " << BCH_n << " " << BCH_q << " " << BCH_t << endl;
	}
	if (f_BCH_dual) {
		cout << "-BCH " << BCH_n << " " << BCH_q << " " << BCH_t << endl;
	}
	if (f_Hamming_space_distance_matrix) {
		cout << "-Hamming_space_distance_matrix " << Hamming_space_n << " " << Hamming_space_q << endl;
	}
	if (f_general_code_binary) {
		cout << "-general_code_binary " << general_code_binary_n << " "
				<< general_code_binary_text << endl;
	}
	if (f_code_diagram) {
		cout << "-code_diagram " << code_diagram_label
				<< " " << code_diagram_codewords_text
				<< " " << code_diagram_n << endl;
	}
	if (f_code_diagram_from_file) {
		cout << "-code_diagram_from_file " << code_diagram_label
				<< " " << code_diagram_from_file_codewords_fname
				<< " " << code_diagram_n << endl;
	}

	if (f_enhance) {
		cout << "-enhance " << enhance_radius << endl;
	}

	if (f_metric_balls) {
		cout << "-metric_balls " << metric_ball_radius << endl;
	}



	if (f_linear_code_through_basis) {
		cout << "-linear_code_through_basis " << linear_code_through_basis_n
				<< " " << linear_code_through_basis_text << endl;
	}

	if (f_linear_code_through_columns_of_parity_check_projectively) {
		cout << "-linear_code_through_columns_of_parity_check_projectively "
				<< linear_code_through_columns_of_parity_check_k
				<< " " << linear_code_through_columns_of_parity_check_text
				<< endl;
	}

	if (f_linear_code_through_columns_of_parity_check) {
		cout << "-linear_code_through_columns_of_parity_check "
				<< linear_code_through_columns_of_parity_check_k
				<< " " << linear_code_through_columns_of_parity_check_text
				<< endl;
	}

	if (f_long_code) {
		cout << "-long_code " << long_code_n << endl;
		for (int h = 0; h < long_code_n; h++) {
			cout << " " << long_code_generators[h] << endl;
		}
	}
	if (f_encode_text_5bits) {
		cout << "-encode_text_5bits " << encode_text_5bits_input << " "
				<< encode_text_5bits_fname << endl;
	}
	if (f_field_induction) {
		cout << "-field_induction " << field_induction_fname_in
				<< " " << field_induction_fname_out
				<< " " << field_induction_nb_bits
				<< endl;
	}
	if (f_crc32) {
		cout << "-crc32 " << crc32_text
				<< endl;
	}
	if (f_crc32_hexdata) {
		cout << "-crc32_hexdata " << crc32_hexdata_text << endl;
	}
	if (f_crc32_test) {
		cout << "-crc32_test " << crc32_test_block_length << endl;
	}
	if (f_crc256_test) {
		cout << "-crc256_test " << crc256_test_message_length << " " << crc256_test_R << " " << crc256_test_k << endl;
	}
	if (f_crc32_remainders) {
		cout << "-crc32_remainders " << crc32_remainders_message_length << endl;
	}
	if (f_crc32_file_based) {
		cout << "-crc32_file_based " << crc32_file_based_fname
				<< " " << crc32_file_based_block_length << endl;
	}
	if (f_crc_new_file_based) {
		cout << "-crc_new_file_based "
				<< crc_new_file_based_fname
				<< endl;
	}
}


void interface_coding_theory::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_coding_theory::worker" << endl;
	}

	if (f_make_macwilliams_system) {

		coding_theory::coding_theory_domain Coding;

		Coding.do_make_macwilliams_system(make_macwilliams_system_q, make_macwilliams_system_n, make_macwilliams_system_k, verbose_level);
	}
	else if (f_table_of_bounds) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_table_of_bounds(table_of_bounds_n_max, table_of_bounds_q, verbose_level);
	}
	else if (f_make_bounds_for_d_given_n_and_k_and_q) {

		coding_theory::coding_theory_domain Coding;
		int d_GV;
		int d_singleton;
		int d_hamming;
		int d_plotkin;
		int d_griesmer;

		d_GV = Coding.gilbert_varshamov_lower_bound_for_d(make_bounds_n, make_bounds_k, make_bounds_q, verbose_level);
		d_singleton = Coding.singleton_bound_for_d(make_bounds_n, make_bounds_k, make_bounds_q, verbose_level);
		d_hamming = Coding.hamming_bound_for_d(make_bounds_n, make_bounds_k, make_bounds_q, verbose_level);
		d_plotkin = Coding.plotkin_bound_for_d(make_bounds_n, make_bounds_k, make_bounds_q, verbose_level);
		d_griesmer = Coding.griesmer_bound_for_d(make_bounds_n, make_bounds_k, make_bounds_q, verbose_level);

		cout << "n = " << make_bounds_n << " k=" << make_bounds_k << " q=" << make_bounds_q << endl;

		cout << "d_GV = " << d_GV << endl;
		cout << "d_singleton = " << d_singleton << endl;
		cout << "d_hamming = " << d_hamming << endl;
		cout << "d_plotkin = " << d_plotkin << endl;
		cout << "d_griesmer = " << d_griesmer << endl;

	}
	else if (f_BCH) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_BCH_codes(BCH_n, BCH_q, BCH_t, 1, FALSE, verbose_level);
	}
	else if (f_BCH_dual) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_BCH_codes(BCH_n, BCH_q, BCH_t, 1, TRUE, verbose_level);
	}
	else if (f_Hamming_space_distance_matrix) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_Hamming_graph_and_write_file(Hamming_space_n, Hamming_space_q,
				FALSE /* f_projective*/, verbose_level);
	}
	else if (f_general_code_binary) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory::coding_theory_domain Codes;


			Lint_vec_scan(general_code_binary_text, set, sz);

			Codes.investigate_code(set, sz, general_code_binary_n, f_embellish, verbose_level);

			FREE_lint(set);

	}

	else if (f_code_diagram) {
			long int *codewords;
			int nb_words;

			coding_theory::coding_theory_domain Codes;


			Lint_vec_scan(code_diagram_codewords_text, codewords, nb_words);



			Codes.code_diagram(
					code_diagram_label,
					codewords,
					nb_words, code_diagram_n, f_metric_balls, metric_ball_radius,
					f_enhance, 0 /*nb_enhance */,
					verbose_level);
	}

	else if (f_code_diagram_from_file) {
			long int *codewords;
			int m, nb_words;
			orbiter_kernel_system::file_io Fio;

			coding_theory::coding_theory_domain Codes;


			Fio.lint_matrix_read_csv(code_diagram_from_file_codewords_fname, codewords, m, nb_words, verbose_level);



			Codes.code_diagram(
					code_diagram_label,
					codewords,
					nb_words, code_diagram_n, f_metric_balls, metric_ball_radius,
					f_enhance, enhance_radius,
					verbose_level);
	}

	else if (f_code_diagram_from_file) {
			long int *codewords;
			int nb_words;

			coding_theory::coding_theory_domain Codes;


			Lint_vec_scan(code_diagram_codewords_text, codewords, nb_words);



			Codes.code_diagram(
					code_diagram_label,
					codewords,
					nb_words, code_diagram_n, f_metric_balls, metric_ball_radius,
					f_enhance, enhance_radius,
					verbose_level);
	}

	else if (f_linear_code_through_basis) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory::coding_theory_domain Codes;


			//Orbiter->Lint_vec.scan(linear_code_through_basis_text, set, sz);
			orbiter_kernel_system::Orbiter->get_lint_vector_from_label(linear_code_through_basis_text, set, sz, verbose_level);

			Codes.do_linear_code_through_basis(
					linear_code_through_basis_n,
					set, sz /*k*/,
					f_embellish,
					verbose_level);

			FREE_lint(set);

	}

	else if (f_linear_code_through_columns_of_parity_check_projectively) {
			long int *set;
			int n;

			coding_theory::coding_theory_domain Codes;


			//Orbiter->Lint_vec.scan(linear_code_through_columns_of_parity_check_text, set, n);
			orbiter_kernel_system::Orbiter->get_lint_vector_from_label(linear_code_through_columns_of_parity_check_text, set, n, verbose_level);

			Codes.do_linear_code_through_columns_of_parity_check_projectively(
					n,
					set, linear_code_through_columns_of_parity_check_k /*k*/,
					verbose_level);

			FREE_lint(set);

	}


	else if (f_linear_code_through_columns_of_parity_check) {
			long int *set;
			int n;

			coding_theory::coding_theory_domain Codes;


			//Orbiter->Lint_vec.scan(linear_code_through_columns_of_parity_check_text, set, n);
			orbiter_kernel_system::Orbiter->get_lint_vector_from_label(linear_code_through_columns_of_parity_check_text, set, n, verbose_level);

			Codes.do_linear_code_through_columns_of_parity_check(
					n,
					set, linear_code_through_columns_of_parity_check_k /*k*/,
					verbose_level);

			FREE_lint(set);

	}

	else if (f_long_code) {
		coding_theory::coding_theory_domain Codes;
			string dummy;

			Codes.do_long_code(
					long_code_n,
					long_code_generators,
					FALSE /* f_nearest_codeword */,
					dummy /* const char *nearest_codeword_text */,
					verbose_level);

	}
	else if (f_encode_text_5bits) {
		coding_theory::coding_theory_domain Codes;

		Codes.encode_text_5bits(encode_text_5bits_input,
				encode_text_5bits_fname,
				verbose_level);

	}
	else if (f_field_induction) {
		coding_theory::coding_theory_domain Codes;

		Codes.field_induction(field_induction_fname_in,
				field_induction_fname_out,
				field_induction_nb_bits,
				verbose_level);

	}
	else if (f_crc32) {
		cout << "-crc32 " << crc32_text
				<< endl;

		coding_theory::coding_theory_domain Codes;
		uint32_t a;

		a = Codes.crc32(crc32_text.c_str(), crc32_text.length());
		cout << "CRC value of " << crc32_text << " is ";

		data_structures::algorithms Algo;

		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (f_crc32_hexdata) {
		cout << "-crc32_hexdata " << crc32_hexdata_text
				<< endl;

		coding_theory::coding_theory_domain Codes;
		data_structures::algorithms Algo;
		uint32_t a;
		char *data;
		int data_size;

		cout << "before Algo.read_hex_data" << endl;
		Algo.read_hex_data(crc32_hexdata_text, data, data_size, verbose_level - 2);
		cout << "after Algo.read_hex_data" << endl;


		int i;
		cout << "data:" << endl;
		for (i = 0; i < data_size; i++) {
			cout << i << " : " << (int) data[i] << endl;
		}
		cout << "data:" << endl;
		for (i = 0; i < data_size; i++) {
			cout << "*";
			Algo.print_repeated_character(cout, '0', 7);
		}
		cout << endl;
		Algo.print_bits(cout, data, data_size);
		cout << endl;


		a = Codes.crc32(data, data_size);
		cout << "CRC value of 0x" << crc32_hexdata_text << " is ";


		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (f_crc32_test) {
		cout << "-crc32_test "
				<< crc32_test_block_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc32_test(crc32_test_block_length, verbose_level - 1);

	}
	else if (f_crc256_test) {
		cout << "-crc256_test "
				<< crc256_test_message_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc256_test_k_subsets(
				crc256_test_message_length,
				crc256_test_R,
				crc256_test_k,
				verbose_level - 1);

	}
	else if (f_crc32_remainders) {
		cout << "-crc32_remainders "
				<< crc32_remainders_message_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc32_remainders(
				crc32_remainders_message_length,
				verbose_level - 1);

	}
	else if (f_crc32_file_based) {
		cout << "-crc32_file_based " << crc32_file_based_fname
				<< " " << crc32_file_based_block_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc32_file_based(crc32_file_based_fname, crc32_file_based_block_length, verbose_level - 1);

	}
	else if (f_crc_new_file_based) {
		cout << "-crc_new_file_based " << crc_new_file_based_fname
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc771_file_based(crc_new_file_based_fname, verbose_level - 1);

	}


	if (f_v) {
		cout << "interface_coding_theory::worker done" << endl;
	}
}



}}}


