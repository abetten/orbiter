/*
 * coding_theoretic_activity_description.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


coding_theoretic_activity_description::coding_theoretic_activity_description()
{

#if 0
	f_BCH = FALSE;
	f_BCH_dual = FALSE;
	BCH_n = 0;
	BCH_q = 0;
	BCH_t = 0;
	//BCH_b = 0;
#endif

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

	f_weight_enumerator = FALSE;

	f_minimum_distance = FALSE;
	//std::string minimum_distance_code_label;

	f_generator_matrix_cyclic_code = FALSE;
	generator_matrix_cyclic_code_n = 0;
	//std::string generator_matrix_cyclic_code_poly

	f_nth_roots = FALSE;
	nth_roots_n = 0;

#if 0
	f_make_BCH_code = FALSE;
	make_BCH_code_n = 0;
	make_BCH_code_d = 0;

	f_make_BCH_code_and_encode = FALSE;
	make_BCH_code_and_encode_n = 0;
	make_BCH_code_and_encode_d = 0;
	//std::string make_BCH_code_and_encode_text;
	//std::string make_BCH_code_and_encode_fname;
#endif

	f_NTT = FALSE;
	NTT_n = 0;
	NTT_q = 0;

	f_fixed_code = FALSE;
	//std::string fixed_code_perm;

	f_export_magma = FALSE;
	//std::string export_magma_fname;

	f_export_codewords = FALSE;
	//std::string export_codewords_fname;

	f_export_codewords_by_weight = FALSE;
	//std::string export_codewords_by_weight_fname;

	f_export_genma = FALSE;
	//std::string export_genma_fname;

	f_export_checkma = FALSE;
	//std::string export_checkma_fname;


	f_crc32 = FALSE;
	//std::string crc32_text;

	f_crc32_hexdata = FALSE;
	//std::string crc32_hexdata_text;

	f_crc32_test = FALSE;
	crc32_test_block_length = 0;

	f_crc256_test = FALSE;
	crc256_test_message_length = 0;
	crc256_test_R = 0;
	crc256_test_k = 0;

	f_crc32_remainders = FALSE;
	crc32_remainders_message_length = 0;

	f_crc_encode_file_based = FALSE;
	//std::string crc_encode_file_based_fname_in;
	//std::string crc_encode_file_based_fname_out;
	//std::string crc_encode_file_based_crc_type;
	crc_encode_file_based_block_length = 0;

#if 0
	f_crc_new_file_based = FALSE;
	//std::string crc_new_file_based_fname;
#endif

	f_find_CRC_polynomials = FALSE;
	find_CRC_polynomials_nb_errors = 0;
	find_CRC_polynomials_information_bits = 0;
	find_CRC_polynomials_check_bits = 0;

	f_write_code_for_division = FALSE;
	//std::string write_code_for_division_fname;
	//std::string write_code_for_division_A;
	//std::string write_code_for_division_B;

	f_polynomial_division_from_file = FALSE;
	//std::string polynomial_division_from_file_fname;
	polynomial_division_from_file_r1 = 0;

	f_polynomial_division_from_file_all_k_bit_error_patterns = FALSE;
	//std::string polynomial_division_from_file_all_k_bit_error_patterns_fname;
	polynomial_division_from_file_all_k_bit_error_patterns_r1 = 0;
	polynomial_division_from_file_all_k_bit_error_patterns_k = 0;

}

coding_theoretic_activity_description::~coding_theoretic_activity_description()
{
}


int coding_theoretic_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "coding_theoretic_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


#if 0
		if (ST.stringcmp(argv[i], "-BCH") == 0) {
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
#endif

		if (ST.stringcmp(argv[i], "-general_code_binary") == 0) {
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
						snprintf(str, sizeof(str), "%ld", S.set[j]);
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
		else if (ST.stringcmp(argv[i], "-weight_enumerator") == 0) {
			f_weight_enumerator = TRUE;
			if (f_v) {
				cout << "-weight_enumerator " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-minimum_distance") == 0) {
			f_minimum_distance = TRUE;
			minimum_distance_code_label.assign(argv[++i]);
			if (f_v) {
				cout << "-minimum_distance " << minimum_distance_code_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-generator_matrix_cyclic_code") == 0) {
			f_generator_matrix_cyclic_code = TRUE;
			generator_matrix_cyclic_code_n = ST.strtoi(argv[++i]);
			generator_matrix_cyclic_code_poly.assign(argv[++i]);
			if (f_v) {
				cout << "-generator_matrix_cyclic_code " << generator_matrix_cyclic_code_n
					<< " " << generator_matrix_cyclic_code_poly
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nth_roots") == 0) {
			f_nth_roots = TRUE;
			nth_roots_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nth_roots " << nth_roots_n << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-make_BCH_code") == 0) {
			f_make_BCH_code = TRUE;
			make_BCH_code_n = ST.strtoi(argv[++i]);
			make_BCH_code_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-make_BCH_code "
						<< " " << make_BCH_code_n
						<< " " << make_BCH_code_d
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-make_BCH_code_and_encode") == 0) {
			f_make_BCH_code_and_encode = TRUE;
			make_BCH_code_and_encode_n = ST.strtoi(argv[++i]);
			make_BCH_code_and_encode_d = ST.strtoi(argv[++i]);
			make_BCH_code_and_encode_text.assign(argv[++i]);
			make_BCH_code_and_encode_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-make_BCH_code_and_encode "
						<< " " << make_BCH_code_and_encode_n
						<< " " << make_BCH_code_and_encode_d
						<< " " << make_BCH_code_and_encode_text
						<< " " << make_BCH_code_and_encode_fname
						<< endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-NTT") == 0) {
			f_NTT = TRUE;
			NTT_n = ST.strtoi(argv[++i]);
			NTT_q = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-NTT " << NTT_n
						<< " " << NTT_q
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fixed_code") == 0) {
			f_fixed_code = TRUE;
			fixed_code_perm.assign(argv[++i]);
			if (f_v) {
				cout << "-fixed_code "
					<< " " << fixed_code_perm
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			export_magma_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_magma "
					<< " " << export_magma_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_codewords") == 0) {
			f_export_codewords = TRUE;
			export_codewords_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_codewords "
					<< " " << export_codewords_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_codewords_by_weight") == 0) {
			f_export_codewords_by_weight = TRUE;
			export_codewords_by_weight_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_codewords_by_weight "
					<< " " << export_codewords_by_weight_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_genma") == 0) {
			f_export_genma = TRUE;
			export_genma_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_genma "
					<< " " << export_genma_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_checkma") == 0) {
			f_export_checkma = TRUE;
			export_checkma_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_checkma "
					<< " " << export_checkma_fname
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
		else if (ST.stringcmp(argv[i], "-crc_encode_file_based") == 0) {
			f_crc_encode_file_based = TRUE;
			crc_encode_file_based_fname_in.assign(argv[++i]);
			crc_encode_file_based_fname_out.assign(argv[++i]);
			crc_encode_file_based_crc_type.assign(argv[++i]);
			crc_encode_file_based_block_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-crc_encode_file_based "
						<< crc_encode_file_based_fname_in << " "
						<< crc_encode_file_based_fname_out << " "
						<< crc_encode_file_based_crc_type << " "
						<< crc_encode_file_based_block_length << endl;
			}
		}

#if 0
		else if (ST.stringcmp(argv[i], "-crc_new_file_based") == 0) {
			f_crc_new_file_based = TRUE;
			crc_new_file_based_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-crc_new_file_based "
						<< crc_new_file_based_fname
						<< endl;
			}
		}
#endif

		else if (ST.stringcmp(argv[i], "-find_CRC_polynomials") == 0) {
			f_find_CRC_polynomials = TRUE;
			find_CRC_polynomials_nb_errors = ST.strtoi(argv[++i]);
			find_CRC_polynomials_information_bits = ST.strtoi(argv[++i]);
			find_CRC_polynomials_check_bits = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_CRC_polynomials "
					<< " " << find_CRC_polynomials_nb_errors
					<< " " << find_CRC_polynomials_information_bits
					<< " " << find_CRC_polynomials_check_bits
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-write_code_for_division") == 0) {
			f_write_code_for_division = TRUE;
			write_code_for_division_fname.assign(argv[++i]);
			write_code_for_division_A.assign(argv[++i]);
			write_code_for_division_B.assign(argv[++i]);
			if (f_v) {
				cout << "-write_code_for_division "
						<< write_code_for_division_fname << " "
						<< write_code_for_division_A << " "
						<< write_code_for_division_B << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_division_from_file") == 0) {
			f_polynomial_division_from_file = TRUE;
			polynomial_division_from_file_fname.assign(argv[++i]);
			polynomial_division_from_file_r1 = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-polynomial_division_from_file "
					<< " " << polynomial_division_from_file_fname
					<< " " << polynomial_division_from_file_r1
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-polynomial_division_from_file_all_k_bit_error_patterns") == 0) {
			f_polynomial_division_from_file_all_k_bit_error_patterns = TRUE;
			polynomial_division_from_file_all_k_bit_error_patterns_fname.assign(argv[++i]);
			polynomial_division_from_file_all_k_bit_error_patterns_r1 = ST.strtolint(argv[++i]);
			polynomial_division_from_file_all_k_bit_error_patterns_k = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-polynomial_division_from_file_all_k_bit_error_patterns "
					<< " " << polynomial_division_from_file_all_k_bit_error_patterns_fname
					<< " " << polynomial_division_from_file_all_k_bit_error_patterns_r1
					<< " " << polynomial_division_from_file_all_k_bit_error_patterns_k
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
			}
		else {
			cout << "coding_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "coding_theoretic_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void coding_theoretic_activity_description::print()
{
#if 0
	if (f_BCH) {
		cout << "-BCH " << BCH_n << " " << BCH_q << " " << BCH_t << endl;
	}
	if (f_BCH_dual) {
		cout << "-BCH " << BCH_n << " " << BCH_q << " " << BCH_t << endl;
	}
#endif
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
	if (f_weight_enumerator) {
		cout << "-weight_enumerator " << endl;
	}
	if (f_minimum_distance) {
		cout << "-minimum_distance " << minimum_distance_code_label << endl;
	}
	if (f_generator_matrix_cyclic_code) {
		cout << "-generator_matrix_cyclic_code " << generator_matrix_cyclic_code_n
			<< " " << generator_matrix_cyclic_code_poly
			<< endl;
	}
	if (f_nth_roots) {
		cout << "-nth_roots " << nth_roots_n << endl;
	}
#if 0
	if (f_make_BCH_code) {
			cout << "-make_BCH_code "
					<< " " << make_BCH_code_n
					<< " " << make_BCH_code_d
					<< endl;
	}
	if (f_make_BCH_code_and_encode) {
			cout << "-make_BCH_code_and_encode "
					<< " " << make_BCH_code_and_encode_n
					<< " " << make_BCH_code_and_encode_d
					<< " " << make_BCH_code_and_encode_text
					<< " " << make_BCH_code_and_encode_fname
					<< endl;
	}
#endif
	if (f_NTT) {
		cout << "-NTT " << NTT_n
				<< " " << NTT_q
				<< endl;
	}
	if (f_fixed_code) {
		cout << "-fixed_code "
				<< " " << fixed_code_perm
				<< endl;
	}
	if (f_export_magma) {
		cout << "-export_magma "
			<< " " << export_magma_fname
			<< endl;
	}
	if (f_export_codewords) {
		cout << "-export_codewords "
			<< " " << export_codewords_fname
			<< endl;
	}
	if (f_export_codewords_by_weight) {
		cout << "-export_codewords_by_weight "
			<< " " << export_codewords_by_weight_fname
			<< endl;
	}
	if (f_export_genma) {
			cout << "-export_genma "
				<< " " << export_genma_fname
				<< endl;
	}
	if (f_export_checkma) {
			cout << "-export_checkma "
				<< " " << export_checkma_fname
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
	if (f_crc_encode_file_based) {
		cout << "-crc_encode_file_based "
				<< crc_encode_file_based_fname_in << " "
				<< crc_encode_file_based_fname_out << " "
				<< crc_encode_file_based_crc_type << " "
				<< crc_encode_file_based_block_length << endl;
	}
#if 0
	if (f_crc_new_file_based) {
		cout << "-crc_new_file_based "
				<< crc_new_file_based_fname
				<< endl;
	}
#endif
	if (f_find_CRC_polynomials) {
		cout << "-find_CRC_polynomials "
				<< " " << find_CRC_polynomials_nb_errors
				<< " " << find_CRC_polynomials_information_bits
				<< " " << find_CRC_polynomials_check_bits
				<< endl;
	}
	else if (f_write_code_for_division) {
		cout << "-write_code_for_division "
				<< write_code_for_division_fname << " "
				<< write_code_for_division_A << " "
				<< write_code_for_division_B << endl;
	}
	if (f_polynomial_division_from_file) {
		cout << "-polynomial_division_from_file "
				<< " " << polynomial_division_from_file_fname
				<< " " << polynomial_division_from_file_r1
				<< endl;
	}

	if (f_polynomial_division_from_file_all_k_bit_error_patterns) {
		cout << "-polynomial_division_from_file_all_k_bit_error_patterns "
				<< " " << polynomial_division_from_file_all_k_bit_error_patterns_fname
				<< " " << polynomial_division_from_file_all_k_bit_error_patterns_r1
				<< " " << polynomial_division_from_file_all_k_bit_error_patterns_k
				<< endl;
	}


}


}}}

