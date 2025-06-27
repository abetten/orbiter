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
	Record_birth();

	f_report = false;

	f_general_code_binary = false;
	general_code_binary_n = 0;
	//std::string general_code_binary_label;
	//std::string general_code_binary_text;

	f_encode_text_5bits = false;
	//encode_text_5bits_input;
	//encode_text_5bits_fname;

	f_field_induction = false;
	//std::string field_induction_fname_in;
	//std::string field_induction_fname_out;
	field_induction_nb_bits = 0;

	f_weight_enumerator = false;

	f_minimum_distance = false;
	//std::string minimum_distance_code_label;

	f_generator_matrix_cyclic_code = false;
	generator_matrix_cyclic_code_n = 0;
	//std::string generator_matrix_cyclic_code_poly

	f_Sylvester_Hadamard_code = false;
	Sylvester_Hadamard_code_n = 0;

	f_NTT = false;
	NTT_n = 0;
	NTT_q = 0;

	f_fixed_code = false;
	//std::string fixed_code_perm;

	f_export_magma = false;
	//std::string export_magma_fname;

	f_export_codewords = false;
	//std::string export_codewords_fname;

	f_all_distances = false;

	f_all_external_distances = false;

	f_export_codewords_long = false;
	//std::string export_codewords_long_fname;

	f_export_codewords_by_weight = false;
	//std::string export_codewords_by_weight_fname;

	f_export_genma = false;
	//std::string export_genma_fname;

	f_export_checkma = false;
	//std::string export_checkma_fname;

	f_export_checkma_as_projective_set = false;
	//std::string export_checkma_as_projective_set_fname;

	f_make_diagram = false;

	f_boolean_function_of_code = false;

	f_embellish = false;
	embellish_radius = 0;

	f_metric_balls = false;
	radius_of_metric_ball = 0;

	f_Hamming_space_distance_matrix = false;
	Hamming_space_distance_matrix_n = 0;





	f_crc32 = false;
	//std::string crc32_text;

	f_crc32_hexdata = false;
	//std::string crc32_hexdata_text;

	f_crc32_test = false;
	crc32_test_block_length = 0;

	f_crc256_test = false;
	crc256_test_message_length = 0;
	crc256_test_R = 0;
	crc256_test_k = 0;

	f_crc32_remainders = false;
	crc32_remainders_message_length = 0;

	f_crc_encode_file_based = false;
	//std::string crc_encode_file_based_fname_in;
	//std::string crc_encode_file_based_fname_out;
	//std::string crc_encode_file_based_crc_code;

	f_crc_compare = false;
	//std::string crc_compare_fname_in;
	//std::string crc_compare_code1;
	//std::string crc_compare_code2;
	crc_compare_error_weight = 0;
	crc_compare_nb_tests_per_block = 0;


	f_crc_compare_read_output_file = false;
	//std::string crc_compare_read_output_file_fname_in;
	crc_compare_read_output_file_nb_lines = 0;
	//std::string crc_compare_read_output_file_crc_code1;
	//std::string crc_compare_read_output_file_crc_code2;


	f_all_errors_of_a_given_weight = false;
	//std::string all_errors_of_a_given_weight_fname_in;
	all_errors_of_a_given_weight_block_number = 0;
	//std::string all_errors_of_a_given_weight_crc_code1;
	//std::string all_errors_of_a_given_weight_crc_code2;
	all_errors_of_a_given_weight_max_weight = 0;


	f_weight_enumerator_bottom_up = false;
	//std::string weight_enumerator_bottom_up_crc_code;
	weight_enumerator_bottom_up_max_weight = 0;



	f_convert_data_to_polynomials = false;
	//std::string convert_data_to_polynomials_fname_in;
	//std::string convert_data_to_polynomials_fname_out;
	convert_data_to_polynomials_block_length = 0;
	convert_data_to_polynomials_symbol_size = 0;


	f_find_CRC_polynomials = false;
	find_CRC_polynomials_nb_errors = 0;
	find_CRC_polynomials_information_bits = 0;
	find_CRC_polynomials_check_bits = 0;

	f_write_code_for_division = false;
	//std::string write_code_for_division_fname;
	//std::string write_code_for_division_A;
	//std::string write_code_for_division_B;

	f_polynomial_division_from_file = false;
	//std::string polynomial_division_from_file_fname;
	polynomial_division_from_file_r1 = 0;

	f_polynomial_division_from_file_all_k_bit_error_patterns = false;
	//std::string polynomial_division_from_file_all_k_bit_error_patterns_fname;
	polynomial_division_from_file_all_k_bit_error_patterns_r1 = 0;
	polynomial_division_from_file_all_k_bit_error_patterns_k = 0;

}

coding_theoretic_activity_description::~coding_theoretic_activity_description()
{
	Record_death();
}


int coding_theoretic_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "coding_theoretic_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-general_code_binary") == 0) {
			f_general_code_binary = true;
			general_code_binary_n = ST.strtoi(argv[++i]);
			general_code_binary_label.assign(argv[++i]);
			general_code_binary_text.assign(argv[++i]);
			if (f_v) {
				cout << "-general_code_binary " << general_code_binary_n
						<< " " << general_code_binary_label
						<< " " << general_code_binary_text
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-encode_text_5bits") == 0) {
			f_encode_text_5bits = true;
			encode_text_5bits_input.assign(argv[++i]);
			encode_text_5bits_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-encode_text_5bits " << encode_text_5bits_input << " "
						<< encode_text_5bits_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-field_induction") == 0) {
			f_field_induction = true;
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
			f_weight_enumerator = true;
			if (f_v) {
				cout << "-weight_enumerator " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-minimum_distance") == 0) {
			f_minimum_distance = true;
			minimum_distance_code_label.assign(argv[++i]);
			if (f_v) {
				cout << "-minimum_distance " << minimum_distance_code_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-generator_matrix_cyclic_code") == 0) {
			f_generator_matrix_cyclic_code = true;
			generator_matrix_cyclic_code_n = ST.strtoi(argv[++i]);
			generator_matrix_cyclic_code_poly.assign(argv[++i]);
			if (f_v) {
				cout << "-generator_matrix_cyclic_code " << generator_matrix_cyclic_code_n
					<< " " << generator_matrix_cyclic_code_poly
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Sylvester_Hadamard_code") == 0) {
			f_Sylvester_Hadamard_code = true;
			Sylvester_Hadamard_code_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Sylvester_Hadamard_code " << Sylvester_Hadamard_code_n << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-NTT") == 0) {
			f_NTT = true;
			NTT_n = ST.strtoi(argv[++i]);
			NTT_q = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-NTT " << NTT_n
						<< " " << NTT_q
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fixed_code") == 0) {
			f_fixed_code = true;
			fixed_code_perm.assign(argv[++i]);
			if (f_v) {
				cout << "-fixed_code "
					<< " " << fixed_code_perm
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = true;
			export_magma_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_magma "
					<< " " << export_magma_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_codewords") == 0) {
			f_export_codewords = true;
			export_codewords_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_codewords "
					<< " " << export_codewords_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-all_distances") == 0) {
			f_all_distances = true;
			if (f_v) {
				cout << "-all_distances "
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-all_external_distances") == 0) {
			f_all_external_distances = true;
			if (f_v) {
				cout << "-all_external_distances "
					<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-export_codewords_long") == 0) {
			f_export_codewords_long = true;
			export_codewords_long_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_codewords_long "
					<< " " << export_codewords_long_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_codewords_by_weight") == 0) {
			f_export_codewords_by_weight = true;
			export_codewords_by_weight_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_codewords_by_weight "
					<< " " << export_codewords_by_weight_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_genma") == 0) {
			f_export_genma = true;
			export_genma_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_genma "
					<< " " << export_genma_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_checkma") == 0) {
			f_export_checkma = true;
			export_checkma_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_checkma "
					<< " " << export_checkma_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_checkma_as_projective_set") == 0) {
			f_export_checkma_as_projective_set = true;
			export_checkma_as_projective_set_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-export_checkma_as_projective_set "
					<< " " << export_checkma_as_projective_set_fname
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-make_diagram") == 0) {
			f_make_diagram = true;
			if (f_v) {
				cout << "-make_diagram " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-boolean_function_of_code") == 0) {
			f_boolean_function_of_code = true;
			if (f_v) {
				cout << "-boolean_function_of_code " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-embellish") == 0) {
			f_embellish = true;
			embellish_radius = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-embellish "
					<< " " << embellish_radius
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-metric_balls") == 0) {
			f_metric_balls = true;
			radius_of_metric_ball = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-metric_balls "
					<< " " << radius_of_metric_ball
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Hamming_space_distance_matrix") == 0) {
			f_Hamming_space_distance_matrix = true;
			Hamming_space_distance_matrix_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Hamming_space_distance_matrix "
					<< " " << Hamming_space_distance_matrix_n
					<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-crc32") == 0) {
			f_crc32 = true;
			crc32_text.assign(argv[++i]);
			if (f_v) {
				cout << "-crc32 " << crc32_text
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc32_hexdata") == 0) {
			f_crc32_hexdata = true;
			crc32_hexdata_text.assign(argv[++i]);
			if (f_v) {
				cout << "-crc32_hexdata " << crc32_hexdata_text
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc32_test") == 0) {
			f_crc32_test = true;
			crc32_test_block_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-crc32_test " << crc32_test_block_length << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc256_test") == 0) {
			f_crc256_test = true;
			crc256_test_message_length = ST.strtoi(argv[++i]);
			crc256_test_R = ST.strtoi(argv[++i]);
			crc256_test_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-crc256_test " << crc256_test_message_length
						<< " " << crc256_test_R << " " << crc256_test_k << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc32_remainders") == 0) {
			f_crc32_remainders = true;
			crc32_remainders_message_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-crc32_remainders " << crc32_remainders_message_length << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc_encode_file_based") == 0) {
			f_crc_encode_file_based = true;
			crc_encode_file_based_fname_in.assign(argv[++i]);
			crc_encode_file_based_fname_out.assign(argv[++i]);
			crc_encode_file_based_crc_code.assign(argv[++i]);
			if (f_v) {
				cout << "-crc_encode_file_based "
						<< crc_encode_file_based_fname_in << " "
						<< crc_encode_file_based_fname_out << " "
						<< crc_encode_file_based_crc_code << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-crc_compare") == 0) {
			f_crc_compare = true;
			crc_compare_fname_in.assign(argv[++i]);
			crc_compare_code1.assign(argv[++i]);
			crc_compare_code2.assign(argv[++i]);
			crc_compare_error_weight = ST.strtoi(argv[++i]);
			crc_compare_nb_tests_per_block = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-crc_compare "
						<< crc_compare_fname_in << " "
						<< crc_compare_code1 << " "
						<< crc_compare_code2 << " "
						<< crc_compare_error_weight << " "
						<< crc_compare_nb_tests_per_block << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-crc_compare_read_output_file") == 0) {
			f_crc_compare_read_output_file = true;
			crc_compare_read_output_file_fname_in.assign(argv[++i]);
			crc_compare_read_output_file_nb_lines = ST.strtoi(argv[++i]);
			crc_compare_read_output_file_crc_code1.assign(argv[++i]);
			crc_compare_read_output_file_crc_code2.assign(argv[++i]);
			if (f_v) {
				cout << "-crc_compare_read_output_file "
						<< crc_compare_read_output_file_fname_in << " "
						<< crc_compare_read_output_file_nb_lines << " "
						<< crc_compare_read_output_file_crc_code1 << " "
						<< crc_compare_read_output_file_crc_code2 << " "
						<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-all_errors_of_a_given_weight") == 0) {
			f_all_errors_of_a_given_weight = true;
			all_errors_of_a_given_weight_fname_in.assign(argv[++i]);
			all_errors_of_a_given_weight_block_number = ST.strtoi(argv[++i]);
			all_errors_of_a_given_weight_crc_code1.assign(argv[++i]);
			all_errors_of_a_given_weight_crc_code2.assign(argv[++i]);
			all_errors_of_a_given_weight_max_weight = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-all_errors_of_a_given_weight "
						<< all_errors_of_a_given_weight_fname_in << " "
						<< all_errors_of_a_given_weight_block_number << " "
						<< all_errors_of_a_given_weight_crc_code1 << " "
						<< all_errors_of_a_given_weight_crc_code2 << " "
						<< all_errors_of_a_given_weight_max_weight << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-weight_enumerator_bottom_up") == 0) {
			f_weight_enumerator_bottom_up = true;
			weight_enumerator_bottom_up_crc_code.assign(argv[++i]);
			weight_enumerator_bottom_up_max_weight = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-weight_enumerator_bottom_up "
						<< weight_enumerator_bottom_up_crc_code << " "
						<< weight_enumerator_bottom_up_max_weight << " "
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-convert_data_to_polynomials") == 0) {
			f_convert_data_to_polynomials = true;
			convert_data_to_polynomials_fname_in.assign(argv[++i]);
			convert_data_to_polynomials_fname_out.assign(argv[++i]);
			convert_data_to_polynomials_block_length = ST.strtoi(argv[++i]);
			convert_data_to_polynomials_symbol_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-convert_data_to_polynomials "
						<< convert_data_to_polynomials_fname_in << " "
						<< convert_data_to_polynomials_fname_out << " "
						<< convert_data_to_polynomials_block_length << " "
						<< convert_data_to_polynomials_symbol_size << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-find_CRC_polynomials") == 0) {
			f_find_CRC_polynomials = true;
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
			f_write_code_for_division = true;
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
			f_polynomial_division_from_file = true;
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
			f_polynomial_division_from_file_all_k_bit_error_patterns = true;
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
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_general_code_binary) {
		cout << "-general_code_binary " << general_code_binary_n
				<< " " << general_code_binary_label
				<< " " << general_code_binary_text
				<< endl;
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
	if (f_Sylvester_Hadamard_code) {
		cout << "-Sylvester_Hadamard_code " << Sylvester_Hadamard_code_n << endl;
	}
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
	if (f_all_distances) {
		cout << "-all_distances "
			<< endl;
	}
	if (f_all_external_distances) {
		cout << "-all_external_distances "
			<< endl;
	}
	if (f_export_codewords_long) {
		cout << "-export_codewords_long "
			<< " " << export_codewords_long_fname
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
	if (f_export_checkma_as_projective_set) {
		cout << "-export_checkma_as_projective_set "
			<< " " << export_checkma_as_projective_set_fname
			<< endl;
	}
	if (f_make_diagram) {
		cout << "-make_diagram " << endl;
	}
	if (f_boolean_function_of_code) {
		cout << "-boolean_function_of_code " << endl;
	}
	if (f_embellish) {
		cout << "-embellish "
			<< " " << embellish_radius
			<< endl;
	}
	if (f_metric_balls) {
		cout << "-metric_balls "
			<< " " << radius_of_metric_ball
			<< endl;
	}
	if (f_Hamming_space_distance_matrix) {
		cout << "-Hamming_space_distance_matrix "
			<< " " << Hamming_space_distance_matrix_n
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
				<< crc_encode_file_based_crc_code << " "
				<< endl;
	}
	if (f_crc_compare) {
		cout << "-crc_compare "
				<< crc_compare_fname_in << " "
				<< crc_compare_code1 << " "
				<< crc_compare_code2 << " "
				<< crc_compare_error_weight << " "
				<< crc_compare_nb_tests_per_block << " "
				<< endl;
	}
	if (f_crc_compare_read_output_file) {
		cout << "-crc_compare_read_output_file "
				<< crc_compare_read_output_file_fname_in << " "
				<< crc_compare_read_output_file_nb_lines << " "
				<< crc_compare_read_output_file_crc_code1 << " "
				<< crc_compare_read_output_file_crc_code2 << " "
				<< endl;
	}

	if (f_all_errors_of_a_given_weight) {
		cout << "-all_errors_of_a_given_weight "
				<< all_errors_of_a_given_weight_fname_in << " "
				<< all_errors_of_a_given_weight_block_number << " "
				<< all_errors_of_a_given_weight_crc_code1 << " "
				<< all_errors_of_a_given_weight_crc_code2 << " "
				<< all_errors_of_a_given_weight_max_weight << " "
				<< endl;
	}

	if (f_weight_enumerator_bottom_up) {
		cout << "-weight_enumerator_bottom_up "
				<< weight_enumerator_bottom_up_crc_code << " "
				<< weight_enumerator_bottom_up_max_weight << " "
				<< endl;
	}

	if (f_convert_data_to_polynomials) {
		cout << "-convert_data_to_polynomials "
				<< convert_data_to_polynomials_fname_in << " "
				<< convert_data_to_polynomials_fname_out << " "
				<< convert_data_to_polynomials_block_length << " "
				<< convert_data_to_polynomials_symbol_size << endl;
	}
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

