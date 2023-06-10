/*
 * coding_theoretic_activity.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


coding_theoretic_activity::coding_theoretic_activity()
{
	Descr = NULL;

	f_has_finite_field = false;
	F = NULL;

	f_has_code = false;
	Code = NULL;
}

coding_theoretic_activity::~coding_theoretic_activity()
{
}


void coding_theoretic_activity::init_field(
		coding_theoretic_activity_description *Descr,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::init_field" << endl;
	}

	coding_theoretic_activity::Descr = Descr;
	f_has_finite_field = true;
	coding_theoretic_activity::F = F;

	if (f_v) {
		cout << "coding_theoretic_activity::init_field, "
				"field of order q = " << F->q << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::init_field done" << endl;
	}
}

void coding_theoretic_activity::init_code(
		coding_theoretic_activity_description *Descr,
		create_code *Code,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::init_code" << endl;
	}

	coding_theoretic_activity::Descr = Descr;

	f_has_code = true;
	coding_theoretic_activity::Code = Code;

	if (f_v) {
		cout << "coding_theoretic_activity::init_code "
				"code=" << Code->label_txt << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::init_code done" << endl;
	}
}


void coding_theoretic_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::perform_activity" << endl;
	}
	data_structures::string_tools ST;

	if (Descr->f_report) {
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity f_report" << endl;
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity before Code->report" << endl;
		}
		Code->report(verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity after Code->report" << endl;
		}

	}
	else if (Descr->f_general_code_binary) {
		long int *Words;
		int nb_words;
		int n;

		//coding_theory::coding_theory_domain Codes;


		n = Descr->general_code_binary_n;

		Get_vector_or_set(Descr->general_code_binary_text, Words, nb_words);

		coding_theory::code_diagram *Diagram;


		Diagram = NEW_OBJECT(coding_theory::code_diagram);

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Diagram->init" << endl;
		}

		Diagram->init(Descr->general_code_binary_label /* label */,
				Words, nb_words, n, verbose_level);

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Diagram->init" << endl;
		}


		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before do_diagram" << endl;
		}
		do_diagram(Diagram, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after do_diagram" << endl;
		}

		FREE_lint(Words);

	}
	else if (Descr->f_encode_text_5bits) {
		coding_theory::coding_theory_domain Codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Codes.encode_text_5bits" << endl;
		}
		Codes.encode_text_5bits(
				Descr->encode_text_5bits_input,
				Descr->encode_text_5bits_fname,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Codes.encode_text_5bits" << endl;
		}

	}
	else if (Descr->f_field_induction) {
		coding_theory::coding_theory_domain Codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Codes.field_induction" << endl;
		}
		Codes.field_induction(
				Descr->field_induction_fname_in,
				Descr->field_induction_fname_out,
				Descr->field_induction_nb_bits,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Codes.field_induction" << endl;
		}

	}
	else if (Descr->f_weight_enumerator) {

		cout << "-weight_enumerator" << endl;

		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_weight_enumerator needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Codes.weight_enumerator" << endl;
		}
		Code->weight_enumerator(verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Codes.weight_enumerator" << endl;
		}

	}
	else if (Descr->f_minimum_distance) {

		coding_theory::coding_theory_domain Codes;

		int *v;
		int m, n;

		Get_matrix(Descr->minimum_distance_code_label, v, m, n);


		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Codes.do_minimum_distance" << endl;
		}
		Codes.do_minimum_distance(F,
				v, m, n,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Codes.do_minimum_distance" << endl;
		}

		FREE_int(v);
	}

	else if (Descr->f_generator_matrix_cyclic_code) {

		cout << "before generator_matrix_cyclic_code" << endl;

		coding_theory::cyclic_codes Cyclic_codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Cyclic_codes.generator_matrix_cyclic_code" << endl;
		}
		Cyclic_codes.generator_matrix_cyclic_code(F,
				Descr->generator_matrix_cyclic_code_n,
				Descr->generator_matrix_cyclic_code_poly,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Cyclic_codes.generator_matrix_cyclic_code" << endl;
		}

	}

	else if (Descr->f_Sylvester_Hadamard_code) {
		cout << "-Sylvester_Hadamard_code n=" << Descr->Sylvester_Hadamard_code_n << endl;

		coding_theory::coding_theory_domain Codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Codes.do_sylvester_hadamard" << endl;
		}
		Codes.do_sylvester_hadamard(F,
				Descr->Sylvester_Hadamard_code_n,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Codes.do_sylvester_hadamard" << endl;
		}


	}


	else if (Descr->f_NTT) {
		number_theory::number_theoretic_transform NTT;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before NTT.init" << endl;
		}
		NTT.init(F, Descr->NTT_n, Descr->NTT_q, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after NTT.init" << endl;
		}

	}

	else if (Descr->f_fixed_code) {
		cout << "-fixed_code " << Descr->fixed_code_perm << endl;

		coding_theory::coding_theory_domain Codes;

		long int *perm;
		int n;

		Get_vector_or_set(Descr->fixed_code_perm, perm, n);

		if (!f_has_code) {
			cout << "-fixed_code needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->fixed_code" << endl;
		}
		Code->fixed_code(
					perm, n,
					verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->fixed_code" << endl;
		}

		FREE_lint(perm);

	}

	else if (Descr->f_export_magma) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_magma" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_magma needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_magma" << endl;
		}
		Code->export_magma(Descr->export_magma_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_magma" << endl;
		}

	}

	else if (Descr->f_export_codewords) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_codewords" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_codewords needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_codewords" << endl;
		}
		Code->export_codewords(Descr->export_codewords_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_codewords" << endl;
		}

	}

	else if (Descr->f_export_codewords_long) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_codewords_long" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_codewords needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_codewords_long" << endl;
		}
		Code->export_codewords_long(Descr->export_codewords_long_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_codewords_long" << endl;
		}

	}

	else if (Descr->f_export_codewords_by_weight) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_codewords_by_weight" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_codewords_by_weight needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_codewords_by_weight" << endl;
		}
		Code->export_codewords_by_weight(
				Descr->export_codewords_by_weight_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_codewords_by_weight" << endl;
		}

	}


	else if (Descr->f_export_genma) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_genma" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_genma needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_genma" << endl;
		}
		Code->export_genma(Descr->export_genma_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_genma" << endl;
		}

	}


	else if (Descr->f_export_checkma) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_checkma" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_checkma needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_checkma" << endl;
		}
		Code->export_checkma(Descr->export_checkma_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_checkma" << endl;
		}

	}
	else if (Descr->f_export_checkma_as_projective_set) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_checkma_as_projective_set" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_export_checkma_as_projective_set needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->export_checkma_as_projective_set" << endl;
		}
		Code->export_checkma_as_projective_set(
				Descr->export_checkma_as_projective_set_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->export_checkma_as_projective_set" << endl;
		}

	}



	else if (Descr->f_make_diagram) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_make_diagram" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_make_diagram needs a code" << endl;
			exit(1);
		}

		coding_theory::code_diagram *Diagram;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->make_diagram" << endl;
		}
		Code->make_diagram(
				Descr->f_embellish, Descr->embellish_radius,
				Descr->f_metric_balls, Descr->radius_of_metric_ball,
				Diagram,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->make_diagram" << endl;
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before do_diagram" << endl;
		}
		do_diagram(Diagram, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after do_diagram" << endl;
		}


	}
	else if (Descr->f_boolean_function_of_code) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_boolean_function_of_code" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_boolean_function_of_code needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Code->polynomial_representation_of_boolean_function" << endl;
		}
		Code->polynomial_representation_of_boolean_function(
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Code->polynomial_representation_of_boolean_function" << endl;
		}
	}
	else if (Descr->f_Hamming_space_distance_matrix) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_Hamming_space_distance_matrix" << endl;
		}
		if (!f_has_finite_field) {
			cout << "coding_theoretic_activity::perform_activity "
					"f_boolean_function_of_code needs a finite field" << endl;
			exit(1);
		}

		coding_theory::coding_theory_domain Codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Codes.make_Hamming_space_distance_matrix" << endl;
		}

		Codes.make_Hamming_space_distance_matrix(
				Descr->Hamming_space_distance_matrix_n, F,
				false /* f_projective */, verbose_level - 1);

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Codes.make_Hamming_space_distance_matrix" << endl;
		}
	}



	else if (Descr->f_crc32) {
		cout << "-crc32 " << Descr->crc32_text << endl;

		coding_theory::crc_codes Crc_codes;
		uint32_t a;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.crc32" << endl;
		}
		a = Crc_codes.crc32((const uint8_t *) Descr->crc32_text.c_str(), Descr->crc32_text.length());
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.crc32" << endl;
		}

		cout << "CRC value of " << Descr->crc32_text << " is ";

		data_structures::algorithms Algo;

		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (Descr->f_crc32_hexdata) {
		cout << "-crc32_hexdata " << Descr->crc32_hexdata_text << endl;

		coding_theory::crc_codes Crc_codes;
		data_structures::algorithms Algo;
		uint32_t a;
		char *data;
		int data_size;

		cout << "before Algo.read_hex_data" << endl;
		Algo.read_hex_data(Descr->crc32_hexdata_text, data, data_size, verbose_level - 2);
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


		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.crc32" << endl;
		}
		a = Crc_codes.crc32((const uint8_t *) data, data_size);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.crc32" << endl;
		}
		cout << "CRC value of 0x" << Descr->crc32_hexdata_text << " is ";


		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (Descr->f_crc32_test) {
		cout << "-crc32_test "
				<< Descr->crc32_test_block_length
				<< endl;

		coding_theory::crc_codes Crc_codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.crc32_test" << endl;
		}
		Crc_codes.crc32_test(Descr->crc32_test_block_length, verbose_level - 1);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.crc32_test" << endl;
		}

	}
	else if (Descr->f_crc256_test) {
		cout << "-crc256_test "
				<< Descr->crc256_test_message_length
				<< endl;

		coding_theory::crc_codes Crc_codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.crc256_test_k_subsets" << endl;
		}
		Crc_codes.crc256_test_k_subsets(
				Descr->crc256_test_message_length,
				Descr->crc256_test_R,
				Descr->crc256_test_k,
				verbose_level - 1);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.crc256_test_k_subsets" << endl;
		}

	}
	else if (Descr->f_crc32_remainders) {
		cout << "-crc32_remainders "
				<< Descr->crc32_remainders_message_length
				<< endl;

		coding_theory::crc_codes Crc_codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.crc32_remainders" << endl;
		}
		Crc_codes.crc32_remainders(
				Descr->crc32_remainders_message_length,
				verbose_level - 1);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.crc32_remainders" << endl;
		}

	}
	else if (Descr->f_crc_encode_file_based) {
		cout << "-crc_encode_file_based " << Descr->crc_encode_file_based_fname_in
				<< " " << Descr->crc_encode_file_based_block_length
				<< " " << Descr->crc_encode_file_based_crc_type
				<< " " << Descr->crc_encode_file_based_block_length
				<< endl;

		coding_theory::crc_codes Crc_codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.crc_encode_file_based" << endl;
		}
		Crc_codes.crc_encode_file_based(
				Descr->crc_encode_file_based_fname_in,
				Descr->crc_encode_file_based_fname_out,
				Descr->crc_encode_file_based_crc_type,
				Descr->crc_encode_file_based_block_length,
				verbose_level - 1);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.crc_encode_file_based" << endl;
		}

	}
	else if (Descr->f_convert_data_to_polynomials) {
		cout << "-convert_data_to_polynomials "
				<< " " << Descr->convert_data_to_polynomials_fname_in
				<< " " << Descr->convert_data_to_polynomials_fname_out
				<< " " << Descr->convert_data_to_polynomials_block_length
				<< " " << Descr->convert_data_to_polynomials_symbol_size
				<< endl;

		coding_theory::crc_codes Crc_codes;


		if (Descr->convert_data_to_polynomials_symbol_size == 256) {
			if (f_v) {
				cout << "coding_theoretic_activity::perform_activity "
						"before Crc_codes.split_binary_file_to_ascii_polynomials_256" << endl;
			}
			Crc_codes.split_binary_file_to_ascii_polynomials_256(
					Descr->convert_data_to_polynomials_fname_in,
					Descr->convert_data_to_polynomials_fname_out,
					Descr->convert_data_to_polynomials_block_length,
					verbose_level - 1);
			if (f_v) {
				cout << "coding_theoretic_activity::perform_activity "
						"after Crc_codes.split_binary_file_to_ascii_polynomials_256" << endl;
			}
		}
		else {
			cout << "I do not have an encoder with symbol size " << Descr->convert_data_to_polynomials_symbol_size << endl;
			exit(1);
		}

	}

#if 0
	else if (Descr->f_crc_new_file_based) {
		cout << "-crc_new_file_based " << Descr->crc_new_file_based_fname
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc771_file_based(Descr->crc_new_file_based_fname, verbose_level - 1);

	}
#endif
	else if (Descr->f_find_CRC_polynomials) {

		coding_theory::crc_codes Crc_codes;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before Crc_codes.find_CRC_polynomials" << endl;
		}
		Crc_codes.find_CRC_polynomials(F,
				Descr->find_CRC_polynomials_nb_errors,
				Descr->find_CRC_polynomials_information_bits,
				Descr->find_CRC_polynomials_check_bits,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after Crc_codes.find_CRC_polynomials" << endl;
		}
	}
	else if (Descr->f_write_code_for_division) {

		ring_theory::ring_theory_global R;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before R.write_code_for_division" << endl;
		}
		R.write_code_for_division(F,
				Descr->write_code_for_division_fname,
				Descr->write_code_for_division_A,
				Descr->write_code_for_division_B,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after R.write_code_for_division" << endl;
		}
	}

	else if (Descr->f_polynomial_division_from_file) {

		ring_theory::ring_theory_global R;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before R.polynomial_division_from_file_with_report" << endl;
		}
		R.polynomial_division_from_file_with_report(F,
				Descr->polynomial_division_from_file_fname,
				Descr->polynomial_division_from_file_r1,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after R.polynomial_division_from_file_with_report" << endl;
		}
	}

	else if (Descr->f_polynomial_division_from_file_all_k_bit_error_patterns) {

		ring_theory::ring_theory_global R;

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"before R.polynomial_division_from_file_all_k_error_patterns_with_report" << endl;
		}
		R.polynomial_division_from_file_all_k_error_patterns_with_report(F,
				Descr->polynomial_division_from_file_all_k_bit_error_patterns_fname,
				Descr->polynomial_division_from_file_all_k_bit_error_patterns_r1,
				Descr->polynomial_division_from_file_all_k_bit_error_patterns_k,
				verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity "
					"after R.polynomial_division_from_file_all_k_error_patterns_with_report" << endl;
		}
	}


	if (f_v) {
		cout << "coding_theoretic_activity::perform_activity done" << endl;
	}
}

void coding_theoretic_activity::do_diagram(
		coding_theory::code_diagram *Diagram,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram" << endl;
	}

	if (Descr->f_metric_balls) {
		Diagram->place_metric_balls(Descr->radius_of_metric_ball, verbose_level);
	}

	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"before Diagram->save_distance" << endl;
	}
	Diagram->save_distance(verbose_level);
	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"after Diagram->save_distance" << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"before Diagram->save_distance_H" << endl;
	}
	Diagram->save_distance_H(verbose_level);
	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"after Diagram->save_distance_H" << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"before Diagram->save_diagram" << endl;
	}
	Diagram->save_diagram(verbose_level);
	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"after Diagram->save_diagram" << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"before Diagram->save_char_func" << endl;
	}
	Diagram->save_char_func(verbose_level);
	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"after Diagram->save_char_func" << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"before Diagram->report" << endl;
	}
	Diagram->report(verbose_level);
	if (f_v) {
		cout << "coding_theoretic_activity::do_diagram "
				"after Diagram->report" << endl;
	}


}




}}}





