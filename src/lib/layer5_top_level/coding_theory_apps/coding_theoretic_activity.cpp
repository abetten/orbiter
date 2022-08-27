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

	f_has_finite_field = FALSE;
	F = NULL;

	f_has_code = FALSE;
	Code = NULL;
}

coding_theoretic_activity::~coding_theoretic_activity()
{
}


void coding_theoretic_activity::init_field(coding_theoretic_activity_description *Descr,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::init_field" << endl;
	}

	coding_theoretic_activity::Descr = Descr;
	f_has_finite_field = TRUE;
	coding_theoretic_activity::F = F;

	if (f_v) {
		cout << "coding_theoretic_activity::init_field, field of order q = " << F->q << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::init_field done" << endl;
	}
}

void coding_theoretic_activity::init_code(coding_theoretic_activity_description *Descr,
		create_code *Code,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::init_code" << endl;
	}

	coding_theoretic_activity::Descr = Descr;

	f_has_code = TRUE;
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

	if (Descr->f_BCH) {

		coding_theory::coding_theory_domain Coding;

		// this function creates a finite field
		Coding.make_BCH_codes(
				Descr->BCH_n,
				Descr->BCH_q,
				Descr->BCH_t, 1, FALSE,
				verbose_level);
	}
	else if (Descr->f_BCH_dual) {

		coding_theory::coding_theory_domain Coding;

		// this function creates a finite field
		Coding.make_BCH_codes(
				Descr->BCH_n,
				Descr->BCH_q,
				Descr->BCH_t, 1, TRUE,
				verbose_level);
	}
	else if (Descr->f_general_code_binary) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory::coding_theory_domain Codes;


			//Lint_vec_scan(Descr->general_code_binary_text, set, sz);

			Get_vector_or_set(Descr->general_code_binary_text, set, sz);


			Codes.investigate_code(set, sz, Descr->general_code_binary_n, f_embellish, verbose_level);

			FREE_lint(set);

	}

	else if (Descr->f_code_diagram) {
			long int *codewords;
			int nb_words;

			coding_theory::coding_theory_domain Codes;


			//Lint_vec_scan(Descr->code_diagram_codewords_text, codewords, nb_words);

			Get_vector_or_set(Descr->code_diagram_codewords_text, codewords, nb_words);


			Codes.code_diagram(
					Descr->code_diagram_label,
					codewords,
					nb_words, Descr->code_diagram_n, Descr->f_metric_balls, Descr->metric_ball_radius,
					Descr->f_enhance, 0 /*nb_enhance */,
					verbose_level);
	}

	else if (Descr->f_code_diagram_from_file) {
			long int *codewords;
			int m, nb_words;
			orbiter_kernel_system::file_io Fio;

			coding_theory::coding_theory_domain Codes;


			Fio.lint_matrix_read_csv(Descr->code_diagram_from_file_codewords_fname, codewords, m, nb_words, verbose_level);



			Codes.code_diagram(
					Descr->code_diagram_label,
					codewords,
					nb_words, Descr->code_diagram_n, Descr->f_metric_balls, Descr->metric_ball_radius,
					Descr->f_enhance, Descr->enhance_radius,
					verbose_level);
	}

	else if (Descr->f_code_diagram_from_file) {
			long int *codewords;
			int nb_words;

			coding_theory::coding_theory_domain Codes;


			//Lint_vec_scan(Descr->code_diagram_codewords_text, codewords, nb_words);

			Get_vector_or_set(Descr->code_diagram_codewords_text, codewords, nb_words);


			Codes.code_diagram(
					Descr->code_diagram_label,
					codewords,
					nb_words, Descr->code_diagram_n, Descr->f_metric_balls, Descr->metric_ball_radius,
					Descr->f_enhance, Descr->enhance_radius,
					verbose_level);
	}


	else if (Descr->f_long_code) {
		coding_theory::coding_theory_domain Codes;
			string dummy;

			Codes.do_long_code(
					Descr->long_code_n,
					Descr->long_code_generators,
					FALSE /* f_nearest_codeword */,
					dummy /* const char *nearest_codeword_text */,
					verbose_level);

	}
	else if (Descr->f_encode_text_5bits) {
		coding_theory::coding_theory_domain Codes;

		Codes.encode_text_5bits(
				Descr->encode_text_5bits_input,
				Descr->encode_text_5bits_fname,
				verbose_level);

	}
	else if (Descr->f_field_induction) {
		coding_theory::coding_theory_domain Codes;

		Codes.field_induction(
				Descr->field_induction_fname_in,
				Descr->field_induction_fname_out,
				Descr->field_induction_nb_bits,
				verbose_level);

	}
	else if (Descr->f_weight_enumerator) {

		coding_theory::coding_theory_domain Codes;

		int *v;
		int m, n;

		orbiter_kernel_system::Orbiter->get_matrix_from_label(Descr->weight_enumerator_input_matrix, v, m, n);


		Codes.do_weight_enumerator(F,
				v, m, n,
				FALSE /* f_normalize_from_the_left */,
				FALSE /* f_normalize_from_the_right */,
				verbose_level);

		FREE_int(v);
	}
	else if (Descr->f_minimum_distance) {

		coding_theory::coding_theory_domain Codes;

		int *v;
		int m, n;

		orbiter_kernel_system::Orbiter->get_matrix_from_label(Descr->minimum_distance_code_label, v, m, n);


		Codes.do_minimum_distance(F,
				v, m, n,
				verbose_level);

		FREE_int(v);
	}

	else if (Descr->f_generator_matrix_cyclic_code) {

		cout << "before generator_matrix_cyclic_code" << endl;

		coding_theory::coding_theory_domain Coding;

		Coding.generator_matrix_cyclic_code(F,
				Descr->generator_matrix_cyclic_code_n,
				Descr->generator_matrix_cyclic_code_poly,
				verbose_level);

	}

	else if (Descr->f_nth_roots) {
		cout << "-nth_roots n=" << Descr->nth_roots_n << endl;

		field_theory::nth_roots *Nth;

		Nth = NEW_OBJECT(field_theory::nth_roots);

		Nth->init(F, Descr->nth_roots_n, verbose_level);

		orbiter_kernel_system::file_io Fio;
		{

			string fname;
			string author;
			string title;
			string extra_praeamble;


			char str[1000];

			snprintf(str, 1000, "Nth_roots_q%d_n%d.tex", F->q, Descr->nth_roots_n);
			fname.assign(str);
			snprintf(str, 1000, "Nth roots");
			title.assign(str);




			{
				ofstream ost(fname);
				number_theory::number_theory_domain NT;



				orbiter_kernel_system::latex_interface L;

				L.head(ost,
						FALSE /* f_book*/,
						TRUE /* f_title */,
						title, author,
						FALSE /* f_toc */,
						FALSE /* f_landscape */,
						TRUE /* f_12pt */,
						TRUE /* f_enlarged_page */,
						TRUE /* f_pagenumbers */,
						extra_praeamble /* extra_praeamble */);


				Nth->report(ost, verbose_level);

				L.foot(ost);


			}

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		}

	}
	else if (Descr->f_make_BCH_code_and_encode) {

		coding_theory::coding_theory_domain Codes;
		field_theory::nth_roots *Nth;
		ring_theory::unipoly_object P;

		int n;
		int degree;
		int *generator_polynomial;
		int i;

		n = Descr->make_BCH_code_n;

		Codes.make_BCH_code(n,
				F,
				Descr->make_BCH_code_d,
				Nth, P,
				verbose_level);

		cout << "generator polynomial is ";
		Nth->FX->print_object(P, cout);
		cout << endl;

		degree = Nth->FX->degree(P);
		generator_polynomial = NEW_int(degree + 1);
		for (i = 0; i <= degree; i++) {
			generator_polynomial[i] = Nth->FX->s_i(P, i);
		}

		// Descr->make_BCH_code_and_encode_text

		Codes.CRC_encode_text(Nth, P,
				Descr->make_BCH_code_and_encode_text,
				Descr->make_BCH_code_and_encode_fname,
				verbose_level);


	}
	else if (Descr->f_NTT) {
		number_theory::number_theoretic_transform NTT;

		NTT.init(F, Descr->NTT_n, Descr->NTT_q, verbose_level);

	}

	else if (Descr->f_export_magma) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity f_export_magma" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity f_export_magma needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity before Code->export_magma" << endl;
		}
		Code->export_magma(Descr->export_magma_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity after Code->export_magma" << endl;
		}

	}

	else if (Descr->f_export_codewords) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity f_export_codewords" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity f_export_codewords needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity before Code->export_codewords" << endl;
		}
		Code->export_codewords(Descr->export_codewords_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity after Code->export_codewords" << endl;
		}

	}


	else if (Descr->f_export_genma) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity f_export_genma" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity f_export_genma needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity before Code->export_genma" << endl;
		}
		Code->export_genma(Descr->export_genma_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity after Code->export_genma" << endl;
		}

	}


	else if (Descr->f_export_checkma) {

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity f_export_checkma" << endl;
		}
		if (!f_has_code) {
			cout << "coding_theoretic_activity::perform_activity f_export_checkma needs a code" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity before Code->export_checkma" << endl;
		}
		Code->export_checkma(Descr->export_checkma_fname, verbose_level);
		if (f_v) {
			cout << "coding_theoretic_activity::perform_activity after Code->export_checkma" << endl;
		}

	}

	else if (Descr->f_crc32) {
		cout << "-crc32 " << Descr->crc32_text << endl;

		coding_theory::coding_theory_domain Codes;
		uint32_t a;

		a = Codes.crc32(Descr->crc32_text.c_str(), Descr->crc32_text.length());
		cout << "CRC value of " << Descr->crc32_text << " is ";

		data_structures::algorithms Algo;

		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (Descr->f_crc32_hexdata) {
		cout << "-crc32_hexdata " << Descr->crc32_hexdata_text << endl;

		coding_theory::coding_theory_domain Codes;
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


		a = Codes.crc32(data, data_size);
		cout << "CRC value of 0x" << Descr->crc32_hexdata_text << " is ";


		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (Descr->f_crc32_test) {
		cout << "-crc32_test "
				<< Descr->crc32_test_block_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc32_test(Descr->crc32_test_block_length, verbose_level - 1);

	}
	else if (Descr->f_crc256_test) {
		cout << "-crc256_test "
				<< Descr->crc256_test_message_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc256_test_k_subsets(
				Descr->crc256_test_message_length,
				Descr->crc256_test_R,
				Descr->crc256_test_k,
				verbose_level - 1);

	}
	else if (Descr->f_crc32_remainders) {
		cout << "-crc32_remainders "
				<< Descr->crc32_remainders_message_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc32_remainders(
				Descr->crc32_remainders_message_length,
				verbose_level - 1);

	}
	else if (Descr->f_crc_encode_file_based) {
		cout << "-crc_encode_file_based " << Descr->crc_encode_file_based_fname_in
				<< " " << Descr->crc_encode_file_based_block_length
				<< " " << Descr->crc_encode_file_based_crc_type
				<< " " << Descr->crc_encode_file_based_block_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc_encode_file_based(
				Descr->crc_encode_file_based_fname_in,
				Descr->crc_encode_file_based_fname_out,
				Descr->crc_encode_file_based_crc_type,
				Descr->crc_encode_file_based_block_length,
				verbose_level - 1);

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

		coding_theory::coding_theory_domain Coding;

		Coding.find_CRC_polynomials(F,
				Descr->find_CRC_polynomials_nb_errors,
				Descr->find_CRC_polynomials_information_bits,
				Descr->find_CRC_polynomials_check_bits,
				verbose_level);
	}
	else if (Descr->f_write_code_for_division) {

		ring_theory::ring_theory_global R;

		R.write_code_for_division(F,
				Descr->write_code_for_division_fname,
				Descr->write_code_for_division_A,
				Descr->write_code_for_division_B,
				verbose_level);
	}

	else if (Descr->f_polynomial_division_from_file) {

		ring_theory::ring_theory_global R;

		R.polynomial_division_from_file_with_report(F,
				Descr->polynomial_division_from_file_fname,
				Descr->polynomial_division_from_file_r1,
				verbose_level);
	}

	else if (Descr->f_polynomial_division_from_file_all_k_bit_error_patterns) {

		ring_theory::ring_theory_global R;

		R.polynomial_division_from_file_all_k_error_patterns_with_report(F,
				Descr->polynomial_division_from_file_all_k_bit_error_patterns_fname,
				Descr->polynomial_division_from_file_all_k_bit_error_patterns_r1,
				Descr->polynomial_division_from_file_all_k_bit_error_patterns_k,
				verbose_level);
	}



	if (f_v) {
		cout << "coding_theoretic_activity::perform_activity done" << endl;
	}
}

}}}





