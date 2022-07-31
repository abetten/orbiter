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
	F = NULL;
}

coding_theoretic_activity::~coding_theoretic_activity()
{
}


void coding_theoretic_activity::init(coding_theoretic_activity_description *Descr,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theoretic_activity::init" << endl;
	}

	coding_theoretic_activity::Descr = Descr;
	coding_theoretic_activity::F = F;

	if (f_v) {
		cout << "coding_theoretic_activity::init, field of order q = " << F->q << endl;
	}

	if (f_v) {
		cout << "coding_theoretic_activity::init done" << endl;
	}
}

void coding_theoretic_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theoretic_activity::perform_activity, q=" << F->q << endl;
	}
	data_structures::string_tools ST;

	if (Descr->f_make_macwilliams_system) {

		coding_theory::coding_theory_domain Coding;

		Coding.do_make_macwilliams_system(
				Descr->make_macwilliams_system_q,
				Descr->make_macwilliams_system_n,
				Descr->make_macwilliams_system_k, verbose_level);
	}
	else if (Descr->f_table_of_bounds) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_table_of_bounds(
				Descr->table_of_bounds_n_max,
				Descr->table_of_bounds_q,
				verbose_level);
	}
	else if (Descr->f_make_bounds_for_d_given_n_and_k_and_q) {

		coding_theory::coding_theory_domain Coding;
		int d_GV;
		int d_singleton;
		int d_hamming;
		int d_plotkin;
		int d_griesmer;

		int n, k, q;

		n = Descr->make_bounds_n;
		k = Descr->make_bounds_k;
		q = Descr->make_bounds_q;

		d_GV = Coding.gilbert_varshamov_lower_bound_for_d(n, k, q, verbose_level);
		d_singleton = Coding.singleton_bound_for_d(n, k, q, verbose_level);
		d_hamming = Coding.hamming_bound_for_d(n, k, q, verbose_level);
		d_plotkin = Coding.plotkin_bound_for_d(n, k, q, verbose_level);
		d_griesmer = Coding.griesmer_bound_for_d(n, k, q, verbose_level);

		cout << "n = " << n << " k=" << k << " q=" << q << endl;

		cout << "d_GV = " << d_GV << endl;
		cout << "d_singleton = " << d_singleton << endl;
		cout << "d_hamming = " << d_hamming << endl;
		cout << "d_plotkin = " << d_plotkin << endl;
		cout << "d_griesmer = " << d_griesmer << endl;

	}
	else if (Descr->f_BCH) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_BCH_codes(
				Descr->BCH_n,
				Descr->BCH_q,
				Descr->BCH_t, 1, FALSE,
				verbose_level);
	}
	else if (Descr->f_BCH_dual) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_BCH_codes(
				Descr->BCH_n,
				Descr->BCH_q,
				Descr->BCH_t, 1, TRUE,
				verbose_level);
	}
	else if (Descr->f_Hamming_space_distance_matrix) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_Hamming_graph_and_write_file(
				Descr->Hamming_space_n,
				Descr->Hamming_space_q,
				FALSE /* f_projective*/, verbose_level);
	}
	else if (Descr->f_general_code_binary) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory::coding_theory_domain Codes;


			Lint_vec_scan(Descr->general_code_binary_text, set, sz);

			Codes.investigate_code(set, sz, Descr->general_code_binary_n, f_embellish, verbose_level);

			FREE_lint(set);

	}

	else if (Descr->f_code_diagram) {
			long int *codewords;
			int nb_words;

			coding_theory::coding_theory_domain Codes;


			Lint_vec_scan(Descr->code_diagram_codewords_text, codewords, nb_words);



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


			Lint_vec_scan(Descr->code_diagram_codewords_text, codewords, nb_words);



			Codes.code_diagram(
					Descr->code_diagram_label,
					codewords,
					nb_words, Descr->code_diagram_n, Descr->f_metric_balls, Descr->metric_ball_radius,
					Descr->f_enhance, Descr->enhance_radius,
					verbose_level);
	}

	else if (Descr->f_linear_code_through_basis) {
			long int *set;
			int sz;
			int f_embellish = FALSE;

			coding_theory::coding_theory_domain Codes;


			//Orbiter->Lint_vec.scan(linear_code_through_basis_text, set, sz);
			orbiter_kernel_system::Orbiter->get_lint_vector_from_label(Descr->linear_code_through_basis_text, set, sz, verbose_level);

			Codes.do_linear_code_through_basis(
					F,
					Descr->linear_code_through_basis_n,
					set, sz /*k*/,
					f_embellish,
					verbose_level);

			FREE_lint(set);

	}

	else if (Descr->f_linear_code_through_columns_of_parity_check_projectively) {
			long int *set;
			int n;

			coding_theory::coding_theory_domain Codes;


			//Orbiter->Lint_vec.scan(linear_code_through_columns_of_parity_check_text, set, n);
			orbiter_kernel_system::Orbiter->get_lint_vector_from_label(
					Descr->linear_code_through_columns_of_parity_check_text, set, n, verbose_level);

			Codes.do_linear_code_through_columns_of_parity_check_projectively(
					F,
					n,
					set,
					Descr->linear_code_through_columns_of_parity_check_k /*k*/,
					verbose_level);

			FREE_lint(set);

	}


	else if (Descr->f_linear_code_through_columns_of_parity_check) {
			long int *set;
			int n;

			coding_theory::coding_theory_domain Codes;


			//Orbiter->Lint_vec.scan(linear_code_through_columns_of_parity_check_text, set, n);
			orbiter_kernel_system::Orbiter->get_lint_vector_from_label(
					Descr->linear_code_through_columns_of_parity_check_text,
					set, n, verbose_level);

			Codes.do_linear_code_through_columns_of_parity_check(
					F,
					n,
					set,
					Descr->linear_code_through_columns_of_parity_check_k /*k*/,
					verbose_level);

			FREE_lint(set);

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
	else if (Descr->f_crc32) {
		cout << "-crc32 " << Descr->crc32_text
				<< endl;

		coding_theory::coding_theory_domain Codes;
		uint32_t a;

		a = Codes.crc32(Descr->crc32_text.c_str(), Descr->crc32_text.length());
		cout << "CRC value of " << Descr->crc32_text << " is ";

		data_structures::algorithms Algo;

		Algo.print_uint32_hex(cout, a);
		cout << endl;

	}
	else if (Descr->f_crc32_hexdata) {
		cout << "-crc32_hexdata " << Descr->crc32_hexdata_text
				<< endl;

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
	else if (Descr->f_crc32_file_based) {
		cout << "-crc32_file_based " << Descr->crc32_file_based_fname
				<< " " << Descr->crc32_file_based_block_length
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc32_file_based(Descr->crc32_file_based_fname,
				Descr->crc32_file_based_block_length,
				verbose_level - 1);

	}
	else if (Descr->f_crc_new_file_based) {
		cout << "-crc_new_file_based " << Descr->crc_new_file_based_fname
				<< endl;

		coding_theory::coding_theory_domain Codes;

		Codes.crc771_file_based(Descr->crc_new_file_based_fname, verbose_level - 1);

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
	else if (Descr->f_make_gilbert_varshamov_code) {

		coding_theory::coding_theory_domain Coding;


		Coding.make_gilbert_varshamov_code(
				Descr->make_gilbert_varshamov_code_n,
				Descr->make_gilbert_varshamov_code_k,
				Descr->make_gilbert_varshamov_code_d,
				F,
				verbose_level);
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
	else if (Descr->f_make_BCH_code) {

#if 0
		coding_theory_domain Codes;
		nth_roots *Nth;
		unipoly_object P;

		int n;
		int *Genma;
		int degree;
		int *generator_polynomial;
		int i;

		n = Descr->make_BCH_code_n;
		Codes.make_BCH_code(n, F, Descr->make_BCH_code_d,
					Nth, P,
					verbose_level);

		cout << "generator polynomial is:" << endl;

		cout << "-dense \"";
		Nth->FX->print_object_dense(P, cout);
		cout << "\"" << endl;
		cout << endl;

		cout << "-sparse \"";
		Nth->FX->print_object_sparse(P, cout);
		cout << "\"" << endl;
		cout << endl;

		Nth->FX->print_object(P, cout);
		cout << endl;

		degree = Nth->FX->degree(P);
		generator_polynomial = NEW_int(degree + 1);
		for (i = 0; i <= degree; i++) {
			generator_polynomial[i] = Nth->FX->s_i(P, i);
		}

		Codes.generator_matrix_cyclic_code(n,
					degree, generator_polynomial, Genma);

		int k = n - degree;

#if 0
		cout << "generator matrix:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, Genma,
				k, n, n, F->log10_of_q);
#endif

#else
		coding_theory::create_BCH_code *C;

		C = NEW_OBJECT(coding_theory::create_BCH_code);

		C->init(F, Descr->make_BCH_code_n,
				Descr->make_BCH_code_d, verbose_level);

		orbiter_kernel_system::file_io Fio;
#if 0
		{
			char str[1000];
			string fname;

			fname.assign("genma_BCH");
			sprintf(str, "_n%d", n);
			fname.append(str);
			sprintf(str, "_k%d", k);
			fname.append(str);
			sprintf(str, "_q%d", F->q);
			fname.append(str);
			fname.append(".csv");

			Fio.int_matrix_write_csv(fname, Genma, k, n);

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}
#endif
		{

			string fname;
			string author;
			string title;
			string extra_praeamble;


			char str[1000];

			snprintf(str, 1000, "BCH_codes_q%d_n%d_d%d.tex",
					F->q,
					Descr->make_BCH_code_n,
					Descr->make_BCH_code_d
					);
			fname.assign(str);
			snprintf(str, 1000, "BCH codes");
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


				C->report(ost, verbose_level);

				L.foot(ost);


			}

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		}

#endif

	}
	else if (Descr->f_make_BCH_code_and_encode) {

		coding_theory::coding_theory_domain Codes;
		field_theory::nth_roots *Nth;
		ring_theory::unipoly_object P;

		int n;
		//int *Genma;
		int degree;
		int *generator_polynomial;
		int i;

		n = Descr->make_BCH_code_n;

		Codes.make_BCH_code(n, F, Descr->make_BCH_code_d,
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





