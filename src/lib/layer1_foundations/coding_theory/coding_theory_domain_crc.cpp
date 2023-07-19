/*
 * coding_theory_domain_crc.cpp
 *
 *  Created on: Jul 19, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {


void coding_theory_domain::crc_encode_file_based(
		std::string &fname_in,
		std::string &fname_out,
		crc_object *Crc_object,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc_encode_file_based "
				"fname_in=" << fname_in << endl;
	}


	orbiter_kernel_system::file_io Fio;

	long int N, L, nb_blocks, cnt;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "coding_theory_domain::crc_encode_file_based "
				"input file size = " << N << endl;
	}

	nb_blocks = Crc_object->get_nb_blocks(N);

	if (f_v) {
		cout << "coding_theory_domain::crc_encode_file_based "
				"nb_blocks = " << nb_blocks << endl;
	}

	data_structures::algorithms Algo;

	unsigned char *Data0;
	unsigned char *Data1;
	unsigned char *Check1;
	unsigned char *Check1c;

	Data0 = (unsigned char *) NEW_char(Crc_object->Len_total_in_symbols);
	Data1 = (unsigned char *) NEW_char(Crc_object->Len_total_in_symbols);
	Check1 = (unsigned char *) NEW_char(Crc_object->Len_check_in_symbols);
	Check1c = (unsigned char *) NEW_char(Crc_object->Len_check_in_bytes);



	ifstream ist(fname_in, ios::binary);

	{
		ofstream ost(fname_out, ios::binary);


		for (cnt = 0; cnt < nb_blocks; cnt++) {


			L = Crc_object->get_this_block_size(N, cnt);


			Algo.uchar_zero(Data0, Crc_object->Len_total_in_symbols);
			Algo.uchar_zero(Data1, Crc_object->Len_total_in_symbols);


			// read one block of information:

			ist.read((char *) Data0 + Crc_object->Len_check_in_bytes, L);

			//Algo.uchar_move(Data0, Data1, Crc_object1->Len_total_in_bytes);
			//Algo.uchar_move(Data0, Data2, Crc_object1->Len_total_in_bytes);

			Crc_object->expand(Data0, Data1);


			Crc_object->divide(Data1, Check1);

			Crc_object->compress_check(Check1, Check1c);





			// write information + check to file:

			ost.write((char *)Data0 + Crc_object->Len_check_in_bytes, L);
			ost.write((char *)Check1c, Crc_object->Len_check_in_bytes);


		}

	}

	cout << "Written file " << fname_out << " of size "
			<< Fio.file_size(fname_out) << endl;

	cout << "nb_blocks = " << nb_blocks << endl;


	FREE_char((char *) Data0);
	FREE_char((char *) Data1);
	FREE_char((char *) Check1);
	FREE_char((char *) Check1c);


	if (f_v) {
		cout << "coding_theory_domain::crc_encode_file_based done" << endl;
	}

}




void coding_theory_domain::crc_simulate_errors(
		std::string &fname_in,
		crc_object *Crc_object1,
		crc_object *Crc_object2,
		int error_pattern_weight,
		int nb_tests_per_block,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc_simulate_errors "
				" fname_in=" << fname_in
				<< " error_pattern_weight=" << error_pattern_weight
				<< " nb_tests_per_block=" << nb_tests_per_block
				<< endl;
	}


	if (Crc_object1->info_length_in_bytes != Crc_object2->info_length_in_bytes) {
		cout << "CRC codes must have the same info_length_in_bytes" << endl;
		cout << "first code has " << Crc_object1->info_length_in_bytes << endl;
		cout << "second code has " << Crc_object2->info_length_in_bytes << endl;
		exit(1);
	}
	if (Crc_object1->Len_check_in_bytes != Crc_object2->Len_check_in_bytes) {
		cout << "CRC codes must have the same Len_check_in_bytes" << endl;
		exit(1);
	}

	data_structures::algorithms Algo;


	orbiter_kernel_system::file_io Fio;

	long int N, L, nb_blocks, cnt;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "coding_theory_domain::crc_simulate_errors "
				"input file size = " << N << endl;
	}

	nb_blocks = Crc_object1->get_nb_blocks(N);

	if (f_v) {
		cout << "coding_theory_domain::crc_simulate_errors "
				"nb_blocks = " << nb_blocks << endl;
	}


	unsigned char *Data0;
	unsigned char *Data0e;
	unsigned char *Data1;
	unsigned char *Data2;
	unsigned char *Check1;
	unsigned char *Check2;
	unsigned char *Check1a;
	unsigned char *Check2a;
	unsigned char *Check1b;
	unsigned char *Check2b;

	Data0 = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Data0e = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Data1 = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Data2 = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Check1 = (unsigned char *) NEW_char(Crc_object1->Len_check_in_symbols);
	Check2 = (unsigned char *) NEW_char(Crc_object1->Len_check_in_symbols);
	Check1a = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);
	Check2a = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);
	Check1b = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);
	Check2b = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);


	error_pattern_generator *Error;


	Error = NEW_OBJECT(error_pattern_generator);

	Error->init(Crc_object1, error_pattern_weight, verbose_level);


	long int nb1 = 0;
	long int nb2 = 0;
	long int nb_error_pattern_zero = 0;
	int nb_blocks_100;

	nb_blocks_100 = (nb_blocks / 100) + 1;

	ifstream ist(fname_in, ios::binary);

	{

		for (cnt = 0; cnt < nb_blocks; cnt++) {


			if ((cnt % nb_blocks_100) == 0) {
				if (f_v) {
					cout << "coding_theory_domain::crc_simulate_errors "
							"block " << cnt << " / " << nb_blocks << " = "
							<< ((double)cnt / (double) nb_blocks_100) << " percent "
									"nb undetected errors =  " << nb1 << "," << nb2 << endl;
				}
			}

			L = Crc_object1->get_this_block_size(N, cnt);


			Algo.uchar_zero(Data0, Crc_object1->Len_total_in_symbols);
			Algo.uchar_zero(Data1, Crc_object1->Len_total_in_symbols);
			Algo.uchar_zero(Data2, Crc_object1->Len_total_in_symbols);


			// read one block of information:

			ist.read((char *) Data0 + Crc_object1->Len_check_in_bytes, L);

			//Algo.uchar_move(Data0, Data1, Crc_object1->Len_total_in_bytes);
			//Algo.uchar_move(Data0, Data2, Crc_object1->Len_total_in_bytes);

			Crc_object1->expand(Data0, Data1);
			Crc_object2->expand(Data0, Data2);


			Crc_object1->divide(Data1, Check1);
			Crc_object2->divide(Data2, Check2);

			Crc_object1->compress_check(Check1, Check1a);
			Crc_object2->compress_check(Check2, Check2a);



			int e;

			for (e = 0; e < nb_tests_per_block; e++) {

				Error->create_error_pattern(0 /*verbose_level*/);

				int i;

				for (i = 0; i < Crc_object1->Len_total_in_bytes; i++) {
					if (Error->Error_in_bytes[i]) {
						break;
					}
				}

				if (i == Crc_object1->Len_total_in_bytes) {

					// don't do anything at all, the error pattern is zero;

					nb_error_pattern_zero++;
				}
				else {

					//Algo.print_hex(cout, Error->Error, Crc_object1->Len_total_in_bytes);

					Algo.uchar_xor(Data0, Error->Error_in_bytes, Data0e, Crc_object1->Len_total_in_bytes);

					Crc_object1->expand(Data0e, Data1);
					Crc_object2->expand(Data0e, Data2);

					Crc_object1->divide(Data1, Check1);
					Crc_object2->divide(Data2, Check2);

					Crc_object1->compress_check(Check1, Check1b);
					Crc_object2->compress_check(Check2, Check2b);

					if (Algo.uchar_compare(Check1a, Check1b, Crc_object1->Len_check_in_bytes) == 0) {
						nb1++;
						cout << "undetected error code 1 : " << nb1
								<< " in block " << cnt << " / " << nb_blocks << endl;
						Algo.print_hex(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
						Algo.print_binary(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
					}
					if (Algo.uchar_compare(Check2a, Check2b, Crc_object1->Len_check_in_bytes) == 0) {
						nb2++;
						cout << "undetected error code 2 : " << nb2
								<< " in block " << cnt << " / " << nb_blocks << endl;
						Algo.print_hex(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
						Algo.print_binary(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
					}
				}
			} // next e
		}

	}

	cout << "nb_blocks = " << nb_blocks << endl;

	cout << "nb_error_pattern_zero = " << nb_error_pattern_zero << endl;

	cout << "number of undetected errors code 1 = " << nb1 << endl;
	cout << "number of undetected errors code 2 = " << nb2 << endl;


	FREE_char((char *) Data0);
	FREE_char((char *) Data0e);
	FREE_char((char *) Data1);
	FREE_char((char *) Data2);
	FREE_char((char *) Check1);
	FREE_char((char *) Check2);
	FREE_char((char *) Check1a);
	FREE_char((char *) Check2a);
	FREE_char((char *) Check1b);
	FREE_char((char *) Check2b);


	if (f_v) {
		cout << "coding_theory_domain::crc_simulate_errors done" << endl;
	}

}

void coding_theory_domain::crc_all_errors_of_a_given_weight(
		std::string &fname_in,
		int block_number,
		crc_object *Crc_object1,
		crc_object *Crc_object2,
		int error_pattern_max_weight,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc_all_errors_of_a_given_weight "
				" fname_in=" << fname_in
				<< " error_pattern_max_weight=" << error_pattern_max_weight
				<< " block_number=" << block_number
				<< endl;
	}


	if (Crc_object1->info_length_in_bytes != Crc_object2->info_length_in_bytes) {
		cout << "CRC codes must have the same info_length_in_bytes" << endl;
		cout << "first code has " << Crc_object1->info_length_in_bytes << endl;
		cout << "second code has " << Crc_object2->info_length_in_bytes << endl;
		exit(1);
	}
	if (Crc_object1->Len_check_in_bytes != Crc_object2->Len_check_in_bytes) {
		cout << "CRC codes must have the same Len_check_in_bytes" << endl;
		exit(1);
	}

	data_structures::algorithms Algo;


	orbiter_kernel_system::file_io Fio;

	long int N, L, nb_blocks, cnt;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "coding_theory_domain::crc_all_errors_of_a_given_weight "
				"input file size = " << N << endl;
	}

	nb_blocks = Crc_object1->get_nb_blocks(N);

	if (f_v) {
		cout << "coding_theory_domain::crc_all_errors_of_a_given_weight "
				"nb_blocks = " << nb_blocks << endl;
	}


	unsigned char *Data0;
	unsigned char *Data0e;
	unsigned char *Data1;
	unsigned char *Data2;
	unsigned char *Check1;
	unsigned char *Check2;
	unsigned char *Check1a;
	unsigned char *Check2a;
	unsigned char *Check1b;
	unsigned char *Check2b;

	Data0 = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Data0e = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Data1 = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Data2 = (unsigned char *) NEW_char(Crc_object1->Len_total_in_symbols);
	Check1 = (unsigned char *) NEW_char(Crc_object1->Len_check_in_symbols);
	Check2 = (unsigned char *) NEW_char(Crc_object1->Len_check_in_symbols);
	Check1a = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);
	Check2a = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);
	Check1b = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);
	Check2b = (unsigned char *) NEW_char(Crc_object1->Len_check_in_bytes);


	error_pattern_generator *Error;


	Error = NEW_OBJECT(error_pattern_generator);

	Error->init(Crc_object1, error_pattern_max_weight + 1, verbose_level);


	long int nb1 = 0;
	long int nb2 = 0;
	long int nb_error_pattern_zero = 0;
	int f_show = false;


	ifstream ist(fname_in, ios::binary);

	{

		for (cnt = 0; cnt < nb_blocks; cnt++) {


			if (cnt != block_number) {
				continue;
			}


			L = Crc_object1->get_this_block_size(N, cnt);


			Algo.uchar_zero(Data0, Crc_object1->Len_total_in_symbols);
			Algo.uchar_zero(Data1, Crc_object1->Len_total_in_symbols);
			Algo.uchar_zero(Data2, Crc_object1->Len_total_in_symbols);


			// read one block of information:

			ist.read((char *) Data0 + Crc_object1->Len_check_in_bytes, L);

			//Algo.uchar_move(Data0, Data1, Crc_object1->Len_total_in_bytes);
			//Algo.uchar_move(Data0, Data2, Crc_object1->Len_total_in_bytes);

			if (f_show) {
				cout << "Data0 in bytes: len = " << Crc_object1->Len_total_in_bytes << endl;
				Algo.print_hex(cout, Data0, Crc_object1->Len_total_in_bytes);
				Algo.print_binary(cout, Data0, Crc_object1->Len_total_in_bytes);
			}

			Crc_object1->expand(Data0, Data1);

			if (f_show) {
				cout << "after expand Data1:" << endl;
				Algo.print_hex(cout, Data1, Crc_object1->Len_total_in_symbols);
				Algo.print_binary(cout, Data1, Crc_object1->Len_total_in_symbols);
			}

			Crc_object2->expand(Data0, Data2);


			Crc_object1->divide(Data1, Check1);
			Crc_object2->divide(Data2, Check2);

			if (f_show) {
				cout << "Check1:" << endl;
				Algo.print_hex(cout, Check1, Crc_object1->Len_check_in_symbols);
				Algo.print_binary(cout, Check1, Crc_object1->Len_check_in_symbols);
			}


			Crc_object1->compress_check(Check1, Check1a);
			Crc_object2->compress_check(Check2, Check2a);

			if (f_show) {
				cout << "Check1 compressed:" << endl;
				Algo.print_hex(cout, Check1a, Crc_object1->Len_check_in_bytes);
				Algo.print_binary(cout, Check1a, Crc_object1->Len_check_in_bytes);
			}


			int wt;
			long int N, N100;
			combinatorics::combinatorics_domain Combi;
			data_structures::data_structures_global DataStructures;

			for (wt = error_pattern_max_weight; wt <= error_pattern_max_weight; wt++) {


				cout << "wt = " << wt << endl;
				nb1 = 0;
				nb2 = 0;
				nb_error_pattern_zero = 0;

				N = Error->number_of_bit_error_patters(wt, verbose_level);

				N100 = N / 100;

				long int counter = 0;

				Error->first_bit_error_pattern_of_given_weight(
						Combi,
						Algo,
						DataStructures,
						wt,
						0 /*verbose_level */);




				while (true) {



					if ((counter % N100) == 0) {
						if (f_v) {
							cout << "coding_theory_domain::crc_all_errors_of_a_given_weight "
									"counter " << counter << " / " << N << " = "
									<< ((double)counter / (double) N100) << " percent "
											"nb undetected errors =  " << nb1 << "," << nb2 << endl;
						}
					}

#if 0
					if (counter != 1394203) {
						 goto go_to_next;
					}
#endif

					int i;

					for (i = 0; i < Crc_object1->Len_total_in_bytes; i++) {
						if (Error->Error_in_bytes[i]) {
							break;
						}
					}

					if (i == Crc_object1->Len_total_in_bytes) {

						// don't do anything at all, the error pattern is zero;

						nb_error_pattern_zero++;
					}
					else {

						//Algo.print_hex(cout, Error->Error, Crc_object1->Len_total_in_bytes);

						if (f_show) {
							cout << "Error_in_bytes:" << endl;
							Algo.print_hex(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
							Algo.print_binary(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);

							cout << "before adding error in bytes: "
									"len = " << Crc_object1->Len_total_in_bytes << endl;
							Algo.print_hex(cout, Data0, Crc_object1->Len_total_in_bytes);
							Algo.print_binary(cout, Data0, Crc_object1->Len_total_in_bytes);
						}

						Algo.uchar_xor(Data0, Error->Error_in_bytes, Data0e, Crc_object1->Len_total_in_bytes);


						if (f_show) {
							cout << "after adding error in bytes: "
									"len = " << Crc_object1->Len_total_in_bytes << endl;
							Algo.print_hex(cout, Data0e, Crc_object1->Len_total_in_bytes);
							Algo.print_binary(cout, Data0e, Crc_object1->Len_total_in_bytes);
						}


						Crc_object1->expand(Data0e, Data1);
						Crc_object2->expand(Data0e, Data2);

						if (f_show) {
							cout << "after expand: "
									"len = " << Crc_object1->Len_total_in_symbols << endl;
							Algo.print_hex(cout, Data1, Crc_object1->Len_total_in_symbols);
							Algo.print_binary(cout, Data1, Crc_object1->Len_total_in_symbols);
						}

						Crc_object1->divide(Data1, Check1);
						Crc_object2->divide(Data2, Check2);

						if (f_show) {
							cout << "Check1:" << endl;
							Algo.print_hex(cout, Check1, Crc_object1->Len_check_in_symbols);
							Algo.print_binary(cout, Check1, Crc_object1->Len_check_in_symbols);
						}


						Crc_object1->compress_check(Check1, Check1b);
						Crc_object2->compress_check(Check2, Check2b);

						if (f_show) {
							cout << "Check1b compressed:" << endl;
							Algo.print_hex(cout, Check1b, Crc_object1->Len_check_in_bytes);
							Algo.print_binary(cout, Check1b, Crc_object1->Len_check_in_bytes);
						}


						if (Algo.uchar_compare(Check1a, Check1b, Crc_object1->Len_check_in_bytes) == 0) {
							nb1++;
							if (f_show) {
								cout << "wt = " << wt << " undetected error code 1 : " << nb1
										<< " counter " << counter << " / " << N << " = "
										<< ((double)counter / (double) N100) << " % " << endl;
								Algo.print_hex(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
								Algo.print_binary(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
							}
						}
						if (Algo.uchar_compare(Check2a, Check2b, Crc_object1->Len_check_in_bytes) == 0) {
							nb2++;
							if (f_show) {
								cout << "wt = " << wt << " undetected error code 2 : " << nb2
										<< " counter " << counter << " / " << N << " = "
										<< ((double)counter / (double) N100) << " % " << endl;
								//Algo.print_hex(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
								//Algo.print_binary(cout, Error->Error_in_bytes, Crc_object1->Len_total_in_bytes);
							}
						}
					}


//go_to_next:
					if (!Error->next_bit_error_pattern_of_given_weight(
							Combi,
							Algo,
							DataStructures,
							wt,
							0 /*verbose_level */)) {
						break;
					}
					counter++;

				}

				cout << "nb_error_pattern_zero = " << nb_error_pattern_zero << endl;

				cout << "wt = " << wt << " N = " << N << ", # undetected errors = " << nb1 << ", " << nb2 << endl;


			} // next wt
		}

	}

	cout << "nb_blocks = " << nb_blocks << endl;


	FREE_char((char *) Data0);
	FREE_char((char *) Data0e);
	FREE_char((char *) Data1);
	FREE_char((char *) Data2);
	FREE_char((char *) Check1);
	FREE_char((char *) Check2);
	FREE_char((char *) Check1a);
	FREE_char((char *) Check2a);
	FREE_char((char *) Check1b);
	FREE_char((char *) Check2b);


	if (f_v) {
		cout << "coding_theory_domain::crc_all_errors_of_a_given_weight done" << endl;
	}

}


void coding_theory_domain::crc_weight_enumerator_bottom_up(
		crc_object *Crc_object,
		int error_pattern_max_weight,
		int f_collect_words,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc_weight_enumerator_bottom_up "
				<< " error_pattern_max_weight=" << error_pattern_max_weight
				<< endl;
	}



	unsigned char *Data0;
	unsigned char *Data0e;
	unsigned char *Data1;
	unsigned char *Check1;
	unsigned char *Check1b;

	Data0 = (unsigned char *) NEW_char(Crc_object->Len_total_in_symbols);
	Data0e = (unsigned char *) NEW_char(Crc_object->Len_total_in_symbols);
	Data1 = (unsigned char *) NEW_char(Crc_object->Len_total_in_symbols);
	Check1 = (unsigned char *) NEW_char(Crc_object->Len_check_in_symbols);
	Check1b = (unsigned char *) NEW_char(Crc_object->Len_check_in_bytes);


	error_pattern_generator *Error;


	Error = NEW_OBJECT(error_pattern_generator);

	Error->init(Crc_object, error_pattern_max_weight + 1, verbose_level);


	int f_show = false;
	long int nb1;
	long int nb_error_pattern_zero;
	int wt;
	long int N, N100;
	combinatorics::combinatorics_domain Combi;
	data_structures::algorithms Algo;
	data_structures::data_structures_global DataStructures;
	long int *Weight_enumerator;


	Weight_enumerator = NEW_lint(error_pattern_max_weight + 1);
	Lint_vec_zero(Weight_enumerator, error_pattern_max_weight + 1);
	Weight_enumerator[0] = 1;

	Algo.uchar_zero(Data0, Crc_object->Len_total_in_symbols);
	Algo.uchar_zero(Data1, Crc_object->Len_total_in_symbols);

	for (wt = 1; wt <= error_pattern_max_weight; wt++) {


		vector<long int> Words;


		cout << "wt = " << wt << endl;
		nb1 = 0;
		nb_error_pattern_zero = 0;

		N = Error->number_of_bit_error_patters(wt, verbose_level);

		N100 = (N / 100) + 1;

		long int counter = 0;

		Error->first_bit_error_pattern_of_given_weight(
				Combi,
				Algo,
				DataStructures,
				wt,
				0 /*verbose_level */);




		while (true) {



			if ((counter % N100) == 0) {
				if (f_v) {
					cout << "coding_theory_domain::crc_simulate_errors "
							" wt = " << wt << " counter " << counter << " / " << N << " = "
							<< ((double)counter / (double) N100) << " percent "
									"nb undetected errors =  " << nb1 << endl;
				}
			}


			int i;

			for (i = 0; i < Crc_object->Len_total_in_bytes; i++) {
				if (Error->Error_in_bytes[i]) {
					break;
				}
			}

			if (i == Crc_object->Len_total_in_bytes) {

				// don't do anything at all, the error pattern is zero;

				nb_error_pattern_zero++;
			}
			else {

				//Algo.print_hex(cout, Error->Error, Crc_object1->Len_total_in_bytes);

				if (f_show) {
					cout << "Error_in_bytes:" << endl;
					Algo.print_hex(cout, Error->Error_in_bytes, Crc_object->Len_total_in_bytes);
					Algo.print_binary(cout, Error->Error_in_bytes, Crc_object->Len_total_in_bytes);

					cout << "before adding error in bytes: len = " << Crc_object->Len_total_in_bytes << endl;
					Algo.print_hex(cout, Data0, Crc_object->Len_total_in_bytes);
					Algo.print_binary(cout, Data0, Crc_object->Len_total_in_bytes);
				}

				Algo.uchar_xor(
						Data0,
						Error->Error_in_bytes,
						Data0e,
						Crc_object->Len_total_in_bytes);


				if (f_show) {
					cout << "after adding error in bytes: len = " << Crc_object->Len_total_in_bytes << endl;
					Algo.print_hex(cout, Data0e, Crc_object->Len_total_in_bytes);
					Algo.print_binary(cout, Data0e, Crc_object->Len_total_in_bytes);
				}


				Crc_object->expand(Data0e, Data1);

				if (f_show) {
					cout << "after expand: len = " << Crc_object->Len_total_in_symbols << endl;
					Algo.print_hex(cout, Data1, Crc_object->Len_total_in_symbols);
					Algo.print_binary(cout, Data1, Crc_object->Len_total_in_symbols);
				}

				Crc_object->divide(Data1, Check1);

				if (f_show) {
					cout << "Check1:" << endl;
					Algo.print_hex(cout, Check1, Crc_object->Len_check_in_symbols);
					Algo.print_binary(cout, Check1, Crc_object->Len_check_in_symbols);
				}


				Crc_object->compress_check(Check1, Check1b);

				if (f_show) {
					cout << "Check1b compressed:" << endl;
					Algo.print_hex(cout, Check1b, Crc_object->Len_check_in_bytes);
					Algo.print_binary(cout, Check1b, Crc_object->Len_check_in_bytes);
				}


				if (Algo.uchar_is_zero(Check1b, Crc_object->Len_check_in_bytes)) {
					nb1++;
					if (f_show) {
						cout << "wt = " << wt << " undetected error : " << nb1
								<< " counter " << counter << " / " << N << " = "
								<< ((double)counter / (double) N100) << " % " << endl;
						Algo.print_hex(cout, Error->Error_in_bytes, Crc_object->Len_total_in_bytes);
						Algo.print_binary(cout, Error->Error_in_bytes, Crc_object->Len_total_in_bytes);
					}

					if (f_collect_words) {
						Words.push_back(counter);
					}
				}
			}


			if (!Error->next_bit_error_pattern_of_given_weight(
					Combi,
					Algo,
					DataStructures,
					wt,
					0 /*verbose_level */)) {
				break;
			}
			counter++;

		}

		cout << "nb_error_pattern_zero = " << nb_error_pattern_zero << endl;

		cout << "wt = " << wt << " N = " << N << ", # undetected errors = " << nb1 << endl;

		Weight_enumerator[wt] = nb1;

		orbiter_kernel_system::file_io Fio;
		string fname;

		fname = Crc_object->label_txt + "_weight_enumerator.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, Weight_enumerator, wt + 1, 1);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		if (f_collect_words) {
			long int *Long_words;
			long int i, a;

			Long_words = NEW_lint(Words.size());
			for (i = 0; i < Words.size(); i++) {
				a = Words[i];
				Long_words[i] = a;
			}

			string fname;

			fname = Crc_object->label_txt + "_words_of_weight_" + std::to_string(wt) + ".csv";

			Fio.Csv_file_support->lint_matrix_write_csv(
					fname, Long_words, Words.size(), 1);

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

			FREE_lint(Long_words);

		}

	} // next wt

	FREE_char((char *) Data0);
	FREE_char((char *) Data0e);
	FREE_char((char *) Data1);
	FREE_char((char *) Check1);
	FREE_char((char *) Check1b);

	FREE_OBJECT(Error);

	if (f_v) {
		cout << "coding_theory_domain::crc_weight_enumerator_bottom_up done" << endl;
	}
}

void coding_theory_domain::read_error_pattern_from_output_file(
		std::string &fname_in,
		int nb_lines,
		crc_object *Crc_object1,
		crc_object *Crc_object2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::read_error_pattern_from_output_file "
				" fname_in=" << fname_in
				<< endl;
	}

	orbiter_kernel_system::file_io Fio;
	std::vector<std::vector<int> > Error1;
	std::vector<std::vector<int> > Error2;

	Fio.read_error_pattern_from_output_file(
			fname_in,
			nb_lines,
			Error1,
			Error2,
			verbose_level);


	cout << "Number of errors in code 1 = " << Error1.size() << endl;
	cout << "Number of errors in code 2 = " << Error2.size() << endl;

	int h;

	for (h = 0; h < 2; h++) {

		std::vector<std::vector<int> > *Error;

		if (h == 0) {
			Error = &Error1;
		}
		else {
			Error = &Error2;
		}
		int *E;
		int m, n;
		int i, j;

		m = (*Error).size();
		if (m == 0) {
			continue;
		}
		n = (*Error)[0].size();

		E = NEW_int(m * n);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				E[i * n + j] = (*Error)[i][j];
			}
		}
		orbiter_kernel_system::file_io Fio;
		data_structures::string_tools ST;
		string fname;

		fname = fname_in;
		ST.chop_off_extension_and_path(fname);

		fname += "_error_" + std::to_string(h + 1) + ".csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, E, m, n);

		cout << "written file "
			<< fname << " of size " << Fio.file_size(fname) << endl;

	}


	if (f_v) {
		cout << "coding_theory_domain::read_error_pattern_from_output_file "
				"done" << endl;
	}

}




}}}

