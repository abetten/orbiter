/*
 * crc_process.cpp
 *
 *  Created on: Dec 9, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


crc_process::crc_process()
{
	Descr = NULL;

	Code = NULL;

	block_length = 0;
	information_length = 0;
	check_size = 0;

	N = 0;
	nb_blocks = 0;
	buffer = NULL;
	check_data = NULL;
}

crc_process::~crc_process()
{
}


void crc_process::init(
		crc_process_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "crc_process::init" << endl;
	}

	crc_process::Descr = Descr;

	if (!Descr->f_code) {
		cout << "crc_process::init please use -code <code_label>" << endl;
		exit(1);
	}
	if (!Descr->f_crc_options) {
		cout << "crc_process::init please use -crc_options <CRC-options>" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "crc_process::init "
				"code=" << Descr->code_label << endl;
	}


	Code = Get_object_of_type_code(Descr->code_label);

	if (f_v) {
		cout << "crc_process::init "
				"found code " << Descr->code_label << endl;
	}

	if (f_v) {
		cout << "crc_process::init done" << endl;
	}
}


#if 0
void crc_process::encode_file(
		std::string &fname_in,
		std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "crc_process::encode_file fname_in=" << fname_in << endl;
	}

	data_structures::string_tools ST;
	int block_length = 771;
	int redundancy = 30;
	int information_length = block_length - redundancy;

	orbiter_kernel_system::file_io Fio;

	//long int N, C, L;
	char *buffer;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "crc_process::encode_file input file size = " << N << endl;
	}
	buffer = NEW_char(block_length);


	ifstream ist(fname_in, ios::binary);

	{
		ofstream ost(fname_out, ios::binary);
		//uint32_t crc;
		//char *p_crc;
		int i;

		//p_crc = (char *) &crc;
		C = 0;

		while (C < N) {

			if (C + information_length > N) {
				L = C + information_length - N;
			}
			else {
				L = information_length;
			}
			ist.read(buffer + redundancy, L);
			for (i = 0; i < redundancy; i++) {
				buffer[i] = 0;
			}
			for (i = L; i < block_length; i++) {
				buffer[i] = 0;
			}

			CRC_BCH256_771_divide(buffer, buffer);

			ost.write(buffer, block_length);


			C += information_length;
		}

	}
	cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;



	if (f_v) {
		cout << "crc_process::encode_file done" << endl;
	}

}
#endif

void crc_process::encode_file(
		std::string &fname_in, std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "crc_process::encode_file "
				"fname_in=" << fname_in << endl;
		cout << "crc_process::encode_file "
				"block_length=" << block_length << endl;
	}

	data_structures::string_tools ST;

	orbiter_kernel_system::file_io Fio;

	//long int N, L, nb_blocks, cnt;
	//char *buffer;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "crc_process::encode_file input file size = " << N << endl;
	}

	check_size = 4;
	information_length = block_length - check_size;
	nb_blocks = (N + information_length - 1) / information_length;
	if (f_v) {
		cout << "crc_process::encode_file nb_blocks = " << nb_blocks << endl;
	}

	buffer = NEW_char(block_length);


	check_data = NEW_char(check_size);

	ifstream ist(fname_in, ios::binary);

	{
		ofstream ost(fname_out, ios::binary);
		//uint32_t crc;
		//char *p_crc;
		long int cnt;
		long int L;
		int i;

		//p_crc = (char *) &crc;
		//C = 0;

		for (cnt = 0; cnt < nb_blocks; cnt++) {
		//while (C < N) {

			if ((cnt + 1) * information_length > N) {
				L = N - cnt * information_length;
			}
			else {
				L = information_length;
			}

			// read information_length bytes
			// (or less, in case we reached the end of he file)

			ist.read(buffer, L);

			// create 4 byte check and add to the block:

			encode_block(L, verbose_level);

			for (i = 0; i < check_size; i++) {
				buffer[L + i] = check_data[i];
			}


			// write information_length + 4 bytes to file:
			// (or less in case we have reached the end of the input file):

			ost.write(buffer, L + check_size);


			// count the bytes read,
			// so C is the position in the input file:

			//C += L;
		}

	}
	cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;



	if (f_v) {
		cout << "crc_process::encode_file done" << endl;
	}

}

void crc_process::encode_block(
		long int L,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "crc_process::encode_block" << endl;
	}

}


}}}

