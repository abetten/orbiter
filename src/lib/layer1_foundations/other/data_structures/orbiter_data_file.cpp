/*
 * orbiter_data_file.cpp
 *
 *  Created on: Oct 23, 2026
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {



orbiter_data_file::orbiter_data_file()
{
	Record_birth();


	//std::string fname;

	f_casenumbers = false;
	nb_cases = -1;
	Set_sizes = NULL;
	Sets = NULL;
	Ago_ascii = NULL;
	Aut_ascii = NULL;
	Casenumbers = NULL;
}

orbiter_data_file::~orbiter_data_file()
{
	Record_death();

	int i;

	if (Ago_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_char(Ago_ascii[i]);
		}
		FREE_pchar(Ago_ascii);
	}
	if (Aut_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_char(Aut_ascii[i]);
		}
		FREE_pchar(Aut_ascii);
	}
	if (Sets) {
		for (i = 0; i < nb_cases; i++) {
			FREE_lint(Sets[i]);
		}
		FREE_plint(Sets);
	}
	if (Set_sizes) {
		FREE_int(Set_sizes);
	}
	if (Casenumbers) {
		FREE_int(Casenumbers);
	}

}


void orbiter_data_file::init_from_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);


	if (f_v) {
		cout << "orbiter_data_file::init_from_file" << endl;
	}


	other::orbiter_kernel_system::file_io Fio;

	orbiter_data_file::fname = fname;

	if (f_v) {
		cout << "orbiter_data_file::init_from_file "
				"reading file " << fname << endl;
	}

	Fio.read_and_parse_data_file_fancy(
			fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 1);

	if (f_vv) {
		cout << "orbiter_data_file::init_from_file "
				"after read_and_parse_data_file_fancy" << endl;
	}


	if (f_v) {
		cout << "orbiter_data_file::init_from_file done" << endl;
	}
}


}}}}

