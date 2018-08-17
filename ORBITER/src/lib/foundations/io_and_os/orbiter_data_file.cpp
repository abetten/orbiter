// orbiter_data_file.C
// 
// Anton Betten
// July 30, 2018
//
//
// 
// pulled out of ovoid: Jul 30, 2018
//

#include "foundations.h"


orbiter_data_file::orbiter_data_file()
{
	null();
}

orbiter_data_file::~orbiter_data_file()
{
	freeself();
}

void orbiter_data_file::null()
{
}

void orbiter_data_file::freeself()
{
	INT i;
	
	for (i = 0; i < nb_cases; i++) {
		FREE_BYTE(Ago_ascii[i]);
		FREE_BYTE(Aut_ascii[i]);
		}
	FREE_PBYTE(Ago_ascii);
	FREE_PBYTE(Aut_ascii);

	FREE_INT(set_sizes);
	FREE_INT(Casenumbers);
}

void orbiter_data_file::load(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE **data;
	INT i;


	if (f_v) {
		cout << "orbiter_data_file::load loading file " << fname << endl;
		}


	if (!try_to_read_file(fname, nb_cases, data, verbose_level)) {
		cout << "orbiter_data_file::load couldn't read file " 
			<< fname << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "file read containing " << nb_cases << " cases" << endl;
		}

	if (f_v) {
		cout << "read_and_parse_data_file: parsing sets" << endl;
		}
	
	parse_sets(nb_cases, data, FALSE /*f_casenumbers */, 
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level - 2*/);
	if (f_v) {
		cout << "read_and_parse_data_file: parsing sets done" << endl;
		}

	for (i = 0; i < nb_cases; i++) {
		FREE_BYTE(data[i]);
		}
	FREE_PBYTE(data);
	
	if (f_v) {
		cout << "orbiter_data_file::load done nb_cases = " << nb_cases << endl;
		}
}


