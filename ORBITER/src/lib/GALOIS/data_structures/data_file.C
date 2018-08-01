// data_file.C
//
// Anton Betten
//
//
// October 13, 2011
//
//
// 
//
//

#include "galois.h"

data_file::data_file()
{
	null();
}

data_file::~data_file()
{
	freeself();
}

void data_file::null()
{
	nb_cases = -1;
	set_sizes = NULL;
	sets = NULL;
	casenumbers = NULL;
	Ago_ascii = NULL;
	Aut_ascii = NULL;
	f_has_candidates = FALSE;
	nb_candidates = NULL;
	candidates = NULL;
}

void data_file::freeself()
{
	INT i;
	
	cout << "data_file::freeself" << endl;
	if (nb_cases >= 0) {
		for (i = 0; i < nb_cases; i++) {
			FREE_INT(sets[i]);
			FREE_BYTE(Ago_ascii[i]);
			FREE_BYTE(Aut_ascii[i]);
			}
		FREE_INT(set_sizes);
		FREE_PINT(sets);
		FREE_INT(casenumbers);
		FREE_PBYTE(Ago_ascii);
		FREE_PBYTE(Aut_ascii);
		if (f_has_candidates) {
			FREE_INT(nb_candidates);
			for (i = 0; i < nb_cases; i++) {
				FREE_INT(candidates[i]);
				}
			FREE_PINT(candidates);
			}
		}
	null();
	cout << "data_file::freeself done" << endl;
}

void data_file::read(const BYTE *fname, INT f_casenumbers, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_file::read trying to read file " << fname << " of size " << file_size(fname) << endl;
		cout << "f_casenumbers=" << f_casenumbers << endl;
		}
	strcpy(data_file::fname, fname);
	
	read_and_parse_data_file_fancy(fname, 
		f_casenumbers, 
		nb_cases, 
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		casenumbers, 
		verbose_level);
	if (f_v) {
		cout << "data_file::read finished" << endl;
		}	
}

void data_file::read_candidates(const BYTE *candidates_fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, cnt, a, b;

	if (f_v) {
		cout << "data_file::read_candidates trying to read candidates file " << candidates_fname << " of size " << file_size(candidates_fname) << endl;
		}

	nb_candidates = NEW_INT(nb_cases);
	candidates = NEW_PINT(nb_cases);
	{
	ifstream fp(candidates_fname);
	cnt = 0;
	while (TRUE) {
		fp >> a;
		if (a == -1) {
			break;
			}
		for (i = 0; i < a; i++) {
			fp >> b;
			}
		fp >> b;
		if (b != -1) {
			cout << "data_file::read_candidates b != -1" << endl;
			exit(1);
			}
		fp >> nb_candidates[cnt];
		candidates[cnt] = NEW_INT(nb_candidates[cnt]);
		for (i = 0; i < nb_candidates[cnt]; i++) {
			fp >> candidates[cnt][i];
			}
		fp >> b; // read final -1
		cnt++;
		if (cnt > nb_cases) {
			cout << "data_file::read_candidates cnt > nb_cases" << endl;
			exit(1);
			}
		}
	if (cnt != nb_cases) {
		cout << "data_file::read_candidates cnt != nb_cases" << endl;
		exit(1);
		}
	}
	f_has_candidates = TRUE;
	if (f_v) {
		cout << "data_file::read_candidates finished" << endl;
		}	
}


