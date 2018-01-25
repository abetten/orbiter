// file_output.C
//
// Anton Betten
// January 8, 2016
//

#include "galois.h"

file_output::file_output()
{
	null();
}

file_output::~file_output()
{
	freeself();
}

void file_output::null()
{
	f_file_is_open = FALSE;
	fp = NULL;
}

void file_output::freeself()
{
	if (f_file_is_open) {
		close();
		}
	null();
}


void file_output::open(const BYTE *fname, void *user_data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_output::open" << endl;
		}
	strcpy(file_output::fname, fname);
	file_output::user_data = user_data;
	
	fp = new ofstream;
	fp->open(fname);
	f_file_is_open = TRUE;
	

	
	if (f_v) {
		cout << "file_output::open done" << endl;
		}
}

void file_output::close()
{
	*fp << "-1" << endl;
	delete fp;
	fp = NULL;
	f_file_is_open = FALSE;
}

void file_output::write_line(INT nb, INT *data, INT verbose_level)
{
	INT i;

	if (!f_file_is_open) {
		cout << "file_output::write_line file is not open" << endl;
		exit(1);
		}
	*fp << nb;
	for (i = 0; i < nb; i++) {
		*fp << " " << data[i];
		}
	*fp << endl;
}



