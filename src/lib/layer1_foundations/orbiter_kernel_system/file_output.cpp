// file_output.cpp
//
// Anton Betten
// January 8, 2016
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {


file_output::file_output()
{
	//std::string fname;
	f_file_is_open = false;
	fp = NULL;
	user_data = NULL;
}

file_output::~file_output()
{
	if (f_file_is_open) {
		close();
	}
}


void file_output::open(std::string &fname,
		void *user_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_output::open" << endl;
	}
	file_output::fname.assign(fname);
	file_output::user_data = user_data;
	
	fp = new ofstream;
	fp->open(fname);
	f_file_is_open = true;
	

	
	if (f_v) {
		cout << "file_output::open done" << endl;
	}
}

void file_output::close()
{
	//*fp << "-1" << endl;
	delete fp;
	fp = NULL;
	f_file_is_open = false;
}

void file_output::write_line(int nb, int *data,
		int verbose_level)
{
	int i;

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

void file_output::write_EOF(int nb_sol, int verbose_level)
{
	*fp << "-1 " << nb_sol << endl;
}


}}}

