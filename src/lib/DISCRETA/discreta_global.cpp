// discreta_global.cpp
//
// Anton Betten
// Nov 19, 2007

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace discreta {

void free_global_data()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "discreta_global free_global_data freeing global data" << endl;
	}
	orthogonal_points_free_global_data();
	combinatorics_domain_free_global_data();
	combinatorics_domain_free_tab_q_binomials();
}

void the_end(int t0)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	file_io Fio;
	os_interface Os;

	if (f_v) {
		cout << "***************** The End **********************" << endl;
		//cout << "nb_calls_to_finite_field_init="
		//		<< nb_calls_to_finite_field_init << endl;
	}
	free_global_data();
	if (f_v) {
		if (f_memory_debug) {
			//registry_dump();
			//registry_dump_sorted();
			}
	}
	Os.time_check(cout, t0);
	cout << endl;


	if (f_v) {
		int mem_usage;
		string fname;

		mem_usage = Os.os_memory_usage();
		fname.assign("memory_usage.csv");
		Fio.int_matrix_write_csv(fname, &mem_usage, 1, 1);
	}
}

void the_end_quietly(int t0)
{
	os_interface Os;

	//cout << "discreta_global the_end_quietly: freeing global data" << endl;
	free_global_data();
	Os.time_check(cout, t0);
	cout << endl;
}



}}

