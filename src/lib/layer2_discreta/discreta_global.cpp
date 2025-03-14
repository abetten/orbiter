// discreta_global.cpp
//
// Anton Betten
// Nov 19, 2007

#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

using namespace std;


namespace orbiter {
namespace layer2_discreta {
namespace typed_objects {


void free_global_data()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "discreta_global free_global_data" << endl;
	}
	//orthogonal_points_free_global_data();
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Combi.free_global_data();
	Combi.free_tab_q_binomials();
	if (f_v) {
		cout << "discreta_global free_global_data done" << endl;
	}
}

void the_end(int t0)
{
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		 cout << "the_end" << endl;
	}
	if (f_v) {
		cout << "***************** The End **********************" << endl;
		//cout << "nb_calls_to_finite_field_init="
		//		<< nb_calls_to_finite_field_init << endl;
	}
	free_global_data();
	if (f_v) {
		 cout << "the_end after free_global_data" << endl;
	}
	if (f_v) {
		if (other::orbiter_kernel_system::Orbiter->f_memory_debug) {
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
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, &mem_usage, 1, 1);
	}
	if (f_v) {
		 cout << "the_end done" << endl;
	}
}

void the_end_quietly(int t0)
{
	other::orbiter_kernel_system::os_interface Os;

	//cout << "discreta_global the_end_quietly: freeing global data" << endl;
	free_global_data();
	Os.time_check(cout, t0);
	cout << endl;
}



}}}

