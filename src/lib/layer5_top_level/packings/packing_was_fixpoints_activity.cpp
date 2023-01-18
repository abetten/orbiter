/*
 * packing_was_fixpoints_activity.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


packing_was_fixpoints_activity::packing_was_fixpoints_activity()
{
	Descr = NULL;
	PWF = NULL;

}

packing_was_fixpoints_activity::~packing_was_fixpoints_activity()
{

}



void packing_was_fixpoints_activity::init(
		packing_was_fixpoints_activity_description *Descr,
		packing_was_fixpoints *PWF,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints_activity::init" << endl;
	}

	packing_was_fixpoints_activity::Descr = Descr;
	packing_was_fixpoints_activity::PWF = PWF;

	if (f_v) {
		cout << "packing_was_fixpoints_activity::init done" << endl;
	}
}

void packing_was_fixpoints_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Descr->f_report) {


		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity before PW->report" << endl;
		}

		PWF->report(verbose_level);

		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity after PW->report" << endl;
		}


	}
	else if (Descr->f_print_packing) {


		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity f_print_packing" << endl;
		}

		long int *packing;
		int sz;

		Lint_vec_scan(Descr->print_packing_text, packing, sz);

		PWF->print_packing(packing, sz, verbose_level);




		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity f_print_packing" << endl;
		}


	}
	else if (Descr->f_compare_files_of_packings) {


		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity f_print_packing" << endl;
		}

		orbiter_kernel_system::file_io Fio;
		int *M1;
		int *M2;
		int m1, n1;
		int m2, n2;
		int len1, len2;

		Fio.int_matrix_read_csv(Descr->compare_files_of_packings_fname1, M1,
				m1, n1, verbose_level);

		Fio.int_matrix_read_csv(Descr->compare_files_of_packings_fname2, M2,
				m2, n2, verbose_level);


		len1 = m1 * n1;
		len2 = m2 * n2;

		data_structures::sorting Sorting;

		Sorting.int_vec_sort_and_remove_duplicates(M1, len1);
		Sorting.int_vec_sort_and_remove_duplicates(M2, len2);


		int *v3;
		int len3;

		v3 = NEW_int(len1 + len2);
		Sorting.int_vec_intersect_sorted_vectors(M1, len1,
				M2, len2, v3, len3);


		cout << "The intersection has size " << len3 << ":" << endl;
		Int_vec_print(cout, v3, len3);
		cout << endl;




		if (f_v) {
			cout << "packing_was_fixpoints_activity::perform_activity f_print_packing" << endl;
		}


	}





	if (f_v) {
		cout << "packing_was_fixpoints_activity::perform_activity" << endl;
	}

}

}}}



