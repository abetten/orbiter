/*
 * spread_table_activity.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_table_activity::spread_table_activity()
{
	Descr = NULL;
	P = NULL;

}

spread_table_activity::~spread_table_activity()
{

}



void spread_table_activity::init(
		spreads::spread_table_activity_description *Descr,
		packings::packing_classify *P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_activity::init" << endl;
	}

	spread_table_activity::Descr = Descr;
	spread_table_activity::P = P;

	if (f_v) {
		cout << "spread_table_activity::init done" << endl;
	}
}

void spread_table_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Descr->f_find_spread) {
		cout << "f_find_spread" << endl;
		long int *spread_elts;
		int sz;
		int idx;
		data_structures::sorting Sorting;



		Lint_vec_scan(Descr->find_spread_text, spread_elts, sz);

		if (sz != P->spread_size) {
			cout << "the set does not have the right size" << endl;
			cout << "sz=" << sz << endl;
			cout << "P->spread_size=" << P->spread_size << endl;
			exit(1);
		}

		Sorting.lint_vec_heapsort(spread_elts, sz);

		idx = P->find_spread(spread_elts, verbose_level);

		cout << "The given spread has index " << idx << " in the spread table" << endl;


	}
	else if (Descr->f_find_spread_and_dualize) {
		cout << "f_find_spread_and_dualize" << endl;
		long int *spread_elts;
		int sz;
		int a, b;
		data_structures::sorting Sorting;



		Lint_vec_scan(Descr->find_spread_and_dualize_text, spread_elts, sz);

		if (sz != P->spread_size) {
			cout << "the set does not have the right size" << endl;
			cout << "sz=" << sz << endl;
			cout << "P->spread_size=" << P->spread_size << endl;
			exit(1);
		}

		Sorting.lint_vec_heapsort(spread_elts, sz);


		a = P->find_spread(spread_elts, verbose_level);

		cout << "The given spread has index " << a << " in the spread table" << endl;

		b = P->Spread_table_with_selection->Spread_tables->dual_spread_idx[a];

		cout << "The dual spread has index " << b << " in the spread table" << endl;

	}
	else if (Descr->f_dualize_packing) {
		cout << "f_dualize_packing" << endl;
		long int *packing;
		int sz;
		long int *dual_packing;
		int a, b;
		data_structures::sorting Sorting;



		Lint_vec_scan(Descr->dualize_packing_text, packing, sz);

		cout << "The packing is : ";
		Lint_vec_print(cout, packing, sz);
		cout << endl;

		dual_packing = NEW_lint(sz);
		for (int i = 0; i < sz; i++) {
			a = packing[i];
			b = P->Spread_table_with_selection->Spread_tables->dual_spread_idx[a];
			dual_packing[i] = b;
		}

		cout << "The dual packing is : ";
		Lint_vec_print(cout, dual_packing, sz);
		cout << endl;

	}
	else if (Descr->f_print_spreads) {
		cout << "f_print_spread" << endl;

		int *idx;
		int nb;

		Int_vec_scan(Descr->print_spreads_idx_text, idx, nb);

		cout << "before report_spreads" << endl;
		report_spreads(idx, nb, verbose_level);

	}

	else if (Descr->f_export_spreads_to_csv) {
		cout << "f_export_spreads_to_csv" << endl;

		int *idx;
		int nb;

		Int_vec_scan(Descr->export_spreads_to_csv_idx_text, idx, nb);

		cout << "before export_spreads_to_csv" << endl;
		export_spreads_to_csv(Descr->export_spreads_to_csv_fname, idx, nb, verbose_level);

	}

	else if (Descr->f_find_spreads_containing_two_lines) {
		cout << "f_find_spreads_containing_two_lines" << endl;

		std::vector<int> v;
		int line1 = Descr->find_spreads_containing_two_lines_line1;
		int line2 = Descr->find_spreads_containing_two_lines_line2;
		int i;

		P->Spread_table_with_selection->find_spreads_containing_two_lines(v,
					line1,
					line2,
					verbose_level);

		cout << "We found " << v.size() << " spreads containing " << line1 << " and " << line2 << endl;
		cout << "They are:" << endl;
		for (i = 0; i < v.size(); i++) {
			cout << v[i];
			if (i < v.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;

	}

	else if (Descr->f_find_spreads_containing_one_line) {
		cout << "f_find_spreads_containing_one_line" << endl;

		int line1 = Descr->find_spreads_containing_one_line_line_idx;
		int line2;
		int *N;

		N = NEW_int(P->P3->N_lines);
		Int_vec_zero(N, P->P3->N_lines);

		for (line2 = 0; line2 < P->P3->N_lines; line2++) {
			if (line2 == line1) {
				continue;
			}
			{
				std::vector<int> v;
				P->Spread_table_with_selection->find_spreads_containing_two_lines(v,
						line1,
						line2,
						verbose_level);
				N[line2] = v.size();
			}
		}
		data_structures::tally N_t;

		N_t.init(N, P->P3->N_lines, FALSE, 0);
		cout << "type of covering based on all lines together with line " << line1 << ":" << endl;
		N_t.print(TRUE);
		cout << endl;

	}
	if (f_v) {
		cout << "spread_table_activity::perform_activity" << endl;

	}

}


void spread_table_activity::export_spreads_to_csv(
		std::string &fname,
		int *spread_idx, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "spread_table_activity::export_spreads_to_csv" << endl;
	}

	long int *T;
	int i, j, idx;
	orbiter_kernel_system::file_io Fio;

	T = NEW_lint(nb * P->spread_size);
	for (i = 0; i < nb; i++) {
		long int *spread_elts;

		idx = spread_idx[i];
		spread_elts = P->Spread_table_with_selection->get_spread(idx);
		for (j = 0; j < P->spread_size; j++) {
			T[i * P->spread_size + j] = spread_elts[j];
		}
	}
	Fio.lint_matrix_write_csv(fname, T, nb, P->spread_size);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "spread_table_activity::export_spreads_to_csv done" << endl;
	}
}

void spread_table_activity::report_spreads(
		int *spread_idx, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "spread_table_activity::report_spreads" << endl;
	}

	{
		char str[1000];
		string fname;
		string title, author, extra_praeamble;

		snprintf(str, 1000, "Spreads");
		title.assign(str);


		snprintf(str, sizeof(str), "Spreads");
		fname.assign(str);



		int i, idx;


		for (i = 0; i < nb; i++) {
			idx = spread_idx[i];
			snprintf(str, sizeof(str), "_%d", idx);
			fname.append(str);
		}
		fname.append(".tex");

		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "spread_table_activity::report_spread before report_spread2" << endl;
			}



			for (i = 0; i < nb; i++) {
				idx = spread_idx[i];
				report_spread2(ost, idx, verbose_level);
			}
			if (f_v) {
				cout << "spread_table_activity::report_spread after report_spread2" << endl;
			}

			if (f_v) {
				cout << "spread_table_activity::report_spread after report_spread2" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "spread_table_activity::report_spread done" << endl;
	}
}

void spread_table_activity::report_spread2(std::ostream &ost, int spread_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_activity::report_spread2" << endl;
	}

	long int *spread_elts;


	spread_elts = P->Spread_table_with_selection->get_spread(spread_idx);

	ost << "The spread " << spread_idx << " is:\\\\" << endl;
	Lint_vec_print(ost, spread_elts, P->spread_size);
	ost << "\\\\" << endl;

	P->P3->Grass_lines->print_set_tex(ost, spread_elts, P->spread_size, 0 /* verbose_level */);

	if (f_v) {
		cout << "spread_table_activity::report_spread2 done" << endl;
	}
}




}}}


