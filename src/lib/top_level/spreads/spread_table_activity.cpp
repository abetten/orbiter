/*
 * spread_table_activity.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


spread_table_activity::spread_table_activity()
{
	Descr = NULL;
	P = NULL;

}

spread_table_activity::~spread_table_activity()
{

}



void spread_table_activity::init(spread_table_activity_description *Descr,
		packing_classify *P,
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
		sorting Sorting;



		Orbiter->Lint_vec.scan(Descr->find_spread_text, spread_elts, sz);

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
	else if (Descr->f_print_spreads) {
		cout << "f_print_spread" << endl;

		int *idx;
		int nb;

		Orbiter->Int_vec.scan(Descr->print_spreads_idx_text, idx, nb);

		cout << "before report_spreads" << endl;
		report_spreads(idx, nb, verbose_level);

	}

	else if (Descr->f_export_spreads_to_csv) {
		cout << "f_export_spreads_to_csv" << endl;

		int *idx;
		int nb;

		Orbiter->Int_vec.scan(Descr->export_spreads_to_csv_idx_text, idx, nb);

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
		Orbiter->Int_vec.zero(N, P->P3->N_lines);

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
		tally N_t;

		N_t.init(N, P->P3->N_lines, FALSE, 0);
		cout << "type of covering based on all lines together with line " << line1 << ":" << endl;
		N_t.print(TRUE);
		cout << endl;

	}
	if (f_v) {
		cout << "spread_table_activity::perform_activity" << endl;

	}

}


void spread_table_activity::export_spreads_to_csv(std::string &fname, int *spread_idx, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "spread_table_activity::export_spreads_to_csv" << endl;
	}

	long int *T;
	int i, j, idx;
	file_io Fio;

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

void spread_table_activity::report_spreads(int *spread_idx, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "spread_table_activity::report_spreads" << endl;
	}

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Spreads");
		//strcpy(author, "");
		author[0] = 0;

		sprintf(str, "Spreads");
		fname.assign(str);



		int i, idx;


		for (i = 0; i < nb; i++) {
			idx = spread_idx[i];
			sprintf(str, "_%d", idx);
			fname.append(str);
		}
		fname.append(".tex");

		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


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
		file_io Fio;

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
	Orbiter->Lint_vec.print(ost, spread_elts, P->spread_size);
	ost << "\\\\" << endl;

	P->P3->Grass_lines->print_set_tex(ost, spread_elts, P->spread_size);

	if (f_v) {
		cout << "spread_table_activity::report_spread2 done" << endl;
	}
}




}}


