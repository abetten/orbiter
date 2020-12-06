/*
 * orthogonal_io.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void orthogonal::list_points_by_type(int verbose_level)
{
	int t;

	for (t = 1; t <= nb_point_classes; t++) {
		list_points_of_given_type(t, verbose_level);
	}
}

void orthogonal::report_points_by_type(std::ostream &ost, int verbose_level)
{
	int t;

	for (t = 1; t <= nb_point_classes; t++) {
		report_points_of_given_type(ost, t, verbose_level);
	}
}

void orthogonal::list_points_of_given_type(int t, int verbose_level)
{
	long int i, j, rk, u;

	cout << "points of type P" << t << ":" << endl;
	for (i = 0; i < P[t - 1]; i++) {
		rk = type_and_index_to_point_rk(t, i, verbose_level);
		cout << i << " : " << rk << " : ";
		unrank_point(v1, 1, rk, verbose_level - 1);
		int_vec_print(cout, v1, n);
		point_rk_to_type_and_index(rk, u, j, verbose_level);
		cout << " : " << u << " : " << j << endl;
		if (u != t) {
			cout << "type wrong" << endl;
			exit(1);
		}
		if (j != i) {
			cout << "index wrong" << endl;
			exit(1);
		}
	}
	cout << endl;
}

void orthogonal::report_points_of_given_type(std::ostream &ost, int t, int verbose_level)
{
	long int i, j, rk, u;

	ost << "points of type P" << t << ":\\\\" << endl;
	for (i = 0; i < P[t - 1]; i++) {
		rk = type_and_index_to_point_rk(t, i, verbose_level);
		ost << i << " : " << rk << " : ";
		unrank_point(v1, 1, rk, verbose_level - 1);
		int_vec_print(ost, v1, n);
		ost << "\\\\" << endl;
		point_rk_to_type_and_index(rk, u, j, verbose_level);
		//ost << " : " << u << " : " << j << endl;
		if (u != t) {
			cout << "type wrong" << endl;
			exit(1);
		}
		if (j != i) {
			cout << "index wrong" << endl;
			exit(1);
		}
	}
	//ost << endl;
}

void orthogonal::report_points(std::ostream &ost, int verbose_level)
{
	long int rk;

	ost << "points:\\\\" << endl;
	for (rk = 0; rk < nb_points; rk++) {
		unrank_point(v1, 1, rk, 0 /*verbose_level*/);
		ost << "$P_{" << rk << "} = ";
		int_vec_print(ost, v1, n);
		ost << "$\\\\" << endl;
	}
	//ost << endl;
}

void orthogonal::report_lines(std::ostream &ost, int verbose_level)
{
	int len;
	int i, a, d = n + 1;
	long int p1, p2;
	latex_interface Li;
	sorting Sorting;

	ost << "The number of lines is " << nb_lines << "\\\\" << endl;
	len = nb_lines; // O.L[0] + O.L[1] + O.L[2];


	long int *Line;
	int *L;

	Line = NEW_lint(q + 1);
	L = NEW_int(2 * d);

	for (i = 0; i < len; i++) {
		ost << "$L_{" << i << "} = ";
		unrank_line(p1, p2, i, 0 /* verbose_level - 1*/);
		//cout << "(" << p1 << "," << p2 << ") : ";

		unrank_point(v1, 1, p1, 0);
		unrank_point(v2, 1, p2, 0);

		int_vec_copy(v1, L, d);
		int_vec_copy(v2, L + d, d);

		ost << "\\left[" << endl;
		Li.print_integer_matrix_tex(ost, L, 2, d);
		ost << "\\right]" << endl;

		a = evaluate_bilinear_form(v1, v2, 1);
		if (a) {
			cout << "not orthogonal" << endl;
			exit(1);
		}

#if 0
		cout << " & ";
		j = O.rank_line(p1, p2, 0 /*verbose_level - 1*/);
		if (i != j) {
			cout << "error: i != j" << endl;
			exit(1);
		}
#endif

#if 1

		points_on_line(p1, p2, Line, 0 /*verbose_level - 1*/);
		Sorting.lint_vec_heapsort(Line, q + 1);

		Li.lint_set_print_masked_tex(ost, Line, q + 1, "P_{", "}");
		ost << "$\\\\" << endl;
#if 0
		for (r1 = 0; r1 <= q; r1++) {
			for (r2 = 0; r2 <= q; r2++) {
				if (r1 == r2)
					continue;
				//p3 = p1;
				//p4 = p2;
				p3 = O.line1[r1];
				p4 = O.line1[r2];
				cout << p3 << "," << p4 << " : ";
				j = O.rank_line(p3, p4, verbose_level - 1);
				cout << " : " << j << endl;
				if (i != j) {
					cout << "error: i != j" << endl;
					exit(1);
				}
			}
		}
		cout << endl;
#endif
#endif
	}
	FREE_lint(Line);
	FREE_int(L);

}
void orthogonal::list_all_points_vs_points(int verbose_level)
{
	int t1, t2;

	for (t1 = 1; t1 <= nb_point_classes; t1++) {
		for (t2 = 1; t2 <= nb_point_classes; t2++) {
			list_points_vs_points(t1, t2, verbose_level);
		}
	}
}

void orthogonal::list_points_vs_points(int t1, int t2, int verbose_level)
{
	int i, j, rk1, rk2, u, cnt;

	cout << "lines between points of type P" << t1
			<< " and points of type P" << t2 << endl;
	for (i = 0; i < P[t1 - 1]; i++) {
		rk1 = type_and_index_to_point_rk(t1, i, verbose_level);
		cout << i << " : " << rk1 << " : ";
		unrank_point(v1, 1, rk1, verbose_level - 1);
		int_vec_print(cout, v1, n);
		cout << endl;
		cout << "is incident with:" << endl;

		cnt = 0;

		for (j = 0; j < P[t2 - 1]; j++) {
			rk2 = type_and_index_to_point_rk(t2, j, verbose_level);
			unrank_point(v2, 1, rk2, verbose_level - 1);

			//cout << "testing: " << j << " : " << rk2 << " : ";
			//int_vec_print(cout, v2, n);
			//cout << endl;

			u = evaluate_bilinear_form(v1, v2, 1);
			if (u == 0 && rk2 != rk1) {
				//cout << "yes" << endl;
				if (test_if_minimal_on_line(v2, v1, v3)) {
					cout << cnt << " : " << j << " : " << rk2 << " : ";
					int_vec_print(cout, v2, n);
					cout << endl;
					cnt++;
				}
			}
		}
		cout << endl;
	}

}


void orthogonal::print_schemes()
{
	int i, j;


	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << setw(7) << L[j];
	}
	cout << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << setw(7) << A[i * nb_line_classes + j];
		}
		cout << endl;
	}
	cout << endl;
	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << setw(7) << L[j];
	}
	cout << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << setw(7) << B[i * nb_line_classes + j];
		}
		cout << endl;
	}
	cout << endl;

}

void orthogonal::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::report" << endl;
	}

	report_schemes(ost);

	report_points(ost, verbose_level);

	//report_points_by_type(ost, verbose_level);

	report_lines(ost, verbose_level);

	if (f_v) {
		cout << "orthogonal::report done" << endl;
	}
}

void orthogonal::report_schemes(std::ostream &ost)
{
	int i, j;
	int f, l;
	int nb_rows = nb_points;
	int nb_cols = nb_lines;
	partitionstack *Stack;
	int *set;
	int *row_classes;
	int *col_classes;
	int *row_scheme;
	int *col_scheme;

	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(nb_rows + nb_cols, 0 /* verbose_level */);
	Stack->subset_continguous(nb_rows, nb_cols);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();
#if 0
	row_classes = NEW_int(nb_point_classes);
	for (i = 0; i < nb_point_classes; i++) {
		row_classes[i] = P[i];
	}
	col_classes = NEW_int(nb_line_classes);
	for (i = 0; i < nb_line_classes; i++) {
		col_classes[i] = L[i];
	}
#endif
	row_scheme = NEW_int(nb_point_classes * nb_line_classes);
	for (i = 0; i < nb_point_classes * nb_line_classes; i++) {
		row_scheme[i] = A[i];
	}
	col_scheme = NEW_int(nb_point_classes * nb_line_classes);
	for (i = 0; i < nb_point_classes * nb_line_classes; i++) {
		col_scheme[i] = B[i];
	}
	set = NEW_int(nb_rows);
	f = 0;
	for (i = 0; i < nb_point_classes - 1; i++) {
		l = P[i];
		if (l == 0) {
			continue;
		}
		for (j = 0; j < l; j++) {
			set[j] = f + j;
		}
		Stack->split_cell_front_or_back(set, l, TRUE /* f_front */, 0 /*verbose_level*/);
		f += l;
	}
	FREE_int(set);
	set = NEW_int(nb_lines);
	f = nb_points;
	for (i = 0; i < nb_line_classes - 1; i++) {
		l = L[i];
		if (l == 0) {
			continue;
		}
		for (j = 0; j < l; j++) {
			set[j] = f + j;
		}
		Stack->split_cell_front_or_back(set, l, TRUE /* f_front */, 0 /*verbose_level*/);
		f += l;
	}
	FREE_int(set);


	int nb_row_classes;
	int nb_col_classes;

	row_classes = NEW_int(Stack->ht);
	col_classes = NEW_int(Stack->ht);

	Stack->get_row_and_col_classes(
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 0 /* verbose_level*/);

	Stack->print_row_tactical_decomposition_scheme_tex(ost, TRUE /*f_enter_math*/,
			row_classes, nb_row_classes,
			col_classes, nb_col_classes,
			row_scheme, FALSE /* f_print_subscripts */);

	Stack->print_column_tactical_decomposition_scheme_tex(ost, TRUE /*f_enter_math*/,
			row_classes, nb_row_classes,
			col_classes, nb_col_classes,
			col_scheme, FALSE /* f_print_subscripts */);

	FREE_int(row_classes);
	FREE_int(col_classes);
	FREE_int(row_scheme);
	FREE_int(col_scheme);

#if 0
	ost << "$$" << endl;
	ost << "\\begin{array}{r||*{" << nb_line_classes << "}{r}}" << endl;
	ost << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		ost << " & " << setw(7) << L[j];
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_point_classes; i++) {
		ost << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			ost << " & " << setw(7) << A[i * nb_line_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{r||*{" << nb_line_classes << "}{r}}" << endl;
	ost << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		ost << " & " << setw(7) << L[j];
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_point_classes; i++) {
		ost << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			ost << " & " << setw(7) << B[i * nb_line_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
#endif

	FREE_OBJECT(Stack);
}


void orthogonal::create_latex_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "orthogonal::create_latex_report" << endl;
	}

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "O_%d_%d_%d.tex", epsilon, n, F->q);
		fname.assign(str);
		snprintf(title, 1000, "Orthogonal Space  ${\\rm O}(%d,%d,%d)$", epsilon, n, F->q);
		//strcpy(author, "");
		author[0] = 0;


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
				cout << "orthogonal::create_latex_report before P->report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "orthogonal::create_latex_report after P->report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "orthogonal::create_latex_report done" << endl;
	}
}


}}



