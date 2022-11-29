/*
 * orthogonal_io.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {


void orthogonal::list_points_by_type(int verbose_level)
{
	int t;

	for (t = 1; t <= Hyperbolic_pair->nb_point_classes; t++) {
		list_points_of_given_type(t, verbose_level);
	}
}

void orthogonal::report_points_by_type(std::ostream &ost, int verbose_level)
{
	int t;

	for (t = 1; t <= Hyperbolic_pair->nb_point_classes; t++) {
		report_points_of_given_type(ost, t, verbose_level);
	}
}

void orthogonal::list_points_of_given_type(int t, int verbose_level)
{
	long int i, j, rk, u;

	cout << "points of type P" << t << ":" << endl;
	for (i = 0; i < Hyperbolic_pair->P[t - 1]; i++) {
		rk = Hyperbolic_pair->type_and_index_to_point_rk(t, i, verbose_level);
		cout << i << " : " << rk << " : ";
		Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, rk, verbose_level - 1);
		Int_vec_print(cout, Hyperbolic_pair->v1, n);
		Hyperbolic_pair->point_rk_to_type_and_index(rk, u, j, verbose_level);
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
	for (i = 0; i < Hyperbolic_pair->P[t - 1]; i++) {
		rk = Hyperbolic_pair->type_and_index_to_point_rk(t, i, verbose_level);
		ost << i << " : " << rk << " : ";
		Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, rk, verbose_level - 1);
		Int_vec_print(ost, Hyperbolic_pair->v1, n);
		ost << "\\\\" << endl;
		Hyperbolic_pair->point_rk_to_type_and_index(rk, u, j, verbose_level);
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

	ost << "The number of points is " << Hyperbolic_pair->nb_points << "\\\\" << endl;
	if (Hyperbolic_pair->nb_points < 3000) {

		long int *Pts;

		Pts = NEW_lint(Hyperbolic_pair->nb_points);
		for (rk = 0; rk < Hyperbolic_pair->nb_points; rk++) {
			Pts[rk] = rk;
		}
		report_given_point_set(ost, Pts, Hyperbolic_pair->nb_points, verbose_level);

#if 0
		ost << "points:\\\\" << endl;
		for (rk = 0; rk < nb_points; rk++) {
			unrank_point(v1, 1, rk, 0 /*verbose_level*/);
			ost << "$P_{" << rk << "} = ";
			Int_vec_print(ost, v1, n);
			ost << "$\\\\" << endl;
		}
#endif

	}
	else {
		ost << "Too many points to print.\\\\" << endl;
	}
	//ost << endl;
}

void orthogonal::report_given_point_set(std::ostream &ost,
		long int *Pts, int nb_pts, int verbose_level)
{
	long int rk;
	int i;

	ost << "A set of points of size " << nb_pts << "\\\\" << endl;
	ost << "The Points:\\\\" << endl;
	for (i = 0; i < nb_pts; i++) {
		rk = Pts[i];

		Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, rk, 0 /*verbose_level*/);
		ost << i << " : $P_{" << rk << "} = ";
		Int_vec_print(ost, Hyperbolic_pair->v1, n);
		ost << "$\\\\" << endl;
	}
	//ost << endl;
}

void orthogonal::report_lines(std::ostream &ost, int verbose_level)
{
	int len;
	int i, a, d = n;
	long int p1, p2;
	orbiter_kernel_system::latex_interface Li;
	data_structures::sorting Sorting;

	ost << "The number of lines is " << Hyperbolic_pair->nb_lines << "\\\\" << endl;

	if (Hyperbolic_pair->nb_lines < 1000) {

		len = Hyperbolic_pair->nb_lines; // O.L[0] + O.L[1] + O.L[2];


		long int *Line;
		int *L;

		Line = NEW_lint(q + 1);
		L = NEW_int(2 * d);

		for (i = 0; i < len; i++) {
			ost << "$L_{" << i << "} = ";
			Hyperbolic_pair->unrank_line(p1, p2, i, 0 /* verbose_level - 1*/);
			//cout << "(" << p1 << "," << p2 << ") : ";

			Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, p1, 0);
			Hyperbolic_pair->unrank_point(Hyperbolic_pair->v2, 1, p2, 0);

			Int_vec_copy(Hyperbolic_pair->v1, L, d);
			Int_vec_copy(Hyperbolic_pair->v2, L + d, d);

			ost << "\\left[" << endl;
			Li.print_integer_matrix_tex(ost, L, 2, d);
			ost << "\\right]" << endl;

			a = evaluate_bilinear_form(Hyperbolic_pair->v1, Hyperbolic_pair->v2, 1);
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
	else {
		//ost << "Too many lines to print. \\\\" << endl;
	}
}


void orthogonal::report_given_line_set(std::ostream &ost,
		long int *Lines, int nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::report_given_line_set" << endl;
		cout << "orthogonal::report_given_line_set Lines=";
		Lint_vec_print(cout, Lines, nb_lines);
		cout << endl;
	}
	int i, a, d = n;
	long int p1, p2, rk;
	orbiter_kernel_system::latex_interface Li;
	data_structures::sorting Sorting;

	ost << "A set of lines of size " << nb_lines << "\\\\" << endl;
	ost << "The Lines:\\\\" << endl;

	long int *Points_on_line;
	int *L;

	Points_on_line = NEW_lint(q + 1);
	L = NEW_int(2 * d);

	for (i = 0; i < nb_lines; i++) {

		rk = Lines[i];

		if (f_v) {
			cout << "orthogonal::report_given_line_set i=" << i << " / " << nb_lines << " rk=" << rk << endl;
		}

		ost << i << " : $L_{" << rk << "} = ";
		Hyperbolic_pair->unrank_line(p1, p2, rk, 0 /*verbose_level - 1*/);
		//cout << "(" << p1 << "," << p2 << ") : ";

		Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, p1, 0);
		Hyperbolic_pair->unrank_point(Hyperbolic_pair->v2, 1, p2, 0);

		Int_vec_copy(Hyperbolic_pair->v1, L, d);
		Int_vec_copy(Hyperbolic_pair->v2, L + d, d);

		ost << "\\left[" << endl;
		Li.print_integer_matrix_tex(ost, L, 2, d);
		ost << "\\right]" << endl;

		a = evaluate_bilinear_form(Hyperbolic_pair->v1, Hyperbolic_pair->v2, 1);
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

		points_on_line(p1, p2, Points_on_line, 0 /*verbose_level - 1*/);
		Sorting.lint_vec_heapsort(Points_on_line, q + 1);

		Li.lint_set_print_masked_tex(ost, Points_on_line, q + 1, "P_{", "}");
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

	cout << "before FREE_lint(Points_on_line)" << endl;
	FREE_lint(Points_on_line);
	cout << "before FREE_lint(L)" << endl;
	FREE_int(L);
	cout << "after FREE_lint(L)" << endl;
	if (f_v) {
		cout << "orthogonal::report_given_line_set done" << endl;
	}
}



void orthogonal::list_all_points_vs_points(int verbose_level)
{
	int t1, t2;

	for (t1 = 1; t1 <= Hyperbolic_pair->nb_point_classes; t1++) {
		for (t2 = 1; t2 <= Hyperbolic_pair->nb_point_classes; t2++) {
			list_points_vs_points(t1, t2, verbose_level);
		}
	}
}

void orthogonal::list_points_vs_points(int t1, int t2, int verbose_level)
{
	int i, j, rk1, rk2, u, cnt;

	cout << "lines between points of type P" << t1
			<< " and points of type P" << t2 << endl;
	for (i = 0; i < Hyperbolic_pair->P[t1 - 1]; i++) {
		rk1 = Hyperbolic_pair->type_and_index_to_point_rk(t1, i, verbose_level);
		cout << i << " : " << rk1 << " : ";
		Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, rk1, verbose_level - 1);
		Int_vec_print(cout, Hyperbolic_pair->v1, n);
		cout << endl;
		cout << "is incident with:" << endl;

		cnt = 0;

		for (j = 0; j < Hyperbolic_pair->P[t2 - 1]; j++) {
			rk2 = Hyperbolic_pair->type_and_index_to_point_rk(t2, j, verbose_level);
			Hyperbolic_pair->unrank_point(Hyperbolic_pair->v2, 1, rk2, verbose_level - 1);

			//cout << "testing: " << j << " : " << rk2 << " : ";
			//int_vec_print(cout, v2, n);
			//cout << endl;

			u = evaluate_bilinear_form(Hyperbolic_pair->v1, Hyperbolic_pair->v2, 1);
			if (u == 0 && rk2 != rk1) {
				//cout << "yes" << endl;
				if (test_if_minimal_on_line(Hyperbolic_pair->v2, Hyperbolic_pair->v1, Hyperbolic_pair->v3)) {
					cout << cnt << " : " << j << " : " << rk2 << " : ";
					Int_vec_print(cout, Hyperbolic_pair->v2, n);
					cout << endl;
					cnt++;
				}
			}
		}
		cout << endl;
	}

}



void orthogonal::report_quadratic_form(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::report_quadratic_form" << endl;
	}

	ost << "The quadratic form is: " << endl;
	ost << "$$" << endl;
	Poly->print_equation_tex(ost, the_quadratic_form);
	ost << " = 0";
	ost << "$$" << endl;

	if (f_v) {
		cout << "orthogonal::report_quadratic_form done" << endl;
	}

}


void orthogonal::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::report" << endl;
	}

	report_quadratic_form(ost, verbose_level);


	if (f_v) {
		cout << "orthogonal::report before report_schemes_easy" << endl;
	}

	report_schemes_easy(ost);

	if (f_v) {
		cout << "orthogonal::report after report_schemes_easy" << endl;
	}

#if 0
	if (f_v) {
		cout << "orthogonal::report before report_schemes" << endl;
	}

	report_schemes(ost, verbose_level);

	if (f_v) {
		cout << "orthogonal::report after report_schemes" << endl;
	}
#endif

	if (f_v) {
		cout << "orthogonal::report before report_points" << endl;
	}

	report_points(ost, verbose_level);

	if (f_v) {
		cout << "orthogonal::report after report_points" << endl;
	}


	//report_points_by_type(ost, verbose_level);

	if (f_v) {
		cout << "orthogonal::report before report_lines" << endl;
	}


	report_lines(ost, verbose_level);

	if (f_v) {
		cout << "orthogonal::report after report_lines" << endl;
	}


	if (f_v) {
		cout << "orthogonal::report done" << endl;
	}
}

void orthogonal::report_schemes(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int f, l;
	int nb_rows = 0;
	int nb_cols = 0;
	data_structures::partitionstack *Stack;
	int *set;
	int *row_classes;
	int *col_classes;
	int *row_scheme;
	int *col_scheme;

	if (f_v) {
		cout << "orthogonal::report_schemes" << endl;
	}


	if (Hyperbolic_pair->nb_points < 10000 && Hyperbolic_pair->nb_lines < 1000) {
		Stack = NEW_OBJECT(data_structures::partitionstack);


		if (f_v) {
			cout << "orthogonal::report_schemes nb_points=" << Hyperbolic_pair->nb_points << endl;
			cout << "orthogonal::report_schemes nb_lines=" << Hyperbolic_pair->nb_lines << endl;
		}

		if (f_v) {
			cout << "orthogonal::report_schemes before Stack->allocate_with_two_classes" << endl;
		}

		Stack->allocate_with_two_classes(Hyperbolic_pair->nb_points + Hyperbolic_pair->nb_lines, Hyperbolic_pair->nb_points, Hyperbolic_pair->nb_lines, 0 /* verbose_level */);

		if (f_v) {
			cout << "orthogonal::report_schemes after Stack->allocate_with_two_classes" << endl;
		}
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

		int *original_row_class;
		int *original_col_class;

		original_row_class = NEW_int(Hyperbolic_pair->nb_point_classes);
		original_col_class = NEW_int(Hyperbolic_pair->nb_line_classes);
		nb_rows = 0;
		for (i = 0; i < Hyperbolic_pair->nb_point_classes; i++) {
			if (Hyperbolic_pair->P[i]) {
				original_row_class[nb_rows] = i;
				nb_rows++;
			}
		}
		nb_cols = 0;
		for (i = 0; i < Hyperbolic_pair->nb_line_classes; i++) {
			if (Hyperbolic_pair->L[i]) {
				original_col_class[nb_cols] = i;
				nb_cols++;
			}
		}
		row_scheme = NEW_int(nb_rows * nb_cols);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				row_scheme[i * nb_cols + j] = Hyperbolic_pair->A[original_row_class[i] * Hyperbolic_pair->nb_line_classes + original_col_class[j]];
			}
		}
		col_scheme = NEW_int(nb_rows * nb_cols);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				col_scheme[i * nb_cols + j] = Hyperbolic_pair->B[original_row_class[i] * Hyperbolic_pair->nb_line_classes + original_col_class[j]];
			}
		}
		set = NEW_int(Hyperbolic_pair->nb_points);
		f = 0;
		for (i = 0; i < nb_rows - 1; i++) {
			l = Hyperbolic_pair->P[original_row_class[i]];
			if (l == 0) {
				cout << "l == 0" << endl;
				exit(1);
			}
			for (j = 0; j < l; j++) {
				set[j] = f + j;
			}
			if (f_v) {
				cout << "orthogonal::report_schemes before Stack->split_cell_front_or_back for points" << endl;
			}
			Stack->split_cell_front_or_back(set, l, TRUE /* f_front */, 0 /*verbose_level*/);
			f += l;
		}
		FREE_int(set);


		set = NEW_int(Hyperbolic_pair->nb_lines);
		f = Hyperbolic_pair->nb_points;
		for (i = 0; i < nb_cols - 1; i++) {
			l = Hyperbolic_pair->L[original_col_class[i]];
			if (l == 0) {
				cout << "l == 0" << endl;
				exit(1);
			}
			for (j = 0; j < l; j++) {
				set[j] = f + j;
			}
			if (f_v) {
				cout << "orthogonal::report_schemes before Stack->split_cell_front_or_back for lines" << endl;
			}
			Stack->split_cell_front_or_back(set, l, TRUE /* f_front */, 0 /*verbose_level*/);
			f += l;
		}
		FREE_int(set);


		int nb_row_classes;
		int nb_col_classes;

		row_classes = NEW_int(Stack->ht);
		col_classes = NEW_int(Stack->ht);

		if (f_v) {
			cout << "orthogonal::report_schemes before Stack->get_row_and_col_classes" << endl;
		}
		Stack->get_row_and_col_classes(
			row_classes, nb_row_classes,
			col_classes, nb_col_classes, 0 /* verbose_level*/);

		if (f_v) {
			cout << "orthogonal::report_schemes before Stack->print_row_tactical_decomposition_scheme_tex" << endl;
		}
		Stack->print_row_tactical_decomposition_scheme_tex(ost, TRUE /*f_enter_math*/,
				row_classes, nb_row_classes,
				col_classes, nb_col_classes,
				row_scheme, FALSE /* f_print_subscripts */);

		if (f_v) {
			cout << "orthogonal::report_schemes before Stack->print_column_tactical_decomposition_scheme_tex" << endl;
		}
		Stack->print_column_tactical_decomposition_scheme_tex(ost, TRUE /*f_enter_math*/,
				row_classes, nb_row_classes,
				col_classes, nb_col_classes,
				col_scheme, FALSE /* f_print_subscripts */);

		FREE_int(row_classes);
		FREE_int(col_classes);
		FREE_int(row_scheme);
		FREE_int(col_scheme);

		FREE_int(original_row_class);
		FREE_int(original_col_class);


		FREE_OBJECT(Stack);
	}
	if (f_v) {
		cout << "orthogonal::report_schemes done" << endl;
	}
}


void orthogonal::report_schemes_easy(std::ostream &ost)
{
	orbiter_kernel_system::latex_interface Li;

	Li.print_row_tactical_decomposition_scheme_tex(
			ost, TRUE /* f_enter_math_mode */,
			Hyperbolic_pair->P, Hyperbolic_pair->nb_point_classes,
			Hyperbolic_pair->L, Hyperbolic_pair->nb_line_classes,
			Hyperbolic_pair->A);

	Li.print_column_tactical_decomposition_scheme_tex(
			ost, TRUE /* f_enter_math_mode */,
			Hyperbolic_pair->P, Hyperbolic_pair->nb_point_classes,
			Hyperbolic_pair->L, Hyperbolic_pair->nb_line_classes,
			Hyperbolic_pair->B);

}

void orthogonal::create_latex_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "orthogonal::create_latex_report" << endl;
	}

	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "O_%d_%d_%d.tex", epsilon, n, F->q);
		fname.assign(str);
		snprintf(str, 1000, "Orthogonal Space  ${\\rm O}(%d,%d,%d)$", epsilon, n, F->q);
		title.assign(str);


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
				cout << "orthogonal::create_latex_report before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "orthogonal::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "orthogonal::create_latex_report done" << endl;
	}
}

void orthogonal::export_incidence_matrix_to_csv(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::export_incidence_matrix_to_csv" << endl;
	}


	int N_points = Hyperbolic_pair->nb_points;
	int N_lines = Hyperbolic_pair->nb_lines;

	long int *line;

	long int i, line_rk, h;
	int *T;
	orbiter_kernel_system::file_io Fio;

	line = NEW_lint(q + 1);
	T = NEW_int(N_points * N_lines);
	Int_vec_zero(T, N_points * N_lines);

	for (line_rk = 0; line_rk < N_lines; line_rk++) {


		points_on_line_by_line_rank(line_rk,
				line, verbose_level);

		for (h = 0; h < q + 1; h++) {
			i = line[h];
			T[i * N_lines + line_rk] = 1;
		}
	}
	string fname;

	make_fname_incidence_matrix_csv(fname);

	Fio.int_matrix_write_csv(fname, T, N_points, N_lines);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(T);
	FREE_lint(line);

	if (f_v) {
		cout << "orthogonal::export_incidence_matrix_to_csv done" << endl;
	}
}

void orthogonal::make_fname_incidence_matrix_csv(std::string &fname)
{
	fname.assign(label_txt);
	fname.append("_incidence_matrix.csv");
}



}}}



