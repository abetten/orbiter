/*
 * projective_space_reporting.cpp
 *
 *  Created on: Dec 3, 2022
 *      Author: betten
 */





#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {


projective_space_reporting::projective_space_reporting()
{
	P = NULL;
}

projective_space_reporting::~projective_space_reporting()
{
	P = NULL;
}

void projective_space_reporting::init(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::init" << endl;
	}
	projective_space_reporting::P = P;
	if (f_v) {
		cout << "projective_space_reporting::init done" << endl;
	}
}

void projective_space_reporting::create_latex_report(
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_reporting::create_latex_report" << endl;
	}

	{
		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "PG_%d_%d.tex", P->n, P->F->q);
		fname.assign(str);
		snprintf(str, 1000, "Cheat Sheet PG($%d,%d$)", P->n, P->F->q);
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
				cout << "projective_space_reporting::create_latex_report "
						"before P->report" << endl;
			}
			report(ost, O, verbose_level);
			if (f_v) {
				cout << "projective_space_reporting::create_latex_report "
						"after P->report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "projective_space_reporting::create_latex_report done" << endl;
	}
}

void projective_space_reporting::report_summary(std::ostream &ost)
{
	//ost << "\\parindent=0pt" << endl;
	ost << "$q = " << P->F->q << "$\\\\" << endl;
	ost << "$p = " << P->F->p << "$\\\\" << endl;
	ost << "$e = " << P->F->e << "$\\\\" << endl;
	ost << "$n = " << P->n << "$\\\\" << endl;
	ost << "Number of points = " << P->N_points << "\\\\" << endl;
	ost << "Number of lines = " << P->N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << P->r << "\\\\" << endl;
	ost << "Number of points on a line = " << P->k << "\\\\" << endl;
}

void projective_space_reporting::report(std::ostream &ost,
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::report" << endl;
	}

	ost << "\\subsection*{The projective space ${\\rm \\PG}(" << P->n << "," << P->F->q << ")$}" << endl;
	ost << "\\noindent" << endl;
	ost << "\\arraycolsep=2pt" << endl;
	ost << "\\parindent=0pt" << endl;
	ost << "$q = " << P->F->q << "$\\\\" << endl;
	ost << "$p = " << P->F->p << "$\\\\" << endl;
	ost << "$e = " << P->F->e << "$\\\\" << endl;
	ost << "$n = " << P->n << "$\\\\" << endl;
	ost << "Number of points = " << P->N_points << "\\\\" << endl;
	ost << "Number of lines = " << P->N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << P->r << "\\\\" << endl;
	ost << "Number of points on a line = " << P->k << "\\\\" << endl;

	//ost<< "\\clearpage" << endl << endl;
	//ost << "\\section{The Finite Field with $" << q << "$ Elements}" << endl;
	//F->cheat_sheet(ost, verbose_level);

#if 0
	if (f_v) {
		cout << "projective_space_reporting::report before incidence_matrix_save_csv" << endl;
	}
	incidence_matrix_save_csv();
	if (f_v) {
		cout << "projective_space_reporting::report after incidence_matrix_save_csv" << endl;
	}
#endif

	if (P->n == 2) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\subsection*{The plane}" << endl;


		if (P->N_points < 1000) {
			string fname_base;
			char str[1000];
			long int *set;
			int i;

			set = NEW_lint(P->N_points);
			for (i = 0; i < P->N_points; i++) {
				set[i] = i;
			}
			snprintf(str, sizeof(str), "plane_of_order_%d", P->q);
			fname_base.assign(str);

			graphics::plot_tools Pt;

			Pt.draw_point_set_in_plane(fname_base,
					O,
					P,
					set, P->N_points,
					TRUE /*f_point_labels*/,
					verbose_level);
			FREE_lint(set);
			ost << "{\\scriptsize" << endl;
			ost << "$$" << endl;
			ost << "\\input " << fname_base << "_draw.tex" << endl;
			ost << "$$" << endl;
			ost << "}%%" << endl;
		}
		else {
			ost << "Too many points to draw. \\\\" << endl;
		}
	}

	//ost << "\\clearpage" << endl << endl;
	ost << "\\subsection*{The points of ${\\rm \\PG}(" << P->n << "," << P->F->q << ")$}" << endl;
	cheat_sheet_points(ost, verbose_level);

	//cheat_sheet_point_table(ost, verbose_level);


#if 0
	//ost << "\\clearpage" << endl << endl;
	cheat_sheet_points_on_lines(ost, verbose_level);

	//ost << "\\clearpage" << endl << endl;
	cheat_sheet_lines_on_points(ost, verbose_level);
#endif

	// report subspaces:
	int k;

	for (k = 1; k < P->n; k++) {
		//ost << "\\clearpage" << endl << endl;
		if (k == 1) {
			ost << "\\subsection*{The lines of ${\\rm \\PG}(" << P->n << "," << P->F->q << ")$}" << endl;
		}
		else if (k == 2) {
			ost << "\\subsection*{The planes of ${\\rm \\PG}(" << P->n << "," << P->F->q << ")$}" << endl;
		}
		else {
			ost << "\\subsection*{The subspaces of dimension " << k << " of ${\\rm \\PG}(" << P->n << "," << P->F->q << ")$}" << endl;
		}
		//ost << "\\section{Subspaces of dimension " << k << "}" << endl;


		if (f_v) {
			cout << "projective_space_reporting::report "
					"before report_subspaces_of_dimension" << endl;
		}
		report_subspaces_of_dimension(ost, k + 1, verbose_level);
		//Grass_stack[k + 1]->cheat_sheet_subspaces(ost, verbose_level);
		if (f_v) {
			cout << "projective_space_reporting::report "
					"after report_subspaces_of_dimension" << endl;
		}
	}


#if 0
	if (n >= 2 && N_lines < 25) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\section*{Line intersections}" << endl;
		cheat_sheet_line_intersection(ost, verbose_level);
	}


	if (n >= 2 && N_points < 25) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\section*{Line through point-pairs}" << endl;
		cheat_sheet_line_through_pairs_of_points(ost, verbose_level);
	}
#endif

	ring_theory::homogeneous_polynomial_domain *Poly1;
	ring_theory::homogeneous_polynomial_domain *Poly2;
	ring_theory::homogeneous_polynomial_domain *Poly3;
	ring_theory::homogeneous_polynomial_domain *Poly4;

	Poly1 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	ost << "\\subsection*{The polynomial rings associated "
			"with ${\\rm \\PG}(" << P->n << "," << P->F->q << ")$}" << endl;
	Poly1->init(P->F,
			P->n + 1 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	Poly2->init(P->F,
			P->n + 1 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	Poly3->init(P->F,
			P->n + 1 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	Poly4->init(P->F,
			P->n + 1 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);

	Poly1->print_monomial_ordering(ost);
	Poly2->print_monomial_ordering(ost);
	Poly3->print_monomial_ordering(ost);
	Poly4->print_monomial_ordering(ost);

	FREE_OBJECT(Poly1);
	FREE_OBJECT(Poly2);
	FREE_OBJECT(Poly3);
	FREE_OBJECT(Poly4);

	if (f_v) {
		cout << "projective_space_reporting::report done" << endl;
	}

}

void projective_space_reporting::report_subspaces_of_dimension(std::ostream &ost,
		int vs_dimension, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::report_subspaces_of_dimension" << endl;
	}

	if (f_v) {
		cout << "projective_space_reporting::report "
				"before report_subspaces_of_dimension, "
				"vs_dimension=" << vs_dimension << endl;
	}
	P->Grass_stack[vs_dimension]->cheat_sheet_subspaces(ost, verbose_level);
	if (f_v) {
		cout << "projective_space_reporting::report "
				"after report_subspaces_of_dimension, "
				"vs_dimension=" << vs_dimension << endl;
	}

	if (f_v) {
		cout << "projective_space_reporting::report_subspaces_of_dimension done" << endl;
	}
}

void projective_space_reporting::cheat_sheet_points(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_points" << endl;
	}
	int i, d;
	int *v;
	string symbol_for_print;

	d = P->n + 1;

	symbol_for_print.assign("\\alpha");
	v = NEW_int(d);

	f << "PG$(" << P->n << ", " << P->q << ")$ has "
			<< P->N_points << " points:\\\\" << endl;

	if (P->N_points < 1000) {
		if (P->F->e == 1) {
			f << "\\begin{multicols}{4}" << endl;
			for (i = 0; i < P->N_points; i++) {
				P->F->PG_element_unrank_modified(v, 1, d, i);
				f << "$P_{" << i << "}=\\bP";
				Int_vec_print(f, v, d);
				f << "$\\\\" << endl;
			}
			f << "\\end{multicols}" << endl;
		}
		else {
			f << "\\begin{multicols}{2}" << endl;
			for (i = 0; i < P->N_points; i++) {
				P->F->PG_element_unrank_modified(v, 1, d, i);
				f << "$P_{" << i << "}=\\bP";
				Int_vec_print(f, v, d);
				f << "=";
				P->F->int_vec_print_elements_exponential(f, v, d, symbol_for_print);
				f << "$\\\\" << endl;
			}
			f << "\\end{multicols}" << endl;

			f << "\\begin{multicols}{2}" << endl;
			for (i = 0; i < P->N_points; i++) {
				P->F->PG_element_unrank_modified(v, 1, d, i);
				f << "$P_{" << i << "}=\\bP";
				Int_vec_print(f, v, d);
				//f << "=";
				//F->int_vec_print_elements_exponential(f, v, d, symbol_for_print);
				f << "$\\\\" << endl;
			}
			f << "\\end{multicols}" << endl;

		}
	}
	else {
		f << "Too many to list. \\\\" << endl;

	}

	if (P->F->has_quadratic_subfield()) {
		int cnt = 0;

		f << "Baer subgeometry:\\\\" << endl;
		if (P->N_points < 1000) {
			f << "\\begin{multicols}{4}" << endl;
			int j;
			for (i = 0; i < P->N_points; i++) {
				P->F->PG_element_unrank_modified(v, 1, d, i);
				P->F->PG_element_normalize_from_front(v, 1, d);
				for (j = 0; j < d; j++) {
					if (!P->F->belongs_to_quadratic_subfield(v[j])) {
						break;
					}
				}
				if (j == d) {
					cnt++;
					f << "$P_{" << i << "}=\\bP";
					Int_vec_print(f, v, d);
					f << "$\\\\" << endl;
				}
			}
			f << "\\end{multicols}" << endl;
		}
		else {
			f << "Too many to list. \\\\" << endl;
			int j;
			for (i = 0; i < P->N_points; i++) {
				P->F->PG_element_unrank_modified(v, 1, d, i);
				P->F->PG_element_normalize_from_front(v, 1, d);
				for (j = 0; j < d; j++) {
					if (!P->F->belongs_to_quadratic_subfield(v[j])) {
						break;
					}
				}
				if (j == d) {
					cnt++;
				}
			}
		}
		f << "There are " << cnt << " elements in the Baer subgeometry.\\\\" << endl;

	}
	//f << "\\clearpage" << endl << endl;

	f << "Normalized from the left:\\\\" << endl;
	if (P->N_points < 1000) {
		f << "\\begin{multicols}{4}" << endl;
		for (i = 0; i < P->N_points; i++) {
			P->F->PG_element_unrank_modified(v, 1, d, i);
			P->F->PG_element_normalize_from_front(v, 1, d);
			f << "$P_{" << i << "}=\\bP";
			Int_vec_print(f, v, d);
			f << "$\\\\" << endl;
			}
		f << "\\end{multicols}" << endl;
		f << "\\clearpage" << endl << endl;
	}
	else {
		f << "Too many to list. \\\\" << endl;

	}


	cheat_polarity(f, verbose_level);

	FREE_int(v);
	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_points done" << endl;
	}
}

void projective_space_reporting::cheat_polarity(std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_polarity" << endl;
	}

	f << "Standard polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;

	if (P->Standard_polarity == NULL) {
		cout << "projective_space_reporting::cheat_polarity NULL pointer" << endl;
		return;
	}
	P->Standard_polarity->report(f);

	f << "Reversal polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;

	if (P->Reversal_polarity == NULL) {
		cout << "projective_space_reporting::cheat_polarity NULL pointer" << endl;
		return;
	}
	P->Reversal_polarity->report(f);

	if (f_v) {
		cout << "projective_space_reporting::cheat_polarity done" << endl;
	}
}

void projective_space_reporting::cheat_sheet_point_table(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_point_table" << endl;
	}
	int I, i, j, a, d, nb_rows, nb_cols = 5;
	int nb_rows_per_page = 40, nb_tables;
	int nb_r;
	int *v;

	d = P->n + 1;

	v = NEW_int(d);

	f << "PG$(" << P->n << ", " << P->q << ")$ has " << P->N_points
			<< " points:\\\\" << endl;

	nb_rows = (P->N_points + nb_cols - 1) / nb_cols;
	nb_tables = (nb_rows + nb_rows_per_page - 1) / nb_rows_per_page;

	for (I = 0; I < nb_tables; I++) {
		f << "$$" << endl;
		f << "\\begin{array}{r|*{" << nb_cols << "}{r}}" << endl;
		f << "P_{" << nb_cols << "\\cdot i+j}";
		for (j = 0; j < nb_cols; j++) {
			f << " & " << j;
		}
		f << "\\\\" << endl;
		f << "\\hline" << endl;

		if (I == nb_tables - 1) {
			nb_r = nb_rows - I * nb_rows_per_page;
		}
		else {
			nb_r = nb_rows_per_page;
		}

		for (i = 0; i < nb_r; i++) {
			f << (I * nb_rows_per_page + i) * nb_cols;
			for (j = 0; j < nb_cols; j++) {
				a = (I * nb_rows_per_page + i) * nb_cols + j;
				f << " & ";
				if (a < P->N_points) {
					P->F->PG_element_unrank_modified(v, 1, d, a);
					Int_vec_print(f, v, d);
					}
				}
			f << "\\\\" << endl;
			}
		f << "\\end{array}" << endl;
		f << "$$" << endl;
		}

	FREE_int(v);
	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_point_table done" << endl;
	}
}


void projective_space_reporting::cheat_sheet_points_on_lines(
	std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_points_on_lines" << endl;
	}
	orbiter_kernel_system::latex_interface L;


	f << "PG$(" << P->n << ", " << P->q << ")$ has " << P->N_lines
			<< " lines, each with " << P->k << " points:\\\\" << endl;
	if (P->Implementation->Lines == NULL) {
		f << "Don't have Lines table\\\\" << endl;
	}
	else {
		int *row_labels;
		int *col_labels;
		int i, nb;

		row_labels = NEW_int(P->N_lines);
		col_labels = NEW_int(P->k);
		for (i = 0; i < P->N_lines; i++) {
			row_labels[i] = i;
		}
		for (i = 0; i < P->k; i++) {
			col_labels[i] = i;
		}
		//int_matrix_print_tex(f, Lines, N_lines, k);
		for (i = 0; i < P->N_lines; i += 40) {
			nb = MINIMUM(P->N_lines - i, 40);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;
			L.print_integer_matrix_with_labels(f,
					P->Implementation->Lines + i * P->k, nb, P->k, row_labels + i,
					col_labels, TRUE /* f_tex */);
			f << "$$" << endl;
		}
		FREE_int(row_labels);
		FREE_int(col_labels);
	}
	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_points_on_lines done" << endl;
	}
}

void projective_space_reporting::cheat_sheet_lines_on_points(
	std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_lines_on_points" << endl;
	}
	orbiter_kernel_system::latex_interface L;

	f << "PG$(" << P->n << ", " << P->q << ")$ has " << P->N_points
			<< " points, each with " << P->r << " lines:\\\\" << endl;
	if (P->Implementation->Lines_on_point == NULL) {
		f << "Don't have Lines\\_on\\_point table\\\\" << endl;
	}
	else {
		int *row_labels;
		int *col_labels;
		int i, nb;

		row_labels = NEW_int(P->N_points);
		col_labels = NEW_int(P->r);
		for (i = 0; i < P->N_points; i++) {
			row_labels[i] = i;
		}
		for (i = 0; i < P->r; i++) {
			col_labels[i] = i;
		}
		for (i = 0; i < P->N_points; i += 40) {
			nb = MINIMUM(P->N_points - i, 40);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;
			L.print_integer_matrix_with_labels(f,
					P->Implementation->Lines_on_point + i * P->r, nb, P->r,
				row_labels + i, col_labels, TRUE /* f_tex */);
			f << "$$" << endl;
		}
		FREE_int(row_labels);
		FREE_int(col_labels);

#if 0
		f << "$$" << endl;
		int_matrix_print_tex(f, Lines_on_point, N_points, r);
		f << "$$" << endl;
#endif
	}
	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_lines_on_points done" << endl;
	}
}



void projective_space_reporting::cheat_sheet_line_intersection(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_line_intersection" << endl;
	}
	int i, j, a;


	f << "intersection of 2 lines:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|*{" << P->N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < P->N_points; j++) {
		f << "& " << j << endl;
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < P->N_points; i++) {
		f << i;
		for (j = 0; j < P->N_points; j++) {
			a = P->Implementation->Line_intersection[i * P->N_lines + j];
			f << " & ";
			if (i != j) {
				f << a;
			}
		}
		f << "\\\\[-3pt]" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	f << "\\clearpage" << endl;

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_line_intersection done" << endl;
	}

}

void projective_space_reporting::cheat_sheet_line_through_pairs_of_points(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_line_through_pairs_of_points" << endl;
	}
	int i, j, a;



	f << "line through 2 points:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|*{" << P->N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < P->N_points; j++) {
		f << "& " << j << endl;
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < P->N_points; i++) {
		f << i;
		for (j = 0; j < P->N_points; j++) {

			a = P->Implementation->Line_through_two_points[i * P->N_points + j];
			f << " & ";
			if (i != j) {
				f << a;
			}
		}
		f << "\\\\[-3pt]" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	f << "\\clearpage" << endl;

	if (f_v) {
		cout << "projective_space_reporting::cheat_sheet_line_through_pairs_of_points done" << endl;
	}

}

void projective_space_reporting::print_set_numerical(
		std::ostream &ost, long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(P->n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		P->unrank_point(v, a);
		ost << setw(3) << i << " : " << setw(5) << a << " : ";
		Int_vec_print(ost, v, P->n + 1);
		ost << "=";
		P->F->PG_element_normalize_from_front(v, 1, P->n + 1);
		Int_vec_print(ost, v, P->n + 1);
		ost << "\\\\" << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_set(long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(P->n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		P->unrank_point(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		P->F->int_vec_print_field_elements(cout, v, P->n + 1);
		cout << "=";
		P->F->PG_element_normalize_from_front(v, 1, P->n + 1);
		P->F->int_vec_print_field_elements(cout, v, P->n + 1);
		cout << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_line_set_numerical(
		long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(2 * (P->n + 1));
	for (i = 0; i < set_size; i++) {
		a = set[i];
		P->unrank_line(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		Int_vec_print(cout, v, 2 * (P->n + 1));
		cout << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_set_of_points(
		std::ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;

	v = NEW_int(P->n + 1);

	for (I = 0; I < (nb_pts + 39) / 40; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & \\mbox{Rank} & \\mbox{Point} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 40; h++) {
			if (I * 40 + h < nb_pts) {
				P->unrank_point(v, Pts[I * 40 + h]);
				ost << I * 40 + h << " & " << Pts[I * 40 + h] << " & ";
				Int_vec_print(ost, v, P->n + 1);
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_all_points()
{
	int *v;
	int i;

	v = NEW_int(P->n + 1);
	cout << "All points in PG(" << P->n << "," << P->q << "):" << endl;
	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v, i);
		cout << setw(3) << i << " : ";
		Int_vec_print(cout, v, P->n + 1);
		cout << endl;
	}
	FREE_int(v);
}



}}}
