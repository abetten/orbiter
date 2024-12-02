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
namespace projective_geometry {


projective_space_reporting::projective_space_reporting()
{
	Record_birth();
	P = NULL;
}

projective_space_reporting::~projective_space_reporting()
{
	Record_death();
	P = NULL;
}

void projective_space_reporting::init(
		projective_space *P, int verbose_level)
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
		other::graphics::layered_graph_draw_options *O,
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



		fname = "PG_" + std::to_string(P->Subspaces->n) + "_" + std::to_string(P->Subspaces->q) + ".tex";
		title = "Cheat Sheet PG($" + std::to_string(P->Subspaces->n) + "," + std::to_string(P->Subspaces->q) + "$)";




		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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
		other::orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "projective_space_reporting::create_latex_report done" << endl;
	}
}

void projective_space_reporting::report_summary(
		std::ostream &ost)
{
	//ost << "\\parindent=0pt" << endl;
	ost << "$q = " << P->Subspaces->F->q << "$\\\\" << endl;
	ost << "$p = " << P->Subspaces->F->p << "$\\\\" << endl;
	ost << "$e = " << P->Subspaces->F->e << "$\\\\" << endl;
	ost << "$n = " << P->Subspaces->n << "$\\\\" << endl;
	ost << "Number of points = " << P->Subspaces->N_points << "\\\\" << endl;
	ost << "Number of lines = " << P->Subspaces->N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << P->Subspaces->r << "\\\\" << endl;
	ost << "Number of points on a line = " << P->Subspaces->k << "\\\\" << endl;
}

void projective_space_reporting::report(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::report" << endl;
	}

	ost << "\\subsection*{The projective space ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
	ost << "\\noindent" << endl;
	ost << "\\arraycolsep=2pt" << endl;
	ost << "\\parindent=0pt" << endl;
	ost << "$q = " << P->Subspaces->F->q << "$\\\\" << endl;
	ost << "$p = " << P->Subspaces->F->p << "$\\\\" << endl;
	ost << "$e = " << P->Subspaces->F->e << "$\\\\" << endl;
	ost << "$n = " << P->Subspaces->n << "$\\\\" << endl;
	ost << "Number of points = " << P->Subspaces->N_points << "\\\\" << endl;
	ost << "Number of lines = " << P->Subspaces->N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << P->Subspaces->r << "\\\\" << endl;
	ost << "Number of points on a line = " << P->Subspaces->k << "\\\\" << endl;

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

	if (P->Subspaces->n == 2) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\subsection*{The plane}" << endl;


		if (f_v) {
			cout << "projective_space_reporting::report "
					"before create_drawing_of_plane" << endl;
		}
		create_drawing_of_plane(ost, Draw_options, verbose_level);
		if (f_v) {
			cout << "projective_space_reporting::report "
					"after create_drawing_of_plane" << endl;
		}


	}

	//ost << "\\clearpage" << endl << endl;
	ost << "\\subsection*{The points of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
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

	for (k = 1; k < P->Subspaces->n; k++) {
		//ost << "\\clearpage" << endl << endl;
		if (k == 1) {
			ost << "\\subsection*{The lines of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
		}
		else if (k == 2) {
			ost << "\\subsection*{The planes of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
		}
		else {
			ost << "\\subsection*{The subspaces of dimension " << k << " of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
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


	if (f_v) {
		cout << "projective_space_reporting::report "
				"before report_polynomial_rings" << endl;
	}
	report_polynomial_rings(
			ost,
			verbose_level);
	if (f_v) {
		cout << "projective_space_reporting::report "
				"after report_polynomial_rings" << endl;
	}


	if (f_v) {
		cout << "projective_space_reporting::report done" << endl;
	}

}

void projective_space_reporting::report_polynomial_rings(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::report_polynomial_rings" << endl;
	}

	algebra::ring_theory::homogeneous_polynomial_domain *Poly1;
	algebra::ring_theory::homogeneous_polynomial_domain *Poly2;
	algebra::ring_theory::homogeneous_polynomial_domain *Poly3;
	algebra::ring_theory::homogeneous_polynomial_domain *Poly4;

	Poly1 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	Poly4 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);

	ost << "\\subsection*{The polynomial rings associated "
			"with ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
	Poly1->init(P->Subspaces->F,
			P->Subspaces->n + 1 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	Poly2->init(P->Subspaces->F,
			P->Subspaces->n + 1 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	Poly3->init(P->Subspaces->F,
			P->Subspaces->n + 1 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	Poly4->init(P->Subspaces->F,
			P->Subspaces->n + 1 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);

	Poly1->print_monomial_ordering_latex(ost);
	Poly2->print_monomial_ordering_latex(ost);
	Poly3->print_monomial_ordering_latex(ost);
	Poly4->print_monomial_ordering_latex(ost);

	FREE_OBJECT(Poly1);
	FREE_OBJECT(Poly2);
	FREE_OBJECT(Poly3);
	FREE_OBJECT(Poly4);

	if (f_v) {
		cout << "projective_space_reporting::report_polynomial_rings done" << endl;
	}
}

void projective_space_reporting::create_drawing_of_plane(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::create_drawing_of_plane" << endl;
	}

	if (P->Subspaces->N_points < 1000) {
		string fname_base;
		long int *set;
		int i;

		set = NEW_lint(P->Subspaces->N_points);
		for (i = 0; i < P->Subspaces->N_points; i++) {
			set[i] = i;
		}
		fname_base = "plane_of_order_" + std::to_string(P->Subspaces->q);

		other::graphics::plot_tools Pt;

		Pt.draw_point_set_in_plane(
				fname_base,
				Draw_options,
				P,
				set, P->Subspaces->N_points,
				true /*f_point_labels*/,
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

	if (f_v) {
		cout << "projective_space_reporting::create_drawing_of_plane done" << endl;
	}
}

void projective_space_reporting::report_subspaces_of_dimension(
		std::ostream &ost,
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
	P->Subspaces->Grass_stack[vs_dimension]->cheat_sheet_subspaces(ost, verbose_level);
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

	d = P->Subspaces->n + 1;

	symbol_for_print.assign("\\alpha");
	v = NEW_int(d);

	f << "PG$(" << P->Subspaces->n << ", " << P->Subspaces->q << ")$ has "
			<< P->Subspaces->N_points << " points:\\\\" << endl;

	if (P->Subspaces->N_points < 1000) {
		if (P->Subspaces->F->e == 1) {
			f << "\\begin{multicols}{4}" << endl;
			for (i = 0; i < P->Subspaces->N_points; i++) {
				P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
						v, 1, d, i);
				f << "$P_{" << i << "}=\\bP";
				Int_vec_print(f, v, d);
				f << "$\\\\" << endl;
			}
			f << "\\end{multicols}" << endl;
		}
		else {
			f << "\\begin{multicols}{2}" << endl;
			for (i = 0; i < P->Subspaces->N_points; i++) {
				P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
						v, 1, d, i);
				f << "$P_{" << i << "}=\\bP";
				Int_vec_print(f, v, d);
				f << "=";
				P->Subspaces->F->Io->int_vec_print_elements_exponential(f, v, d, symbol_for_print);
				f << "$\\\\" << endl;
			}
			f << "\\end{multicols}" << endl;

			f << "\\begin{multicols}{2}" << endl;
			for (i = 0; i < P->Subspaces->N_points; i++) {
				P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
						v, 1, d, i);
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

	if (P->Subspaces->F->has_quadratic_subfield()) {
		int cnt = 0;

		f << "Baer subgeometry:\\\\" << endl;
		if (P->Subspaces->N_points < 1000) {
			f << "\\begin{multicols}{4}" << endl;
			int j;
			for (i = 0; i < P->Subspaces->N_points; i++) {
				P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
						v, 1, d, i);
				P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
						v, 1, d);
				for (j = 0; j < d; j++) {
					if (!P->Subspaces->F->belongs_to_quadratic_subfield(v[j])) {
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
			for (i = 0; i < P->Subspaces->N_points; i++) {
				P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
						v, 1, d, i);
				P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
						v, 1, d);
				for (j = 0; j < d; j++) {
					if (!P->Subspaces->F->belongs_to_quadratic_subfield(v[j])) {
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
	if (P->Subspaces->N_points < 1000) {
		f << "\\begin{multicols}{4}" << endl;
		for (i = 0; i < P->Subspaces->N_points; i++) {
			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
					v, 1, d, i);
			P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
					v, 1, d);
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

void projective_space_reporting::cheat_polarity(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_reporting::cheat_polarity" << endl;
	}

	f << "Standard polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;

	if (P->Subspaces->Standard_polarity == NULL) {
		cout << "projective_space_reporting::cheat_polarity NULL pointer" << endl;
		return;
	}
	P->Subspaces->Standard_polarity->report(f);

	f << "Reversal polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;

	if (P->Subspaces->Reversal_polarity == NULL) {
		cout << "projective_space_reporting::cheat_polarity NULL pointer" << endl;
		return;
	}
	P->Subspaces->Reversal_polarity->report(f);

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

	d = P->Subspaces->n + 1;

	v = NEW_int(d);

	f << "PG$(" << P->Subspaces->n << ", " << P->Subspaces->q << ")$ has " << P->Subspaces->N_points
			<< " points:\\\\" << endl;

	nb_rows = (P->Subspaces->N_points + nb_cols - 1) / nb_cols;
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
				if (a < P->Subspaces->N_points) {
					P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
							v, 1, d, a);
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
	other::l1_interfaces::latex_interface L;


	f << "PG$(" << P->Subspaces->n << ", " << P->Subspaces->q << ")$ has " << P->Subspaces->N_lines
			<< " lines, each with " << P->Subspaces->k << " points:\\\\" << endl;
	if (!P->Subspaces->Implementation->has_lines()) {
		f << "Don't have Lines table\\\\" << endl;
	}
	else {
		int *row_labels;
		int *col_labels;
		int i, j, h, nb;
		int row_block_size = 40;
		int *v;

		v = NEW_int(row_block_size * P->Subspaces->k);
		row_labels = NEW_int(P->Subspaces->N_lines);
		col_labels = NEW_int(P->Subspaces->k);
		for (i = 0; i < P->Subspaces->N_lines; i++) {
			row_labels[i] = i;
		}
		for (i = 0; i < P->Subspaces->k; i++) {
			col_labels[i] = i;
		}
		//int_matrix_print_tex(f, Lines, N_lines, k);
		for (i = 0; i < P->Subspaces->N_lines; i += row_block_size) {
			nb = MINIMUM(P->Subspaces->N_lines - i, row_block_size);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;

			for (h = 0; h < nb; h++) {
				for (j = 0; j < P->Subspaces->k; j++) {
					v[h * P->Subspaces->k + j] = P->Subspaces->Implementation->lines(i + h, j);
				}
			}
			L.print_integer_matrix_with_labels(f,
					v, nb, P->Subspaces->k, row_labels + i,
					col_labels, true /* f_tex */);
			f << "$$" << endl;
		}
		FREE_int(v);
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
	other::l1_interfaces::latex_interface L;

	f << "PG$(" << P->Subspaces->n << ", " << P->Subspaces->q << ")$ has " << P->Subspaces->N_points
			<< " points, each with " << P->Subspaces->r << " lines:\\\\" << endl;
	if (!P->Subspaces->Implementation->has_lines_on_point()) {
		f << "Don't have Lines\\_on\\_point table\\\\" << endl;
	}
	else {
		int *row_labels;
		int *col_labels;
		int i, h, j, nb;
		int block_size = 40;
		int *v;

		v = NEW_int(block_size * P->Subspaces->r);
		row_labels = NEW_int(P->Subspaces->N_points);
		col_labels = NEW_int(P->Subspaces->r);
		for (i = 0; i < P->Subspaces->N_points; i++) {
			row_labels[i] = i;
		}
		for (i = 0; i < P->Subspaces->r; i++) {
			col_labels[i] = i;
		}
		for (i = 0; i < P->Subspaces->N_points; i += block_size) {
			nb = MINIMUM(P->Subspaces->N_points - i, block_size);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;

			for (h = 0; h < nb; h++) {
				for (j = 0; j < P->Subspaces->k; j++) {
					v[h * P->Subspaces->r + j] = P->Subspaces->Implementation->lines_on_point(i + h, j);
				}
			}
			L.print_integer_matrix_with_labels(f,
					v, nb, P->Subspaces->r,
				row_labels + i, col_labels, true /* f_tex */);
			f << "$$" << endl;
		}
		FREE_int(v);
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
	f << "\\begin{array}{|r|*{" << P->Subspaces->N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < P->Subspaces->N_points; j++) {
		f << "& " << j << endl;
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		f << i;
		for (j = 0; j < P->Subspaces->N_points; j++) {
			a = P->Subspaces->Implementation->line_intersection(i, j);
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
	f << "\\begin{array}{|r|*{" << P->Subspaces->N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < P->Subspaces->N_points; j++) {
		f << "& " << j << endl;
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		f << i;
		for (j = 0; j < P->Subspaces->N_points; j++) {

			a = P->Subspaces->Implementation->line_through_two_points(i, j);
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

	v = NEW_int(P->Subspaces->n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		P->unrank_point(v, a);
		ost << setw(3) << i << " : " << setw(5) << a << " : ";
		Int_vec_print(ost, v, P->Subspaces->n + 1);
		ost << "=";
		P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
				v, 1, P->Subspaces->n + 1);
		Int_vec_print(ost, v, P->Subspaces->n + 1);
		ost << "\\\\" << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_set(
		long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(P->Subspaces->n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		P->unrank_point(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		P->Subspaces->F->Io->int_vec_print_field_elements(cout, v, P->Subspaces->n + 1);
		cout << "=";
		P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
				v, 1, P->Subspaces->n + 1);
		P->Subspaces->F->Io->int_vec_print_field_elements(cout, v, P->Subspaces->n + 1);
		cout << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_line_set_numerical(
		long int *set, int set_size)
{
	long int i, a;
	int *v;

	v = NEW_int(2 * (P->Subspaces->n + 1));
	for (i = 0; i < set_size; i++) {
		a = set[i];
		P->unrank_line(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		Int_vec_print(cout, v, 2 * (P->Subspaces->n + 1));
		cout << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_set_of_points(
		std::ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;

	v = NEW_int(P->Subspaces->n + 1);

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
				Int_vec_print(ost, v, P->Subspaces->n + 1);
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}

void projective_space_reporting::print_set_of_points_easy(
		std::ostream &ost, long int *Pts, int nb_pts)
{
	int h;
	int *v;

	v = NEW_int(P->Subspaces->n + 1);

	for (h = 0; h < nb_pts; h++) {
		P->unrank_point(v, Pts[h]);
		ost << h << ": $" << Pts[h] << " = ";
		Int_vec_print(ost, v, P->Subspaces->n + 1);
		ost << "$";
		if (h < nb_pts - 1) {
			ost << ",";
		}
		ost << endl;
	}
	ost << "\\\\" << endl;
	FREE_int(v);
}



void projective_space_reporting::print_all_points()
{
	int *v;
	int i;

	v = NEW_int(P->Subspaces->n + 1);
	cout << "All points in PG(" << P->Subspaces->n << "," << P->Subspaces->q << "):" << endl;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		P->unrank_point(v, i);
		cout << setw(3) << i << " : ";
		Int_vec_print(cout, v, P->Subspaces->n + 1);
		cout << endl;
	}
	FREE_int(v);
}



}}}}

