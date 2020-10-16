// surface_domain_io.cpp
//
// Anton Betten
//
// moved here from surface.cpp: Dec 26, 2018
//
//
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


void surface_domain::print_equation(ostream &ost, int *coeffs)
{
	Poly3_4->print_equation(ost, coeffs);
}

void surface_domain::print_equation_tex(ostream &ost, int *coeffs)
{
	Poly3_4->print_equation_tex(ost, coeffs);
}

void surface_domain::print_equation_tex_lint(ostream &ost, long int *coeffs)
{
	Poly3_4->print_equation_lint_tex(ost, coeffs);
}

void surface_domain::latex_double_six(ostream &ost, long int *double_six)
{
	long int i, j, a, u, v;

	ost << "\\begin{array}{cc}" << endl;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 2; j++) {
			a = double_six[j * 6 + i];
			Gr->unrank_lint(a, 0);
			ost << "\\left[" << endl;
			ost << "\\begin{array}{*{6}{c}}" << endl;
			for (u = 0; u < 2; u++) {
				for (v = 0; v < 4; v++) {
					ost << Gr->M[u * 4 + v];
					if (v < 4 - 1) {
						ost << ", ";
					}
				}
				ost << "\\\\" << endl;
			}
			ost << "\\end{array}" << endl;
			ost << "\\right]" << endl;
			if (j < 2 - 1) {
				ost << ", " << endl;
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
}

void surface_domain::make_spreadsheet_of_lines_in_three_kinds(
	spreadsheet *&Sp,
	long int *Wedge_rk, long int *Line_rk, long int *Klein_rk, int nb_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a;
	char str[1000];
	int w[6];
	int Basis[8];
	char **Text_wedge;
	char **Text_line;
	char **Text_klein;

	if (f_v) {
		cout << "surface_domain::make_spreadsheet_of_lines_in_three_kinds" << endl;
		}

	Text_wedge = NEW_pchar(nb_lines);
	Text_line = NEW_pchar(nb_lines);
	Text_klein = NEW_pchar(nb_lines);

	for (i = 0; i < nb_lines; i++) {
		a = Wedge_rk[i];
		F->PG_element_unrank_modified_lint(w, 1, 6 /*wedge_dimension*/, a);
		int_vec_print_to_str(str, w, 6);
		Text_wedge[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_wedge[i], str);
		}
	for (i = 0; i < nb_lines; i++) {
		a = Line_rk[i];
		Gr->unrank_lint_here(Basis, a, 0 /* verbose_level */);
		int_vec_print_to_str(str, Basis, 8);
		Text_line[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_line[i], str);
		}
	for (i = 0; i < nb_lines; i++) {
		a = Klein_rk[i];
		O->unrank_point(w, 1, a, 0 /* verbose_level*/);
			// error corrected: w was v which was v[4], so too short.
			// Aug 25, 2018
		int_vec_print_to_str(str, w, 6);
			// w was v, error corrected
		Text_klein[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_klein[i], str);
		}

	Sp = NEW_OBJECT(spreadsheet);
	Sp->init_empty_table(nb_lines + 1, 7);
	Sp->fill_column_with_row_index(0, "Idx");
	Sp->fill_column_with_lint(1, Wedge_rk, "Wedge_rk");
	Sp->fill_column_with_text(2,
			(const char **) Text_wedge, "Wedge coords");
	Sp->fill_column_with_lint(3, Line_rk, "Line_rk");
	Sp->fill_column_with_text(4,
			(const char **) Text_line, "Line basis");
	Sp->fill_column_with_lint(5, Klein_rk, "Klein_rk");
	Sp->fill_column_with_text(6,
			(const char **) Text_klein, "Klein coords");

	for (i = 0; i < nb_lines; i++) {
		FREE_char(Text_wedge[i]);
		}
	FREE_pchar(Text_wedge);
	for (i = 0; i < nb_lines; i++) {
		FREE_char(Text_line[i]);
		}
	FREE_pchar(Text_line);
	for (i = 0; i < nb_lines; i++) {
		FREE_char(Text_klein[i]);
		}
	FREE_pchar(Text_klein);


	if (f_v) {
		cout << "surface_domain::make_spreadsheet_of_lines_"
				"in_three_kinds done" << endl;
		}
}



#if 0
void surface_domain::print_web_of_cubic_curves(ostream &ost,
	int *Web_of_cubic_curves)
// curves[45 * 10]
{
	latex_interface L;

	ost << "Web of cubic curves:\\\\" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
		Web_of_cubic_curves, 45, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;
}
#endif

void surface_domain::print_equation_in_trihedral_form(ostream &ost,
	int *the_six_plane_equations, int lambda, int *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = F_0F_1F_2 + \\lambda G_0G_1G_2\\\\" << endl;
	ost << "& = " << endl;
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 0 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 1 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 2 * 4);
	ost << "\\Big)";
	ost << "+ " << lambda;
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 3 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 4 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 5 * 4);
	ost << "\\Big)\\\\";
	ost << "& \\equiv " << endl;
	Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}

void surface_domain::print_equation_wrapped(ostream &ost, int *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = " << endl;
	Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}

void surface_domain::print_lines_tex(ostream &ost, long int *Lines, int nb_lines)
{
	int i;
	latex_interface L;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		Gr->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "}";

		if (nb_lines == 27) {
			ost << " = " << Schlaefli->Line_label_tex[i];
		}
		ost << " = " << endl;
		//print_integer_matrix_width(cout,
		// Gr->M, k, n, n, F->log10_of_q + 1);
		Gr->latex_matrix(ost, Gr->M);
		//print_integer_matrix_tex(ost, Gr->M, 2, 4);
		//ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "_{" << Lines[i] << "}" << endl;
		ost << "=" << endl;
		ost << "\\left[" << endl;
		L.print_integer_matrix_tex(ost, Gr->M, 2, 4);
		ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "$$" << endl;
		}

}

void surface_domain::print_clebsch_P(ostream &ost)
{
	int h, i, f_first;

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::print_clebsch_P f_has_large_polynomial_"
				"domains is FALSE" << endl;
		//exit(1);
		return;
		}
	ost << "\\clearpage" << endl;
	ost << "\\subsection*{The Clebsch system $P$}" << endl;

	ost << "$$" << endl;
	print_clebsch_P_matrix_only(ost);
	ost << "\\cdot \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "x_0\\\\" << endl;
	ost << "x_1\\\\" << endl;
	ost << "x_2\\\\" << endl;
	ost << "x_3\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "= \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;


	ost << "\\begin{align*}" << endl;
	for (h = 0; h < 4; h++) {
		ost << "x_" << h << " &= C_" << h
				<< "(y_0,y_1,y_2)=\\\\" << endl;
		f_first = TRUE;
		for (i = 0; i < Poly3->get_nb_monomials(); i++) {

			if (Poly3_24->is_zero(CC[h * Poly3->get_nb_monomials() + i])) {
				continue;
				}
			ost << "&";

			if (f_first) {
				f_first = FALSE;
				}
			else {
				ost << "+";
				}
			ost << "\\Big(";
			Poly3_24->print_equation_with_line_breaks_tex(
					ost, CC[h * Poly3->get_nb_monomials() + i],
					6, "\\\\\n&");
			ost << "\\Big)" << endl;

			ost << "\\cdot" << endl;

			Poly3->print_monomial(ost, i);
			ost << "\\\\" << endl;
			}
		}
	ost << "\\end{align*}" << endl;
}

void surface_domain::print_clebsch_P_matrix_only(ostream &ost)
{
	int i, j;

	if (!f_has_large_polynomial_domains) {
		cout << "surface::print_clebsch_P_matrix_only "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	ost << "\\left[" << endl;
	ost << "\\begin{array}{cccc}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			//cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(ost, Clebsch_P[i * 4 + j]);
			if (j < 4 - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void surface_domain::print_clebsch_cubics(ostream &ost)
{
	int i, h;

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::print_clebsch_cubics "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	ost << "The Clebsch coefficients are:" << endl;
	for (h = 0; h < 4; h++) {
		ost << "C[" << h << "]:" << endl;
		for (i = 0; i < Poly3->get_nb_monomials(); i++) {

			if (Poly3_24->is_zero(CC[h * Poly3->get_nb_monomials() + i])) {
				continue;
				}

			Poly3->print_monomial(ost, i);
			ost << " \\cdot \\Big(";
			Poly3_24->print_equation(ost, CC[h * Poly3->get_nb_monomials() + i]);
			ost << "\\Big)" << endl;
			}
		}
}

void surface_domain::print_system(ostream &ost, int *system)
{
	int i, j;

	//ost << "The system:\\\\";
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	ost << "\\begin{array}{cccc}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			int *p = system + (i * 4 + j) * 3;
			Poly1->print_equation(ost, p);
			if (j < 4 - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "\\cdot \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "x_0\\\\" << endl;
	ost << "x_1\\\\" << endl;
	ost << "x_2\\\\" << endl;
	ost << "x_3\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "= \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;
}

void surface_domain::print_trihedral_pair_in_dual_coordinates_in_GAP(
	long int *F_planes_rank, long int *G_planes_rank)
{
	int i;
	int F_planes[12];
	int G_planes[12];

	for (i = 0; i < 3; i++) {
		P->unrank_point(F_planes + i * 4, F_planes_rank[i]);
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(G_planes + i * 4, G_planes_rank[i]);
		}
	cout << "[";
	for (i = 0; i < 3; i++) {
		int_vec_print_GAP(cout, F_planes + i * 4, 4);
		cout << ", ";
		}
	for (i = 0; i < 3; i++) {
		int_vec_print_GAP(cout, G_planes + i * 4, 4);
		if (i < 3 - 1) {
			cout << ", ";
			}
		}
	cout << "];";
}

void surface_domain::print_basics(ostream &ost)
{
	print_polynomial_domains(ost);
	Schlaefli->print_Schlaefli_labelling(ost);


	cout << "surface_domain::print_basics "
			"before print_Steiner_and_Eckardt" << endl;
	Schlaefli->print_Steiner_and_Eckardt(ost);
	cout << "surface_domain::print_basics "
			"after print_Steiner_and_Eckardt" << endl;

	cout << "surface_domain::print_basics "
			"before print_clebsch_P" << endl;
	print_clebsch_P(ost);
	cout << "surface_domain::print_basics "
			"after print_clebsch_P" << endl;

}


void surface_domain::print_polynomial_domains(ostream &ost)
{
	ost << "The polynomial domain Poly3\\_4 is:" << endl;
	Poly3_4->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly1\\_x123 is:" << endl;
	Poly1_x123->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly2\\_x123 is:" << endl;
	Poly2_x123->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly3\\_x123 is:" << endl;
	Poly3_x123->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly4\\_x123 is:" << endl;
	Poly4_x123->print_monomial_ordering(ost);

}



void surface_domain::sstr_line_label(stringstream &sstr, long int pt)
{
	if (pt >= 27) {
		cout << "surface_domain::sstr_line_label pt >= 27, pt=" << pt << endl;
		exit(1);
	}
	if (pt < 0) {
		cout << "surface_domain::sstr_line_label pt < 0, pt=" << pt << endl;
		exit(1);
	}
	sstr << Schlaefli->Line_label_tex[pt];
}

void callback_surface_domain_sstr_line_label(stringstream &sstr, long int pt, void *data)
{
	surface_domain *D = (surface_domain *) data;

	//cout << "callback_surface_domain_sstr_line_label pt=" << pt << endl;
	D->sstr_line_label(sstr, pt);
}



}}
