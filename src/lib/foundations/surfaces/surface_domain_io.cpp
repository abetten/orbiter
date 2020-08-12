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

void surface_domain::print_line(ostream &ost, int rk)
{
	combinatorics_domain Combi;

	if (rk < 6) {
		ost << "a_" << rk + 1 << endl;
		}
	else if (rk < 12) {
		ost << "b_" << rk - 6 + 1 << endl;
		}
	else {
		int i, j;

		rk -= 12;
		Combi.k2ij(rk, i, j, 6);
		ost << "c_{" << i + 1 << j + 1 << "}";
		}
}

void surface_domain::print_Steiner_and_Eckardt(ostream &ost)
{
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Double Sixes}" << endl;
	latex_table_of_double_sixes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	latex_table_of_tritangent_planes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	latex_table_of_trihedral_pairs(ost);

}

void surface_domain::latex_abstract_trihedral_pair(ostream &ost, int t_idx)
{
	latex_trihedral_pair(ost, Trihedral_pairs + t_idx * 9,
		Trihedral_to_Eckardt + t_idx * 6);
}

void surface_domain::latex_trihedral_pair(ostream &ost, int *T, long int *TE)
{
	int i, j;

	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			print_line(ost, T[i * 3 + j]);
			ost << " & ";
			}
		ost << "\\pi_{";
		Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}\\\\" << endl;
		}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
		}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}

void surface_domain::latex_table_of_trihedral_pairs(ostream &ost)
{
	int i;

	cout << "surface::latex_table_of_trihedral_pairs" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Trihedral_pair_labels[i] << "} = \\\\" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair(ost, Trihedral_pairs + i * 9,
			Trihedral_to_Eckardt + i * 6);
		//ost << "\\end{array}" << endl;
		//ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
#if 0
		ost << "planes: $";
		int_vec_print(ost, Trihedral_to_Eckardt + i * 6, 6);
		ost << "$\\\\" << endl;
#endif
		}
	ost << "\\end{multicols}" << endl;

	print_trihedral_pairs(ost);

	cout << "surface_domain::latex_table_of_trihedral_pairs done" << endl;
}

void surface_domain::print_trihedral_pairs(ostream &ost)
{
	latex_interface L;
	int i, j;

	ost << "List of trihedral pairs:\\\\" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << i << " / " << nb_trihedral_pairs
			<< ": $T_{" << i << "} =  T_{"
			<< Trihedral_pair_labels[i] << "}=(";
		for (j = 0; j < 6; j++) {
			ost << "\\pi_{" << Trihedral_to_Eckardt[i * 6 + j]
				<< "}";
			if (j == 2) {
				ost << "; ";
				}
			else if (j < 6 - 1) {
				ost << ", ";
				}
			}
		ost << ")$\\\\" << endl;
		}
	ost << "List of trihedral pairs numerically:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Trihedral_to_Eckardt, 40, 6, 0, 0, TRUE /* f_tex*/);
	ost << "\\;";
	//ost << "$$" << endl;
	//ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0, TRUE /* f_tex*/);
	ost << "\\;";
	//ost << "$$" << endl;
	//ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void surface_domain::latex_table_of_double_sixes(ostream &ost)
{
	int i, j, h;
	long int D[12];

	cout << "surface::latex_table_of_double_sixes" << endl;



	//ost << "\\begin{multicols}{2}" << endl;
	for (h = 0; h < 36; h++) {

		lint_vec_copy(Double_six + h * 12, D, 12);

		ost << "$D_{" << h << "} = " << Double_six_label_tex[h] << endl;

		ost << " = \\left[";
		ost << "\\begin{array}{cccccc}" << endl;
		for (i = 0; i < 2; i++) {
			for (j = 0; j < 6; j++) {
				ost << Line_label_tex[D[i * 6 + j]];
				if (j < 6 - 1) {
					ost << " & ";
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
		ost << " = \\left[";
		ost << "\\begin{array}{cccccc}" << endl;
		for (i = 0; i < 2; i++) {
			for (j = 0; j < 6; j++) {
				ost << D[i * 6 + j];
				if (j < 6 - 1) {
					ost << " & ";
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
		}
	//ost << "\\end{multicols}" << endl;

	cout << "surface::latex_table_of_double_sixes done" << endl;

}


void surface_domain::latex_table_of_half_double_sixes(ostream &ost)
{
	int i, j;
	long int H[6];

	cout << "surface::latex_table_of_half_double_sixes" << endl;



	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < 72; i++) {

		lint_vec_copy(Half_double_sixes + i * 6, H, 6);

		ost << "$H_{" << i << "} = " << Half_double_six_label_tex[i] << endl;

		ost << " = \\{";
		for (j = 0; j < 6; j++) {
			ost << Line_label_tex[H[j]];
			if (j < 6 - 1) {
				ost << ", ";
				}
			}
		ost << "\\}$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;



	cout << "surface::latex_table_of_double_sixes done" << endl;
}


void surface_domain::latex_table_of_Eckardt_points(ostream &ost)
{
	int i, j;
	int three_lines[3];

	cout << "surface::latex_table_of_Eckardt_points" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(this, three_lines);

		ost << "$E_{" << i << "} = " << endl;
		Eckardt_points[i].latex(ost);
		ost << " = ";
		for (j = 0; j < 3; j++) {
			ost << Line_label_tex[three_lines[j]];
			if (j < 3 - 1) {
				ost << " \\cap ";
				}
			}
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
	cout << "surface::latex_table_of_Eckardt_points done" << endl;
}

void surface_domain::latex_table_of_tritangent_planes(ostream &ost)
{
	int i, j;
	int three_lines[3];

	cout << "surface::latex_table_of_tritangent_planes" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(this, three_lines);

		ost << "$\\pi_{" << i << "} = \\pi_{" << endl;
		Eckardt_points[i].latex_index_only(ost);
		ost << "} = ";
		for (j = 0; j < 3; j++) {
			ost << Line_label_tex[three_lines[j]];
			}
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
	cout << "surface_domain::latex_table_of_tritangent_planes done" << endl;
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

void surface_domain::print_lines_tex(ostream &ost, long int *Lines)
{
	int i;
	latex_interface L;

	for (i = 0; i < 27; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		Gr->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = "
			<< Line_label_tex[i] << " = " << endl;
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
	print_Schlaefli_labelling(ost);


	cout << "surface_domain::print_basics "
			"before print_Steiner_and_Eckardt" << endl;
	print_Steiner_and_Eckardt(ost);
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

void surface_domain::print_Schlaefli_labelling(ostream &ost)
{
	int j, h;

	ost << "The Schlaefli labeling of lines:\\\\" << endl;
	ost << "$$" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\begin{array}{|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "h &  \\mbox{line} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 9; h++) {
			ost << j * 9 + h << " & "
				<< Line_label_tex[j * 9 + h] << "\\\\" << endl;
			}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		if (j < 3 - 1) {
			ost << "\\qquad" << endl;
			}
		}
	ost << "$$" << endl;
}

void surface_domain::print_set_of_lines_tex(ostream &ost, long int *v, int len)
{
	int i;

	ost << "\\{";
	for (i = 0; i < len; i++) {
		ost << Line_label_tex[v[i]];
		if (i < len - 1) {
			ost << ", ";
			}
		}
	ost << "\\}";
}

void surface_domain::latex_table_of_clebsch_maps(ostream &ost)
{
	int e, line, j, l1, l2, t1, t2, t3, t4, c1, c2, cnt;
	int three_lines[3];
	int transversal_line;
	//int intersecting_lines[10];

	cnt = 0;
	cout << "surface_domain::latex_table_of_clebsch_maps" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	for (e = 0; e < nb_Eckardt_points; e++) {

		Eckardt_points[e].three_lines(this, three_lines);

		for (line = 0; line < 3; line++) {

			transversal_line = three_lines[line];
			if (line == 0) {
				c1 = three_lines[1];
				c2 = three_lines[2];
			}
			else if (line == 1) {
				c1 = three_lines[0];
				c2 = three_lines[2];
			}
			else if (line == 2) {
				c1 = three_lines[0];
				c2 = three_lines[1];
			}

			for (l1 = 0; l1 < 27; l1++) {
				if (l1 == c1 || l1 == c2) {
					continue;
				}
				if (get_adjacency_matrix_of_lines(
						transversal_line, l1) == 0) {
					continue;
				}
				for (l2 = l1 + 1; l2 < 27; l2++) {
					if (l2 == c1 || l2 == c2) {
						continue;
					}
					if (get_adjacency_matrix_of_lines(
							transversal_line, l2) == 0) {
						continue;
					}



					cout << "e=" << e << endl;
					cout << "transversal_line=" << transversal_line << endl;
					cout << "c1=" << c1 << endl;
					cout << "c2=" << c2 << endl;
					cout << "l1=" << l1 << endl;
					cout << "l2=" << l2 << endl;

					for (t1 = 0; t1 < 27; t1++) {
						if (t1 == three_lines[0] ||
								t1 == three_lines[1] ||
								t1 == three_lines[2]) {
							continue;
						}
						if (t1 == l1 || t1 == l2) {
							continue;
						}
						if (get_adjacency_matrix_of_lines(l1, t1) == 0 ||
								get_adjacency_matrix_of_lines(l2, t1) == 0) {
							continue;
						}
						cout << "t1=" << t1 << endl;

						for (t2 = t1 + 1; t2 < 27; t2++) {
							if (t2 == three_lines[0] ||
									t2 == three_lines[1] ||
									t2 == three_lines[2]) {
								continue;
							}
							if (t2 == l1 || t2 == l2) {
								continue;
							}
							if (get_adjacency_matrix_of_lines(l1, t2) == 0 ||
									get_adjacency_matrix_of_lines(l2, t2) == 0) {
								continue;
							}
							cout << "t2=" << t2 << endl;

							for (t3 = t2 + 1; t3 < 27; t3++) {
								if (t3 == three_lines[0] ||
										t3 == three_lines[1] ||
										t3 == three_lines[2]) {
									continue;
								}
								if (t3 == l1 || t3 == l2) {
									continue;
								}
								if (get_adjacency_matrix_of_lines(l1, t3) == 0 ||
										get_adjacency_matrix_of_lines(l2, t3) == 0) {
									continue;
								}
								cout << "t3=" << t3 << endl;

								for (t4 = t3 + 1; t4 < 27; t4++) {
									if (t4 == three_lines[0] ||
											t4 == three_lines[1] ||
											t4 == three_lines[2]) {
										continue;
									}
									if (t4 == l1 || t4 == l2) {
										continue;
									}
									if (get_adjacency_matrix_of_lines(l1, t4) == 0 ||
											get_adjacency_matrix_of_lines(l2, t4) == 0) {
										continue;
									}
									cout << "t4=" << t4 << endl;


									int tc1[4], tc2[4];
									int n1 = 0, n2 = 0;

									if (get_adjacency_matrix_of_lines(t1, c1)) {
										tc1[n1++] = t1;
									}
									if (get_adjacency_matrix_of_lines(t1, c2)) {
										tc2[n2++] = t1;
									}
									if (get_adjacency_matrix_of_lines(t2, c1)) {
										tc1[n1++] = t2;
									}
									if (get_adjacency_matrix_of_lines(t2, c2)) {
										tc2[n2++] = t2;
									}
									if (get_adjacency_matrix_of_lines(t3, c1)) {
										tc1[n1++] = t3;
									}
									if (get_adjacency_matrix_of_lines(t3, c2)) {
										tc2[n2++] = t3;
									}
									if (get_adjacency_matrix_of_lines(t4, c1)) {
										tc1[n1++] = t4;
									}
									if (get_adjacency_matrix_of_lines(t4, c2)) {
										tc2[n2++] = t4;
									}
									cout << "n1=" << n1 << endl;
									cout << "n2=" << n2 << endl;

									ost << cnt << " : $\\pi_{" << e << "} = \\pi_{";
									Eckardt_points[e].latex_index_only(ost);
									ost << "}$, $\\;$ ";

#if 0
									ost << " = ";
									for (j = 0; j < 3; j++) {
										ost << Line_label_tex[three_lines[j]];
										}
									ost << "$, $\\;$ " << endl;
#endif

									ost << "$" << Line_label_tex[transversal_line] << "$, $\\;$ ";
									//ost << "$(" << Line_label_tex[c1] << ", " << Line_label_tex[c2];
									//ost << ")$, $\\;$ ";

									ost << "$(" << Line_label_tex[l1] << "," << Line_label_tex[l2] << ")$, $\\;$ ";
#if 0
									ost << "$(" << Line_label_tex[t1]
										<< "," << Line_label_tex[t2]
										<< "," << Line_label_tex[t3]
										<< "," << Line_label_tex[t4]
										<< ")$, $\\;$ ";
#endif
									ost << "$"
											<< Line_label_tex[c1] << " \\cap \\{";
									for (j = 0; j < n1; j++) {
										ost << Line_label_tex[tc1[j]];
										if (j < n1 - 1) {
											ost << ", ";
										}
									}
									ost << "\\}$ ";
									ost << "$"
											<< Line_label_tex[c2] << " \\cap \\{";
									for (j = 0; j < n2; j++) {
										ost << Line_label_tex[tc2[j]];
										if (j < n2 - 1) {
											ost << ", ";
										}
									}
									ost << "\\}$ ";
									ost << "\\\\" << endl;
									cnt++;

								} // next t4
							} // next t3
						} // next t2
					} // next t1
					//ost << "\\hline" << endl;
				} // next l2
			} // next l1

		} // line
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
	} // e
	//ost << "\\end{multicols}" << endl;
	cout << "surface_domain::latex_table_of_clebsch_maps done" << endl;
}

void surface_domain::print_half_double_sixes_in_GAP()
{
	int i, j;

	cout << "[";
	for (i = 0; i < 72; i++) {
		cout << "[";
		for (j = 0; j < 6; j++) {
			cout << Half_double_sixes[i * 6 + j] + 1;
			if (j < 6 - 1) {
				cout << ", ";
			}
		}
		cout << "]";
		if (i < 72 - 1) {
			cout << "," << endl;
		}
	}
	cout << "];" << endl;
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
	sstr << Line_label_tex[pt];
}

void callback_surface_domain_sstr_line_label(stringstream &sstr, long int pt, void *data)
{
	surface_domain *D = (surface_domain *) data;

	//cout << "callback_surface_domain_sstr_line_label pt=" << pt << endl;
	D->sstr_line_label(sstr, pt);
}



}}
