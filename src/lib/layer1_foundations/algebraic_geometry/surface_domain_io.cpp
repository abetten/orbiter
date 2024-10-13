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
namespace layer1_foundations {
namespace algebraic_geometry {


void surface_domain::print_equation(
		std::ostream &ost, int *coeffs)
{
	PolynomialDomains->Poly3_4->print_equation(ost, coeffs);
}

void surface_domain::print_equation_maple(
		std::stringstream &ost, int *coeffs)
{
	PolynomialDomains->Poly3_4->print_equation_str(ost, coeffs);
}


void surface_domain::print_equation_tex(
		std::ostream &ost, int *coeffs)
{
	PolynomialDomains->Poly3_4->print_equation_tex(ost, coeffs);
}

void surface_domain::print_equation_with_line_breaks_tex(
		std::ostream &ost, int *coeffs)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;
	PolynomialDomains->Poly3_4->print_equation_with_line_breaks_tex(
			ost, coeffs, 10 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
	ost << "=0" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}

void surface_domain::print_equation_tex_lint(
		std::ostream &ost, long int *coeffs)
{
	PolynomialDomains->Poly3_4->print_equation_lint_tex(ost, coeffs);
}

#if 0
void surface_domain::latex_double_six(
		std::ostream &ost, long int *double_six)
{

#if 0
	long int i, j, a, u, v;

	ost << "\\begin{array}{cc}" << endl;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 2; j++) {
			a = double_six[j * 6 + i];
			Gr->unrank_lint(a, 0);

			ost << Schlaefli->Labels->Line_label[i + j * 6];
			ost << " = ";
			ost << "\\left[" << endl;
			ost << "\\begin{array}{*{4}{c}}" << endl;
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
			ost << "\\right]_{" << a << "}" << endl;

			if (j < 2 - 1) {
				ost << ", " << endl;
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
#else

	print_lines_tex(ost, double_six, 12 /* nb_lines */);

#endif
}
#endif


void surface_domain::latex_double_six(
		std::ostream &ost, long int *double_six)
{

	print_lines_tex(ost, double_six, 12 /* nb_lines */);
}


void surface_domain::make_spreadsheet_of_lines_in_three_kinds(
		data_structures::spreadsheet *&Sp,
	long int *Wedge_rk, long int *Line_rk,
	long int *Klein_rk, int nb_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a;
	string s;
	int w[6];
	int Basis[8];
	string *Text_wedge;
	string *Text_line;
	string *Text_klein;

	if (f_v) {
		cout << "surface_domain::make_spreadsheet_of_lines_in_three_kinds" << endl;
	}

	Text_wedge = new string[nb_lines];
	Text_line = new string[nb_lines];
	Text_klein = new string[nb_lines];

	for (i = 0; i < nb_lines; i++) {
		a = Wedge_rk[i];
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				w, 1, 6 /*wedge_dimension*/, a);
		Int_vec_print_to_str(Text_wedge[i], w, 6);
	}
	for (i = 0; i < nb_lines; i++) {
		a = Line_rk[i];
		Gr->unrank_lint_here(Basis, a, 0 /* verbose_level */);
		Int_vec_print_to_str(Text_line[i], Basis, 8);
	}
	for (i = 0; i < nb_lines; i++) {
		a = Klein_rk[i];
		O->Hyperbolic_pair->unrank_point(w, 1, a, 0 /* verbose_level*/);
			// error corrected: w was v which was v[4], so too short.
			// Aug 25, 2018
		Int_vec_print_to_str(Text_klein[i], w, 6);
	}

	Sp = NEW_OBJECT(data_structures::spreadsheet);
	Sp->init_empty_table(nb_lines + 1, 7);
	Sp->fill_column_with_row_index(0, "Idx");
	Sp->fill_column_with_lint(1, Wedge_rk, "Wedge_rk");
	Sp->fill_column_with_text(2,
			Text_wedge, "Wedge coords");
	Sp->fill_column_with_lint(3, Line_rk, "Line_rk");
	Sp->fill_column_with_text(4,
			Text_line, "Line basis");
	Sp->fill_column_with_lint(5, Klein_rk, "Klein_rk");
	Sp->fill_column_with_text(6,
			Text_klein, "Klein coords");

	delete [] Text_wedge;
	delete [] Text_line;
	delete [] Text_klein;


	if (f_v) {
		cout << "surface_domain::make_spreadsheet_of_lines_in_three_kinds done" << endl;
	}
}



#if 0
void surface_domain::print_web_of_cubic_curves(std::ostream &ost,
	int *Web_of_cubic_curves)
// curves[45 * 10]
{
	latex_interface L;

	ost << "Web of cubic curves:\\\\" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
		Web_of_cubic_curves, 45, 10, true /* f_tex*/);
	ost << "$$" << endl;
}
#endif


void surface_domain::print_equation_wrapped(
		std::ostream &ost, int *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = " << endl;
	PolynomialDomains->Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}

void surface_domain::print_lines_tex(
		std::ostream &ost, long int *Lines, int nb_lines)
{
	int idx;
	long int *Rk;
	int vv[6];

	Rk = NEW_lint(nb_lines);

	ost << "The lines and their Pluecker coordinates are:\\\\" << endl;

	for (idx = 0; idx < nb_lines; idx++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		int v6[6];

		Gr->Pluecker_coordinates(Lines[idx], v6, 0 /* verbose_level */);

		Int_vec_copy(v6, vv, 6); // mistake found by Alice Hui

		Rk[idx] = O->Orthogonal_indexing->Qplus_rank(vv, 1, 5, 0 /* verbose_level*/);

	}

	for (idx = 0; idx < nb_lines; idx++) {

		print_one_line_tex(ost, Lines, nb_lines, idx);

	}

	ost << "Rank of lines: ";
	Lint_vec_print(ost, Lines, nb_lines);
	ost << "\\\\" << endl;
	ost << "Rank of points on Klein quadric: ";
	Lint_vec_print(ost, Rk, nb_lines);
	ost << "\\\\" << endl;

	//alice(ost, Lines, nb_lines);

	FREE_lint(Rk);

}


#if 0
void surface_domain::alice(std::ostream &ost, long int *Lines, int nb_lines)
{
	int Pa6[6];
	int Pb6[6];
	int Pa2[6];
	int P_line[6];
	int tmp[6];
	long int rk_a6;
	long int rk_b6;
	int h;

	P->Pluecker_coordinates(Lines[5], Pa6, 0 /* verbose_level */);
	P->Pluecker_coordinates(Lines[11], Pb6, 0 /* verbose_level */);
	P->Pluecker_coordinates(Lines[1], Pa2, 0 /* verbose_level */);

	Int_vec_copy(Pa6, tmp, 6);
	rk_a6 = F->Orthogonal_indexing->Qplus_rank(tmp, 1, 5, 0 /* verbose_level*/);

	Int_vec_copy(Pb6, tmp, 6);
	rk_b6 = F->Orthogonal_indexing->Qplus_rank(tmp, 1, 5, 0 /* verbose_level*/);

	if (rk_a6 != 1) {
		return;
	}
	if (rk_b6 != 0) {
		return;
	}
	Int_vec_copy(Pa2, tmp, 6);
	if (Pa2[2] || Pa2[3] || Pa2[4]) {
		return;
	}

	int v[3];
	long int rk;

	ost << "\\section*{Projected points:}" << endl;
	for (h = 0; h < 27; h++) {
		if (h == 5 || h == 11 || h == 1) {
			continue;
		}

		P->Pluecker_coordinates(Lines[h], P_line, 0 /* verbose_level */);
		Int_vec_copy(P_line + 2, v, 3);

		rk = P2->rank_point(v);


		ost << "$" << Schlaefli->Labels->Line_label_tex[h];
		ost << " = ";
		Int_vec_print(ost, v, 3);
		ost << "_{";
		ost << rk;
		ost << "}$\\\\" << endl;


	}
	for (h = 0; h < 27; h++) {
		if (h == 5 || h == 11 || h == 1) {
			continue;
		}
		P->Pluecker_coordinates(Lines[h], P_line, 0 /* verbose_level */);
		Int_vec_copy(P_line + 2, v, 3);

		rk = P2->rank_point(v);
		ost << rk << ",";
	}
	ost << "\\\\" << endl;


}
#endif


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
		Int_vec_print_GAP(cout, F_planes + i * 4, 4);
		cout << ", ";
	}
	for (i = 0; i < 3; i++) {
		Int_vec_print_GAP(cout, G_planes + i * 4, 4);
		if (i < 3 - 1) {
			cout << ", ";
		}
	}
	cout << "];";
}

void surface_domain::print_basics(
		std::ostream &ost)
{
	PolynomialDomains->print_polynomial_domains_latex(ost);

	Schlaefli->print_Schlaefli_labelling(ost);


	cout << "surface_domain::print_basics "
			"before print_Steiner_and_Eckardt" << endl;
	Schlaefli->print_Steiner_and_Eckardt(ost);
	cout << "surface_domain::print_basics "
			"after print_Steiner_and_Eckardt" << endl;

	cout << "surface_domain::print_basics "
			"before print_clebsch_P" << endl;
	PolynomialDomains->print_clebsch_P(ost);
	cout << "surface_domain::print_basics "
			"after print_clebsch_P" << endl;

}


void surface_domain::print_point_with_orbiter_rank(
		std::ostream &ost, long int rk, int *v)
{
	ost << "P_{" << rk << "}";
	ost << "=";
	ost << "\\bP";
	Int_vec_print_fully(ost, v, 4);
}

void surface_domain::print_point(
		std::ostream &ost, int *v)
{
	ost << "\\bP";
	Int_vec_print_fully(ost, v, 4);
}

#if 0
void surface_domain::sstr_line_label(
		std::stringstream &sstr, long int pt)
{
	if (pt >= 27) {
		cout << "surface_domain::sstr_line_label pt >= 27, pt=" << pt << endl;
		exit(1);
	}
	if (pt < 0) {
		cout << "surface_domain::sstr_line_label pt < 0, pt=" << pt << endl;
		exit(1);
	}
	sstr << Schlaefli->Labels->Line_label_tex[pt];
}

void surface_domain::sstr_tritangent_plane_label(
		std::stringstream &sstr, long int pt)
{
	if (pt >= 45) {
		cout << "surface_domain::sstr_tritangent_plane_label pt >= 45, pt=" << pt << endl;
		exit(1);
	}
	if (pt < 0) {
		cout << "surface_domain::sstr_tritangent_plane_label pt < 0, pt=" << pt << endl;
		exit(1);
	}
	sstr << Schlaefli->Labels->Tritangent_plane_label_tex[pt];
}
#endif

void surface_domain::print_one_line_tex(
		std::ostream &ost,
		long int *Lines, int nb_lines, int idx)
{
	l1_interfaces::latex_interface L;
	int vv[6];
	long int line_rk;
	int Basis1[8];
	int Basis2[8];


	line_rk = Lines[idx];

	Gr->unrank_lint_here(
			Basis1, line_rk, 0 /*verbose_level*/);

	Int_vec_copy(Basis1, Basis2, 8);

	Gr->F->Linear_algebra->Gauss_easy_from_the_back(
			Basis2, 2, 4);


	//Gr->unrank_lint(Lines[idx], 0 /*verbose_level*/);


	ost << "$$" << endl;
	ost << "\\ell_{" << idx << "}";

	if (nb_lines == 27) {
		ost << " = " << Schlaefli->Labels->Line_label_tex[idx];
	}
	ost << " = " << endl;
	//print_integer_matrix_width(cout,
	// Gr->M, k, n, n, F->log10_of_q + 1);

#if 1
	Gr->latex_matrix(ost, Basis1);

	ost << " = ";
	Gr->latex_matrix(ost, Basis2);


	//print_integer_matrix_tex(ost, Gr->M, 2, 4);
	//ost << "\\right]_{" << Lines[i] << "}" << endl;


	ost << "_{" << line_rk << "}" << endl;
#endif

#if 0
	ost << "=" << endl;
	ost << "\\left[" << endl;
	L.print_integer_matrix_tex(ost, Basis1, 2, 4);
	ost << "\\right]";
	//ost << "_{" << Lines[idx] << "}";
	ost << endl;
#endif

	int v6[6];

	Gr->Pluecker_coordinates(line_rk, v6, 0 /* verbose_level */);

	Int_vec_copy(v6, vv, 6); // mistake found by Alice Hui

	long int klein_rk;
	klein_rk = O->Orthogonal_indexing->Qplus_rank(
			vv, 1, 5, 0 /* verbose_level*/);

	ost << "={\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
			<< v6[2] << "," << v6[3] << "," << v6[4]
			<< "," << v6[5] << " ";
	ost << ")";
	ost << "_{" << klein_rk << "}";
	ost << endl;
	ost << "$$" << endl;
}





}}}

