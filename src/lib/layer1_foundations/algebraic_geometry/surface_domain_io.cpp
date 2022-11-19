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


void surface_domain::print_equation(std::ostream &ost, int *coeffs)
{
	Poly3_4->print_equation(ost, coeffs);
}

void surface_domain::print_equation_maple(std::stringstream &ost, int *coeffs)
{
	Poly3_4->print_equation_str(ost, coeffs);
}


void surface_domain::print_equation_tex(std::ostream &ost, int *coeffs)
{
	Poly3_4->print_equation_tex(ost, coeffs);
}

void surface_domain::print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;
	Poly3_4->print_equation_with_line_breaks_tex(
			ost, coeffs, 10 /* nb_terms_per_line*/,
			"\\\\\n" /* const char *new_line_text*/);
	ost << "=0" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}

void surface_domain::print_equation_tex_lint(std::ostream &ost, long int *coeffs)
{
	Poly3_4->print_equation_lint_tex(ost, coeffs);
}

void surface_domain::latex_double_six(std::ostream &ost, long int *double_six)
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

void surface_domain::make_spreadsheet_of_lines_in_three_kinds(
		data_structures::spreadsheet *&Sp,
	long int *Wedge_rk, long int *Line_rk, long int *Klein_rk, int nb_lines,
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
		F->PG_element_unrank_modified_lint(w, 1, 6 /*wedge_dimension*/, a);
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

void surface_domain::print_equation_wrapped(std::ostream &ost, int *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = " << endl;
	Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}

void surface_domain::print_lines_tex(std::ostream &ost, long int *Lines, int nb_lines)
{
	int i;
	orbiter_kernel_system::latex_interface L;
	long int *Rk;
	int vv[6];

	Rk = NEW_lint(nb_lines);

	ost << "The lines and their Pluecker coordinates are:\\\\" << endl;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		Gr->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "}";

		if (nb_lines == 27) {
			ost << " = " << Schlaefli->Labels->Line_label_tex[i];
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

		int v6[6];

		Gr->Pluecker_coordinates(Lines[i], v6, 0 /* verbose_level */);

		Int_vec_copy(v6, vv, 6); // mistake found by Alice Hui

		Rk[i] = F->Orthogonal_indexing->Qplus_rank(vv, 1, 5, 0 /* verbose_level*/);

		ost << "={\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
				<< v6[2] << "," << v6[3] << "," << v6[4]
				<< "," << v6[5] << " ";
		ost << ")_{" << Rk[i] << "}";
		ost << "$$" << endl;
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
	sstr << Schlaefli->Labels->Line_label_tex[pt];
}


void surface_domain::make_table_of_surfaces(int verbose_level)
{


	//int f_v = (verbose_level >= 1);


	string fname;
	string author;
	string title;
	string extras_for_preamble;

	fname.assign("surfaces_report.tex");

	author.assign("Orbiter");

	title.assign("Cubic Surfaces with 27 Lines over Finite Fields");


	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);


		{
			int Q[] = {
					4,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,
					41,43,47,49,53,59,61,64,67,71,73,79,81,83, 89, 97, 101, 103, 107, 109, 113, 121, 128
				};
			int nb_Q = sizeof(Q) / sizeof(int);

			fp << "\\section*{Cubic Surfaces}" << endl;

			make_table_of_surfaces2(fp, Q, nb_Q, verbose_level);
		}

		fp << endl;

#if 0
		fp << "\\clearpage" << endl;

		fp << endl;


		{
			int Q_even[] = {
				4,8,16,32,64,128
				};
			int nb_Q_even = 6;

			fp << "\\section*{Even Characteristic}" << endl;

			make_table_of_surfaces2(fp, Q_even, nb_Q_even, verbose_level);
		}

		fp << endl;

		fp << "\\clearpage" << endl;

		fp << endl;


		{
			int Q_odd[] = {
					7,9,11,13,17,19,23,25,27,29,31,37,
					41,43,47,49,53,59,61,67,71,73,79,81,83, 89, 97, 101, 103, 107, 109, 113, 121
				};
			int nb_Q_odd = sizeof(Q_odd) / sizeof(int);


			fp << "\\section*{Odd Characteristic}" << endl;

			make_table_of_surfaces2(fp, Q_odd, nb_Q_odd, verbose_level);
		}
#endif

		L.foot(fp);
	}

}


void surface_domain::make_table_of_surfaces_detailed(
		int *Q_table, int Q_table_len, int verbose_level)
{
	int i, j, q, cur, nb_E;
	int nb_reps_total;
	int *Nb_reps;
	knowledge_base K;
	long int *Big_table;
	orbiter_kernel_system::file_io Fio;

	Nb_reps = NEW_int(Q_table_len);

	nb_reps_total = 0;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		Nb_reps[i] = K.cubic_surface_nb_reps(q);
		nb_reps_total += Nb_reps[i];
	}
	Big_table = NEW_lint(nb_reps_total * 4);

	cur = 0;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		for (j = 0; j < Nb_reps[i]; j++, cur++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			Big_table[cur * 4 + 0] = q;
			Big_table[cur * 4 + 1] = nb_E;
			Big_table[cur * 4 + 2] = j;

			int *data;
			int nb_gens;
			int data_size;
			string stab_order;
			long int ago;
			data_structures::string_tools ST;

			K.cubic_surface_stab_gens(q, j, data, nb_gens, data_size, stab_order);
			ago = ST.strtolint(stab_order);

			Big_table[cur * 4 + 3] = ago;
		}
	}
	std::string fname;

	fname.assign("table_of_cubic_surfaces_QECA.csv");

	std::string *headers;

	headers = new string[4];


	headers[0].assign("Q");
	headers[1].assign("E");
	headers[2].assign("OCN");
	headers[3].assign("AUT");


	Fio.lint_matrix_write_csv_override_headers(fname, headers, Big_table, nb_reps_total, 4);

	FREE_lint(Big_table);
}

void surface_domain::make_table_of_surfaces2(std::ostream &ost,
		int *Q_table, int Q_table_len, int verbose_level)
{
#if 0
	int Q_table[] = {
		4,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,
		41,43,47,49,53,59,61,64,67,71,73,79,81,83, 89, 97};
	int Q_table[] = {
		4,8,16,32,64,128
	};
#endif
	//int Q_table_len = sizeof(Q_table) / sizeof(int);
	int q, nb_reps;
	int i, j, nb_E;
	int *data;
	int nb_gens;
	int data_size;
	knowledge_base K;
	orbiter_kernel_system::file_io Fio;


	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		cout << q << " : " << nb_reps << "\\\\" << endl;
	}

#if 0
	const char *fname_ago = "ago.csv";
	{
	ofstream f(fname_ago);

	f << "q,j,nb_E,stab_order" << endl;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			K.cubic_surface_stab_gens(q, j,
					data, nb_gens, data_size, stab_order);
			f << q << "," << j << ", " << nb_E << ", "
					<< stab_order << endl;
			}
		}
	f << "END" << endl;
	}
	cout << "Written file " << fname_ago << " of size "
			<< Fio.file_size(fname_ago) << endl;

	const char *fname_dist = "ago_dist.csv";
	{
	ofstream f(fname_dist);
	int *Ago;

	f << "q,ago" << endl;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		Ago = NEW_int(nb_reps);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			K.cubic_surface_stab_gens(q, j, data,
					nb_gens, data_size, stab_order);
			sscanf(stab_order, "%d", &Ago[j]);
			//f << q << "," << j << ", " << nb_E << ", " << stab_order << endl;
			}
		tally C;

		C.init(Ago, nb_reps, FALSE, 0);
		f << q << ", ";
		C.print_naked_tex(f, TRUE /* f_backwards*/);
		f << endl;

		FREE_int(Ago);
		}
	f << "END" << endl;
	}
	cout << "Written file " << fname_dist << " of size "
			<< Fio.file_size(fname_dist) << endl;
#endif

	long int *Table;
	long int *Table2;
	int *Q;
	int nb_Q;
	int *E;
	int nb_E_types;

	compute_table_E(Q_table, Q_table_len,
			Table, Q, nb_Q, E, nb_E_types, verbose_level);

	Table2 = NEW_lint(nb_Q * nb_E_types + 1);
	for (i = 0; i < nb_Q; i++) {
		Table2[i * (nb_E_types + 1) + 0] = Q[i];
		for (j = 0; j < nb_reps; j++) {
			Table2[i * (nb_E_types + 1) + 1 + j] = Table[i * nb_E_types + j];
		}
	}

	//file_io Fio;
	std::string fname;

	fname.assign("table_of_cubic_surfaces_QE.csv");

	std::string *headers;

	headers = new string[nb_E_types + 1];


	headers[0].assign("Q");
	for (j = 0; j < nb_E_types; j++) {
		char str[1000];

		snprintf(str, sizeof(str), "E%d", E[j]);
		headers[1 + j].assign(str);
	}


	Fio.lint_matrix_write_csv_override_headers(fname, headers, Table2, nb_Q, nb_E_types + 1);
	FREE_lint(Table2);

	//LG->report(fp, f_sylow, f_group_table, verbose_level);



	int Nb_total = 0;

	ost << "$$" << endl;
	ost << "\\begin{array}{|r||r||*{" << nb_E_types << "}{r|}}" << endl;
	ost << "\\hline" << endl;
	ost << "q  & \\mbox{total} ";
	for (j = 0; j < nb_E_types; j++) {
		ost << " & " << E[j];
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_Q; i++) {
		q = Q[i];
		ost << q;
		nb_reps = K.cubic_surface_nb_reps(q);
		Nb_total += nb_reps;
		ost << " & ";
		ost << nb_reps;
		for (j = 0; j < nb_E_types; j++) {
			ost << " & " << Table[i * nb_E_types + j];
			}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	//cout << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	ost << "Total: " << Nb_total << endl;





	ost << "\\bigskip" << endl;

	for (j = 0; j < nb_E_types; j++) {
		ost << "\\section*{" << E[j] << " Eckardt Points}" << endl;

		Nb_total = 0;

		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|p{8cm}|}" << endl;
		ost << "\\hline" << endl;
		ost << "q & \\mbox{total} & \\mbox{Ago} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;

		for (i = 0; i < nb_Q; i++) {
			q = Q[i];
			nb_reps = Table[i * nb_E_types + j];
			Nb_total += nb_reps;
			if (nb_reps) {

				int *Ago;
				int go;
				int h, u, nb_total;
				data_structures::string_tools ST;

				nb_total = K.cubic_surface_nb_reps(q);
				Ago = NEW_int(nb_reps);
				u = 0;
				for (h = 0; h < nb_total; h++) {
					nb_E = K.cubic_surface_nb_Eckardt_points(q, h);
					if (nb_E != E[j]) {
						continue;
					}
					string stab_order;

					K.cubic_surface_stab_gens(q, h, data,
							nb_gens, data_size, stab_order);

					go = ST.strtolint(stab_order);
					Ago[u++] = go;
				}

				if (u != nb_reps) {
					cout << "u != nb_reps" << endl;
					exit(1);
				}
				data_structures::tally C;

				C.init(Ago, nb_reps, FALSE, 0);
				ost << q << " & " << nb_reps << " & ";
				ost << "$";
				C.print_naked_tex(ost, TRUE /* f_backwards*/);
				ost << "$\\\\" << endl;

				FREE_int(Ago);



				ost << "\\hline" << endl;
			}
		}


		ost << "\\end{array}" << endl;
		ost << "$$" << endl;

		ost << "Total: " << Nb_total << endl;


		ost << "\\bigskip" << endl;


	} // next j

#if 0
	table_top(ost);

	h = 0;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		nb_reps = K.cubic_surface_nb_reps(q);




		for (j = 0; j < nb_reps; j++, h++) {

			int *data;
			int nb_gens;
			int data_size;
			const char *stab_order;

			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			K.cubic_surface_stab_gens(q, j,
					data, nb_gens, data_size, stab_order);
			ost << q << " & " << j << " & " << stab_order
					<< " & " << nb_E << " & \\\\" << endl;
			if ((h + 1) % 30 == 0) {
				table_bottom(ost);
				if ((h + 1) % 60 == 0) {
					ost << endl;
					ost << "\\bigskip" << endl;
					ost << endl;
					}
				table_top(ost);
				}
			}
		ost << "\\hline" << endl;
		}
	table_bottom(ost);
#endif

	FREE_lint(Table);
	FREE_int(Q);
	FREE_int(E);


	make_table_of_surfaces_detailed(Q_table, Q_table_len, verbose_level);


}

void surface_domain::table_top(std::ostream &ost)
{
	ost << "$" << endl;
	ost << "\\begin{array}{|c|c||c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "q & \\mbox{Iso} & \\mbox{Ago} & \\# E & "
			"\\mbox{Comment}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
}

void surface_domain::table_bottom(std::ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	//ost << "\\quad" << endl;
	ost << "$" << endl;
}

void surface_domain::compute_table_E(
		int *field_orders, int nb_fields,
		long int *&Table,
		int *&Q, int &nb_Q,
		int *&E, int &nb_E_types, int verbose_level)
{
	//int Q_table[] = {4,7,8,9,11,13,16,17,19,23,25,27,29,
	//		31,32,37,41,43,47,49,53,59,61,64,67,71,73,79,81,83,89, 97};
	//int Q_table_len = sizeof(Q_table) / sizeof(int);
	int i, j, q, nb_reps, nb_E, nb_E_max, idx;
	int *Table_idx;
	knowledge_base K;

	nb_Q = nb_fields;
	Q = NEW_int(nb_Q);
	Int_vec_copy(field_orders, Q, nb_Q);

	nb_E_max = 0;
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			nb_E_max = MAXIMUM(nb_E_max, nb_E);
			}
		}
	cout << "nb_E_max=" << nb_E_max << endl;
	int *E_freq;
	E_freq = NEW_int(nb_E_max + 1);
	Int_vec_zero(E_freq, nb_E_max + 1);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			E_freq[nb_E]++;
			}
		}



	cout << "E_freq=";
	Int_vec_print(cout, E_freq, nb_E_max + 1);
	cout << endl;


	E = NEW_int(nb_E_max + 1);
	nb_E_types = 0;

	Table_idx = NEW_int(nb_E_max + 1);
	for (j = 0; j <= nb_E_max; j++) {
		if (E_freq[j]) {
			E[nb_E_types] = j;
			Table_idx[j] = nb_E_types;
			nb_E_types++;
			}
		else {
			Table_idx[j] = -1;
			}
		}


	Table = NEW_lint(nb_Q * nb_E_types);
	orbiter_kernel_system::Orbiter->Lint_vec->zero(Table, nb_Q * nb_E_types);
	for (i = 0; i < nb_fields; i++) {
		q = Q[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			idx = Table_idx[nb_E];
			Table[i * nb_E_types + idx]++;
			}
		}
	cout << "Table:" << endl;
	orbiter_kernel_system::Orbiter->Lint_vec->matrix_print(Table, nb_Q, nb_E_types);

	FREE_int(Table_idx);
}



void callback_surface_domain_sstr_line_label(std::stringstream &sstr, long int pt, void *data)
{
	surface_domain *D = (surface_domain *) data;

	//cout << "callback_surface_domain_sstr_line_label pt=" << pt << endl;
	D->sstr_line_label(sstr, pt);
}




}}}

