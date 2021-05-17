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


void surface_domain::print_equation(std::ostream &ost, int *coeffs)
{
	Poly3_4->print_equation(ost, coeffs);
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
	long int i, j, a, u, v;

	ost << "\\begin{array}{cc}" << endl;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 2; j++) {
			a = double_six[j * 6 + i];
			Gr->unrank_lint(a, 0);
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
		Orbiter->Int_vec.print_to_str(str, w, 6);
		Text_wedge[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_wedge[i], str);
		}
	for (i = 0; i < nb_lines; i++) {
		a = Line_rk[i];
		Gr->unrank_lint_here(Basis, a, 0 /* verbose_level */);
		Orbiter->Int_vec.print_to_str(str, Basis, 8);
		Text_line[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_line[i], str);
		}
	for (i = 0; i < nb_lines; i++) {
		a = Klein_rk[i];
		O->unrank_point(w, 1, a, 0 /* verbose_level*/);
			// error corrected: w was v which was v[4], so too short.
			// Aug 25, 2018
		Orbiter->Int_vec.print_to_str(str, w, 6);
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

		int v6[6];

		P->Pluecker_coordinates(Lines[i], v6, 0 /* verbose_level */);

		Orbiter->Int_vec.copy(v6, vv, 6); // mistake found by Alice Hui

		Rk[i] = F->Qplus_rank(vv, 1, 5, 0 /* verbose_level*/);

		ost << "={\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
				<< v6[2] << "," << v6[3] << "," << v6[4]
				<< "," << v6[5] << " ";
		ost << ")_{" << Rk[i] << "}";
		ost << "$$" << endl;
	}
	ost << "Rank of lines: ";
	Orbiter->Lint_vec.print(ost, Lines, nb_lines);
	ost << "\\\\" << endl;
	ost << "Rank of points on Klein quadric: ";
	Orbiter->Lint_vec.print(ost, Rk, nb_lines);
	ost << "\\\\" << endl;

	FREE_lint(Rk);

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
		Orbiter->Int_vec.print_GAP(cout, F_planes + i * 4, 4);
		cout << ", ";
		}
	for (i = 0; i < 3; i++) {
		Orbiter->Int_vec.print_GAP(cout, G_planes + i * 4, 4);
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


void surface_domain::make_table_of_surfaces(int verbose_level)
{


	//int f_v = (verbose_level >= 1);


	char fname[1000];
	char title[1000];
	const char *author = "Orbiter";
	const char *extras_for_preamble = "";

	sprintf(fname, "surfaces_report.tex");
	sprintf(title, "Cubic Surfaces with 27 Lines over Finite Fields");

	{
		ofstream fp(fname);
		latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);


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


		L.foot(fp);
	}

}


void surface_domain::make_table_of_surfaces2(ostream &ost,
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
	const char *stab_order;
	knowledge_base K;
	file_io Fio;


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

	int *Table;
	int *Q;
	int nb_Q;
	int *E;
	int nb_E_types;

	compute_table_E(Q_table, Q_table_len,
			Table, Q, nb_Q, E, nb_E_types, verbose_level);



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

				nb_total = K.cubic_surface_nb_reps(q);
				Ago = NEW_int(nb_reps);
				u = 0;
				for (h = 0; h < nb_total; h++) {
					nb_E = K.cubic_surface_nb_Eckardt_points(q, h);
					if (nb_E != E[j]) {
						continue;
					}
					K.cubic_surface_stab_gens(q, h, data,
							nb_gens, data_size, stab_order);
					sscanf(stab_order, "%d", &go);
					Ago[u++] = go;
				}

				if (u != nb_reps) {
					cout << "u != nb_reps" << endl;
					exit(1);
				}
				tally C;

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

	FREE_int(Table);
	FREE_int(Q);
	FREE_int(E);


}

void surface_domain::table_top(ostream &ost)
{
	ost << "$" << endl;
	ost << "\\begin{array}{|c|c||c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "q & \\mbox{Iso} & \\mbox{Ago} & \\# E & "
			"\\mbox{Comment}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
}

void surface_domain::table_bottom(ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	//ost << "\\quad" << endl;
	ost << "$" << endl;
}

void surface_domain::compute_table_E(
		int *field_orders, int nb_fields,
		int *&Table,
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
	Orbiter->Int_vec.copy(field_orders, Q, nb_Q);

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
	Orbiter->Int_vec.zero(E_freq, nb_E_max + 1);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			E_freq[nb_E]++;
			}
		}



	cout << "E_freq=";
	Orbiter->Int_vec.print(cout, E_freq, nb_E_max + 1);
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


	Table = NEW_int(nb_Q * nb_E_types);
	Orbiter->Int_vec.zero(Table, nb_Q * nb_E_types);
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
	Orbiter->Int_vec.matrix_print(Table, nb_Q, nb_E_types);

	FREE_int(Table_idx);
}



void callback_surface_domain_sstr_line_label(stringstream &sstr, long int pt, void *data)
{
	surface_domain *D = (surface_domain *) data;

	//cout << "callback_surface_domain_sstr_line_label pt=" << pt << endl;
	D->sstr_line_label(sstr, pt);
}



}}
