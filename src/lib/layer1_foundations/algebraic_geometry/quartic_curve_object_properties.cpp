/*
 * quartic_curve_object_properties.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */






#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


quartic_curve_object_properties::quartic_curve_object_properties()
{
	QO = NULL;
	pts_on_lines = NULL;
	f_is_on_line = NULL;
	Bitangent_line_type = NULL;
	//line_type_distribution[3];
	lines_on_point = NULL;
	Point_type = NULL;
	f_fullness_has_been_established = FALSE;
	f_is_full = FALSE;
	nb_Kovalevski = 0;
	nb_Kovalevski_on = 0;
	nb_Kovalevski_off = 0;
	Kovalevski_point_idx = NULL;
	Kovalevski_points = NULL;
	Pts_off = NULL;
	nb_pts_off = 0;
	pts_off_on_lines = NULL;
	f_is_on_line2 = NULL;
	lines_on_points_off = NULL;
	Point_off_type = NULL;
	gradient = NULL;

	singular_pts = NULL;

	nb_singular_pts = 0;
	nb_non_singular_pts = 0;

	tangent_line_rank_global = NULL;
	tangent_line_rank_dual = NULL;


}

quartic_curve_object_properties::~quartic_curve_object_properties()
{
	if (pts_on_lines) {
		FREE_OBJECT(pts_on_lines);
	}
	if (f_is_on_line) {
		FREE_int(f_is_on_line);
	}
	if (Bitangent_line_type) {
		FREE_OBJECT(Bitangent_line_type);
	}
	if (lines_on_point) {
		FREE_OBJECT(lines_on_point);
	}
	if (Point_type) {
		FREE_OBJECT(Point_type);
	}
	if (Kovalevski_point_idx) {
		FREE_int(Kovalevski_point_idx);
	}
	if (Kovalevski_points) {
		FREE_lint(Kovalevski_points);
	}
	if (Pts_off) {
		FREE_lint(Pts_off);
	}
	if (pts_off_on_lines) {
		FREE_OBJECT(pts_off_on_lines);
	}
	if (f_is_on_line2) {
		FREE_int(f_is_on_line2);
	}
	if (lines_on_points_off) {
		FREE_OBJECT(lines_on_points_off);
	}
	if (Point_off_type) {
		FREE_OBJECT(Point_off_type);
	}
	if (gradient) {
		FREE_int(gradient);
	}
	if (singular_pts) {
		FREE_lint(singular_pts);
	}

	if (tangent_line_rank_global) {
		FREE_lint(tangent_line_rank_global);
	}
	if (tangent_line_rank_dual) {
		FREE_lint(tangent_line_rank_dual);
	}
}

void quartic_curve_object_properties::init(quartic_curve_object *QO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::init" << endl;
	}
	quartic_curve_object_properties::QO = QO;




	if (QO->f_has_bitangents) {
		if (f_v) {
			cout << "quartic_curve_object_properties::init before points_on_curve_on_lines" << endl;
		}
		points_on_curve_on_lines(verbose_level);
		if (f_v) {
			cout << "quartic_curve_object_properties::init after points_on_curve_on_lines" << endl;
		}
	}


	if (f_v) {
		cout << "quartic_curve_object_properties::init before compute_singular_points_and_tangent_lines" << endl;
	}
	compute_singular_points_and_tangent_lines(verbose_level);
	if (f_v) {
		cout << "quartic_curve_object_properties::init after compute_singular_points_and_tangent_lines" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::init done" << endl;
	}
}


void quartic_curve_object_properties::create_summary_file(std::string &fname,
		std::string &surface_label, std::string &col_postfix, int verbose_level)
{
#if 0
	string col_lab_surface_label;
	string col_lab_nb_lines;
	string col_lab_nb_points;
	string col_lab_nb_singular_points;
	string col_lab_nb_Eckardt_points;
	string col_lab_nb_double_points;
	string col_lab_nb_Single_points;
	string col_lab_nb_pts_not_on_lines;
	string col_lab_nb_Hesse_planes;
	string col_lab_nb_axes;


	col_lab_surface_label.assign("Surface");


	col_lab_nb_lines.assign("#L");
	col_lab_nb_lines.append(col_postfix);

	col_lab_nb_points.assign("#P");
	col_lab_nb_points.append(col_postfix);

	col_lab_nb_singular_points.assign("#S");
	col_lab_nb_singular_points.append(col_postfix);

	col_lab_nb_Eckardt_points.assign("#E");
	col_lab_nb_Eckardt_points.append(col_postfix);

	col_lab_nb_double_points.assign("#D");
	col_lab_nb_double_points.append(col_postfix);

	col_lab_nb_Single_points.assign("#U");
	col_lab_nb_Single_points.append(col_postfix);

	col_lab_nb_pts_not_on_lines.assign("#OFF");
	col_lab_nb_pts_not_on_lines.append(col_postfix);

	col_lab_nb_Hesse_planes.assign("#H");
	col_lab_nb_Hesse_planes.append(col_postfix);

	col_lab_nb_axes.assign("#AX");
	col_lab_nb_axes.append(col_postfix);

#if 0
	SO->nb_lines;

	SO->nb_pts;

	nb_singular_pts;

	nb_Eckardt_points;

	nb_Double_points;

	nb_Single_points;

	nb_pts_not_on_lines;

	nb_Hesse_planes;

	nb_axes;
#endif


	file_io Fio;

	{
		ofstream f(fname);

		f << col_lab_surface_label << ",";
		f << col_lab_nb_lines << ",";
		f << col_lab_nb_points << ",";
		f << col_lab_nb_singular_points << ",";
		f << col_lab_nb_Eckardt_points << ",";
		f << col_lab_nb_double_points << ",";
		f << col_lab_nb_Single_points << ",";
		f << col_lab_nb_pts_not_on_lines << ",";
		f << col_lab_nb_Hesse_planes << ",";
		f << col_lab_nb_axes << ",";
		f << endl;

		f << surface_label << ",";
		f << SO->nb_lines << ",";
		f << SO->nb_pts << ",";
		f << nb_singular_pts << ",";
		f << nb_Eckardt_points << ",";
		f << nb_Double_points << ",";
		f << nb_Single_points << ",";
		f << nb_pts_not_on_lines << ",";
		f << nb_Hesse_planes << ",";
		f << nb_axes << ",";
		f << endl;

		f << "END" << endl;
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

}

void quartic_curve_object_properties::report_properties_simple(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple before print_equation" << endl;
	}
	print_equation(ost);

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple before print_gradient" << endl;
	}
	print_gradient(ost);

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple before print_general" << endl;
	}
	print_general(ost);

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple before print_points" << endl;
	}
	print_points(ost);



	if (pts_on_lines) {
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple before print_lines_with_points_on_them" << endl;
		}
		print_lines_with_points_on_them(ost, QO->bitangents28, 28, pts_on_lines);
	}
	else {
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple before print_bitangents" << endl;
		}
		print_bitangents(ost);
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple before report_bitangent_line_type" << endl;
	}
	report_bitangent_line_type(ost);

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple done" << endl;
	}
}

void quartic_curve_object_properties::print_equation(std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the quartic curve ";
	ost << " is :" << endl;

	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;
	QO->Dom->print_equation_with_line_breaks_tex(ost, QO->eqn15);
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	ost << "$$" << endl;
	Int_vec_print(ost, QO->eqn15, 15);
	ost << "$$" << endl;

#if 0
	long int rk;

	QO->F->PG_element_rank_modified_lint(QO->eqn15, 1, 15, rk);
	ost << "The point rank of the equation over GF$(" << QO->F->q << ")$ is " << rk << "\\\\" << endl;
#endif

	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}

void quartic_curve_object_properties::print_gradient(std::ostream &ost)
{
	int i;

	ost << "\\subsection*{The gradient}" << endl;
	ost << "The gradient of the quartic curve ";
	ost << " is :" << endl;

	for (i = 0; i < 3; i++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{c}" << endl;
		QO->Dom->print_gradient_with_line_breaks_tex(ost, gradient + i * QO->Dom->Poly3_3->get_nb_monomials());
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;

		ost << "$$" << endl;
		Int_vec_print(ost, gradient + i * QO->Dom->Poly3_3->get_nb_monomials(), QO->Dom->Poly3_3->get_nb_monomials());
		ost << "$$" << endl;
	}

#if 0
	long int rk;

	QO->F->PG_element_rank_modified_lint(QO->eqn15, 1, 15, rk);
	ost << "The point rank of the equation over GF$(" << QO->F->q << ")$ is " << rk << "\\\\" << endl;
#endif

	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}



void quartic_curve_object_properties::print_general(std::ostream &ost)
{
	ost << "\\subsection*{General information}" << endl;


	int nb_bitangents;


	if (QO->f_has_bitangents) {
		nb_bitangents = 28;
	}
	else {
		nb_bitangents = 0;
	}

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of bitangents} & " << nb_bitangents << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points} & " << QO->nb_pts << "\\\\" << endl;
	ost << "\\hline" << endl;

	if (f_fullness_has_been_established) {
		if (f_is_full) {
			ost << "\\mbox{Fullness} &  \\mbox{is full}\\\\" << endl;
			ost << "\\hline" << endl;
		}
		else {
			ost << "\\mbox{Fullness} &  \\mbox{not full}\\\\" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\mbox{Number of Kovalevski points} & " << nb_Kovalevski << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Bitangent line type $(a_0,a_1,a_2)$} & ";
	ost << "(";
	ost << line_type_distribution[0];
	ost << "," << endl;
	ost << line_type_distribution[1];
	ost << "," << endl;
	ost << line_type_distribution[2];
	ost << ")";
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of singular points} & " << nb_singular_pts << "\\\\" << endl;
	ost << "\\hline" << endl;



#if 0
	ost << "\\mbox{Number of Eckardt points} & " << nb_Eckardt_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of double points} & " << nb_Double_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of single points} & " << nb_Single_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points off lines} & " << nb_pts_not_on_lines << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of Hesse planes} & " << nb_Hesse_planes << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of axes} & " << nb_axes << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of points on lines} & ";
	Type_pts_on_lines->print_naked_tex(ost, TRUE);
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of lines on points} & ";
	Type_lines_on_point->print_naked_tex(ost, TRUE);
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
#endif


	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
#if 0
	ost << "Points on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
	ost << "Lines on points:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
#endif
}


void quartic_curve_object_properties::print_points(std::ostream &ost)
{
	ost << "\\subsection*{All points on the curve}" << endl;

	cout << "quartic_curve_object_properties::print_points before print_all_points" << endl;
	print_all_points(ost);

#if 0
	ost << "\\subsubsection*{Eckardt Points}" << endl;
	cout << "quartic_curve_object_properties::print_points before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);

	ost << "\\subsubsection*{Singular Points}" << endl;
	cout << "quartic_curve_object_properties::print_points before print_singular_points" << endl;
	print_singular_points(ost);

	ost << "\\subsubsection*{Double Points}" << endl;
	cout << "quartic_curve_object_properties::print_points before print_double_points" << endl;
	print_double_points(ost);

	ost << "\\subsubsection*{Points on lines}" << endl;
	cout << "quartic_curve_object_properties::print_points before print_points_on_lines" << endl;
	print_points_on_lines(ost);

	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	cout << "quartic_curve_object_properties::print_points before print_points_on_surface_but_not_on_a_line" << endl;
	print_points_on_surface_but_not_on_a_line(ost);
#endif
}



void quartic_curve_object_properties::print_all_points(std::ostream &ost)
{
	//latex_interface L;
	//int i;
	//int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << QO->nb_pts << " points:\\\\" << endl;

	if (QO->nb_pts < 1000) {
		//ost << "$$" << endl;
		//L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, TRUE /* f_tex */);
		//ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the quartic curve are:\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i, j;
		int v[3];
		int a;
		long int b;

		for (i = 0; i < QO->nb_pts; i++) {
			QO->Dom->unrank_point(v, QO->Pts[i]);
			ost << i << " : $P_{" << QO->Pts[i] << "}=";
			Int_vec_print_fully(ost, v, 3);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		ost << "The points by rank are: " << endl;
		Lint_vec_print_fully(ost, QO->Pts, QO->nb_pts);
		ost << "\\\\" << endl;


		algebraic_geometry::schlaefli_labels *Labels;

		Labels = NEW_OBJECT(algebraic_geometry::schlaefli_labels);
		if (FALSE) {
			cout << "quartic_curve_object_properties::print_all_points before Labels->init" << endl;
		}
		Labels->init(0 /*verbose_level*/);
		if (FALSE) {
			cout << "quartic_curve_object_properties::print_all_points after Labels->init" << endl;
		}



		ost << "The Kovalevski points are: \\\\" << endl;
		for (i = 0; i < nb_Kovalevski; i++) {
			a = Kovalevski_point_idx[i];
			QO->Dom->unrank_point(v, Kovalevski_points[i]);
			ost << i << " : $P_{" << Kovalevski_points[i] << "}=";
			Int_vec_print_fully(ost, v, 3);

			ost << " = ";

			for (j = 0; j < 4; j++) {
				b = lines_on_points_off->Sets[a][j];
				//ost << "\\ell_{" << b << "}";
				ost << Labels->Line_label_tex[b];
				if (j < 4 - 1) {
					ost << " \\cap ";
				}
			}
			ost << "$\\\\" << endl;
		}

		FREE_OBJECT(Labels);

		ost << "The Kovalevski points by rank are: " << endl;
		Lint_vec_print_fully(ost, Kovalevski_points, nb_Kovalevski);
		ost << "\\\\" << endl;

		ost << "The points off the curve are: \\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_pts_off; i++) {
			QO->Dom->unrank_point(v, Pts_off[i]);
			ost << i << " : $P_{" << Pts_off[i] << "}=";
			Int_vec_print_fully(ost, v, 3);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		Lint_vec_print_fully(ost, Pts_off, nb_pts_off);
		ost << "\\\\" << endl;

	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void quartic_curve_object_properties::print_bitangents(std::ostream &ost)
{
	if (QO->f_has_bitangents) {
		ost << "\\subsection*{The 28 Bitangents}" << endl;
		QO->Dom->print_lines_tex(ost, QO->bitangents28, 28);
		ost << "Curve Points on Bitangents:\\\\" << endl;
		pts_on_lines->print_table_tex(ost);
	}
}

void quartic_curve_object_properties::print_lines_with_points_on_them(std::ostream &ost,
		long int *Lines, int nb_lines, data_structures::set_of_sets *SoS)
{
	int i, j, h;
	int verbose_level = 1;
	long int a;
	orbiter_kernel_system::latex_interface L;
	int w[3];
	int coeff_out[5];
	int Basis[6];

	ost << "The lines and their points of contact are:\\\\" << endl;
	//ost << "\\begin{multicols}{2}" << endl;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		QO->Dom->P->Grass_lines->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$" << endl;

		ost << QO->Dom->Schlaefli->Line_label_tex[i];
		//ost << "\\ell_{" << i << "}";

#if 0
		if (nb_lines == 27) {
			ost << " = " << Schlaefli->Line_label_tex[i];
		}
#endif
		ost << " = " << endl;
		//print_integer_matrix_width(cout,
		// P->Grass_lines->M, k, n, n, F->log10_of_q + 1);
		QO->Dom->P->Grass_lines->latex_matrix(ost, QO->Dom->P->Grass_lines->M);


		//print_integer_matrix_tex(ost, P->Grass_lines->M, 2, 4);
		//ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "_{" << Lines[i] << "}" << endl;

		if (QO->Dom->F->e > 1) {
			ost << "=" << endl;
			ost << "\\left[" << endl;
			L.print_integer_matrix_tex(ost, QO->Dom->P->Grass_lines->M, 2, 3);
			ost << "\\right]_{" << Lines[i] << "}" << endl;
		}

		for (j = 0; j < SoS->Set_size[i]; j++) {
			a = SoS->Sets[i][j];
			QO->Dom->unrank_point(w, QO->Pts[a]);
			ost << "P_{" << QO->Pts[a] << "}";
			ost << "=\\bP";
			Int_vec_print(ost, w, 3);

			Int_vec_copy(QO->Dom->P->Grass_lines->M, Basis, 6);

			QO->Dom->F->Linear_algebra->adjust_basis(Basis, w,
					3 /* n */, 2 /* k */, 1 /* d */, verbose_level);

			QO->Dom->Poly4_3->substitute_line(
					QO->eqn15 /* int *coeff_in */, coeff_out,
					Basis /* int *Pt1_coeff */, Basis + 3,
					0 /*verbose_level*/);
			// coeff_in[nb_monomials], coeff_out[degree + 1]
			for (h = 0; h <= 4; h++) {
				if (coeff_out[h]) {
					break;
				}
			}

			ost << "\\;" << h << " \\times ";

			if (j < SoS->Set_size[i] - 1) {
				ost << ", ";
			}

		}
		ost << "$\\\\" << endl;
	}
	//ost << "\\end{multicols}" << endl;
	ost << "Rank of lines: ";
	Lint_vec_print(ost, Lines, nb_lines);
	ost << "\\\\" << endl;

}



#if 0
void quartic_curve_object_properties::print_bitangents_with_points_on_them(std::ostream &ost)
{
	latex_interface L;

	if (QO->f_has_bitangents) {
		ost << "\\subsection*{The 28 bitangents with points on them}" << endl;
		int i; //, j;
		//int pt;

		for (i = 0; i < 28; i++) {
			//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
			QO->Dom->P->Grass_lines->unrank_lint(QO->bitangents28[i], 0 /*verbose_level*/);
			ost << "$$" << endl;
			ost << "\\ell_{" << i << "} ";
#if 0
			if (SO->nb_lines == 27) {
				ost << " = " << SO->Surf->Schlaefli->Line_label_tex[i];
			}
#endif
			ost << " = \\left[" << endl;
			//print_integer_matrix_width(cout, Gr->M,
			// k, n, n, F->log10_of_q + 1);
			L.print_integer_matrix_tex(ost, QO->Dom->P->Grass_lines->M, 2, 4);
			ost << "\\right]_{" << QO->bitangents28[i] << "}" << endl;
			ost << "$$" << endl;

#if 0
			ost << "which contains the point set " << endl;
			ost << "$$" << endl;
			ost << "\\{ P_{i} \\mid i \\in ";
			L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
					pts_on_lines->Set_size[i]);
			ost << "\\}." << endl;
			ost << "$$" << endl;

			{
				std::vector<long int> plane_ranks;

				SO->Surf->P->planes_through_a_line(
						SO->Lines[i], plane_ranks,
						0 /*verbose_level*/);

				// print the tangent planes associated with the points on the line:
				ost << "The tangent planes associated with the points on this line are:\\\\" << endl;
				for (j = 0; j < pts_on_lines->Set_size[i]; j++) {

					int w[4];

					pt = pts_on_lines->Sets[i][j];
					ost << j << " : " << pt << " : ";
					SO->Surf->unrank_point(w, SO->Pts[pt]);
					Int_vec_print(ost, w, 4);
					ost << " : ";
					if (tangent_plane_rank_global[pt] == -1) {
						ost << " is singular\\\\" << endl;
					}
					else {
						ost << tangent_plane_rank_global[pt] << "\\\\" << endl;
					}
				}
				ost << "The planes in the pencil through the line are:\\\\" << endl;
				for (j = 0; j < plane_ranks.size(); j++) {
					ost << j << " : " << plane_ranks[j] << "\\\\" << endl;

				}
			}
#endif
		}
	}
}
#endif

void quartic_curve_object_properties::points_on_curve_on_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines" << endl;
	}

	if (!QO->f_has_bitangents) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines !QO->f_has_bitangents" << endl;
		exit(1);
	}
	combinatorics::combinatorics_domain Combi;


	Pts_off = NEW_lint(QO->Dom->P->N_points);

	Combi.set_complement_lint(QO->Pts, QO->nb_pts, Pts_off /*complement*/,
			nb_pts_off, QO->Dom->P->N_points);

	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines before "
				"Surf->compute_points_on_lines" << endl;
	}
	QO->Dom->compute_points_on_lines(QO->Pts, QO->nb_pts,
		QO->bitangents28, 28,
		pts_on_lines,
		f_is_on_line,
		0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines after "
				"Surf->compute_points_on_lines" << endl;
	}

	pts_on_lines->sort();

	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines pts_on_lines:" << endl;
		pts_on_lines->print_table();
	}

	Bitangent_line_type = NEW_OBJECT(data_structures::tally);
	Bitangent_line_type->init_lint(pts_on_lines->Set_size,
		pts_on_lines->nb_sets, FALSE, 0);

	if (f_v) {
		cout << "Line type:" << endl;
		Bitangent_line_type->print_naked_tex(cout, TRUE);
		cout << endl;
	}

	int i, j;

	for (i = 0; i <= 2; i++) {
		j = Bitangent_line_type->determine_class_by_value(i);
		if (j == -1) {
			line_type_distribution[i] = 0;
		}
		else {
			line_type_distribution[i] = Bitangent_line_type->type_len[j];
		}
	}


	pts_on_lines->dualize(lines_on_point, 0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines lines_on_point:" << endl;
		lines_on_point->print_table();
	}

	Point_type = NEW_OBJECT(data_structures::tally);
	Point_type->init_lint(lines_on_point->Set_size,
		lines_on_point->nb_sets, FALSE, 0);
	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines type of lines_on_point:" << endl;
		Point_type->print_naked_tex(cout, TRUE);
		cout << endl;
	}

	int f, l, a, b;
	vector<int> K;


	f_is_full = TRUE;
	nb_Kovalevski_on = 0;
	for (i = Point_type->nb_types - 1; i >= 0; i--) {
		f = Point_type->type_first[i];
		l = Point_type->type_len[i];
		a = Point_type->data_sorted[f];
		if (a == 0) {
			f_is_full = FALSE;
		}
		if (a == 4) {
			nb_Kovalevski_on += l;
			for (j = 0; j < l; j++) {
				b = Point_type->sorting_perm_inv[f + j];
				K.push_back(b);
			}
		}
	}
	f_fullness_has_been_established = TRUE;



	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines before "
				"Surf->compute_points_on_lines for complement" << endl;
	}
	QO->Dom->compute_points_on_lines(Pts_off, nb_pts_off,
		QO->bitangents28, 28,
		pts_off_on_lines,
		f_is_on_line2,
		0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines after "
				"Surf->compute_points_on_lines" << endl;
	}

	pts_off_on_lines->sort();

	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines pts_off_on_lines:" << endl;
		pts_off_on_lines->print_table();
	}

	pts_off_on_lines->dualize(lines_on_points_off, 0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines lines_on_point:" << endl;
		lines_on_point->print_table();
	}

	Point_off_type = NEW_OBJECT(data_structures::tally);
	Point_off_type->init_lint(lines_on_points_off->Set_size,
			lines_on_points_off->nb_sets, FALSE, 0);
	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines type of lines_on_point off:" << endl;
		Point_off_type->print_naked_tex(cout, TRUE);
		cout << endl;
	}


	nb_Kovalevski_off = 0;
	for (i = Point_off_type->nb_types - 1; i >= 0; i--) {
		f = Point_off_type->type_first[i];
		l = Point_off_type->type_len[i];
		a = Point_off_type->data_sorted[f];
		if (a == 4) {
			nb_Kovalevski_off += l;
			for (j = 0; j < l; j++) {
				b = Point_off_type->sorting_perm_inv[f + j];
				K.push_back(b);
			}
		}
	}
	nb_Kovalevski = nb_Kovalevski_on + nb_Kovalevski_off;
	if (K.size() != nb_Kovalevski) {
		cout << "K.size() != nb_Kovalevski" << endl;
		cout << "K.size()=" << K.size() << endl;
		cout << "nb_Kovalevski=" << nb_Kovalevski << endl;
		exit(1);
	}

	Kovalevski_point_idx = NEW_int(nb_Kovalevski);
	Kovalevski_points = NEW_lint(nb_Kovalevski);
	for (j = 0; j < nb_Kovalevski; j++) {
		Kovalevski_point_idx[j] = K[j];
		Kovalevski_points[j] = Pts_off[K[j]];
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::points_on_curve_on_lines done" << endl;
	}
}

void quartic_curve_object_properties::report_bitangent_line_type(std::ostream &ost)
{
	if (QO->f_has_bitangents) {
		orbiter_kernel_system::latex_interface L;
		int i;
		long int a;
		int v[3];


		ost << "Line type: $" << endl;
		Bitangent_line_type->print_naked_tex(ost, TRUE);
		ost << "$\\\\" << endl;

		ost << "$$" << endl;
		Bitangent_line_type->print_array_tex(ost, TRUE /* f_backwards */);
		ost << "$$" << endl;


		ost << "point types: $" << endl;
		Point_type->print_naked_tex(ost, TRUE);
		ost << "$\\\\" << endl;

		ost << "$$" << endl;
		Point_type->print_array_tex(ost, TRUE /* f_backwards */);
		ost << "$$" << endl;

		ost << "point types for points off the curve: $" << endl;
		Point_off_type->print_naked_tex(ost, TRUE);
		ost << "$\\\\" << endl;

		ost << "$$" << endl;
		Point_off_type->print_array_tex(ost, TRUE /* f_backwards */);
		ost << "$$" << endl;

		ost << "Lines on points off the curve:\\\\" << endl;
		//lines_on_points_off->print_table_tex(ost);
		for (i = 0; i < lines_on_points_off->nb_sets; i++) {

			a = Pts_off[i];
			QO->Dom->unrank_point(v, a);


			ost << "Off point " << i << " = $P_{" << a << "} = ";

			Int_vec_print(ost, v, 3);

			ost << "$ lies on " << lines_on_points_off->Set_size[i] << " bisecants : ";
			L.lint_set_print_tex(ost, lines_on_points_off->Sets[i], lines_on_points_off->Set_size[i]);
			ost << "\\\\" << endl;
		}


	}
}

void quartic_curve_object_properties::compute_gradient(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object_properties::compute_gradient before QO->Dom->compute_gradient" << endl;
	}

	QO->Dom->compute_gradient(QO->eqn15, gradient, verbose_level);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_gradient done" << endl;
	}
}



void quartic_curve_object_properties::compute_singular_points_and_tangent_lines(int verbose_level)
// a singular point is a point where all partials vanish
// We compute the set of singular points into Pts[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines" << endl;
	}
	int h, i;
	long int rk;
	int nb_eqns = 3;
	int v[3];
	int w[3];


	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines before compute_gradient" << endl;
	}
	compute_gradient(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines after compute_gradient" << endl;
	}

	nb_singular_pts = 0;
	nb_non_singular_pts = 0;

	singular_pts = NEW_lint(QO->nb_pts);
	tangent_line_rank_global = NEW_lint(QO->nb_pts);
	tangent_line_rank_dual = NEW_lint(QO->nb_pts);
	for (h = 0; h < QO->nb_pts; h++) {
		if (f_vv) {
			cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
					"h=" << h << " / " << QO->nb_pts << endl;
		}
		rk = QO->Pts[h];
		if (f_vv) {
			cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
					"rk=" << rk << endl;
		}
		QO->Dom->unrank_point(v, rk);
		if (f_vv) {
			cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
					"v=";
			Int_vec_print(cout, v, 4);
			cout << endl;
		}
		for (i = 0; i < nb_eqns; i++) {
			if (f_vv) {
				cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
						"gradient i=" << i << " / " << nb_eqns << endl;
			}
			if (FALSE) {
				cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
						"gradient " << i << " = ";
				Int_vec_print(cout,
						gradient + i * QO->Dom->Poly3_3->get_nb_monomials(),
						QO->Dom->Poly3_3->get_nb_monomials());
				cout << endl;
			}
			w[i] = QO->Dom->Poly3_3->evaluate_at_a_point(
					gradient + i * QO->Dom->Poly3_3->get_nb_monomials(), v);
			if (f_vv) {
				cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
						"value = " << w[i] << endl;
			}
		}
		for (i = 0; i < nb_eqns; i++) {
			if (w[i]) {
				break;
			}
		}

		if (i == nb_eqns) {
			singular_pts[nb_singular_pts++] = rk;
			tangent_line_rank_global[h] = -1;
		}
		else {
			long int line_rk;

			line_rk = QO->Dom->P->line_rank_using_dual_coordinates_in_plane(
					w /* eqn3 */,
					0 /* verbose_level*/);
			tangent_line_rank_global[h] = line_rk;
			tangent_line_rank_dual[nb_non_singular_pts++] =
					QO->Dom->P->dual_rank_of_line_in_plane(
							line_rk, 0 /* verbose_level*/);
		}
	}

	data_structures::sorting Sorting;
	int nb_tangent_lines;

	nb_tangent_lines = nb_non_singular_pts;

	Sorting.lint_vec_sort_and_remove_duplicates(tangent_line_rank_dual, nb_tangent_lines);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines nb_tangent_lines " << nb_tangent_lines << endl;
	}

#if 0
	string fname_tangents;
	file_io Fio;

	fname_tangents.assign("tangents.txt");

	Fio.write_set_to_file_lint(fname_tangents,
			tangent_plane_rank_dual, nb_tangent_planes, verbose_level);

	if (f_v) {
		cout << "Written file " << fname_tangents << " of size " << Fio.file_size(fname_tangents) << endl;
	}
#endif


	int *Kernel;
	int *w1;
	int *w2;
	int r, ns;

	Kernel = NEW_int(QO->Dom->Poly2_3->get_nb_monomials() * QO->Dom->Poly2_3->get_nb_monomials());
	w1 = NEW_int(QO->Dom->Poly2_3->get_nb_monomials());
	w2 = NEW_int(QO->Dom->Poly2_3->get_nb_monomials());



	QO->Dom->Poly2_3->vanishing_ideal(tangent_line_rank_dual, nb_non_singular_pts,
			r, Kernel, 0 /*verbose_level */);

	ns = QO->Dom->Poly2_3->get_nb_monomials() - r; // dimension of null space
	if (f_v) {
		cout << "The system has rank " << r << endl;
		cout << "The ideal has dimension " << ns << endl;
#if 0
		cout << "and is generated by:" << endl;
		int_matrix_print(Kernel, ns, QO->Dom->Poly2_3->get_nb_monomials());
		cout << "corresponding to the following basis "
				"of polynomials:" << endl;
		for (h = 0; h < ns; h++) {
			SO->Surf->Poly2_3->print_equation(cout, Kernel + h * QO->Dom->Poly2_3->get_nb_monomials());
			cout << endl;
		}
#endif
	}

	FREE_int(Kernel);
	FREE_int(w1);
	FREE_int(w2);


	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines done" << endl;
	}
}

}}}


