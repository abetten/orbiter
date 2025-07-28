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
namespace geometry {
namespace algebraic_geometry {


quartic_curve_object_properties::quartic_curve_object_properties()
{
	Record_birth();
	QO = NULL;

	Kovalevski = NULL;

	gradient = NULL;

	singular_pts = NULL;

	nb_singular_pts = 0;
	nb_non_singular_pts = 0;

	tangent_line_rank_global = NULL;
	tangent_line_rank_dual = NULL;


	dual_of_bitangents = NULL;
	Kernel = NULL;

}

quartic_curve_object_properties::~quartic_curve_object_properties()
{
	Record_death();
	if (Kovalevski) {
		FREE_OBJECT(Kovalevski);
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
	if (dual_of_bitangents) {
		FREE_lint(dual_of_bitangents);
	}
	if (Kernel) {
		FREE_OBJECT(Kernel);
	}
}

void quartic_curve_object_properties::init(
		quartic_curve_object *QO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::init" << endl;
	}
	quartic_curve_object_properties::QO = QO;




	if (QO->f_has_bitangents) {


		Kovalevski = NEW_OBJECT(kovalevski_points);

		Kovalevski->init(QO, verbose_level);

		if (f_v) {
			cout << "quartic_curve_object_properties::init "
					"before Kovalevski->compute_Kovalevski_points" << endl;
		}
		Kovalevski->compute_Kovalevski_points(verbose_level);
		if (f_v) {
			cout << "quartic_curve_object_properties::init "
					"after Kovalevski->compute_Kovalevski_points" << endl;
		}
	}


	if (QO->f_has_bitangents) {

		dual_of_bitangents = NEW_lint(28);

		if (QO->get_nb_lines() != 28) {
			cout << "quartic_curve_object_properties::init QO->get_nb_lines() != 28" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "quartic_curve_object_properties::init "
					"before dualize_lines_to_points" << endl;
		}

		QO->Dom->P->Plane->dualize_lines_to_points(
			QO->get_nb_lines(),
			QO->get_lines(),
			dual_of_bitangents,
			verbose_level);

		if (f_v) {
			cout << "quartic_curve_object_properties::init "
					"after dualize_lines_to_points" << endl;
		}

		int rk;

		if (f_v) {
			cout << "quartic_curve_object_properties::init "
					"before vanishing_ideal" << endl;
		}
		QO->Dom->Poly4_3->vanishing_ideal(
				dual_of_bitangents, 28 /*nb_pts*/,
				rk,
				Kernel,
				verbose_level - 1);
		if (f_v) {
			cout << "quartic_curve_object_properties::init "
					"after vanishing_ideal" << endl;
		}

	}

	if (f_v) {
		cout << "quartic_curve_object_properties::init "
				"before compute_singular_points_and_tangent_lines" << endl;
	}
	compute_singular_points_and_tangent_lines(verbose_level);
	if (f_v) {
		cout << "quartic_curve_object_properties::init "
				"after compute_singular_points_and_tangent_lines" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::init done" << endl;
	}
}


void quartic_curve_object_properties::create_summary_file(
		std::string &fname,
		std::string &surface_label,
		std::string &col_postfix,
		int verbose_level)
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


	col_lab_surface_label = "Surface";


	col_lab_nb_lines = "#L" + col_postfix;

	col_lab_nb_points = "#P" + col_postfix;

	col_lab_nb_singular_points = "#S" + col_postfix;

	col_lab_nb_Eckardt_points = "#E" + col_postfix;

	col_lab_nb_double_points = "#D" + col_postfix;

	col_lab_nb_Single_points = "#U" + col_postfix;

	col_lab_nb_pts_not_on_lines = "#OFF" + col_postfix;

	col_lab_nb_Hesse_planes = "#H" + col_postfix;

	col_lab_nb_axes = "#AX" + col_postfix;

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

void quartic_curve_object_properties::report_properties_simple(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"before print_equation" << endl;
	}
	print_equation(ost);
	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"after print_equation" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"before print_gradient" << endl;
	}
	print_gradient(ost);
	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"after print_gradient" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"before print_general" << endl;
	}
	print_general(ost);
	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"after print_general" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"before print_points" << endl;
	}
	print_points(ost, verbose_level);
	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"after print_points" << endl;
	}



	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"before Kovalevski->print_lines_with_points_on_them" << endl;
	}
	if (Kovalevski) {
		Kovalevski->print_lines_with_points_on_them(ost);
	}
	else {
		cout << "no Kovalevski" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"after Kovalevski->print_lines_with_points_on_them" << endl;
	}

#if 0
	if (pts_on_lines) {
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"before print_lines_with_points_on_them" << endl;
		}
		print_lines_with_points_on_them(ost, QO->bitangents28, 28, pts_on_lines);
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"after print_lines_with_points_on_them" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"before print_bitangents" << endl;
		}
		print_bitangents(ost);
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"after print_bitangents" << endl;
		}
	}
#endif

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"before Kovalevski->report_bitangent_line_type" << endl;
	}
	if (Kovalevski) {
		Kovalevski->report_bitangent_line_type(ost);
	}
	else {
		cout << "no Kovalevski" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"after Kovalevski->report_bitangent_line_type" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_object_properties::report_properties_simple "
				"done" << endl;
	}
}

void quartic_curve_object_properties::print_equation(
		std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the quartic curve ";
	ost << " is :" << endl;

	ost << "$$" << endl;
	ost << "\\begin{array}{c}" << endl;


	int eqn15[15];
	//Int_vec_copy(QO->eqn15, eqn15, 15);
	Int_vec_copy(QO->Variety_object->eqn, eqn15, 15);

	QO->Dom->F->Projective_space_basic->PG_element_normalize_from_front(
			eqn15, 1, 15);


	QO->Dom->print_equation_with_line_breaks_tex(ost, eqn15);
	//QO->Dom->print_equation_with_line_breaks_tex(ost, QO->eqn15);
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;


	ost << "$$" << endl;
	Int_vec_print(ost, eqn15, 15);
	ost << "$$" << endl;


	QO->Variety_object->print_equation_verbatim(
			eqn15,
			ost);


#if 0
	long int rk;

	QO->F->PG_element_rank_modified_lint(QO->eqn15, 1, 15, rk);
	ost << "The point rank of the equation over GF$(" << QO->F->q << ")$ is " << rk << "\\\\" << endl;
#endif





	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}

void quartic_curve_object_properties::print_gradient(
		std::ostream &ost)
{
	int i;

	ost << "\\subsection*{The gradient}" << endl;
	ost << "The gradient of the quartic curve ";
	ost << " is :" << endl;

	for (i = 0; i < 3; i++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{c}" << endl;
		QO->Dom->print_gradient_with_line_breaks_tex(ost,
				gradient + i * QO->Dom->Poly3_3->get_nb_monomials());
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;

		ost << "$$" << endl;
		Int_vec_print(ost,
				gradient + i * QO->Dom->Poly3_3->get_nb_monomials(),
				QO->Dom->Poly3_3->get_nb_monomials());
		ost << "$$" << endl;
	}

#if 0
	long int rk;

	QO->F->PG_element_rank_modified_lint(QO->eqn15, 1, 15, rk);
	ost << "The point rank of the equation over GF$(" << QO->F->q << ")$ is " << rk << "\\\\" << endl;
#endif

	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}



void quartic_curve_object_properties::print_general(
		std::ostream &ost)
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
	ost << "\\mbox{Number of points} & " << QO->get_nb_points() << "\\\\" << endl;
	ost << "\\hline" << endl;

	if (Kovalevski) {
		Kovalevski->print_general(ost);
	}
	else {
		cout << "no Kovalevski" << endl;
	}

	ost << "\\mbox{Number of singular points} & " << nb_singular_pts << "\\\\" << endl;
	ost << "\\hline" << endl;



	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;

}


void quartic_curve_object_properties::print_points(
		std::ostream &ost, int verbose_level)
{
	ost << "\\subsection*{All points on the curve}" << endl;

	cout << "quartic_curve_object_properties::print_points "
			"before print_all_points" << endl;
	print_all_points(ost, verbose_level);

	if (dual_of_bitangents) {
		ost << "\\subsection*{Duals of Bitangents}" << endl;
		print_dual_of_bitangents(ost, verbose_level);

		//r = QO->Dom->Poly4_3->get_nb_monomials() - rk;

		int h, r;
		r = Kernel->m;

		ost << "Dimension of the vanishing ideal is " << r << "\\\\" << endl;
		ost << "\\begin{verbatim}" << endl;
		for (h = 0; h < r; h++) {
			QO->Dom->Poly4_3->print_equation_relaxed(ost, Kernel->M + h * Kernel->n);
			ost << endl;

		}
		ost << "\\end{verbatim}" << endl;

	}

}

void quartic_curve_object_properties::print_dual_of_bitangents(
		std::ostream &ost, int verbose_level)
{
	int i;
	int v[3];

	ost << "dual of bitangents:\\\\" << endl;
	for (i = 0; i < 28; i++) {
		QO->Dom->unrank_point(v, dual_of_bitangents[i]);
		ost << i << " : $P_{" << dual_of_bitangents[i] << "}=";
		Int_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
	}
}

void quartic_curve_object_properties::print_all_points(
		std::ostream &ost, int verbose_level)
{
	//latex_interface L;
	//int i;
	//int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The quartic has " << QO->get_nb_points() << " points:\\\\" << endl;

	if (QO->get_nb_points() < 1000) {
		//ost << "$$" << endl;
		//L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, true /* f_tex */);
		//ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the quartic curve are:\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;
		int v[3];

		for (i = 0; i < QO->get_nb_points(); i++) {
			QO->Dom->unrank_point(v, QO->get_point(i));
			ost << i << " : $P_{" << QO->get_point(i) << "}=";
			Int_vec_print_fully(ost, v, 3);
			ost << "$\\\\" << endl;
		}
		ost << "\\end{multicols}" << endl;
		ost << "The points by rank are: " << endl;
		Lint_vec_print_fully(ost, QO->get_points(), QO->get_nb_points());
		ost << "\\\\" << endl;



		if (Kovalevski) {
			Kovalevski->print_all_points(ost, verbose_level);
		}



	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void quartic_curve_object_properties::print_bitangents(
		std::ostream &ost)
{
	if (QO->f_has_bitangents) {
		ost << "\\subsection*{The 28 Bitangents}" << endl;
		QO->Dom->print_lines_tex(ost, QO->get_lines(), QO->get_nb_lines());

		ost << "Curve Points on Bitangents:\\\\" << endl;
		Kovalevski->pts_on_lines->print_table_tex(ost);
	}
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



void quartic_curve_object_properties::compute_gradient(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object_properties::compute_gradient "
				"before QO->Dom->compute_gradient" << endl;
	}

	if (QO->Variety_object == NULL) {
		cout << "quartic_curve_object_properties::compute_gradient "
				"QO->Variety_object == NULL" << endl;
		exit(1);
	}
	QO->Dom->compute_gradient(
			QO->Variety_object->eqn, gradient, verbose_level);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_gradient done" << endl;
	}
}



void quartic_curve_object_properties::compute_singular_points_and_tangent_lines(
		int verbose_level)
// a singular point is a point where all partials vanish
// We compute the set of singular points into Pts[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines" << endl;
	}
	int h, i;
	long int rk;
	int nb_eqns = 3;
	int v[3];
	int w[3];


	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
				"before compute_gradient" << endl;
	}
	compute_gradient(verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
				"after compute_gradient" << endl;
	}

	nb_singular_pts = 0;
	nb_non_singular_pts = 0;

	singular_pts = NEW_lint(QO->get_nb_points());
	tangent_line_rank_global = NEW_lint(QO->get_nb_points());
	tangent_line_rank_dual = NEW_lint(QO->get_nb_points());
	for (h = 0; h < QO->get_nb_points(); h++) {
		if (f_vv) {
			cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
					"h=" << h << " / " << QO->get_nb_points() << endl;
		}
		rk = QO->get_point(h);
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
			if (false) {
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

			line_rk = QO->Dom->P->Plane->line_rank_using_dual_coordinates_in_plane(
					w /* eqn3 */,
					0 /* verbose_level*/);
			tangent_line_rank_global[h] = line_rk;
			tangent_line_rank_dual[nb_non_singular_pts++] =
					QO->Dom->P->Plane->dual_rank_of_line_in_plane(
							line_rk, 0 /* verbose_level*/);
		}
	}

	other::data_structures::sorting Sorting;
	int nb_tangent_lines;

	nb_tangent_lines = nb_non_singular_pts;

	Sorting.lint_vec_sort_and_remove_duplicates(
			tangent_line_rank_dual, nb_tangent_lines);

	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines "
				"nb_tangent_lines " << nb_tangent_lines << endl;
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


	other::data_structures::int_matrix *Kernel;
	//int *Kernel;
	int *w1;
	int *w2;
	int r, ns;

	//Kernel = NEW_int(QO->Dom->Poly2_3->get_nb_monomials() * QO->Dom->Poly2_3->get_nb_monomials());
	w1 = NEW_int(QO->Dom->Poly2_3->get_nb_monomials());
	w2 = NEW_int(QO->Dom->Poly2_3->get_nb_monomials());


	QO->Dom->Poly2_3->vanishing_ideal(
			tangent_line_rank_dual, nb_non_singular_pts,
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

	FREE_OBJECT(Kernel);
	FREE_int(w1);
	FREE_int(w2);


	if (f_v) {
		cout << "quartic_curve_object_properties::compute_singular_points_and_tangent_lines done" << endl;
	}
}

}}}}


