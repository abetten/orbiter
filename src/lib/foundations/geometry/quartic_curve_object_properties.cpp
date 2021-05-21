/*
 * quartic_curve_object_properties.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */






#include "foundations.h"


using namespace std;



namespace orbiter {
namespace foundations {


quartic_curve_object_properties::quartic_curve_object_properties()
{
	QO = NULL;
}

quartic_curve_object_properties::~quartic_curve_object_properties()
{

}

void quartic_curve_object_properties::init(quartic_curve_object *QO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_properties::init" << endl;
	}
	quartic_curve_object_properties::QO = QO;
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
		cout << "quartic_curve_object_properties::report_properties_simple before print_general" << endl;
	}
	print_general(ost);


}

void quartic_curve_object_properties::print_equation(std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the quartic curve ";
	ost << " is :" << endl;

	QO->Dom->print_equation_with_line_breaks_tex(ost, QO->eqn15);

	Orbiter->Int_vec.print(ost, QO->eqn15, 15);
	ost << "\\\\" << endl;

	long int rk;

	QO->F->PG_element_rank_modified_lint(QO->eqn15, 1, 15, rk);
	ost << "The point rank of the equation over GF$(" << QO->F->q << ")$ is " << rk << "\\\\" << endl;

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
#if 0
	ost << "\\mbox{Number of singular points} & " << nb_singular_pts << "\\\\" << endl;
	ost << "\\hline" << endl;
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




}}

