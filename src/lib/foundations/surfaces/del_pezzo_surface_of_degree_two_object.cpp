/*
 * del_pezzo_surface_of_degree_two_object.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: betten
 */

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


del_pezzo_surface_of_degree_two_object::del_pezzo_surface_of_degree_two_object()
{
	Dom = NULL;

	RHS = NULL;
	Subtrees = NULL;
	Coefficient_vector = NULL;

	Pts = NULL;
	nb_pts = 0;

	Lines = NULL;
	nb_lines = 0;
}

del_pezzo_surface_of_degree_two_object::~del_pezzo_surface_of_degree_two_object()
{
	if (Pts) {
		FREE_lint(Pts);
	}
	if (Lines) {
		FREE_lint(Lines);
	}
}

void del_pezzo_surface_of_degree_two_object::init(del_pezzo_surface_of_degree_two_domain *Dom,
		formula *RHS, syntax_tree_node **Subtrees, int *Coefficient_vector,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::init" << endl;
	}
	del_pezzo_surface_of_degree_two_object::Dom = Dom;
	del_pezzo_surface_of_degree_two_object::RHS = RHS;
	del_pezzo_surface_of_degree_two_object::Subtrees = Subtrees;
	del_pezzo_surface_of_degree_two_object::Coefficient_vector = Coefficient_vector;

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::init done" << endl;
	}
}

void del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines" << endl;
	}

	vector<long int> Points;
	vector<long int> The_Lines;

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines before "
				"Dom->enumerate_points" << endl;
	}
	Dom->enumerate_points(Coefficient_vector,
		Points,
		0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines after "
				"Dom->enumerate_points" << endl;
	}
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines The surface "
				"has " << Points.size() << " points" << endl;
	}
	int i;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}


	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines before "
				"Surf->P->find_lines_which_are_contained" << endl;
	}
	Dom->P->find_lines_which_are_contained(Points, The_Lines, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines after "
				"Surf->P->find_lines_which_are_contained" << endl;
	}

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines The surface "
				"has " << The_Lines.size() << " lines" << endl;
	}


#if 0
	if (F->q == 2) {
		if (f_v) {
			cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines before find_real_lines" << endl;
		}

		find_real_lines(The_Lines, verbose_level);

		if (f_v) {
			cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines after find_real_lines" << endl;
		}
	}
#endif

	nb_lines = The_Lines.size();
	Lines = NEW_lint(nb_lines);
	for (i = 0; i < nb_lines; i++) {
		Lines[i] = The_Lines[i];
	}

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines nb_pts=" << nb_pts << " nb_lines=" << nb_lines << endl;
		cout << "Lines:";
		lint_vec_print(cout, Lines, nb_lines);
		cout << endl;
	}




	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::enumerate_points_and_lines done" << endl;
	}
}

void del_pezzo_surface_of_degree_two_object::create_latex_report(std::string &label, std::string &label_tex, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::create_latex_report" << endl;
	}

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "%s_report.tex", label.c_str());
		fname.assign(str);
		snprintf(title, 1000, "Del Pezzo Surface  %s", label_tex.c_str());
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "del_pezzo_surface_of_degree_two_object::create_latex_report before report_properties" << endl;
			}
			report_properties(ost, verbose_level);
			if (f_v) {
				cout << "del_pezzo_surface_of_degree_two_object::create_latex_report after report_properties" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::create_latex_report done" << endl;
	}
}

void del_pezzo_surface_of_degree_two_object::report_properties(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties" << endl;
	}

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties_simple before print_equation" << endl;
	}
	print_equation(ost);

#if 0
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties before print_general" << endl;
	}
	print_general(ost);
#endif


	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties before print_lines" << endl;
	}
	print_lines(ost);

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties before print_points" << endl;
	}
	print_points(ost);

#if 0
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties print_tritangent_planes" << endl;
	}
	print_tritangent_planes(ost);


	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties "
				"before print_Steiner_and_Eckardt" << endl;
	}
	print_Steiner_and_Eckardt(ost);

	//SOA->SO->print_planes_in_trihedral_pairs(fp);

#if 0
	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties "
				"before print_generalized_quadrangle" << endl;
	}
	print_generalized_quadrangle(ost);
#endif

	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_line_intersection_graph" << endl;
	}
	print_line_intersection_graph(ost);
#endif

	if (f_v) {
		cout << "del_pezzo_surface_of_degree_two_object::report_properties done" << endl;
	}
}

void del_pezzo_surface_of_degree_two_object::print_equation(std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the surface ";
	ost << " is :" << endl;

	Dom->print_equation_with_line_breaks_tex(ost, Coefficient_vector);

	int_vec_print(ost, Coefficient_vector, 15);
	ost << "\\\\" << endl;

	long int rk;

	Dom->F->PG_element_rank_modified_lint(Coefficient_vector, 1, 15, rk);
	ost << "The point rank of the equation over GF$(" << Dom->F->q << ")$ is " << rk << "\\\\" << endl;

	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}

void del_pezzo_surface_of_degree_two_object::print_points(std::ostream &ost)
{
	ost << "\\subsection*{All Points on surface}" << endl;

	cout << "surface_object_properties::print_points before print_points_on_surface" << endl;
	//print_points_on_surface(ost);
	print_all_points_on_surface(ost);

#if 0
	ost << "\\subsubsection*{Eckardt Points}" << endl;
	cout << "surface_object_properties::print_points before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);

	ost << "\\subsubsection*{Singular Points}" << endl;
	cout << "surface_object_properties::print_points before print_singular_points" << endl;
	print_singular_points(ost);

	ost << "\\subsubsection*{Double Points}" << endl;
	cout << "surface_object_properties::print_points before print_double_points" << endl;
	print_double_points(ost);

	ost << "\\subsubsection*{Points on lines}" << endl;
	cout << "surface_object_properties::print_points before print_points_on_lines" << endl;
	print_points_on_lines(ost);

	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	cout << "surface_object_properties::print_points before print_points_on_surface_but_not_on_a_line" << endl;
	print_points_on_surface_but_not_on_a_line(ost);

#if 0
	ost << "\\clearpage" << endl;
	ost << "\\section*{Lines through points}" << endl;
	lines_on_point->print_table_tex(ost);
#endif
#endif

}

void del_pezzo_surface_of_degree_two_object::print_all_points_on_surface(std::ostream &ost)
{
	//latex_interface L;
	//int i;
	//int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_pts << " points:\\\\" << endl;

	if (nb_pts < 1000) {
		//ost << "$$" << endl;
		//L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, TRUE /* f_tex */);
		//ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;
		int v[4];

		for (i = 0; i < nb_pts; i++) {
			Dom->unrank_point(v, Pts[i]);
			ost << i << " : $P_{" << Pts[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		lint_vec_print_fully(ost, Pts, nb_pts);
		ost << "\\\\" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void del_pezzo_surface_of_degree_two_object::print_lines(std::ostream &ost)
{
	ost << "\\subsection*{The " << nb_lines << " Lines}" << endl;
	Dom->print_lines_tex(ost, Lines, nb_lines);
}





}}


