/*
 * points_and_lines.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {




points_and_lines::points_and_lines()
{
	P = NULL;

	Pts = NULL;
	nb_pts = 0;

	Lines = NULL;
	nb_lines = 0;

	Quadratic_form = NULL;

}

points_and_lines::~points_and_lines()
{
	if (Pts) {
		FREE_lint(Pts);
	}
	if (Lines) {
		FREE_lint(Lines);
	}
	if (Quadratic_form) {
		FREE_OBJECT(Quadratic_form);
	}
}

void points_and_lines::init(
		projective_space *P,
		std::vector<long int> &Points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "points_and_lines::init" << endl;
	}

	points_and_lines::P = P;

	vector<long int> The_Lines;

	if (f_v) {
		cout << "points_and_lines::init The object "
				"has " << Points.size() << " points" << endl;
	}
	int i;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}


	geometry_global Geo;

	if (f_v) {
		cout << "points_and_lines::init before "
				"Geo.find_lines_which_are_contained" << endl;
	}
	Geo.find_lines_which_are_contained(P, Points, The_Lines, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "points_and_lines::init after "
				"Geo.find_lines_which_are_contained" << endl;
	}

	if (f_v) {
		cout << "points_and_lines::init The object "
				"has " << The_Lines.size() << " lines" << endl;
	}


	nb_lines = The_Lines.size();
	Lines = NEW_lint(nb_lines);
	for (i = 0; i < nb_lines; i++) {
		Lines[i] = The_Lines[i];
	}

	if (f_v) {
		cout << "points_and_lines::init "
				"nb_pts=" << nb_pts << " nb_lines=" << nb_lines << endl;
		cout << "Lines:";
		Lint_vec_print(cout, Lines, nb_lines);
		cout << endl;
	}


	if (P->Subspaces->n == 3) {
		Quadratic_form = NEW_OBJECT(orthogonal_geometry::quadratic_form);

		if (f_v) {
			cout << "points_and_lines::init "
					"before Quadratic_form->init" << endl;
		}
		Quadratic_form->init(1 /* epsilon */, 4, P->Subspaces->F, verbose_level);
		if (f_v) {
			cout << "points_and_lines::init "
					"after Quadratic_form->init" << endl;
		}
	}

	if (f_v) {
		cout << "points_and_lines::init done" << endl;
	}

}

void points_and_lines::unrank_point(int *v, long int rk)
{
	P->unrank_point(v, rk);
}

long int points_and_lines::rank_point(int *v)
{
	long int rk;

	rk = P->rank_point(v);
	return rk;
}



void points_and_lines::print_all_points(std::ostream &ost)
{
	//latex_interface L;
	//int i;
	//int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_pts << " points:\\\\" << endl;

	if (nb_pts < 1000) {
		//ost << "$$" << endl;
		//L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, true /* f_tex */);
		//ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;
		int v[4];

		for (i = 0; i < nb_pts; i++) {
			unrank_point(v, Pts[i]);
			ost << i << " : $P_{" << Pts[i] << "}=";
			Int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		Lint_vec_print_fully(ost, Pts, nb_pts);
		ost << "\\\\" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void points_and_lines::print_all_lines(std::ostream &ost)
{
	ost << "\\subsection*{The " << nb_lines << " Lines}" << endl;
	print_lines_tex(ost);
}

void points_and_lines::print_lines_tex(std::ostream &ost)
{
	int i;
	l1_interfaces::latex_interface L;
	long int *Rk;

	Rk = NEW_lint(nb_lines);

	ost << "The lines and their Pluecker coordinates are:\\\\" << endl;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		P->Subspaces->Grass_lines->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "}";

#if 0
		if (nb_lines == 27) {
			ost << " = " << Schlaefli->Line_label_tex[i];
		}
#endif
		ost << " = " << endl;
		//print_integer_matrix_width(cout,
		// Gr->M, k, n, n, F->log10_of_q + 1);
		P->Subspaces->Grass_lines->latex_matrix(ost, P->Subspaces->Grass_lines->M);
		//print_integer_matrix_tex(ost, Gr->M, 2, 4);
		//ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "_{" << Lines[i] << "}" << endl;
		ost << "=" << endl;
		ost << "\\left[" << endl;
		L.print_integer_matrix_tex(ost, P->Subspaces->Grass_lines->M, 2, 4);
		ost << "\\right]_{" << Lines[i] << "}" << endl;

		int v6[6];

		P->Subspaces->Grass_lines->Pluecker_coordinates(Lines[i], v6, 0 /* verbose_level */);



		Rk[i] = Quadratic_form->Orthogonal_indexing->Qplus_rank(v6, 1, 5, 0 /* verbose_level*/);

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

	FREE_lint(Rk);

}

void points_and_lines::write_points_to_txt_file(
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "points_and_lines::write_points_to_txt_file" << endl;
	}
	string fname;
	fname = label + "_points.txt";

	orbiter_kernel_system::file_io Fio;

	{
		ofstream ost(fname);
		int i;
		ost << nb_pts;
		for (i = 0; i < nb_pts; i++) {
			ost << " " << Pts[i];
		}
		ost << endl;
		ost << -1 << endl;
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "points_and_lines::write_points_to_txt_file done" << endl;
	}
}


}}}


