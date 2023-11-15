/*
 * kovalevski_points.cpp
 *
 *  Created on: Oct 8, 2023
 *      Author: betten
 */



#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {



kovalevski_points::kovalevski_points()
{
	QO = NULL;

	pts_on_lines = NULL;
	f_is_on_line = NULL;

	Contact_multiplicity = NULL;

	Bitangent_line_type = NULL;
	//line_type_distribution[3];
	lines_on_point = NULL;
	Point_type = NULL;
	f_fullness_has_been_established = false;
	f_is_full = false;


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
}

kovalevski_points::~kovalevski_points()
{
	if (pts_on_lines) {
		FREE_OBJECT(pts_on_lines);
	}
	if (f_is_on_line) {
		FREE_int(f_is_on_line);
	}

	if (Contact_multiplicity) {
		int i;

		for (i = 0; i < 28; i++) {
			FREE_int(Contact_multiplicity[i]);
		}
		FREE_pint(Contact_multiplicity);
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
}

void kovalevski_points::init(
		quartic_curve_object *QO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "kovalevski_points::init" << endl;
	}
	kovalevski_points::QO = QO;
	if (f_v) {
		cout << "kovalevski_points::init done" << endl;
	}
}


void kovalevski_points::compute_Kovalevski_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points" << endl;
	}

	if (!QO->f_has_bitangents) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"!QO->f_has_bitangents" << endl;
		exit(1);
	}
	combinatorics::combinatorics_domain Combi;


	Pts_off = NEW_lint(QO->Dom->P->Subspaces->N_points);

	Combi.set_complement_lint(
			QO->Pts, QO->nb_pts, Pts_off /*complement*/,
			nb_pts_off, QO->Dom->P->Subspaces->N_points);

	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points before "
				"compute_points_on_lines" << endl;
	}
	compute_points_on_lines(verbose_level - 2);
	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points after "
				"compute_points_on_lines" << endl;
	}

	Bitangent_line_type = NEW_OBJECT(data_structures::tally);
	Bitangent_line_type->init_lint(
			pts_on_lines->Set_size,
			pts_on_lines->nb_sets, false, 0);

	if (f_v) {
		cout << "Line type:" << endl;
		Bitangent_line_type->print_bare_tex(cout, true);
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
		cout << "kovalevski_points::compute_Kovalevski_points "
				"lines_on_point:" << endl;
		lines_on_point->print_table();
	}

	Point_type = NEW_OBJECT(data_structures::tally);
	Point_type->init_lint(
			lines_on_point->Set_size,
			lines_on_point->nb_sets, false, 0);
	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"type of lines_on_point:" << endl;
		Point_type->print_bare_tex(cout, true);
		cout << endl;
	}

	int f, l, a, b;
	vector<int> K;


	f_is_full = true;
	nb_Kovalevski_on = 0;
	for (i = Point_type->nb_types - 1; i >= 0; i--) {
		f = Point_type->type_first[i];
		l = Point_type->type_len[i];
		a = Point_type->data_sorted[f];
		if (a == 0) {
			f_is_full = false;
		}
		if (a == 4) {
			nb_Kovalevski_on += l;
			for (j = 0; j < l; j++) {
				b = Point_type->sorting_perm_inv[f + j];
				K.push_back(b);
			}
		}
	}
	f_fullness_has_been_established = true;



	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"before compute_off_points_on_lines" << endl;
	}
#if 0
	QO->Dom->compute_points_on_lines(
			Pts_off, nb_pts_off,
		QO->bitangents28, 28,
		pts_off_on_lines,
		f_is_on_line2,
		0 /* verbose_level */);
#endif
	compute_off_points_on_lines(verbose_level - 2);
	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"after compute_off_points_on_lines" << endl;
	}

	pts_off_on_lines->sort();

	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"pts_off_on_lines:" << endl;
		pts_off_on_lines->print_table();
	}

	pts_off_on_lines->dualize(
			lines_on_points_off, 0 /* verbose_level */);
	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"lines_on_point:" << endl;
		lines_on_point->print_table();
	}

	Point_off_type = NEW_OBJECT(data_structures::tally);
	Point_off_type->init_lint(
			lines_on_points_off->Set_size,
			lines_on_points_off->nb_sets, false, 0);
	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points "
				"type of lines_on_point off:" << endl;
		Point_off_type->print_bare_tex(cout, true);
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
	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points nb_Kovalevski     = " << nb_Kovalevski << endl;
		cout << "kovalevski_points::compute_Kovalevski_points nb_Kovalevski_on  = " << nb_Kovalevski_on << endl;
		cout << "kovalevski_points::compute_Kovalevski_points nb_Kovalevski_off = " << nb_Kovalevski_off << endl;
	}

	Kovalevski_point_idx = NEW_int(nb_Kovalevski);
	Kovalevski_points = NEW_lint(nb_Kovalevski);
	for (j = 0; j < nb_Kovalevski; j++) {
		Kovalevski_point_idx[j] = K[j];
		Kovalevski_points[j] = Pts_off[K[j]];
	}

	if (f_v) {
		cout << "kovalevski_points::compute_Kovalevski_points done" << endl;
	}
}

void kovalevski_points::compute_points_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *Pts;
	int nb_points;
	long int *Lines;
	int nb_lines;

	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines" << endl;
	}

	Pts = QO->Pts;
	nb_points = QO->nb_pts;
	Lines = QO->bitangents28;
	nb_lines = 28;


	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines "
				"before compute_points_on_lines_worker" << endl;
	}
	compute_points_on_lines_worker(
			Pts, nb_points,
			Lines, nb_lines,
			pts_on_lines,
			f_is_on_line,
			verbose_level);
	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines "
				"after compute_points_on_lines_worker" << endl;
	}

	pts_on_lines->sort();

	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines "
				"pts_on_lines:" << endl;
		pts_on_lines->print_table();
	}


	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines "
				"before compute_contact_multiplicity" << endl;
	}
	compute_contact_multiplicity(
			Lines, nb_lines,
			pts_on_lines,
			verbose_level);
	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines "
				"after compute_contact_multiplicity" << endl;
	}


	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines done" << endl;
	}
}


void kovalevski_points::compute_off_points_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *Pts;
	int nb_points;
	long int *Lines;
	int nb_lines;

	if (f_v) {
		cout << "kovalevski_points::compute_off_points_on_lines" << endl;
	}

	Pts = Pts_off;
	nb_points = nb_pts_off;
	Lines = QO->bitangents28;
	nb_lines = 28;

	compute_points_on_lines_worker(
			Pts, nb_points,
			Lines, nb_lines,
			pts_off_on_lines,
			f_is_on_line2,
			verbose_level);

	if (f_v) {
		cout << "kovalevski_points::compute_off_points_on_lines done" << endl;
	}
}



void kovalevski_points::compute_points_on_lines_worker(
		long int *Pts, int nb_points,
		long int *Lines, int nb_lines,
		data_structures::set_of_sets *&pts_on_lines,
		int *&f_is_on_line,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	long int l;
	int *pt_coords;
	int Basis[6];
	int Mtx[9];

	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines_worker" << endl;
	}
	f_is_on_line = NEW_int(nb_points);
	Int_vec_zero(f_is_on_line, nb_points);

	pts_on_lines = NEW_OBJECT(data_structures::set_of_sets);
	pts_on_lines->init_basic_constant_size(nb_points,
		nb_lines, QO->Dom->F->q + 1, 0 /* verbose_level */);
	pt_coords = NEW_int(nb_points * 3);
	for (i = 0; i < nb_points; i++) {
		QO->Dom->P->unrank_point(pt_coords + i * 3, Pts[i]);
	}

	Lint_vec_zero(pts_on_lines->Set_size, nb_lines);
	for (i = 0; i < nb_lines; i++) {
		l = Lines[i];
		QO->Dom->P->unrank_line(Basis, l);
		for (j = 0; j < nb_points; j++) {
			Int_vec_copy(Basis, Mtx, 6);
			Int_vec_copy(pt_coords + j * 3, Mtx + 6, 3);
			r = QO->Dom->F->Linear_algebra->Gauss_easy(Mtx, 3, 3);
			if (r == 2) {
				pts_on_lines->add_element(i, j);
				f_is_on_line[j] = true;
			}
		}
	}

	FREE_int(pt_coords);

	if (f_v) {
		cout << "kovalevski_points::compute_points_on_lines_worker done" << endl;
	}
}


void kovalevski_points::compute_contact_multiplicity(
		long int *Lines, int nb_lines,
		data_structures::set_of_sets *Pts_on_lines,
		int verbose_level)
// Pts_on_lines are the points of contact with the quartic curve for each line.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "kovalevski_points::compute_contact_multiplicity" << endl;
	}
	int i, j, h;
	long int a;
	int w[3];
	int coeff_out[5];
	int Basis[6];


	Contact_multiplicity = NEW_pint(nb_lines);

	for (i = 0; i < nb_lines; i++) {

		Contact_multiplicity[i] = NEW_int(Pts_on_lines->Set_size[i]);

		QO->Dom->P->Subspaces->Grass_lines->unrank_lint(Lines[i], 0 /*verbose_level*/);

		for (j = 0; j < Pts_on_lines->Set_size[i]; j++) {
			a = Pts_on_lines->Sets[i][j];

			// unrank the point of the line to w[]:
			QO->Dom->unrank_point(w, QO->Pts[a]);

			Int_vec_copy(QO->Dom->P->Subspaces->Grass_lines->M, Basis, 6);

			QO->Dom->F->Linear_algebra->adjust_basis(Basis, w,
					3 /* n */, 2 /* k */, 1 /* d */, verbose_level);

			// now the first row of Basis is w
			// and the second row is another point on the line

			QO->Dom->Poly4_3->substitute_line(
					QO->eqn15 /* int *coeff_in */, coeff_out,
					Basis /* int *Pt1_coeff */, Basis + 3,
					0 /*verbose_level*/);
			// coeff_in[nb_monomials], coeff_out[degree + 1]

			// count the multiplicity of the root:
			for (h = 0; h <= 4; h++) {
				if (coeff_out[h]) {
					break;
				}
			}

			Contact_multiplicity[i][j] = h;


		}

	}

	if (f_v) {
		cout << "kovalevski_points::compute_contact_multiplicity done" << endl;
	}
}




void kovalevski_points::print_general(
		std::ostream &ost)
{

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

}

void kovalevski_points::print_lines_with_points_on_them(
		std::ostream &ost)
{
	int f_v = false;

	if (pts_on_lines) {
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"before print_lines_and_points_of_contact" << endl;
		}
		print_lines_and_points_of_contact(ost, QO->bitangents28, 28);
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"after print_lines_and_points_of_contact" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"before print_bitangents" << endl;
		}
		QO->QP->print_bitangents(ost);
		if (f_v) {
			cout << "quartic_curve_object_properties::report_properties_simple "
					"after print_bitangents" << endl;
		}
	}

}

void kovalevski_points::print_all_points(
		std::ostream &ost)
{
	int i, j;
	int v[3];
	int a;
	long int b;


	algebraic_geometry::schlaefli_labels *Labels;

	Labels = NEW_OBJECT(algebraic_geometry::schlaefli_labels);
	if (false) {
		cout << "kovalevski_points::print_all_points "
				"before Labels->init" << endl;
	}
	Labels->init(0 /*verbose_level*/);
	if (false) {
		cout << "kovalevski_points::print_all_points "
				"after Labels->init" << endl;
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


void kovalevski_points::report_bitangent_line_type(
		std::ostream &ost)
{
	if (QO->f_has_bitangents) {
		l1_interfaces::latex_interface L;
		int i;
		long int a;
		int v[3];


		ost << "Line type: $" << endl;
		Bitangent_line_type->print_bare_tex(ost, true);
		ost << "$\\\\" << endl;

		ost << "$$" << endl;
		Bitangent_line_type->print_array_tex(ost, true /* f_backwards */);
		ost << "$$" << endl;


		ost << "point types: $" << endl;
		Point_type->print_bare_tex(ost, true);
		ost << "$\\\\" << endl;

		ost << "$$" << endl;
		Point_type->print_array_tex(ost, true /* f_backwards */);
		ost << "$$" << endl;

		ost << "point types for points off the curve: $" << endl;
		Point_off_type->print_bare_tex(ost, true);
		ost << "$\\\\" << endl;

		ost << "$$" << endl;
		Point_off_type->print_array_tex(ost, true /* f_backwards */);
		ost << "$$" << endl;

		ost << "Lines on points off the curve:\\\\" << endl;
		//lines_on_points_off->print_table_tex(ost);
		for (i = 0; i < lines_on_points_off->nb_sets; i++) {

			a = Pts_off[i];
			QO->Dom->unrank_point(v, a);


			ost << "Off point " << i << " = $P_{" << a << "} = ";

			Int_vec_print(ost, v, 3);

			ost << "$ lies on " << lines_on_points_off->Set_size[i] << " bisecants : ";
			L.lint_set_print_tex(ost,
					lines_on_points_off->Sets[i],
					lines_on_points_off->Set_size[i]);
			ost << "\\\\" << endl;
		}


	}
}

void kovalevski_points::print_lines_and_points_of_contact(
		std::ostream &ost,
		long int *Lines, int nb_lines)
{
	int i, j;
	//int verbose_level = 1;
	long int a;
	l1_interfaces::latex_interface L;
	int w[3];

	ost << "The lines and their points of contact are:\\\\" << endl;
	//ost << "\\begin{multicols}{2}" << endl;

	for (i = 0; i < nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		QO->Dom->P->Subspaces->Grass_lines->unrank_lint(Lines[i], 0 /*verbose_level*/);
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
		QO->Dom->P->Subspaces->Grass_lines->latex_matrix(ost,
				QO->Dom->P->Subspaces->Grass_lines->M);


		//print_integer_matrix_tex(ost, P->Grass_lines->M, 2, 4);
		//ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "_{" << Lines[i] << "}" << endl;

		if (QO->Dom->F->e > 1) {
			ost << "=" << endl;
			ost << "\\left[" << endl;
			L.print_integer_matrix_tex(ost,
					QO->Dom->P->Subspaces->Grass_lines->M, 2, 3);
			ost << "\\right]_{" << Lines[i] << "}" << endl;
		}

		for (j = 0; j < pts_on_lines->Set_size[i]; j++) {
			a = pts_on_lines->Sets[i][j];

			// unrank the point of the line to w[]:
			QO->Dom->unrank_point(w, QO->Pts[a]);

			ost << "P_{" << QO->Pts[a] << "}";
			ost << "=\\bP";
			Int_vec_print(ost, w, 3);

			ost << "\\;" << Contact_multiplicity[i][j] << " \\times ";

			if (j < pts_on_lines->Set_size[i] - 1) {
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





}}}

