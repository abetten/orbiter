/*
 * schlaefli_tritangent_planes.cpp
 *
 *  Created on: Nov 15, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


schlaefli_tritangent_planes::schlaefli_tritangent_planes()
{

	Schlaefli = NULL;

	nb_Eckardt_points = 0;
	Eckardt_points = NULL;
	Eckard_point_label = NULL;
	Eckard_point_label_tex = NULL;

	incidence_lines_vs_tritangent_planes = NULL;
	Lines_in_tritangent_planes = NULL;

}


schlaefli_tritangent_planes::~schlaefli_tritangent_planes()
{
	if (Eckardt_points) {
		FREE_OBJECTS(Eckardt_points);
	}

	if (Eckard_point_label) {
		delete [] Eckard_point_label;
	}
	if (Eckard_point_label_tex) {
		delete [] Eckard_point_label_tex;
	}
	if (incidence_lines_vs_tritangent_planes) {
		FREE_int(incidence_lines_vs_tritangent_planes);
	}
	if (Lines_in_tritangent_planes) {
		FREE_lint(Lines_in_tritangent_planes);
	}

}


void schlaefli_tritangent_planes::init(
		schlaefli *Schlaefli, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli_tritangent_planes::init" << endl;
	}

	schlaefli_tritangent_planes::Schlaefli = Schlaefli;


	if (f_v) {
		cout << "schlaefli_tritangent_planes::init "
				"before make_Eckardt_points" << endl;
	}
	make_Eckardt_points(verbose_level);
	if (f_v) {
		cout << "schlaefli_tritangent_planes::init "
				"after make_Eckardt_points" << endl;
	}

	if (f_v) {
		cout << "schlaefli_tritangent_planes::init "
				"before init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}
	init_incidence_matrix_of_lines_vs_tritangent_planes(verbose_level);
	if (f_v) {
		cout << "schlaefli_tritangent_planes::init "
				"after init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}

}



void schlaefli_tritangent_planes::make_Eckardt_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "schlaefli_tritangent_planes::make_Eckardt_points" << endl;
	}
	nb_Eckardt_points = 45;
	Eckardt_points = NEW_OBJECTS(eckardt_point, nb_Eckardt_points);
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].init_by_rank(i);
	}

	Eckard_point_label = new string [nb_Eckardt_points];
	Eckard_point_label_tex = new string [nb_Eckardt_points];

	for (i = 0; i < nb_Eckardt_points; i++) {

		string str;

		str = Eckardt_points[i].make_label();
		//Eckardt_points[i].latex_to_str_without_E(str);
		Eckard_point_label[i] = str;
		Eckard_point_label_tex[i] = str;
	}
	if (f_v) {
		cout << "schlaefli_tritangent_planes::make_Eckardt_points done" << endl;
	}
}


void schlaefli_tritangent_planes::init_incidence_matrix_of_lines_vs_tritangent_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
	int three_lines[3];

	if (f_v) {
		cout << "schlaefli_tritangent_planes::init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}

	incidence_lines_vs_tritangent_planes = NEW_int(27 * 45);
	Int_vec_zero(incidence_lines_vs_tritangent_planes, 27 * 45);


	Lines_in_tritangent_planes = NEW_lint(45 * 3);
	Lint_vec_zero(Lines_in_tritangent_planes, 45 * 3);

	for (j = 0; j < nb_Eckardt_points; j++) {
		eckardt_point *E;

		E = Eckardt_points + j;
		E->three_lines(Schlaefli->Surf, three_lines);
		for (h = 0; h < 3; h++) {
			Lines_in_tritangent_planes[j * 3 + h] = three_lines[h];
				// conversion to long int
		}
		for (h = 0; h < 3; h++) {
			i = three_lines[h];
			incidence_lines_vs_tritangent_planes[i * 45 + j] = 1;
		}
	}



	if (f_v) {
		cout << "schlaefli_tritangent_planes::init_incidence_matrix_of_lines_vs_tritangent_planes done" << endl;
	}
}

void schlaefli_tritangent_planes::find_tritangent_planes_intersecting_in_a_line(
	int line_idx,
	int &plane1, int &plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;
	int three_lines[3];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schlaefli::find_tritangent_planes_intersecting_in_a_line" << endl;
	}
	for (plane1 = 0; plane1 < nb_Eckardt_points; plane1++) {

		Eckardt_points[plane1].three_lines(Schlaefli->Surf, three_lines);
		if (Sorting.int_vec_search_linear(three_lines, 3, line_idx, idx)) {
			for (plane2 = plane1 + 1;
					plane2 < nb_Eckardt_points;
					plane2++) {

				Eckardt_points[plane2].three_lines(Schlaefli->Surf, three_lines);
				if (Sorting.int_vec_search_linear(three_lines, 3, line_idx, idx)) {
					if (f_v) {
						cout << "schlaefli_tritangent_planes::find_tritangent_planes_intersecting_in_a_line done" << endl;
						}
					return;
				}
			}
		}
	}
	cout << "schlaefli_tritangent_planes::find_tritangent_planes_intersecting_in_a_line could not find "
			"two planes" << endl;
	exit(1);
}

void schlaefli_tritangent_planes::make_tritangent_plane_disjointness_graph(
		int *&Adj, int &nb_vertices, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	int *T;
	int i, j;
	int nb_tritangent_planes = 45;

	if (f_v) {
		cout << "schlaefli_tritangent_planes::make_tritangent_plane_disjointness_graph" << endl;
		cout << "nb_tritangent_planes=" << nb_tritangent_planes << endl;
	}

	nb_vertices = nb_tritangent_planes;

	Adj = NEW_int(nb_tritangent_planes * nb_tritangent_planes);
	T = NEW_int(nb_tritangent_planes * 3);
	for (i = 0; i < nb_tritangent_planes; i++) {
		Lint_vec_copy_to_int(Lines_in_tritangent_planes + i * 3, T + i * 3, 3);
		Sorting.int_vec_heapsort(T + i * 3, 3);
	}
	Int_vec_zero(Adj, nb_tritangent_planes * nb_tritangent_planes);
	for (i = 0; i < nb_tritangent_planes; i++) {
		for (j = i + 1; j < nb_tritangent_planes; j++) {
			if (Sorting.int_vecs_are_disjoint(T + i * 3, 3, T + j * 3, 3)) {
				Adj[i * nb_tritangent_planes + j] = 1;
				Adj[j * nb_tritangent_planes + 1] = 1;
			}
			else {
			}
		}
	}
	FREE_int(T);



	if (f_v) {
		cout << "schlaefli_tritangent_planes::make_tritangent_plane_disjointness_graph done" << endl;
	}
}

int schlaefli_tritangent_planes::choose_tritangent_plane_for_Clebsch_map(
		int line_a, int line_b,
			int transversal_line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j, nb;
	int planes[45];

	if (f_v) {
		cout << "schlaefli_tritangent_planes::choose_tritangent_plane_for_Clebsch_map" << endl;
	}

	nb = 0;
	for (j = 0; j < 45; j++) {
		if (incidence_lines_vs_tritangent_planes[line_a * 45 + j] == 0 &&
				incidence_lines_vs_tritangent_planes[line_b * 45 + j] == 0 &&
				incidence_lines_vs_tritangent_planes[transversal_line * 45 + j]) {
			planes[nb++] = j;
		}
	}
	if (nb != 3) {
		cout << "schlaefli_tritangent_planes::choose_tritangent_plane_for_Clebsch_map nb != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "schlaefli_tritangent_planes::choose_tritangent_plane_for_Clebsch_map done" << endl;
	}
	return planes[0];
}

void schlaefli_tritangent_planes::latex_tritangent_planes(
		std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i, j, a;

	cout << "schlaefli_tritangent_planes::latex_tritangent_planes" << endl;

	ost << "\\subsection*{Tritangent Planes}" << endl;
	ost << "The 45 tritangent planes are:\\\\" << endl;

	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(
			ost,
			Lines_in_tritangent_planes, 45, 3, 0, 0, true /* f_tex*/);
	ost << "$$";

	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < 45; i++) {
		ost << "\\noindent $";
		ost << "{\\cal T}_{" << i << "}";
		ost << "={\\cal T}_{" << Eckard_point_label_tex[i] << "}";
		ost << "= \\{";
		for (j = 0; j < 3; j++) {
			a = Lines_in_tritangent_planes[i * 3 + j];
			ost << Schlaefli->Labels->Line_label_tex[a];
			if (j < 3 - 1) {
				ost << ", ";
			}
		}
		ost << "\\}$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;

	cout << "schlaefli_tritangent_planes::latex_tritangent_planes done" << endl;
}

void schlaefli_tritangent_planes::latex_table_of_Eckardt_points(
		std::ostream &ost)
{
	int i, j;
	int three_lines[3];

	//cout << "schlaefli_tritangent_planes::latex_table_of_Eckardt_points" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(Schlaefli->Surf, three_lines);

		ost << "$E_{" << i << "} = " << endl;
		Eckardt_points[i].latex(ost);
		ost << " = ";
		for (j = 0; j < 3; j++) {
			ost << Schlaefli->Labels->Line_label_tex[three_lines[j]];
			if (j < 3 - 1) {
				ost << " \\cap ";
			}
		}
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
	//cout << "schlaefli_tritangent_planes::latex_table_of_Eckardt_points done" << endl;
}

void schlaefli_tritangent_planes::latex_table_of_tritangent_planes(
		std::ostream &ost)
{
	int i, j;
	int three_lines[3];

	//cout << "schlaefli_tritangent_planes::latex_table_of_tritangent_planes" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(Schlaefli->Surf, three_lines);

		ost << "$\\pi_{" << i << "} = \\pi_{" << endl;
		Eckardt_points[i].latex_index_only(ost);
		ost << "} = ";
		for (j = 0; j < 3; j++) {
			ost << Schlaefli->Labels->Line_label_tex[three_lines[j]];
		}
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
	//cout << "schlaefli_tritangent_planes::latex_table_of_tritangent_planes done" << endl;
}

void schlaefli_tritangent_planes::write_lines_vs_tritangent_planes(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli_tritangent_planes::write_lines_vs_tritangent_planes" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = prefix + "_lines_tritplanes_incma.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, incidence_lines_vs_tritangent_planes,
			27, 45);

	fname = prefix + "_lines_tritplanes.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, Lines_in_tritangent_planes,
			45, 3);

	if (f_v) {
		cout << "schlaefli_tritangent_planes::write_lines_vs_tritangent_planes done" << endl;
	}
}

int schlaefli_tritangent_planes::third_line_in_tritangent_plane(
		int l1, int l2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, k, l, m, n;

	if (f_v) {
		cout << "schlaefli_tritangent_planes::third_line_in_tritangent_plane" << endl;
	}
	if (l1 > l2) {
		int t = l1;
		l1 = l2;
		l2 = t;
	}
	// now l1 < l2.
	if (l1 < 6) {
		// l1 = ai line
		i = l1;
		if (l2 < 6) {
			cout << "schlaefli_tritangent_planes::third_line_in_tritangent_plane impossible (1)" << endl;
			exit(1);
		}
		if (l2 < 12) {
			j = l2 - 6;
			return Schlaefli->line_cij(i, j);
		}
		else {
			Schlaefli->index_of_line(l2, h, k);
			if (h == i) {
				return Schlaefli->line_bi(k);
			}
			else if (k == i) {
				return Schlaefli->line_bi(h);
			}
			else {
				cout << "schlaefli_tritangent_planes::third_line_in_tritangent_plane impossible (2)" << endl;
				exit(1);
			}
		}
	}
	else if (l1 < 12) {
		// l1 = bh line
		h = l1 - 6;
		if (l2 < 12) {
			cout << "schlaefli_tritangent_planes::third_line_in_tritangent_plane impossible (3)" << endl;
			exit(1);
		}
		Schlaefli->index_of_line(l2, i, j);
		if (i == h) {
			return Schlaefli->line_ai(j);
		}
		else if (h == j) {
			return Schlaefli->line_ai(i);
		}
		else {
			cout << "schlaefli_tritangent_planes::third_line_in_tritangent_plane impossible (4)" << endl;
			exit(1);
		}
	}
	else {
		// now we must be in a tritangent plane c_{ij,kl,mn}
		Schlaefli->index_of_line(l1, i, j);
		Schlaefli->index_of_line(l2, k, l);

		Schlaefli->ijkl2mn(i, j, k, l, m, n);

		return Schlaefli->line_cij(m, n);
	}
}




int schlaefli_tritangent_planes::Eckardt_point_from_tritangent_plane(
		int *tritangent_plane)
{
	int a, b, c, rk;
	eckardt_point E;
	data_structures::sorting Sorting;

	Sorting.int_vec_heapsort(tritangent_plane, 3);
	a = tritangent_plane[0];
	b = tritangent_plane[1];
	c = tritangent_plane[2];
	if (a < 6) {
		E.init2(a, b - 6);
	}
	else {
		if (a < 12) {
			cout << "schlaefli_tritangent_planes::Eckardt_point_from_tritangent_plane a < 12" << endl;
			exit(1);
		}
		a -= 12;
		b -= 12;
		c -= 12;
		E.init3(a, b, c);
	}
	rk = E.rank();
	return rk;
}




}}}


