/*
 * quartic_curve_object_with_group.cpp
 *
 *  Created on: May 22, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {



quartic_curve_object_with_group::quartic_curve_object_with_group()
{
	Record_birth();
	DomA = NULL;
	QO = NULL;
	Aut_gens = NULL;
	f_has_nice_gens = false;
	nice_gens = NULL;
	projectivity_group_gens = NULL;
	Syl = NULL;
	A_on_points = NULL;
	Orbits_on_points = NULL;
}

quartic_curve_object_with_group::~quartic_curve_object_with_group()
{
	Record_death();
}

void quartic_curve_object_with_group::init(
		quartic_curve_domain_with_action *DomA,
		geometry::algebraic_geometry::quartic_curve_object *QO,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_group::init" << endl;
	}
	quartic_curve_object_with_group::DomA = DomA;
	quartic_curve_object_with_group::QO = QO;
	quartic_curve_object_with_group::Aut_gens = Aut_gens;

	if (f_v) {
		cout << "quartic_curve_object_with_group::init done" << endl;
	}
}

void quartic_curve_object_with_group::export_something(
		std::string &what,
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_group::export_something" << endl;
	}

	other::data_structures::string_tools ST;
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	if (ST.stringcmp(what, "points") == 0) {

		fname = fname_base + "_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->get_points(), 1, QO->get_nb_points());

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "equation") == 0) {

		fname = fname_base + "_equation.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, QO->Variety_object->eqn, 1, 15);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "bitangents") == 0) {

		fname = fname_base + "_bitangents.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->get_lines(), 1, QO->get_nb_lines());

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "Kovalevski_points") == 0) {

		fname = fname_base + "_Kovalevski_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->QP->Kovalevski->Kovalevski_points,
				1, QO->QP->Kovalevski->nb_Kovalevski);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "singular_points") == 0) {

		fname = fname_base + "_singular_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, QO->QP->singular_pts, 1, QO->QP->nb_singular_pts);

		cout << "quartic_curve_object_with_group::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else {
		cout << "quartic_curve_object_with_group::export_something "
				"unrecognized export target: " << what << endl;
	}


	if (f_v) {
		cout << "quartic_curve_object_with_group::export_something done" << endl;
	}

}


void quartic_curve_object_with_group::export_col_headings(
		std::string *&Col_headings, int &nb_cols,
		int verbose_level)
{
	nb_cols = 15;
	Col_headings = new std::string [nb_cols];

	Col_headings[0] = "n";
	Col_headings[1] = "q";
	Col_headings[2] = "d";
	Col_headings[3] = "label_txt";
	Col_headings[4] = "label_tex";
	Col_headings[5] = "equation_af";
	Col_headings[6] = "equation_vec";
	Col_headings[7] = "Ago";
	Col_headings[8] = "SetStab";
	Col_headings[9] = "NbPoints";
	Col_headings[10] = "NbLines";
	Col_headings[11] = "NbSingPoints";
	Col_headings[12] = "Points";
	Col_headings[13] = "Lines";
	Col_headings[14] = "LinesKlein";
}

void quartic_curve_object_with_group::export_data(
		std::vector<std::string> &Table, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_object_with_group::export_data" << endl;
	}

	algebra::ring_theory::longinteger_object ago;
	algebra::ring_theory::longinteger_object set_stab_go;


	QO->Variety_object;

	Aut_gens->group_order(ago);

	set_stab_go.create(-1);


	int nb_points;
	int nb_lines;
	int nb_singular_points = -1;

	nb_points = QO->Variety_object->get_nb_points();
	nb_lines = QO->Variety_object->get_nb_lines();

	if (QO->Variety_object->f_has_singular_points) {
		nb_singular_points = QO->Variety_object->Singular_points.size();
	}

	string s_Pts, s_Lines, s_Lines_Klein;

	s_Pts = "\"" + QO->Variety_object->stringify_points() + "\"";
	s_Lines = "\"" + QO->Variety_object->stringify_lines() + "\"";


	if (DomA->PA->P->Subspaces->n == 3 && QO->Variety_object->Line_sets) {



		geometry::orthogonal_geometry::orthogonal *O;
		geometry::projective_geometry::klein_correspondence *Klein;

		O = DomA->PA->Surf_A->Surf->O;
		Klein = DomA->PA->Surf_A->Surf->Klein;

		int v[6];
		long int *Lines;
		long int *Points_on_Klein_quadric;
		long int line_rk;
		int i;

		Lines = QO->Variety_object->Line_sets->Sets[0];

		Points_on_Klein_quadric = NEW_lint(nb_lines);

		for (i = 0; i < nb_lines; i++) {
			line_rk = Lines[i];

			Klein->line_to_Pluecker(
				line_rk, v, 0 /* verbose_level*/);

			Points_on_Klein_quadric[i] = O->Orthogonal_indexing->Qplus_rank(
					v,
					1, 5, 0 /*verbose_level */);

		}

		other::data_structures::sorting Sorting;
		Sorting.lint_vec_heapsort(Points_on_Klein_quadric, nb_lines);


		s_Lines_Klein = "\"" + Lint_vec_stringify(Points_on_Klein_quadric, nb_lines) + "\"";

		FREE_lint(Points_on_Klein_quadric);
	}
	else {
		s_Lines_Klein = "\"\"";
	}

	int n, q, d;
	string s_eqn;
	string s_eqn_vec;

	n = DomA->PA->P->Subspaces->n;
	q = DomA->PA->P->Subspaces->F->q;
	d = QO->Variety_object->Ring->degree;

	s_eqn = "\"" + QO->Variety_object->Ring->stringify_equation(QO->Variety_object->eqn) + "\"";
	s_eqn_vec = "\"" + Int_vec_stringify(QO->Variety_object->eqn, QO->Variety_object->Ring->get_nb_monomials()) + "\"";

	Table.push_back(std::to_string(n));
	Table.push_back(std::to_string(q));
	Table.push_back(std::to_string(d));
	Table.push_back("\"" + QO->Variety_object->label_txt + "\"");
	Table.push_back("\"" + QO->Variety_object->label_tex + "\"");

	Table.push_back(s_eqn);
	Table.push_back(s_eqn_vec);



	Table.push_back(ago.stringify());
	Table.push_back(set_stab_go.stringify());
	Table.push_back(std::to_string(nb_points));
	Table.push_back(std::to_string(nb_lines));
	Table.push_back(std::to_string(nb_singular_points));
	Table.push_back(s_Pts);
	Table.push_back(s_Lines);
	Table.push_back(s_Lines_Klein);

	if (f_v) {
		cout << "quartic_curve_object_with_group::export_data done" << endl;
	}
}







}}}}

