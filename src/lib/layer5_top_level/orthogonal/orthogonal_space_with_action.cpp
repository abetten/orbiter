/*
 * orthogonal_space_with_action.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


orthogonal_space_with_action::orthogonal_space_with_action()
{
	Descr = NULL;
	P = NULL;
	O = NULL;
	f_semilinear = FALSE;
	A = NULL;
	AO = NULL;
	Blt_Set_domain = NULL;
}

orthogonal_space_with_action::~orthogonal_space_with_action()
{
	if (O) {
		FREE_OBJECT(O);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (Blt_Set_domain) {
		FREE_OBJECT(Blt_Set_domain);
	}
}

void orthogonal_space_with_action::init(
		orthogonal_space_with_action_description *Descr,
		int verbose_level)
// creates a projective space and an orthogonal space.
// For n == 5, it also creates a blt_set_domain
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init" << endl;
	}
	orthogonal_space_with_action::Descr = Descr;

	P = NEW_OBJECT(geometry::projective_space);

	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"before P->projective_space_init" << endl;
	}

	P->projective_space_init(Descr->n - 1, Descr->F,
		FALSE /* f_init_incidence_structure */,
		verbose_level);

	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"after P->projective_space_init" << endl;
	}

	O = NEW_OBJECT(orthogonal_geometry::orthogonal);


	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"before O->init" << endl;
	}
	O->init(Descr->epsilon, Descr->n, Descr->F, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"after O->init" << endl;
	}


	if (Descr->f_label_txt) {
		O->label_txt.assign(Descr->label_txt);
	}
	if (Descr->f_label_tex) {
		O->label_tex.assign(Descr->label_tex);
	}




	if (!Descr->f_without_group) {

		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"before init_group" << endl;
		}
		init_group(verbose_level - 2);
		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"after init_group" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "orthogonal_space_with_action::init without group" << endl;
		}

	}

	if (Descr->n == 5) {

		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"allocating Blt_Set_domain" << endl;
		}
		Blt_Set_domain = NEW_OBJECT(orthogonal_geometry::blt_set_domain);

		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"before Blt_Set_domain->init_blt_set_domain" << endl;
		}
		Blt_Set_domain->init_blt_set_domain(O, P, verbose_level - 2);
		if (f_v) {
			cout << "orthogonal_space_with_action::init_blt_set_domain "
					"after Blt_Set_domain->init" << endl;
		}
	}


	if (f_v) {
		cout << "orthogonal_space_with_action::init done" << endl;
	}
}

void orthogonal_space_with_action::init_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group" << endl;
	}

	number_theory::number_theory_domain NT;

	f_semilinear = TRUE;
	if (NT.is_prime(Descr->F->q)) {
		f_semilinear = FALSE;
	}


	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group before "
				"A->Known_groups->init_orthogonal_group_with_O" << endl;
	}

	A->Known_groups->init_orthogonal_group_with_O(O,
			TRUE /* f_on_points */,
			FALSE /* f_on_lines */,
			FALSE /* f_on_points_and_lines */,
			f_semilinear,
			TRUE /* f_basis */,
			verbose_level - 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"after A->Known_groups->init_orthogonal_group_with_O" << endl;
	}

	if (f_v) {
		cout << "A->make_element_size = "
			<< A->make_element_size << endl;
		cout << "orthogonal_space_with_action::init_group "
				"degree = " << A->degree << endl;
	}

	if (!A->f_has_sims) {
		cout << "orthogonal_space_with_action::init_group "
				"!A->f_has_sims" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"before A->lex_least_base_in_place" << endl;
	}
	A->lex_least_base_in_place(A->Sims, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"after A->lex_least_base_in_place" << endl;
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group base: ";
		Lint_vec_print(cout, A->get_base(), A->base_len());
		cout << endl;
	}



	AO = A->G.AO;
	O = AO->O;


	if (f_v) {
		cout << "orthogonal_space_with_action::init_group done" << endl;
	}
}

void orthogonal_space_with_action::report(
		graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::report" << endl;
	}

	{
		string fname_report;
		fname_report.assign(O->label_txt);
		fname_report.append("_report.tex");
		orbiter_kernel_system::latex_interface L;
		orbiter_kernel_system::file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

			if (f_v) {
				cout << "orthogonal_space_with_action::report "
						"before report2" << endl;
			}
			report2(ost, LG_Draw_options, verbose_level);
			if (f_v) {
				cout << "orthogonal_space_with_action::report "
						"after report2" << endl;
			}

			L.foot(ost);
		}

		cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}

}

void orthogonal_space_with_action::report2(
		std::ostream &ost,
		graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::report2" << endl;
	}


	if (!Descr->f_without_group) {
		if (f_v) {
			cout << "orthogonal_space_with_action::report2 "
					"before A>report" << endl;
		}

		A->report(ost,
				FALSE /* f_sims */, NULL,
				FALSE /* f_strong_gens */, NULL,
				LG_Draw_options,
				verbose_level - 1);

		if (f_v) {
			cout << "orthogonal_space_with_action::report2 "
					"after A->report" << endl;
		}
	}
	else {
		ost << "The group is not available.\\\\" << endl;
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::report2 before O->report" << endl;
	}
	O->report(ost, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::report2 after O->report" << endl;
	}


	if (f_v) {
		cout << "orthogonal_space_with_action::report2 done" << endl;
	}
}

void orthogonal_space_with_action::report_point_set(
		long int *Pts, int nb_pts,
		std::string &label_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::report_point_set" << endl;
	}

	{
		string fname_report;
		fname_report.assign(label_txt);
		fname_report.append("_set_report.tex");
		orbiter_kernel_system::latex_interface L;
		orbiter_kernel_system::file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

			if (f_v) {
				cout << "orthogonal_space_with_action::report_point_set "
						"before report_given_point_set" << endl;
			}
			//report2(ost, LG_Draw_options, verbose_level);

			O->report_given_point_set(ost, Pts, nb_pts, verbose_level);


			if (f_v) {
				cout << "orthogonal_space_with_action::report_point_set "
						"after report_given_point_set" << endl;
			}

			L.foot(ost);
		}

		if (f_v) {
			cout << "orthogonal_space_with_action::report_point_set "
					"Written file " << fname_report << " of size "
					<< Fio.file_size(fname_report) << endl;
		}
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::report_point_set done" << endl;
	}
}



void orthogonal_space_with_action::report_line_set(
		long int *Lines, int nb_lines,
		std::string &label_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::report_line_set" << endl;
	}

	{
		string fname_report;
		fname_report.assign(label_txt);
		fname_report.append("_set_of_lines_report.tex");
		orbiter_kernel_system::latex_interface L;
		orbiter_kernel_system::file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

			if (f_v) {
				cout << "orthogonal_space_with_action::report_line_set "
						"before report_given_line_set" << endl;
			}
			//report2(ost, LG_Draw_options, verbose_level);

			O->report_given_line_set(ost, Lines, nb_lines, verbose_level);


			if (f_v) {
				cout << "orthogonal_space_with_action::report_line_set "
						"after report_given_line_set" << endl;
			}

			L.foot(ost);
		}

		if (f_v) {
			cout << "orthogonal_space_with_action::report_line_set "
					"Written file " << fname_report << " of size "
					<< Fio.file_size(fname_report) << endl;
		}
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::report_line_set done" << endl;
	}
}

void orthogonal_space_with_action::make_table_of_blt_sets(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets" << endl;
	}

	if (O->Quadratic_form->n != 5) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"we need a five-dimensional orthogonal space" << endl;
		exit(1);
	}

	table_of_blt_sets *T;

	T = NEW_OBJECT(table_of_blt_sets);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"before T->init" << endl;
	}
	T->init(this, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"after T->init" << endl;
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"before T->do_export" << endl;
	}
	T->do_export(verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"after T->do_export" << endl;
	}

	FREE_OBJECT(T);


	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets done" << endl;
	}

}

void orthogonal_space_with_action::make_collinearity_graph(
		int *&Adj, int &N,
		long int *Set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph" << endl;
	}

	int i, j;
	int d, nb_e, nb_inc;
	int *v1, *v2;
	long int Nb_points;
	geometry::geometry_global Gg;
	//orthogonal_geometry::quadratic_form *Quadratic_form;


	d = O->Quadratic_form->n; // algebraic dimension

	v1 = NEW_int(d);
	v2 = NEW_int(d);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph" << endl;
	}


	Nb_points = O->Quadratic_form->nb_points;
	//Gg.nb_pts_Qepsilon(epsilon, n, F->q);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph "
				"number of points = " << Nb_points << endl;
	}

	N = sz;

	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph field:" << endl;
		O->Quadratic_form->F->Io->print();
	}

#if 0
	Quadratic_form = NEW_OBJECT(orthogonal_geometry::quadratic_form);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph "
				"before Quadratic_form->init" << endl;
	}
	Quadratic_form->init(epsilon, d, F, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph "
				"after Quadratic_form->init" << endl;
	}
#endif



#if 0
	if (f_list_points) {
		for (i = 0; i < N; i++) {
			F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
			cout << i << " : ";
			int_vec_print(cout, v, n + 1);
			j = F->Q_epsilon_rank(v, 1, epsilon, n, c1, c2, c3, 0 /* verbose_level */);
			cout << " : " << j << endl;

			}
		}
#endif


	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph "
				"allocating adjacency matrix" << endl;
	}
	Adj = NEW_int(N * N);
	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph "
				"allocating adjacency matrix was successful" << endl;
	}

	long int a, b;
	int val;


	for (i = 0; i < sz; i++) {

		a = Set[i];

		if (a < 0 || a >= Nb_points) {
			cout << "orthogonal_space_with_action::make_collinearity_graph out of range" << endl;
			exit(1);
		}
	}

	nb_e = 0;
	nb_inc = 0;
	for (i = 0; i < sz; i++) {

		a = Set[i];


		O->Quadratic_form->unrank_point(v1, a, 0 /* verbose_level */);

		for (j = i + 1; j < sz; j++) {

			b = Set[j];

			O->Quadratic_form->unrank_point(v2, b, 0 /* verbose_level */);

			val = O->Quadratic_form->evaluate_bilinear_form(v1, v2, 1);

			if (val == 0) {
				nb_e++;
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
			else {
				Adj[i * N + j] = 0;
				Adj[j * N + i] = 0;
				nb_inc++;
			}
		}
		Adj[i * N + i] = 0;
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph "
				"The adjacency matrix of the collinearity graph has been computed" << endl;
	}


	FREE_int(v1);
	FREE_int(v2);
	//FREE_OBJECT(Quadratic_form);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_collinearity_graph done" << endl;
	}
}



}}}

