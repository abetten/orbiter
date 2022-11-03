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
	//std::string label_txt;
	//std::string label_tex;
	O = NULL;
	f_semilinear = FALSE;
	A = NULL;
	AO = NULL;
}

orthogonal_space_with_action::~orthogonal_space_with_action()
{
	if (O) {
		FREE_OBJECT(O);
	}
	if (A) {
		FREE_OBJECT(A);
	}
}

void orthogonal_space_with_action::init(
		orthogonal_space_with_action_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init" << endl;
	}
	orthogonal_space_with_action::Descr = Descr;

	O = NEW_OBJECT(orthogonal_geometry::orthogonal);

	if (Descr->f_label_txt) {
		label_txt.assign(Descr->label_txt);
	}
	else {
		char str[1000];

		snprintf(str, sizeof(str), "O_%d_%d_%d", Descr->epsilon, Descr->n, Descr->F->q);
		label_txt.assign(str);
	}
	if (Descr->f_label_tex) {
		label_tex.assign(Descr->label_tex);
	}
	else {
		char str[1000];

		if (Descr->epsilon == 1) {
			snprintf(str, sizeof(str), "O^+(%d,%d)", Descr->n, Descr->F->q);
		}
		else if (Descr->epsilon == 0) {
			snprintf(str, sizeof(str), "O(%d,%d)", Descr->n, Descr->F->q);
		}
		else if (Descr->epsilon == -1) {
			snprintf(str, sizeof(str), "O^-(%d,%d)", Descr->n, Descr->F->q);
		}
		else {
			cout << "orthogonal_space_with_action::init illegal value of epsilon" << endl;
			exit(1);
		}
		label_tex.assign(str);
	}



	if (f_v) {
		cout << "orthogonal_space_with_action::init before O->init" << endl;
	}
	O->init(Descr->epsilon, Descr->n, Descr->F, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::init after O->init" << endl;
	}


	if (!Descr->f_without_group) {

		if (f_v) {
			cout << "orthogonal_space_with_action::init before init_group" << endl;
		}
		init_group(verbose_level);
		if (f_v) {
			cout << "orthogonal_space_with_action::init after init_group" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "orthogonal_space_with_action::init without group" << endl;
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
				"A->init_orthogonal_group_with_O" << endl;
	}

	A->init_orthogonal_group_with_O(O,
			TRUE /* f_on_points */,
			FALSE /* f_on_lines */,
			FALSE /* f_on_points_and_lines */,
			f_semilinear,
			TRUE /* f_basis */,
			verbose_level);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"after A->init_orthogonal_group_with_O" << endl;
	}

	if (f_v) {
		cout << "A->make_element_size = "
			<< A->make_element_size << endl;
		cout << "orthogonal_space_with_action::init_group "
				"degree = " << A->degree << endl;
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group computing "
				"lex-least base" << endl;
	}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group computing "
				"lex-least base done" << endl;
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
		fname_report.assign(label_txt);
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

void orthogonal_space_with_action::report2(std::ostream &ost,
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




}}}

