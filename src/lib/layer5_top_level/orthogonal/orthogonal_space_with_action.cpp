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
	Blt_set_domain_with_action = NULL;
}

orthogonal_space_with_action::~orthogonal_space_with_action()
{
	if (O) {
		FREE_OBJECT(O);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (Blt_set_domain_with_action) {
		FREE_OBJECT(Blt_set_domain_with_action);
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

	P->projective_space_init(
			Descr->n - 1, Descr->F,
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
		Blt_set_domain_with_action = NEW_OBJECT(
				orthogonal_geometry_applications::blt_set_domain_with_action);


		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"before Blt_set_domain_with_action->init" << endl;
		}
		Blt_set_domain_with_action->init(A, P, O, verbose_level);
		if (f_v) {
			cout << "orthogonal_space_with_action::init_blt_set_domain "
					"after Blt_set_domain_with_action->init" << endl;
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

	A->Known_groups->init_orthogonal_group_with_O(
			O,
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



}}}

