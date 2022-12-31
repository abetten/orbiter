/*
 * action_group_theory.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: betten
 */

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {





void action::report_groups_and_normalizers(
		std::ostream &ost,
		int nb_subgroups,
		groups::strong_generators *H_gens,
		groups::strong_generators *N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u;
	ring_theory::longinteger_object go1, go2;

	if (f_v) {
		cout << "action::report_groups_and_normalizers" << endl;
	}

	for (u = 0; u < nb_subgroups; u++) {

		ost << "\\subsection*{Class " << u << " / " << nb_subgroups << "}" << endl;

		H_gens[u].group_order(go1);
		N_gens[u].group_order(go2);

		ost << "Group order = " << go1 << "\\\\" << endl;
		ost << "Normalizer order = " << go2 << "\\\\" << endl;

		ost << "Generators for $H$:\\\\" << endl;

		H_gens[u].print_generators_in_latex_individually(ost);
		H_gens[u].print_generators_as_permutations_tex(ost, this);

		ost << "\\bigskip" << endl;

		ost << "Generators for $N(H)$:\\\\" << endl;

		N_gens[u].print_generators_in_latex_individually(ost);
		N_gens[u].print_generators_as_permutations_tex(ost, this);

	}


	if (f_v) {
		cout << "action::report_groups_and_normalizers done" << endl;
	}
}

void action::report_fixed_objects(int *Elt,
		char *fname_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j, cnt;
	//int v[4];
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::report_fixed_objects" << endl;
		}


	{
		ofstream fp(fname_latex);
		string title, author, extra_praeamble;
		orbiter_kernel_system::latex_interface L;

		title.assign("Fixed Objects");

		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author /* const char *author */,
			FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);

		fp << "\\section{Fixed Objects}" << endl;



		fp << "The element" << endl;
		fp << "$$" << endl;
		element_print_latex(Elt, fp);
		fp << "$$" << endl;
		fp << "has the following fixed objects:\\\\" << endl;


	#if 0
		fp << "\\subsection{Fixed Points}" << endl;

		cnt = 0;
		for (i = 0; i < P3->N_points; i++) {
			j = element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed points, they are: \\\\" << endl;
		for (i = 0; i < P3->N_points; i++) {
			j = element_image_of(i, Elt, 0 /* verbose_level */);
			F->PG_element_unrank_modified(v, 1, 4, i);
			if (j == i) {
				fp << i << " : ";
				int_vec_print(fp, v, 4);
				fp << "\\\\" << endl;
				cnt++;
				}
			}

		fp << "\\subsection{Fixed Lines}" << endl;

		{
		action *A2;

		A2 = induced_action_on_grassmannian(2, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				fp << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
				fp << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
		}

		fp << "\\subsection{Fixed Planes}" << endl;

		{
		action *A2;

		A2 = induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		fp << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				fp << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
				fp << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
		}
	#endif


		L.foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;


	if (f_v) {
		cout << "action::report_fixed_objects done" << endl;
		}
}

void action::element_conjugate_bvab(int *Elt_A,
		int *Elt_B, int *Elt_C,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2;


	if (f_v) {
		cout << "action::element_conjugate_bvab" << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	if (f_v) {
		cout << "action::element_conjugate_bvab A=" << endl;
		element_print_quick(Elt_A, cout);
		cout << "action::element_conjugate_bvab B=" << endl;
		element_print_quick(Elt_B, cout);
		}

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt1, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt_B, Elt_C, 0);
	if (f_v) {
		cout << "action::element_conjugate_bvab C=B^-1 * A * B" << endl;
		element_print_quick(Elt_C, cout);
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "action::element_conjugate_bvab done" << endl;
		}
}

void action::element_conjugate_babv(int *Elt_A,
		int *Elt_B, int *Elt_C,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2;


	if (f_v) {
		cout << "action::element_conjugate_babv" << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt_B, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt1, Elt_C, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
}

void action::element_commutator_abavbv(int *Elt_A,
		int *Elt_B, int *Elt_C,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2, *Elt3, *Elt4;


	if (f_v) {
		cout << "action::element_commutator_abavbv" << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);

	element_invert(Elt_A, Elt1, 0);
	element_invert(Elt_B, Elt2, 0);
	element_mult(Elt_A, Elt_B, Elt3, 0);
	element_mult(Elt3, Elt1, Elt4, 0);
	element_mult(Elt4, Elt2, Elt_C, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
}

void action::compute_projectivity_subgroup(
		groups::strong_generators *&projectivity_gens,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_projectivity_subgroup" << endl;
	}

	if (is_semilinear_matrix_group()) {
		if (f_v) {
			cout << "action::compute_projectivity_subgroup "
					"computing projectivity subgroup" << endl;
		}

		projectivity_gens = NEW_OBJECT(groups::strong_generators);
		{
			groups::sims *S;

			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"before Aut_gens->create_sims" << endl;
			}
			S = Aut_gens->create_sims(0 /*verbose_level */);
			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"after Aut_gens->create_sims" << endl;
			}
			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"before projectivity_group_gens->projectivity_subgroup" << endl;
			}
			projectivity_gens->projectivity_subgroup(S, verbose_level - 3);
			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"after projectivity_group_gens->projectivity_subgroup" << endl;
			}
			FREE_OBJECT(S);
		}
		if (f_v) {
			cout << "action::compute_projectivity_subgroup "
					"computing projectivity subgroup done" << endl;
		}
	}
	else {
		projectivity_gens = NULL;
	}


	if (f_v) {
		cout << "action::compute_projectivity_subgroup done" << endl;
	}
}



}}}

