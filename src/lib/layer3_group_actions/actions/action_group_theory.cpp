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





#if 0
void action::report_fixed_objects(
		int *Elt,
		std::string &fname_latex,
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
		l1_interfaces::latex_interface L;

		title.assign("Fixed Objects");

		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author /* const char *author */,
			false /* f_toc */, false /* f_landscape */, true /* f_12pt */,
			true /* f_enlarged_page */, true /* f_pagenumbers */,
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
#endif





}}}

