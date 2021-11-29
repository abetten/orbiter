/*
 * action_projective.cpp
 *
 *  Created on: Feb 18, 2019
 *      Author: betten
 */



#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

strong_generators *action::set_stabilizer_in_projective_space(
	projective_space *P,
	long int *set, int set_size, //int &canonical_pt,
	int *canonical_set_or_NULL,
	int verbose_level)
// assuming we are in a linear action.
// added 2/28/2011, called from analyze.cpp
// November 17, 2014 moved here from TOP_LEVEL/extra.cpp
// December 31, 2014, moved here from projective_space.cpp
{
	int f_v = (verbose_level >= 1);
	object_with_canonical_form *OwCF;
	nauty_interface_with_group Nau;

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
	}


	OwCF = NEW_OBJECT(object_with_canonical_form);

	OwCF->init_point_set(set, set_size, verbose_level);

	OwCF->P = P;

	int nb_rows, nb_cols;
	bitvector *Canonical_form = NULL;

	OwCF->encoding_size(
			nb_rows, nb_cols,
			verbose_level);



	strong_generators *SG;
	nauty_output *NO;


	NO = NEW_OBJECT(nauty_output);
	NO->allocate(nb_rows + nb_cols, 0 /* verbose_level */);

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space before Nau.set_stabilizer_of_object" << endl;
	}
	SG = Nau.set_stabilizer_of_object(
			OwCF,
		this /* A_linear */,
		FALSE /* f_compute_canonical_form */, Canonical_form,
		NO,
		verbose_level - 2);
	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space after Nau.set_stabilizer_of_object" << endl;
	}

	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"go = " << *NO->Ago << endl;

		NO->print_stats();


	}


	FREE_OBJECT(NO);


	FREE_OBJECT(OwCF);

	return SG;
}

int action::reverse_engineer_semilinear_map(
	projective_space *P,
	int *Elt, int *Mtx, int &frobenius,
	int verbose_level)
// uses the function A->element_image_of
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map, before P->reverse_engineer_semilinear_map" << endl;
	}
	ret = P->reverse_engineer_semilinear_map(Elt, Mtx, frobenius, verbose_level);
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map, after P->reverse_engineer_semilinear_map" << endl;
	}
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map done" << endl;
	}
	return ret;
}


void action::report_fixed_objects_in_P3(ostream &ost,
	projective_space *P3,
	int *Elt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, cnt;
	int v[4];
	finite_field *F;

	if (f_v) {
		cout << "action::report_fixed_objects_in_P3" << endl;
	}
	//ost << "\\section{Fixed Objects}" << endl;

	F = P3->F;

	ost << "\\bigskip" << endl;

	ost << "The element" << endl;
	ost << "$$" << endl;
	element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "\\bigskip" << endl;
	//ost << "Fixed Points:\\" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
		}
	}

	ost << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = element_image_of(i, Elt, 0 /* verbose_level */);
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			Orbiter->Int_vec.print(ost, v, 4);
			ost << "\\\\" << endl;
			cnt++;
		}
	}

	ost << "\\bigskip" << endl;
	//ost << "Fixed Lines\\\\" << endl;

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

		ost << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $";
				A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "$\\\\" << endl;
				cnt++;
			}
		}

		FREE_OBJECT(A2);
	}

	ost << "\\bigskip" << endl;
	//ost << "Fixed Planes\\\\" << endl;

	{
		action *A3;

		A3 = induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A3->degree; i++) {
			j = A3->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
			}
		}

		ost << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A3->degree; i++) {
			j = A3->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $";
				A3->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "$\\\\" << endl;
				cnt++;
			}
		}

		FREE_OBJECT(A3);
	}
	if (f_v) {
		cout << "action::report_fixed_objects_in_P3 done" << endl;
	}
}


}}

