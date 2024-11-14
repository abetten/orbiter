/*
 * fixed_objects_in_PG.cpp
 *
 *  Created on: Oct 1, 2024
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace combinatorics_with_groups {


fixed_objects_in_PG::fixed_objects_in_PG()
{

	A_base = NULL;
	A = NULL;
	P = NULL;
	Elt = NULL;

	up_to_which_rank = -1;
	//std::vector<std::vector<long int> > Fix;

}


fixed_objects_in_PG::~fixed_objects_in_PG()
{
	if (Elt) {
		FREE_int(Elt);
	}
}

void fixed_objects_in_PG::init(
		actions::action *A_base,
		actions::action *A,
		int *Elt,
		int up_to_which_rank,
		geometry::projective_space *P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "fixed_objects_in_PG::init "
				"computing fixed points" << endl;
	}

	fixed_objects_in_PG::A_base = A_base;
	fixed_objects_in_PG::A = A;
	fixed_objects_in_PG::up_to_which_rank = up_to_which_rank;
	fixed_objects_in_PG::P = P;
	fixed_objects_in_PG::Elt = NEW_int(A->elt_size_in_int);
	Int_vec_copy(Elt, fixed_objects_in_PG::Elt, A->elt_size_in_int);

	{
		vector<long int> fixed_points;

		if (f_v) {
			cout << "fixed_objects_in_PG::init "
					"before compute_fixed_points" << endl;
		}
		compute_fixed_points(
				fixed_points, 0 /* verbose_level */);
		if (f_v) {
			cout << "fixed_objects_in_PG::init "
					"after compute_fixed_points" << endl;
		}

		Fix.push_back(fixed_points);
	}

	int dimension;

	for (dimension = 2; dimension <= up_to_which_rank; dimension++) {

		if (f_v) {
			cout << "fixed_objects_in_PG::init "
					"computing fixed subspaces of rank " << dimension << endl;
		}

		vector<long int> fixpoints;


		if (f_v) {
			cout << "fixed_objects_in_PG::init "
					"before compute_fixed_points_in_induced_action_on_grassmannian" << endl;
		}
		compute_fixed_points_in_induced_action_on_grassmannian(
			dimension,
			fixpoints,
			0 /*verbose_level*/);
		if (f_v) {
			cout << "fixed_objects_in_PG::init "
					"after compute_fixed_points_in_induced_action_on_grassmannian" << endl;
		}

		Fix.push_back(fixpoints);


	}

	if (f_v) {
		cout << "fixed_objects_in_PG::init done" << endl;
	}


}

void fixed_objects_in_PG::compute_fixed_points(
		std::vector<long int> &fixed_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j;

	if (f_v) {
		cout << "fixed_objects_in_PG::compute_fixed_points "
				"computing fixed points in action "
				<< A->label << " of degree " << A->degree << endl;
	}
	for (i = 0; i < A->degree; i++) {
		j = A->Group_element->element_image_of(i, Elt, 0);
		if (j == i) {
			fixed_points.push_back(i);
		}
	}
	if (f_v) {
		cout << "group_element::compute_fixed_points "
				"found " << fixed_points.size() << " fixed points" << endl;
	}
	if (f_v) {
		cout << "fixed_objects_in_PG::compute_fixed_points done" << endl;
	}
}

void fixed_objects_in_PG::compute_fixed_points_in_induced_action_on_grassmannian(
		int dimension,
		std::vector<long int> &fixed_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "fixed_objects_in_PG::compute_fixed_points_in_induced_action_on_grassmannian "
				"computing fixed points in action "
				<< A->label << " of degree " << A->degree << endl;
	}
	actions::action *A_induced;

	if (f_v) {
		cout << "fixed_objects_in_PG::compute_fixed_points_in_induced_action_on_grassmannian "
				"before A->Induced_action->induced_action_on_grassmannian" << endl;
	}
	A_induced = A->Induced_action->induced_action_on_grassmannian(
			dimension, 0 /* verbose_level*/);
	if (f_v) {
		cout << "fixed_objects_in_PG::compute_fixed_points_in_induced_action_on_grassmannian "
				"after A->Induced_action->induced_action_on_grassmannian" << endl;
	}

	long int a, b;

	for (a = 0; a < A_induced->degree; a++) {
		b = A_induced->Group_element->element_image_of(
				a, Elt, 0 /* verbose_level */);
		if (b == a) {
			fixed_points.push_back(a);
		}
	}


	FREE_OBJECT(A_induced);

	if (f_v) {
		cout << "fixed_objects_in_PG::compute_fixed_points_in_induced_action_on_grassmannian done" << endl;
	}
}

void fixed_objects_in_PG::report(
		std::ostream &ost,
	int verbose_level)
// creates temporary actions using induced_action_on_grassmannian
{
	int f_v = (verbose_level >= 1);
	int j, h, cnt;
	int v[4];
	//field_theory::finite_field *F;

	if (f_v) {
		cout << "fixed_objects_in_PG::report" << endl;
	}


	//ost << "\\section{Fixed Objects}" << endl;

	//F = PG->F;

#if 0
	//int up_to_which_rank = P->Subspaces->n;
	std::vector<std::vector<long int>> Fix;
	long int a;

	if (f_v) {
		cout << "fixed_objects_in_PG::report "
				"before compute_fixed_objects_in_PG" << endl;
	}
	compute_fixed_objects_in_PG(
			up_to_which_rank,
			P,
			Elt,
			Fix,
			verbose_level);
#endif

	ost << "\\bigskip" << endl;

	ost << "The element" << endl;
	ost << "$$" << endl;
	A->Group_element->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "\\bigskip" << endl;
	//ost << "Fixed Points:\\" << endl;

	long int a;

	cnt = Fix[0].size();
	ost << "There are " << cnt << " / " << P->Subspaces->N_points
			<< " fixed points, they are: \\\\" << endl;
	for (j = 0; j < cnt; j++) {
		a = Fix[0][j];

		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified_lint(
				v, 1, 4, a);

		ost << j << " / " << cnt << " = " << a << " : ";
		Int_vec_print(ost, v, 4);
		ost << "\\\\" << endl;
	}

	ost << "\\bigskip" << endl;

	for (h = 2; h <= up_to_which_rank; h++) {

		if (f_v) {
			cout << "fixed_objects_in_PG::report "
					"listing fixed subspaces of rank " << h << endl;
		}
		vector<long int> fix;
		actions::action *Ah;

		if (f_v) {
			cout << "fixed_objects_in_PG::report "
					"before A->Induced_action->induced_action_on_grassmannian" << endl;
		}
		Ah = A->Induced_action->induced_action_on_grassmannian(
				h, 0 /* verbose_level*/);
		if (f_v) {
			cout << "fixed_objects_in_PG::report "
					"after A->Induced_action->induced_action_on_grassmannian" << endl;
		}

		cnt = Fix[h - 1].size();
		ost << "There are " << cnt << " / " << Ah->degree
				<< " fixed subspaces of "
				"rank " << h << ", they are: \\\\" << endl;

		for (j = 0; j < cnt; j++) {
			a = Fix[h - 1][j];

			ost << j << " / " << cnt << " = " << a << " : $";
			Ah->G.AG->G->print_single_generator_matrix_tex(ost, a);
			ost << "$\\\\" << endl;
		}
		FREE_OBJECT(Ah);
	}


	if (f_v) {
		cout << "fixed_objects_in_PG::report done" << endl;
	}
}




}}}


