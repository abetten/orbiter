// action_on_cosets.cpp
//
// Anton Betten
// Dec 24, 2013

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_cosets::action_on_cosets()
{
	A_linear = NULL;
	F = NULL;
	dimension_of_subspace = 0;
	n = 0;
	subspace_basis = NULL;
	base_cols = NULL;

	f_lint = FALSE;
	nb_points = 0;
	Points = NULL;
	lint_Points = NULL;

	v1 = NULL;
	v2 = NULL;

	unrank_point = NULL;
	rank_point = NULL;
	unrank_point_lint = NULL;
	rank_point_lint = NULL;
	rank_unrank_data = NULL;
}

action_on_cosets::~action_on_cosets()
{
	int f_v = FALSE;
	//int f_vv = FALSE;

	if (f_v) {
		cout << "action_on_cosets::~action_on_coset" << endl;
		}
	if (v1) {
		FREE_int(v1);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (f_v) {
		cout << "action_on_cosets::~action_on_coset done" << endl;
		}
}

void action_on_cosets::init(int nb_points, int *Points, 
		actions::action *A_linear,
	field_theory::finite_field *F,
	int dimension_of_subspace, 
	int n, 
	int *subspace_basis, 
	int *base_cols, 
	void (*unrank_point)(int *v, int a, void *data), 
	int (*rank_point)(int *v, void *data), 
	void *rank_unrank_data, 
	int verbose_level)
{
	int f_v = FALSE;
	int i;

	if (f_v) {
		cout << "action_on_cosets::init nb_points=" << nb_points
				<< " dimension_of_subspace=" << dimension_of_subspace
				<< " n=" << n << endl;
		}
	f_lint = FALSE;
	action_on_cosets::nb_points = nb_points;
	action_on_cosets::Points = Points;
	action_on_cosets::A_linear = A_linear;
	action_on_cosets::F = F;
	action_on_cosets::dimension_of_subspace = dimension_of_subspace;
	action_on_cosets::n = n;
	action_on_cosets::subspace_basis = subspace_basis;
	action_on_cosets::base_cols = base_cols;
	action_on_cosets::unrank_point = unrank_point;
	action_on_cosets::rank_point = rank_point;
	action_on_cosets::rank_unrank_data = rank_unrank_data;
	v1 = NEW_int(n);
	v2 = NEW_int(n);
	for (i = 0; i < nb_points - 1; i++) {
		if (Points[i] >= Points[i + 1]) {
			cout << "action_on_cosets::init the array Points[] "
					"is not sorted increasingly" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "action_on_cosets::init done" << endl;
		}
}

void action_on_cosets::init_lint(int nb_points, long int *Points,
		actions::action *A_linear,
	field_theory::finite_field *F,
	int dimension_of_subspace,
	int n,
	int *subspace_basis,
	int *base_cols,
	void (*unrank_point)(int *v, long int a, void *data),
	long int (*rank_point)(int *v, void *data),
	void *rank_unrank_data,
	int verbose_level)
{
	int f_v = FALSE;
	int i;

	if (f_v) {
		cout << "action_on_cosets::init_lint nb_points=" << nb_points
				<< " dimension_of_subspace=" << dimension_of_subspace
				<< " n=" << n << endl;
		}

	f_lint = TRUE;
	action_on_cosets::nb_points = nb_points;
	action_on_cosets::lint_Points = Points;
	action_on_cosets::A_linear = A_linear;
	action_on_cosets::F = F;
	action_on_cosets::dimension_of_subspace = dimension_of_subspace;
	action_on_cosets::n = n;
	action_on_cosets::subspace_basis = subspace_basis;
	action_on_cosets::base_cols = base_cols;
	action_on_cosets::unrank_point_lint = unrank_point;
	action_on_cosets::rank_point_lint = rank_point;
	action_on_cosets::rank_unrank_data = rank_unrank_data;
	v1 = NEW_int(n);
	v2 = NEW_int(n);
	for (i = 0; i < nb_points - 1; i++) {
		if (Points[i] >= Points[i + 1]) {
			cout << "action_on_cosets::init_lint the array Points[] "
					"is not sorted increasingly" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "action_on_cosets::init done" << endl;
		}
}

void action_on_cosets::reduce_mod_subspace(int *v, int verbose_level)
{
	F->Linear_algebra->reduce_mod_subspace(dimension_of_subspace, n,
		subspace_basis, base_cols, v, verbose_level);
}


long int action_on_cosets::compute_image(int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int idx;
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "action_on_cosets::compute_image i = " << i << endl;
		}
	if (i >= nb_points) {
		cout << "action_on_cosets::compute_image "
				"i = " << i << " i >= nb_points" << endl;
		exit(1);
		}
	if (f_lint) {
		(*unrank_point_lint)(v1, lint_Points[i], rank_unrank_data);
	}
	else {
		(*unrank_point)(v1, Points[i], rank_unrank_data);
	}
	if (f_vv) {
		cout << "action_on_cosets::compute_image after unrank:";
		Int_vec_print(cout, v1, n);
		cout << endl;
		}
	
	A_linear->element_image_of_low_level(v1, v2,
			Elt, 0/*verbose_level - 1*/);

	if (f_vv) {
		cout << "action_on_cosets::compute_image "
				"after element_image_of_low_level:";
		Int_vec_print(cout, v2, n);
		cout << endl;
		}

	reduce_mod_subspace(v2, 0 /* verbose_level */);
	
	if (f_vv) {
		cout << "action_on_cosets::compute_image "
				"after reduce_mod_subspace:";
		Int_vec_print(cout, v2, n);
		cout << endl;
		}
	if (f_lint) {
		long int R;

		R = (*rank_point_lint)(v2, rank_unrank_data);
		if (f_vv) {
			cout << "action_on_cosets::compute_image after "
					"rank, R = " << R << endl;
			}
		if (!Sorting.lint_vec_search(lint_Points, nb_points, R, idx, verbose_level)) {
			cout << "action_on_cosets::compute_image image "
					<< R << " not found in list of points (using lint)" << endl;

			Sorting.lint_vec_search(lint_Points, nb_points, R, idx, 2 /*verbose_level*/);

			cout << "action_on_cosets::compute_image image "
					<< R << " not found in list of points (using lint)" << endl;


			exit(1);
			}
	}
	else {
		int r;

		r = (*rank_point)(v2, rank_unrank_data);
		if (f_vv) {
			cout << "action_on_cosets::compute_image after "
					"rank, r = " << r << endl;
			}
		if (!Sorting.int_vec_search(Points, nb_points, r, idx)) {
			cout << "action_on_cosets::compute_image image "
					<< r << " not found in list of points (using int)" << endl;
			exit(1);
			}
	}
	if (f_v) {
		cout << "action_on_cosets::compute_image image "
				"of " << i << " is " << idx << endl;
		}
	return idx;
}



}}}

