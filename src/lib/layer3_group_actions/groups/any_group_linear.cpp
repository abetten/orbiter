/*
 * any_group_linear.cpp
 *
 *  Created on: Sep 27, 2021
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace groups {


any_group_linear::any_group_linear()
{
	Any_group = NULL;
}

any_group_linear::~any_group_linear()
{
}


void any_group_linear::init(
		any_group *Any_group, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group_linear::init" << endl;
	}

	any_group_linear::Any_group = Any_group;
	if (f_v) {
		cout << "any_group_linear::init done" << endl;
	}
}


void any_group_linear::classes_based_on_normal_form(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group_linear::classes_based_on_normal_form" << endl;
	}

	if (!Any_group->f_linear_group) {
		cout << "any_group_linear::classes_based_on_normal_form !Any_group->f_linear_group" << endl;
		exit(1);
	}
	groups::sims *G;
	group_theory_global Group_theory_global;

	G = Any_group->LG->Strong_gens->create_sims(verbose_level);


	if (f_v) {
		cout << "any_group_linear::classes_based_on_normal_form "
				"before Group_theory_global.conjugacy_classes_based_on_normal_forms" << endl;
	}
	Group_theory_global.conjugacy_classes_based_on_normal_forms(
			Any_group->LG->A_linear,
			G,
			Any_group->label,
			Any_group->label_tex,
			verbose_level);
	if (f_v) {
		cout << "any_group_linear::classes_based_on_normal_form "
				"after Group_theory_global.conjugacy_classes_based_on_normal_forms" << endl;
	}

	FREE_OBJECT(G);

	if (f_v) {
		cout << "any_group_linear::classes_based_on_normal_form done" << endl;
	}
}

void any_group_linear::find_singer_cycle(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group_linear::find_singer_cycle" << endl;
	}
	if (!Any_group->f_linear_group) {
		cout << "any_group_linear::find_singer_cycle "
				"not a linear group" << endl;
		exit(1);
	}

	group_theory_global Group_theory_global;

	if (f_v) {
		cout << "any_group_linear::find_singer_cycle "
				"before Group_theory_global.find_singer_cycle" << endl;
	}
	Group_theory_global.find_singer_cycle(Any_group,
			Any_group->A, Any_group->A,
			verbose_level);
	if (f_v) {
		cout << "any_group_linear::find_singer_cycle "
				"after Group_theory_global.find_singer_cycle" << endl;
	}

	if (f_v) {
		cout << "any_group_linear::find_singer_cycle done" << endl;
	}
}

void any_group_linear::isomorphism_Klein_quadric(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);

	if (f_v) {
		cout << "any_group_linear::isomorphism_Klein_quadric" << endl;
	}

	if (!Any_group->f_linear_group) {
		cout << "any_group_linear::isomorphism_Klein_quadric "
				"not a linear group" << endl;
		exit(1);
	}

	field_theory::finite_field *F;
	groups::sims *H;
	orbiter_kernel_system::file_io Fio;

	F = Any_group->LG->F;
	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = Any_group->LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;

	Elt = NEW_int(Any_group->A->elt_size_in_int);


	cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;

	int *M;
	int m, n;
	Fio.Csv_file_support->int_matrix_read_csv(
			fname, M, m, n, verbose_level);

	cout << "Read a set of size " << m << endl;

	if (n != Any_group->A->make_element_size) {
		cout << "n != A->make_element_size" << endl;
		exit(1);
	}





	int i, j, c;
	int Basis1[] = {
#if 1
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1,
#else
			1,0,0,0,0,0,
			0,0,0,0,0,1,
			0,1,0,0,0,0,
			0,0,0,0,-1,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
#endif
	};
	//int Basis1b[36];
	int Basis2[36];
	int An2[37];
	int v[6];
	int w[6];
	int C[36];
	int D[36];
	int E[36];
	int B[] = {
			1,0,0,0,0,0,
			0,0,0,2,0,0,
			1,3,0,0,0,0,
			0,0,0,1,3,0,
			1,0,2,0,0,0,
			0,0,0,2,0,4,
	};
	int Target[] = {
			1,0,0,0,0,0,
			3,2,2,0,0,0,
			1,4,2,0,0,0,
			0,0,0,1,0,0,
			0,0,0,3,2,2,
			0,0,0,1,4,2,
	};
	int Bv[36];
	data_structures::sorting Sorting;

#if 0
	for (i = 0; i < 6; i++) {
		if (Basis1[i] == -1) {
			Basis1b[i] = F->negate(1);
		}
		else {
			Basis1b[i] = Basis1[i];
		}
	}
#endif

	geometry::geometry_global Geo;

	for (i = 0; i < 6; i++) {
		Geo.klein_to_wedge(F, Basis1 + i * 6, Basis2 + i * 6);
	}

	F->Linear_algebra->matrix_inverse(B, Bv, 6, 0 /* verbose_level */);


	for (i = 0; i < m; i++) {

		Any_group->A->Group_element->make_element(
				Elt, M + i * Any_group->A->make_element_size, 0);

		if ((i % 10000) == 0) {
			cout << i << " / " << m << endl;
		}

		if (f_vv) {
			cout << "Element " << i << " / " << m << endl;
			Any_group->A->Group_element->element_print(Elt, cout);
			cout << endl;
		}

		F->Linear_algebra->exterior_square(Elt, An2, 4, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "Exterior square:" << endl;
			Int_matrix_print(An2, 6, 6);
			cout << endl;
		}

		for (j = 0; j < 6; j++) {
			F->Linear_algebra->mult_vector_from_the_left(
					Basis2 + j * 6, An2, v, 6, 6);
					// v[m], A[m][n], vA[n]
			Geo.wedge_to_klein(F, v /* W */, w /*K*/);
			Int_vec_copy(w, C + j * 6, 6);
		}

		int Gram[] = {
				0,1,0,0,0,0,
				1,0,0,0,0,0,
				0,0,0,1,0,0,
				0,0,1,0,0,0,
				0,0,0,0,0,1,
				0,0,0,0,1,0,
		};
		int new_Gram[36];

		F->Linear_algebra->transform_form_matrix(C, Gram,
				new_Gram, 6, 0 /* verbose_level*/);

		if (f_vv) {
			cout << "Transformed Gram matrix:" << endl;
			Int_matrix_print(new_Gram, 6, 6);
			cout << endl;
		}


		if (f_vv) {
			cout << "orthogonal matrix :" << endl;
			Int_matrix_print(C, 6, 6);
			cout << endl;
		}

		F->Linear_algebra->mult_matrix_matrix(Bv, C, D, 6, 6, 6, 0 /*verbose_level */);
		F->Linear_algebra->mult_matrix_matrix(D, B, E, 6, 6, 6, 0 /*verbose_level */);

		F->Projective_space_basic->PG_element_normalize_from_front(
				E, 1, 36);

		if (f_vv) {
			cout << "orthogonal matrix in the special form:" << endl;
			Int_matrix_print(E, 6, 6);
			cout << endl;
		}

		int special_Gram[] = {
				0,0,0,3,4,1,
				0,0,0,4,1,3,
				0,0,0,1,3,4,
				3,4,1,0,0,0,
				4,1,3,0,0,0,
				1,3,4,0,0,0,
		};
		int new_special_Gram[36];

		F->Linear_algebra->transform_form_matrix(E, special_Gram,
				new_special_Gram, 6, 0 /* verbose_level*/);

		if (f_vv) {
			cout << "Transformed special Gram matrix:" << endl;
			Int_matrix_print(new_special_Gram, 6, 6);
			cout << endl;
		}



		c = Sorting.integer_vec_compare(E, Target, 36);
		if (c == 0) {
			cout << "We found it! i=" << i << " element = ";
			Int_vec_print(cout, M + i * Any_group->A->make_element_size, Any_group->A->make_element_size);
			cout << endl;

			cout << "Element :" << endl;
			Any_group->A->Group_element->element_print(Elt, cout);
			cout << endl;

			cout << "exterior square :" << endl;
			Int_matrix_print(An2, 6, 6);
			cout << endl;

			cout << "orthogonal matrix :" << endl;
			Int_matrix_print(C, 6, 6);
			cout << endl;

			cout << "orthogonal matrix in the special form:" << endl;
			Int_matrix_print(E, 6, 6);
			cout << endl;

			//exit(1);
		}


	}

	FREE_int(Elt);
	FREE_int(M);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "any_group_linear::isomorphism_Klein_quadric" << endl;
	}
}



int any_group_linear::subspace_orbits_test_set(
		int len, long int *S, int verbose_level)
{

	//cout << "any_group::subspace_orbits_test_set temporarily disabled" << endl;
	///exit(1);

#if 1
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = true;
	int rk;
	int n;
	field_theory::finite_field *F;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;



	if (f_v) {
		cout << "any_group_linear::subspace_orbits_test_set" << endl;
	}

	if (!Any_group->A->f_is_linear) {
		cout << "any_group_linear::subspace_orbits_test_set !A->f_is_linear" << endl;
		exit(1);
	}

	n = Any_group->A->matrix_group_dimension();
	//n = Group->LG->n;
	if (f_v) {
		cout << "any_group_linear::subspace_orbits_test_set n = " << n << endl;
	}

	F = Any_group->A->matrix_group_finite_field();
	//q = F->q;



	if (f_v) {
		cout << "Testing set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "n=" << n << endl;
	}

	//n = LG->n;
	//F = LG->F;

	orbits_on_subspaces_M = NEW_int(len * n);
	orbits_on_subspaces_base_cols = NEW_int(n);

	F->Projective_space_basic->PG_elements_unrank_lint(
			orbits_on_subspaces_M, len, n, S);

	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				orbits_on_subspaces_M, len, n, n, F->log10_of_q);
	}

	rk = F->Linear_algebra->Gauss_simple(orbits_on_subspaces_M, len, n,
			orbits_on_subspaces_base_cols, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "the matrix has rank " << rk << endl;
	}

	FREE_int(orbits_on_subspaces_base_cols);
	FREE_int(orbits_on_subspaces_M);

	if (rk < len) {
		ret = false;
	}

#if 0
	if (ret) {
		if (f_has_extra_test_func) {
			ret = (*extra_test_func)(this,
					len, S, extra_test_func_data, verbose_level);
		}
	}
#endif

	if (ret) {
		if (f_v) {
			cout << "any_group_linear::subspace_orbits_test_set OK" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "any_group_linear::subspace_orbits_test_set not OK" << endl;
		}
	}
	return ret;
#endif
}


}}}



