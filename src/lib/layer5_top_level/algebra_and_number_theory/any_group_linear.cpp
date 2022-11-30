/*
 * any_group_linear.cpp
 *
 *  Created on: Sep 27, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {

void any_group::classes_based_on_normal_form(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::classes_based_on_normal_form" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::classes_based_on_normal_form !f_linear_group" << endl;
		exit(1);
	}
	groups::sims *G;
	algebra_global_with_action Algebra;

	G = LG->Strong_gens->create_sims(verbose_level);


	Algebra.conjugacy_classes_based_on_normal_forms(LG->A_linear,
			G,
			label,
			label_tex,
			verbose_level);

	FREE_OBJECT(G);
	if (f_v) {
		cout << "any_group::classes_based_on_normal_form done" << endl;
	}
}


void any_group::classes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::classes" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::classes !f_linear_group" << endl;
		exit(1);
	}

	groups::sims *G;

	G = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::classes "
				"before A2->conjugacy_classes_and_normalizers" << endl;
	}
	LG->A2->conjugacy_classes_and_normalizers(G,
			label, label_tex, verbose_level);
	if (f_v) {
		cout << "any_group::classes "
				"after A2->conjugacy_classes_and_normalizers" << endl;
	}

	FREE_OBJECT(G);
	if (f_v) {
		cout << "any_group::classes done" << endl;
	}
}

void any_group::find_singer_cycle(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::find_singer_cycle" << endl;
	}
	if (!f_linear_group) {
		cout << "any_group::find_singer_cycle !f_linear_group" << endl;
		exit(1);
	}

	algebra_global_with_action Algebra;

	Algebra.find_singer_cycle(this,
			A, A,
			verbose_level);
	if (f_v) {
		cout << "any_group::find_singer_cycle done" << endl;
	}
}

void any_group::search_element_of_order(int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::search_element_of_order" << endl;
	}
	if (!f_linear_group) {
		cout << "any_group::search_element_of_order !f_linear_group" << endl;
		exit(1);
	}
	algebra_global_with_action Algebra;

	Algebra.search_element_of_order(this,
			A, A,
			order, verbose_level);

	if (f_v) {
		cout << "any_group::search_element_of_order done" << endl;
	}
}

void any_group::find_standard_generators(int order_a,
		int order_b,
		int order_ab,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::find_standard_generators" << endl;
	}
	if (!f_linear_group) {
		cout << "any_group::find_standard_generators !f_linear_group" << endl;
		exit(1);
	}
	algebra_global_with_action Algebra;

	Algebra.find_standard_generators(this,
			A, A,
			order_a, order_b, order_ab, verbose_level);

	if (f_v) {
		cout << "any_group::find_standard_generators done" << endl;
	}

}


void any_group::isomorphism_Klein_quadric(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);

	if (f_v) {
		cout << "any_group::isomorphism_Klein_quadric" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::isomorphism_Klein_quadric !f_linear_group" << endl;
		exit(1);
	}

	field_theory::finite_field *F;
	groups::sims *H;
	orbiter_kernel_system::file_io Fio;

	F = LG->F;
	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);


	cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;

	int *M;
	int m, n;
	Fio.int_matrix_read_csv(fname, M, m, n, verbose_level);

	cout << "Read a set of size " << m << endl;

	if (n != A->make_element_size) {
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

	for (i = 0; i < 6; i++) {
		F->klein_to_wedge(Basis1 + i * 6, Basis2 + i * 6);
	}

	F->Linear_algebra->matrix_inverse(B, Bv, 6, 0 /* verbose_level */);


	for (i = 0; i < m; i++) {

		A->make_element(Elt, M + i * A->make_element_size, 0);

		if ((i % 10000) == 0) {
			cout << i << " / " << m << endl;
		}

		if (f_vv) {
			cout << "Element " << i << " / " << m << endl;
			A->element_print(Elt, cout);
			cout << endl;
		}

		F->Linear_algebra->exterior_square(Elt, An2, 4, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "Exterior square:" << endl;
			Int_matrix_print(An2, 6, 6);
			cout << endl;
		}

		for (j = 0; j < 6; j++) {
			F->Linear_algebra->mult_vector_from_the_left(Basis2 + j * 6, An2, v, 6, 6);
					// v[m], A[m][n], vA[n]
			F->wedge_to_klein(v /* W */, w /*K*/);
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

		F->PG_element_normalize_from_front(E, 1, 36);

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
			Int_vec_print(cout, M + i * A->make_element_size, A->make_element_size);
			cout << endl;

			cout << "Element :" << endl;
			A->element_print(Elt, cout);
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
		cout << "any_group::isomorphism_Klein_quadric" << endl;
	}
}

void any_group::do_orbits_on_subspaces(
		poset_classification::poset_classification_control *Control,
		orbits_on_subspaces *&OoS,
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_orbits_on_subspaces" << endl;
	}





	if (!f_linear_group) {
		cout << "any_group::do_orbits_on_subspaces !f_linear_group" << endl;
		exit(1);
	}


	//orbits_on_subspaces *OoS;

	OoS = NEW_OBJECT(orbits_on_subspaces);

	OoS->init(this, Control, depth, verbose_level);


	//finite_field *F;

	//F = LG->F;


	//FREE_OBJECT(OoS);


	if (f_v) {
		cout << "any_group::do_orbits_on_subspaces done" << endl;
	}
}

void any_group::do_tensor_classify(
		std::string &control_label,
		apps_geometry::tensor_classify *&T,
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_tensor_classify" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::do_tensor_classify !f_linear_group" << endl;
		exit(1);
	}

	field_theory::finite_field *F;

	F = LG->F;

	poset_classification::poset_classification_control *Control =
			Get_object_of_type_poset_classification_control(control_label);



	//apps_geometry::tensor_classify *T;

	T = NEW_OBJECT(apps_geometry::tensor_classify);

	if (f_v) {
		cout << "any_group::do_tensor_classify before T->init" << endl;
	}
	T->init(F, LG, verbose_level - 1);
	if (f_v) {
		cout << "any_group::do_tensor_classify after T->init" << endl;
	}

	if (f_v) {
		cout << "any_group::do_tensor_classify before classify_poset" << endl;
	}
	T->classify_poset(depth,
			Control,
			verbose_level);
	if (f_v) {
		cout << "any_group::do_tensor_classify after classify_poset" << endl;
	}



	//FREE_OBJECT(T);

	if (f_v) {
		cout << "any_group::do_tensor_classify done" << endl;
	}
}


void any_group::do_tensor_permutations(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_tensor_permutations" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::do_tensor_permutations !f_linear_group" << endl;
		exit(1);
	}
	field_theory::finite_field *F;

	F = LG->F;


	apps_geometry::tensor_classify *T;

	T = NEW_OBJECT(apps_geometry::tensor_classify);

	T->init(F, LG, verbose_level - 1);


	FREE_OBJECT(T);

	if (f_v) {
		cout << "any_group::do_tensor_permutations done" << endl;
	}
}


void any_group::do_linear_codes(
		std::string &control_label,
		int minimum_distance,
		int target_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_linear_codes" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::do_linear_codes !f_linear_group" << endl;
		exit(1);
	}

	poset_classification::poset_classification_control *Control =
			Get_object_of_type_poset_classification_control(control_label);

	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "any_group::do_linear_codes before "
				"Algebra.linear_codes_with_bounded_minimum_distance" << endl;
	}

	Algebra.linear_codes_with_bounded_minimum_distance(
			Control, LG,
			minimum_distance, target_size, verbose_level);

	if (f_v) {
		cout << "any_group::do_linear_codes after "
				"Algebra.linear_codes_with_bounded_minimum_distance" << endl;
	}


	if (f_v) {
		cout << "any_group::do_linear_codes done" << endl;
	}
}

void any_group::do_classify_ovoids(
		apps_geometry::ovoid_classify_description *Ovoid_classify_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_classify_ovoids" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::do_classify_ovoids !f_linear_group" << endl;
		exit(1);
	}

	apps_geometry::ovoid_classify *Ovoid_classify;


	Ovoid_classify = NEW_OBJECT(apps_geometry::ovoid_classify);


	Ovoid_classify->init(Ovoid_classify_description,
			LG,
			verbose_level);

	FREE_OBJECT(Ovoid_classify);

	if (f_v) {
		cout << "any_group::do_classify_ovoids done" << endl;
	}
}



int any_group::subspace_orbits_test_set(
		int len, long int *S, int verbose_level)
{

	//cout << "any_group::subspace_orbits_test_set temporarily disabled" << endl;
	///exit(1);

#if 1
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int rk;
	int n;
	field_theory::finite_field *F;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;



	if (f_v) {
		cout << "any_group::subspace_orbits_test_set" << endl;
	}

	if (!f_linear_group) {
		cout << "any_group::subspace_orbits_test_set !f_linear_group" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "Testing set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "LG->n=" << LG->n << endl;
	}

	n = LG->n;
	F = LG->F;

	orbits_on_subspaces_M = NEW_int(len * n);
	orbits_on_subspaces_base_cols = NEW_int(n);

	F->PG_elements_unrank_lint(
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
		ret = FALSE;
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
			cout << "any_group::subspace_orbits_test_set OK" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "any_group::subspace_orbits_test_set not OK" << endl;
		}
	}
	return ret;
#endif
}


}}}



