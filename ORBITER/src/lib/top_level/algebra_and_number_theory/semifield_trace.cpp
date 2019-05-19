/*
 * semifield_trace.cpp
 *
 *  Created on: Apr 18, 2019
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_trace::semifield_trace()
{
	SC = NULL;
	SL = NULL;
	L2 = NULL;
	A = NULL;
	F = NULL;
	n = k = k2 = 0;
	ELT1 = ELT2 = ELT3 = NULL;
	M1 = NULL;
	Basis = NULL;
	basis_tmp = NULL;
	base_cols = NULL;
	R1 = NULL;
	//null();
}

semifield_trace::~semifield_trace()
{
	if (ELT1) {
		FREE_int(ELT1);
	}
	if (ELT2) {
		FREE_int(ELT2);
	}
	if (ELT3) {
		FREE_int(ELT3);
	}
	if (M1) {
		FREE_int(M1);
	}
	if (Basis) {
		FREE_int(Basis);
	}
	if (basis_tmp) {
		FREE_int(basis_tmp);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (R1) {
		FREE_OBJECT(R1);
	}
	//freeself();
}

void semifield_trace::init(semifield_lifting *SL)
{
	semifield_trace::SL = SL;
	SC = SL->SC;
	L2 = SL->L2;
	A = SC->A;
	F = SC->F;
	k = SC->k;
	k2 = k * k;
	n = 2 * k;
	ELT1 = NEW_int(A->elt_size_in_int);
	ELT2 = NEW_int(A->elt_size_in_int);
	ELT3 = NEW_int(A->elt_size_in_int);


	M1 = NEW_int(n * n);
	Basis = NEW_int(k * k);
	basis_tmp = NEW_int(k /* basis_sz */ * k2);
	base_cols = NEW_int(k2);

	R1 = NEW_OBJECT(gl_class_rep);
}

void semifield_trace::trace_very_general(
	int cur_level,
	int *input_basis, int basis_sz,
	int *basis_after_trace, int *transporter,
	int &trace_po, int &trace_so,
	int verbose_level)
// input basis is input_basis of size basis_sz x k2
// there is a check if input_basis defines a semifield
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, idx, d, d0, po, c0, c1;

	if (f_v) {
		cout << "semifield_trace::trace_very_general" << endl;
		}
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"input basis:" << endl;
		SC->basis_print(input_basis, basis_sz);
		}
	if (!SC->test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial semifield condition" << endl;
		exit(1);
		}




	// Step 1:
	// trace the first matrix (which becomes the identity matrix):

	// create the n x n matrix which is a 2 x 2 block matrix
	// (A 0)
	// (0 I)
	// where A is input_basis
	// the resulting matrix will be put in transporter
	int_vec_zero(M1, n * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			M1[i * n + j] = input_basis[i * k + j];
			}
		}
	for (i = k; i < n; i++) {
		M1[i * n + i] = 1;
		}
	A->make_element(transporter, M1, 0);

	if (f_vv) {
		cout << "transformation matrix transporter=" << endl;
		int_matrix_print(transporter, n, n);
		cout << "transformation matrix M1=" << endl;
		int_matrix_print(M1, n, n);
		}

	// apply transporter to elements 0,...,basis_sz - 1 of input_basis
	SC->apply_element_and_copy_back(transporter,
		input_basis, basis_tmp,
		0, basis_sz, verbose_level);
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"after transform (1):" << endl;
		SC->basis_print(input_basis, basis_sz);
		}
	if (!F->is_identity_matrix(input_basis, k)) {
		cout << "semifield_trace::trace_very_general "
				"basis_tmp is not the identity matrix" << endl;
		exit(1);
		}



	// Step 2:
	// Trace the second matrix using rational normal forms.
	// Do adjustment to get the right coset representative.
	// Apply fusion element if necessary

	L2->C->identify_matrix(input_basis + 1 * k2, R1, Basis, 0 /* verbose_level */);

	idx = L2->C->find_class_rep(L2->R, L2->nb_classes, R1, 0 /* verbose_level */);
	d = L2->class_to_flag_orbit[idx];
	if (f_vv) {
		cout << "semifield_starter::trace_very_general "
				"the second matrix belongs to conjugacy class "
				<< idx << " which is in down orbit " << d << endl;
		}

	L2->multiply_to_the_right(transporter, Basis, ELT2, ELT3, 0 /* verbose_level */);
	A->element_move(ELT3, transporter, 0);

	// apply ELT2 (i.e., Basis) to input_basis elements 1, .., basis_sz - 1
	SC->apply_element_and_copy_back(ELT2,
		input_basis, basis_tmp,
		1, basis_sz, verbose_level);
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"after transform (2):" << endl;
		SC->basis_print(input_basis, basis_sz);
		}


	c0 = L2->flag_orbit_classes[d * 2 + 0];
	c1 = L2->flag_orbit_classes[d * 2 + 1];
	if (c0 != idx && c1 == idx) {
		if (f_vv) {
			cout << "Adjusting" << endl;
			}
		L2->multiply_to_the_right(transporter,
				L2->class_rep_plus_I_Basis_inv[c0],
				ELT2, ELT3,
				0 /* verbose_level */);
		A->element_move(ELT3, transporter, 0);

		// apply ELT2 to the basis elements 1,...,basis_sz - 1:
		SC->apply_element_and_copy_back(ELT2,
			input_basis, basis_tmp,
			1, basis_sz, verbose_level);
		if (f_vv) {
			cout << "semifield_trace::trace_very_general "
					"after transform because of adjustment:" << endl;
			SC->basis_print(input_basis, basis_sz);
			}
		// subtract off the first matrix (is this really the identity?)
		for (i = 0; i < k2; i++) {
			input_basis[1 * k2 + i] = F->add(
					input_basis[1 * k2 + i], F->negate(input_basis[i]));
			}
		if (f_vv) {
			cout << "semifield_trace::trace_very_general "
					"after subtracting the identity:" << endl;
			SC->basis_print(input_basis, basis_sz);
			}
		}
	else {
		if (f_vv) {
			cout << "No adjustment needed" << endl;
			}
		}

	if (L2->f_Fusion[d]) {
		if (f_vv) {
			cout << "Applying fusion element" << endl;
			}
		if (L2->Fusion_elt[d] == NULL) {
			cout << "Fusion_elt[d] == NULL" << endl;
			exit(1);
			}
		d0 = L2->Fusion_idx[d];
		A->element_mult(transporter, L2->Fusion_elt[d], ELT3, 0);
		A->element_move(ELT3, transporter, 0);

		// apply Fusion_elt[d] to the basis elements 0,1,...,basis_sz - 1
		SC->apply_element_and_copy_back(L2->Fusion_elt[d],
			input_basis, basis_tmp,
			0, basis_sz, verbose_level);
		if (f_vv) {
			cout << "semifield_starter::trace_very_general "
					"after transform (3):" << endl;
			SC->basis_print(input_basis, basis_sz);
			}
		if (input_basis[0] == 0) {
			// add the second matrix to the first:
			for (j = 0; j < k2; j++) {
				input_basis[j] = F->add(
						input_basis[j], input_basis[k2 + j]);
				}
			}
		// now, input_basis[0] != 0
		if (input_basis[0] == 0) {
			cout << "input_basis[0] == 0" << endl;
			exit(1);
			}
		if (input_basis[0] != 1) {
			int lambda;

			lambda = F->inverse(input_basis[0]);
			for (j = 0; j < k2; j++) {
				input_basis[j] = F->mult(input_basis[j], lambda);
				}
			}
		if (input_basis[0] != 1) {
			cout << "input_basis[0] != 1" << endl;
			exit(1);
			}
		if (input_basis[k2]) {
			int lambda;
			lambda = F->negate(input_basis[k2]);
			for (j = 0; j < k2; j++) {
				input_basis[k2 + j] = F->add(
						input_basis[k2 + j],
						F->mult(input_basis[j], lambda));
				}
			}
		if (input_basis[k]) {
			int lambda;
			lambda = F->negate(input_basis[k]);
			for (j = 0; j < k2; j++) {
				input_basis[j] = F->add(
						input_basis[j],
						F->mult(input_basis[k2 + j], lambda));
				}
			}
		if (input_basis[k]) {
			cout << "input_basis[k] (should be zero by now)" << endl;
			exit(1);
			}
		if (f_vv) {
			cout << "semifield_starter::trace_very_general "
					"after gauss elimination:" << endl;
			SC->basis_print(input_basis, basis_sz);
			}
		}
	else {
		if (f_vv) {
			cout << "No fusion" << endl;
			}
		d0 = d;
		}
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"d0 = " << d0 << endl;
		}
	if (L2->Fusion_elt[d0]) {
		cout << "Fusion_elt[d0]" << endl;
		exit(1);
		}
	po = L2->Fusion_idx[d0];
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"po = " << po << endl;
		}


	// Step 2 almost finished.
	// Next we need to compute the reduced coset representatives
	// for the remaining elements
	// w.r.t. the basis and the pivots in base_col

	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"we will now compute the reduced coset reps:" << endl;
		}

	base_cols[0] = 0;
	base_cols[1] = k;
	if (f_vv) {
		cout << "semifield_trace::trace_very_general base_cols=";
		int_vec_print(cout, base_cols, 2);
		cout << endl;
		}
	for (i = 0; i < 2; i++) {
		for (j = 2; j < basis_sz; j++) {
			F->Gauss_step(input_basis + i * k2,
					input_basis + j * k2, k2, base_cols[i],
					0 /*verbose_level*/);
			}
		}
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"reduced basis=" << endl;
		int_matrix_print(input_basis, basis_sz, k2);
		cout << "Which is:" << endl;
		SC->basis_print(input_basis, basis_sz);
		}
	if (!SC->test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial "
				"semifield condition" << endl;
		exit(1);
		}


#if 0

	// ToDo


	int a, a_local, pos, so;


	// Step 3:
	// Locate the third matrix, compute its rank,
	// and find the rank in the candidates array to compute the local point.
	// Then find the point in the schreier structure
	// and compute a coset representative.
	// This coset representative stabilizes the subspace which is
	// generated by the first two vectors, so when applying the mapping,
	// we can skip the first two vectors.

	a = SC->matrix_rank(input_basis + 2 * k2);
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"a = " << a << endl;
		}

	a_local = Level_two_down[po].find_point(a);
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"a_local = " << a_local << endl;
		}

	pos = Level_two_down[po].Sch->orbit_inv[a_local];
	so = Level_two_down[po].Sch->orbit_number(a_local);
		// Level_two_down[po].Sch->orbit_no[pos];

	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"so = " << so << endl;
		}
	trace_po = po;
	trace_so = so;

	Level_two_down[po].Sch->coset_rep_inv(pos);
	A->element_mult(transporter,
			Level_two_down[po].Sch->cosetrep,
			ELT3,
			0 /* verbose_level */);
	A->element_move(ELT3, transporter, 0);
	// apply cosetrep to base elements 2,...,basis_sz - 1:
	apply_element_and_copy_back(
		Level_two_down[po].Sch->cosetrep,
		input_basis, basis_tmp,
		2, basis_sz, verbose_level);
#if 0
	for (i = 2; i < basis_sz; i++) {
		SF->A_on_S->compute_image_low_level(
				Level_two_down[po].Sch->cosetrep,
				input_basis + i * k2,
				basis_tmp + i * k2,
				0 /* verbose_level */);
		}
	int_vec_copy(basis_tmp + 2 * k2,
			input_basis + 2 * k2,
			(basis_sz - 2) * k2);
#endif
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"after transforming with cosetrep from "
				"secondary orbit (4):" << endl;
		basis_print(input_basis, basis_sz);
		}
	base_cols[0] = 0;
	base_cols[1] = k;
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"base_cols=";
		int_vec_print(cout, base_cols, 2);
		cout << endl;
		}
	for (i = 0; i < 2; i++) {
		for (j = 2; j < basis_sz; j++) {
			F->Gauss_step(input_basis + i * k2,
					input_basis + j * k2, k2,
					base_cols[i],
					0 /*verbose_level*/);
			}
		}
	if (f_vv) {
		cout << "semifield_trace::trace_very_general "
				"reduced basis(2)=" << endl;
		int_matrix_print(input_basis, basis_sz, k2);
		cout << "Which is:" << endl;
		basis_print(input_basis, basis_sz);
		}

#if 0
	if (!test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial semifield condition" << endl;
		exit(1);
		}
#endif


	if (cur_level >= 3) {
		// we need to keep going, since we are working on level 4 or higher:
		//f_vv = TRUE;

		if (f_vv) {
			cout << "semifield_trace::trace_very_general "
					"keep going since cur_level >= 3" << endl;
			cout << "po=" << po << " so=" << so << endl;
			}

		semifield_downstep_node *D;
		semifield_downstep_node *D1;
		middle_layer_node *M;

		D = Level_two_down;
		D1 = Level_three_down;
		M = Level_two_middle;

		trace_step(3 /* step */,
			po, so,
			input_basis, basis_sz, basis_tmp,
			transporter, ELT3,
			D,
			D1,
			M,
			verbose_level);

#if 0
		if (!test_partial_semifield(input_basis,
				basis_sz, 0 /* verbose_level */)) {
			cout << "does not satisfy the partial "
					"semifield condition" << endl;
			exit(1);
			}
#endif
		trace_po = po;
		trace_so = so;


		}

	if (cur_level >= 4) {
		// we need to keep going, since we are working on level 5 or higher:
		//f_vv = TRUE;

		if (f_vv) {
			cout << "semifield_trace::trace_very_general "
					"keep going since cur_level >= 4" << endl;
			cout << "po=" << po << " so=" << so << endl;
			}

		semifield_downstep_node *D;
		semifield_downstep_node *D1;
		semifield_middle_layer_node *M;

		D = Level_three_down;
		D1 = Level_four_down;
		M = Level_three_middle;

		trace_step(4 /* step */,
			po, so,
			input_basis, basis_sz, basis_tmp,
			transporter, ELT3,
			D,
			D1,
			M,
			verbose_level);

#if 0
		if (!test_partial_semifield(input_basis,
				basis_sz, 0 /* verbose_level */)) {
			cout << "does not satisfy the partial "
					"semifield condition" << endl;
			exit(1);
			}
#endif
		trace_po = po;
		trace_so = so;


		}
#endif


	if (f_v) {
		cout << "semifield_trace::trace_very_general done" << endl;
		}
}


#if 0
int semifield_trace::trace_to_level_three(semifield_lifting *SL,
	int *input_basis, int basis_sz, int *transporter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int trace_po;
	int trace_so;
	int *Elt1;
	int *basis_tmp;


	if (f_v) {
		cout << "semifield_trace::trace_to_level_three" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	basis_tmp = NEW_int(basis_sz * k2);

	trace_very_general(2,
		input_basis, basis_sz, basis_tmp, transporter,
		trace_po, trace_so,
		verbose_level - 4);

	semifield_downstep_node *D;
	semifield_downstep_node *D1;
	semifield_middle_layer_node *M;

	D = Level_two_down;
	D1 = Level_three_down;
	M = Level_two_middle;

	if (f_vv) {
		cout << "semifield_trace::trace_to_level_three "
				"before trace_step_up" << endl;
		}

	trace_step_up(3 /* step */,
		trace_po, trace_so,
		input_basis, basis_sz, basis_tmp,
		transporter, Elt1,
		D,
		D1,
		M,
		verbose_level - 4);

	FREE_int(Elt1);
	FREE_int(basis_tmp);

	if (f_v) {
		cout << "semifield_trace::trace_to_level_three "
				"done" << endl;
		}
	return trace_po;
}


void semifield_trace::trace_general(
	int cur_level, int cur_po, int cur_so,
	int *input_basis, int *basis_after_trace, int *transporter,
	int &trace_po, int &trace_so,
	int verbose_level)
// input_basis has size (cur_level + 1) x k2
// there is a check if input_basis defines a semifield
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_trace::trace_general "
				"before trace_very_general" << endl;
		}
	trace_very_general(cur_level,
		input_basis,
		cur_level + 1,
		basis_after_trace,
		transporter,
		trace_po, trace_so,
		verbose_level - 1);
	if (f_v) {
		cout << "semifield_trace::trace_general "
				"after trace_very_general" << endl;
		}
}

void semifield_trace::trace_step(
	int step,
	int &po, int &so,
	int *changed_basis, int basis_sz, int *basis_tmp,
	int *transporter, int *ELT3,
	semifield_downstep_node *D,
	semifield_downstep_node *D1,
	semifield_middle_layer_node *M,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_trace::trace_step "
				"step = " << step << endl;
		}


	trace_step_up(step,
		po, so,
		changed_basis, basis_sz, basis_tmp,
		transporter, ELT3,
		D,
		D1,
		M,
		verbose_level - 1);


	trace_step_down(step,
		po, so,
		changed_basis, basis_sz, basis_tmp,
		transporter, ELT3,
		D,
		D1,
		M,
		verbose_level - 1);



	if (f_v) {
		cout << "semifield_trace::trace_step "
				"step = " << step << " done" << endl;
		}

}

void semifield_trace::trace_step_up(
	int step,
	int &po, int &so,
	int *changed_basis, int basis_sz, int *basis_tmp,
	int *transporter, int *ELT3,
	semifield_downstep_node *D,
	semifield_downstep_node *D1,
	semifield_middle_layer_node *M,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int mo, m0;
	int i, j;

	if (f_v) {
		cout << "semifield_trace::trace_step_up "
				"step = " << step << endl;
		}
	mo = D[po].first_middle_orbit + so;
	if (f_vv) {
		cout << "semifield_trace::trace_step_up "
				"mo = " << mo << endl;
		}
	if (M[mo].f_fusion_node) {
		if (f_vv) {
			cout << "semifield_trace::trace_step_up "
					"fusion node" << endl;
			}
		m0 = M[mo].fusion_with;
		A->element_mult(transporter,
				M[mo].fusion_elt,
				ELT3,
				0 /* verbose_level */);
		A->element_move(ELT3, transporter, 0);
		apply_element_and_copy_back(M[mo].fusion_elt,
			changed_basis, basis_tmp,
			0, basis_sz, verbose_level);
#if 0
		for (i = 0; i < basis_sz; i++) {
			SF->A_on_S->compute_image_low_level(
					M[mo].fusion_elt,
					changed_basis + i * k2,
					basis_tmp + i * k2,
					0 /* verbose_level */);
			}
		int_vec_copy(basis_tmp + 0 * k2,
				changed_basis + 0 * k2,
				(basis_sz - 0) * k2);
#endif
		if (f_vv) {
			cout << "semifield_trace::trace_step_up "
					"after fusion:" << endl;
			int_matrix_print(changed_basis, basis_sz, k2);
			basis_print(changed_basis, basis_sz);
			}
#if 0
		if (!test_partial_semifield(changed_basis,
				basis_sz, 0 /* verbose_level */)) {
			cout << "does not satisfy the partial semifield condition" << endl;
			exit(1);
			}
#endif
		//exit(1);
		}
	else {
		m0 = mo;
		}
	if (f_vv) {
		cout << "semifield_trace::trace_step_up "
				"m0 = " << m0 << endl;
		}
	po = M[m0].upstep_orbit;
	if (f_vv) {
		cout << "semifield_trace::trace_step_up "
				"po = " << po << endl;
		}
	if (po == -1) {
		cout << "semifield_trace::trace_step_up "
				"po == -1" << endl;
		exit(1);
		}

	int *pivots;

	pivots = NEW_int(step);
	get_pivots(step /* level */, M[m0].upstep_orbit,
			pivots, verbose_level - 3);

	if (f_vv) {
		cout << "semifield_trace::trace_step_up "
				"pivots=";
		int_vec_print(cout, pivots, step);
		cout << endl;
		}
	F->Gauss_int_with_given_pivots(
		changed_basis,
		FALSE /* f_special */,
		TRUE /* f_complete */,
		pivots,
		step /* nb_pivots */,
		basis_sz /* m */,
		k2 /* n */,
		0 /*verbose_level*/);
	if (f_vv) {
		cout << "semifield_trace::trace_step_up "
				"after Gauss_int_with_given_pivots:" << endl;
		int_matrix_print(changed_basis, basis_sz, k2);
		}
#if 0
	if (!test_partial_semifield(changed_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial "
				"semifield condition" << endl;
		exit(1);
		}
#endif
	for (i = 0; i < step; i++) {
		for (j = step; j < basis_sz; j++) {
			F->Gauss_step(changed_basis + i * k2,
					changed_basis + j * k2, k2,
					pivots[i], 0 /*verbose_level*/);
			}
		}
	if (f_vv) {
		cout << "semifield_trace::trace_step_up "
				"after reducing:" << endl;
		int_matrix_print(changed_basis, basis_sz, k2);
		basis_print(changed_basis, basis_sz);
		}
#if 0
	if (!test_partial_semifield(changed_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial semifield condition" << endl;
		exit(1);
		}
#endif

	FREE_int(pivots);
	if (f_v) {
		cout << "semifield_trace::trace_step_up "
				"step = " << step << " done" << endl;
		}
}

void semifield_trace::trace_step_down(
	int step,
	int &po, int &so,
	int *changed_basis, int basis_sz, int *basis_tmp,
	int *transporter, int *ELT3,
	semifield_downstep_node *D,
	semifield_downstep_node *D1,
	semifield_middle_layer_node *M,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int a, a_local, pos;

	if (f_v) {
		cout << "semifield_trace::trace_step_down "
				"step = " << step << endl;
		}
	if (f_vv) {
		cout << "Elt " << step << endl;
		int_matrix_print(changed_basis + step * k2, k, k);
		}
	a = matrix_rank(changed_basis + step * k2);
	if (f_vv) {
		cout << "semifield_trace::trace_step_down "
				"a = " << a << " po = " << po << endl;
		}

	a_local = D1[po].find_point(a);
	if (f_vv) {
		cout << "semifield_trace::trace_step_down "
				"a_local = " << a_local << endl;
		}

	pos = D1[po].Sch->orbit_inv[a_local];
	so = D1[po].Sch->orbit_number(a_local);
		// D1[po].Sch->orbit_no[pos];


	if (f_vv) {
		cout << "semifield_trace::trace_step_down "
				"so = " << so << endl;
		}
	D1[po].Sch->coset_rep_inv(pos);
	A->element_mult(transporter,
			D1[po].Sch->cosetrep,
			ELT3,
			0 /* verbose_level */);
	A->element_move(ELT3, transporter, 0);
	apply_element_and_copy_back(D1[po].Sch->cosetrep,
		changed_basis, basis_tmp,
		step, basis_sz, verbose_level);
#if 0
	for (i = step; i < basis_sz; i++) {
		SF->A_on_S->compute_image_low_level(
				D1[po].Sch->cosetrep,
				changed_basis + i * k2,
				basis_tmp + i * k2,
				0 /* verbose_level */);
		}
	int_vec_copy(basis_tmp + step * k2,
			changed_basis + step * k2,
			(basis_sz - step) * k2);
#endif
	if (f_vv) {
		cout << "semifield_trace::trace_step_down "
				"after transforming with cosetrep from "
				"secondary orbit:" << endl;
		basis_print(changed_basis, basis_sz);
#if 0
		for (i = 0; i < basis_sz; i++) {
			cout << "Elt i = " << i << endl;
			int_matrix_print(changed_basis + i * k2, k, k);
			}
#endif
		}
	if (f_v) {
		cout << "semifield_trace::trace_step_down "
				"step = " << step << " done" << endl;
		}
}

#endif








}}


