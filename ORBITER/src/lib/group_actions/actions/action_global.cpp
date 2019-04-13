// action_global.C
//
// Anton Betten
// October 10, 2013

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

void action_print_symmetry_group_type(ostream &ost,
		symmetry_group_type a)
{
	if (a == unknown_symmetry_group_t) {
		ost << "unknown_symmetry_group_t";
		}
	else if (a == matrix_group_t) {
		ost << "matrix_group_t";
		}
	else if (a == perm_group_t) {
		ost << "perm_group_t";
		}
	else if (a == wreath_product_t) {
		ost << "wreath_product_t";
		}
	else if (a == direct_product_t) {
		ost << "direct_product_t";
		}
	else if (a == action_on_sets_t) {
		ost << "action_on_sets_t";
		}
	else if (a == action_on_set_partitions_t) {
		ost << "action_on_set_partitions_t";
		}
	else if (a == action_on_subgroups_t) {
		ost << "action_on_subgroups_t";
		}
	else if (a == action_on_pairs_t) {
		ost << "action_on_pairs_t";
		}
	else if (a == action_on_ordered_pairs_t) {
		ost << "action_on_ordered_pairs_t";
		}
	else if (a == base_change_t) {
		ost << "base_change_t";
		}
	else if (a == product_action_t) {
		ost << "product_action_t";
		}
	else if (a == action_by_right_multiplication_t) {
		ost << "action_by_right_multiplication_t";
		}
	else if (a == action_by_restriction_t) {
		ost << "action_by_restriction_t";
		}
	else if (a == action_by_conjugation_t) {
		ost << "action_by_conjugation_t";
		}
	else if (a == action_by_representation_t) {
		ost << "action_by_representation_t";
		}
	else if (a == action_by_subfield_structure_t) {
		ost << "action_by_subfield_structure_t";
		}
	else if (a == action_on_determinant_t) {
		ost << "action_on_determinant_t";
		}
	else if (a == action_on_galois_group_t) {
		ost << "action_on_galois_group_t";
		}
	else if (a == action_on_sign_t) {
		ost << "action_on_sign_t";
		}
	else if (a == action_on_grassmannian_t) {
		ost << "action_on_grassmannian_t";
		}
	else if (a == action_on_spread_set_t) {
		ost << "action_on_spread_set_t";
		}
	else if (a == action_on_cosets_t) {
		ost << "action_on_cosets_t";
		}
	else if (a == action_on_factor_space_t) {
		ost << "action_on_factor_space_t";
		}
	else if (a == action_on_wedge_product_t) {
		ost << "action_on_wedge_product_t";
		}
	else if (a == action_on_bricks_t) {
		ost << "action_on_bricks_t";
		}
	else if (a == action_on_andre_t) {
		ost << "action_on_andre_t";
		}
	else if (a == action_on_orthogonal_t) {
		ost << "action_on_orthogonal_t";
		}
	else if (a == action_on_orbits_t) {
		ost << "action_on_orbits_t";
		}
	else if (a == action_on_flags_t) {
		ost << "action_on_flags_t";
		}
	else if (a == action_on_homogeneous_polynomials_t) {
		ost << "action_on_homogeneous_polynomials_t";
		}
	else {
		ost << "unknown symmetry_group_type" << endl;
		}
}



void make_generators_stabilizer_of_two_components(
	action *A_PGL_n_q, action *A_PGL_k_q,
	int k, vector_ge *gens, int verbose_level)
// used in semifield.C
// does not include the swap
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Q;
	int *Elt1;
	int *Zero;
	int *Id;
	int *Center;
	int *minusId;
	int n, i, len;
	int *P;
	matrix_group *Mtx;
	finite_field *Fq;
	int minus_one, alpha;
	strong_generators *gens_PGL_k;
	//vector_ge *gens_PGL_k;


	if (f_v) {
		cout << "make_generators_stabilizer_of_two_components" << endl;
		}
	n = 2 * k;

	Zero = NEW_int(k * k);
	Id = NEW_int(k * k);
	Center = NEW_int(k * k);
	minusId = NEW_int(k * k);
	Q = NEW_int(n * n + 1);
	Elt1 = NEW_int(A_PGL_n_q->elt_size_in_int);


	Mtx = A_PGL_k_q->G.matrix_grp;
	Fq = Mtx->GFq;
	minus_one = Fq->negate(1);
	alpha = Fq->primitive_root();

	int_vec_zero(Zero, k * k);
	int_vec_zero(Id, k * k);
	int_vec_zero(Center, k * k);
	int_vec_zero(minusId, k * k);
	for (i = 0; i < k; i++) {
		Id[i * k + i] = 1;
		}
	for (i = 0; i < k; i++) {
		Center[i * k + i] = alpha;
		}
	for (i = 0; i < k; i++) {
		minusId[i * k + i] = minus_one;
		}

	gens_PGL_k = A_PGL_k_q->Strong_gens;
	//gens_PGL_k = A_PGL_k_q->strong_generators;
	
	len = gens_PGL_k->gens->len;
	//len = gens_PGL_k->len;

	int *Data;
	int new_len, sz, idx, h;

	new_len = 2 * len + 2;
	sz = n * n;
	if (Mtx->f_semilinear) {
		sz++;
		}
	

	Data = NEW_int(new_len * sz);
	idx = 0;
	for (h = 0; h < 2 * len; h++) {

		P = gens_PGL_k->gens->ith(h / 2);
		//P = gens_PGL_k->ith(h / 2);

		if (EVEN(h)) {
			// Q := diag(P,Id)
			int_matrix_make_block_matrix_2x2(Q, k, P, Zero, Zero, Id);
			}
		else {
			// Q := diag(Id,P)
			int_matrix_make_block_matrix_2x2(Q, k, Id, Zero, Zero, P);
			}
		if (Mtx->f_semilinear) {
			Q[n * n] = P[k * k];
			}
		int_vec_copy(Q, Data + idx * sz, sz);
		idx++;
		}

#if 0
	// Q := matrix(0,I,I,0):
	int_matrix_make_block_matrix_2x2(Q, k, Zero, Id, Id, Zero);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
		}
	int_vec_copy(Q, Data + idx * sz, sz);
	idx++;
#endif

	// Q := matrix(Center,0,0,I):
	int_matrix_make_block_matrix_2x2(Q, k, Center, Zero, Zero, Id);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
		}
	int_vec_copy(Q, Data + idx * sz, sz);
	idx++;

	// Q := matrix(I,0,0,Center):
	int_matrix_make_block_matrix_2x2(Q, k, Id, Zero, Zero, Center);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
		}
	int_vec_copy(Q, Data + idx * sz, sz);
	idx++;


	if (idx != new_len) {
		cout << "make_generators_stabilizer_of_two_components "
				"idx != new_len" << endl;
		exit(1);
		}



	gens->init(A_PGL_n_q);
	gens->allocate(new_len);
	for (h = 0; h < new_len; h++) {
		A_PGL_n_q->make_element(Elt1, Data + h * sz, 0);
		if (f_vv) {
			cout << "make_generators_stabilizer_of_two_components "
					"after make_element generator " << h << " : " << endl;
			A_PGL_n_q->print_quick(cout, Elt1);
			}
		A_PGL_n_q->move(Elt1, gens->ith(h));
		}
	

	FREE_int(Data);

	FREE_int(Zero);
	FREE_int(Id);
	FREE_int(Center);
	FREE_int(minusId);
	FREE_int(Q);
	FREE_int(Elt1);
	if (f_v) {
		cout << "make_generators_stabilizer_of_two_components done" << endl;
		}
}


void make_generators_stabilizer_of_three_components(
	action *A_PGL_n_q, action *A_PGL_k_q,
	int k, vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Q;
	int *Elt1;
	int *Zero;
	int *Id;
	int *minusId;
	int n, i, len;
	int *P;
	matrix_group *Mtx;
	finite_field *Fq;
	int minus_one;
	strong_generators *gens_PGL_k;

	if (f_v) {
		cout << "make_generators_stabilizer_of_three_components" << endl;
		}
	n = 2 * k;

	Zero = NEW_int(k * k);
	Id = NEW_int(k * k);
	minusId = NEW_int(k * k);
	Q = NEW_int(n * n + 1);
	Elt1 = NEW_int(A_PGL_n_q->elt_size_in_int);


	Mtx = A_PGL_k_q->G.matrix_grp;
	Fq = Mtx->GFq;
	minus_one = Fq->negate(1);


	int_vec_zero(Zero, k * k);
	int_vec_zero(Id, k * k);
	int_vec_zero(minusId, k * k);
	for (i = 0; i < k; i++) {
		Id[i * k + i] = 1;
		}
	for (i = 0; i < k; i++) {
		minusId[i * k + i] = minus_one;
		}

	gens_PGL_k = A_PGL_k_q->Strong_gens;
	//gens_PGL_k = A_PGL_k_q->strong_generators;
	
	len = gens_PGL_k->gens->len;
	//len = gens_PGL_k->len;

	int *Data;
	int new_len, sz, idx, h;

	new_len = len + 2;
	sz = n * n;
	if (Mtx->f_semilinear) {
		sz++;
		}
	

	Data = NEW_int(new_len * sz);
	idx = 0;
	for (h = 0; h < len; h++) {

		P = gens_PGL_k->gens->ith(h);
		//P = gens_PGL_k->ith(h);

		// Q := diag(P,P)
		int_matrix_make_block_matrix_2x2(Q, k, P, Zero, Zero, P);
		if (Mtx->f_semilinear) {
			Q[n * n] = P[k * k];
			}
		int_vec_copy(Q, Data + idx * sz, sz);
		idx++;
		}

	// Q := matrix(0,I,I,0):
	int_matrix_make_block_matrix_2x2(Q, k, Zero, Id, Id, Zero);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
		}
	int_vec_copy(Q, Data + idx * sz, sz);
	idx++;

	// Q := matrix(0,I,-I,-I):
	int_matrix_make_block_matrix_2x2(Q, k, Zero, Id, minusId, minusId);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
		}
	int_vec_copy(Q, Data + idx * sz, sz);
	idx++;


	if (idx != new_len) {
		cout << "make_generators_stabilizer_of_three_components "
				"idx != new_len" << endl;
		exit(1);
		}



	gens->init(A_PGL_n_q);
	gens->allocate(new_len);
	for (h = 0; h < new_len; h++) {
		A_PGL_n_q->make_element(Elt1, Data + h * sz, 0);
		if (f_vv) {
			cout << "make_generators_stabilizer_of_three_components "
					"after make_element generator " << h << " : " << endl;
			A_PGL_n_q->print_quick(cout, Elt1);
			}
		A_PGL_n_q->move(Elt1, gens->ith(h));
		}
	

	FREE_int(Data);

	FREE_int(Zero);
	FREE_int(Id);
	FREE_int(minusId);
	FREE_int(Q);
	FREE_int(Elt1);
	if (f_v) {
		cout << "make_generators_stabilizer_of_three_components done" << endl;
		}
}

void compute_generators_GL_n_q(int *&Gens,
		int &nb_gens, int &elt_size, int n, finite_field *F,
		vector_ge *&nice_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	action *A;
	vector_ge *gens;
	int *Elt;
	int h, i, l, alpha;

	if (f_v) {
		cout << "compute_generators_GL_n_q" << endl;
		}
	A = NEW_OBJECT(action);

	A->init_projective_group(n, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */,
			nice_gens,
			verbose_level - 2);

	gens = A->Strong_gens->gens;

	l = gens->len;
	nb_gens = l + 1;
	elt_size = n * n;
	Gens = NEW_int(nb_gens * elt_size);
	for (h = 0; h < nb_gens; h++) {
		if (h < l) {
			Elt = gens->ith(h);
			for (i = 0; i < n * n; i++) {
				Gens[h * elt_size + i] = Elt[i];
				}
			}
		else {
			for (i = 0; i < n * n; i++) {
				Gens[h * elt_size + i] = 0;
				}
			alpha = F->primitive_root();
			for (i = 0; i < n; i++) {
				Gens[h * elt_size + i * n + i] = alpha;
				}
			}
		}
	if (f_vv) {
		for (h = 0; h < nb_gens; h++) {
			cout << "Generator " << h << ":" << endl;
			int_matrix_print(Gens + h * elt_size, n, n);
			}
		
		}
	FREE_OBJECT(A);
	if (f_v) {
		cout << "compute_generators_GL_n_q done" << endl;
		}
}

void order_of_PGGL_n_q(longinteger_object &go,
		int n, int q, int f_semilinear)
{
	int verbose_level = 0;
	action *A;
	finite_field *F;
	vector_ge *nice_gens;

	F = NEW_OBJECT(finite_field);
	A = NEW_OBJECT(action);

	F->init(q, 0);
	A->init_projective_group(n, F, 
		f_semilinear, 
		TRUE /* f_basis */,
		nice_gens,
		verbose_level - 2);
	A->group_order(go);
	
	FREE_OBJECT(nice_gens);
	FREE_OBJECT(F);
	FREE_OBJECT(A);
}


// callbacks for Schreier Sims:


	int f_generator_orthogonal_siegel = TRUE;
	int f_generator_orthogonal_reflection = TRUE;
	int f_generator_orthogonal_similarity = TRUE;
	int f_generator_orthogonal_semisimilarity = TRUE;


void set_orthogonal_group_type(int f_siegel,
		int f_reflection,
		int f_similarity,
		int f_semisimilarity)
{
	f_generator_orthogonal_siegel = f_siegel;
	f_generator_orthogonal_reflection = f_reflection;
	f_generator_orthogonal_similarity = f_similarity;
	f_generator_orthogonal_semisimilarity = f_semisimilarity;
}

int get_orthogonal_group_type_f_reflection()
{
	return f_generator_orthogonal_reflection;
}

void callback_choose_random_generator_orthogonal(int iteration, 
	int *Elt, void *data, int verbose_level)
{
	//verbose_level += 5;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "callback_choose_random_generator_orthogonal "
				"iteration=" << iteration << endl;
		}

	schreier_sims *ss = (schreier_sims *) data;
	action *A = ss->GA;
	action *subaction = ss->KA;
	matrix_group *M;
#if 0
	int f_siegel = TRUE;
	int f_reflection = TRUE;
	int f_similarity = TRUE;
	int f_semisimilarity = TRUE;
#endif

	action_on_orthogonal *AO;
	orthogonal *O;
	
	AO = A->G.AO;
	O = AO->O;
	
	M = subaction->G.matrix_grp;
	if (f_v) {
		cout << "callback_choose_random_generator_orthogonal "
				"iteration=" << iteration
				<< " before M->orthogonal_group_random_generator"
				<< endl;
		}
	M->orthogonal_group_random_generator(ss->GA, O, 
		f_generator_orthogonal_siegel, 
		f_generator_orthogonal_reflection, 
		f_generator_orthogonal_similarity, 
		f_generator_orthogonal_semisimilarity, 
		Elt, verbose_level - 2);
	//M->GL_invert_internal(Elt, Elt + M->elt_size_int_half, 0);
	if (f_v) {
		cout << "callback_choose_random_generator_orthogonal "
				"iteration=" << iteration
				<< " after M->orthogonal_group_random_generator"
				<< endl;
		}

	if (f_v) {
		cout << "callback_choose_random_generator_orthogonal "
				"iteration=" << iteration << " done" << endl;
		}
}



void test_matrix_group(int k, int q, int f_semilinear, int verbose_level)
{
	action A;
	finite_field *F;
	int f_basis = TRUE;
	vector_ge *nice_gens;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	A.init_projective_group(k, F, f_semilinear, f_basis,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	FREE_OBJECT(F);
}

void lift_generators(vector_ge *gens_in, vector_ge *&gens_out, 
	action *Aq, subfield_structure *S, int n, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *EltQ;
	int *Eltq;
	int *Mtx;
	int nb_gens, m, t;


	if (f_v) {
		cout << "lift_generators" << endl;
		}

	nb_gens = gens_in->len;

	m = n / S->s;

	gens_out = NEW_OBJECT(vector_ge);

	Eltq = NEW_int(Aq->elt_size_in_int);
	Mtx = NEW_int(n * n);

	if (f_v) {
		cout << "lift_generators lifting generators" << endl;
		}
	gens_out->init(Aq);
	gens_out->allocate(nb_gens);
	for (t = 0; t < nb_gens; t++) {
		if (f_vv) {
			cout << "lift_generators " << t << " / " << nb_gens << endl;
			}
		EltQ = gens_in->ith(t);
		S->lift_matrix(EltQ, m, Mtx, 0 /* verbose_level */);
		if (f_vv) {
			cout << "lift_generators lifted matrix:" << endl;
			int_matrix_print(Mtx, n, n);
			}
		Aq->make_element(Eltq, Mtx, 0 /*verbose_level - 4 */);
		if (f_vv) {
			cout << "lift_generators after make_element:" << endl;
			Aq->element_print_quick(Eltq, cout);
			}
		Aq->element_move(Eltq, gens_out->ith(t), 0);
		if (f_vv) {
			cout << "lift_generators " << t << " / "
					<< nb_gens << " done" << endl;
			}
		}
	FREE_int(Eltq);
	FREE_int(Mtx);
	if (f_v) {
		cout << "lift_generators done" << endl;
		}

}

void retract_generators(vector_ge *gens_in,
	vector_ge *&gens_out,
	action *AQ, subfield_structure *S, int n, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *EltQ;
	int *Eltq;
	int *Mtx;
	int nb_gens, m, t;


	if (f_v) {
		cout << "retract_generators" << endl;
		}

	nb_gens = gens_in->len;

	m = n / S->s;

	gens_out = NEW_OBJECT(vector_ge);

	EltQ = NEW_int(AQ->elt_size_in_int);
	Mtx = NEW_int(m * m);

	if (f_v) {
		cout << "retract_generators retracting generators" << endl;
		}
	gens_out->init(AQ);
	gens_out->allocate(nb_gens);
	for (t = 0; t < nb_gens; t++) {
		if (f_vv) {
			cout << "retract_generators " << t
					<< " / " << nb_gens << endl;
			}
		Eltq = gens_in->ith(t);
		S->retract_matrix(Eltq, n, Mtx, m, 0 /* verbose_level */);
		if (f_vv) {
			cout << "retract_generators retracted matrix:" << endl;
			int_matrix_print(Mtx, m, m);
			}
		AQ->make_element(EltQ, Mtx, 0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "retract_generators after make_element:" << endl;
			AQ->element_print_quick(EltQ, cout);
			}
		AQ->element_move(EltQ, gens_out->ith(t), 0);
		if (f_vv) {
			cout << "retract_generators " << t
					<< " / " << nb_gens << " done" << endl;
			}
		}
	FREE_int(EltQ);
	FREE_int(Mtx);
	if (f_v) {
		cout << "retract_generators done" << endl;
		}

}

void lift_generators_to_subfield_structure(
	int n, int s, 
	subfield_structure *S, 
	action *Aq, action *AQ, 
	strong_generators *&Strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int q, Q, m;
	finite_field *Fq;
	//finite_field *FQ;
	sims *Sims;
	number_theory_domain NT;

	if (f_v) {
		cout << "lift_generators_to_subfield_structure" << endl;
		}
	Fq = S->Fq;
	//FQ = S->FQ;
	q = Fq->q;
	Q = NT.i_power_j(q, s);
	m = n / s;
	if (m * s != n) {
		cout << "lift_generators_to_subfield_structure "
				"s must divide n" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"creating subfield structure" << endl;
		}
	if (f_v) {
		cout << "n=" << n << endl;
		cout << "s=" << s << endl;
		cout << "m=" << m << endl;
		cout << "q=" << q << endl;
		cout << "Q=" << Q << endl;
		}

	longinteger_object order_GLmQ;
	longinteger_object target_go;
	longinteger_domain D;
	int r;

	AQ->group_order(order_GLmQ);
	

	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"order of GL(m,Q) = " << order_GLmQ << endl;
		}
	D.integral_division_by_int(order_GLmQ, 
		q - 1, target_go, r);
	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"target_go = " << target_go << endl;
		}



	vector_ge *gens;
	vector_ge *gens1;


	gens = AQ->Strong_gens->gens;


	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"before lift_generators" << endl;
		}
	lift_generators(gens, gens1, Aq, S, n, verbose_level);
		// ACTION/action_global.C
	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"after lift_generators" << endl;
		}


	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"creating lifted group:" << endl;
		}
	//Aq->group_order(target_go);
	Sims = Aq->create_sims_from_generators_with_target_group_order(
		gens1, 
		target_go, 
		0 /* verbose_level */);

#if 0
	Sims = A1->create_sims_from_generators_without_target_group_order(
		gens1, MINIMUM(2, verbose_level - 3));
#endif

	if (f_v) {
		cout << "lift_generators_to_subfield_structure "
				"creating lifted group done" << endl;
		}

	longinteger_object go;

	Sims->group_order(go);

	if (f_v) {
		cout << "go=" << go << endl;
		}


	Strong_gens = NEW_OBJECT(strong_generators);

	Strong_gens->init_from_sims(Sims, 0 /* verbose_level */);
	if (f_vv) {
		cout << "lift_generators_to_subfield_structure "
				"strong generators are:" << endl;
		Strong_gens->print_generators();
		}


	FREE_OBJECT(gens1);
	FREE_OBJECT(Sims);
	if (f_v) {
		cout << "lift_generators_to_subfield_structure done" << endl;
		}

}


int group_ring_element_size(action *A, sims *S)
{
	int goi;

	goi = S->group_order_int();
	return goi;
}

void group_ring_element_create(action *A, sims *S, int *&elt)
{
	int goi;

	goi = S->group_order_int();
	elt = NEW_int(goi);
	group_ring_element_zero(A, S, elt);
}

void group_ring_element_free(action *A, sims *S, int *elt)
{
	FREE_int(elt);
}

void group_ring_element_print(action *A, sims *S, int *elt)
{
	int goi;

	goi = S->group_order_int();
	int_vec_print(cout, elt, goi);
}

void group_ring_element_copy(action *A, sims *S,
		int *elt_from, int *elt_to)
{
	int goi;

	goi = S->group_order_int();
	int_vec_copy(elt_from, elt_to, goi);
}

void group_ring_element_zero(action *A, sims *S, int *elt)
{
	int goi;

	goi = S->group_order_int();
	int_vec_zero(elt, goi);
}

void group_ring_element_mult(action *A,
		sims *S, int *elt1, int *elt2, int *elt3)
{
	int goi;
	int i, j, k;
	int a, b, c;

	goi = S->group_order_int();
	int_vec_zero(elt3, goi);
	for (i = 0; i < goi; i++) {
		a = elt1[i];
		for (j = 0; j < goi; j++) {
			b = elt2[j];
			c = a * b;
			k = S->mult_by_rank(i, j, 0 /* verbose_level */);
			elt3[k] += c;
			}
		}
}


void perm_print_cycles_sorted_by_length(ostream &ost,
		int degree, int *perm, int verbose_level)
{
	perm_print_cycles_sorted_by_length_offset(ost,
			degree, perm, 0, FALSE, TRUE, verbose_level);
}

void perm_print_cycles_sorted_by_length_offset(ostream &ost, 
	int degree, int *perm, int offset,
	int f_do_it_anyway_even_for_big_degree,
	int f_print_cycles_of_length_one, int verbose_level)
{
	int nb_gens = 1;
	int i;
	vector_ge Gens;
	action *A;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_big = FALSE;
	int f_doit = TRUE;
	
	if (f_v) {
		cout << "perm_print_cycles_sorted_by_length, "
				"degree=" << degree << endl;
		}
	
	if (degree > 500) {
		f_big = TRUE;
		}
	A = NEW_OBJECT(action);
	
	A->init_permutation_group(degree, 0/*verbose_level*/);
	Gens.init(A);
	Gens.allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		Gens.copy_in(i, perm + i * degree);
		}
	if (f_vv) {
		Gens.print(cout);
		}
	
	schreier S;
	
	S.init(A);
	S.init_generators(Gens);
	S.compute_all_point_orbits(verbose_level);
	if (f_v) {
		cout << "after S.compute_all_point_orbits, "
				"nb_orbits=" << S.nb_orbits << endl;
		}
	//S.print_orbit_lengths(cout);
	//S.print_orbit_length_distribution(ost);

	int j, f, l, length, F, L, h, a, b, m, orbit_idx;
	int *orbit_len_sorted;
	int *sorting_perm;
	int *sorting_perm_inv;
	int nb_types;
	int *type_first;
	int *type_len;
	
	int_vec_classify(S.nb_orbits, S.orbit_len, orbit_len_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);

#if 0
	ost << "permutation of degree " << degree << " with "
			<< S.nb_orbits << " orbits: " << endl;
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		length = orbit_len_sorted[f];
		if (l > 1) {
			ost << l << " \\times ";
			}
		ost << length;
		if (i < nb_types - 1)
			ost << ", ";
		}
	ost << endl;
	ost << "cycles in increasing length:" << endl;
#endif
	if (f_big) {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			length = orbit_len_sorted[f];
			ost << l << " cycles of length " << length << endl;
			}
		}
	if (f_big && !f_do_it_anyway_even_for_big_degree) {
		f_doit = FALSE;
		}
	if (f_doit) {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			length = orbit_len_sorted[f];
			if (length == 1 && !f_print_cycles_of_length_one) {
				continue;
				}
			for (j = 0; j < l; j++) {
				orbit_idx = sorting_perm_inv[f + j];
				//ost << "orbit " << orbit_idx << ": ";
				F = S.orbit_first[orbit_idx];
				L = S.orbit_len[orbit_idx];
				m = S.orbit[F];
				for (h = 1; h < L; h++) {
					if (S.orbit[F + h] < m)
						m = S.orbit[F + h];
					}
				// now m is the least lement in the orbit
				ost << "(";
				a = m;
				ost << (a + offset);
				while (TRUE) {
					b = perm[a];
					if (b == m)
						break;
					ost << ", " << (b + offset);
					a = b;
					}
				ost << ")";
				if (length > 20) {
					//ost << endl;
					}
				} // next j
			//ost << endl;
			} // next i
		} // if
	//ost << "done" << endl;

#if 0
	classify C;

	C.init(S.orbit_len, S.nb_orbits, FALSE, 0);
	ost << " cycle type: ";
	C.print_file(ost, TRUE /* f_backwards */);
#endif

	FREE_int(orbit_len_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);
	
	FREE_OBJECT(A);
}





}}
