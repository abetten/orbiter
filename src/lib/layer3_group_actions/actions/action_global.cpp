// action_global.cpp
//
// Anton Betten
// October 10, 2013

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


void action_global::action_print_symmetry_group_type(std::ostream &ost,
		symmetry_group_type a)
{
	std::string txt;
	std::string tex;

	get_symmetry_group_type_text(txt, tex, a);
	ost << txt;
}

void action_global::get_symmetry_group_type_text(
		std::string &txt, std::string &tex,
		symmetry_group_type a)
{
	if (a == unknown_symmetry_group_t) {
		txt.assign("unknown_symmetry_group_t");
		tex.assign("unknown");
	}
	else if (a == matrix_group_t) {
		txt.assign("matrix_group_t");
		tex.assign("matrix type");
	}
	else if (a == perm_group_t) {
		txt.assign("perm_group_t");
		tex.assign("permutation group");
	}
	else if (a == wreath_product_t) {
		txt.assign("wreath_product_t");
		tex.assign("wreath product");
	}
	else if (a == direct_product_t) {
		txt.assign("direct_product_t");
		tex.assign("direct product");
	}
	else if (a == permutation_representation_t) {
		txt.assign("permutation_representation_t");
		tex.assign("permutation representation");
	}
	else if (a == action_on_sets_t) {
		txt.assign("action_on_sets_t");
		tex.assign("action on subsets");
	}
	else if (a == action_on_set_partitions_t) {
		txt.assign("action_on_set_partitions_t");
		tex.assign("action on set partitions");
	}
	else if (a == action_on_subgroups_t) {
		txt.assign("action_on_subgroups_t");
		tex.assign("action on subgroups");
	}
	else if (a == action_on_pairs_t) {
		txt.assign("action_on_pairs_t");
		tex.assign("action on pairs");
	}
	else if (a == action_on_ordered_pairs_t) {
		txt.assign("action_on_ordered_pairs_t");
		tex.assign("action on ordered pairs");
	}
	else if (a == base_change_t) {
		txt.assign("base_change_t");
		tex.assign("base change");
	}
	else if (a == product_action_t) {
		txt.assign("product_action_t");
		tex.assign("product action");
	}
	else if (a == action_by_right_multiplication_t) {
		txt.assign("action_by_right_multiplication_t");
		tex.assign("action by right multiplication");
	}
	else if (a == action_by_restriction_t) {
		txt.assign("action_by_restriction_t");
		tex.assign("action by restriction");
	}
	else if (a == action_by_conjugation_t) {
		txt.assign("action_by_conjugation_t");
		tex.assign("action by conjugation");
	}
	else if (a == action_by_representation_t) {
		txt.assign("action_by_representation_t");
		tex.assign("action by representation");
	}
	else if (a == action_by_subfield_structure_t) {
		txt.assign("action_by_subfield_structure_t");
		tex.assign("action by subfield structure");
	}
	else if (a == action_on_determinant_t) {
		txt.assign("action_on_determinant_t");
		tex.assign("action on determinant");
	}
	else if (a == action_on_galois_group_t) {
		txt.assign("action_on_galois_group_t");
		tex.assign("action on galois group");
	}
	else if (a == action_on_sign_t) {
		txt.assign("action_on_sign_t");
		tex.assign("action on sign");
	}
	else if (a == action_on_grassmannian_t) {
		txt.assign("action_on_grassmannian_t");
		tex.assign("action on grassmannian");
	}
	else if (a == action_on_spread_set_t) {
		txt.assign("action_on_spread_set_t");
		tex.assign("action on spread set");
	}
	else if (a == action_on_cosets_t) {
		txt.assign("action_on_cosets_t");
		tex.assign("action on cosets");
	}
	else if (a == action_on_factor_space_t) {
		txt.assign("action_on_factor_space_t");
		tex.assign("action on factor space");
	}
	else if (a == action_on_wedge_product_t) {
		txt.assign("action_on_wedge_product_t");
		tex.assign("action on wedge product");
	}
	else if (a == action_on_bricks_t) {
		txt.assign("action_on_bricks_t");
		tex.assign("action on bricks");
	}
	else if (a == action_on_andre_t) {
		txt.assign("action_on_andre_t");
		tex.assign("action on andre");
	}
	else if (a == action_on_orthogonal_t) {
		txt.assign("action_on_orthogonal_t");
		tex.assign("action on orthogonal");
	}
	else if (a == action_on_orbits_t) {
		txt.assign("action_on_orbits_t");
		tex.assign("action on orbits");
	}
	else if (a == action_on_flags_t) {
		txt.assign("action_on_flags_t");
		tex.assign("action on flags");
	}
	else if (a == action_on_homogeneous_polynomials_t) {
		txt.assign("action_on_homogeneous_polynomials_t");
		tex.assign("action on homogeneous polynomials");
	}
	else if (a == action_on_k_subsets_t) {
		txt.assign("action_on_k_subsets_t");
		tex.assign("action on k-subsets");
	}
	else if (a == action_on_interior_direct_product_t) {
		txt.assign("action_on_interior_direct_product_t");
		tex.assign("action on interior direct product");
	}
	else {
		txt.assign("unknown symmetry_group_type");
		tex.assign("unknown");
		exit(1);
	}
}


void action_global::make_generators_stabilizer_of_two_components(
	action *A_PGL_n_q, action *A_PGL_k_q,
	int k,
	data_structures_groups::vector_ge *gens,
	int verbose_level)
// used in semifield.cpp
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
	algebra::matrix_group *Mtx;
	field_theory::finite_field *Fq;
	int minus_one, alpha;
	groups::strong_generators *gens_PGL_k;
	//vector_ge *gens_PGL_k;


	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_two_components" << endl;
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

	Int_vec_zero(Zero, k * k);
	Int_vec_zero(Id, k * k);
	Int_vec_zero(Center, k * k);
	Int_vec_zero(minusId, k * k);
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
			orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
					Q, k, P, Zero, Zero, Id);
		}
		else {
			// Q := diag(Id,P)
			orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
					Q, k, Id, Zero, Zero, P);
		}
		if (Mtx->f_semilinear) {
			Q[n * n] = P[k * k];
		}
		Int_vec_copy(Q, Data + idx * sz, sz);
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
	orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
			Q, k, Center, Zero, Zero, Id);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
	}
	Int_vec_copy(Q, Data + idx * sz, sz);
	idx++;

	// Q := matrix(I,0,0,Center):
	orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
			Q, k, Id, Zero, Zero, Center);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
	}
	Int_vec_copy(Q, Data + idx * sz, sz);
	idx++;


	if (idx != new_len) {
		cout << "action_global::make_generators_stabilizer_of_two_components "
				"idx != new_len" << endl;
		exit(1);
	}



	gens->init(A_PGL_n_q, verbose_level - 2);
	gens->allocate(new_len, verbose_level - 2);
	for (h = 0; h < new_len; h++) {
		A_PGL_n_q->Group_element->make_element(Elt1, Data + h * sz, 0);
		if (f_vv) {
			cout << "action_global::make_generators_stabilizer_of_two_components "
					"after make_element generator " << h << " : " << endl;
			A_PGL_n_q->Group_element->print_quick(cout, Elt1);
		}
		A_PGL_n_q->Group_element->move(Elt1, gens->ith(h));
	}
	

	FREE_int(Data);

	FREE_int(Zero);
	FREE_int(Id);
	FREE_int(Center);
	FREE_int(minusId);
	FREE_int(Q);
	FREE_int(Elt1);
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_two_components done" << endl;
	}
}


void action_global::make_generators_stabilizer_of_three_components(
	action *A_PGL_n_q, action *A_PGL_k_q,
	int k,
	data_structures_groups::vector_ge *gens,
	int verbose_level)
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
	algebra::matrix_group *Mtx;
	field_theory::finite_field *Fq;
	int minus_one;
	groups::strong_generators *gens_PGL_k;

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components" << endl;
		cout << "A_PGL_n_q:" << endl;
		A_PGL_n_q->print_info();
		cout << "A_PGL_k_q:" << endl;
		A_PGL_k_q->print_info();
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


	Int_vec_zero(Zero, k * k);
	Int_vec_zero(Id, k * k);
	Int_vec_zero(minusId, k * k);
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
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components sz = " << sz << endl;
	}
	Data = NEW_int(new_len * sz);
	

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components step 1" << endl;
	}
	idx = 0;
	for (h = 0; h < len; h++) {

		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components step 1: " << h << " / " << len << endl;
		}
		P = gens_PGL_k->gens->ith(h);
		//P = gens_PGL_k->ith(h);

		// Q := diag(P,P)
		orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(Q, k, P, Zero, Zero, P);
		if (Mtx->f_semilinear) {
			Q[n * n] = P[k * k];
		}
		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components Q=" << endl;
			Int_matrix_print(Q, n, n);
		}
		Int_vec_copy(Q, Data + idx * sz, sz);
		idx++;
	}

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components step 2" << endl;
	}
	// Q := matrix(0,I,I,0):
	orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(Q, k, Zero, Id, Id, Zero);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
	}
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components Q=" << endl;
		Int_matrix_print(Q, n, n);
	}
	Int_vec_copy(Q, Data + idx * sz, sz);
	idx++;

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components step 3" << endl;
	}
	// Q := matrix(0,I,-I,-I):
	orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(Q, k, Zero, Id, minusId, minusId);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
	}
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components Q=" << endl;
		Int_matrix_print(Q, n, n);
	}
	Int_vec_copy(Q, Data + idx * sz, sz);
	idx++;


	if (idx != new_len) {
		cout << "action_global::make_generators_stabilizer_of_three_components "
				"idx != new_len" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components step 4" << endl;
	}


	gens->init(A_PGL_n_q, verbose_level - 2);
	gens->allocate(new_len, verbose_level - 2);
	for (h = 0; h < new_len; h++) {

		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components step 4: " << h << " / " << new_len << endl;
		}

		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components generator=" << endl;
			Int_matrix_print(Data + h * sz, n, n);
		}
		A_PGL_n_q->Group_element->make_element(Elt1, Data + h * sz, 0);
		if (f_vv) {
			cout << "action_global::make_generators_stabilizer_of_three_components "
					"after make_element generator " << h << " : " << endl;
			A_PGL_n_q->Group_element->print_quick(cout, Elt1);
		}
		A_PGL_n_q->Group_element->move(Elt1, gens->ith(h));
	}
	

	FREE_int(Data);

	FREE_int(Zero);
	FREE_int(Id);
	FREE_int(minusId);
	FREE_int(Q);
	FREE_int(Elt1);
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components done" << endl;
	}
}

void action_global::compute_generators_GL_n_q(
		int *&Gens,
		int &nb_gens, int &elt_size, int n,
		field_theory::finite_field *F,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level)
// puts generators for the kernel back in to get from PGL to GL
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	action *A;
	data_structures_groups::vector_ge *gens;
	int *Elt;
	int h, i, l, alpha;

	if (f_v) {
		cout << "action_global::compute_generators_GL_n_q" << endl;
	}
	A = NEW_OBJECT(action);

	if (f_v) {
		cout << "action_global::compute_generators_GL_n_q "
				"before A->Known_groups->init_projective_group" << endl;
	}
	A->Known_groups->init_projective_group(n, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */, TRUE /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_generators_GL_n_q "
				"after A->Known_groups->init_projective_group" << endl;
	}

	gens = A->Strong_gens->gens;

	l = gens->len;
	nb_gens = l + 1;
	elt_size = n * n;
	Gens = NEW_int(nb_gens * elt_size);
	for (h = 0; h < nb_gens; h++) {
		if (h < l) {
			Elt = gens->ith(h);
			Int_vec_copy(Elt, Gens + h * elt_size, elt_size);
#if 0
			for (i = 0; i < n * n; i++) {
				Gens[h * elt_size + i] = Elt[i];
			}
#endif
		}
		else {
			Int_vec_zero(Gens + h * elt_size, elt_size);
#if 0
			for (i = 0; i < n * n; i++) {
				Gens[h * elt_size + i] = 0;
			}
#endif
			alpha = F->primitive_root();
			for (i = 0; i < n; i++) {
				Gens[h * elt_size + i * n + i] = alpha;
			}
		}
	}
	if (f_vv) {
		for (h = 0; h < nb_gens; h++) {
			cout << "Generator " << h << ":" << endl;
			Int_matrix_print(Gens + h * elt_size, n, n);
		}
		
	}
	FREE_OBJECT(A);
	if (f_v) {
		cout << "action_global::compute_generators_GL_n_q done" << endl;
	}
}



// callbacks for Schreier Sims:


	int f_generator_orthogonal_siegel = TRUE;
	int f_generator_orthogonal_reflection = TRUE;
	int f_generator_orthogonal_similarity = TRUE;
	int f_generator_orthogonal_semisimilarity = TRUE;


void action_global::set_orthogonal_group_type(
		int f_siegel,
		int f_reflection,
		int f_similarity,
		int f_semisimilarity)
{
	f_generator_orthogonal_siegel = f_siegel;
	f_generator_orthogonal_reflection = f_reflection;
	f_generator_orthogonal_similarity = f_similarity;
	f_generator_orthogonal_semisimilarity = f_semisimilarity;
}

int action_global::get_orthogonal_group_type_f_reflection()
{
	return f_generator_orthogonal_reflection;
}



void action_global::lift_generators(
		data_structures_groups::vector_ge *gens_in,
		data_structures_groups::vector_ge *&gens_out,
	action *Aq,
	field_theory::subfield_structure *S, int n,
	int verbose_level)
// gens_in are m x m (i.e., small matrices over the large field),
// gens_out are n x n (i.e., large matrices over the small field).
// Here, m * s = n.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *EltQ;
	int *Eltq;
	int *Mtx;
	int nb_gens, m, t;


	if (f_v) {
		cout << "action_global::lift_generators" << endl;
	}

	nb_gens = gens_in->len;

	m = n / S->s;

	gens_out = NEW_OBJECT(data_structures_groups::vector_ge);

	Eltq = NEW_int(Aq->elt_size_in_int);
	Mtx = NEW_int(n * n);

	if (f_v) {
		cout << "action_global::lift_generators lifting generators" << endl;
	}
	gens_out->init(Aq, verbose_level - 2);
	gens_out->allocate(nb_gens, verbose_level - 2);
	for (t = 0; t < nb_gens; t++) {
		if (f_vv) {
			cout << "lift_generators " << t << " / " << nb_gens << endl;
			}
		EltQ = gens_in->ith(t);
		S->lift_matrix(EltQ, m, Mtx, 0 /* verbose_level */);
		if (f_vv) {
			cout << "action_global::lift_generators lifted matrix:" << endl;
			Int_matrix_print(Mtx, n, n);
			}
		Aq->Group_element->make_element(Eltq, Mtx, 0 /*verbose_level - 4 */);
		if (f_vv) {
			cout << "action_global::lift_generators after make_element:" << endl;
			Aq->Group_element->element_print_quick(Eltq, cout);
			}
		Aq->Group_element->element_move(Eltq, gens_out->ith(t), 0);
		if (f_vv) {
			cout << "action_global::lift_generators " << t << " / "
					<< nb_gens << " done" << endl;
			}
		}
	FREE_int(Eltq);
	FREE_int(Mtx);
	if (f_v) {
		cout << "action_global::lift_generators done" << endl;
	}
}

void action_global::retract_generators(
		data_structures_groups::vector_ge *gens_in,
		data_structures_groups::vector_ge *&gens_out,
	action *AQ,
	field_theory::subfield_structure *S, int n,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *EltQ;
	int *Eltq;
	int *Mtx;
	int nb_gens, m, t;


	if (f_v) {
		cout << "action_global::retract_generators" << endl;
	}

	nb_gens = gens_in->len;

	m = n / S->s;

	gens_out = NEW_OBJECT(data_structures_groups::vector_ge);

	EltQ = NEW_int(AQ->elt_size_in_int);
	Mtx = NEW_int(m * m);

	if (f_v) {
		cout << "action_global::retract_generators retracting generators" << endl;
	}
	gens_out->init(AQ, verbose_level - 2);
	gens_out->allocate(nb_gens, verbose_level - 2);
	for (t = 0; t < nb_gens; t++) {
		if (f_vv) {
			cout << "action_global::retract_generators " << t
					<< " / " << nb_gens << endl;
		}
		Eltq = gens_in->ith(t);
		S->retract_matrix(Eltq, n, Mtx, m, 0 /* verbose_level */);
		if (f_vv) {
			cout << "action_global::retract_generators retracted matrix:" << endl;
			Int_matrix_print(Mtx, m, m);
		}
		AQ->Group_element->make_element(EltQ, Mtx, 0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "action_global::retract_generators after make_element:" << endl;
			AQ->Group_element->element_print_quick(EltQ, cout);
		}
		AQ->Group_element->element_move(EltQ, gens_out->ith(t), 0);
		if (f_vv) {
			cout << "action_global::retract_generators " << t
					<< " / " << nb_gens << " done" << endl;
		}
	}
	FREE_int(EltQ);
	FREE_int(Mtx);
	if (f_v) {
		cout << "action_global::retract_generators done" << endl;
	}
}

void action_global::lift_generators_to_subfield_structure(
	int n, int s, 
	field_theory::subfield_structure *S,
	action *Aq, action *AQ, 
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int q, Q, m;
	field_theory::finite_field *Fq;
	//finite_field *FQ;
	groups::sims *Sims;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure" << endl;
	}
	Fq = S->Fq;
	//FQ = S->FQ;
	q = Fq->q;
	Q = NT.i_power_j(q, s);
	m = n / s;
	if (m * s != n) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"s must divide n" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"creating subfield structure" << endl;
	}
	if (f_v) {
		cout << "n=" << n << endl;
		cout << "s=" << s << endl;
		cout << "m=" << m << endl;
		cout << "q=" << q << endl;
		cout << "Q=" << Q << endl;
	}

	ring_theory::longinteger_object order_GLmQ;
	ring_theory::longinteger_object target_go;
	ring_theory::longinteger_domain D;
	int r;

	AQ->group_order(order_GLmQ);
	

	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"order of GL(m,Q) = " << order_GLmQ << endl;
	}
	D.integral_division_by_int(order_GLmQ, 
		q - 1, target_go, r);
	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"target_go = " << target_go << endl;
	}



	data_structures_groups::vector_ge *gens;
	data_structures_groups::vector_ge *gens1;


	gens = AQ->Strong_gens->gens;


	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"before lift_generators" << endl;
	}
	lift_generators(gens, gens1, Aq, S, n, verbose_level);

	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"after lift_generators" << endl;
	}


	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure "
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
		cout << "action_global::lift_generators_to_subfield_structure "
				"creating lifted group done" << endl;
	}

	ring_theory::longinteger_object go;

	Sims->group_order(go);

	if (f_v) {
		cout << "go=" << go << endl;
	}


	Strong_gens = NEW_OBJECT(groups::strong_generators);

	Strong_gens->init_from_sims(Sims, 0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"strong generators are:" << endl;
		Strong_gens->print_generators(cout);
	}


	FREE_OBJECT(gens1);
	FREE_OBJECT(Sims);
	if (f_v) {
		cout << "action_global::lift_generators_to_subfield_structure done" << endl;
	}
}


void action_global::perm_print_cycles_sorted_by_length(
		std::ostream &ost,
		int degree, int *perm, int verbose_level)
{
	perm_print_cycles_sorted_by_length_offset(ost,
			degree, perm, 0, FALSE, TRUE, verbose_level);
}

void action_global::perm_print_cycles_sorted_by_length_offset(
		std::ostream &ost,
	int degree, int *perm, int offset,
	int f_do_it_anyway_even_for_big_degree,
	int f_print_cycles_of_length_one, int verbose_level)
{
	int nb_gens = 1;
	int i;
	data_structures_groups::vector_ge Gens;
	action *A;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_big = FALSE;
	int f_doit = TRUE;
	
	if (f_v) {
		cout << "action_global::perm_print_cycles_sorted_by_length, "
				"degree=" << degree << endl;
	}
	
	if (degree > 500) {
		f_big = TRUE;
	}
	A = NEW_OBJECT(action);
	int f_no_base = FALSE;
	
	if (f_v) {
		cout << "action_global::perm_print_cycles_sorted_by_length "
				"before A->Known_groups->init_permutation_group" << endl;
	}
	A->Known_groups->init_permutation_group(
			degree, f_no_base, 0/*verbose_level*/);
	if (f_v) {
		cout << "action_global::perm_print_cycles_sorted_by_length "
				"after A->Known_groups->init_permutation_group" << endl;
	}

	Gens.init(A, verbose_level - 2);
	Gens.allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		Gens.copy_in(i, perm + i * degree);
	}
	if (f_vv) {
		Gens.print(cout);
	}
	
	groups::schreier S;
	
	S.init(A, verbose_level - 2);
	S.init_generators(Gens, verbose_level - 2);
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
	data_structures::sorting Sorting;
	
	Sorting.int_vec_classify(
			S.nb_orbits, S.orbit_len, orbit_len_sorted,
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
				// now m is the least element in the orbit
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



action *action_global::init_direct_product_group_and_restrict(
		algebra::matrix_group *M1,
		algebra::matrix_group *M2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_direct_product;
	action *Adp;
	groups::direct_product *P;
	long int *points;
	int nb_points;
	int i;

	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict" << endl;
		cout << "M1=" << M1->label << endl;
		cout << "M2=" << M2->label << endl;
	}
	A_direct_product = NEW_OBJECT(action);
	A_direct_product = init_direct_product_group(M1, M2, verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict "
				"after A_direct_product->init_direct_product_group" << endl;
	}

	P = A_direct_product->G.direct_product_group;
	nb_points = P->degree_of_product_action;
	points = NEW_lint(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = P->perm_offset_i[2] + i;
	}


	std::string label_of_set;

	label_of_set.assign("direct_product");

	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict "
				"before A_direct_product->Induced_action->restricted_action" << endl;
	}
	Adp = A_direct_product->Induced_action->restricted_action(
			points, nb_points, label_of_set,
			verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict "
				"after A_direct_product->Induced_action->restricted_action" << endl;
	}
	Adp->f_is_linear = FALSE;


	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict "
				"after A_direct_product->restricted_action" << endl;
	}
	return Adp;
}

action *action_global::init_direct_product_group(
		algebra::matrix_group *M1,
		algebra::matrix_group *M2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::direct_product *P;
	action *A;

	if (f_v) {
		cout << "action_global::init_direct_product_group" << endl;
		cout << "M1=" << M1->label << endl;
		cout << "M2=" << M2->label << endl;
	}

	A = NEW_OBJECT(action);
	P = NEW_OBJECT(groups::direct_product);



	A->type_G = direct_product_t;
	A->G.direct_product_group = P;
	A->f_allocated = TRUE;

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before P->init" << endl;
	}
	P->init(M1, M2, verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after P->init" << endl;
	}

	A->f_is_linear = FALSE;
	A->dimension = 0;


	A->low_level_point_size = 0;
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"low_level_point_size="
			<< A->low_level_point_size<< endl;
	}

	A->label.assign(P->label);
	A->label_tex.assign(P->label_tex);


	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"label=" << A->label << endl;
	}

	A->degree = P->degree_overall;
	A->make_element_size = P->make_element_size;

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_direct_product_group();

	A->elt_size_in_int = P->elt_size_int;
	A->coded_elt_size_in_char = P->char_per_elt;
	A->allocate_element_data();




	A->degree = P->degree_overall;
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"degree=" << A->degree << endl;
	}

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(
			A, P->base_length, verbose_level);

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"base_len=" << A->base_len() << endl;
	}


	Lint_vec_copy(P->the_base, A->get_base(), A->base_len());
	Int_vec_copy(P->the_transversal_length,
			A->get_transversal_length(), A->base_len());

	int *gens_data;
	int gens_size;
	int gens_nb;

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before P->make_strong_generators_data" << endl;
	}
	P->make_strong_generators_data(gens_data,
			gens_size, gens_nb, verbose_level - 1);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after P->make_strong_generators_data" << endl;
	}
	A->Strong_gens = NEW_OBJECT(groups::strong_generators);

	data_structures_groups::vector_ge *nice_gens;

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before A->Strong_gens->init_from_data" << endl;
	}
	A->Strong_gens->init_from_data(A,
			gens_data, gens_nb, gens_size,
			A->get_transversal_length(),
			nice_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after A->Strong_gens->init_from_data" << endl;
	}
	FREE_OBJECT(nice_gens);
	A->f_has_strong_generators = TRUE;
	FREE_int(gens_data);

	groups::sims *S;

	S = NEW_OBJECT(groups::sims);

	S->init(A, verbose_level - 2);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before S->init_generators" << endl;
	}
	S->init_generators(*A->Strong_gens->gens, verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after S->init_generators" << endl;
	}
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before S->compute_base_orbits_known_length" << endl;
	}
	S->compute_base_orbits_known_length(
			A->get_transversal_length(), verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after S->compute_base_orbits_known_length" << endl;
	}


	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before init_sims_only" << endl;
	}

	A->init_sims_only(S, verbose_level);

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after init_sims_only" << endl;
	}

	A->compute_strong_generators_from_sims(
			0/*verbose_level - 2*/);

	if (f_v) {
		cout << "action_global::init_direct_product_group, "
				"finished setting up " << A->label;
		cout << ", a permutation group of degree " << A->degree << " ";
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
	return A;
}



void action_global::compute_decomposition_based_on_orbits(
		geometry::projective_space *P,
		groups::schreier *Sch1, groups::schreier *Sch2,
		geometry::incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits" << endl;
	}

	data_structures::partitionstack *S1;
	data_structures::partitionstack *S2;


	S1 = NEW_OBJECT(data_structures::partitionstack);
	S2 = NEW_OBJECT(data_structures::partitionstack);

	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits "
				"before S1->allocate" << endl;
	}
	S1->allocate(P->Subspaces->N_points, 0 /* verbose_level */);
	S2->allocate(P->Subspaces->N_lines, 0 /* verbose_level */);

	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits "
				"before Sch1->get_orbit_partition" << endl;
	}
	Sch1->get_orbit_partition(*S1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits "
				"before Sch2->get_orbit_partition" << endl;
	}
	Sch2->get_orbit_partition(*S2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits "
				"after Sch2->get_orbit_partition" << endl;
	}




	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits "
				"before P->compute_decomposition" << endl;
	}
	P->Subspaces->compute_decomposition(S1, S2, Inc, Stack, verbose_level);

	FREE_OBJECT(S1);
	FREE_OBJECT(S2);

	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbits done" << endl;
	}
}


void action_global::compute_decomposition_based_on_orbit_length(
		geometry::projective_space *P,
		groups::schreier *Sch1, groups::schreier *Sch2,
		geometry::incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbit_length" << endl;
	}

	int *L1, *L2;

	Sch1->get_orbit_length(L1, 0 /* verbose_level */);
	Sch2->get_orbit_length(L2, 0 /* verbose_level */);

	data_structures::tally T1, T2;

	T1.init(L1, Sch1->A->degree, FALSE, 0);

	T2.init(L2, Sch2->A->degree, FALSE, 0);



	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbit_length "
				"before P->Subspaces->compute_decomposition_based_on_tally" << endl;
	}
	P->Subspaces->compute_decomposition_based_on_tally(
			&T1, &T2, Inc, Stack, verbose_level);


	FREE_int(L1);
	FREE_int(L2);

	if (f_v) {
		cout << "action_global::compute_decomposition_based_on_orbit_length done" << endl;
	}
}




void action_global::orbits_on_equations(
		action *A,
		ring_theory::homogeneous_polynomial_domain *HPD,
	int *The_equations,
	int nb_equations, groups::strong_generators *gens,
	actions::action *&A_on_equations,
	groups::schreier *&Orb,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::orbits_on_equations" << endl;
	}

	if (f_v) {
		cout << "action_global::orbits_on_equations "
				"creating the induced action on the equations:" << endl;
	}
	A_on_equations =
			A->Induced_action->induced_action_on_homogeneous_polynomials_given_by_equations(
		HPD,
		The_equations, nb_equations,
		FALSE /* f_induce_action */, NULL /* sims *old_G */,
		verbose_level);
	if (f_v) {
		cout << "action_global::orbits_on_equations "
				"The induced action on the equations has been created, "
				"degree = " << A_on_equations->degree << endl;
	}

	if (f_v) {
		cout << "action_global::orbits_on_equations "
				"computing orbits on the equations:" << endl;
	}
	Orb = gens->orbits_on_points_schreier(
			A_on_equations,
			verbose_level - 2);

	if (FALSE) {
		cout << "action_global::orbits_on_equations "
				"We found " << Orb->nb_orbits
				<< " orbits on the equations:" << endl;
		Orb->print_and_list_orbits_tex(cout);
	}

	if (f_v) {
		cout << "action_global::orbits_on_equations done" << endl;
	}
}



groups::strong_generators *action_global::set_stabilizer_in_projective_space(
		action *A_linear,
		geometry::projective_space *P,
	long int *set, int set_size, //int &canonical_pt,
	int *canonical_set_or_NULL,
	int verbose_level)
// assuming we are in a linear action.
// added 2/28/2011, called from analyze.cpp
// November 17, 2014 moved here from TOP_LEVEL/extra.cpp
// December 31, 2014, moved here from projective_space.cpp
{
	int f_v = (verbose_level >= 1);
	geometry::object_with_canonical_form *OwCF;
	interfaces::nauty_interface_with_group Nau;

	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
	}


	OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

	OwCF->init_point_set(set, set_size, verbose_level);

	OwCF->P = P;

	int nb_rows, nb_cols;
	data_structures::bitvector *Canonical_form = NULL;

	OwCF->encoding_size(
			nb_rows, nb_cols,
			verbose_level);



	groups::strong_generators *SG;
	data_structures::nauty_output *NO;


	NO = NEW_OBJECT(data_structures::nauty_output);
	NO->allocate(nb_rows + nb_cols, 0 /* verbose_level */);

	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space "
				"before Nau.set_stabilizer_of_object" << endl;
	}

	SG = Nau.set_stabilizer_of_object(
			OwCF,
		A_linear,
		FALSE /* f_compute_canonical_form */, Canonical_form,
		NO,
		verbose_level - 2);

	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space "
				"after Nau.set_stabilizer_of_object" << endl;
	}

	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space "
				"go = " << *NO->Ago << endl;
		NO->print_stats();
	}

	FREE_OBJECT(NO);

	FREE_OBJECT(OwCF);

	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space done" << endl;
	}
	return SG;
}

void action_global::stabilizer_of_dual_hyperoval_representative(
		action *A,
		int k, int n, int no,
		data_structures_groups::vector_ge *&gens,
		std::string &stab_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *data, nb_gens, data_size;
	//int i;
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative" << endl;
	}
	K.DH_stab_gens(k, n, no, data, nb_gens, data_size, stab_order);

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	//gens->init(A, verbose_level - 2);
	//gens->allocate(nb_gens, verbose_level - 2);


	if (f_vv) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(A, data,
			nb_gens, data_size,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative "
				"after gens->init_from_data" << endl;
	}


#if 0
	if (f_vv) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative "
				"creating stabilizer generators:" << endl;
	}
	for (i = 0; i < nb_gens; i++) {
		A->make_element(gens->ith(i), data + i * data_size, 0 /*verbose_level*/);
	}
#endif

	if (f_v) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative done"
				<< endl;
	}
}

void action_global::stabilizer_of_spread_representative(
		action *A,
		int q, int k, int no,
		data_structures_groups::vector_ge *&gens,
		std::string &stab_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *data, nb_gens, data_size;
	//int i;
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative"
				<< endl;
	}
	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative "
				"before K.Spread_stab_gens" << endl;
	}
	K.Spread_stab_gens(q, k, no, data, nb_gens, data_size, stab_order);
	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative "
				"after K.Spread_stab_gens" << endl;
	}

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	//gens->init(A, verbose_level - 2);
	//gens->allocate(nb_gens, verbose_level - 2);


	if (f_vv) {
		cout << "action_global::stabilizer_of_spread_representative "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(A, data,
			nb_gens, data_size, 0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::stabilizer_of_spread_representative "
				"after gens->init_from_data" << endl;
	}
#if 0
	for (i = 0; i < nb_gens; i++) {
		A->make_element(gens->ith(i),
				data + i * data_size,
				0 /*verbose_level*/);
	}
#endif

	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative done"
				<< endl;
	}
}

void action_global::stabilizer_of_quartic_curve_representative(
		action *A,
		int q, int no,
		data_structures_groups::vector_ge *&gens,
		std::string &stab_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *data, nb_gens, data_size;
	//int i;
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "action_global::stabilizer_of_quartic_curve_representative" << endl;
	}
	if (f_v) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"before K.quartic_curves_stab_gens" << endl;
	}
	K.quartic_curves_stab_gens(q, no, data, nb_gens, data_size, stab_order);
	if (f_v) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"after K.quartic_curves_stab_gens" << endl;
	}

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	//gens->init(A, verbose_level - 2);
	//gens->allocate(nb_gens, verbose_level - 2);



	if (f_vv) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(A, data,
			nb_gens, data_size,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"after gens->init_from_data" << endl;
	}

#if 0
	if (f_vv) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"creating stabilizer generators:" << endl;
	}
	for (i = 0; i < nb_gens; i++) {
		A->make_element(gens->ith(i),
				data + i * data_size,
				0 /*verbose_level*/);
	}
#endif

	if (f_v) {
		cout << "action_global::stabilizer_of_quartic_curve_representative done"
				<< endl;
	}
}

void action_global::perform_tests(
		action *A,
		groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::perform_tests" << endl;
	}
	int r1, r2;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *perm1;
	int *perm2;
	int *perm3;
	int *perm4;
	int *perm5;
	int cnt;
	int i;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::os_interface Os;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	perm1 = NEW_int(A->degree);
	perm2 = NEW_int(A->degree);
	perm3 = NEW_int(A->degree);
	perm4 = NEW_int(A->degree);
	perm5 = NEW_int(A->degree);

	for (cnt = 0; cnt < 10; cnt++) {
		r1 = Os.random_integer(SG->gens->len);
		r2 = Os.random_integer(SG->gens->len);
		if (f_v) {
			cout << "r1=" << r1 << endl;
			cout << "r2=" << r2 << endl;
		}
		A->Group_element->element_move(SG->gens->ith(r1), Elt1, 0);
		A->Group_element->element_move(SG->gens->ith(r2), Elt2, 0);
		if (f_v) {
			cout << "Elt1 = " << endl;
			A->Group_element->element_print_quick(Elt1, cout);
		}
		A->Group_element->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, A->degree);
			cout << endl;
		}

		if (f_v) {
			cout << "Elt2 = " << endl;
			A->Group_element->element_print_quick(Elt2, cout);
		}
		A->Group_element->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm2, A->degree);
			cout << endl;
		}

		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
		if (f_v) {
			cout << "Elt3 = " << endl;
			A->Group_element->element_print_quick(Elt3, cout);
		}
		A->Group_element->element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm3, A->degree);
			cout << endl;
		}

		Combi.perm_mult(perm1, perm2, perm4, A->degree);
		if (f_v) {
			cout << "perm1 * perm2= " << endl;
			Combi.perm_print(cout, perm4, A->degree);
			cout << endl;
		}

		for (i = 0; i < A->degree; i++) {
			if (perm3[i] != perm4[i]) {
				cout << "test " << cnt
						<< " failed; something is wrong" << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "action_global::perform_tests test 1 passed" << endl;
	}

	for (cnt = 0; cnt < 10; cnt++) {
		r1 = Os.random_integer(SG->gens->len);
		if (f_v) {
			cout << "r1=" << r1 << endl;
		}
		A->Group_element->element_move(SG->gens->ith(r1), Elt1, 0);
		if (f_v) {
			cout << "Elt1 = " << endl;
			A->Group_element->element_print_quick(Elt1, cout);
		}
		A->Group_element->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, A->degree);
			cout << endl;
		}
		A->Group_element->element_invert(Elt1, Elt2, 0);
		if (f_v) {
			cout << "Elt2 = " << endl;
			A->Group_element->element_print_quick(Elt2, cout);
		}
		A->Group_element->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm2, A->degree);
			cout << endl;
		}

		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
		if (f_v) {
			cout << "Elt3 = " << endl;
			A->Group_element->element_print_quick(Elt3, cout);
		}
		A->Group_element->element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm3, A->degree);
			cout << endl;
		}

		if (!Combi.perm_is_identity(perm3, A->degree)) {
			cout << "fails the inverse test" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "action_global::perform_tests test 2 passed" << endl;
	}


	for (cnt = 0; cnt < 10; cnt++) {
		r1 = Os.random_integer(SG->gens->len);
		r2 = Os.random_integer(SG->gens->len);
		if (f_v) {
			cout << "r1=" << r1 << endl;
			cout << "r2=" << r2 << endl;
		}
		A->Group_element->element_move(SG->gens->ith(r1), Elt1, 0);
		A->Group_element->element_move(SG->gens->ith(r2), Elt2, 0);
		if (f_v) {
			cout << "Elt1 = " << endl;
			A->Group_element->element_print_quick(Elt1, cout);
		}
		A->Group_element->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, A->degree);
			cout << endl;
		}

		if (f_v) {
			cout << "Elt2 = " << endl;
			A->Group_element->element_print_quick(Elt2, cout);
		}
		A->Group_element->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm2, A->degree);
			cout << endl;
		}

		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
		if (f_v) {
			cout << "Elt3 = " << endl;
			A->Group_element->element_print_quick(Elt3, cout);
		}

		A->Group_element->element_invert(Elt3, Elt4, 0);
		if (f_v) {
			cout << "Elt4 = Elt3^-1 = " << endl;
			A->Group_element->element_print_quick(Elt4, cout);
		}


		A->Group_element->element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as Elt3 as permutation: " << endl;
			Combi.perm_print(cout, perm3, A->degree);
			cout << endl;
		}

		A->Group_element->element_as_permutation(Elt4, perm4, 0 /* verbose_level */);
		if (f_v) {
			cout << "as Elt4 as permutation: " << endl;
			Combi.perm_print(cout, perm4, A->degree);
			cout << endl;
		}

		Combi.perm_mult(perm3, perm4, perm5, A->degree);
		if (f_v) {
			cout << "perm3 * perm4= " << endl;
			Combi.perm_print(cout, perm5, A->degree);
			cout << endl;
		}

		for (i = 0; i < A->degree; i++) {
			if (perm5[i] != i) {
				cout << "test " << cnt
						<< " failed; something is wrong" << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "action_global::perform_tests test 3 passed" << endl;
	}


	if (f_v) {
		cout << "performing test 4:" << endl;
	}

	int data[] = {2,0,1, 0,1,1,0, 1,0,0,1, 1,0,0,1 };
	A->Group_element->make_element(Elt1, data, verbose_level);
	A->Group_element->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
	if (f_v) {
		cout << "as Elt1 as permutation: " << endl;
		Combi.perm_print(cout, perm1, A->degree);
		cout << endl;
	}

	A->Group_element->element_invert(Elt1, Elt2, 0);
	A->Group_element->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
	if (f_v) {
		cout << "as Elt2 as permutation: " << endl;
		Combi.perm_print(cout, perm2, A->degree);
		cout << endl;
	}


	A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
	if (f_v) {
		cout << "Elt3 = " << endl;
		A->Group_element->element_print_quick(Elt3, cout);
	}

	Combi.perm_mult(perm1, perm2, perm3, A->degree);
	if (f_v) {
		cout << "perm1 * perm2= " << endl;
		Combi.perm_print(cout, perm3, A->degree);
		cout << endl;
	}

	for (i = 0; i < A->degree; i++) {
		if (perm3[i] != i) {
			cout << "test 4 failed; something is wrong" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "action_global::perform_tests test 4 passed" << endl;
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(perm1);
	FREE_int(perm2);
	FREE_int(perm3);
	FREE_int(perm4);
	FREE_int(perm5);
}


void action_global::apply_based_on_text(
		action *A,
		std::string &input_text,
		std::string &input_group_element,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::apply_based_on_text" << endl;
	}
	if (f_v) {
		cout << "applying" << endl;
		cout << "input = " << input_text << endl;
		cout << "group element = " << input_group_element << endl;
	}
	long int *v;
	long int *w;
	int i, sz;

	int *Elt2;

	Elt2 = NEW_int(A->elt_size_in_int);

	Lint_vec_scan(input_text, v, sz);
	if (f_v) {
		cout << "v=" << endl;
		Lint_vec_print(cout, v, sz);
	}

	w = NEW_lint(sz);

	A->Group_element->make_element_from_string(Elt2, input_group_element, verbose_level);
	if (f_v) {
		cout << "B=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
	}

	for (i = 0; i < sz; i++) {
		w[i] = A->Group_element->element_image_of(v[i], Elt2, verbose_level - 1);
		if (f_v) {
			cout << "mapping " << v[i] << " -> " << w[i] << endl;
		}
	}




	{



		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		fname.assign(A->label);
		fname.append("_apply.tex");

		snprintf(str, 1000, "Application of Group Element in $%s$", A->label_tex.c_str());
		title.assign(str);



		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			A->Group_element->element_print_latex(Elt2, ost);
			ost << "$$" << endl;

			A->Group_element->element_print_for_make_element(Elt2, ost);
			ost << "\\\\" << endl;

			ost << "maps: \\\\" << endl;

			for (i = 0; i < sz; i++) {
				ost << "$" << v[i] << " \\mapsto " << w[i] << "$\\\\" << endl;
			}


			ost << "image set: \\\\" << endl;
			Lint_vec_print(ost, w, sz);
			ost << "\\\\" << endl;


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(Elt2);

	FREE_lint(v);
	FREE_lint(w);


	if (f_v) {
		cout << "action_global::apply_based_on_text" << endl;
	}
}



void action_global::multiply_based_on_text(
		action *A,
		std::string &data_A,
		std::string &data_B, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::multiply_based_on_text" << endl;
	}
	if (f_v) {
		cout << "multiplying" << endl;
		cout << "A=" << data_A << endl;
		cout << "B=" << data_B << endl;
	}
	int *Elt1;
	int *Elt2;
	int *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->make_element_from_string(Elt2, data_B, verbose_level);
	if (f_v) {
		cout << "B=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
	}

	A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
	if (f_v) {
		cout << "A*B=" << endl;
		A->Group_element->element_print_quick(Elt3, cout);
		A->Group_element->element_print_for_make_element(Elt3, cout);
		cout << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		fname.assign(A->label);
		fname.append("_mult.tex");

		snprintf(str, 1000, "Multiplication of Group Elements in $%s$", A->label_tex.c_str());
		title.assign(str);

		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			A->Group_element->element_print_latex(Elt1, ost);
			ost << "\\cdot" << endl;
			A->Group_element->element_print_latex(Elt2, ost);
			ost << "=" << endl;
			A->Group_element->element_print_latex(Elt3, ost);
			ost << "$$" << endl;

			A->Group_element->element_print_for_make_element(Elt1, ost);
			ost << "\\\\" << endl;
			A->Group_element->element_print_for_make_element(Elt2, ost);
			ost << "\\\\" << endl;
			A->Group_element->element_print_for_make_element(Elt3, ost);
			ost << "\\\\" << endl;

			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);


	if (f_v) {
		cout << "action_global::multiply_based_on_text" << endl;
	}
}

void action_global::inverse_based_on_text(
		action *A,
		std::string &data_A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::inverse_based_on_text" << endl;
	}

	if (f_v) {
		cout << "computing the inverse" << endl;
		cout << "A=" << data_A << endl;
	}
	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->element_invert(Elt1, Elt2, 0);
	if (f_v) {
		cout << "A^-1=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
		A->Group_element->element_print_for_make_element(Elt2, cout);
		cout << endl;
	}



	{


		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		fname.assign(A->label);
		fname.append("_inv.tex");

		snprintf(str, 1000, "Inverse of Group Element in $%s$", A->label_tex.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			ost << "{" << endl;
			A->Group_element->element_print_latex(Elt1, ost);
			ost << "}^{-1}" << endl;
			ost << "=" << endl;
			A->Group_element->element_print_latex(Elt2, ost);
			ost << "$$" << endl;

			A->Group_element->element_print_for_make_element(Elt1, ost);
			ost << "\\\\" << endl;
			A->Group_element->element_print_for_make_element(Elt2, ost);
			ost << "\\\\" << endl;

			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);


	if (f_v) {
		cout << "action_global::inverse_based_on_text done" << endl;
	}
}

void action_global::consecutive_powers_based_on_text(
		action *A,
		std::string &data_A,
		std::string &exponent_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::consecutive_powers_based_on_text" << endl;
	}

	if (f_v) {
		cout << "computing the power" << endl;
		cout << "A=" << data_A << endl;
		cout << "exponent=" << exponent_text << endl;
	}

	int exponent;
	data_structures::string_tools ST;

	exponent = ST.strtoi(exponent_text);

	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}



	{



		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		fname.assign(A->label);
		fname.append("_all_powers.tex");

		snprintf(str, 1000, "Consecutive Powers of Group Element in $%s$", A->label_tex.c_str());
		title.assign(str);



		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;
			int i;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			ost << "{" << endl;
			A->Group_element->element_print_latex(Elt1, ost);
			ost << "}^i" << endl;
			ost << "=" << endl;
			//element_print_latex(Elt2, ost);
			ost << "$$" << endl;

			ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|r|l|}" << endl;
			ost << "\\hline" << endl;

			ost << "i & {" << endl;
			A->Group_element->element_print_latex(Elt1, ost);
			ost << "}^i\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;


			for (i = 1; i <= exponent; i++) {
				A->Group_element->move(Elt1, Elt2);


				A->Group_element->element_power_int_in_place(Elt2,
						i, 0 /* verbose_level*/);

				if (f_v) {
					cout << "A^" << i << "=" << endl;
					A->Group_element->element_print_quick(Elt2, cout);
					A->Group_element->element_print_for_make_element(Elt2, cout);
					cout << endl;
				}



				ost << i << " & $";
				//ost << "$i=" << i << "$:" << endl;
				//ost << "$$" << endl;
				//ost << "{" << endl;
				//element_print_latex(Elt1, ost);
				//ost << "}^{" << i << "}" << endl;
				//ost << "=" << endl;
				A->Group_element->element_print_latex(Elt2, ost);
				//ost << "$$" << endl;
				ost << "$\\\\" << endl;
				ost << "\\hline" << endl;

				//element_print_for_make_element(Elt1, ost);
				//ost << "\\\\" << endl;
				//element_print_for_make_element(Elt2, ost);
				//ost << "\\\\" << endl;
			}
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
			ost << "}" << endl;

			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);


	if (f_v) {
		cout << "action_global::consecutive_powers_based_on_text done" << endl;
	}
}


void action_global::raise_to_the_power_based_on_text(
		action *A,
		std::string &data_A,
		std::string &exponent_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::raise_to_the_power_based_on_text" << endl;
	}

	if (f_v) {
		cout << "computing the power" << endl;
		cout << "A=" << data_A << endl;
		cout << "exponent=" << exponent_text << endl;
	}

	int exponent;
	data_structures::string_tools ST;

	exponent = ST.strtoi(exponent_text);

	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->move(Elt1, Elt2);


	A->Group_element->element_power_int_in_place(Elt2,
			exponent, 0 /* verbose_level*/);

	if (f_v) {
		cout << "A^" << exponent << "=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
		A->Group_element->element_print_for_make_element(Elt2, cout);
		cout << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		fname.assign(A->label);
		fname.append("_power.tex");

		snprintf(str, 1000, "Power of Group Element in $%s$", A->label_tex.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			ost << "{" << endl;
			A->Group_element->element_print_latex(Elt1, ost);
			ost << "}^{" << exponent << "}" << endl;
			ost << "=" << endl;
			A->Group_element->element_print_latex(Elt2, ost);
			ost << "$$" << endl;

			A->Group_element->element_print_for_make_element(Elt1, ost);
			ost << "\\\\" << endl;
			A->Group_element->element_print_for_make_element(Elt2, ost);
			ost << "\\\\" << endl;

			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);


	if (f_v) {
		cout << "action_global::raise_to_the_power_based_on_text done" << endl;
	}
}


void action_global::compute_orbit_of_point(
		actions::action *A,
		data_structures_groups::vector_ge &strong_generators,
		int pt, int *orbit, int &len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::schreier Schreier;
	int i, f;

	if (f_v) {
		cout << "action_global::compute_orbit_of_point "
				"computing orbit of point " << pt << endl;
	}
	Schreier.init(A, verbose_level - 2);
	Schreier.init_generators(strong_generators, verbose_level - 2);
	Schreier.compute_point_orbit(pt, 0);
	f = Schreier.orbit_first[0];
	len = Schreier.orbit_len[0];
	for (i = 0; i < len; i++) {
		orbit[i] = Schreier.orbit[f + i];
	}
	if (f_v) {
		cout << "action_global::compute_orbit_of_point done" << endl;
	}
}

void action_global::compute_orbit_of_point_generators_by_handle(
		actions::action *A,
		int nb_gen,
	int *gen_handle, int pt, int *orbit, int &len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge gens;
	int i;

	if (f_v) {
		cout << "action_global::compute_orbit_of_point_generators_by_handle" << endl;
	}
	gens.init(A, verbose_level - 2);
	gens.allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		A->Group_element->element_retrieve(gen_handle[i], gens.ith(i), 0);
	}
	compute_orbit_of_point(A, gens, pt, orbit, len, verbose_level);
	if (f_v) {
		cout << "action_global::compute_orbit_of_point_generators_by_handle done" << endl;
	}
}


int action_global::least_image_of_point(
		actions::action *A,
		data_structures_groups::vector_ge &strong_generators,
	int pt, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::schreier Schreier;
	int len, image, pos, i;

	if (f_v) {
		cout << "action_global::least_image_of_point: "
				"computing least image of " << pt << endl;
	}
	Schreier.init(A, verbose_level - 2);
	Schreier.init_generators(strong_generators, verbose_level - 2);
	Schreier.compute_point_orbit(pt, 0);
	len = Schreier.orbit_len[0];
	image = orbiter_kernel_system::Orbiter->Int_vec->minimum(Schreier.orbit, len);
	pos = Schreier.orbit_inv[image];
	Schreier.coset_rep(pos, 0 /* verbose_level */);
	A->Group_element->element_move(Schreier.cosetrep, transporter, 0);
	// we check it:
	i = A->Group_element->element_image_of(pt, transporter, 0);
	if (i != image) {
		cout << "action_global::least_image_of_point i != image" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "action_global::least_image_of_point: "
				"least image of " << pt << " is " << image << endl;
	}
	return image;
}

int action_global::least_image_of_point_generators_by_handle(
		actions::action *A,
	std::vector<int> &gen_handle,
	int pt, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge gens;
	int i;
	int nb_gen;
	int ret;

	if (f_v) {
		cout << "action_global::least_image_of_point_generators_by_handle" << endl;
	}
	nb_gen = gen_handle.size();

	if (nb_gen == 0) {
		A->Group_element->element_one(transporter, 0);
		return pt;
	}
	gens.init(A, verbose_level - 2);
	gens.allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		A->Group_element->element_retrieve(gen_handle[i], gens.ith(i), 0);
	}
	ret = least_image_of_point(A, gens, pt, transporter, verbose_level);
	if (f_v) {
		cout << "action_global::least_image_of_point_generators_by_handle done" << endl;
	}
	return ret;
}

int action_global::least_image_of_point_generators_by_handle(
		actions::action *A,
	int nb_gen, int *gen_handle,
	int pt, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge gens;
	int i, ret;

	if (f_v) {
		cout << "action_global::least_image_of_point_generators_by_handle" << endl;
	}
	if (nb_gen == 0) {
		A->Group_element->element_one(transporter, 0);
		return pt;
	}
	gens.init(A, verbose_level - 2);
	gens.allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		A->Group_element->element_retrieve(gen_handle[i], gens.ith(i), 0);
	}
	ret = least_image_of_point(A, gens, pt, transporter, verbose_level);
	if (f_v) {
		cout << "action_global::least_image_of_point_generators_by_handle done" << endl;
	}
	return ret;
}

void action_global::all_point_orbits(
		actions::action *A,
		groups::schreier &Schreier, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::all_point_orbits" << endl;
	}
	Schreier.init(A, verbose_level - 2);
	if (!A->f_has_strong_generators) {
		cout << "action_global::all_point_orbits "
				"!A->f_has_strong_generators" << endl;
		exit(1);
	}
	Schreier.init_generators(
			*A->Strong_gens->gens /* *strong_generators */,
			verbose_level - 2);
	if (f_v) {
		cout << "action_global::all_point_orbits "
				"before Schreier.compute_all_point_orbits" << endl;
	}
	Schreier.compute_all_point_orbits(verbose_level);
	if (f_v) {
		cout << "action_global::all_point_orbits "
				"after Schreier.compute_all_point_orbits" << endl;
	}
	if (f_v) {
		cout << "action_global::all_point_orbits done" << endl;
	}
}

void action_global::all_point_orbits_from_generators(
		actions::action *A,
		groups::schreier &Schreier,
		groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::all_point_orbits_from_generators" << endl;
	}
	Schreier.init(A, verbose_level - 2);
	Schreier.init_generators(
			*SG->gens /* *strong_generators */,
			verbose_level - 2);
	Schreier.compute_all_point_orbits(verbose_level);
	if (f_v) {
		cout << "action_global::all_point_orbits_from_generators done" << endl;
	}
}

void action_global::all_point_orbits_from_single_generator(
		actions::action *A,
		groups::schreier &Schreier,
		int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::all_point_orbits_from_single_generator" << endl;
	}
	data_structures_groups::vector_ge gens;

	gens.init(A, verbose_level - 2);
	gens.allocate(1, verbose_level - 2);
	A->Group_element->element_move(Elt, gens.ith(0), 0);

	Schreier.init(A, verbose_level - 2);
	Schreier.init_generators(gens, verbose_level - 2);
	if (f_v) {
		cout << "action_global::all_point_orbits_from_single_generator "
				"before Schreier.compute_all_point_orbits" << endl;
	}
	Schreier.compute_all_point_orbits(verbose_level);
	if (f_v) {
		cout << "action_global::all_point_orbits_from_single_generator "
				"after Schreier.compute_all_point_orbits" << endl;
	}
	if (f_v) {
		cout << "action_global::all_point_orbits_from_single_generator done" << endl;
	}
}



#if 0
void action_global::compute_set_orbit(
		actions::action *A,
		data_structures_groups::vector_ge &gens,
	int size, long int *set,
	int &nb_sets, long int **&Sets, int **&Transporter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *image_set;
	long int **New_Sets;
	int **New_Transporter;
	int nb_finished, allocated_nb_sets;
	int new_allocated_nb_sets, nb_gens, i, j, h;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_global::compute_set_orbit" << endl;
	}

	if (f_vv) {
		cout << "action_global::compute_set_orbit set=";
		Lint_vec_print(cout, set, size);
		cout << endl;
	}

	nb_gens = gens.len;

	allocated_nb_sets = 100;
	Sets = NEW_plint(allocated_nb_sets);
	Transporter = NEW_pint(allocated_nb_sets);
	nb_sets = 0;

	image_set = NEW_lint(size);
	Sets[0] = NEW_lint(size);
	for (i = 0; i < size; i++) {
		Sets[0][i] = set[i];
	}
	Sorting.lint_vec_heapsort(Sets[0], size);

	Transporter[0] = NEW_int(A->elt_size_in_int);
	A->element_one(Transporter[0], FALSE);

	nb_sets = 1;
	nb_finished = 0;

	while (nb_finished < nb_sets) {
		if (f_vv) {
			cout << "action_global::compute_set_orbit "
					"nb_finished=" << nb_finished
					<< " nb_sets=" << nb_sets << endl;
		}
		for (i = 0; i < nb_gens; i++) {
			A->map_a_set_and_reorder(Sets[nb_finished], image_set, size,
				gens.ith(i), 0);
			if (FALSE) {
				cout << "action_global::compute_set_orbit "
						"image under generator " << i << ":";
				Lint_vec_print(cout, image_set, size);
				cout << endl;
			}
			for (j = 0; j < nb_sets; j++) {
				if (Sorting.lint_vec_compare(Sets[j], image_set, size) == 0) {
					break;
				}
			}
			if (j < nb_sets) {
				continue;
			}
			// found a new set in the orbit:
			if (f_vv) {
				cout << "action_global::compute_set_orbit "
						"new set " << nb_sets << ":";
				Lint_vec_print(cout, image_set, size);
				cout << endl;
			}
			Sets[nb_sets] = image_set;
			image_set = NEW_lint(size);
			Transporter[nb_sets] = NEW_int(A->elt_size_in_int);
			A->element_mult(Transporter[nb_finished],
					gens.ith(i), Transporter[nb_sets], 0);
			nb_sets++;
			if (nb_sets == allocated_nb_sets) {
				new_allocated_nb_sets = allocated_nb_sets + 100;
				if (f_vv) {
					cout << "action_global::compute_set_orbit reallocating to size "
						<< new_allocated_nb_sets << endl;
				}
				New_Sets = NEW_plint(new_allocated_nb_sets);
				New_Transporter = NEW_pint(new_allocated_nb_sets);
				for (h = 0; h < nb_sets; h++) {
					New_Sets[h] = Sets[h];
					New_Transporter[h] = Transporter[h];
					}
				FREE_plint(Sets);
				FREE_pint(Transporter);
				Sets = New_Sets;
				Transporter = New_Transporter;
				allocated_nb_sets = new_allocated_nb_sets;
			}
		} // next i
		nb_finished++;
	}
	FREE_lint(image_set);
	if (f_v) {
		cout << "action_global::compute_set_orbit "
				"found an orbit of size " << nb_sets << endl;
	}
	if (f_vv) {
		cout << "action_global::compute_set_orbit "
				"the set orbit of size " << nb_sets << " is" << endl;
		for (i = 0; i < nb_sets; i++) {
			cout << i << " : ";
			Lint_vec_print(cout, Sets[i], size);
			cout << endl;
			A->element_print(Transporter[i], cout);
		}
	}
	if (f_v) {
		cout << "action_global::compute_set_orbit done" << endl;
	}
}

void action_global::delete_set_orbit(
		actions::action *A,
		int nb_sets, long int **Sets, int **Transporter)
{
	int i;

	for (i = 0; i < nb_sets; i++) {
		FREE_lint(Sets[i]);
		FREE_int(Transporter[i]);
	}
	FREE_plint(Sets);
	FREE_pint(Transporter);
}

void action_global::compute_minimal_set(
		actions::action *A,
		data_structures_groups::vector_ge &gens,
		int size, long int *set,
	long int *minimal_set, int *transporter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int **Sets;
	int **Transporter;
	int nb_sets, i;
	int min_set;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_global::compute_minimal_set" << endl;
	}


	compute_set_orbit(A, gens, size, set,
		nb_sets, Sets, Transporter,
		verbose_level);

	min_set = 0;
	for (i = 1; i < nb_sets; i++) {
		if (Sorting.lint_vec_compare(Sets[i], Sets[min_set], size) < 0) {
			min_set = i;
		}
	}
	for (i = 0; i < size; i++) {
		minimal_set[i] = Sets[min_set][i];
	}
	A->element_move(Transporter[min_set], transporter, 0);
	delete_set_orbit(A, nb_sets, Sets, Transporter);
	if (f_v) {
		cout << "action_global::compute_minimal_set done" << endl;
	}
}
#endif





void action_global::induce(
		action *old_action,
		action *new_action,
		groups::sims *old_G,
	int base_of_choice_len, long int *base_of_choice,
	int verbose_level)

// after this procedure, new_action will have
// a sims for the group and the kernel
// it will also have strong generators

// the old_action may not have a stabilizer chain,
// but it's subaction does.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	if (f_v) {
		cout << "action_global::induce "
				"verbose_level=" << verbose_level << endl;
	}

	if (f_v) {
		cout << "action_global::induce" << endl;
		cout << "action_global::induce old_action = ";
		old_action->print_info();
		cout << endl;
		cout << "action_global::induce new_action = ";
		new_action->print_info();
		cout << endl;

		cout << "action_global::induce "
				"new_action->Stabilizer_chain->A = " << new_action->Stabilizer_chain->get_A()->label << endl;

		cout << "action_global::induce old_G->A = ";
		old_G->A->print_info();
		cout << endl;
	}

	// the old action may not have a base,
	// so we don't print old_action->base_len()

	if (f_v) {

		// old_action may not have a base, so the next command is bad:
		//cout << "action_global::induce old_action->base_len=" << old_action->base_len() << endl;

		cout << "action_global::induce "
				"new_action->base_len=" << new_action->base_len() << endl;
	}

	action *subaction;
	groups::sims *G, *K;
		// will become part of the action object
		// 'this' by the end of this procedure
	ring_theory::longinteger_object go, /*go1,*/ go2, go3;
	ring_theory::longinteger_object G_order, K_order;
	ring_theory::longinteger_domain D;
	int b, i, old_base_len;
	action *fallback_action;

	if (f_v) {
		cout << "action_global::induce old_action" << endl;
		old_action->print_info();

		cout << "action_global::induce "
				"the old group is in action:" << endl;
		old_G->A->print_info();
	}

	if (old_action->subaction) {
		if (f_vv) {
			cout << "action_global::induce "
					"the old action has a subaction" << endl;
		}
		subaction = old_action->subaction;
		if (f_vv) {
			cout << "subaction is ";
			subaction->print_info();
		}
	}
	else {
		if (f_vv) {
			cout << "action_global::induce "
					"does not have subaction" << endl;
		}
		subaction = old_action;
	}
	old_G->group_order(go);
	//old_action->group_order(go1);
	subaction->group_order(go2);
	if (f_v) {
		cout << "action_global::induce" << endl;
		cout << "from old action " << old_action->label << endl;
		cout << "subaction " << subaction->label << endl;
		cout << "target order = " << go << endl;
		//cout << "old_action order = " << go1 << endl;
		cout << "subaction order = " << go2 << endl;
		cout << "old action has degree " << old_action->degree << endl;
		cout << "subaction has degree " << subaction->degree << endl;

		// old_action may not have a base, so the next command is bad:
		//cout << "base_length = " << old_action->base_len() << endl;

		cout << "subaction->base_len = " << subaction->base_len() << endl;
		if (base_of_choice_len) {
			cout << "base of choice:" << endl;
			Lint_vec_print(cout, base_of_choice, base_of_choice_len);
			cout << endl;
		}
		else {
			cout << "no base of choice" << endl;
		}
	}

	G = NEW_OBJECT(groups::sims);
	K = NEW_OBJECT(groups::sims);

	// action of G is new_action
	// action of K is fallback_action


	if (f_v) {
		cout << "action_global::induce "
				"new_action=" << new_action->label << endl;
	}

	if (f_v) {
		cout << "action_global::induce "
				"before G->init_without_base(this);" << endl;
	}
	G->init_without_base(new_action, verbose_level - 2);
	if (f_v) {
		cout << "action_global::induce "
				"after G->init_without_base(this);" << endl;
	}


	if (base_of_choice_len) {
		if (f_v) {
			cout << "action_global::induce "
					"initializing base of choice" << endl;

			// old_action may not have a base, so the next command is bad:
			//cout << "action_global::induce old_action->base_len=" << old_action->base_len() << endl;

			cout << "action_global::induce "
					"new_action->base_len=" << new_action->base_len() << endl;
		}
		for (i = 0; i < base_of_choice_len; i++) {
			b = base_of_choice[i];
			if (f_v) {
				cout << i << "-th base point is " << b << endl;
			}
			//old_base_len = old_action->base_len();
			old_base_len = new_action->base_len();

			if (f_v) {
				cout << "action_global::induce "
						"before new_action->Stabilizer_chain->reallocate_base" << endl;
			}
			new_action->Stabilizer_chain->reallocate_base(b, verbose_level);
			if (f_v) {
				cout << "action_global::induce "
						"after new_action->Stabilizer_chain->reallocate_base" << endl;
			}

			if (f_v) {
				cout << "action_global::induce "
						"before G->reallocate_base" << endl;
			}
			G->reallocate_base(old_base_len, verbose_level - 2);
			if (f_v) {
				cout << "action_global::induce "
						"after G->reallocate_base" << endl;
			}
		}
		if (f_vv) {
			cout << "action_global::induce initializing base of choice finished"
					<< endl;
		}
	}
	else {
		if (f_vv) {
			cout << "action_global::induce no base of choice given" << endl;
		}

	}

	fallback_action = subaction; // changed A. Betten Dec 27, 2011 !!!
	//fallback_action = old_action; // changed back A. Betten, May 27, 2012 !!!
		// The BLT search needs old_action
		// the translation plane search needs subaction
	if (fallback_action->base_len() == 0) {
		if (f_vv) {
			cout << "WARNING: action_global::induce "
					"fallback_action->base_len == 0" << endl;
			cout << "fallback_action=" << fallback_action->label << endl;
			cout << "subaction=" << subaction->label << endl;
			cout << "old_action=" << old_action->label << endl;
			cout << "old_G->A=" << old_G->A->label << endl;
		}
		fallback_action = old_G->A;
		if (f_vv) {
			cout << "changing fallback action to " << fallback_action->label
					<< endl;
		}
	}


	if (f_v) {
		cout << "action_global::induce "
				"new_action=" << new_action->label
				<< " of degree " << new_action->degree << endl;
		cout << "action_global::induce "
				"fallback_action=" << fallback_action->label
				<< " of degree " << fallback_action->degree << endl;
	}


	if (f_v) {
		cout << "action_global::induce "
				"before K->init" << endl;
	}
	K->init(fallback_action, verbose_level - 2);
	if (f_v) {
		cout << "action_global::induce "
				"after K->init" << endl;
	}

	if (f_v) {
		cout << "action_global::induce "
				"before G->init_trivial_group" << endl;
	}
	G->init_trivial_group(verbose_level - 2);
	if (f_v) {
		cout << "action_global::induce "
				"after G->init_trivial_group" << endl;
	}

	if (f_v) {
		cout << "action_global::induce "
				"before K->init_trivial_group" << endl;
	}
	K->init_trivial_group(verbose_level - 2);
	if (f_v) {
		cout << "action_global::induce "
				"after K->init_trivial_group" << endl;
	}

	if (f_v) {
		cout << "action_global::induce "
				"before G->build_up_group_random_process" << endl;
	}
	G->build_up_group_random_process(
			K, old_G, go,
		FALSE /*f_override_chose_next_base_point*/,
		NULL /*choose_next_base_point_method*/,
		verbose_level - 1);
	if (f_v) {
		cout << "action_global::induce "
				"after G->build_up_group_random_process" << endl;
	}
	if (f_v) {
		cout << "action_global::induce "
				"new_action=" << new_action->label
				<< " of degree " << new_action->degree << endl;
		cout << "action_global::induce "
				"G->A->label=" << G->A->label
				<< " of degree " << G->A->degree << endl;
	}

	G->group_order(G_order);
	K->group_order(K_order);
	if (f_v) {
		cout << "action_global::induce ";
		cout << "found a group in action " << G->A->label
				<< " of order " << G_order << " ";
		cout << "transversal lengths: ";
		for (int t = 0; t < G->A->base_len(); t++) {
			cout << G->get_orbit_length(t) << ", ";
		}
		cout << " base: ";
		for (int t = 0; t < G->A->base_len(); t++) {
			cout << G->A->base_i(t) << ", ";
		}
		//int_vec_print(cout, G->get_orbit_length(i), G->A->base_len());
		cout << endl;

		cout << "action_global::induce kernel in action " << fallback_action->label
				<< " of order " << K_order << " ";
		cout << "transversal lengths: ";
		for (int t = 0; t < fallback_action->base_len(); t++) {
			cout << K->get_orbit_length(t) << ", ";
		}
		cout << " base: ";
		for (int t = 0; t < fallback_action->base_len(); t++) {
			cout << fallback_action->base_i(t) << ", ";
		}
		//int_vec_print(cout, K->get_orbit_length(), K->A->base_len());
		cout << endl;
	}
	D.mult(G_order, K_order, go3);
	if (D.compare(go3, go) != 0) {
		cout << "action_global::induce "
				"group orders do not match: "
				<< go3 << " != " << go << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "action_global::induce "
				"product of group orders equals "
				"old group order" << endl;
	}
	if (f_vv) {
		cout << "action_global::induce "
				"before init_sims_only" << endl;
	}
	new_action->init_sims_only(G, verbose_level - 2);
	if (f_vv) {
		cout << "action_global::induce "
				"after init_sims_only" << endl;
	}
	new_action->f_has_kernel = TRUE;
	new_action->Kernel = K;

	//init_transversal_reps_from_stabilizer_chain(G, verbose_level - 2);
	if (f_vv) {
		cout << "action_global::induce "
				"before new_action->compute_strong_generators_from_sims" << endl;
	}
	new_action->compute_strong_generators_from_sims(verbose_level - 2);
	if (f_vv) {
		cout << "action_global::induce "
				"after new_action->compute_strong_generators_from_sims" << endl;
	}
	if (f_v) {
		cout << "action_global::induce done" << endl;
	}
}


void action_global::induced_action_override_sims(
	action *old_action, action *new_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::induced_action_override_sims" << endl;
		cout << "action_global::induced_action_override_sims "
				"old_action = ";
		old_action->print_info();
		cout << endl;
		cout << "action_global::induced_action_override_sims "
				"old_G->A = ";
		old_G->A->print_info();
		cout << endl;
	}
	if (f_v) {
		cout << "action_global::induced_action_override_sims "
				"before induce" << endl;
	}
	induce(old_action, new_action,
			old_G,
		0 /* base_of_choice_len */, NULL /* base_of_choice */,
		verbose_level - 1);
	if (f_v) {
		cout << "action_global::induced_action_override_sims "
				"after induce" << endl;
	}

	if (f_v) {
		cout << "action_global::induced_action_override_sims done" << endl;
	}
}

void action_global::make_canonical(
		action *A, groups::sims *Sims,
		int size, long int *set,
	long int *canonical_set, int *transporter,
	long int &total_backtrack_nodes,
	int f_get_automorphism_group, groups::sims *Aut,
	int verbose_level)
{
	//verbose_level += 10;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action_global::make_canonical" << endl;
	}



	int *Elt1, *Elt2, *Elt3;
	long int *set1;
	long int *set2;
	int backtrack_level;
	long int backtrack_nodes, cnt = 0;
	//int f_get_automorphism_group = TRUE;
	//sims Aut;

	total_backtrack_nodes = 0;
	if (f_v) {
		cout << "action_global::make_canonical" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "elt_size_in_int=" << A->elt_size_in_int << endl;
	}
	if (f_vv) {
		cout << "the input set is ";
		Lint_vec_print(cout, set, size);
		cout << endl;
	}

#if 0
	if (!f_has_sims) {
		cout << "action_global::make_canonical  sims is not available" << endl;
		exit(1);
	}
#endif

	ring_theory::longinteger_object go;
	Sims->group_order(go);
	if (f_v) {
		cout << "action_global::make_canonical "
				"group order = " << go << endl;
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	set1 = NEW_lint(size);
	set2 = NEW_lint(size);

	Lint_vec_copy(set, set1, size);
	A->Group_element->element_one(Elt1, FALSE);

	int c;

	while (TRUE) {
		cnt++;
		//if (cnt == 4) verbose_level += 10;
		if (f_v) {
			cout << "action_global::make_canonical iteration "
						<< cnt << " before is_minimal_witness" << endl;
		}
		c = A->is_minimal_witness(/*default_action,*/ size, set1, Sims,
			backtrack_level, set2, Elt2,
			backtrack_nodes,
			f_get_automorphism_group, *Aut,
			verbose_level - 1);
		if (f_v) {
			cout << "action_global::make_canonical iteration "
						<< cnt << " after is_minimal_witness c=" << c << endl;
		}

		if (c) {
			total_backtrack_nodes += backtrack_nodes;
			if (f_v) {
				cout << "action_global::make_canonical: is minimal, "
						"after iteration " << cnt << " with "
					<< backtrack_nodes << " backtrack nodes, total:"
					<< total_backtrack_nodes << endl;
			}
			break;
		}
		//if (cnt == 4) verbose_level -= 10;
		total_backtrack_nodes += backtrack_nodes;
		if (f_v) {
			cout << "action_global::make_canonical "
					"finished iteration " << cnt;
			if (f_vv) {
				Lint_vec_print(cout, set2, size);
			}
			cout << " with "
				<< backtrack_nodes << " backtrack nodes, total:"
				<< total_backtrack_nodes << endl;
		}
		Lint_vec_copy(set2, set1, size);
		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
		A->Group_element->element_move(Elt3, Elt1, 0);

	}
	Lint_vec_copy(set1, canonical_set, size);
	A->Group_element->element_move(Elt1, transporter, FALSE);

	if (!A->Group_element->check_if_transporter_for_set(transporter,
			size, set, canonical_set, verbose_level - 3)) {
		cout << "action_global::make_canonical "
				"check_if_transporter_for_set returns FALSE" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "action_global::make_canonical succeeds in " << cnt
				<< " iterations, total_backtrack_nodes="
				<< total_backtrack_nodes << endl;
		ring_theory::longinteger_object go;
		Aut->group_order(go);
		cout << "the automorphism group has order " << go << endl;
	}
	if (f_vv) {
		cout << "the canonical set is ";
		Lint_vec_print(cout, canonical_set, size);
		cout << endl;
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_lint(set1);
	FREE_lint(set2);
	//exit(1);
	if (f_v) {
		cout << "action_global::make_canonical done" << endl;
	}
}



void action_global::make_element_which_moves_a_line_in_PG3q(
		action *A,
		geometry::projective_space_of_dimension_three *P3,
		//geometry::grassmann *Gr,
		long int line_rk, int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::make_element_which_moves_a_line_in_PG3q" << endl;
	}

#if 0
	int M[4 * 4];
	int N[4 * 4 + 1]; // + 1 if f_semilinear
	int base_cols[4];
	int r, c, i, j;

	//int_vec_zero(M, 16);
	Gr->unrank_lint_here(M, line_rk, 0 /*verbose_level*/);
	r = Gr->F->Linear_algebra->Gauss_simple(
			M, 2, 4, base_cols, 0 /* verbose_level */);
	Gr->F->Linear_algebra->kernel_columns(
			4, r, base_cols, base_cols + r);

	for (i = r; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			if (j == base_cols[i]) {
				c = 1;
			}
			else {
				c = 0;
			}
			M[i * 4 + j] = c;
		}
	}
	Gr->F->Linear_algebra->matrix_inverse(M, N, 4, 0 /* verbose_level */);
#endif

	int Mtx17[17];

	if (f_v) {
		cout << "action_global::make_element_which_moves_a_line_in_PG3q "
				"before P3->make_element_which_moves_a_line_in_PG3q" << endl;
	}
	P3->make_element_which_moves_a_line_in_PG3q(
			line_rk, Mtx17,
			verbose_level);
	if (f_v) {
		cout << "action_global::make_element_which_moves_a_line_in_PG3q "
				"after P3->make_element_which_moves_a_line_in_PG3q" << endl;
	}

	Mtx17[16] = 0;



	//N[4 * 4] = 0;
	A->Group_element->make_element(Elt, Mtx17, 0);

	if (f_v) {
		cout << "action_global::make_element_which_moves_a_line_in_PG3q done" << endl;
	}
}


void action_global::orthogonal_group_random_generator(
		action *A,
		orthogonal_geometry::orthogonal *O,
		algebra::matrix_group *M,
	int f_siegel,
	int f_reflection,
	int f_similarity,
	int f_semisimilarity,
	int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	int *Mtx;

	if (f_v) {
		cout << "action_global::orthogonal_group_random_generator" << endl;
		cout << "f_siegel=" << f_siegel << endl;
		cout << "f_reflection=" << f_reflection << endl;
		cout << "f_similarity=" << f_similarity << endl;
		cout << "f_semisimilarity=" << f_semisimilarity << endl;
		cout << "n=" << M->n << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Mtx = NEW_int(M->n * M->n + 1);

	if (f_v) {
		cout << "action_global::orthogonal_group_random_generator "
				"before O->random_generator_for_orthogonal_group" << endl;
	}

	O->Orthogonal_group->random_generator_for_orthogonal_group(
			M->f_semilinear /* f_action_is_semilinear */,
		f_siegel,
		f_reflection,
		f_similarity,
		f_semisimilarity,
		Mtx, verbose_level - 1);

	if (f_v) {
		cout << "action_global::orthogonal_group_random_generator "
				"after O->random_generator_for_orthogonal_group" << endl;
		cout << "Mtx=" << endl;
		Int_matrix_print(Mtx, M->n, M->n);
	}
	A->Group_element->make_element(Elt, Mtx, verbose_level - 1);


	FREE_int(Mtx);


	if (f_vvv) {
		cout << "action_global::orthogonal_group_random_generator "
				"random generator:" << endl;
		A->Group_element->element_print_quick(Elt, cout);
	}
	if (f_v) {
		cout << "action_global::orthogonal_group_random_generator done" << endl;
	}
}





//#############################################################################

void callback_choose_random_generator_orthogonal(int iteration,
	int *Elt, void *data, int verbose_level)
{
	//verbose_level += 5;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::callback_choose_random_generator_orthogonal "
				"iteration=" << iteration << endl;
		}

	groups::schreier_sims *ss = (groups::schreier_sims *) data;
	action *A = ss->GA;
	action *subaction = ss->KA;
	algebra::matrix_group *M;
#if 0
	int f_siegel = TRUE;
	int f_reflection = TRUE;
	int f_similarity = TRUE;
	int f_semisimilarity = TRUE;
#endif

	induced_actions::action_on_orthogonal *AO;
	orthogonal_geometry::orthogonal *O;
	action_global AG;

	AO = A->G.AO;
	O = AO->O;

	M = subaction->G.matrix_grp;
	if (f_v) {
		cout << "action_global::callback_choose_random_generator_orthogonal "
				"iteration=" << iteration
				<< " before M->orthogonal_group_random_generator"
				<< endl;
	}
	AG.orthogonal_group_random_generator(
			ss->GA,
			O,
			M,
		f_generator_orthogonal_siegel,
		f_generator_orthogonal_reflection,
		f_generator_orthogonal_similarity,
		f_generator_orthogonal_semisimilarity,
		Elt, verbose_level - 2);
	//M->GL_invert_internal(Elt, Elt + M->elt_size_int_half, 0);
	if (f_v) {
		cout << "action_global::callback_choose_random_generator_orthogonal "
				"iteration=" << iteration
				<< " after M->orthogonal_group_random_generator"
				<< endl;
	}

	if (f_v) {
		cout << "action_global::callback_choose_random_generator_orthogonal "
				"iteration=" << iteration << " done" << endl;
	}
}


void action_global::init_base(
		actions::action *A, algebra::matrix_group *M, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action_global::init_base" << endl;
	}
	if (M->f_projective) {
		if (f_vv) {
			cout << "action_global::init_base "
					"before init_base_projective" << endl;
		}
		init_base_projective(A, M, verbose_level - 2);
		if (f_vv) {
			cout << "action_global::init_base "
					"after init_base_projective" << endl;
		}
	}
	else if (M->f_affine) {
		if (f_vv) {
			cout << "action_global::init_base "
					"before init_base_affine" << endl;
		}
		init_base_affine(A, M, verbose_level - 2);
		if (f_vv) {
			cout << "action_global::init_base "
					"after init_base_affine" << endl;
		}
	}
	else if (M->f_general_linear) {
		if (f_vv) {
			cout << "action_global::init_base "
					"before init_base_general_linear" << endl;
		}
		init_base_general_linear(A, M, verbose_level - 2);
		if (f_vv) {
			cout << "action_global::init_base "
					"after init_base_general_linear" << endl;
		}
	}
	else {
		cout << "action_global::init_base  "
				"group type unknown" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "action_global::init_base done" << endl;
	}
}

void action_global::init_base_projective(
		actions::action *A, algebra::matrix_group *M, int verbose_level)
// initializes A->degree, A->Stabilizer_chain
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int q = M->GFq->q;
	algebra::group_generators_domain GG;
	int base_len;

	if (f_v) {
		cout << "action_global::init_base_projective "
				"verbose_level=" << verbose_level << endl;
	}
	A->degree = M->degree;
	if (f_vv) {
		cout << "action_global::init_base_projective "
				"degree=" << M->degree << endl;
	}
	if (f_vv) {
		cout << "action_global::init_base_projective "
				"before GG.matrix_group_base_len_projective_group" << endl;
	}
	base_len = GG.matrix_group_base_len_projective_group(
			M->n, q, M->f_semilinear, verbose_level);
	if (f_vv) {
		cout << "action_global::init_base_projective "
				"after GG.matrix_group_base_len_projective_group" << endl;
	}

	A->Stabilizer_chain = NEW_OBJECT(actions::stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A, base_len, verbose_level);
	//A->Stabilizer_chain->base_len = base_len;
	//A->allocate_base_data(A->base_len);
	if (f_vv) {
		cout << "action_global::init_base_projective "
				"A->base_len()=" << A->base_len() << endl;
	}

	if (f_v) {
		cout << "action_global::init_base_projective "
				"before init_projective_matrix_group" << endl;
	}

	A->Stabilizer_chain->init_projective_matrix_group(
			M->GFq, M->n, M->f_semilinear, A->degree,
			verbose_level);

	if (f_v) {
		cout << "action_global::init_base_projective "
				"after init_projective_matrix_group" << endl;
	}

	if (f_v) {
		cout << "action_global::init_base_projective: finished" << endl;
	}
}

void action_global::init_base_affine(
		actions::action *A, algebra::matrix_group *M, int verbose_level)
// initializes A->degree, A->Stabilizer_chain
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int q = M->GFq->q;
	algebra::group_generators_domain GG;
	int base_len;

	if (f_v) {
		cout << "action_global::init_base_affine "
				"verbose_level=" << verbose_level << endl;
	}
	A->degree = M->degree;
	if (f_vv) {
		cout << "action_global::init_base_affine degree="
				<< A->degree << endl;
	}
	base_len = GG.matrix_group_base_len_affine_group(
			M->n, q, M->f_semilinear, verbose_level - 1);
	if (f_vv) {
		cout << "action_global::init_base_affine base_len="
				<< base_len << endl;
	}

	A->Stabilizer_chain = NEW_OBJECT(actions::stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A, base_len, verbose_level);
	//A->Stabilizer_chain->base_len = base_len;
	//A->allocate_base_data(A->base_len);

	if (f_v) {
		cout << "action_global::init_base_affine before "
				"init_affine_matrix_group" << endl;
	}
	A->Stabilizer_chain->init_affine_matrix_group(
			M->GFq, M->n, M->f_semilinear, A->degree,
			verbose_level);
	if (f_v) {
		cout << "action_global::init_base_affine after "
				"init_affine_matrix_group" << endl;
	}

	if (f_v) {
		cout << "action_global::init_base_affine: finished" << endl;
	}
}

void action_global::init_base_general_linear(
		actions::action *A, algebra::matrix_group *M, int verbose_level)
// initializes A->degree, A->Stabilizer_chain
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int q = M->GFq->q;
	algebra::group_generators_domain GG;
	int base_len;

	if (f_v) {
		cout << "action_global::init_base_general_linear "
				"verbose_level=" << verbose_level << endl;
	}
	A->degree = M->degree;
	if (f_vv) {
		cout << "action_global::init_base_general_linear "
				"degree=" << A->degree << endl;
	}
	base_len = GG.matrix_group_base_len_general_linear_group(
			M->n, q, M->f_semilinear, verbose_level - 1);

	if (f_vv) {
		cout << "action_global::init_base_general_linear "
				"base_len=" << base_len << endl;
	}

	A->Stabilizer_chain = NEW_OBJECT(actions::stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A, base_len, verbose_level);
	//A->Stabilizer_chain->base_len = base_len;
	//A->allocate_base_data(A->base_len);

	if (f_v) {
		cout << "action_global::init_base_general_linear before "
				"init_linear_matrix_group" << endl;
	}
	A->Stabilizer_chain->init_linear_matrix_group(
			M->GFq, M->n, M->f_semilinear, A->degree,
			verbose_level);
	if (f_v) {
		cout << "action_global::init_base_general_linear after "
				"init_linear_matrix_group" << endl;
	}

	if (f_v) {
		cout << "action_global::init_base_affine: finished" << endl;
	}
}

void action_global::substitute_semilinear(
		action *A,
		ring_theory::homogeneous_polynomial_domain *HPD,
		int *Elt,
		int *input, int *output,
		int verbose_level)
{
	int f_v = (verbose_level > 1);


	if (f_v) {
		cout << "action_global::substitute_semilinear" << endl;
	}

	int *Elt1;
	algebra::matrix_group *mtx;
	int f_semilinear;
	int n;

	Elt1 = NEW_int(A->elt_size_in_int);

	mtx = A->G.matrix_grp;
	f_semilinear = mtx->f_semilinear;
	n = mtx->n;


	A->Group_element->element_invert(Elt, Elt1, 0);


	if (f_semilinear) {
		HPD->substitute_semilinear(input, output,
				f_semilinear, Elt[n * n], Elt1,
				0 /* verbose_level */);
	}
	else {
		HPD->substitute_linear(input, output, Elt1,
				0 /* verbose_level */);
	}


	FREE_int(Elt1);

	if (f_v) {
		cout << "action_global::substitute_semilinear done" << endl;
	}
}




}}}

