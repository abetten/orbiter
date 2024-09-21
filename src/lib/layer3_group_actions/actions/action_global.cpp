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


void action_global::action_print_symmetry_group_type(
		std::ostream &ost,
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
	else if (a == polarity_extension_t) {
		txt.assign("polarity_extension_t");
		tex.assign("polarity extension");
	}
	else if (a == permutation_representation_t) {
		txt.assign("permutation_representation_t");
		tex.assign("permutation representation");
	}
	else if (a == action_on_sets_t) {
		txt.assign("action_on_sets_t");
		tex.assign("action on subsets");
	}
	else if (a == action_on_subgroups_t) {
		txt.assign("action_on_subgroups_t");
		tex.assign("action on subgroups");
	}
	else if (a == action_on_k_subsets_t) {
		txt.assign("action_on_k_subsets_t");
		tex.assign("action on k-subsets");
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
	else if (a == action_on_orthogonal_t) {
		txt.assign("action_on_orthogonal_t");
		tex.assign("action on orthogonal");
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
	else if (a == action_by_representation_t) {
		txt.assign("action_by_representation_t");
		tex.assign("action by representation");
	}
	else if (a == action_by_subfield_structure_t) {
		txt.assign("action_by_subfield_structure_t");
		tex.assign("action by subfield structure");
	}
	else if (a == action_on_bricks_t) {
		txt.assign("action_on_bricks_t");
		tex.assign("action on bricks");
	}
	else if (a == action_on_andre_t) {
		txt.assign("action_on_andre_t");
		tex.assign("action on andre");
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
	else if (a == action_on_set_partitions_t) {
		txt.assign("action_on_set_partitions_t");
		tex.assign("action on set partitions");
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



void action_global::automorphism_group_as_permutation_group(
		l1_interfaces::nauty_output *NO,
		actions::action *&A_perm,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "action_global::automorphism_group_as_permutation_group" << endl;
	}
	A_perm = NEW_OBJECT(actions::action);


	if (f_v) {
		cout << "action_global::automorphism_group_as_permutation_group "
				"before A_perm->Known_groups->init_permutation_group_from_generators" << endl;
	}
	A_perm->Known_groups->init_permutation_group_from_nauty_output(
			NO,
			verbose_level - 2);
	if (f_v) {
		cout << "action_global::automorphism_group_as_permutation_group "
				"after A_perm->Known_groups->init_permutation_group_from_generators" << endl;
	}

	if (f_vv) {
		cout << "action_global::automorphism_group_as_permutation_group "
				"create_automorphism_group_of_incidence_structure: created action ";
		A_perm->print_info();
		cout << endl;
	}


	if (f_v) {
		cout << "action_global::automorphism_group_as_permutation_group done" << endl;
	}
}

void action_global::reverse_engineer_linear_group_from_permutation_group(
		actions::action *A_linear,
		geometry::projective_space *P,
		groups::strong_generators *&SG,
		actions::action *&A_perm,
		l1_interfaces::nauty_output *NO,
		int verbose_level)
// called from
// combinatorial_object_with_properties::lift_generators_to_matrix_group
// combinatorial_object_with_properties::init_object_in_projective_space
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group" << endl;
	}

	if (f_v) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"before automorphism_group_as_permutation_group" << endl;
	}

	automorphism_group_as_permutation_group(
				NO,
				A_perm,
				verbose_level - 2);

	if (f_v) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"after automorphism_group_as_permutation_group" << endl;
	}

	data_structures_groups::vector_ge *gens_in; // permutations from nauty
	data_structures_groups::vector_ge *gens_out; // matrices


	gens_in = A_perm->Strong_gens->gens;

	if (f_v) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"before reverse_engineer_semilinear_group" << endl;
	}
	reverse_engineer_semilinear_group(
			A_perm, A_linear,
			P,
			gens_in,
			gens_out,
			verbose_level - 2);
	if (f_v) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"after reverse_engineer_semilinear_group" << endl;
	}


	if (f_vvv) {
		gens_out->print(cout);
	}


	if (f_vv) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"we are now testing the generators:" << endl;
	}


	test_if_two_actions_agree_vector(
			A_linear, A_perm,
			gens_out, gens_in,
			verbose_level - 2);


	if (f_vv) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"the generators are OK" << endl;
	}



	ring_theory::longinteger_object target_go;

	NO->Ago->assign_to(target_go);

	if (f_vv) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"before A_linear->generators_to_strong_generators" << endl;
	}
	A_linear->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens_out, SG, 0 /*verbose_level - 3*/);
	if (f_vv) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group "
				"after A_linear->generators_to_strong_generators" << endl;
	}




	// ToDo what about gens_out, should it be freed ?




	if (f_v) {
		cout << "action_global::reverse_engineer_linear_group_from_permutation_group done" << endl;
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
			cout << "action_global::make_generators_stabilizer_of_three_components "
					"step 1: " << h << " / " << len << endl;
		}
		P = gens_PGL_k->gens->ith(h);
		//P = gens_PGL_k->ith(h);

		// Q := diag(P,P)
		orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
				Q, k, P, Zero, Zero, P);
		if (Mtx->f_semilinear) {
			Q[n * n] = P[k * k];
		}
		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components "
					"Q=" << endl;
			Int_matrix_print(Q, n, n);
		}
		Int_vec_copy(Q, Data + idx * sz, sz);
		idx++;
	}

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components "
				"step 2" << endl;
	}
	// Q := matrix(0,I,I,0):
	orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
			Q, k, Zero, Id, Id, Zero);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
	}
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components "
				"Q=" << endl;
		Int_matrix_print(Q, n, n);
	}
	Int_vec_copy(Q, Data + idx * sz, sz);
	idx++;

	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components "
				"step 3" << endl;
	}
	// Q := matrix(0,I,-I,-I):
	orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(
			Q, k, Zero, Id, minusId, minusId);
	if (Mtx->f_semilinear) {
		Q[n * n] = 0;
	}
	if (f_v) {
		cout << "action_global::make_generators_stabilizer_of_three_components "
				"Q=" << endl;
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
		cout << "action_global::make_generators_stabilizer_of_three_components "
				"step 4" << endl;
	}


	gens->init(A_PGL_n_q, verbose_level - 2);
	gens->allocate(new_len, verbose_level - 2);
	for (h = 0; h < new_len; h++) {

		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components "
					"step 4: " << h << " / " << new_len << endl;
		}

		if (f_v) {
			cout << "action_global::make_generators_stabilizer_of_three_components "
					"generator=" << endl;
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
	A->Known_groups->init_projective_group(
			n, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
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
		}
		else {
			Int_vec_zero(Gens + h * elt_size, elt_size);
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


	int f_generator_orthogonal_siegel = true;
	int f_generator_orthogonal_reflection = true;
	int f_generator_orthogonal_similarity = true;
	int f_generator_orthogonal_semisimilarity = true;


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
		cout << "action_global::retract_generators "
				"retracting generators" << endl;
	}
	gens_out->init(AQ, verbose_level - 2);
	gens_out->allocate(nb_gens, verbose_level - 2);
	for (t = 0; t < nb_gens; t++) {
		if (f_vv) {
			cout << "action_global::retract_generators " << t
					<< " / " << nb_gens << endl;
		}
		Eltq = gens_in->ith(t);
		S->retract_matrix(
				Eltq, n, Mtx, m, 0 /* verbose_level */);
		if (f_vv) {
			cout << "action_global::retract_generators "
					"retracted matrix:" << endl;
			Int_matrix_print(Mtx, m, m);
		}
		AQ->Group_element->make_element(
				EltQ, Mtx, 0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "action_global::retract_generators "
					"after make_element:" << endl;
			AQ->Group_element->element_print_quick(EltQ, cout);
		}
		AQ->Group_element->element_move(
				EltQ, gens_out->ith(t), 0);
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

	Strong_gens->init_from_sims(
			Sims, 0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::lift_generators_to_subfield_structure "
				"strong generators are:" << endl;
		Strong_gens->print_generators(cout, verbose_level - 1);
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
			degree, perm, 0, false, true, verbose_level);
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
	int f_big = false;
	int f_doit = true;
	
	if (f_v) {
		cout << "action_global::perm_print_cycles_sorted_by_length, "
				"degree=" << degree << endl;
	}
	
	if (degree > 500) {
		f_big = true;
	}
	A = NEW_OBJECT(action);
	int f_no_base = false;
	
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
		f_doit = false;
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
				while (true) {
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

	C.init(S.orbit_len, S.nb_orbits, false, 0);
	ost << " cycle type: ";
	C.print_file(ost, true /* f_backwards */);
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
	group_constructions::direct_product *P;
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
	std::string label_of_set_tex;

	label_of_set.assign("_direct_product");
	label_of_set_tex.assign("\\_direct\\_product");

	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict "
				"before A_direct_product->Induced_action->restricted_action" << endl;
	}
	Adp = A_direct_product->Induced_action->restricted_action(
			points, nb_points,
			label_of_set, label_of_set_tex,
			verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group_and_restrict "
				"after A_direct_product->Induced_action->restricted_action" << endl;
	}
	Adp->f_is_linear = false;


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
	group_constructions::direct_product *P;
	action *A;

	if (f_v) {
		cout << "action_global::init_direct_product_group" << endl;
		cout << "M1=" << M1->label << endl;
		cout << "M2=" << M2->label << endl;
	}

	A = NEW_OBJECT(action);
	P = NEW_OBJECT(group_constructions::direct_product);



	A->type_G = direct_product_t;
	A->G.direct_product_group = P;
	A->f_allocated = true;

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before P->init" << endl;
	}
	P->init(M1, M2, verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after P->init" << endl;
	}

	A->f_is_linear = false;
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
	A->Group_element->allocate_element_data();




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
	P->make_strong_generators_data(
			gens_data,
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
	A->Strong_gens->init_from_data(
			A,
			gens_data, gens_nb, gens_size,
			A->get_transversal_length(),
			nice_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after A->Strong_gens->init_from_data" << endl;
	}
	FREE_OBJECT(nice_gens);
	A->f_has_strong_generators = true;
	FREE_int(gens_data);

#if 0
	groups::sims *S;

	S = NEW_OBJECT(groups::sims);

	S->init(A, verbose_level - 2);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before S->init_generators" << endl;
	}
	S->init_generators(
			*A->Strong_gens->gens, verbose_level);
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

	A->init_sims_only(
			S, verbose_level);

	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after init_sims_only" << endl;
	}

	A->compute_strong_generators_from_sims(
			0/*verbose_level - 2*/);

#else
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"before compute_sims" << endl;
	}
	compute_sims(
			A, verbose_level);
	if (f_v) {
		cout << "action_global::init_direct_product_group "
				"after compute_sims" << endl;
	}

#endif

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


action *action_global::init_polarity_extension_group_and_restrict(
		actions::action *A,
		geometry::projective_space *P,
		geometry::polarity *Polarity,
		int f_on_middle_layer_grassmannian,
		int f_on_points_and_hyperplanes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_new;
	action *A_new_r;
	group_constructions::polarity_extension *Polarity_extension;
	long int *points;
	int nb_points;
	int i;

	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict" << endl;
		cout << "action_global::init_polarity_extension_group_and_restrict A=" << A->label << endl;
		cout << "action_global::init_polarity_extension_group_and_restrict P=" << P->label_txt << endl;
		cout << "action_global::init_polarity_extension_group_and_restrict Polarity=" << Polarity->label_txt << ", degree sequence = " << Polarity->degree_sequence_txt << endl;
		cout << "action_global::init_polarity_extension_group_and_restrict f_on_middle_layer_grassmannian=" << f_on_middle_layer_grassmannian << endl;
		cout << "action_global::init_polarity_extension_group_and_restrict f_on_points_and_hyperplanes=" << f_on_points_and_hyperplanes << endl;
	}
	A_new = NEW_OBJECT(action);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"before init_polarity_extension_group" << endl;
	}
	A_new = init_polarity_extension_group(A, P, Polarity, verbose_level);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"after init_polarity_extension_group" << endl;
	}
	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"A_new:" << endl;
		A_new->print_info();
	}
	if (A_new->Strong_gens == NULL) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"A_new->Strong_gens == NULL" << endl;
		exit(1);
	}


	Polarity_extension = A_new->G.Polarity_extension;
	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict" << endl;
		cout << "Polarity_extension->Polarity->total_degree=" << Polarity_extension->Polarity->total_degree << endl;
	}

	if (f_on_middle_layer_grassmannian) {

#if 0
		int nb_ranks;
		int *rank_sequence;
		int *rank_sequence_opposite;
		long int *nb_objects;
		long int *offset;
		int total_degree;
#endif

		if (EVEN(Polarity_extension->Polarity->nb_ranks)) {
			cout << "action_global::init_polarity_extension_group_and_restrict "
					"error: for on_middle_layer_grassmannian we need an odd number of ranks" << endl;
			exit(1);
		}

		int half, middle_rank;

		half = Polarity_extension->Polarity->nb_ranks >> 1;
		middle_rank = Polarity_extension->Polarity->rank_sequence[half];
		nb_points = Polarity_extension->Polarity->nb_objects[half];

		if (f_v) {
			cout << "action_global::init_polarity_extension_group_and_restrict f_on_middle_layer_grassmannian" << endl;
			cout << "action_global::init_polarity_extension_group_and_restrict half = " << half << endl;
			cout << "action_global::init_polarity_extension_group_and_restrict middle_rank = " << middle_rank << endl;
			cout << "action_global::init_polarity_extension_group_and_restrict nb_points = " << nb_points << endl;
		}

		points = NEW_lint(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = Polarity_extension->perm_offset_i[1] + Polarity_extension->Polarity->offset[half] + i;
		}

	}
	else if (f_on_points_and_hyperplanes) {

		int point_idx, hyperplane_idx, j;

		point_idx = 0;
		hyperplane_idx = Polarity_extension->Polarity->nb_ranks - 1;
		nb_points = Polarity_extension->Polarity->nb_objects[point_idx] + Polarity_extension->Polarity->nb_objects[hyperplane_idx];

		if (f_v) {
			cout << "action_global::init_polarity_extension_group_and_restrict f_on_points_and_hyperplanes" << endl;
			cout << "action_global::init_polarity_extension_group_and_restrict point_idx = " << point_idx << endl;
			cout << "action_global::init_polarity_extension_group_and_restrict hyperplane_idx = " << hyperplane_idx << endl;
			cout << "action_global::init_polarity_extension_group_and_restrict nb_points = " << nb_points << endl;
		}

		points = NEW_lint(nb_points);
		j = 0;
		for (i = 0; i < Polarity_extension->Polarity->nb_objects[point_idx]; i++) {
			points[j++] = Polarity_extension->perm_offset_i[1] + Polarity_extension->Polarity->offset[point_idx] + i;
		}
		for (i = 0; i < Polarity_extension->Polarity->nb_objects[hyperplane_idx]; i++) {
			points[j++] = Polarity_extension->perm_offset_i[1] + Polarity_extension->Polarity->offset[hyperplane_idx] + i;
		}


	}
	else {
		nb_points = Polarity_extension->Polarity->total_degree;
		points = NEW_lint(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = Polarity_extension->perm_offset_i[1] + i;
		}

	}







	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_polarity_extension");
	label_of_set_tex.assign("{\\rm \\_polext}");

	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"before A_new->Induced_action->restricted_action" << endl;
	}
	A_new_r = A_new->Induced_action->restricted_action(
			points, nb_points,
			label_of_set, label_of_set_tex,
			verbose_level);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"after A_new->Induced_action->restricted_action" << endl;
	}
	A_new_r->f_is_linear = false;
	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"A_new_r:" << endl;
		A_new_r->print_info();
	}


	if (f_v) {
		cout << "action_global::init_polarity_extension_group_and_restrict "
				"after A_direct_product->restricted_action" << endl;
	}
	return A_new_r;
}



action *action_global::init_polarity_extension_group(
		actions::action *A,
		geometry::projective_space *P,
		geometry::polarity *Polarity,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension *Polarity_extension;
	action *A_polarity;

	if (f_v) {
		cout << "action_global::init_polarity_extension_group" << endl;
		cout << "action_global::init_polarity_extension_group A=" << A->label << endl;
		cout << "action_global::init_polarity_extension_group P=" << P->label_txt << endl;
		cout << "action_global::init_polarity_extension_group Polarity=" << Polarity->label_txt
				<< ", degree sequence = " << Polarity->degree_sequence_txt << endl;
	}


	//algebra::matrix_group *M;


	if (!A->is_matrix_group()) {
		cout << "action_global::init_polarity_extension_group "
				"the given group is not a matrix group" << endl;
		exit(1);
	}
	//M = A->get_matrix_group();


	A_polarity = NEW_OBJECT(action);
	Polarity_extension = NEW_OBJECT(group_constructions::polarity_extension);



	A_polarity->type_G = polarity_extension_t;
	A_polarity->G.Polarity_extension = Polarity_extension;
	A_polarity->f_allocated = true;

	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"before Polarity_extension->init" << endl;
	}
	Polarity_extension->init(A, P, Polarity, verbose_level);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"after Polarity_extension->init" << endl;
	}

	A_polarity->f_is_linear = false;
	A_polarity->dimension = 0;


	A_polarity->low_level_point_size = 0;
	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"low_level_point_size="
			<< A_polarity->low_level_point_size<< endl;
	}

	A_polarity->label.assign(Polarity_extension->label);
	A_polarity->label_tex.assign(Polarity_extension->label_tex);


	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"label=" << A_polarity->label << endl;
	}

	A_polarity->degree = Polarity_extension->degree_overall;
	A_polarity->make_element_size = Polarity_extension->make_element_size;

	A_polarity->ptr = NEW_OBJECT(action_pointer_table);
	A_polarity->ptr->init_function_pointers_polarity_extension();

	A_polarity->elt_size_in_int = Polarity_extension->elt_size_int;
	A_polarity->coded_elt_size_in_char = Polarity_extension->char_per_elt;
	A_polarity->Group_element->allocate_element_data();




	A_polarity->degree = Polarity_extension->degree_overall;
	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"degree=" << A_polarity->degree << endl;
	}

	A_polarity->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A_polarity->Stabilizer_chain->allocate_base_data(
			A_polarity, Polarity_extension->base_length, verbose_level);

	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"base_len=" << A_polarity->base_len() << endl;
	}


	Lint_vec_copy(Polarity_extension->the_base, A_polarity->get_base(), A_polarity->base_len());
	Int_vec_copy(Polarity_extension->the_transversal_length,
			A_polarity->get_transversal_length(), A_polarity->base_len());

	int *gens_data;
	int gens_size;
	int gens_nb;

	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"before Polarity_extension->make_strong_generators_data" << endl;
	}
	Polarity_extension->make_strong_generators_data(
			gens_data,
			gens_size, gens_nb, verbose_level - 1);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"after Polarity_extension->make_strong_generators_data" << endl;
	}
	A_polarity->Strong_gens = NEW_OBJECT(groups::strong_generators);

	data_structures_groups::vector_ge *nice_gens;

	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"before A->Strong_gens->init_from_data" << endl;
	}
	A_polarity->Strong_gens->init_from_data(
			A_polarity,
			gens_data, gens_nb, gens_size,
			A_polarity->get_transversal_length(),
			nice_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"after A->Strong_gens->init_from_data" << endl;
	}
	FREE_OBJECT(nice_gens);
	A_polarity->f_has_strong_generators = true;
	FREE_int(gens_data);


	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"before compute_sims" << endl;
	}
	compute_sims(
			A_polarity, verbose_level - 1);
	if (f_v) {
		cout << "action_global::init_polarity_extension_group "
				"after compute_sims" << endl;
	}

	if (f_v) {
		cout << "action_global::init_polarity_extension_group, "
				"finished setting up " << A_polarity->label;
		cout << ", a permutation group of degree " << A_polarity->degree << " ";
		cout << "and of order ";
		A_polarity->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
	return A_polarity;
}


void action_global::compute_sims(
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::compute_sims" << endl;
		cout << "action_global::compute_sims verbose_level = " << verbose_level << endl;
	}

	groups::sims *S; // will become part of A


	S = NEW_OBJECT(groups::sims);

	S->init(A, verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_sims "
				"before S->init_generators" << endl;
	}
	S->init_generators(
			*A->Strong_gens->gens, verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_sims "
				"after S->init_generators" << endl;
	}
	if (f_v) {
		cout << "action_global::compute_sims "
				"before S->compute_base_orbits_known_length" << endl;
	}
	S->compute_base_orbits_known_length(
			A->get_transversal_length(), verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_sims "
				"after S->compute_base_orbits_known_length" << endl;
	}


	if (f_v) {
		cout << "action_global::compute_sims "
				"before init_sims_only" << endl;
	}

	A->init_sims_only(
			S, verbose_level - 2);

	if (f_v) {
		cout << "action_global::compute_sims "
				"after init_sims_only" << endl;
	}

	if (f_v) {
		cout << "action_global::compute_sims "
				"before A->compute_strong_generators_from_sims" << endl;
	}
	A->compute_strong_generators_from_sims(
			verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_sims "
				"after A->compute_strong_generators_from_sims" << endl;
	}

	if (f_v) {
		cout << "action_global::compute_sims done" << endl;
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
		false /* f_induce_action */,
		NULL /* sims *old_G */,
		verbose_level);
	if (f_v) {
		cout << "action_global::orbits_on_equations "
				"The induced action on the equations has been created, "
				"degree = " << A_on_equations->degree << endl;
	}

	if (f_v) {
		cout << "action_global::orbits_on_equations "
				"before compute_all_point_orbits_schreier" << endl;
	}
	Orb = gens->compute_all_point_orbits_schreier(
			A_on_equations,
			verbose_level - 2);
	if (f_v) {
		cout << "action_global::orbits_on_equations "
				"after compute_all_point_orbits_schreier" << endl;
	}

	if (false) {
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
	long int *set, int set_size,
	int f_save_nauty_input_graphs,
	int verbose_level)
// used only by hermitian_spreads_classify
// assuming we are in a linear action.
// added 2/28/2011, called from analyze.cpp
// November 17, 2014 moved here from TOP_LEVEL/extra.cpp
// December 31, 2014, moved here from projective_space.cpp
{
	int f_v = (verbose_level >= 1);
	interfaces::nauty_interface_with_group Nau;

	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
	}


	groups::strong_generators *Set_stab;

	data_structures::bitvector *Canonical_form;
	l1_interfaces::nauty_output *NO;


	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space "
				"before Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
	}
	Nau.set_stabilizer_in_projective_space_using_nauty(
			P,
			A_linear,
			set, set_size,
			f_save_nauty_input_graphs,
			Set_stab,
			Canonical_form,
			NO,
			verbose_level - 1);
	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space "
				"after Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
	}

	FREE_OBJECT(Canonical_form);
	FREE_OBJECT(NO);


	if (f_v) {
		cout << "action_global::set_stabilizer_in_projective_space done" << endl;
	}
	return Set_stab;
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
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative" << endl;
	}
	K.DH_stab_gens(
			k, n, no, data, nb_gens, data_size, stab_order);

	gens = NEW_OBJECT(data_structures_groups::vector_ge);


	if (f_vv) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::stabilizer_of_dual_hyperoval_representative "
				"after gens->init_from_data" << endl;
	}


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
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative"
				<< endl;
	}
	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative "
				"before K.Spread_stab_gens" << endl;
	}
	K.Spread_stab_gens(
			q, k, no, data, nb_gens, data_size, stab_order);
	if (f_v) {
		cout << "action_global::stabilizer_of_spread_representative "
				"after K.Spread_stab_gens" << endl;
	}

	gens = NEW_OBJECT(data_structures_groups::vector_ge);


	if (f_vv) {
		cout << "action_global::stabilizer_of_spread_representative "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A, data,
			nb_gens, data_size, 0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::stabilizer_of_spread_representative "
				"after gens->init_from_data" << endl;
	}

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
	K.quartic_curves_stab_gens(
			q, no, data, nb_gens, data_size, stab_order);
	if (f_v) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"after K.quartic_curves_stab_gens" << endl;
	}

	gens = NEW_OBJECT(data_structures_groups::vector_ge);



	if (f_vv) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "action_global::stabilizer_of_quartic_curve_representative "
				"after gens->init_from_data" << endl;
	}

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
		A->Group_element->compute_permutation(
				Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, A->degree);
			cout << endl;
		}

		if (f_v) {
			cout << "Elt2 = " << endl;
			A->Group_element->element_print_quick(Elt2, cout);
		}
		A->Group_element->compute_permutation(
				Elt2, perm2, 0 /* verbose_level */);
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
		A->Group_element->compute_permutation(
				Elt3, perm3, 0 /* verbose_level */);
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
		A->Group_element->compute_permutation(
				Elt1, perm1, 0 /* verbose_level */);
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
		A->Group_element->compute_permutation(
				Elt2, perm2, 0 /* verbose_level */);
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
		A->Group_element->compute_permutation(
				Elt3, perm3, 0 /* verbose_level */);
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
		A->Group_element->compute_permutation(
				Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, A->degree);
			cout << endl;
		}

		if (f_v) {
			cout << "Elt2 = " << endl;
			A->Group_element->element_print_quick(Elt2, cout);
		}
		A->Group_element->compute_permutation(
				Elt2, perm2, 0 /* verbose_level */);
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


		A->Group_element->compute_permutation(
				Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as Elt3 as permutation: " << endl;
			Combi.perm_print(cout, perm3, A->degree);
			cout << endl;
		}

		A->Group_element->compute_permutation(
				Elt4, perm4, 0 /* verbose_level */);
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
	A->Group_element->compute_permutation(
			Elt1, perm1, 0 /* verbose_level */);
	if (f_v) {
		cout << "as Elt1 as permutation: " << endl;
		Combi.perm_print(cout, perm1, A->degree);
		cout << endl;
	}

	A->Group_element->element_invert(Elt1, Elt2, 0);
	A->Group_element->compute_permutation(
			Elt2, perm2, 0 /* verbose_level */);
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
		cout << endl;
	}

	w = NEW_lint(sz);

	A->Group_element->make_element_from_string(
			Elt2, input_group_element, verbose_level);
	if (f_v) {
		cout << "B=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
	}

	for (i = 0; i < sz; i++) {
		w[i] = A->Group_element->element_image_of(
				v[i], Elt2, verbose_level - 1);
		if (f_v) {
			cout << "mapping " << v[i] << " -> " << w[i] << endl;
		}
	}




	{



		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = A->label + "_apply.tex";

		title = "Application of Group Element in $" + A->label_tex + " $";



		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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
		cout << "action_global::multiply_based_on_text multiplying" << endl;
		cout << "action_global::multiply_based_on_text A=" << data_A << endl;
		cout << "action_global::multiply_based_on_text B=" << data_B << endl;
	}
	int *Elt1;
	int *Elt2;
	int *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = true;
	int f_print_cycles_of_length_one = true;



	A->Group_element->make_element_from_string(
			Elt1, data_A, verbose_level);

	if (f_v) {
		cout << "action_global::multiply_based_on_text A=" << endl;
		A->Group_element->element_print_quick(
				Elt1, cout);

		A->Group_element->element_print_for_make_element(Elt1, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
				Elt1, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}

	A->Group_element->make_element_from_string(
			Elt2, data_B, verbose_level);

	if (f_v) {
		cout << "action_global::multiply_based_on_text B=" << endl;
		A->Group_element->element_print_quick(
				Elt2, cout);

		A->Group_element->element_print_for_make_element(Elt2, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
				Elt2, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}

	if (f_v) {
		cout << "action_global::multiply_based_on_text "
				"before A->Group_element->element_mult" << endl;
	}
	A->Group_element->element_mult(
			Elt1, Elt2, Elt3, verbose_level - 1);
	if (f_v) {
		cout << "action_global::multiply_based_on_text "
				"after A->Group_element->element_mult" << endl;
	}


	if (f_v) {
		cout << "action_global::multiply_based_on_text A*B=" << endl;
		A->Group_element->element_print_quick(Elt3, cout);

		A->Group_element->element_print_for_make_element(Elt3, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
				Elt3, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = A->label + "_mult.tex";

		title = "Multiplication of Group Elements in $" + A->label_tex + "$";

		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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

	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = true;
	int f_print_cycles_of_length_one = true;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(
			Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
		A->Group_element->element_print_as_permutation_with_offset(
				Elt1, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}

	A->Group_element->element_invert(
			Elt1, Elt2, 0);

	if (f_v) {
		cout << "A^-1=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
		A->Group_element->element_print_for_make_element(Elt2, cout);
		cout << endl;
		A->Group_element->element_print_as_permutation_with_offset(
				Elt2, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}



	{


		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = A->label + "_inv.tex";

		title = "Inverse of Group Element in $" + A->label_tex + "$";


		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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

	A->Group_element->make_element_from_string(
			Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}



	{



		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = A->label + "_all_powers.tex";

		title = "Consecutive Powers of Group Element in $" + A->label_tex + "$";



		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;
			int i;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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

			ost << "i & \\left({" << endl;
			A->Group_element->element_print_latex(Elt1, ost);
			ost << "}\\right)^i\\\\" << endl;
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

	A->Group_element->make_element_from_string(
			Elt1, data_A, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->move(Elt1, Elt2);


	A->Group_element->element_power_int_in_place(Elt2,
			exponent, 0 /* verbose_level*/);

	if (f_v) {
		cout << "A^" << exponent << "=" << endl;
		A->Group_element->element_print_quick(
				Elt2, cout);
		A->Group_element->element_print_for_make_element(
				Elt2, cout);
		cout << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = A->label + "_power.tex";

		title = "Power of Group Element in $" + A->label_tex + "$";


		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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
		A->Group_element->element_retrieve(
				gen_handle[i], gens.ith(i), 0);
	}
	compute_orbit_of_point(
			A, gens, pt, orbit, len, verbose_level);
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
	image = Int_vec_minimum(Schreier.orbit, len);
	pos = Schreier.orbit_inv[image];
	Schreier.coset_rep(pos, 0 /* verbose_level */);
	A->Group_element->element_move(
			Schreier.cosetrep, transporter, 0);
	// we check it:
	i = A->Group_element->element_image_of(
			pt, transporter, 0);
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
		A->Group_element->element_retrieve(
				gen_handle[i], gens.ith(i), 0);
	}
	ret = least_image_of_point(
			A, gens, pt, transporter, verbose_level);
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
		A->Group_element->element_retrieve(
				gen_handle[i], gens.ith(i), 0);
	}
	ret = least_image_of_point(
			A, gens, pt, transporter, verbose_level);
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

void action_global::get_orbits_on_points_as_characteristic_vector(
		actions::action *A,
		int *&orbit_no,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::get_orbits_on_points_as_characteristic_vector" << endl;
	}

	groups::schreier Schreier;

	if (f_v) {
		cout << "action_global::get_orbits_on_points_as_characteristic_vector "
				"before all_point_orbits" << endl;
	}
	all_point_orbits(
			A,
			Schreier, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "action_global::get_orbits_on_points_as_characteristic_vector "
				"after all_point_orbits" << endl;
	}

	orbit_no = NEW_int(A->degree);

	int pt;

	for (pt = 0; pt < A->degree; pt++) {
		orbit_no[pt] = Schreier.orbit_number(pt);
	}

	if (f_v) {
		cout << "action_global::get_orbits_on_points_as_characteristic_vector done" << endl;
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
	Schreier.init_generators(
			gens, verbose_level - 2);
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
				"new_action->Stabilizer_chain->A = "
				<< new_action->Stabilizer_chain->get_A()->label << endl;

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
			new_action->Stabilizer_chain->reallocate_base(
					b, verbose_level);
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
		false /*f_override_chose_next_base_point*/,
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
	new_action->f_has_kernel = true;
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
	//int f_get_automorphism_group = true;
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
	A->Group_element->element_one(Elt1, false);

	int c;

	while (true) {
		cnt++;
		//if (cnt == 4) verbose_level += 10;
		if (f_v) {
			cout << "action_global::make_canonical iteration "
						<< cnt << " before is_minimal_witness" << endl;
		}
		c = A->is_minimal_witness(
				/*default_action,*/ size, set1, Sims,
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
	A->Group_element->element_move(Elt1, transporter, false);

	if (!A->Group_element->check_if_transporter_for_set(
			transporter,
			size, set, canonical_set,
			verbose_level - 3)) {
		cout << "action_global::make_canonical "
				"check_if_transporter_for_set returns false" << endl;
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
		long int line_rk, int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::make_element_which_moves_a_line_in_PG3q" << endl;
	}

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
	A->Group_element->make_element(
			Elt, Mtx17, 0);

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




void action_global::init_base(
		actions::action *A,
		algebra::matrix_group *M,
		int verbose_level)
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
		actions::action *A,
		algebra::matrix_group *M,
		int verbose_level)
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
	if (f_vv) {
		cout << "action_global::init_base_projective "
				"before A->Stabilizer_chain->allocate_base_data" << endl;
	}
	A->Stabilizer_chain->allocate_base_data(
			A, base_len, verbose_level);
	if (f_vv) {
		cout << "action_global::init_base_projective "
				"after A->Stabilizer_chain->allocate_base_data" << endl;
	}
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
		actions::action *A,
		algebra::matrix_group *M,
		int verbose_level)
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
		actions::action *A,
		algebra::matrix_group *M,
		int verbose_level)
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


	A->Group_element->element_invert(
			Elt, Elt1, 0);


	if (f_semilinear) {
		HPD->substitute_semilinear(
				input, output,
				f_semilinear, Elt[n * n], Elt1,
				0 /* verbose_level */);
	}
	else {
		HPD->substitute_linear(
				input, output, Elt1,
				0 /* verbose_level */);
	}


	FREE_int(Elt1);

	if (f_v) {
		cout << "action_global::substitute_semilinear done" << endl;
	}
}


void action_global::test_if_two_actions_agree_vector(
		action *A1, action *A2,
		data_structures_groups::vector_ge *gens1,
		data_structures_groups::vector_ge *gens2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::test_if_two_actions_agree_vector" << endl;
	}

	if (gens1->len != gens2->len) {
		cout << "action_global::test_if_two_actions_agree_vector "
				"vectors of different length" << endl;
		exit(1);
	}


	int h;

	for (h = 0; h < gens1->len; h++) {
		if (f_v) {
			cout << "generator " << h << " / " << gens1->len << " : " << endl;
		}
		//A_linear->element_print(gens1->ith(g), cout);

		test_if_two_actions_agree(
				A1, A2, gens1->ith(h), gens2->ith(h), verbose_level);

	}
	if (f_v) {
		cout << "action_global::test_if_two_actions_agree_vector done" << endl;
	}
}

void action_global::test_if_two_actions_agree(
		action *A1, action *A2, int *Elt1, int *Elt2,
		int verbose_level)
// The degree of A2 can be larger than the degree of A1.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::test_if_two_actions_agree" << endl;
	}

#if 0
	if (A1->degree != A2->degree) {
		cout << "action_global::test_if_two_actions_agree the degrees differ" << endl;
		exit(1);
	}
#endif

	int i, j1, j2;

	for (i = 0; i < A1->degree; i++) {
		j1 = A1->Group_element->element_image_of(i, Elt1, 0);
		j2 = A2->Group_element->element_image_of(i, Elt2, 0);
		if (j1 != j2) {
			cout << "action_global::test_if_two_actions_agree "
					"j1 != j2" << endl;
			cout << "i=" << i << endl;
			cout << "j1=" << j1 << endl;
			cout << "j2=" << j2 << endl;
			cout << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "action_global::test_if_two_actions_agree return true" << endl;
	}
}

void action_global::reverse_engineer_semilinear_group(
		action *A_perm, action *A_linear,
		geometry::projective_space *P,
		data_structures_groups::vector_ge *gens_in,
		data_structures_groups::vector_ge *&gens_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action_global::reverse_engineer_semilinear_group" << endl;
	}


	linear_algebra::linear_algebra_global LA;

	//action *A_perm;

	int d;

	d = A_linear->matrix_group_dimension();


	//action *A_linear;

	//A_linear = A;

	if (A_linear == NULL) {
		cout << "action_global::reverse_engineer_semilinear_group "
				"A_linear == NULL" << endl;
		exit(1);
	}

	//data_structures_groups::vector_ge *gens; // permutations from nauty
	//data_structures_groups::vector_ge *gens1; // matrices
	int g, frobenius, pos;
	int *Mtx;
	int *Elt1;
	int c;

	//gens = A_perm->Strong_gens->gens;

	gens_out = NEW_OBJECT(data_structures_groups::vector_ge);
	gens_out->init(A_linear, verbose_level - 2);
	gens_out->allocate(gens_in->len, verbose_level - 2);
	Elt1 = NEW_int(A_linear->elt_size_in_int);

	Mtx = NEW_int(d * d + 1); // leave space for frobenius

	pos = 0;
	for (g = 0; g < gens_in->len; g++) {
		if (f_vv) {
			cout << "action_global::reverse_engineer_semilinear_group "
					"strong generator " << g << ":" << endl;
			//A_perm->element_print(gens->ith(g), cout);
			cout << endl;
		}

		c = LA.reverse_engineer_semilinear_map(
				P->Subspaces->F,
				P->Subspaces->n,
				gens_in->ith(g), Mtx, frobenius,
				0 /*verbose_level - 2*/);

		if (c) {

			Mtx[d * d] = frobenius;
			A_linear->Group_element->make_element(
					Elt1, Mtx, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "action_global::reverse_engineer_semilinear_group "
						"semi-linear group element:" << endl;
				A_linear->Group_element->element_print(Elt1, cout);
			}
			A_linear->Group_element->element_move(
					Elt1, gens_out->ith(pos), 0);


			pos++;
		}
		else {
			//if (f_vv) {
				cout << "action_global::reverse_engineer_semilinear_group "
						"generator " << g << " does not "
						"correspond to a semilinear mapping" << endl;
				exit(1);
			//}
		}
	}
	gens_out->reallocate(pos, verbose_level - 2);
	if (f_v) {
		cout << "action_global::reverse_engineer_semilinear_group "
				"we found " << gens_out->len << " generators" << endl;
	}

	FREE_int(Mtx);
	FREE_int(Elt1);

	if (f_v) {
		cout << "action_global::reverse_engineer_semilinear_group done" << endl;
	}
}


groups::strong_generators *action_global::scan_generators(
		action *A0,
		std::string &gens_text,
		std::string &group_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::strong_generators *Strong_gens;

	if (f_v) {
		cout << "action_global::scan_generators" << endl;
	}
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	int *data;
	int sz;
	int nb_gens;
	data_structures_groups::vector_ge *nice_gens;

	Int_vec_scan(gens_text, data, sz);

	nb_gens = sz / A0->make_element_size;

	if (f_v) {
		cout << "action_global::scan_generators "
				"before Strong_gens->init_from_data_with_target_go_ascii" << endl;
	}

	if (f_v) {
		cout << "action_global::scan_generators "
				"nb_gens=" << nb_gens << endl;
	}

	Strong_gens->init_from_data_with_target_go_ascii(
			A0,
			data,
			nb_gens, A0->make_element_size,
			group_order,
			nice_gens,
			verbose_level + 2);

	FREE_OBJECT(nice_gens);
	FREE_int(data);

	if (f_v) {
		cout << "action_global::scan_generators "
				"after Strong_gens->init_from_data_with_target_go_ascii" << endl;
	}
	if (f_v) {
		cout << "action_global::scan_generators done" << endl;
	}
	return Strong_gens;
}

void action_global::multiply_all_elements_in_lex_order(
		groups::sims *Sims, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::multiply_all_elements_in_lex_order" << endl;
	}

	long int go, go100;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	action *A;

	A = Sims->A;

	go = Sims->group_order_lint();
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	int rk;

	A->Group_element->element_one(Elt1, 0 /*verbose_level*/);

	cout << "action_global::multiply_all_elements_in_lex_order "
			"go=" << go << endl;

	go100 = go / 100;
	if (go100 == 0) {
		go100++;
	}
	for (rk = 0; rk < go; rk++) {

		if ((rk % go100) == 0) {
			cout << "action_global::multiply_all_elements_in_lex_order " << rk / go100 << "%" << endl;
		}
		Sims->element_unrank_lint(
				rk, Elt2, 0 /*verbose_level*/);

		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0 /*verbose_level*/);

		A->Group_element->element_move(Elt3, Elt1, 0 /*verbose_level*/);

	}

	A->Group_element->element_move(Elt1, Elt, 0 /*verbose_level*/);


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "action_global::multiply_all_elements_in_lex_order done" << endl;
	}
}

void action_global::get_generators_from_ascii_coding(
		action *A,
		std::string &ascii_coding,
		data_structures_groups::vector_ge *&gens, int *&tl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go;
	data_structures_groups::group_container *G;

	if (f_v) {
		cout << "action_global::get_generators_from_ascii_coding" << endl;
	}
	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(A, verbose_level - 2);
	if (f_vv) {
		cout << "action_global::get_generators_from_ascii_coding "
				"before G->init_ascii_coding_to_sims" << endl;
	}
	G->init_ascii_coding_to_sims(ascii_coding, verbose_level - 2);
	if (f_vv) {
		cout << "action_global::get_generators_from_ascii_coding "
				"after G->init_ascii_coding_to_sims" << endl;
	}


	G->S->group_order(go);

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	tl = NEW_int(A->base_len());
	G->S->extract_strong_generators_in_order(*gens, tl,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "action_global::get_generators_from_ascii_coding "
				"Group order = " << go << endl;
	}

	FREE_OBJECT(G);
	if (f_v) {
		cout << "action_global::get_generators_from_ascii_coding done" << endl;
	}
}


void action_global::lexorder_test(
		action *A,
		long int *set, int set_sz,
	int &set_sz_after_test,
	data_structures_groups::vector_ge *gens, int max_starter,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_v5 = false; //(verbose_level  >= 1);
	groups::schreier *Sch;
	int i, orb, first, a, a0;

	if (f_v) {
		cout << "action_global::lexorder_test" << endl;
	}

	Sch = NEW_OBJECT(groups::schreier);

	if (f_v) {
		cout << "action_global::lexorder_test computing orbits in action "
				"of degree " << A->degree << ", max_starter="
				<< max_starter << endl;
	}
	Sch->init(A, verbose_level - 2);
	Sch->init_generators(*gens, verbose_level - 2);

	//Sch->compute_all_point_orbits(0);
	if (f_v) {
		cout << "action_global::lexorder_test "
				"before compute_all_orbits_on_invariant_subset" << endl;
	}
	Sch->compute_all_orbits_on_invariant_subset(set_sz,
		set, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_global::lexorder_test "
				"after compute_all_orbits_on_invariant_subset" << endl;
	}

	if (f_v) {
		cout << "action_global::lexorder_test: there are "
				<< Sch->nb_orbits << " orbits on set" << endl;
		Sch->print_orbit_length_distribution(cout);
	}
	if (f_v5) {
		Sch->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "action_global::lexorder_test "
				"max_starter=" << max_starter << endl;
	}
	set_sz_after_test = 0;
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		if (false) {
			cout << "action_global::lexorder_test "
					"Looking at point " << a << endl;
		}
		orb = Sch->orbit_number(a);
		first = Sch->orbit_first[orb];
		a0 = Sch->orbit[first];
		if (a0 < max_starter) {
			if (f_v) {
				cout << "action_global::lexorder_test  Point " << a
						<< " maps to " << a0 << " which is less than "
						"max_starter = " << max_starter
						<< " so we eliminate" << endl;
			}
		}
		else {
			set[set_sz_after_test++] = a;
		}
	}
	if (f_v) {
		cout << "action_global::lexorder_test Of the " << set_sz
				<< " points, we accept " << set_sz_after_test
				<< " and we reject " << set_sz - set_sz_after_test << endl;
	}
	FREE_OBJECT(Sch);
	if (f_v) {
		cout << "action_global::lexorder_test done" << endl;
	}

}

void action_global::compute_orbits_on_points(
		action *A,
		groups::schreier *&Sch,
		data_structures_groups::vector_ge *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::compute_orbits_on_points" << endl;
	}
	Sch = NEW_OBJECT(groups::schreier);
	if (f_v) {
		cout << "action_global::compute_orbits_on_points in action ";
		A->print_info();
	}
	if (f_v) {
		cout << "action_global::compute_orbits_on_points "
				"before Sch->init" << endl;
	}
	Sch->init(A, verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_orbits_on_points "
				"before Sch->init_generators" << endl;
	}
	Sch->init_generators(*gens, verbose_level - 2);
	if (f_v) {
		cout << "action_global::compute_orbits_on_points "
				"before Sch->compute_all_point_orbits, "
				"degree = " << A->degree << endl;
	}
	Sch->compute_all_point_orbits(verbose_level - 3);
	if (f_v) {
		cout << "action_global::compute_orbits_on_points "
				"after Sch->compute_all_point_orbits" << endl;
		cout << "action_global::compute_orbits_on_points "
				"Sch->nb_orbits=" << Sch->nb_orbits << endl;
	}
	//Sch.print_and_list_orbits(cout);
	if (f_v) {
		cout << "action_global::compute_orbits_on_points done, we found "
				<< Sch->nb_orbits << " orbits" << endl;
	}
}

void action_global::point_stabilizer_any_point(
		action *A,
		int &pt,
		groups::schreier *&Sch, groups::sims *&Stab,
		groups::strong_generators *&stab_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point" << endl;
	}

	int f; //, len;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point "
				"before compute_all_point_orbits_schreier" << endl;
	}
	Sch = A->Strong_gens->compute_all_point_orbits_schreier(
			A, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_global::point_stabilizer_any_point "
				"after compute_all_point_orbits_schreier" << endl;
	}
	//compute_all_point_orbits(Sch,
	//*Strong_gens->gens, 0 /* verbose_level */);
	if (f_v) {
		cout << "computing all point orbits done, found "
				<< Sch->nb_orbits << " orbits" << endl;
	}


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point "
				"orbit rep = "
				<< pt << endl;
	}

	A->group_order(go);
	if (f_v) {
		cout << "action_global::point_stabilizer_any_point "
				"Computing point stabilizer:" << endl;
	}
	Sch->point_stabilizer(A, go,
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);

	Stab->group_order(go);

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point "
				"Computing point stabilizer done:" << endl;
		cout << "action_global::point_stabilizer_any_point "
				"point stabilizer is a group of order " << go << endl;
	}

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point computing "
				"strong generators for the point stabilizer:" << endl;
	}
	stab_gens = NEW_OBJECT(groups::strong_generators);
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_global::point_stabilizer_any_point strong generators "
				"for the point stabilizer have been computed" << endl;
	}

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point done" << endl;
	}
}

void action_global::point_stabilizer_any_point_with_given_group(
		action *A,
		groups::strong_generators *input_gens,
	int &pt,
	groups::schreier *&Sch, groups::sims *&Stab,
	groups::strong_generators *&stab_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group" << endl;
	}

	int f; //, len;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"before compute_all_point_orbits_schreier" << endl;
	}
	Sch = input_gens->compute_all_point_orbits_schreier(A, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"after compute_all_point_orbits_schreier" << endl;
	}
	//compute_all_point_orbits(Sch, *Strong_gens->gens, 0 /* verbose_level */);
	cout << "computing all point orbits done, found "
			<< Sch->nb_orbits << " orbits" << endl;


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"orbit rep = " << pt << endl;
	}

	input_gens->group_order(go);
	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"Computing point stabilizer:" << endl;
	}
	Sch->point_stabilizer(A, go,
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);

	Stab->group_order(go);

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"Computing point stabilizer done:" << endl;
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"point stabilizer is a group of order " << go << endl;
	}

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"computing strong generators for the point stabilizer:"
				<< endl;
	}
	stab_gens = NEW_OBJECT(groups::strong_generators);
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group "
				"strong generators for the point stabilizer "
				"have been computed" << endl;
	}

	if (f_v) {
		cout << "action_global::point_stabilizer_any_point_with_given_group done"
				<< endl;
	}
}


void action_global::move_a_to_b_and_stabilizer_of_b(
		actions::action *A_base,
		actions::action *A2,
		groups::strong_generators *SG,
		int a, int b,
		int *&transporter_a_b,
		groups::strong_generators *&Stab_b,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b" << endl;
	}
	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b A2 = " << endl;
		A2->print_info();
	}

	ring_theory::longinteger_object full_group_order;

	SG->group_order(full_group_order);
	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"group order = " << full_group_order << endl;
	}


	groups::schreier *Schreier;

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"before compute_all_point_orbits_schreier" << endl;
	}

	Schreier = SG->gens->compute_all_point_orbits_schreier(
			A2, verbose_level - 2);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"after compute_all_point_orbits_schreier" << endl;
		cout << "We found " << Schreier->nb_orbits
				<< " orbits of the group" << endl;
	}

	int *Elt1, *Elt2;

	Elt1 = NEW_int(A2->elt_size_in_int);
	Elt2 = NEW_int(A2->elt_size_in_int);
	transporter_a_b = NEW_int(A2->elt_size_in_int);

	int idx1, idx2;

	idx1 = Schreier->orbit_number(a);
	idx2 = Schreier->orbit_number(b);
	if (idx1 != idx2) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"the two points lie in different orbits" << endl;
		exit(1);
	}
	int orbit_idx;
	int c;

	Schreier->transporter_from_point_to_orbit_rep(
			a,
		orbit_idx, Elt1, verbose_level);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"Elt1:" << endl;
		A2->Group_element->element_print_latex(
				Elt1, cout);
	}

	Schreier->transporter_from_orbit_rep_to_point(
			b,
		orbit_idx, Elt2, verbose_level);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"Elt2:" << endl;
		A2->Group_element->element_print_latex(
				Elt2, cout);
	}
	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"before element_mult" << endl;
	}

	A2->Group_element->element_mult(
			Elt1, Elt2, transporter_a_b, verbose_level);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"transporter_a_b:" << endl;
		A2->Group_element->element_print_latex(
				transporter_a_b, cout);
	}

	c = A2->Group_element->element_image_of(
			a, transporter_a_b, 0 /*verbose_level*/);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"transporter_a_b "
				"the element maps a to c where a=" << a << " c=" << c << endl;
	}
	if (c != b) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"c != b, error" << endl;
		exit(1);

	}

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"before Schreier->stabilizer_any_point" << endl;
	}

	Stab_b = Schreier->stabilizer_any_point(
			A_base,
		full_group_order,
		b,
		0 /*verbose_level*/);

	if (f_v) {
		cout << "action_global::move_a_to_b_and_stabilizer_of_b "
				"after Schreier->stabilizer_any_point" << endl;
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_OBJECT(Schreier);


	if (f_v) {
		cout << "action_global::move_a_to_b done" << endl;
	}
}


void action_global::rational_normal_form(
		actions::action *A,
		std::string &element_given,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::rational_normal_form" << endl;
	}
	if (f_v) {
		cout << "action_global::rational_normal_form A = " << endl;
		A->print_info();
	}

	algebra::matrix_group *M;

	M = A->G.matrix_grp;

	int n;

	n = M->n;

	linear_algebra::gl_classes *C;


#if 0
	if (f_v) {
		cout << "action_global::rational_normal_form "
				"before M->init_gl_classes M->n=" << M->n << endl;
	}
	M->init_gl_classes(0 /*verbose_level*/);
	if (f_v) {
		cout << "action_global::rational_normal_form "
				"after M->init_gl_classes" << endl;
	}
	linear_algebra::gl_classes *C;

	C = M->C;
#endif

	C = NEW_OBJECT(linear_algebra::gl_classes);
	if (f_v) {
		cout << "action_global::rational_normal_form "
				"before C->init" << endl;
	}
	C->init(M->n, M->GFq, verbose_level - 3);
	if (f_v) {
		cout << "action_global::rational_normal_form "
				"after C->init" << endl;
	}

	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *B;
	int *Bv;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	B = NEW_int(A->elt_size_in_int);
	Bv = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(
			Elt1, element_given, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action_global::rational_normal_form "
				"element_given=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}




	int *Mtx1;
	int *Basis1;
	linear_algebra::gl_class_rep *R1;

	//Mtx1 = NEW_int(n);
	Mtx1 = Elt1;


	Basis1 = NEW_int(n * n);

	R1 = NEW_OBJECT(linear_algebra::gl_class_rep);

	if (f_v) {
		cout << "action_global::rational_normal_form "
				"before identify_matrix Mtx1" << endl;
	}
	C->identify_matrix(Mtx1, R1, Basis1, verbose_level);
	if (f_v) {
		cout << "action_global::rational_normal_form "
				"after identify_matrix Mtx1" << endl;
	}

	A->Group_element->make_element(
			B, Basis1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "B=" << endl;
		A->Group_element->element_print_quick(B, cout);
	}

	A->Group_element->element_invert(
			B, Bv, verbose_level - 2);
	if (f_v) {
		cout << "Bv=" << endl;
		A->Group_element->element_print_quick(Bv, cout);
	}


	A->Group_element->element_mult(
			Bv, Elt1, Elt2, 0 /*verbose_level*/);
	A->Group_element->element_mult(
			Elt2, B, Elt3, 0 /*verbose_level*/);

	if (f_v) {
		cout << "action_global::rational_normal_form "
				"Bv * Elt1 * B=" << endl;
		A->Group_element->element_print_quick(Elt3, cout);
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(B);
	FREE_int(Bv);
	FREE_int(Basis1);
	FREE_OBJECT(R1);
	FREE_OBJECT(C);

	if (f_v) {
		cout << "action_global::rational_normal_form done" << endl;
	}
}




void action_global::find_conjugating_element(
		actions::action *A,
		std::string &element_from,
		std::string &element_to,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::find_conjugating_element" << endl;
	}
	if (f_v) {
		cout << "action_global::find_conjugating_element A = " << endl;
		A->print_info();
	}

	algebra::matrix_group *M;

	M = A->G.matrix_grp;

	int n;

	n = M->n;

	linear_algebra::gl_classes *C;


#if 0
	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"before M->init_gl_classes M->n=" << M->n << endl;
	}
	M->init_gl_classes(0 /*verbose_level*/);
	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"after M->init_gl_classes" << endl;
	}
	linear_algebra::gl_classes *C;

	C = M->C;
#endif

	C = NEW_OBJECT(linear_algebra::gl_classes);
	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"before C->init" << endl;
	}
	C->init(M->n, M->GFq, verbose_level);
	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"after C->init" << endl;
	}

	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->make_element_from_string(
			Elt1, element_from, verbose_level);
	if (f_v) {
		cout << "Elt_from=" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->make_element_from_string(
			Elt2, element_to, verbose_level);
	if (f_v) {
		cout << "Elt_to=" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
	}



	int *Mtx1;
	int *Mtx2;
	int *Basis1;
	int *Basis2;
	linear_algebra::gl_class_rep *R1;
	linear_algebra::gl_class_rep *R2;

	//Mtx1 = NEW_int(n);
	Mtx1 = Elt1;
	Mtx2 = Elt2;


	Basis1 = NEW_int(n * n);
	Basis2 = NEW_int(n * n);

	R1 = NEW_OBJECT(linear_algebra::gl_class_rep);
	R2 = NEW_OBJECT(linear_algebra::gl_class_rep);

	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"before identify_matrix Mtx1" << endl;
	}
	C->identify_matrix(Mtx1, R1, Basis1, verbose_level);
	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"after identify_matrix Mtx1" << endl;
	}

	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"before identify_matrix Mtx2" << endl;
	}
	C->identify_matrix(Mtx2, R2, Basis2, verbose_level);
	if (f_v) {
		cout << "action_global::find_conjugating_element "
				"after identify_matrix Mtx2" << endl;
	}



	if (f_v) {
		cout << "action_global::find_conjugating_element done" << endl;
	}
}




void action_global::read_orbit_rep_and_candidates_from_files_and_process(
		action *A,
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level),
	void *early_test_func_callback_data,
	long int *&starter,
	int &starter_sz,
	groups::sims *&Stab,
	groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *candidates1;
	int nb_candidates1;
	int h; //, i;

	if (f_v) {
		cout << "action_global::read_orbit_rep_and_candidates_from_files_and_process" << endl;
	}

	if (f_v) {
		cout << "action_global::read_orbit_rep_and_candidates_from_files_and_process "
				"before read_orbit_rep_and_candidates_from_files" << endl;
	}
	read_orbit_rep_and_candidates_from_files(
			A, prefix,
		level, orbit_at_level, level_of_candidates_file,
		starter,
		starter_sz,
		Stab,
		Strong_gens,
		candidates1,
		nb_candidates1,
		nb_cases,
		verbose_level);
	if (f_v) {
		cout << "action_global::read_orbit_rep_and_candidates_from_files_and_process "
				"after read_orbit_rep_and_candidates_from_files" << endl;
	}

	for (h = level_of_candidates_file; h < level; h++) {

		long int *candidates2;
		int nb_candidates2;

		if (f_vv) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files_and_process "
					"testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
			}
		candidates2 = NEW_lint(nb_candidates1);

		(*early_test_func_callback)(starter, h + 1,
			candidates1, nb_candidates1,
			candidates2, nb_candidates2,
			early_test_func_callback_data, verbose_level - 1);

		if (f_vv) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files_and_process "
					"number of candidates at level " << h + 1
					<< " reduced from " << nb_candidates1 << " to "
					<< nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
			}

		Lint_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_lint(candidates2);
		}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "action_global::read_orbit_rep_and_candidates_from_files_and_process "
				"done" << endl;
		}
}

void action_global::read_orbit_rep_and_candidates_from_files(
		action *A,
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	long int *&starter,
	int &starter_sz,
	groups::sims *&Stab,
	groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "action_global::read_orbit_rep_and_candidates_from_files "
				"prefix=" << prefix << endl;
	}

	{
		candidates = NULL;
		//longinteger_object stab_go;

		string fname1;
		fname1 = prefix + "_lvl_" + std::to_string(level);

		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"before read_set_and_stabilizer fname1=" << fname1 << endl;
		}
		read_set_and_stabilizer(
				A,
				fname1,
			orbit_at_level, starter, starter_sz, Stab,
			Strong_gens,
			nb_cases,
			verbose_level);
		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"after read_set_and_stabilizer" << endl;
		}



		//Stab->group_order(stab_go);

		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"Read starter " << orbit_at_level << " / "
					<< nb_cases << " : ";
			Lint_vec_print(cout, starter, starter_sz);
			cout << endl;
			//cout << "read_orbit_rep_and_candidates_from_files "
			//"Group order=" << stab_go << endl;
		}

		if (level == level_of_candidates_file) {
			orbit_at_candidate_level = orbit_at_level;
		}
		else {
			// level_of_candidates_file < level
			// Now, we need to find out the orbit representative
			// at level_of_candidates_file
			// that matches with the prefix of starter
			// so that we can retrieve it's set of candidates.
			// Once we have the candidates for the prefix, we run it through the
			// test function to find the candidate set of starter as a subset
			// of this set.

			orbit_at_candidate_level = Fio.find_orbit_index_in_data_file(
					prefix,
					level_of_candidates_file, starter,
					verbose_level);
		}
		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"Found starter, orbit_at_candidate_level="
					<< orbit_at_candidate_level << endl;
		}


		// read the set of candidates from the binary file:

		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"before generator_read_candidates_of_orbit" << endl;
		}
		string fname2;
		fname2 = prefix + "_lvl_" + std::to_string(level_of_candidates_file) + "_candidates.bin";


		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"before Fio.poset_classification_read_candidates_of_orbit" << endl;
		}
		Fio.poset_classification_read_candidates_of_orbit(
			fname2, orbit_at_candidate_level,
			candidates, nb_candidates, verbose_level - 1);

		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"after Fio.poset_classification_read_candidates_of_orbit" << endl;
		}


		if (candidates == NULL) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"could not read the candidates" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "action_global::read_orbit_rep_and_candidates_from_files "
					"Found " << nb_candidates << " candidates at level "
					<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "action_global::read_orbit_rep_and_candidates_from_files done" << endl;
	}
}


void action_global::read_representatives(
		std::string &fname,
		int *&Reps, int &nb_reps, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_casenumbers = false;
	int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i, j;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action_global::read_representatives "
			"reading file " << fname << endl;
	}

	Fio.read_and_parse_data_file_fancy(
			fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level*/);
	nb_reps = nb_cases;
	size = Set_sizes[0];
	Reps = NEW_int(nb_cases * size);
	for (i = 0; i < nb_cases; i++) {
		for (j = 0; j < size; j++) {
			Reps[i * size + j] = Sets[i][j];
		}
	}
	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);
	if (f_v) {
		cout << "action_global::read_representatives done" << endl;
	}
}

void action_global::read_representatives_and_strong_generators(
	std::string &fname, int *&Reps,
	char **&Aut_ascii, int &nb_reps, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_casenumbers = false;
	int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	//char **Aut_ascii;
	int *Casenumbers;
	int i, j;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "action_global::read_representatives_and_strong_generators "
			"reading file " << fname << endl;
	}

	if (f_v) {
		cout << "action_global::read_representatives_and_strong_generators "
			"before Fio.read_and_parse_data_file_fancy" << endl;
	}
	Fio.read_and_parse_data_file_fancy(
			fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level*/);
	if (f_v) {
		cout << "action_global::read_representatives_and_strong_generators "
			"after Fio.read_and_parse_data_file_fancy" << endl;
	}
	nb_reps = nb_cases;
	if (f_v) {
		cout << "action_global::read_representatives_and_strong_generators "
			"nb_reps = " << nb_reps << endl;
	}
	size = Set_sizes[0];
	Reps = NEW_int(nb_cases * size);
	for (i = 0; i < nb_cases; i++) {
		for (j = 0; j < size; j++) {
			Reps[i * size + j] = Sets[i][j];
		}
	}
	Fio.free_data_fancy(
			nb_cases,
		Set_sizes, Sets,
		Ago_ascii, NULL /*Aut_ascii*/,
		Casenumbers);
	if (f_v) {
		cout << "action_global::read_representatives_and_strong_generators done" << endl;
	}
}

void action_global::read_file_and_print_representatives(
		action *A,
		std::string &fname,
		int f_print_stabilizer_generators, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_casenumbers = false;
	int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action_global::read_file_and_print_representatives "
				"reading file "
				<< fname << endl;
	}

	Fio.read_and_parse_data_file_fancy(
			fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level*/);

	for (i = 0; i < nb_cases; i++) {
		cout << "Orbit " << i << " / " << nb_cases << " representative = ";
		Lint_vec_print(cout, Sets[i], Set_sizes[i]);
		cout << endl;

		data_structures_groups::group_container *G;
		data_structures_groups::vector_ge *gens;
		int *tl;

		G = NEW_OBJECT(data_structures_groups::group_container);
		G->init(A, verbose_level - 2);

		string s;

		s.assign(Aut_ascii[i]);
		G->init_ascii_coding_to_sims(s, verbose_level - 2);


		ring_theory::longinteger_object go;

		G->S->group_order(go);

		gens = NEW_OBJECT(data_structures_groups::vector_ge);
		tl = NEW_int(A->base_len());
		G->S->extract_strong_generators_in_order(
				*gens, tl,
				0 /* verbose_level */);
		cout << "Stabilizer has order " << go << " tl=";
		Int_vec_print(cout, tl, A->base_len());
		cout << endl;

		if (f_print_stabilizer_generators) {
			cout << "The stabilizer is generated by:" << endl;
			gens->print(cout);
		}

		FREE_OBJECT(G);
		FREE_OBJECT(gens);
		FREE_int(tl);

	}
	Fio.free_data_fancy(
			nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);

}

void action_global::read_set_and_stabilizer(
		action *A,
		std::string &fname,
	int no, long int *&set, int &set_sz, groups::sims *&stab,
	groups::strong_generators *&Strong_gens,
	int &nb_cases,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_vv = (verbose_level  >= 2);
	int f_casenumbers = false;
	//int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	data_structures_groups::group_container *G;
	int i;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "action_global::read_set_and_stabilizer "
				"reading file " << fname
				<< " no=" << no << endl;
	}

	Fio.read_and_parse_data_file_fancy(
			fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 1);

	if (f_vv) {
		cout << "action_global::read_set_and_stabilizer "
				"after read_and_parse_data_file_fancy" << endl;
		cout << "Aut_ascii[no]=" << Aut_ascii[no] << endl;
		cout << "Set_sizes[no]=" << Set_sizes[no] << endl;
	}

	set_sz = Set_sizes[no];
	set = NEW_lint(set_sz);
	for (i = 0; i < set_sz; i ++) {
		set[i] = Sets[no][i];
	}


	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(A, verbose_level - 2);
	if (f_vv) {
		cout << "action_global::read_set_and_stabilizer "
				"before G->init_ascii_coding_to_sims" << endl;
	}

	string s;

	s.assign(Aut_ascii[no]);
	G->init_ascii_coding_to_sims(s, verbose_level - 2);
	if (f_vv) {
		cout << "action_global::read_set_and_stabilizer "
				"after G->init_ascii_coding_to_sims" << endl;
	}

	stab = G->S;
	G->S = NULL;
	G->f_has_sims = false;

	ring_theory::longinteger_object go;

	stab->group_order(go);


	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(stab, 0);
	A->f_has_strong_generators = true;

	if (f_vv) {
		cout << "action_global::read_set_and_stabilizer "
				"Group order=" << go << endl;
	}

	FREE_OBJECT(G);
	if (f_vv) {
		cout << "action_global::read_set_and_stabilizer "
				"after FREE_OBJECT  G" << endl;
	}
	Fio.free_data_fancy(
			nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);
	if (f_v) {
		cout << "action_global::read_set_and_stabilizer done" << endl;
	}

}

data_structures::set_of_sets *action_global::set_of_sets_copy_and_apply(
		action *A,
		int *Elt,
		data_structures::set_of_sets *old_one,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);

	if (f_v) {
		cout << "action_global::set_of_sets_copy_and_apply" << endl;
	}

	data_structures::set_of_sets *SoS;

	SoS = old_one->copy();

	int i, j;
	long int a, b;

	for (i = 0; i < SoS->nb_sets; i++) {
		for (j = 0; j < SoS->Set_size[i]; j++) {
			a = SoS->Sets[i][j];
			b = A->Group_element->image_of(Elt, a);
			SoS->Sets[i][j] = b;
		}
	}



	if (f_v) {
		cout << "action_global::set_of_sets_copy_and_apply done" << endl;
	}

	return SoS;

}

actions::action *action_global::create_action_on_k_subspaces(
		actions::action *A_previous,
		int k,
		int verbose_level)
{
	int f_v = (verbose_level  >= 1);

	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces" << endl;
	}

	if (!A_previous->f_is_linear) {
		cout << "action_global::create_action_on_k_subspaces "
				"previous action is not linear" << endl;
		exit(1);
	}

	action *A_modified;
	algebra::matrix_group *M;
	field_theory::finite_field *Fq;
	int n;

	M = A_previous->get_matrix_group();
	n = M->n;
	Fq = M->GFq;

	induced_actions::action_on_grassmannian *AonG;
	geometry::grassmann *Grass;

	AonG = NEW_OBJECT(induced_actions::action_on_grassmannian);

	Grass = NEW_OBJECT(geometry::grassmann);


	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces "
				"before Grass->init" << endl;
	}

	Grass->init(n, k, Fq, 0 /* verbose_level */);

	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces "
				"after Grass->init" << endl;
	}


	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces "
				"before AonG->init" << endl;
	}

	AonG->init(*A_previous, Grass, verbose_level - 2);

	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces "
				"after AonG->init" << endl;
	}



	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces "
				"before induced_action_on_grassmannian_preloaded" << endl;
	}

	A_modified = A_previous->Induced_action->induced_action_on_grassmannian_preloaded(AonG,
		false /* f_induce_action */, NULL /*sims *old_G */,
		verbose_level - 2);

	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces "
				"after induced_action_on_grassmannian_preloaded" << endl;
	}

	A_modified->f_is_linear = true;

	A_modified->dimension = A_previous->dimension;


	if (f_v) {
		cout << "action_global::create_action_on_k_subspaces done" << endl;
	}
	return A_modified;
}

void action_global::report_strong_generators(
		std::ostream &ost,
		graphics::layered_graph_draw_options *LG_Draw_options,
		groups::strong_generators *SG,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_global::report_strong_generators" << endl;
	}

	// GAP:

	ost << "GAP export: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"before SG->print_generators_gap" << endl;
	}
	SG->print_generators_gap(ost, verbose_level - 1);
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"after SG->print_generators_gap" << endl;
	}
	ost << "\\end{verbatim}" << endl;


	// Fining:

	ost << "Fining export: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"before SG->export_fining" << endl;
	}
	SG->export_fining(A, ost, verbose_level);
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"after SG->export_fining" << endl;
	}
	ost << "\\end{verbatim}" << endl;


	// Magma:

	ost << "Magma export: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"before SG->export_magma" << endl;
	}
	SG->export_magma(A, ost, verbose_level);
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"after SG->export_magma" << endl;
	}
	ost << "\\end{verbatim}" << endl;


	// Orbiter compact form:

	ost << "Compact form: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"before SG->print_generators_compact" << endl;
	}
	SG->print_generators_compact(ost, verbose_level - 1);
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"after SG->print_generators_compact" << endl;
	}
	ost << "\\end{verbatim}" << endl;



	if (f_v) {
		cout << "action_global::report_strong_generators done" << endl;
	}
}



//#############################################################################

void callback_choose_random_generator_orthogonal(
		int iteration,
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
	int f_siegel = true;
	int f_reflection = true;
	int f_similarity = true;
	int f_semisimilarity = true;
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





}}}

