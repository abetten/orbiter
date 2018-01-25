// interface.C
//
// Anton Betten
//
// started:  November 13, 2007
// last change:  November 9, 2010




#include "galois.h"
#include "action.h"




// ####################################################################################
// interface functions: induced action
// ####################################################################################


INT induced_action_element_image_of(action &A, INT a, void *elt, INT verbose_level)
{
	INT *Elt = (INT *) elt;
	INT b = 0;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "induced_action_element_image_of computing image of " << a << " in action " << A.label << endl;
		}
	if (A.type_G == action_by_right_multiplication_t) {
		if (f_v) {
			cout << "action_by_right_multiplication_t" << endl;
			}
		action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		ABRM->compute_image(sub, Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_by_restriction_t) {
		if (f_v) {
			cout << "action_by_restriction_t" << endl;
			}
		action_by_restriction *ABR = A.G.ABR;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = ABR->compute_image(sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_by_conjugation_t) {
		if (f_v) {
			cout << "action_by_conjugation_t" << endl;
			}
		action_by_conjugation *ABC = A.G.ABC;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = ABC->compute_image(sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_by_representation_t) {
		if (f_v) {
			cout << "action_by_representation_t" << endl;
			}
		action_by_representation *Rep = A.G.Rep;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = Rep->compute_image_INT(*sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_determinant_t) {
		if (f_v) {
			cout << "action_on_determinant_t" << endl;
			}
		action_on_determinant *AD = A.G.AD;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		AD->compute_image(sub, Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_on_sign_t) {
		if (f_v) {
			cout << "action_on_sign_t" << endl;
			}
		action_on_sign *OnSign = A.G.OnSign;

		OnSign->compute_image(Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_on_grassmannian_t) {
		if (f_v) {
			cout << "action_on_grassmannian_t" << endl;
			}
		action_on_grassmannian *AG = A.G.AG;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = AG->compute_image_INT(sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_spread_set_t) {
		if (f_v) {
			cout << "action_on_spread_set_t" << endl;
			}
		action_on_spread_set *AS = A.G.AS;

		b = AS->compute_image_INT(Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_orthogonal_t) {
		if (f_v) {
			cout << "action_on_orthogonal_t" << endl;
			}
		action_on_orthogonal *AO = A.G.AO;

#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
#endif
		b = AO->compute_image_INT(Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_wedge_product_t) {
		if (f_v) {
			cout << "action_on_wedge_product_t" << endl;
			}
		action_on_wedge_product *AW = A.G.AW;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = AW->compute_image_INT(*sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_homogeneous_polynomials_t) {
		if (f_v) {
			cout << "action_on_homogeneous_polynomials_t" << endl;
			}
		action_on_homogeneous_polynomials *OnHP = A.G.OnHP;

		b = OnHP->compute_image_INT(Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_by_subfield_structure_t) {
		if (f_v) {
			cout << "action_by_subfield_structure_t" << endl;
			}
		action_by_subfield_structure *SubfieldStructure = A.G.SubfieldStructure;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = SubfieldStructure->compute_image_INT(*sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_cosets_t) {
		if (f_v) {
			cout << "action_on_cosets_t" << endl;
			}
		action_on_cosets *AC = A.G.OnCosets;

		//cout << "interface.C: action_on_cosets computing image of " << a << endl;
		b = AC->compute_image(Elt, a, verbose_level - 1);
		//cout << "interface.C: action_on_cosets image of " << a << " is " << b << endl;
		}
	else if (A.type_G == action_on_factor_space_t) {
		if (f_v) {
			cout << "action_on_factor_space_t" << endl;
			}
		action_on_factor_space *AF = A.G.AF;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = AF->compute_image(sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_sets_t) {
		if (f_v) {
			cout << "action_on_sets_t" << endl;
			}
		action_on_sets *AOS = A.G.on_sets;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		AOS->compute_image(sub, Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_on_subgroups_t) {
		if (f_v) {
			cout << "action_on_subgroups_t" << endl;
			}
		action_on_subgroups *AOS = A.G.on_subgroups;

		b = AOS->compute_image(Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_k_subsets_t) {
		if (f_v) {
			cout << "action_on_k_subsets_t" << endl;
			}
		action_on_k_subsets *On_k_subsets = A.G.on_k_subsets;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		On_k_subsets->compute_image(Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_on_orbits_t) {
		if (f_v) {
			cout << "action_on_orbits_t" << endl;
			}
		action_on_orbits *On_orbits = A.G.OnOrbits;

		b = On_orbits->compute_image(Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_flags_t) {
		if (f_v) {
			cout << "action_on_flags_t" << endl;
			}
		action_on_flags *On_flags = A.G.OnFlags;

		b = On_flags->compute_image(Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_bricks_t) {
		if (f_v) {
			cout << "action_on_bricks_t" << endl;
			}
		action_on_bricks *On_bricks = A.G.OnBricks;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		On_bricks->compute_image(Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_on_andre_t) {
		if (f_v) {
			cout << "action_on_andre_t" << endl;
			}
		action_on_andre *On_andre = A.G.OnAndre;

#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
#endif

		On_andre->compute_image(Elt, a, b, verbose_level - 1);
		}
	else if (A.type_G == action_on_pairs_t) {
		if (f_v) {
			cout << "action_on_pairs_t" << endl;
			}
		action *sub;
		INT i, j, u, v;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction, type = action_on_pairs_t" << endl;
			exit(1);
			}
		k2ij(a, i, j, sub->degree);
		u = sub->element_image_of(i, elt, verbose_level - 1);
		v = sub->element_image_of(j, elt, verbose_level - 1);
		b = ij2k(u, v, sub->degree);
		}
	else if (A.type_G == action_on_ordered_pairs_t) {
		if (f_v) {
			cout << "action_on_ordered_pairs_t" << endl;
			}
		action *sub;
		INT a2, b2, swap, swap2, i, j, tmp, u, v, u2, v2;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction, type = action_on_ordered_pairs_t" << endl;
			exit(1);
			}
		swap = a % 2;
		a2 = a / 2;
		k2ij(a2, i, j, sub->degree);
		if (swap) {
			tmp = i;
			i = j;
			j = tmp;
			}
		u = sub->element_image_of(i, elt, verbose_level - 1);
		v = sub->element_image_of(j, elt, verbose_level - 1);
		if (u > v) {
			v2 = u;
			u2 = v;
			swap2 = 1;
			}
		else {
			u2 = u;
			v2 = v;
			swap2 = 0;
			}
		b2 = ij2k(u2, v2, sub->degree);
		b = 2 * b2 + swap2;
#if 0
		cout << "induced_action_element_image_of action_on_ordered_pairs_t" << endl;
		cout << a << " -> " << b << endl;
		cout << "(" << i << "," << j << ") -> (" << u << "," << v << ")" << endl;
		cout << "under" << endl;
		sub->element_print(elt, cout);
		cout << endl;
#endif
		}
	else if (A.type_G == base_change_t) {
		if (f_v) {
			cout << "base_change_t" << endl;
			}
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction, type = base_change_t" << endl;
			exit(1);
			}
		b = sub->element_image_of(a, elt, verbose_level - 1);
		}
	else if (A.type_G == product_action_t) {
		if (f_v) {
			cout << "product_action_t" << endl;
			}
		product_action *PA;
		
		PA = A.G.product_action_data;
		b = PA->compute_image(&A, (INT *)elt, a, verbose_level - 1);
		}
	else {
		cout << "induced_action_element_image_of() type_G unknown:: type_G = " << A.type_G << endl;
		action_print_symmetry_group_type(cout, A.type_G);
		cout << "action:" << endl;
		A.print_info();
		exit(1);
		}
	if (f_v) {
		cout << "induced_action_element_image_of()  image of " << a << " is " << b << endl;
		}
	return b;
}

void induced_action_element_image_of_low_level(action &A, INT *input, INT *output, void *elt, INT verbose_level)
{
	INT *Elt = (INT *) elt;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "induced_action_element_image_of_low_level() computing image of ";
		INT_vec_print(cout, input, A.low_level_point_size);
		cout << " in action " << A.label << endl;
		}
	if (A.type_G == action_by_right_multiplication_t) {
		if (f_v) {
			cout << "action_by_right_multiplication_t" << endl;
			}

		cout << "induced_action_element_image_of_low_level() action_by_right_multiplication_t not yet implemented" << endl;
		exit(1);
#if 0
		action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		ABRM->compute_image(sub, Elt, a, b, verbose_level - 1);
#endif
		}
	else if (A.type_G == action_by_restriction_t) {
		if (f_v) {
			cout << "action_by_restriction_t" << endl;
			}

		cout << "induced_action_element_image_of_low_level() action_by_restriction_t not yet implemented" << endl;
		exit(1);
#if 0
		action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		ABRM->compute_image(sub, Elt, a, b, verbose_level - 1);
#endif
		}
	else if (A.type_G == action_by_conjugation_t) {
		if (f_v) {
			cout << "action_by_conjugation_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_by_conjugation_t not yet implemented" << endl;
		exit(1);
#if 0
		action_by_conjugation *ABC = A.G.ABC;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		ABC->compute_image(sub, Elt, a, b, verbose_level - 1);
#endif
		}
	else if (A.type_G == action_by_representation_t) {
		if (f_v) {
			cout << "action_by_representation_t" << endl;
			}
		action_by_representation *Rep = A.G.Rep;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		Rep->compute_image_INT_low_level(*sub, Elt, input, output, verbose_level - 1);
		}
	else if (A.type_G == action_on_determinant_t) {
		if (f_v) {
			cout << "action_on_determinant_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_determinant_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_sign_t) {
		if (f_v) {
			cout << "action_on_sign_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_sign_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_grassmannian_t) {
		if (f_v) {
			cout << "action_on_grassmannian_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_grassmannian_t not yet implemented" << endl;
		exit(1);
#if 0
		action_on_grassmannian *AG = A.G.AG;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = AG->compute_image_INT(sub, Elt, a, verbose_level - 1);
#endif
		}
	else if (A.type_G == action_on_spread_set_t) {
		if (f_v) {
			cout << "action_on_spread_set_t" << endl;
			}
		action_on_spread_set *AS = A.G.AS;

		AS->compute_image_low_level(Elt, input, output, verbose_level - 1);
		}
	else if (A.type_G == action_on_orthogonal_t) {
		if (f_v) {
			cout << "action_on_orthogonal_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_orthogonal_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_wedge_product_t) {
		if (f_v) {
			cout << "action_on_wedge_product_t" << endl;
			}
		action_on_wedge_product *AW = A.G.AW;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		AW->compute_image_INT_low_level(*sub, Elt, input, output, verbose_level - 1);
		}
	else if (A.type_G == action_on_homogeneous_polynomials_t) {
		if (f_v) {
			cout << "action_on_homogeneous_polynomials_t" << endl;
			}
		action_on_homogeneous_polynomials *OnHP = A.G.OnHP;

		OnHP->compute_image_INT_low_level(Elt, input, output, verbose_level - 1);
		}
	else if (A.type_G == action_by_subfield_structure_t) {
		if (f_v) {
			cout << "action_by_subfield_structure_t" << endl;
			}
		action_by_subfield_structure *SubfieldStructure = A.G.SubfieldStructure;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		SubfieldStructure->compute_image_INT_low_level(*sub, Elt, input, output, verbose_level - 1);
		}
	else if (A.type_G == action_on_cosets_t) {
		if (f_v) {
			cout << "action_on_cosets_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_cosets_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_factor_space_t) {
		if (f_v) {
			cout << "action_on_factor_space_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_factor_space_t not yet implemented" << endl;
		exit(1);
#if 0
		action_on_factor_space *AF = A.G.AF;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction" << endl;
			exit(1);
			}
		b = AF->compute_image(sub, Elt, a, verbose_level - 1);
#endif
		}
	else if (A.type_G == action_on_sets_t) {
		if (f_v) {
			cout << "action_on_sets_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_sets_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_subgroups_t) {
		if (f_v) {
			cout << "action_on_subgroups_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_subgroups_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_k_subsets_t) {
		if (f_v) {
			cout << "action_on_k_subsets_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_k_subsets_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_bricks_t) {
		if (f_v) {
			cout << "action_on_bricks_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_bricks_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_andre_t) {
		if (f_v) {
			cout << "action_on_andre_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_andre_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == action_on_pairs_t) {
		if (f_v) {
			cout << "action_on_pairs_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_pairs_t not yet implemented" << endl;
		exit(1);
#if 0
		action *sub;
		INT i, j, u, v;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction, type = action_on_pairs_t" << endl;
			exit(1);
			}
		k2ij(a, i, j, sub->degree);
		u = sub->element_image_of(i, elt, verbose_level - 1);
		v = sub->element_image_of(j, elt, verbose_level - 1);
		b = ij2k(u, v, sub->degree);
#endif
		}
	else if (A.type_G == action_on_ordered_pairs_t) {
		if (f_v) {
			cout << "action_on_ordered_pairs_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() action_on_ordered_pairs_t not yet implemented" << endl;
		exit(1);
		}
	else if (A.type_G == base_change_t) {
		if (f_v) {
			cout << "base_change_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() base_change_t not yet implemented" << endl;
		exit(1);
#if 0
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of no subaction, type = base_change_t" << endl;
			exit(1);
			}
		b = sub->element_image_of(a, elt, verbose_level - 1);
#endif
		}
	else if (A.type_G == product_action_t) {
		if (f_v) {
			cout << "product_action_t" << endl;
			}
		cout << "induced_action_element_image_of_low_level() product_action_t not yet implemented" << endl;
		exit(1);
#if 0
		product_action *PA;
		
		PA = A.G.product_action_data;
		b = PA->compute_image(&A, (INT *)elt, a, verbose_level - 1);
		}
#endif
		}
	else {
		cout << "induced_action_element_image_of_low_level() type_G unknown:: type_G = " << A.type_G << endl;
		exit(1);
		}
	if (f_v) {
		cout << "induced_action_element_image_of_low_level()  done" << endl;
		cout << "image of ";
		INT_vec_print(cout, input, A.low_level_point_size);
		cout << " in action " << A.label << " is ";
		INT_vec_print(cout, output, A.low_level_point_size);
		cout << endl;
		}
}

INT induced_action_element_linear_entry_ij(action &A, void *elt, INT i, INT j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//matrix_group &G = *A.G.matrix_grp;
	INT *Elt = (INT *) elt;
	INT b;

	if (f_v) {
		cout << "induced_action_element_linear_entry_ij() i=" << i << " j=" << j << endl;
		}
	if (A.type_G == action_on_wedge_product_t) {
		if (f_v) {
			cout << "action_on_wedge_product_t" << endl;
			}
		action_on_wedge_product *AW = A.G.AW;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_linear_entry_ij no subaction" << endl;
			exit(1);
			}
		b = AW->element_entry_ij(*sub, Elt, i, j, verbose_level - 1);
		}
	else {
		cout << "induced_action_element_linear_entry_ij() type_G unknown:: type_G = " << A.type_G << endl;
		exit(1);
		}
	return b;
}

INT induced_action_element_linear_entry_frobenius(action &A, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//matrix_group &G = *A.G.matrix_grp;
	INT *Elt = (INT *) elt;
	INT b;

	if (f_v) {
		cout << "induced_action_element_linear_entry_frobenius()" << endl;
		}
	if (A.type_G == action_on_wedge_product_t) {
		if (f_v) {
			cout << "action_on_wedge_product_t" << endl;
			}
		action_on_wedge_product *AW = A.G.AW;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_linear_entry_frobenius no subaction" << endl;
			exit(1);
			}
		b = AW->element_entry_frobenius(*sub, Elt, verbose_level - 1);
		}
	else {
		cout << "induced_action_element_linear_entry_frobenius() type_G unknown:: type_G = " << A.type_G << endl;
		exit(1);
		}
	return b;
}


void induced_action_element_one(action &A, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_one() ";
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_one(&A, (INT *) elt, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_one no subaction" << endl;
			exit(1);
			}
		sub->element_one(elt, verbose_level);
		}
}

INT induced_action_element_is_one(action &A, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_is_one() ";
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		return PA->element_is_one(&A, (INT *) elt, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_is_one no subaction" << endl;
			exit(1);
			}
		return sub->element_is_one(elt, verbose_level);
		}
}

void induced_action_element_unpack(action &A, void *elt, void *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_unpack()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_unpack((UBYTE *)elt, (INT *)Elt, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_unpack no subaction" << endl;
			exit(1);
			}
		sub->element_unpack(elt, Elt, verbose_level);
		}
}

void induced_action_element_pack(action &A, void *Elt, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_pack()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_pack((INT *)Elt, (UBYTE *)elt, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_pack no subaction" << endl;
			exit(1);
			}
		sub->element_pack(Elt, elt, verbose_level);
		}
}

void induced_action_element_retrieve(action &A, INT hdl, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_retrieve()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_retrieve(&A, hdl, (INT *)elt, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_retrieve no subaction" << endl;
			exit(1);
			}
		sub->element_retrieve(hdl, elt, verbose_level);
		}
}

INT induced_action_element_store(action &A, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_store()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		return PA->element_store(&A, (INT *)elt, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_store no subaction" << endl;
			exit(1);
			}
		return sub->element_store(elt, verbose_level);
		}
}

void induced_action_element_mult(action &A, void *a, void *b, void *ab, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_mult()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_mult((INT *)a, (INT *)b, (INT *)ab, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_mult no subaction" << endl;
			exit(1);
			}
		sub->element_mult(a, b, ab, f_v);
		}
}

void induced_action_element_invert(action &A, void *a, void *av, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_invert()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_invert((INT *)a, (INT *)av, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_invert no subaction" << endl;
			exit(1);
			}
		sub->element_invert(a, av, verbose_level);
		}
}

void induced_action_element_transpose(action &A, void *a, void *at, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_transpose" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_transpose((INT *)a, (INT *)at, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_transpose no subaction" << endl;
			exit(1);
			}
		sub->element_transpose(a, at, verbose_level);
		}
}

void induced_action_element_move(action &A, void *a, void *b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_move()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_move((INT *)a, (INT *)b, verbose_level);
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_move no subaction" << endl;
			exit(1);
			}
		sub->element_move(a, b, verbose_level);
		}
}

void induced_action_element_dispose(action &A, INT hdl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_dispose()" << endl;
		}
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		// do nothing!
		}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_dispose no subaction" << endl;
			exit(1);
			}
		sub->element_dispose(hdl, verbose_level);
		}
}

void induced_action_element_print(action &A, void *elt, ostream &ost)
{
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print((INT *)elt, ost);
		}
	else if (A.f_has_subaction) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print no subaction" << endl;
			exit(1);
			}
		sub->element_print_quick(elt, ost);

		INT n;
		INT *fp;
		
		fp = NEW_INT(sub->degree);
		n = sub->find_fixed_points(elt, fp, 0);
		ost << "with " << n << " fixed points in action " << sub->label << endl;
		FREE_INT(fp);
		sub->element_print_base_images((INT *)elt, ost);
		ost << endl;
		}
	else {
		cout << "induced_action_element_print not of type product_action_t and no subaction" << endl;
		exit(1);
		}
}

void induced_action_element_print_quick(action &A, void *elt, ostream &ost)
{
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print((INT *)elt, ost);
		}
	else if (A.f_has_subaction) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print no subaction" << endl;
			exit(1);
			}
		sub->element_print_quick(elt, ost);
		
		}
	else {
		cout << "induced_action_element_print_quick not of type product_action_t and no subaction" << endl;
		exit(1);
		}
}

void induced_action_element_print_latex(action &A, void *elt, ostream &ost)
{
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print_latex((INT *)elt, ost);
		}
	else {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print_latex no subaction" << endl;
			exit(1);
			}
		sub->element_print_latex(elt, ost);
		}
}

void induced_action_element_print_verbose(action &A, void *elt, ostream &ost)
{
	if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print((INT *)elt, ost);
		}
	else {
		action *sub;
	
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print_verbose no subaction" << endl;
			exit(1);
			}
		sub->element_print_verbose(elt, ost);
		}
}

void induced_action_element_code_for_make_element(action &A, void *elt, INT *data)
{
	//INT *Elt = (INT *) elt;

	//cout << "induced_action_element_code_for_make_element not yet implemented" << endl;
	action *sub;
	
	sub = A.subaction;
	if (sub == NULL) {
		cout << "induced_action_element_code_for_make_element no subaction" << endl;
		exit(1);
		}
	sub->element_code_for_make_element(elt, data);
	//exit(1);
}

void induced_action_element_print_for_make_element(action &A, void *elt, ostream &ost)
{
	//INT *Elt = (INT *) elt;

	//cout << "induced_action_element_print_for_make_element not yet implemented" << endl;
	action *sub;
	
	sub = A.subaction;
	if (sub == NULL) {
		cout << "induced_action_element_print_for_make_element no subaction" << endl;
		exit(1);
		}
	sub->element_print_for_make_element(elt, ost);
	//exit(1);
}

void induced_action_element_print_for_make_element_no_commas(action &A, void *elt, ostream &ost)
{
	//INT *Elt = (INT *) elt;

	//cout << "induced_action_element_print_for_make_element_no_commas not yet implemented" << endl;
	action *sub;
	
	sub = A.subaction;
	if (sub == NULL) {
		cout << "induced_action_element_print_for_make_element_no_commas no subaction" << endl;
		exit(1);
		}
	sub->element_print_for_make_element_no_commas(elt, ost);
	//exit(1);
}

void induced_action_print_point(action &A, INT a, ostream &ost)
{

	if (A.type_G == action_by_right_multiplication_t) {
		//action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		//ABRM->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_by_restriction_t) {
		//action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		//ABRM->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_by_conjugation_t) {
		//action_by_conjugation *ABC = A.G.ABC;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		//ABC->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_determinant_t) {
		//action_on_determinant *AD = A.G.AD;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		}
	else if (A.type_G == action_on_sign_t) {
		//action_on_sign *OnSign = A.G.OnSign;
		ost << a;
		}
	else if (A.type_G == action_on_sets_t) {
		//action_on_sets *AOS = A.G.on_sets;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		//AOS->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_subgroups_t) {
		//action_on_subgroups *AOS = A.G.on_subgroups;
		ost << a;
		//AOS->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_k_subsets_t) {
		//action_on_k_subsets *On_k_subsets = A.G.on_k_subsets;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		}
	else if (A.type_G == action_on_orbits_t) {
		//action_on_orbits *On_orbits = A.G.OnOrbits;
#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
#endif
		ost << a;
		}
	else if (A.type_G == action_on_bricks_t) {
		//action_on_bricks *On_bricks = A.G.OnBricks;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		}
	else if (A.type_G == action_on_andre_t) {
		//action_on_andre *OnAndre = A.G.OnAndre;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
		}
	else if (A.type_G == action_on_pairs_t) {
		action *sub;
		INT i, j;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction, type = action_on_pairs_t" << endl;
			exit(1);
			}
		k2ij(a, i, j, sub->degree);
		cout << "a={" << i << "," << j << "}";
		}
	else if (A.type_G == action_on_ordered_pairs_t) {
		action *sub;
		INT a2, swap, tmp, i, j;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction, type = action_on_ordered_pairs_t" << endl;
			exit(1);
			}
		swap = a % 2;
		a2 = a / 2;
		k2ij(a2, i, j, sub->degree);
		if (swap) {
			tmp = i;
			i = j;
			j = tmp;
			}
		cout << "a=(" << i << "," << j << ")";
		}
	else if (A.type_G == base_change_t) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction, type = base_change_t" << endl;
			exit(1);
			}
		ost << a;
		}
	else if (A.type_G == product_action_t) {
		product_action *PA;
		
		PA = A.G.product_action_data;
		ost << a;
		}
	else if (A.type_G == action_on_grassmannian_t) {
		if (FALSE) {
			cout << "action_on_grassmannian_t" << endl;
			}
		action_on_grassmannian *AG = A.G.AG;

#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
#endif
		AG->print_point(a, ost);
		//b = AG->compute_image_INT(sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_spread_set_t) {
		if (FALSE) {
			cout << "action_on_spread_set_t" << endl;
			}
		//action_on_spread_set *AS = A.G.AS;

		ost << a;
		}
	else if (A.type_G == action_on_orthogonal_t) {
		if (FALSE) {
			cout << "action_on_orthogonal_t" << endl;
			}
		//action_on_orthogonal *AO = A.G.AO;

		ost << a;
		
#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
			}
		ost << a;
#endif
		}
	else {
		cout << "induced_action_print_point type_G unknown:: type_G = ";
		action_print_symmetry_group_type(cout, A.type_G);
		cout << endl;
		exit(1);
		}
}




