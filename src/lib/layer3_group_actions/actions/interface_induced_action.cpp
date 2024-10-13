// interface_induced_action.cpp
//
// Anton Betten
//
// started:  November 13, 2007
// last change:  November 9, 2010




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



// #############################################################################
// interface functions: induced action
// #############################################################################

static long int induced_action_element_image_of(
		action &A, long int a,
	void *elt, int verbose_level);
static void induced_action_element_image_of_low_level(
		action &A,
	int *input, int *output, void *elt, int verbose_level);
static void induced_action_element_one(
		action &A,
	void *elt, int verbose_level);
static int induced_action_element_is_one(
		action &A,
	void *elt, int verbose_level);
static void induced_action_element_unpack(
		action &A,
	void *elt, void *Elt, int verbose_level);
static void induced_action_element_pack(
		action &A,
	void *Elt, void *elt, int verbose_level);
static void induced_action_element_retrieve(
		action &A,
	int hdl, void *elt, int verbose_level);
static int induced_action_element_store(
		action &A,
	void *elt, int verbose_level);
static void induced_action_element_mult(
		action &A,
	void *a, void *b, void *ab, int verbose_level);
static void induced_action_element_invert(
		action &A,
	void *a, void *av, int verbose_level);
static void induced_action_element_transpose(
		action &A,
	void *a, void *at, int verbose_level);
static void induced_action_element_move(
		action &A,
	void *a, void *b, int verbose_level);
static void induced_action_element_dispose(
		action &A,
	int hdl, int verbose_level);
static void induced_action_element_print(
		action &A,
	void *elt, std::ostream &ost);
static void induced_action_element_print_quick(
		action &A,
	void *elt, std::ostream &ost);
static void induced_action_element_print_latex(
		action &A,
	void *elt, std::ostream &ost);
static void induced_action_element_print_latex_with_point_labels(
	action &A,
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data);
static void induced_action_element_print_verbose(
		action &A,
	void *elt, std::ostream &ost);
static void induced_action_element_code_for_make_element(
		action &A,
	void *elt, int *data);
static void induced_action_element_print_for_make_element(
		action &A,
	void *elt, std::ostream &ost);
static void induced_action_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
static void induced_action_print_point(
		action &A, long int a, std::ostream &ost, int verbose_level);
static void induced_action_unrank_point(
		action &A, long int rk, int *v, int verbose_level);
static long int induced_action_rank_point(
		action &A, int *v, int verbose_level);


void action_pointer_table::init_function_pointers_induced_action()
{
	label.assign("function_pointers_induced_action");
	//ptr_get_transversal_rep = induced_action_get_transversal_rep;
	ptr_element_image_of = induced_action_element_image_of;
	ptr_element_image_of_low_level = induced_action_element_image_of_low_level;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = induced_action_element_one;
	ptr_element_is_one = induced_action_element_is_one;
	ptr_element_unpack = induced_action_element_unpack;
	ptr_element_pack = induced_action_element_pack;
	ptr_element_retrieve = induced_action_element_retrieve;
	ptr_element_store = induced_action_element_store;
	ptr_element_mult = induced_action_element_mult;
	ptr_element_invert = induced_action_element_invert;
	ptr_element_transpose = induced_action_element_transpose;
	ptr_element_move = induced_action_element_move;
	ptr_element_dispose = induced_action_element_dispose;
	ptr_element_print = induced_action_element_print;
	ptr_element_print_quick = induced_action_element_print_quick;
	ptr_element_print_latex = induced_action_element_print_latex;
	ptr_element_print_latex_with_point_labels =
			induced_action_element_print_latex_with_point_labels;
	ptr_element_print_verbose = induced_action_element_print_verbose;
	ptr_element_code_for_make_element =
			induced_action_element_code_for_make_element;
	ptr_element_print_for_make_element =
			induced_action_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			induced_action_element_print_for_make_element_no_commas;
	ptr_print_point = induced_action_print_point;
	ptr_unrank_point = induced_action_unrank_point;
	ptr_rank_point = induced_action_rank_point;
}


static long int induced_action_element_image_of(
		action &A,
		long int a, void *elt, int verbose_level)
{
	int *Elt = (int *) elt;
	long int b = 0;
	int f_v = (verbose_level >= 1);
	action_global AG;
	
	if (f_v) {
		cout << "induced_action_element_image_of "
				"computing image of " << a
				<< " in action " << A.label << endl;
	}
	if (A.type_G == action_by_right_multiplication_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_by_right_multiplication_t" << endl;
		}
		induced_actions::action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = ABRM->compute_image(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_by_restriction_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_by_restriction_t" << endl;
		}
		induced_actions::action_by_restriction *ABR = A.G.ABR;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "induced_action_element_image_of "
					"before ABR->compute_image a=" << a << endl;
			cout << "verbose_level = " << verbose_level << endl;
		}
		b = ABR->compute_image(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_by_conjugation_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_by_conjugation_t" << endl;
		}
		induced_actions::action_by_conjugation *ABC = A.G.ABC;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = ABC->compute_image(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_by_representation_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_by_representation_t" << endl;
		}
		induced_actions::action_by_representation *Rep = A.G.Rep;
		//action *sub;
		
		//sub = A.subaction;
#if 0
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
#endif
		b = Rep->compute_image_int(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_determinant_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_determinant_t" << endl;
		}
		induced_actions::action_on_determinant *AD = A.G.AD;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AD->compute_image(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_galois_group_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_galois_group_t" << endl;
		}
		induced_actions::action_on_galois_group *AG = A.G.on_Galois_group;

		b = AG->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_sign_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_sign_t" << endl;
		}
		induced_actions::action_on_sign *OnSign = A.G.OnSign;

		b = OnSign->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_grassmannian_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_grassmannian_t" << endl;
		}
		induced_actions::action_on_grassmannian *AG = A.G.AG;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AG->compute_image_int(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_spread_set_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_spread_set_t" << endl;
		}
		induced_actions::action_on_spread_set *AS = A.G.AS;

		b = AS->compute_image_int(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_orthogonal_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_orthogonal_t" << endl;
		}
		induced_actions::action_on_orthogonal *AO = A.G.AO;
		b = AO->compute_image_int(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_wedge_product_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_wedge_product_t" << endl;
		}
		induced_actions::action_on_wedge_product *AW = A.G.AW;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AW->compute_image_int(/**sub,*/ Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_homogeneous_polynomials_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_homogeneous_polynomials_t" << endl;
		}
		induced_actions::action_on_homogeneous_polynomials *OnHP = A.G.OnHP;

		b = OnHP->compute_image_int(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_by_subfield_structure_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_by_subfield_structure_t" << endl;
		}
		induced_actions::action_by_subfield_structure *SubfieldStructure =
				A.G.SubfieldStructure;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = SubfieldStructure->compute_image_int(
				*sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_cosets_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_cosets_t" << endl;
		}
		induced_actions::action_on_cosets *AC = A.G.OnCosets;

		//cout << "interface.cpp: action_on_cosets "
		//"computing image of " << a << endl;
		b = AC->compute_image(Elt, a, verbose_level - 1);
		//cout << "interface.cpp: action_on_cosets image of "
		// << a << " is " << b << endl;
	}
	else if (A.type_G == action_on_factor_space_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_factor_space_t" << endl;
		}
		induced_actions::action_on_factor_space *AF = A.G.AF;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AF->compute_image(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_sets_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_sets_t" << endl;
		}
		induced_actions::action_on_sets *AOS = A.G.on_sets;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AOS->compute_image(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_set_partitions_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_set_partitions_t" << endl;
		}
		induced_actions::action_on_set_partitions *OSP = A.G.OnSetPartitions;
		b = OSP->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_subgroups_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_subgroups_t" << endl;
		}
		induced_actions::action_on_subgroups *AOS = A.G.on_subgroups;

		b = AOS->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_k_subsets_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_k_subsets_t" << endl;
		}
		induced_actions::action_on_k_subsets *On_k_subsets = A.G.on_k_subsets;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = On_k_subsets->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_orbits_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_orbits_t" << endl;
		}
		induced_actions::action_on_orbits *On_orbits = A.G.OnOrbits;

		b = On_orbits->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_flags_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_flags_t" << endl;
		}
		induced_actions::action_on_flags *On_flags = A.G.OnFlags;

		b = On_flags->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_bricks_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_bricks_t" << endl;
		}
		induced_actions::action_on_bricks *On_bricks = A.G.OnBricks;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = On_bricks->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_andre_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_andre_t" << endl;
		}
		induced_actions::action_on_andre *On_andre = A.G.OnAndre;

#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
#endif

		b = On_andre->compute_image(Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_pairs_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_pairs_t" << endl;
		}
		action *sub;
		combinatorics::combinatorics_domain Combi;
		long int i, j, u, v;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction, type = action_on_pairs_t" << endl;
			exit(1);
		}
		Combi.k2ij_lint(a, i, j, sub->degree);
		u = sub->Group_element->element_image_of(i, elt, verbose_level - 1);
		v = sub->Group_element->element_image_of(j, elt, verbose_level - 1);
		b = Combi.ij2k_lint(u, v, sub->degree);
	}
	else if (A.type_G == action_on_ordered_pairs_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_ordered_pairs_t" << endl;
		}
		action *sub;
		combinatorics::combinatorics_domain Combi;
		long int a2, b2, swap, swap2, i, j, tmp, u, v, u2, v2;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction, type = action_on_ordered_pairs_t" << endl;
			exit(1);
		}
		swap = a % 2;
		a2 = a / 2;
		Combi.k2ij_lint(a2, i, j, sub->degree);
		if (swap) {
			tmp = i;
			i = j;
			j = tmp;
		}
		u = sub->Group_element->element_image_of(i, elt, verbose_level - 1);
		v = sub->Group_element->element_image_of(j, elt, verbose_level - 1);
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
		b2 = Combi.ij2k_lint(u2, v2, sub->degree);
		b = 2 * b2 + swap2;
#if 0
		cout << "induced_action_element_image_of "
				"action_on_ordered_pairs_t" << endl;
		cout << a << " -> " << b << endl;
		cout << "(" << i << "," << j << ") -> "
				"(" << u << "," << v << ")" << endl;
		cout << "under" << endl;
		sub->element_print(elt, cout);
		cout << endl;
#endif
	}
	else if (A.type_G == base_change_t) {
		if (f_v) {
			cout << "induced_action_element_image_of base_change_t" << endl;
		}
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction, type = base_change_t" << endl;
			exit(1);
		}
		b = sub->Group_element->element_image_of(a, elt, verbose_level - 1);
	}
	else if (A.type_G == product_action_t) {
		if (f_v) {
			cout << "induced_action_element_image_of product_action_t" << endl;
		}
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		b = PA->compute_image(&A, (int *)elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_interior_direct_product_t) {
		if (f_v) {
			cout << "induced_action_element_image_of action_on_interior_direct_product_t" << endl;
		}
		induced_actions::action_on_interior_direct_product *IDP;

		IDP = A.G.OnInteriorDirectProduct;
		b = IDP->compute_image((int *)elt, a, verbose_level - 1);
	}
	else {
		cout << "induced_action_element_image_of type_G "
				"unknown:: type_G = " << A.type_G << endl;
		AG.action_print_symmetry_group_type(cout, A.type_G);
		cout << "action:" << endl;
		A.print_info();
		exit(1);
	}
	if (f_v) {
		cout << "induced_action_element_image_of type=";
			AG.action_print_symmetry_group_type(cout, A.type_G);
			cout << " image of " << a << " is " << b << endl;
	}
	return b;
}

static void induced_action_element_image_of_low_level(
		action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	int *Elt = (int *) elt;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "induced_action_element_image_of_low_level "
				"computing image of ";
		Int_vec_print(cout, input, A.low_level_point_size);
		cout << " in action " << A.label << endl;
	}
	if (A.type_G == action_by_right_multiplication_t) {
		if (f_v) {
			cout << "action_by_right_multiplication_t" << endl;
		}

		cout << "induced_action_element_image_of_low_level "
				"action_by_right_multiplication_t "
				"not yet implemented" << endl;
		exit(1);
#if 0
		action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		ABRM->compute_image(sub, Elt, a, b, verbose_level - 1);
#endif
	}
	else if (A.type_G == action_by_restriction_t) {
		if (f_v) {
			cout << "action_by_restriction_t" << endl;
		}

		//cout << "induced_action_element_image_of_low_level
		// action_by_restriction_t not yet implemented" << endl;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"type action_by_restriction_t, "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->image_of_low_level(elt, input, output, verbose_level - 1);
	}
	else if (A.type_G == action_by_conjugation_t) {
		if (f_v) {
			cout << "action_by_conjugation_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_by_conjugation_t not yet implemented" << endl;
		exit(1);
#if 0
		action_by_conjugation *ABC = A.G.ABC;
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		ABC->compute_image(sub, Elt, a, b, verbose_level - 1);
#endif
	}
	else if (A.type_G == action_by_representation_t) {
		if (f_v) {
			cout << "action_by_representation_t" << endl;
		}
		induced_actions::action_by_representation *Rep = A.G.Rep;

#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
#endif
		Rep->compute_image_int_low_level(
				Elt, input, output, verbose_level - 1);
	}
	else if (A.type_G == action_on_determinant_t) {
		if (f_v) {
			cout << "action_on_determinant_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_determinant_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_galois_group_t) {
		if (f_v) {
			cout << "action_on_galois_group_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_galois_group_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_sign_t) {
		if (f_v) {
			cout << "action_on_sign_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_sign_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_grassmannian_t) {
		if (f_v) {
			cout << "action_on_grassmannian_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_grassmannian_t not yet implemented" << endl;
		exit(1);
#if 0
		action_on_grassmannian *AG = A.G.AG;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AG->compute_image_int(sub, Elt, a, verbose_level - 1);
#endif
	}
	else if (A.type_G == action_on_spread_set_t) {
		if (f_v) {
			cout << "action_on_spread_set_t" << endl;
		}
		induced_actions::action_on_spread_set *AS = A.G.AS;

		AS->compute_image_low_level(Elt, input, output, verbose_level - 1);
	}
	else if (A.type_G == action_on_orthogonal_t) {
		if (f_v) {
			cout << "action_on_orthogonal_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_orthogonal_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_wedge_product_t) {
		if (f_v) {
			cout << "action_on_wedge_product_t" << endl;
		}
		induced_actions::action_on_wedge_product *AW = A.G.AW;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		AW->compute_image_int_low_level(//*sub,
				Elt, input, output, verbose_level - 1);
	}
	else if (A.type_G == action_on_homogeneous_polynomials_t) {
		if (f_v) {
			cout << "action_on_homogeneous_polynomials_t" << endl;
		}
		induced_actions::action_on_homogeneous_polynomials *OnHP = A.G.OnHP;

		OnHP->compute_image_int_low_level(Elt,
				input, output, verbose_level - 1);
	}
	else if (A.type_G == action_by_subfield_structure_t) {
		if (f_v) {
			cout << "action_by_subfield_structure_t" << endl;
		}
		induced_actions::action_by_subfield_structure *SubfieldStructure =
				A.G.SubfieldStructure;


		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		SubfieldStructure->compute_image_int_low_level(*sub,
				Elt, input, output, verbose_level - 1);
	}
	else if (A.type_G == action_on_cosets_t) {
		if (f_v) {
			cout << "action_on_cosets_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_cosets_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_factor_space_t) {
		if (f_v) {
			cout << "action_on_factor_space_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_factor_space_t not yet implemented" << endl;
		exit(1);
#if 0
		action_on_factor_space *AF = A.G.AF;

		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction" << endl;
			exit(1);
		}
		b = AF->compute_image(sub, Elt, a, verbose_level - 1);
#endif
	}
	else if (A.type_G == action_on_sets_t) {
		if (f_v) {
			cout << "action_on_sets_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_sets_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_set_partitions_t) {
		if (f_v) {
			cout << "induced_action_element_image_of "
					"action_on_set_partitions_t" << endl;
		}
		exit(1);
	}
	else if (A.type_G == action_on_subgroups_t) {
		if (f_v) {
			cout << "action_on_subgroups_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_subgroups_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_k_subsets_t) {
		if (f_v) {
			cout << "action_on_k_subsets_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_k_subsets_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_bricks_t) {
		if (f_v) {
			cout << "action_on_bricks_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_bricks_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_andre_t) {
		if (f_v) {
			cout << "action_on_andre_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_andre_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == action_on_pairs_t) {
		if (f_v) {
			cout << "action_on_pairs_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"action_on_pairs_t not yet implemented" << endl;
		exit(1);
#if 0
		action *sub;
		int i, j, u, v;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction, type = action_on_pairs_t" << endl;
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
		cout << "induced_action_element_image_of_low_level "
				"action_on_ordered_pairs_t not yet implemented" << endl;
		exit(1);
	}
	else if (A.type_G == base_change_t) {
		if (f_v) {
			cout << "base_change_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"base_change_t not yet implemented" << endl;
		exit(1);
#if 0
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_image_of "
					"no subaction, type = base_change_t" << endl;
			exit(1);
		}
		b = sub->element_image_of(a, elt, verbose_level - 1);
#endif
	}
	else if (A.type_G == product_action_t) {
		if (f_v) {
			cout << "product_action_t" << endl;
		}
		cout << "induced_action_element_image_of_low_level "
				"product_action_t not yet implemented" << endl;
		exit(1);
#if 0
		product_action *PA;
		
		PA = A.G.product_action_data;
		b = PA->compute_image(&A, (int *)elt, a, verbose_level - 1);
#endif
	}
	else if (A.type_G == action_on_interior_direct_product_t) {
		if (f_v) {
			cout << "action_on_interior_direct_product_t" << endl;
		}
		//action_on_interior_direct_product *IDP;

		cout << "action_on_interior_direct_product_t "
				"not yet implemented" << endl;
		exit(1);
	}
	else {
		cout << "induced_action_element_image_of_low_level "
				"type_G unknown:: type_G = " << A.type_G << endl;
		exit(1);
	}
	if (f_v) {
		cout << "induced_action_element_image_of_low_level  done" << endl;
		cout << "image of ";
		Int_vec_print(cout, input, A.low_level_point_size);
		cout << " in action " << A.label << " is ";
		Int_vec_print(cout, output, A.low_level_point_size);
		cout << endl;
	}
}


static void induced_action_element_one(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_one ";
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_one(&A, (int *) elt, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_one "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_one(elt, verbose_level);
	}
}

static int induced_action_element_is_one(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_is_one ";
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		return PA->element_is_one(&A, (int *) elt, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_is_one "
					"no subaction" << endl;
			exit(1);
		}
		return sub->Group_element->element_is_one(elt, verbose_level);
	}
}

static void induced_action_element_unpack(
		action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_unpack" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_unpack((uchar *)elt, (int *)Elt, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_unpack "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_unpack(elt, Elt, verbose_level);
	}
}

static void induced_action_element_pack(
		action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_pack" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_pack((int *)Elt, (uchar *)elt, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_pack "
					"no subaction" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "induced_action_element_pack before sub->element_pack" << endl;
		}
		sub->Group_element->element_pack(Elt, elt, verbose_level);
		if (f_v) {
			cout << "induced_action_element_pack after sub->element_pack" << endl;
		}
	}
}

static void induced_action_element_retrieve(
		action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_retrieve" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_retrieve(&A, hdl,
				(int *)elt, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_retrieve "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_retrieve(hdl, elt, verbose_level);
	}
}

static int induced_action_element_store(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_store" << endl;
		}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		return PA->element_store(&A, (int *)elt, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_store "
					"no subaction" << endl;
			exit(1);
		}
		return sub->Group_element->element_store(elt, verbose_level);
	}
}

static void induced_action_element_mult(
		action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_mult" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_mult((int *)a, (int *)b, (int *)ab, verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_mult "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_mult(a, b, ab, verbose_level);
	}
	if (f_v) {
		cout << "induced_action_element_mult done" << endl;
	}
}

static void induced_action_element_invert(
		action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_invert" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_invert((int *)a, (int *)av,
				verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_invert "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_invert(a, av, verbose_level);
	}
}

static void induced_action_element_transpose(
		action &A,
		void *a, void *at, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_transpose" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_transpose((int *)a, (int *)at,
				verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_transpose "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_transpose(a, at, verbose_level);
	}
}

static void induced_action_element_move(
		action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_move" << endl;
	}
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_move((int *)a, (int *)b,
				verbose_level);
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_move "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_move(a, b, verbose_level);
	}
}

static void induced_action_element_dispose(
		action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *sub;
	
	if (f_v) {
		cout << "induced_action_element_dispose" << endl;
	}
	if (A.type_G == product_action_t) {
		//product_action *PA;
		
		//PA = A.G.product_action_data;
		// do nothing!
	}
	else {
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_dispose "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_dispose(hdl, verbose_level);
	}
}

static void induced_action_element_print(
		action &A,
		void *elt, std::ostream &ost)
{
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print((int *)elt, ost);
	}
	else if (A.f_has_subaction) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_print_quick(elt, ost);


#if 0
		int n;
		int *fp;
		
		fp = NEW_int(sub->degree);
		n = sub->find_fixed_points(elt, fp, 0);
		ost << "with " << n << " fixed points in action "
				<< sub->label << endl;
		FREE_int(fp);
		sub->element_print_base_images((int *)elt, ost);
		ost << endl;
#endif

	}
	else {
		cout << "induced_action_element_print "
				"not of type product_action_t and "
				"no subaction" << endl;
		exit(1);
	}
}

static void induced_action_element_print_quick(
		action &A,
		void *elt, std::ostream &ost)
{
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print((int *)elt, ost);
	}
	else if (A.f_has_subaction) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_print_quick(elt, ost);
		
	}
	else {
		cout << "induced_action_element_print_quick "
				"not of type product_action_t and "
				"no subaction" << endl;
		exit(1);
	}
}

static void induced_action_element_print_latex(
		action &A,
		void *elt, std::ostream &ost)
{
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print_latex((int *)elt, ost);
	}

	else if (A.type_G == action_on_wedge_product_t) {
		induced_actions::action_on_wedge_product *AW;

		AW = A.G.AW;
		AW->element_print_latex((int *)elt, ost);
	}

	else {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print_latex "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_print_latex(elt, ost);
	}
}

static void induced_action_element_print_latex_with_point_labels(
	action &A,
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data)
{
	int f_v = false;
	int *Elt = (int *) elt;
	int i, j;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "induced_action_element_print_latex_with_point_labels "
				"degree = " << A.degree << endl;
	}
	int *p = NEW_int(A.degree);
	for (i = 0; i < A.degree; i++) {
		//cout << "matrix_group_element_print_as_permutation
		//computing image of i=" << i << endl;
		//if (i == 3)
			//f_v = true;
		//else
			//f_v = false;
		j = A.Group_element->element_image_of(i, Elt, 0 /* verbose_level */);
		p[i] = j;
	}
	//Combi.perm_print(ost, p, A.degree);
	//ost << ";";
	Combi.Permutations->perm_print_with_point_labels(ost, p,
			A.degree, Point_labels, data);
	FREE_int(p);




#if 0
	if (A.type_G == product_action_t) {
		product_action *PA;

		PA = A.G.product_action_data;
		PA->element_print_latex((int *)elt, ost);
	}
	else {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print_latex "
					"no subaction" << endl;
			exit(1);
		}
		sub->element_print_latex_with_print_point_function(
				elt, ost, point_label, point_label_data);
	}
#endif

}

static void induced_action_element_print_verbose(
		action &A,
		void *elt, std::ostream &ost)
{
	if (A.type_G == product_action_t) {
		induced_actions::product_action *PA;
		
		PA = A.G.product_action_data;
		PA->element_print((int *)elt, ost);
	}
	else {
		action *sub;
	
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_element_print_verbose "
					"no subaction" << endl;
			exit(1);
		}
		sub->Group_element->element_print_verbose(elt, ost);
	}
}

static void induced_action_element_code_for_make_element(
		action &A,
		void *elt, int *data)
{
	//int *Elt = (int *) elt;

	//cout << "induced_action_element_code_for_make_element
	//not yet implemented" << endl;
	action *sub;
	
	sub = A.subaction;
	if (sub == NULL) {
		cout << "induced_action_element_code_for_"
				"make_element no subaction" << endl;
		exit(1);
	}
	sub->Group_element->element_code_for_make_element(elt, data);
	//exit(1);
}

static void induced_action_element_print_for_make_element(
		action &A,
		void *elt, std::ostream &ost)
{
	//int *Elt = (int *) elt;

	//cout << "induced_action_element_print_for_
	// make_element not yet implemented" << endl;
	action *sub;
	
	sub = A.subaction;
	if (sub == NULL) {
		cout << "induced_action_element_print_for_"
				"make_element no subaction" << endl;
		exit(1);
	}
	sub->Group_element->element_print_for_make_element(elt, ost);
	//exit(1);
}

static void induced_action_element_print_for_make_element_no_commas(
		action &A, void *elt, std::ostream &ost)
{
	//int *Elt = (int *) elt;

	//cout << "induced_action_element_print_for_"
	// "make_element_no_commas not yet implemented" << endl;
	action *sub;
	
	sub = A.subaction;
	if (sub == NULL) {
		cout << "induced_action_element_print_for_"
				"make_element_no_commas no subaction" << endl;
		exit(1);
	}
	sub->Group_element->element_print_for_make_element_no_commas(elt, ost);
	//exit(1);
}

static void induced_action_print_point(
		action &A,
		long int a, std::ostream &ost, int verbose_level)
{
	action_global AG;

#if 0
	cout << "induced_action_print_point type=";
	AG.action_print_symmetry_group_type(cout, A.type_G);
	cout << endl;
#endif

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
	else if (A.type_G == action_on_galois_group_t) {
		//action_on_galois_group *AG = A.G.on_Galois_group;

		ost << a;
	}
	else if (A.type_G == action_on_sign_t) {
		//action_on_sign *OnSign = A.G.OnSign;
		ost << a;
	}
	else if (A.type_G == action_on_sets_t) {
		induced_actions::action_on_sets *AOS = A.G.on_sets;
		action *sub;
		int i;
		long int b;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point no subaction" << endl;
			exit(1);
		}
		ost << a << "=";
		Lint_vec_print(ost, AOS->sets[AOS->perm[a]], AOS->set_size);
		ost << endl;
		for (i = 0; i < AOS->set_size; i++) {
			ost << "$$" << endl;
			ost << "$$" << endl;
			b = AOS->sets[AOS->perm[a]][i];
			sub->Group_element->print_point(b, ost);
		}
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
		combinatorics::combinatorics_domain Combi;
		int i, j;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point "
					"no subaction, type = action_on_pairs_t" << endl;
			exit(1);
		}
		Combi.k2ij(a, i, j, sub->degree);
		cout << "a={" << i << "," << j << "}";
	}
	else if (A.type_G == action_on_ordered_pairs_t) {
		action *sub;
		combinatorics::combinatorics_domain Combi;
		int a2, swap, tmp, i, j;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point "
					"no subaction, type = action_on_ordered_pairs_t" << endl;
			exit(1);
		}
		swap = a % 2;
		a2 = a / 2;
		Combi.k2ij(a2, i, j, sub->degree);
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
			cout << "induced_action_print_point "
					"no subaction, type = base_change_t" << endl;
			exit(1);
			}
		ost << a;
	}
	else if (A.type_G == product_action_t) {
		//product_action *PA;
		
		//PA = A.G.product_action_data;
		ost << a;
	}
	else if (A.type_G == action_on_grassmannian_t) {
		if (false) {
			cout << "action_on_grassmannian_t" << endl;
			}
		induced_actions::action_on_grassmannian *AG = A.G.AG;

#if 0
		action *sub;
		
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point "
					"no subaction" << endl;
			exit(1);
			}
		//ost << a;
#endif
		AG->print_point(a, ost);
		//b = AG->compute_image_int(sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_on_spread_set_t) {
		if (false) {
			cout << "action_on_spread_set_t" << endl;
			}
		//action_on_spread_set *AS = A.G.AS;

		ost << a;
	}
	else if (A.type_G == action_on_orthogonal_t) {
		if (false) {
			cout << "action_on_orthogonal_t" << endl;
			}
		induced_actions::action_on_orthogonal *AO = A.G.AO;

		ost << a << " = ";
		
		int *v;

		v = NEW_int(AO->low_level_point_size);
		AO->unrank_point(v, a);
		Int_vec_print(ost, v, AO->low_level_point_size);
		FREE_int(v);
	}
	else if (A.type_G == action_on_interior_direct_product_t) {
		if (false) {
			cout << "action_on_interior_direct_product_t" << endl;
		}
		induced_actions::action_on_interior_direct_product *IDP;
		int i, j;

		IDP = A.G.OnInteriorDirectProduct;
		i = a / IDP->nb_cols;
		j = a % IDP->nb_cols;
		ost << "(" << i << "," << j << ")";
	}
	else if (A.type_G == action_on_wedge_product_t) {
		if (false) {
			cout << "action_on_wedge_product_t" << endl;
		}
		induced_actions::action_on_wedge_product *AW = A.G.AW;

		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_print_point "
					"no subaction" << endl;
			exit(1);
		}
		AW->unrank_point(AW->wedge_v1, a);
		ost << a << " = ";
		Int_vec_print(ost, AW->wedge_v1, AW->wedge_dimension);

	}



	else {
		cout << "induced_action_print_point type_G unknown:: type_G = ";
		AG.action_print_symmetry_group_type(cout, A.type_G);
		cout << endl;
		exit(1);
	}
}


static void induced_action_unrank_point(
		action &A, long int rk, int *v, int verbose_level)
{
	action_global AG;
	//cout << "induced_action_unrank_point" << endl;

	if (A.type_G == action_by_right_multiplication_t) {
		induced_actions::action_by_right_multiplication *ABRM = A.G.ABRM;


		ABRM->Base_group->element_unrank_lint(rk, ABRM->Elt1);
		ABRM->Base_group->A->Group_element->code_for_make_element(v, ABRM->Elt1);
		}
	else if (A.type_G == action_by_restriction_t) {
		induced_actions::action_by_restriction *ABR = A.G.ABR;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		int rk0;
		rk0 = ABR->original_point(rk);
		sub->Group_element->unrank_point(rk0, v);
		}
	else if (A.type_G == action_by_conjugation_t) {
		//action_by_conjugation *ABC = A.G.ABC;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		//ABC->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_determinant_t) {
		//action_on_determinant *AD = A.G.AD;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_galois_group_t) {
		//action_on_galois_group *AG = A.G.on_Galois_group;

		//ost << a;
		}
	else if (A.type_G == action_on_sign_t) {
		//action_on_sign *OnSign = A.G.OnSign;
		//ost << a;
		}
	else if (A.type_G == action_on_sets_t) {
		//action_on_sets *AOS = A.G.on_sets;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		//AOS->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_subgroups_t) {
		//action_on_subgroups *AOS = A.G.on_subgroups;
		//ost << a;
		//AOS->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_k_subsets_t) {
		//action_on_k_subsets *On_k_subsets = A.G.on_k_subsets;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_orbits_t) {
		//action_on_orbits *On_orbits = A.G.OnOrbits;
#if 0
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
#endif
		//ost << a;
		}
	else if (A.type_G == action_on_bricks_t) {
		//action_on_bricks *On_bricks = A.G.OnBricks;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_andre_t) {
		//action_on_andre *OnAndre = A.G.OnAndre;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_pairs_t) {
		action *sub;
		//int i, j;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point "
					"no subaction, type = action_on_pairs_t" << endl;
			exit(1);
			}
		//k2ij(a, i, j, sub->degree);
		//cout << "a={" << i << "," << j << "}";
		}
	else if (A.type_G == action_on_ordered_pairs_t) {
		action *sub;
		//int a2, swap, tmp, i, j;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point "
					"no subaction, type = action_on_ordered_pairs_t" << endl;
			exit(1);
			}
#if 0
		swap = a % 2;
		a2 = a / 2;
		k2ij(a2, i, j, sub->degree);
		if (swap) {
			tmp = i;
			i = j;
			j = tmp;
			}
		cout << "a=(" << i << "," << j << ")";
#endif
		}
	else if (A.type_G == base_change_t) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point "
					"no subaction, type = base_change_t" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == product_action_t) {
		//product_action *PA;

		//PA = A.G.product_action_data;
		//ost << a;
		}
	else if (A.type_G == action_on_grassmannian_t) {
		if (false) {
			cout << "action_on_grassmannian_t" << endl;
			}
		induced_actions::action_on_grassmannian *AG = A.G.AG;

		AG->unrank(rk, v, 0 /*verbose_level*/);
		}
	else if (A.type_G == action_on_spread_set_t) {
		if (false) {
			cout << "action_on_spread_set_t" << endl;
			}
		//action_on_spread_set *AS = A.G.AS;

		//ost << a;
		}
	else if (A.type_G == action_on_orthogonal_t) {
		if (false) {
			cout << "action_on_orthogonal_t" << endl;
			}
		induced_actions::action_on_orthogonal *AO = A.G.AO;

		AO->unrank_point(v, rk);
		}
	else if (A.type_G == action_on_wedge_product_t) {
		if (false) {
			cout << "induced_action_unrank_point "
					"action_on_wedge_product_t" << endl;
		}
		induced_actions::action_on_wedge_product *AW = A.G.AW;

		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_unrank_point "
					"no subaction" << endl;
			exit(1);
		}
		AW->unrank_point(v, rk);
		//b = AW->compute_image_int(*sub, Elt, a, verbose_level - 1);
	}
	else if (A.type_G == action_by_representation_t) {
		if (false) {
			cout << "induced_action_unrank_point "
					"action_by_representation_t" << endl;
		}
		induced_actions::action_by_representation *Rep = A.G.Rep;

		Rep->unrank_point(rk, v, 0 /* verbose_level*/);
		//b = AW->compute_image_int(*sub, Elt, a, verbose_level - 1);
	}
	else {
		cout << "induced_action_unrank_point type_G unknown:: type_G = ";
		AG.action_print_symmetry_group_type(cout, A.type_G);
		cout << endl;
		exit(1);
		}

}

static long int induced_action_rank_point(
		action &A, int *v, int verbose_level)
{
	action_global AG;
	//cout << "induced_action_rank_point" << endl;
	long int rk = -1;

	if (A.type_G == action_by_right_multiplication_t) {
		//action_by_right_multiplication *ABRM = A.G.ABRM;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		//ABRM->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_by_restriction_t) {
		induced_actions::action_by_restriction *ABR = A.G.ABR;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		int rk0;
		rk0 = sub->Group_element->rank_point(v);
		rk = ABR->restricted_point_idx(rk0);
		}
	else if (A.type_G == action_by_conjugation_t) {
		//action_by_conjugation *ABC = A.G.ABC;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		//ABC->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_determinant_t) {
		//action_on_determinant *AD = A.G.AD;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_galois_group_t) {
		//action_on_galois_group *AG = A.G.on_Galois_group;

		//ost << a;
		}
	else if (A.type_G == action_on_sign_t) {
		//action_on_sign *OnSign = A.G.OnSign;
		//ost << a;
		}
	else if (A.type_G == action_on_sets_t) {
		//action_on_sets *AOS = A.G.on_sets;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		//AOS->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_subgroups_t) {
		//action_on_subgroups *AOS = A.G.on_subgroups;
		//ost << a;
		//AOS->compute_image(sub, Elt, a, b, verbose_level);
		}
	else if (A.type_G == action_on_k_subsets_t) {
		//action_on_k_subsets *On_k_subsets = A.G.on_k_subsets;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_orbits_t) {
		//action_on_orbits *On_orbits = A.G.OnOrbits;
#if 0
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
#endif
		//ost << a;
		}
	else if (A.type_G == action_on_bricks_t) {
		//action_on_bricks *On_bricks = A.G.OnBricks;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_andre_t) {
		//action_on_andre *OnAndre = A.G.OnAndre;
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point no subaction" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == action_on_pairs_t) {
		action *sub;
		//int i, j;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point "
					"no subaction, type = action_on_pairs_t" << endl;
			exit(1);
			}
		//k2ij(a, i, j, sub->degree);
		//cout << "a={" << i << "," << j << "}";
		}
	else if (A.type_G == action_on_ordered_pairs_t) {
		action *sub;
		//int a2, swap, tmp, i, j;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point "
					"no subaction, type = action_on_ordered_pairs_t" << endl;
			exit(1);
			}
#if 0
		swap = a % 2;
		a2 = a / 2;
		k2ij(a2, i, j, sub->degree);
		if (swap) {
			tmp = i;
			i = j;
			j = tmp;
			}
		cout << "a=(" << i << "," << j << ")";
#endif
		}
	else if (A.type_G == base_change_t) {
		action *sub;
		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point "
					"no subaction, type = base_change_t" << endl;
			exit(1);
			}
		//ost << a;
		}
	else if (A.type_G == product_action_t) {
		//product_action *PA;

		//PA = A.G.product_action_data;
		//ost << a;
		}
	else if (A.type_G == action_on_grassmannian_t) {
		if (false) {
			cout << "action_on_grassmannian_t" << endl;
			}
		//action_on_grassmannian *AG = A.G.AG;

#if 0
		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point "
					"no subaction" << endl;
			exit(1);
			}
		//ost << a;
#endif
		//AG->print_point(a, ost);
		//b = AG->compute_image_int(sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_on_spread_set_t) {
		if (false) {
			cout << "action_on_spread_set_t" << endl;
			}
		//action_on_spread_set *AS = A.G.AS;

		//ost << a;
		}
	else if (A.type_G == action_on_orthogonal_t) {
		if (false) {
			cout << "action_on_orthogonal_t" << endl;
			}
		induced_actions::action_on_orthogonal *AO = A.G.AO;

		rk = AO->rank_point(v);
		}
	else if (A.type_G == action_on_wedge_product_t) {
		if (false) {
			cout << "induced_action_rank_point "
					"action_on_wedge_product_t" << endl;
			}
		induced_actions::action_on_wedge_product *AW = A.G.AW;

		action *sub;

		sub = A.subaction;
		if (sub == NULL) {
			cout << "induced_action_rank_point "
					"no subaction" << endl;
			exit(1);
			}
		rk = AW->rank_point(v);
		//b = AW->compute_image_int(*sub, Elt, a, verbose_level - 1);
		}
	else if (A.type_G == action_by_representation_t) {
		if (false) {
			cout << "induced_action_rank_point "
					"action_by_representation_t" << endl;
		}
		induced_actions::action_by_representation *Rep = A.G.Rep;

		rk = Rep->rank_point(v, 0 /* verbose_level*/);
		//b = AW->compute_image_int(*sub, Elt, a, verbose_level - 1);
	}
	else {
		cout << "induced_action_rank_point type_G unknown:: type_G = ";
		AG.action_print_symmetry_group_type(cout, A.type_G);
		cout << endl;
		exit(1);
		}

	return rk;
}




}}}




