// action.cpp
//
// Anton Betten
// July 8, 2003

#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {



action::action()
{
	null();
}

action::~action()
{
	freeself();
}

void action::null()
{
	label[0] = 0;
	label_tex[0] = 0;
	
	//user_data_type = 0;
	type_G = unknown_symmetry_group_t;
	
	subaction = NULL;
	f_has_strong_generators = FALSE;
	Strong_gens = NULL;
	//strong_generators = NULL;


	//transversal_reps = NULL;

	null_element_data();

	degree = 0;
	f_is_linear = FALSE;
	dimension = 0;

	f_has_stabilizer_chain = FALSE;

	Stabilizer_chain = NULL;



	elt_size_in_int = 0;
	coded_elt_size_in_char = 0;
	group_prefix[0] = 0;
	//f_has_transversal_reps = FALSE;
	f_group_order_is_small = FALSE;
	make_element_size = 0;
	low_level_point_size = 0;

	ptr = NULL;

	f_allocated = FALSE;
	f_has_subaction = FALSE;
	f_subaction_is_allocated = FALSE;
	f_has_sims = FALSE;
	f_has_kernel = FALSE;
};

void action::freeself()
{
	//int i;
	int f_v = FALSE;
	int f_vv = FALSE;

	if (f_v) {
		cout << "action::freeself deleting action " << label << endl;
		print_info();
		}
	if (f_allocated) {
		if (f_vv) {
			cout << "action::freeself freeing G of type ";
			action_print_symmetry_group_type(cout, type_G);
			cout << endl;
			}
		if (type_G == matrix_group_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.matrix_grp" << endl;
				cout << "G.matrix_grp=" << G.matrix_grp << endl;
				}
			FREE_OBJECT(G.matrix_grp);
			if (f_vv) {
				cout << "action::~action freeing G.matrix_grp finished"
						<< endl;
				}
			G.matrix_grp = NULL;
			}
		else if (type_G == wreath_product_t) {
			if (f_vv) {
				cout << "action::freeself freeing "
						"G.wreath_product_group" << endl;
				cout << "G.wreath_product_group="
						<< G.wreath_product_group << endl;
				}
			FREE_OBJECT(G.wreath_product_group);
			if (f_vv) {
				cout << "action::freeself freeing "
						"G.wreath_product_group finished" << endl;
				}
			G.wreath_product_group = NULL;
			}
		else if (type_G == perm_group_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.perm_group_t" << endl;
				}
			FREE_OBJECT(G.perm_grp);
			if (f_vv) {
				cout << "action::freeself freeing G.perm_group_t finished"
						<< endl;
				}
			G.perm_grp = NULL;
			}
		else if (type_G == action_on_sets_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.on_sets" << endl;
				cout << "G.on_sets=" << G.on_sets << endl;
				}
			FREE_OBJECT(G.on_sets);
			if (f_vv) {
				cout << "action::freeself freeing G.on_sets finished" << endl;
				}
			G.on_sets = NULL;
			}
		else if (type_G == action_on_set_partitions_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnSetPartitions" << endl;
				cout << "G.OnSetPartitions=" << G.OnSetPartitions << endl;
				}
			FREE_OBJECT(G.OnSetPartitions);
			if (f_vv) {
				cout << "action::freeself freeing G.OnSetPartitions finished" << endl;
				}
			G.OnSetPartitions = NULL;
			}
		else if (type_G == action_on_k_subsets_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.on_sets" << endl;
				cout << "G.on_k_subsets=" << G.on_k_subsets << endl;
				}
			FREE_OBJECT(G.on_k_subsets);
			if (f_vv) {
				cout << "action::freeself freeing G.on_k_subsets finished"
						<< endl;
				}
			G.on_k_subsets = NULL;
			}
		else if (type_G == action_on_orbits_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnOrbits" << endl;
				cout << "G.OnOrbits=" << G.OnOrbits << endl;
				}
			FREE_OBJECT(G.OnOrbits);
			if (f_vv) {
				cout << "action::freeself freeing G.OnOrbits finished" << endl;
				}
			G.OnOrbits = NULL;
			}
		else if (type_G == action_on_bricks_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnBricks" << endl;
				cout << "G.OnBricks=" << G.OnBricks << endl;
				}
			FREE_OBJECT(G.OnBricks);
			if (f_vv) {
				cout << "action::freeself freeing G.OnBricks finished" << endl;
				}
			G.OnBricks = NULL;
			}
		else if (type_G == action_on_andre_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnAndre" << endl;
				cout << "G.OnAndre=" << G.OnAndre << endl;
				}
			FREE_OBJECT(G.OnAndre);
			if (f_vv) {
				cout << "action::freeself freeing G.OnAndre finished" << endl;
				}
			G.OnAndre = NULL;
			}
		else if (type_G == action_by_right_multiplication_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.ABRM" << endl;
				}
			FREE_OBJECT(G.ABRM);
			G.ABRM = NULL;
			}
		else if (type_G == action_by_restriction_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.ABR" << endl;
				}
			FREE_OBJECT(G.ABR);
			G.ABR = NULL;
			}
		else if (type_G == action_by_conjugation_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.ABC" << endl;
				}
			FREE_OBJECT(G.ABC);
			G.ABC = NULL;
			}
		else if (type_G == action_by_representation_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.Rep" << endl;
				}
			FREE_OBJECT(G.Rep);
			G.Rep = NULL;
			}
		else if (type_G == action_on_determinant_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.AD" << endl;
				}
			FREE_OBJECT(G.AD);
			G.AD = NULL;
			}
		else if (type_G == action_on_galois_group_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.on_Galois_group" << endl;
				}
			FREE_OBJECT(G.on_Galois_group);
			G.on_Galois_group = NULL;
			}
		else if (type_G == action_on_sign_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnSign" << endl;
				}
			FREE_OBJECT(G.OnSign);
			G.OnSign = NULL;
			}
		else if (type_G == action_on_grassmannian_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.AG" << endl;
				}
			FREE_OBJECT(G.AG);
			G.AG = NULL;
			}
		else if (type_G == action_on_factor_space_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.AF" << endl;
				}
			FREE_OBJECT(G.AF);
			G.AF = NULL;
			}
		else if (type_G == action_on_wedge_product_t) {
			//FREE_OBJECT(G.AW);
			G.AW = NULL;
			}
		else if (type_G == action_on_homogeneous_polynomials_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnHP" << endl;
				}
			FREE_OBJECT(G.OnHP);
			G.OnHP = NULL;
			}
		else if (type_G == action_by_subfield_structure_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.SubfieldStructure" << endl;
				}
			FREE_OBJECT(G.SubfieldStructure);
			G.SubfieldStructure = NULL;
			}
		else {
			cout << "action::freeself don't know "
					"how to free the object; action type is ";
			print_symmetry_group_type(cout);
			cout << endl;
			exit(1);
			}
		f_allocated = FALSE;
		type_G = unknown_symmetry_group_t;
		}
	if (f_v) {
		cout << "action::freeself after freeing G " << endl;
		}

	if (Stabilizer_chain) {
		FREE_OBJECT(Stabilizer_chain);
	}

	if (f_v) {
		cout << "action::freeself after free_base_data" << endl;
		}
	
#if 0
	if (f_has_transversal_reps) {
		if (f_v) {
			cout << "we are freeing the transversal reps" << endl;
			}
		for (i = 0; i < base_len; i++) {
			FREE_int(transversal_reps[i]);
			}
		FREE_pint(transversal_reps);
		f_has_transversal_reps = FALSE;
		}
#endif
	
	if (f_v) {
		cout << "action::freeself after freeing transversal reps" << endl;
		}
		
	free_element_data();

	if (f_v) {
		cout << "action::freeself after free_element_data" << endl;
		}
	
	if (f_has_strong_generators) {
		if (f_v) {
			cout << "we are freeing strong generators" << endl;
			}
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
		//FREE_OBJECT(strong_generators); //delete strong_generators;
		//FREE_int(tl);
		//strong_generators = NULL;
		//tl = NULL;
		f_has_strong_generators = FALSE;
		}

	if (f_v) {
		cout << "action::freeself after freeing strong generators" << endl;
		}

	if (f_has_subaction && f_subaction_is_allocated) {
		if (f_v) {
			cout << "subaction is allocated, so we free it" << endl;
			subaction->print_info();
			}
		FREE_OBJECT(subaction);
		subaction = NULL;
		f_subaction_is_allocated = FALSE;
		f_has_subaction = FALSE;
		}

	if (f_v) {
		cout << "action::freeself after freeing subaction" << endl;
		}
	
	if (f_has_sims) {
		if (f_v) {
			cout << "action::freeself freeing Sims" << endl;
			}
		FREE_OBJECT(Sims);
		Sims = NULL;
		f_has_sims = FALSE;
		if (f_v) {
			cout << "action::freeself freeing Sims finished" << endl;
			}
		}

	if (f_v) {
		cout << "action::freeself after freeing sims" << endl;
		}

	if (f_has_kernel) {
		if (f_v) {
			cout << "action::freeself freeing Kernel" << endl;
			}
		FREE_OBJECT(Kernel);
		Kernel = NULL;
		f_has_kernel = FALSE;
		if (f_v) {
			cout << "action::freeself freeing Kernel finished" << endl;
			}
		}

	if (ptr) {
		FREE_OBJECT(ptr);
	}

	if (f_v) {
		cout << "action::freeself "
				"deleting action " << label << " done" << endl;
		}
}

int action::f_has_base()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_f_has_base();
	}
	else {
		return FALSE;
	}
}


int action::base_len()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_base_len();
	}
	else {
		cout << "action::base_len Stabilizer_chain == NULL" << endl;
		exit(1);
		//return 0;
	}
}

void action::set_base_len(int base_len)
{
	if (Stabilizer_chain) {
		Stabilizer_chain->get_base_len() = base_len;
	}
	else {
		cout << "action::set_base_len no stabilizer chain" << endl;
	}

}

int &action::base_i(int i)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->base_i(i);
	}
	else {
		cout << "action::base_i no Stabilizer_chain" << endl;
		exit(1);
	}
}

int *&action::get_base()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_base();
	}
	else {
		cout << "action::get_base no Stabilizer_chain" << endl;
		exit(1);
	}
}

int &action::transversal_length_i(int i)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->transversal_length_i(i);
	}
	else {
		cout << "action::transversal_length_i no Stabilizer_chain" << endl;
		exit(1);
	}
}

int *&action::get_transversal_length()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_transversal_length();
	}
	else {
		cout << "action::transversal_length no Stabilizer_chain" << endl;
		exit(1);
	}
}

int &action::orbit_ij(int i, int j)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->orbit_ij(i, j);
	}
	else {
		cout << "action::orbit_ij no Stabilizer_chain" << endl;
		exit(1);
	}
}

int &action::orbit_inv_ij(int i, int j)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->orbit_inv_ij(i, j);
	}
	else {
		cout << "action::orbit_inv_ij no Stabilizer_chain" << endl;
		exit(1);
	}
}



void action::null_element_data()
{
	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	element_rw_memory_object = NULL;
}

void action::allocate_element_data()
{
	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);
	Elt5 = NEW_int(elt_size_in_int);
	eltrk1 = NEW_int(elt_size_in_int);
	eltrk2 = NEW_int(elt_size_in_int);
	eltrk3 = NEW_int(elt_size_in_int);
	elt_mult_apply = NEW_int(elt_size_in_int);
	elt1 = NEW_uchar(coded_elt_size_in_char);
	element_rw_memory_object = NEW_char(coded_elt_size_in_char);
}

void action::free_element_data()
{
	if (Elt1) {
		FREE_int(Elt1);
		}
	if (Elt2) {
		FREE_int(Elt2);
		}
	if (Elt3) {
		FREE_int(Elt3);
		}
	if (Elt4) {
		FREE_int(Elt4);
		}
	if (Elt5) {
		FREE_int(Elt5);
		}
	if (eltrk1) {
		FREE_int(eltrk1);
		}
	if (eltrk2) {
		FREE_int(eltrk2);
		}
	if (eltrk3) {
		FREE_int(eltrk3);
		}
	if (elt_mult_apply) {
		FREE_int(elt_mult_apply);
		}
	if (elt1) {
		FREE_uchar(elt1);
		}
	if (element_rw_memory_object) {
		FREE_char(element_rw_memory_object);
		}
	null_element_data();
}



// #############################################################################


int action::find_non_fixed_point(void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "action::find_non_fixed_point" << endl;
		cout << "degree=" << degree << endl;
		}
	for (i = 0; i < degree; i++) {
		j = element_image_of(i, elt, verbose_level - 1);
		if (j != i) {
			if (f_v) {
				cout << "moves " << i << " to " << j << endl;
				}
			return i;
			}
		}
	if (f_v) {
		cout << "cannot find non fixed point" << endl;
		}
	return -1;
}

int action::find_fixed_points(void *elt,
		int *fixed_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, n = 0;
	
	if (f_v) {
		cout << "computing fixed points in action "
				<< label << " of degree " << degree << endl;
		}
	for (i = 0; i < degree; i++) {
		j = element_image_of(i, elt, 0);
		if (j == i) {
			fixed_points[n++] = i;
			}
		}
	if (f_v) {
		cout << "found " << n << " fixed points" << endl;
		}
	return n;
}

int action::test_if_set_stabilizes(int *Elt,
		int size, int *set, int verbose_level)
{
	int *set1, *set2;
	int i, cmp;
	int f_v = (verbose_level >= 1);
	sorting Sorting;

	if (f_v) {
		cout << "action::test_if_set_stabilizes" << endl;
		}
	set1 = NEW_int(size);
	set2 = NEW_int(size);
	for (i = 0; i < size; i++) {
		set1[i] = set[i];
		}
	Sorting.int_vec_quicksort_increasingly(set1, size);
	map_a_set(set1, set2, size, Elt, 0);
	Sorting.int_vec_quicksort_increasingly(set2, size);
	cmp = int_vec_compare(set1, set2, size);
	if (f_v) {
		cout << "the elements takes " << endl;
		int_vec_print(cout, set1, size);
		cout << endl << "to" << endl;
		int_vec_print(cout, set2, size);
		cout << endl;
		cout << "cmp = " << cmp << endl;
		}
	FREE_int(set1);
	FREE_int(set2);
	if (cmp == 0) {
		if (f_v) {
			cout << "action::test_if_set_stabilizes "
					"done, returning TRUE" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "action::test_if_set_stabilizes "
					"done, returning FALSE" << endl;
			}
		return FALSE;
		}
}

void action::map_a_set(int *set,
		int *image_set, int n, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "action::map_a_set" << endl;
		}
	if (f_vv) {
		cout << "group element:" << endl;
		element_print_quick(Elt, cout);
		cout << endl;
		cout << "set: " << endl;
		int_vec_print(cout, set, n);
		cout << endl;
		}
	for (i = 0; i < n; i++) {
		if (f_vv) {
			cout << "i=" << i << " computing image of " << set[i] << endl;
			}
		image_set[i] = element_image_of(set[i], Elt, verbose_level - 2);
		if (f_vv) {
			cout << "i=" << i << " image of "
					<< set[i] << " is " << image_set[i] << endl;
			}
		}
}

void action::map_a_set_and_reorder(int *set,
		int *image_set, int n, int *Elt, int verbose_level)
{
	sorting Sorting;

	map_a_set(set, image_set, n, Elt, verbose_level);
	Sorting.int_vec_heapsort(image_set, n);
}



void action::init_sims(sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, k;

	if (f_v) {
		cout << "action::init_sims action " << label
				<< " base_len = " << base_len() << endl;
		}
	if (f_has_sims) {
		FREE_OBJECT(Sims);
		Sims = NULL;
		f_has_sims = FALSE;
		}
	if (G->A != this) {
		cout << "action::init_sims action " << label
				<< " sims object has different action "
				<< G->A->label << endl;
		exit(1);
		}
	Sims = G;
	f_has_sims = TRUE;
	Stabilizer_chain->init_base_from_sims(G, verbose_level);

	compute_strong_generators_from_sims(0/*verbose_level - 2*/);
#if 0
	f_has_strong_generators = TRUE;
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(G, 0);
#endif
	
	if (f_v) {
		cout << "action::init_sims done" << endl;
		}
}

int action::element_has_order_two(int *E1,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "action::element_has_order_two" << endl;
		}

	element_mult(E1, E1, Elt1, 0);
	if (is_one(Elt1)) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	
	if (f_v) {
		cout << "action::element_has_order_two done" << endl;
		}
	return ret;
}

int action::product_has_order_two(int *E1,
		int *E2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "action::product_has_order_two" << endl;
		}

	element_mult(E1, E2, Elt1, 0);
	element_mult(Elt1, Elt1, Elt2, 0);
	if (is_one(Elt2)) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	
	if (f_v) {
		cout << "action::product_has_order_two done" << endl;
		}
	return ret;
}

int action::product_has_order_three(int *E1,
		int *E2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "action::product_has_order_three" << endl;
		}

	element_mult(E1, E2, Elt1, 0);
	element_mult(Elt1, Elt1, Elt2, 0);
	element_mult(Elt2, Elt1, Elt3, 0);
	if (is_one(Elt3)) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	
	if (f_v) {
		cout << "action::product_has_order_three done" << endl;
		}
	return ret;
}

int action::element_order(void *elt)
{
	int *cycle_type;
	int order;

	cycle_type = NEW_int(degree);
	order = element_order_and_cycle_type_verbose(
			elt, cycle_type, 0);
	FREE_int(cycle_type);
	return order;
}

int action::element_order_and_cycle_type(
		void *elt, int *cycle_type)
{
	return element_order_and_cycle_type_verbose(
			elt, cycle_type, 0);
}

int action::element_order_and_cycle_type_verbose(
		void *elt, int *cycle_type, int verbose_level)
// cycle_type[i - 1] is the number of cycle of length i for 1 le i le n
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *have_seen;
	int l, l1, first, next, len, g, n, order = 1;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "action::element_order_verbose" << endl;
		}
	if (f_vv) {
		cout << "The element is:" << endl;
		element_print_quick(elt, cout);
		cout << "as permutation:" << endl;
		element_print_as_permutation(elt, cout);
		}
	n = degree;
	int_vec_zero(cycle_type, degree);
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
			}
		// work on cycle, starting with l:
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1] = TRUE;
			next = element_image_of(l1, elt, 0);
			if (next > n) {
				cout << "action::element_order_verbose: next = "
					<< next << " > n = " << n << endl;
				// print_list(ost);
				exit(1);
				}
			if (next == first) {
				break;
				}
			if (have_seen[next]) {
				cout << "action::element_order_verbose "
						"have_seen[next]" << endl;
				exit(1);
				}
			l1 = next;
			len++;
			}
		cycle_type[len - 1]++;
		if (len == 1) {
			continue;
			}
		g = NT.gcd_int(len, order);
		order *= len / g;
		}
	FREE_int(have_seen);
	if (f_v) {
		cout << "action::element_order_verbose "
				"done order=" << order << endl;
		}
	return order;
}

int action::element_order_if_divisor_of(void *elt, int o)
// returns the order of the element if o == 0
// if o != 0, returns the order of the element provided it divides o,
// 0 otherwise.
{
	int *have_seen;
	int l, l1, first, next, len, g, n, order = 1;
	number_theory_domain NT;
	
	n = degree;
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
			}
		// work on cycle, starting with l: 
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1] = TRUE;
			next = element_image_of(l1, elt, 0);
			if (next > n) {
				cout << "perm_print(): next = " 
					<< next << " > n = " << n << endl;
				// print_list(ost);
				exit(1);
				}
			if (next == first) {
				break;
				}
			if (have_seen[next]) {
				cout << "action::element_order_if_divisor_of(): "
						"have_seen[next]" << endl;
				exit(1);
				}
			l1 = next;
			len++;
			}
		if (len == 1)
			continue;
		if (o && (o % len)) {
			FREE_int(have_seen);
			return 0;
			}
		g = NT.gcd_int(len, order);
		order *= len / g;
		}
	FREE_int(have_seen);
	return order;
}

void action::compute_all_point_orbits(schreier &S,
		vector_ge &gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_all_point_orbits" << endl;
		}
	S.init(this, verbose_level - 2);
	S.init_generators(gens, verbose_level - 2);
	S.compute_all_point_orbits(verbose_level - 1);
	if (f_v) {
		cout << "action::compute_all_point_orbits done" << endl;
		}
}

int action::depth_in_stab_chain(int *Elt)
// the index of the first moved base point
{
	int i, j, b;
	
	for (i = 0; i < base_len(); i++) {
		b = base_i(i);
		j = element_image_of(b, Elt, 0);
		if (j != b)
			return i;
		}
	return base_len();
}

void action::strong_generators_at_depth(int depth,
		vector_ge &gen, int verbose_level)
// all strong generators that leave base points 0,..., depth - 1 fix
{
	int i, j, l, n;
	
	l = Strong_gens->gens->len;
	gen.init(this, verbose_level - 2);
	gen.allocate(l, verbose_level - 2);
	n = 0;
	for (i = 0; i < l; i++) {
		j = depth_in_stab_chain(Strong_gens->gens->ith(i));
		if (j >= depth) {
			gen.copy_in(n, Strong_gens->gens->ith(i));
			n++;
			}
		}
	gen.len = n;
}

void action::compute_point_stabilizer_chain(vector_ge &gen, 
	sims *S, int *sequence, int len, int verbose_level)
// S points to len + 1 many sims objects
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;
	
	if (f_v) {
		cout << "action::compute_point_stabilizer_chain for sequence ";
		int_vec_print(cout, sequence, len);
		cout << endl;
		}
	for (i = 0; i <= len; i++) {
		S[i].init(this, verbose_level - 2);
		}
	S[0].init_generators(gen, 0);
	S[0].compute_base_orbits(0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "automorphism group has order ";
		S[0].print_group_order(cout);
		cout << endl;
		if (f_vvv) {
			cout << "generators:" << endl;
			S[0].print_generators();
			}
		}
	
	for (i = 0; i < len; i++) {
		if (f_vv) {
			cout << "computing stabilizer of " << i 
				<< "-th point in the sequence" << endl;
			}
		S[i].point_stabilizer_stabchain_with_action(this, 
			S[i + 1], sequence[i], 0 /*verbose_level - 2*/);
		if (f_vv) {
			cout << "stabilizer of " << i << "-th point "
					<< sequence[i] << " has order ";
			S[i + 1].print_group_order(cout);
			cout << endl;
			if (f_vvv) {
				cout << "generators:" << endl;
				S[i + 1].print_generators();
				}
			}
		}
	if (f_v) {
		cout << "action::compute_point_stabilizer_chain for sequence ";
		int_vec_print(cout, sequence, len);
		cout << " finished" << endl;
		cout << "i : order of i-th stabilizer" << endl;
		for (i = 0; i <= len; i++) {
			cout << i << " : ";
			S[i].print_group_order(cout);
			cout << endl;
			}
		if (f_vv) {
			for (i = 0; i <= len; i++) {
				cout << i << " : ";
				cout << "generators:" << endl;
				S[i].print_generators();
				}
			}
		}
}

int action::compute_orbit_of_point(vector_ge &strong_generators,
		int pt, int *orbit, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier Schreier;
	int len, i, f;
	
	if (f_v) {
		cout << "action::compute_orbit_of_point: "
				"computing orbit of point " << pt << endl;
		}
	Schreier.init(this, verbose_level - 2);
	Schreier.init_generators(strong_generators, verbose_level - 2);
	Schreier.compute_point_orbit(pt, 0);
	f = Schreier.orbit_first[0];
	len = Schreier.orbit_len[0];
	for (i = 0; i < len; i++) {
		orbit[i] = Schreier.orbit[f + i];
		}
	return len;
}

int action::compute_orbit_of_point_generators_by_handle(int nb_gen, 
	int *gen_handle, int pt, int *orbit, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	vector_ge gens;
	int i;
	
	gens.init(this, verbose_level - 2);
	gens.allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		element_retrieve(gen_handle[i], gens.ith(i), 0);
		}
	return compute_orbit_of_point(gens, pt, orbit, verbose_level);
}


int action::least_image_of_point(vector_ge &strong_generators,
	int pt, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier Schreier;
	int len, image, pos, i;
	
	if (f_v) {
		cout << "action::least_image_of_point: "
				"computing least image of " << pt << endl;
		}
	Schreier.init(this, verbose_level - 2);
	Schreier.init_generators(strong_generators, verbose_level - 2);
	Schreier.compute_point_orbit(pt, 0);
	len = Schreier.orbit_len[0];
	image = int_vec_minimum(Schreier.orbit, len);
	pos = Schreier.orbit_inv[image];
	Schreier.coset_rep(pos);
	element_move(Schreier.cosetrep, transporter, 0);
	// we check it:
	i = element_image_of(pt, transporter, 0);
	if (i != image) {
		cout << "action::least_image_of_point i != image" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action::least_image_of_point: "
				"least image of " << pt << " is " << image << endl;
		}
	return image;
}

int action::least_image_of_point_generators_by_handle(
	int nb_gen, int *gen_handle,
	int pt, int *transporter, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	vector_ge gens;
	int i;
	
	if (nb_gen == 0) {
		element_one(transporter, 0);
		return pt;
		}
	gens.init(this, verbose_level - 2);
	gens.allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		element_retrieve(gen_handle[i], gens.ith(i), 0);
		}
	return least_image_of_point(gens, pt, transporter, verbose_level);
}

void action::all_point_orbits(schreier &Schreier, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::all_point_orbits" << endl;
		}
	Schreier.init(this, verbose_level - 2);
	if (!f_has_strong_generators) {
		cout << "action::all_point_orbits !f_has_strong_generators" << endl;
		exit(1);
		}
	Schreier.init_generators(*Strong_gens->gens /* *strong_generators */, verbose_level - 2);
	Schreier.compute_all_point_orbits(verbose_level);
}

void action::all_point_orbits_from_generators(schreier &Schreier,
		strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::all_point_orbits_from_generators" << endl;
		}
	Schreier.init(this, verbose_level - 2);
	Schreier.init_generators(*SG->gens /* *strong_generators */, verbose_level - 2);
	Schreier.compute_all_point_orbits(verbose_level);
}

void action::all_point_orbits_from_single_generator(schreier &Schreier,
		int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::all_point_orbits_from_single_generator" << endl;
		}
	vector_ge gens;

	gens.init(this, verbose_level - 2);
	gens.allocate(1, verbose_level - 2);
	element_move(Elt, gens.ith(0), 0);

	Schreier.init(this, verbose_level - 2);
	Schreier.init_generators(gens, verbose_level - 2);
	Schreier.compute_all_point_orbits(verbose_level);
	if (f_v) {
		cout << "action::all_point_orbits_from_single_generator done" << endl;
		}
}

void action::compute_stabilizer_orbits(partitionstack *&Staborbits,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int i;
	vector_ge gen;
	
	if (f_v) {
		cout << "action::compute_stabilizer_orbits" << endl;
		cout << "base_len = " << base_len() << endl;
		for (i = 0; i < base_len(); i++) {
			cout << i << " : " << base_i(i) << " : " << transversal_length_i(i);
			//int_vec_print(cout, Stabilizer_chain->orbit[i], Stabilizer_chain->transversal_length[i]);
			cout << endl;
			}
		cout << "degree = " << degree << endl;
		}
	Staborbits = NEW_OBJECTS(partitionstack, base_len());
		// where is this freed???

	for (i = 0; i < base_len(); i++) {
		strong_generators_at_depth(i, gen, verbose_level - 2);
		if (FALSE) {
			cout << "level " << i << " found "
					<< gen.len << " strong generators" << endl;
			}
		if (FALSE) {
			gen.print(cout);
			}

		partitionstack *S;
		schreier Schreier;


		S = &Staborbits[i];
		S->allocate(degree, FALSE);
	
		if (FALSE) {
			cout << "computing point orbits" << endl;
			}
			
		compute_all_point_orbits(Schreier, gen, 0 /*verbose_level - 2*/);
		
		if (FALSE) {
			Schreier.print(cout);
			}
		
		Schreier.get_orbit_partition(*S, 0 /*verbose_level - 2*/);
		if (FALSE) {
			cout << "found " << S->ht << " orbits" << endl;
			}
		if (f_vv) {
			cout << "level " << i << " with "
					<< gen.len << " strong generators : ";
			//cout << "orbit partition at level " << i << ":" << endl;
			cout << *S;
			}

		}
	if (f_v) {
		cout << "action::compute_stabilizer_orbits finished" << endl;
		}
}


int action::check_if_in_set_stabilizer(int *Elt,
		int size, int *set, int verbose_level)
{
	int i, a, b, idx;
	int *ordered_set;
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	
	ordered_set = NEW_int(size);
	for (i = 0; i < size; i++) {
		ordered_set[i] = set[i];
		}
	Sorting.int_vec_sort(size, ordered_set);
	for (i = 0; i < size; i++) {
		a = ordered_set[i];
		b = element_image_of(a, Elt, 0);
		if (!Sorting.int_vec_search(ordered_set, size, b, idx)) {
			if (f_v) {
				cout << "action::check_if_in_set_stabilizer fails" << endl;
				cout << "set: ";
				int_vec_print(cout, set, size);
				cout << endl;
				cout << "ordered_set: ";
				int_vec_print(cout, ordered_set, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
				}
			FREE_int(ordered_set);
			return FALSE;
			}
		}
	FREE_int(ordered_set);
	return TRUE;
	
}

int action::check_if_transporter_for_set(int *Elt,
		int size, int *set1, int *set2, int verbose_level)
{
	int i, a, b, idx;
	int *ordered_set2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 4);
	sorting Sorting;
	
	if (f_vv) {
		cout << "action::check_if_transporter_for_set "
				"size=" << size << endl;
		int_vec_print(cout, set1, size);
		cout << endl;
		int_vec_print(cout, set2, size);
		cout << endl;
		element_print(Elt, cout);
		cout << endl;
		}
	ordered_set2 = NEW_int(size);
	for (i = 0; i < size; i++) {
		ordered_set2[i] = set2[i];
		}
	Sorting.int_vec_sort(size, ordered_set2);
	if (f_vv) {
		cout << "sorted target set:" << endl;
		int_vec_print(cout, ordered_set2, size);
		cout << endl;
		}
	for (i = 0; i < size; i++) {
		a = set1[i];
		if (FALSE) {
			cout << "i=" << i << " a=" << a << endl;
			}
		b = element_image_of(a, Elt, 0);
		if (FALSE) {
			cout << "i=" << i << " a=" << a << " b=" << b << endl;
			}
		if (!Sorting.int_vec_search(ordered_set2, size, b, idx)) {
			if (f_v) {
				cout << "action::check_if_transporter_for_set fails" << endl;
				cout << "set1   : ";
				int_vec_print(cout, set1, size);
				cout << endl;
				cout << "set2   : ";
				int_vec_print(cout, set2, size);
				cout << endl;
				cout << "ordered: ";
				int_vec_print(cout, ordered_set2, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
				}
			FREE_int(ordered_set2);
			return FALSE;
			}
		}
	FREE_int(ordered_set2);
	return TRUE;
	
}

void action::compute_set_orbit(vector_ge &gens,
	int size, int *set,
	int &nb_sets, int **&Sets, int **&Transporter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *image_set;
	int **New_Sets;
	int **New_Transporter;
	int nb_finished, allocated_nb_sets;
	int new_allocated_nb_sets, nb_gens, i, j, h;
	sorting Sorting;
	
	if (f_v) {
		cout << "action::compute_set_orbit: ";
		int_vec_print(cout, set, size);
		cout << endl;
		}
	nb_gens = gens.len;
	
	allocated_nb_sets = 100;
	Sets = NEW_pint(allocated_nb_sets);
	Transporter = NEW_pint(allocated_nb_sets);
	nb_sets = 0;

	image_set = NEW_int(size);
	Sets[0] = NEW_int(size);
	for (i = 0; i < size; i++) {
		Sets[0][i] = set[i];
		}
	Sorting.int_vec_sort(size, Sets[0]);
	
	Transporter[0] = NEW_int(elt_size_in_int);
	element_one(Transporter[0], FALSE);

	nb_sets = 1;
	nb_finished = 0;

	while (nb_finished < nb_sets) {
		if (f_v) {
			cout << "nb_finished=" << nb_finished
					<< " nb_sets=" << nb_sets << endl;
			}
		for (i = 0; i < nb_gens; i++) {
			map_a_set_and_reorder(Sets[nb_finished], image_set, size, 
				gens.ith(i), 0);
			if (FALSE) {
				cout << "image under generator " << i << ":";
				int_vec_print(cout, image_set, size);
				cout << endl;
				}
			for (j = 0; j < nb_sets; j++) {
				if (int_vec_compare(Sets[j], image_set, size) == 0)
					break;
				}
			if (j < nb_sets) {
				continue;
				}
			// n e w set found:
			if (f_v) {
				cout << "n e w set " << nb_sets << ":";
				int_vec_print(cout, image_set, size);
				cout << endl;
				}
			Sets[nb_sets] = image_set;
			image_set = NEW_int(size);
			Transporter[nb_sets] = NEW_int(elt_size_in_int);
			element_mult(Transporter[nb_finished],
					gens.ith(i), Transporter[nb_sets], 0);
			nb_sets++;
			if (nb_sets == allocated_nb_sets) {
				new_allocated_nb_sets = allocated_nb_sets + 100;
				cout << "reallocating to size "
						<< new_allocated_nb_sets << endl;
				New_Sets = NEW_pint(new_allocated_nb_sets);
				New_Transporter = NEW_pint(new_allocated_nb_sets);
				for (h = 0; h < nb_sets; h++) {
					New_Sets[h] = Sets[h];
					New_Transporter[h] = Transporter[h];
					}
				FREE_pint(Sets);
				FREE_pint(Transporter);
				Sets = New_Sets;
				Transporter = New_Transporter;
				allocated_nb_sets = new_allocated_nb_sets;
				}
			} // next i
		 nb_finished++;
		}
	FREE_int(image_set);
	if (f_v) {
		cout << "action::compute_set_orbit "
				"found an orbit of size " << nb_sets << endl;
		for (i = 0; i < nb_sets; i++) {
			cout << i << " : ";
			int_vec_print(cout, Sets[i], size);
			cout << endl;
			element_print(Transporter[i], cout);
			}
		}
}

void action::delete_set_orbit(int nb_sets, int **Sets, int **Transporter)
{
	int i;
	
	for (i = 0; i < nb_sets; i++) {
		FREE_int(Sets[i]);
		FREE_int(Transporter[i]);
		}
	FREE_pint(Sets);
	FREE_pint(Transporter);
}

void action::compute_minimal_set(vector_ge &gens, int size, int *set, 
	int *minimal_set, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int **Sets;
	int **Transporter;
	int nb_sets, i;
	int min_set;
	
	if (f_v) {
		cout << "action::compute_minimal_set" << endl;
		}
	

	compute_set_orbit(gens, size, set, 
		nb_sets, Sets, Transporter, verbose_level);
	
	min_set = 0;
	for (i = 1; i < nb_sets; i++) {
		if (int_vec_compare(Sets[i], Sets[min_set], size) < 0) {
			min_set = i;
			}
		}
	for (i = 0; i < size; i++) {
		minimal_set[i] = Sets[min_set][i];
		}
	element_move(Transporter[min_set], transporter, 0);
	delete_set_orbit(nb_sets, Sets, Transporter);
}

void action::find_strong_generators_at_level(
	int base_len, int *the_base, int level,
	vector_ge &gens, vector_ge &subset_of_gens,
	int verbose_level)
{
	int nb_generators_found;
	int *gen_idx;
	int nb_gens, i, j, bj, bj_image;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "action::find_strong_generators_at_level "
				"level=" << level << " base: ";
		int_vec_print(cout, the_base, base_len);
		cout << endl;
		}
	nb_gens = gens.len;
	gen_idx = NEW_int(gens.len);
	
	nb_generators_found = 0;
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < level; j++) {
			bj = the_base[j];
			bj_image = element_image_of(bj, gens.ith(i), 0);
			if (bj_image != bj)
				break;
			}
		if (j == level) {
			gen_idx[nb_generators_found++] = i;
			}
		}
	subset_of_gens.init(this, verbose_level - 2);
	subset_of_gens.allocate(nb_generators_found, verbose_level - 2);
	for (i = 0; i < nb_generators_found; i++) {
		j = gen_idx[i];
		element_move(gens.ith(j), subset_of_gens.ith(i), 0);
		}
	FREE_int(gen_idx);
	if (f_v) {
		cout << "action::find_strong_generators_at_level found " 
			<< nb_generators_found << " strong generators" << endl;
		if (f_vv) {
			subset_of_gens.print(cout);
			cout << endl;
			}
		}
}

void action::compute_strong_generators_from_sims(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::compute_strong_generators_from_sims" << endl;
		}
	if (!f_has_sims) {
		cout << "action::compute_strong_generators_from_sims need sims" << endl;
		exit(1);
		}
	if (f_has_strong_generators) {
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
		f_has_strong_generators = FALSE;
		}
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(Sims, verbose_level - 2);
	f_has_strong_generators = TRUE;
	if (f_v) {
		cout << "action::compute_strong_generators_from_sims done" << endl;
		}
}

void action::make_element_from_permutation_representation(
		int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *base_image;
	int i, a;
	
	if (f_v) {
		cout << "action::make_element_from_permutation_representation" << endl;
		}
	base_image = NEW_int(base_len());
	for (i = 0; i < base_len(); i++) {
		a = base_i(i);
		base_image[i] = data[a];
		if (base_image[i] >= degree) {
			cout << "action::make_element_from_permutation_representation "
					"base_image[i] >= degree" << endl;
			cout << "i=" << i << " base[i] = " << a
					<< " base_image[i]=" << base_image[i] << endl;
			exit(1);
			}
		}
	make_element_from_base_image(Elt, base_image, verbose_level);

	FREE_int(base_image);
	if (f_v) {
		cout << "action::make_element_from_permutation_representation done"
				<< endl;
		}
}

void action::make_element_from_base_image(int *Elt,
		int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *base_image;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *Elt5;
	sims *S;
#if 1
	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = TRUE;
	int f_print_cycles_of_length_one = FALSE;
#endif

	int i, j, yi, z, b, c, b_pt;

	if (f_v) {
		cout << "action::make_element_from_base_image" << endl;
		cout << "base images: ";
		int_vec_print(cout, data, base_len());
		cout << endl;
		print_info();
		}
	if (!f_has_sims) {
		cout << "action::make_element_from_base_image "
				"fatal: does not have sims" << endl;
		exit(1);
		}
	S = Sims;
	if (f_v) {
		cout << "action in Sims:" << endl;
		S->A->print_info();
		}
	base_image = NEW_int(base_len());
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);
	Elt5 = NEW_int(elt_size_in_int);
	for (j = 0; j < base_len(); j++) {
		base_image[j] = data[j];
		}
	element_one(Elt3, 0);
	
	for (i = 0; i < base_len(); i++) {
		element_invert(Elt3, Elt4, 0);
		b_pt = base_i(i);
		yi = base_image[i];
		z = element_image_of(yi, Elt4, 0);
		j = S->get_orbit_inv(i, z);
		//j = S->orbit_inv[i][z];
		if (f_vv) {
			cout << "i=" << i << endl;
			cout << "Elt3=" << endl;
			element_print_quick(Elt3, cout);
			element_print_as_permutation_with_offset(Elt3, cout, 
				offset, f_do_it_anyway_even_for_big_degree, 
				f_print_cycles_of_length_one, 0/*verbose_level*/);
			cout << "i=" << i << " b_pt=" << b_pt
					<< " yi=" << yi << " z="
					<< z << " j=" << j << endl;
			}
		S->coset_rep(Elt5, i, j, 0);
		if (f_vv) {
			cout << "cosetrep=" << endl;
			element_print_quick(Elt5, cout);
			element_print_base_images(Elt5);
			element_print_as_permutation_with_offset(Elt5, cout,
				offset, f_do_it_anyway_even_for_big_degree, 
				f_print_cycles_of_length_one, 0/*verbose_level*/);
			}
		element_mult(Elt5, Elt3, Elt4, 0);
		element_move(Elt4, Elt3, 0);

		if (f_vv) {
			cout << "after left multiplying, Elt3=" << endl;
			element_print_quick(Elt3, cout);
			element_print_as_permutation_with_offset(Elt3, cout, 
				offset, f_do_it_anyway_even_for_big_degree, 
				f_print_cycles_of_length_one, 0/*verbose_level*/);

			cout << "computing image of b_pt=" << b_pt << endl;
			}
		
		c = element_image_of(b_pt, Elt3, 0);
		if (f_vv) {
			cout << "b_pt=" << b_pt << " -> " << c << endl;
			}
		if (c != yi) {
			cout << "action::make_element_from_base_image "
					"fatal: element_image_of(b_pt, Elt3, 0) "
					"!= yi" << endl;
			exit(1);
			}
		}
	element_move(Elt3, Elt, 0);
	for (i = 0; i < base_len(); i++) {
		yi = data[i];
		b = element_image_of(base_i(i), Elt, 0);
		if (yi != b) {
			cout << "action::make_element_from_base_image "
					"fatal: yi != b"
					<< endl;
			cout << "i=" << i << endl;
			cout << "base[i]=" << base_i(i) << endl;
			cout << "yi=" << yi << endl;
			cout << "b=" << b << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "action::make_element_from_base_image "
				"created element:" << endl;
		element_print_quick(Elt, cout);
		}
	FREE_int(base_image);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(Elt5);
}

void action::make_element_2x2(int *Elt, int a0, int a1, int a2, int a3)
{
	int data[4];
	
	data[0] = a0;
	data[1] = a1;
	data[2] = a2;
	data[3] = a3;
	make_element(Elt, data, 0);
}

void action::make_element_from_string(int *Elt,
		const char *data_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::make_element_from_string" << endl;
		}
	int *data;
	int data_len;

	int_vec_scan(data_string, data, data_len);

	if (f_v) {
		cout << "action::make_element_from_string data = ";
		int_vec_print(cout, data, data_len);
		cout << endl;
		}

	make_element(Elt, data, verbose_level);

	FREE_int(data);

	if (f_v) {
		cout << "action::make_element_from_string Elt = " << endl;
		element_print_quick(Elt, cout);
		}

	if (f_v) {
		cout << "action::make_element_from_string done" << endl;
		}
}

void action::make_element(int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::make_element" << endl;
		}
	if (type_G == product_action_t) {

		if (f_v) {
			cout << "action::make_element product_action_t" << endl;
			}

		product_action *PA;
		
		PA = G.product_action_data;
		PA->make_element(Elt, data, verbose_level);
		//PA->A1->make_element(Elt, data, verbose_level);
		//PA->A2->make_element(Elt + PA->A1->elt_size_in_int, 
		//	data + PA->A1->make_element_size, verbose_level);
		}
	else if (type_G == action_on_sets_t) {
		if (f_v) {
			cout << "action::make_element action_on_sets_t" << endl;
			}
		subaction->make_element(Elt, data, verbose_level);
		}
	else if (type_G == action_on_pairs_t) {
		if (f_v) {
			cout << "action::make_element action_on_pairs_t" << endl;
			}
		subaction->make_element(Elt, data, verbose_level);
		}
	else if (type_G == matrix_group_t) {
		if (f_v) {
			cout << "action::make_element matrix_group_t" << endl;
			}
		G.matrix_grp->make_element(Elt, data, verbose_level);
		}
	else if (type_G == wreath_product_t) {
		if (f_v) {
			cout << "action::make_element wreath_product_t" << endl;
			}
		G.wreath_product_group->make_element(Elt, data, verbose_level);
		}
	else if (type_G == direct_product_t) {
		if (f_v) {
			cout << "action::make_element direct_product_t" << endl;
			}
		G.direct_product_group->make_element(Elt, data, verbose_level);
		}
	else if (f_has_subaction) {
		if (f_v) {
			cout << "action::make_element subaction" << endl;
			}
		subaction->make_element(Elt, data, verbose_level);
		}
	else if (type_G == perm_group_t) {
		if (f_v) {
			cout << "action::make_element perm_group_t" << endl;
			}
		G.perm_grp->make_element(Elt, data, verbose_level);
		}
	else {
		cout << "action::make_element unknown type_G: ";
		print_symmetry_group_type(cout);
		cout << endl;
		exit(1);
		}
}

void action::build_up_automorphism_group_from_aut_data(
	int nb_auts, int *aut_data,
	sims &S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, coset;
	int *Elt1, *Elt2;
	longinteger_object go;
	
	if (f_v) {
		cout << "action::build_up_automorphism_group_from_aut_data "
				"action=" << label << " nb_auts=" << nb_auts << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	S.init(this, verbose_level - 2);
	S.init_trivial_group(verbose_level - 1);
	for (h = 0; h < nb_auts; h++) {
		if (f_v) {
			cout << "aut_data[" << h << "]=";
			int_vec_print(cout, aut_data + h * base_len(), base_len());
			cout << endl;
			}
		for (i = 0; i < base_len(); i++) {
			coset = aut_data[h * base_len() + i];
			//image_point = Sims->orbit[i][coset];
			Sims->path[i] = coset;
				//Sims->orbit_inv[i][aut_data[h * base_len + i]];
			}
		if (f_v) {
			cout << "path=";
			int_vec_print(cout, Sims->path, base_len());
			cout << endl;
			}
		Sims->element_from_path_inv(Elt1);
		if (S.strip_and_add(Elt1, Elt2, 0/*verbose_level*/)) {
			S.group_order(go);
			if (f_v) {
				cout << "generator " << h
						<< " added, n e w group order " << go << endl;
				S.print_transversal_lengths();
				S.print_transversals_short();
				}
			}
		else {
			if (f_v) {
				cout << "generator " << h << " strips through" << endl;
				}
			}
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
}

void action::element_power_int_in_place(int *Elt,
		int n, int verbose_level)
{
	int *Elt2;
	int *Elt3;
	int *Elt4;
	
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);
	move(Elt, Elt2);
	one(Elt3);
	while (n) {
		if (ODD(n)) {
			mult(Elt2, Elt3, Elt4);
			move(Elt4, Elt3);
			}
		mult(Elt2, Elt2, Elt4);
		move(Elt4, Elt2);
		n >>= 1;
		}
	move(Elt3, Elt);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
}

void action::word_in_ab(int *Elt1, int *Elt2, int *Elt3,
		const char *word, int verbose_level)
{
	int *Elt4;
	int *Elt5;
	int l, i;
	

	Elt4 = NEW_int(elt_size_in_int);
	Elt5 = NEW_int(elt_size_in_int);
	one(Elt4);
	l = strlen(word);
	for (i = 0; i < l; i++) {
		if (word[i] == 'a') {
			mult(Elt4, Elt1, Elt5);
			move(Elt5, Elt4);
			}
		else if (word[i] == 'b') {
			mult(Elt4, Elt2, Elt5);
			move(Elt5, Elt4);
			}
		else {
			cout << "word must consist of a and b" << endl;
			exit(1);
			}
		}
	move(Elt4, Elt3);
	
	FREE_int(Elt4);
	FREE_int(Elt5);
}

void action::init_group_from_generators(
	int *group_generator_data, int group_generator_size,
	int f_group_order_target, const char *group_order_target, 
	vector_ge *gens, strong_generators *&Strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object go, cur_go;
	sims S;
	int *Elt;
	int nb_gens, i;
	int nb_times = 200;

	if (f_v) {
		cout << "action::init_group_from_generators" << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		}
	if (f_group_order_target) {
		cout << "group_order_target=" << group_order_target << endl;
		}
	go.create_from_base_10_string(group_order_target, 0);
	if (f_group_order_target) {
		cout << "group_order_target=" << go << endl;
		}
	S.init(this, verbose_level - 2);
	Elt = NEW_int(elt_size_in_int);
	nb_gens = group_generator_size / make_element_size;
	if (nb_gens * make_element_size != group_generator_size) {
		cout << "action::init_group_from_generators fatal: "
				"group_generator_size is not "
				"divisible by make_element_size"
				<< endl;
		cout << "make_element_size=" << make_element_size << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		exit(1);
		}
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "parsing generator " << i << ":" << endl;
			}
		int_vec_print(cout, group_generator_data + 
			i * make_element_size, make_element_size);
		cout << endl;
		make_element(Elt, 
			group_generator_data + i * make_element_size, verbose_level - 2);
		element_move(Elt, gens->ith(i), 0);
		}
	if (f_v) {
		cout << "done parsing generators" << endl;
		}
	S.init_trivial_group(verbose_level);
	S.init_generators(*gens, verbose_level);
	S.compute_base_orbits(verbose_level);
	while (TRUE) {
		S.closure_group(nb_times, 0/*verbose_level*/);
		S.group_order(cur_go);
		cout << "cur_go=" << cur_go << endl;
		if (!f_group_order_target)
			break;
		if (D.compare(cur_go, go) == 0) {
			cout << "reached target group order" << endl;
			break;
			}
		cout << "did not reach target group order, continuing" << endl;
		}

	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(&S, verbose_level - 1);

	FREE_int(Elt);
}

void action::init_group_from_generators_by_base_images(
	int *group_generator_data, int group_generator_size, 
	int f_group_order_target, const char *group_order_target, 
	vector_ge *gens, strong_generators *&Strong_gens_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object go, cur_go;
	sims S;
	int *Elt;
	int nb_gens, i;
	int nb_times = 200;

	if (f_v) {
		cout << "action::init_group_from_generators_by_base_images" << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		}
	if (f_group_order_target) {
		cout << "group_order_target=" << group_order_target << endl;
		go.create_from_base_10_string(group_order_target, 0);
		}
	if (f_group_order_target) {
		cout << "group_order_target=" << go << endl;
		}
	S.init(this, verbose_level - 2);
	Elt = NEW_int(elt_size_in_int);
	nb_gens = group_generator_size / base_len();
	if (f_v) {
		cout << "nb_gens=" << nb_gens << endl;
		cout << "base_len=" << base_len() << endl;
		}
	if (nb_gens * base_len() != group_generator_size) {
		cout << "action::init_group_from_generators_by_base_images fatal: "
				"group_generator_size is not divisible by base_len" << endl;
		cout << "base_len=" << base_len() << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		exit(1);
		}
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "parsing generator " << i << ":" << endl;
			}
		int_vec_print(cout, group_generator_data + 
			i * base_len(), base_len());
		cout << endl;
		make_element_from_base_image(Elt, 
			group_generator_data + i * base_len(),
			verbose_level - 2);
		element_move(Elt, gens->ith(i), 0);
		}
	if (f_v) {
		cout << "done parsing generators" << endl;
		}
	S.init_trivial_group(verbose_level);
	S.init_generators(*gens, verbose_level);
	S.compute_base_orbits(verbose_level);
	while (TRUE) {
		S.closure_group(nb_times, 0/*verbose_level*/);
		S.group_order(cur_go);
		cout << "cur_go=" << cur_go << endl;
		if (!f_group_order_target)
			break;
		if (D.compare(cur_go, go) == 0) {
			cout << "reached target group order" << endl;
			break;
			}
		cout << "did not reach target group order, continuing" << endl;
		}

	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(&S, verbose_level - 1);
	f_has_strong_generators = TRUE;

	FREE_int(Elt);
}

void action::group_order(longinteger_object &go)
{
	//longinteger_domain D;
	
	if (Stabilizer_chain == NULL) {
		cout << "action::group_order Stabilizer_chain == NULL" << endl;
		go.create(0);
	}
	else {
		Stabilizer_chain->group_order(go);
		//D.multiply_up(go, Stabilizer_chain->transversal_length, base_len());
	}
}



void action::element_print_base_images(int *Elt)
{
	element_print_base_images(Elt, cout);
}

void action::element_print_base_images(int *Elt, ostream &ost)
{
	element_print_base_images_verbose(Elt, cout, 0);
}

void action::element_print_base_images_verbose(
		int *Elt, ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *base_images;
	
	if (f_v) {
		cout << "action::element_print_base_images_verbose" << endl;
		}
	base_images = NEW_int(base_len());
	element_base_images_verbose(Elt, base_images, verbose_level - 1);
	ost << "base images: ";
	int_vec_print(ost, base_images, base_len());
	FREE_int(base_images);
}

void action::element_base_images(int *Elt, int *base_images)
{
	element_base_images_verbose(Elt, base_images, 0);
}

void action::element_base_images_verbose(
		int *Elt, int *base_images, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, bi;
	
	if (f_v) {
		cout << "action::element_base_images_verbose" << endl;
		}
	for (i = 0; i < base_len(); i++) {
		bi = base_i(i);
		if (f_vv) {
			cout << "the " << i << "-th base point is "
					<< bi << " is mapped to:" << endl;
			}
		base_images[i] = element_image_of(bi, Elt, verbose_level - 2);
		if (f_vv) {
			cout << "the " << i << "-th base point is "
					<< bi << " is mapped to: " << base_images[i] << endl;
			}
		}
}

void action::minimize_base_images(int level,
		sims *S, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *base_images1;
	int *base_images2;
	int *Elt1, *Elt2, *Elt3;
	int i, j, /*bi,*/ oj, j0 = 0, image0 = 0, image;


	if (f_v) {
		cout << "action::minimize_base_images" << endl;
		cout << "level=" << level << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	base_images1 = NEW_int(base_len());
	base_images2 = NEW_int(base_len());
	
	element_move(Elt, Elt1, 0);
	for (i = level; i < base_len(); i++) {
		element_base_images(Elt1, base_images1);
		//bi = base[i];
		if (f_vv) {
			cout << "level " << i << " S->orbit_len[i]="
					<< S->get_orbit_length(i) << endl;
			}
		for (j = 0; j < S->get_orbit_length(i); j++) {
			oj = S->get_orbit(i, j);
			image = element_image_of(oj, Elt1, 0);
			if (f_vv) {
				cout << "level " << i << " j=" << j
						<< " oj=" << oj << " image="
						<< image << endl;
				}
			if (j == 0) {
				image0 = image;
				j0 = 0;
				}
			else {
				if (image < image0) {
					if (f_vv) {
						cout << "level " << i << " coset j="
								<< j << " image=" << image
								<< "less that image0 = "
								<< image0 << endl;
						}
					image0 = image;
					j0 = j;
					}
				}
			}
		if (f_vv) {
			cout << "level " << i << " S->orbit_len[i]="
					<< S->get_orbit_length(i) << " j0=" << j0 << endl;
			}
		S->coset_rep(Elt3, i, j0, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "cosetrep=" << endl;
			element_print_quick(Elt3, cout);
			if (degree < 500) {
				element_print_as_permutation(Elt3, cout);
				cout << endl;
				}
			}
		element_mult(Elt3, Elt1, Elt2, 0);
		element_move(Elt2, Elt1, 0);
		element_base_images(Elt1, base_images2);
		if (f_vv) {
			cout << "level " << i << " j0=" << j0 << endl;
			cout << "before: ";
			int_vec_print(cout, base_images1, base_len());
			cout << endl;
			cout << "after : ";
			int_vec_print(cout, base_images2, base_len());
			cout << endl;
			}
		}

	element_move(Elt1, Elt, 0);
	
	FREE_int(base_images1);
	FREE_int(base_images2);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
}


void action::get_generators_from_ascii_coding(
		char *ascii_coding, vector_ge *&gens, int *&tl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object go;
	group *G;

	if (f_v) {
		cout << "action::get_generators_from_ascii_coding" << endl;
		}
	G = NEW_OBJECT(group);
	G->init(this, verbose_level - 2);
	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"before G->init_ascii_coding_to_sims" << endl;
		}
	G->init_ascii_coding_to_sims(ascii_coding, verbose_level - 2);
	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"after G->init_ascii_coding_to_sims" << endl;
		}
		

	G->S->group_order(go);

	gens = NEW_OBJECT(vector_ge);
	tl = NEW_int(base_len());
	G->S->extract_strong_generators_in_order(*gens, tl,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding Group order="
				<< go << endl;
		}

	FREE_OBJECT(G);
	if (f_v) {
		cout << "action::get_generators_from_ascii_coding done" << endl;
		}
}


void action::lexorder_test(int *set, int set_sz,
	int &set_sz_after_test,
	vector_ge *gens, int max_starter,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_v5 = FALSE; //(verbose_level  >= 1);
	schreier *Sch;
	int i, /*loc,*/ orb, first, a, a0;

	if (f_v) {
		cout << "action::lexorder_test" << endl;
		}

	Sch = NEW_OBJECT(schreier);

	if (f_v) {
		cout << "action::lexorder_test computing orbits in action "
				"of degree " << degree << ", max_starter="
				<< max_starter << endl;
		}
	Sch->init(this, verbose_level - 2);
	Sch->init_generators(*gens, verbose_level - 2);

	//Sch->compute_all_point_orbits(0);
	Sch->compute_all_orbits_on_invariant_subset(set_sz, 
		set, 0 /* verbose_level */);

	if (f_v) {
		cout << "action::lexorder_test: there are "
				<< Sch->nb_orbits << " orbits on set" << endl;
		Sch->print_orbit_length_distribution(cout);
		}
	if (f_v5) {
		Sch->print_and_list_orbits(cout);
		}

	if (f_v) {
		cout << "action::lexorder_test "
				"max_starter=" << max_starter << endl;
		}
	set_sz_after_test = 0;
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		if (FALSE) {
			cout << "action::lexorder_test "
					"Looking at point " << a << endl;
			}
		//loc = Sch->orbit_inv[a];
		orb = Sch->orbit_number(a); //Sch->orbit_no[loc];
		first = Sch->orbit_first[orb];
		a0 = Sch->orbit[first];
		if (a0 < max_starter) {
			if (f_v) {
				cout << "action::lexorder_test  Point " << a
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
		cout << "action::lexorder_test Of the " << set_sz
				<< " points, we accept " << set_sz_after_test
				<< " and we reject " << set_sz - set_sz_after_test << endl;
		}
	FREE_OBJECT(Sch);
	if (f_v) {
		cout << "action::lexorder_test done" << endl;
		}
	
}

void action::compute_orbits_on_points(schreier *&Sch,
		vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_orbits_on_points" << endl;
		}
	Sch = NEW_OBJECT(schreier);
	Sch->init(this, verbose_level - 2);
	Sch->init_generators(*gens, verbose_level - 2);
	Sch->compute_all_point_orbits(verbose_level);
	//Sch.print_and_list_orbits(cout);
	if (f_v) {
		cout << "action::compute_orbits_on_points done, we found "
				<< Sch->nb_orbits << " orbits" << endl;
		}
}

void action::stabilizer_of_dual_hyperoval_representative(
		int k, int n, int no,
		vector_ge *&gens, const char *&stab_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *data, nb_gens, data_size;
	int i;
	knowledge_base K;

	if (f_v) {
		cout << "action::stabilizer_of_dual_hyperoval_representative" << endl;
		}
	K.DH_stab_gens(k, n, no, data, nb_gens, data_size, stab_order);

	gens = NEW_OBJECT(vector_ge);
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	if (f_vv) {
		cout << "action::stabilizer_of_dual_hyperoval_representative "
				"creating stabilizer generators:" << endl;
		}
	for (i = 0; i < nb_gens; i++) {
		make_element(gens->ith(i), data + i * data_size, 0 /*verbose_level*/);
		}
	
	if (f_v) {
		cout << "action::stabilizer_of_dual_hyperoval_representative done"
				<< endl;
		}
}

void action::stabilizer_of_spread_representative(
		int q, int k, int no,
		vector_ge *&gens, const char *&stab_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *data, nb_gens, data_size;
	int i;
	knowledge_base K;

	if (f_v) {
		cout << "action::stabilizer_of_spread_representative"
				<< endl;
		}
	K.Spread_stab_gens(q, k, no, data, nb_gens, data_size, stab_order);

	gens = NEW_OBJECT(vector_ge);
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	if (f_vv) {
		cout << "action::stabilizer_of_spread_representative "
				"creating stabilizer generators:" << endl;
		}
	for (i = 0; i < nb_gens; i++) {
		make_element(gens->ith(i),
				data + i * data_size, 0 /*verbose_level*/);
		}
	
	if (f_v) {
		cout << "action::stabilizer_of_spread_representative done"
				<< endl;
		}
}

void action::point_stabilizer_any_point(int &pt, 
	schreier *&Sch, sims *&Stab, strong_generators *&stab_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::point_stabilizer_any_point" << endl;
		}
	
	int f; //, len;
	longinteger_object go;
	
	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"computing all point orbits:" << endl;
		}
	Sch = Strong_gens->orbits_on_points_schreier(
			this, 0 /* verbose_level */);
	//compute_all_point_orbits(Sch,
	//*Strong_gens->gens, 0 /* verbose_level */);
	cout << "computing all point orbits done, found "
			<< Sch->nb_orbits << " orbits" << endl;


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"orbit rep = "
				<< pt << endl;
		}

	group_order(go);
	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"Computing point stabilizer:" << endl;
		}
	Sch->point_stabilizer(this, go, 
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);

	Stab->group_order(go);

	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"Computing point stabilizer done:" << endl;
		cout << "action::point_stabilizer_any_point "
				"point stabilizer is a group of order " << go << endl;
		}

	if (f_v) {
		cout << "action::point_stabilizer_any_point computing "
				"strong generators for the point stabilizer:" << endl;
		}
	stab_gens = NEW_OBJECT(strong_generators);
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "action::point_stabilizer_any_point strong generators "
				"for the point stabilizer have been computed" << endl;
		}

	if (f_v) {
		cout << "action::point_stabilizer_any_point done" << endl;
		}
}

void action::point_stabilizer_any_point_with_given_group(
	strong_generators *input_gens, 
	int &pt, 
	schreier *&Sch, sims *&Stab, strong_generators *&stab_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group" << endl;
		}
	
	int f; //, len;
	longinteger_object go;
	
	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"computing all point orbits:" << endl;
		}
	Sch = input_gens->orbits_on_points_schreier(this, 0 /* verbose_level */);
	//compute_all_point_orbits(Sch, *Strong_gens->gens, 0 /* verbose_level */);
	cout << "computing all point orbits done, found "
			<< Sch->nb_orbits << " orbits" << endl;


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"orbit rep = " << pt << endl;
		}

	input_gens->group_order(go);
	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"Computing point stabilizer:" << endl;
		}
	Sch->point_stabilizer(this, go, 
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);

	Stab->group_order(go);

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"Computing point stabilizer done:" << endl;
		cout << "action::point_stabilizer_any_point_with_given_group "
				"point stabilizer is a group of order " << go << endl;
		}

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"computing strong generators for the point stabilizer:"
				<< endl;
		}
	stab_gens = NEW_OBJECT(strong_generators);
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"strong generators for the point stabilizer "
				"have been computed" << endl;
		}

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group done"
				<< endl;
		}
}

void action::make_element_which_moves_a_line_in_PG3q(
		grassmann *Gr,
		int line_rk, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::make_element_which_moves_a_line_in_PG3q" << endl;
		}

	int M[4 * 4];
	int N[4 * 4 + 1]; // + 1 if f_semilinear
	int base_cols[4];
	int r, c, i, j;

	//int_vec_zero(M, 16);
	Gr->unrank_int_here(M, line_rk, 0 /*verbose_level*/);
	r = Gr->F->Gauss_simple(M, 2, 4, base_cols, 0 /* verbose_level */);
	Gr->F->kernel_columns(4, r, base_cols, base_cols + r);
	
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
	Gr->F->matrix_inverse(M, N, 4, 0 /* verbose_level */);
	N[4 * 4] = 0;
	make_element(Elt, N, 0);

	if (f_v) {
		cout << "action::make_element_which_moves_a_line_in_PG3q "
				"done" << endl;
		}
}


int action::is_matrix_group()
{
	if (type_G == matrix_group_t) {
			return TRUE;
	}
	else {
		return FALSE;
	}
}

int action::is_semilinear_matrix_group()
{
	if (!is_matrix_group()) {
			cout << "action::is_semilinear_matrix_group "
					"is not a matrix group" << endl;
			exit(1);
	}
	else {
		matrix_group *M;

		M = get_matrix_group();
		if (M->f_semilinear) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
}

int action::is_projective()
{
	if (!is_matrix_group()) {
			cout << "action::is_projective "
					"is not a matrix group" << endl;
			exit(1);
	}
	else {
		matrix_group *M;

		M = get_matrix_group();
		if (M->f_projective) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
}

int action::is_affine()
{
	if (!is_matrix_group()) {
			cout << "action::is_affine "
					"is not a matrix group" << endl;
			exit(1);
	}
	else {
		matrix_group *M;

		M = get_matrix_group();
		if (M->f_affine) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
}

int action::is_general_linear()
{
	if (!is_matrix_group()) {
			cout << "action::is_general_linear "
					"is not a matrix group" << endl;
			exit(1);
	}
	else {
		matrix_group *M;

		M = get_matrix_group();
		if (M->f_general_linear) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
}

matrix_group *action::get_matrix_group()
{
	if (type_G == unknown_symmetry_group_t) {
		cout << "action::get_matrix_group type_G == "
				"unknown_symmetry_group_t" << endl;
		exit(1);
	}
	else if (type_G == matrix_group_t) {
		return G.matrix_grp;
	}
	else if (type_G == perm_group_t) {
		cout << "action::get_matrix_group type_G == perm_group_t" << endl;
		exit(1);
	}
	else if (type_G == wreath_product_t) {
		cout << "action::get_matrix_group type_G == wreath_product_t" << endl;
		exit(1);
	}
	else if (type_G == direct_product_t) {
		cout << "action::get_matrix_group type_G == direct_product_t" << endl;
		exit(1);
	}
	else if (type_G == action_on_sets_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_subgroups_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_k_subsets_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_pairs_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_ordered_pairs_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == base_change_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == product_action_t) {
		cout << "action::get_matrix_group type_G == product_action_t" << endl;
		exit(1);
	}
	else if (type_G == action_by_right_multiplication_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_restriction_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_conjugation_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_determinant_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_sign_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_grassmannian_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_spread_set_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_orthogonal_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_cosets_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_factor_space_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_wedge_product_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_representation_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_subfield_structure_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_bricks_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_andre_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_orbits_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_flags_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_homogeneous_polynomials_t) {
		return subaction->get_matrix_group();
	}
	else {
		cout << "action::get_matrix_group unknown type" << endl;
		exit(1);
	}
}

void action::perform_tests(strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::perform_tests" << endl;
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
	combinatorics_domain Combi;
	os_interface Os;

	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);
	perm1 = NEW_int(degree);
	perm2 = NEW_int(degree);
	perm3 = NEW_int(degree);
	perm4 = NEW_int(degree);
	perm5 = NEW_int(degree);

	for (cnt = 0; cnt < 10; cnt++) {
		r1 = Os.random_integer(SG->gens->len);
		r2 = Os.random_integer(SG->gens->len);
		if (f_v) {
			cout << "r1=" << r1 << endl;
			cout << "r2=" << r2 << endl;
		}
		element_move(SG->gens->ith(r1), Elt1, 0);
		element_move(SG->gens->ith(r2), Elt2, 0);
		if (f_v) {
			cout << "Elt1 = " << endl;
			element_print_quick(Elt1, cout);
		}
		element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, degree);
			cout << endl;
		}

		if (f_v) {
			cout << "Elt2 = " << endl;
			element_print_quick(Elt2, cout);
		}
		element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm2, degree);
			cout << endl;
		}

		element_mult(Elt1, Elt2, Elt3, 0);
		if (f_v) {
			cout << "Elt3 = " << endl;
			element_print_quick(Elt3, cout);
		}
		element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm3, degree);
			cout << endl;
		}

		Combi.perm_mult(perm1, perm2, perm4, degree);
		if (f_v) {
			cout << "perm1 * perm2= " << endl;
			Combi.perm_print(cout, perm4, degree);
			cout << endl;
		}

		for (i = 0; i < degree; i++) {
			if (perm3[i] != perm4[i]) {
				cout << "test " << cnt
						<< " failed; something is wrong" << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "action::perform_tests test 1 passed" << endl;
	}

	for (cnt = 0; cnt < 10; cnt++) {
		r1 = Os.random_integer(SG->gens->len);
		if (f_v) {
			cout << "r1=" << r1 << endl;
		}
		element_move(SG->gens->ith(r1), Elt1, 0);
		if (f_v) {
			cout << "Elt1 = " << endl;
			element_print_quick(Elt1, cout);
		}
		element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, degree);
			cout << endl;
		}
		element_invert(Elt1, Elt2, 0);
		if (f_v) {
			cout << "Elt2 = " << endl;
			element_print_quick(Elt2, cout);
		}
		element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm2, degree);
			cout << endl;
		}

		element_mult(Elt1, Elt2, Elt3, 0);
		if (f_v) {
			cout << "Elt3 = " << endl;
			element_print_quick(Elt3, cout);
		}
		element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm3, degree);
			cout << endl;
		}

		if (!Combi.perm_is_identity(perm3, degree)) {
			cout << "fails the inverse test" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "action::perform_tests test 2 passed" << endl;
	}


	for (cnt = 0; cnt < 10; cnt++) {
		r1 = Os.random_integer(SG->gens->len);
		r2 = Os.random_integer(SG->gens->len);
		if (f_v) {
			cout << "r1=" << r1 << endl;
			cout << "r2=" << r2 << endl;
		}
		element_move(SG->gens->ith(r1), Elt1, 0);
		element_move(SG->gens->ith(r2), Elt2, 0);
		if (f_v) {
			cout << "Elt1 = " << endl;
			element_print_quick(Elt1, cout);
		}
		element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm1, degree);
			cout << endl;
		}

		if (f_v) {
			cout << "Elt2 = " << endl;
			element_print_quick(Elt2, cout);
		}
		element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		if (f_v) {
			cout << "as permutation: " << endl;
			Combi.perm_print(cout, perm2, degree);
			cout << endl;
		}

		element_mult(Elt1, Elt2, Elt3, 0);
		if (f_v) {
			cout << "Elt3 = " << endl;
			element_print_quick(Elt3, cout);
		}

		element_invert(Elt3, Elt4, 0);
		if (f_v) {
			cout << "Elt4 = Elt3^-1 = " << endl;
			element_print_quick(Elt4, cout);
		}


		element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		if (f_v) {
			cout << "as Elt3 as permutation: " << endl;
			Combi.perm_print(cout, perm3, degree);
			cout << endl;
		}

		element_as_permutation(Elt4, perm4, 0 /* verbose_level */);
		if (f_v) {
			cout << "as Elt4 as permutation: " << endl;
			Combi.perm_print(cout, perm4, degree);
			cout << endl;
		}

		Combi.perm_mult(perm3, perm4, perm5, degree);
		if (f_v) {
			cout << "perm3 * perm4= " << endl;
			Combi.perm_print(cout, perm5, degree);
			cout << endl;
		}

		for (i = 0; i < degree; i++) {
			if (perm5[i] != i) {
				cout << "test " << cnt
						<< " failed; something is wrong" << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "action::perform_tests test 3 passed" << endl;
	}


	if (f_v) {
		cout << "performing test 4:" << endl;
	}

	int data[] = {2,0,1, 0,1,1,0, 1,0,0,1, 1,0,0,1 };
	make_element(Elt1, data, verbose_level);
	element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
	if (f_v) {
		cout << "as Elt1 as permutation: " << endl;
		Combi.perm_print(cout, perm1, degree);
		cout << endl;
	}

	element_invert(Elt1, Elt2, 0);
	element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
	if (f_v) {
		cout << "as Elt2 as permutation: " << endl;
		Combi.perm_print(cout, perm2, degree);
		cout << endl;
	}


	element_mult(Elt1, Elt2, Elt3, 0);
	if (f_v) {
		cout << "Elt3 = " << endl;
		element_print_quick(Elt3, cout);
	}

	Combi.perm_mult(perm1, perm2, perm3, degree);
	if (f_v) {
		cout << "perm1 * perm2= " << endl;
		Combi.perm_print(cout, perm3, degree);
		cout << endl;
	}

	for (i = 0; i < degree; i++) {
		if (perm3[i] != i) {
			cout << "test 4 failed; something is wrong" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "action::perform_tests test 4 passed" << endl;
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


}}


