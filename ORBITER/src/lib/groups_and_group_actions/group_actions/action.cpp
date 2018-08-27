// action.C
//
// Anton Betten
// July 8, 2003

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"


// #############################################################################
// class action:
// #############################################################################



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

	null_base_data();

	//transversal_reps = NULL;

	null_element_data();

	degree = 0;
	f_is_linear = FALSE;
	dimension = 0;
	base_len = 0;
	elt_size_in_INT = 0;
	coded_elt_size_in_char = 0;
	group_prefix[0] = 0;
	//f_has_transversal_reps = FALSE;
	f_group_order_is_small = FALSE;
	make_element_size = 0;
	low_level_point_size = 0;

	null_function_pointers();
	
	nb_times_image_of_called = 0;
	nb_times_image_of_low_level_called = 0;
	nb_times_unpack_called = 0;
	nb_times_pack_called = 0;
	nb_times_retrieve_called = 0;
	nb_times_store_called = 0;
	nb_times_mult_called = 0;
	nb_times_invert_called = 0;


	f_allocated = FALSE;
	f_has_subaction = FALSE;
	f_subaction_is_allocated = FALSE;
	f_has_sims = FALSE;
	f_has_kernel = FALSE;
};

void action::freeself()
{
	//INT i;
	INT f_v = FALSE;
	INT f_vv = FALSE;

	if (f_v) {
		cout << "deleting action " << label << endl;
		print_info();
		}
	if (f_allocated) {
		if (f_vv) {
			cout << "freeing G of type ";
			action_print_symmetry_group_type(cout, type_G);
			cout << endl;
			}
		if (type_G == matrix_group_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.matrix_grp" << endl;
				cout << "G.matrix_grp=" << G.matrix_grp << endl;
				}
			FREE_OBJECT(G.matrix_grp);
			if (f_vv) {
				cout << "action::~action() freeing G.matrix_grp finished"
						<< endl;
				}
			G.matrix_grp = NULL;
			}
		else if (type_G == wreath_product_t) {
			if (f_vv) {
				cout << "action::~action() freeing "
						"G.wreath_product_group" << endl;
				cout << "G.wreath_product_group="
						<< G.wreath_product_group << endl;
				}
			FREE_OBJECT(G.wreath_product_group);
			if (f_vv) {
				cout << "action::~action() freeing "
						"G.wreath_product_group finished" << endl;
				}
			G.wreath_product_group = NULL;
			}
		else if (type_G == perm_group_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.perm_group_t" << endl;
				}
			FREE_OBJECT(G.perm_grp);
			if (f_vv) {
				cout << "action::~action() freeing G.perm_group_t finished"
						<< endl;
				}
			G.perm_grp = NULL;
			}
		else if (type_G == action_on_sets_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.on_sets" << endl;
				cout << "G.on_sets=" << G.on_sets << endl;
				}
			FREE_OBJECT(G.on_sets);
			if (f_vv) {
				cout << "action::~action() freeing G.on_sets finished" << endl;
				}
			G.on_sets = NULL;
			}
		else if (type_G == action_on_k_subsets_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.on_sets" << endl;
				cout << "G.on_k_subsets=" << G.on_k_subsets << endl;
				}
			FREE_OBJECT(G.on_k_subsets);
			if (f_vv) {
				cout << "action::~action() freeing G.on_k_subsets finished"
						<< endl;
				}
			G.on_k_subsets = NULL;
			}
		else if (type_G == action_on_orbits_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.OnOrbits" << endl;
				cout << "G.OnOrbits=" << G.OnOrbits << endl;
				}
			FREE_OBJECT(G.OnOrbits);
			if (f_vv) {
				cout << "action::~action() freeing G.OnOrbits finished" << endl;
				}
			G.OnOrbits = NULL;
			}
		else if (type_G == action_on_bricks_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.OnBricks" << endl;
				cout << "G.OnBricks=" << G.OnBricks << endl;
				}
			FREE_OBJECT(G.OnBricks);
			if (f_vv) {
				cout << "action::~action() freeing G.OnBricks finished" << endl;
				}
			G.OnBricks = NULL;
			}
		else if (type_G == action_on_andre_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.OnAndre" << endl;
				cout << "G.OnAndre=" << G.OnAndre << endl;
				}
			FREE_OBJECT(G.OnAndre);
			if (f_vv) {
				cout << "action::~action() freeing G.OnAndre finished" << endl;
				}
			G.OnAndre = NULL;
			}
		else if (type_G == action_by_right_multiplication_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.ABRM" << endl;
				}
			FREE_OBJECT(G.ABRM);
			G.ABRM = NULL;
			}
		else if (type_G == action_by_restriction_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.ABR" << endl;
				}
			FREE_OBJECT(G.ABR);
			G.ABR = NULL;
			}
		else if (type_G == action_by_conjugation_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.ABC" << endl;
				}
			FREE_OBJECT(G.ABC);
			G.ABC = NULL;
			}
		else if (type_G == action_by_representation_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.Rep" << endl;
				}
			FREE_OBJECT(G.Rep);
			G.Rep = NULL;
			}
		else if (type_G == action_on_determinant_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.AD" << endl;
				}
			FREE_OBJECT(G.AD);
			G.AD = NULL;
			}
		else if (type_G == action_on_sign_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.OnSign" << endl;
				}
			FREE_OBJECT(G.OnSign);
			G.OnSign = NULL;
			}
		else if (type_G == action_on_grassmannian_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.AG" << endl;
				}
			FREE_OBJECT(G.AG);
			G.AG = NULL;
			}
		else if (type_G == action_on_factor_space_t) {
			//delete G.AF;
			G.AF = NULL;
			}
		else if (type_G == action_on_wedge_product_t) {
			//delete G.AW;
			G.AW = NULL;
			}
		else if (type_G == action_on_homogeneous_polynomials_t) {
			if (f_vv) {
				cout << "action::~action() freeing G.OnHP" << endl;
				}
			FREE_OBJECT(G.OnHP);
			G.OnHP = NULL;
			}
		else {
			cout << "action::~action don't know "
					"how to free the object; action type is ";
			print_symmetry_group_type(cout);
			cout << endl;
			exit(1);
			}
		f_allocated = FALSE;
		type_G = unknown_symmetry_group_t;
		}
	if (f_v) {
		cout << "deleted G " << endl;
		}

	
	free_base_data();

	if (f_v) {
		cout << "after free_base_data" << endl;
		}
	
#if 0
	if (f_has_transversal_reps) {
		if (f_v) {
			cout << "we are freeing the transversal reps" << endl;
			}
		for (i = 0; i < base_len; i++) {
			FREE_INT(transversal_reps[i]);
			}
		FREE_PINT(transversal_reps);
		f_has_transversal_reps = FALSE;
		}
#endif
	
	if (f_v) {
		cout << "after freeing transversal reps" << endl;
		}
		
	free_element_data();

	if (f_v) {
		cout << "after free_element_data" << endl;
		}
	
	if (f_has_strong_generators) {
		if (f_v) {
			cout << "we are freeing strong generators" << endl;
			}
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
		//FREE_OBJECT(strong_generators); //delete strong_generators;
		//FREE_INT(tl);
		//strong_generators = NULL;
		//tl = NULL;
		f_has_strong_generators = FALSE;
		}

	if (f_v) {
		cout << "after freeing strong generators" << endl;
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
		cout << "after freeing subaction" << endl;
		}
	
	if (f_has_sims) {
		if (f_v) {
			cout << "action::~action() freeing Sims" << endl;
			}
		FREE_OBJECT(Sims);
		Sims = NULL;
		f_has_sims = FALSE;
		if (f_v) {
			cout << "action::~action() freeing Sims finished" << endl;
			}
		}

	if (f_v) {
		cout << "after freeing sims" << endl;
		}

	if (f_has_kernel) {
		if (f_v) {
			cout << "action::~action() freeing Kernel" << endl;
			}
		FREE_OBJECT(Kernel);
		Kernel = NULL;
		f_has_kernel = FALSE;
		if (f_v) {
			cout << "action::~action() freeing Kernel finished" << endl;
			}
		}
	if (f_v) {
		cout << "deleting action " << label << " done" << endl;
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
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	Elt3 = NEW_INT(elt_size_in_INT);
	Elt4 = NEW_INT(elt_size_in_INT);
	Elt5 = NEW_INT(elt_size_in_INT);
	eltrk1 = NEW_INT(elt_size_in_INT);
	eltrk2 = NEW_INT(elt_size_in_INT);
	eltrk3 = NEW_INT(elt_size_in_INT);
	elt_mult_apply = NEW_INT(elt_size_in_INT);
	elt1 = NEW_UBYTE(coded_elt_size_in_char);
	element_rw_memory_object = NEW_BYTE(coded_elt_size_in_char);
}

void action::free_element_data()
{
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	if (Elt3) {
		FREE_INT(Elt3);
		}
	if (Elt4) {
		FREE_INT(Elt4);
		}
	if (Elt5) {
		FREE_INT(Elt5);
		}
	if (eltrk1) {
		FREE_INT(eltrk1);
		}
	if (eltrk2) {
		FREE_INT(eltrk2);
		}
	if (eltrk3) {
		FREE_INT(eltrk3);
		}
	if (elt_mult_apply) {
		FREE_INT(elt_mult_apply);
		}
	if (elt1) {
		FREE_UBYTE(elt1);
		}
	if (element_rw_memory_object) {
		FREE_BYTE(element_rw_memory_object);
		}
	null_element_data();
}

void action::null_base_data()
{
	f_has_base = FALSE;
	base = NULL;
	transversal_length = NULL;
	orbit = NULL;
	orbit_inv = NULL;
	path = NULL;
}

void action::allocate_base_data(INT base_len)
{
	INT i, j;
	
	if (f_has_base) {
		free_base_data();
		}
	f_has_base = TRUE;
	action::base_len = base_len;
	base = NEW_INT(base_len);
	transversal_length = NEW_INT(base_len);
	orbit = NEW_PINT(base_len);
	orbit_inv = NEW_PINT(base_len);
	path = NEW_INT(base_len);
	for (i = 0; i < base_len; i++) {
		orbit[i] = NEW_INT(degree);
		orbit_inv[i] = NEW_INT(degree);
		for (j = 0; j < degree; j++) {
			orbit[i][j] = -1;
			orbit_inv[i][j] = -1;
			}
		}
}

void action::reallocate_base(INT new_base_point)
{
	INT i, j;
	INT *old_base;
	INT *old_transversal_length;
	INT **old_orbit;
	INT **old_orbit_inv;
	INT *old_path;
	
	old_base = base;
	old_transversal_length = transversal_length;
	old_orbit = orbit;
	old_orbit_inv = orbit_inv;
	old_path = path;
	
	base = NEW_INT(base_len + 1);
	transversal_length = NEW_INT(base_len + 1);
	orbit = NEW_PINT(base_len + 1);
	orbit_inv = NEW_PINT(base_len + 1);
	path = NEW_INT(base_len + 1);
	orbit[base_len] = NEW_INT(degree);
	orbit_inv[base_len] = NEW_INT(degree);
	for (i = 0; i < base_len; i++) {
		base[i] = old_base[i];
		transversal_length[i] = old_transversal_length[i];
		orbit[i] = old_orbit[i];
		orbit_inv[i] = old_orbit_inv[i];
		path[i] = old_path[i];
		}
	base[base_len] = new_base_point;
	transversal_length[base_len] = 1;
	for (j = 0; j < degree; j++) {
		orbit[base_len][j] = -1;
		orbit_inv[base_len][j] = -1;
		}
	base_len++;
	if (old_base)
		FREE_INT(old_base);
	if (old_transversal_length)
		FREE_INT(old_transversal_length);
	if (old_orbit)
		FREE_PINT(old_orbit);
	if (old_orbit_inv)
		FREE_PINT(old_orbit_inv);
	if (old_path)
		FREE_INT(old_path);
}

void action::free_base_data()
{
	INT i;
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "action::free_base_data" << endl;
		}
	if (base) {
		FREE_INT(base);
		base = NULL;
		}
	if (transversal_length) {
		FREE_INT(transversal_length);
		transversal_length = NULL;
		}
	if (orbit) {
		for (i = 0; i < base_len; i++) {
			if (f_v) {
				cout << "deleting orbit " << i << endl;
				}
			FREE_INT(orbit[i]);
			orbit[i] = NULL;
			if (f_v) {
				cout << "deleting orbit_inv " << i << endl;
				}
			FREE_INT(orbit_inv[i]);
			orbit_inv[i] = NULL;
			}
		FREE_PINT(orbit);
		orbit = NULL;
		FREE_PINT(orbit_inv);
		orbit_inv = NULL;
		}
	if (path) {
		FREE_INT(path);
		path = NULL;
		}
	f_has_base = FALSE;
	null_base_data();
	if (f_v) {
		cout << "action::free_base_data finished" << endl;
		}
}

// ##########################################################################


INT action::find_non_fixed_point(void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
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

INT action::find_fixed_points(void *elt,
		INT *fixed_points, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, n = 0;
	
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

INT action::test_if_set_stabilizes(INT *Elt,
		INT size, INT *set, INT verbose_level)
{
	INT *set1, *set2;
	INT i, cmp;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::test_if_set_stabilizes" << endl;
		}
	set1 = NEW_INT(size);
	set2 = NEW_INT(size);
	for (i = 0; i < size; i++) {
		set1[i] = set[i];
		}
	INT_vec_quicksort_increasingly(set1, size);
	map_a_set(set1, set2, size, Elt, 0);
	INT_vec_quicksort_increasingly(set2, size);
	cmp = INT_vec_compare(set1, set2, size);
	if (f_v) {
		cout << "the elements takes " << endl;
		INT_vec_print(cout, set1, size);
		cout << endl << "to" << endl;
		INT_vec_print(cout, set2, size);
		cout << endl;
		cout << "cmp = " << cmp << endl;
		}
	FREE_INT(set1);
	FREE_INT(set2);
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

void action::map_a_set(INT *set,
		INT *image_set, INT n, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "action::map_a_set" << endl;
		}
	if (f_vv) {
		cout << "group element:" << endl;
		element_print_quick(Elt, cout);
		cout << endl;
		cout << "set: " << endl;
		INT_vec_print(cout, set, n);
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

void action::map_a_set_and_reorder(INT *set,
		INT *image_set, INT n, INT *Elt, INT verbose_level)
{
	map_a_set(set, image_set, n, Elt, verbose_level);
	INT_vec_heapsort(image_set, n);
}



void action::init_sims(sims *G, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, k;
	
	if (f_v) {
		cout << "action::init_sims action " << label
				<< " base_len = " << base_len << endl;
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
	for (i = 0; i < base_len; i++) {
		k = G->orbit_len[i];
		transversal_length[i] = k;
		}
	init_base_from_sims(G, verbose_level);

	f_has_strong_generators = TRUE;
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(G, 0);
	
	if (f_v) {
		cout << "action::init_sims done" << endl;
		}
}

void action::init_base_from_sims(sims *G, INT verbose_level)
{
	INT i, j, k, l;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::init_base_from_sims, "
				"base length " << base_len << endl;
		//G->print(TRUE);
		}
	for (i = 0; i < base_len; i++) {
		//cout << "i = " << i << " base[i]="
		// << base[i] << " tl[i]=" << tl[i] << endl;
		//base[i] = bi = base[i];
		//transversal_length[i] = tl[i];
		//cout << "a" << endl;
		for (j = 0; j < degree; j++) {
			orbit[i][j] = -1;
			orbit_inv[i][j] = -1;
			}
		k = transversal_length[i];
		//cout << "b: bi=" << bi << " k=" << k << endl;
		for (j = 0; j < k; j++) {
			//cout << "j" << j << endl;
			//cout << G->orbit[i][j] << " " << endl;
			orbit[i][j] = l = G->orbit[i][j];
			orbit_inv[i][l] = j;
			}
		//cout << endl;
		//cout << "c" << endl;
		for (j = 0; j < degree; j++) {
			if (orbit_inv[i][j] == -1) {
				//cout << "adding " << j << " : k=" << k << endl;
				orbit[i][k] = j;
				orbit_inv[i][j] = k;
				k++;
				}
			}
		if (k != degree) {
			cout << "k != degree" << endl;
			cout << "transversal " << i << " k = " << k << endl;
			exit(1);
			}
		
		}
}

INT action::element_has_order_two(INT *E1,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret;
	
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

INT action::product_has_order_two(INT *E1,
		INT *E2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret;
	
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

INT action::product_has_order_three(INT *E1,
		INT *E2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret;
	
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

INT action::element_order(void *elt)
{
	return element_order_verbose(elt, 0);
}

INT action::element_order_verbose(void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *have_seen;
	INT l, l1, first, next, len, g, n, order = 1;
	
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
	have_seen = NEW_INT(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
			}
		/* work on cycle, starting with l: */
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
						"have_seen[next]\n";
				exit(1);
				}
			l1 = next;
			len++;
			}
		if (len == 1) {
			continue;
			}
		g = gcd_INT(len, order);
		order *= len / g;
		}
	FREE_INT(have_seen);
	if (f_v) {
		cout << "action::element_order_verbose "
				"done order=" << order << endl;
		}
	return order;
}

INT action::element_order_if_divisor_of(void *elt, INT o)
// returns the order of the element if o == 0
// if o != 0, returns the order of the element provided it divides o,
// 0 otherwise.
{
	INT *have_seen;
	INT l, l1, first, next, len, g, n, order = 1;
	
	n = degree;
	have_seen = NEW_INT(n);
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
			FREE_INT(have_seen);
			return 0;
			}
		g = gcd_INT(len, order);
		order *= len / g;
		}
	FREE_INT(have_seen);
	return order;
}

void action::compute_all_point_orbits(schreier &S,
		vector_ge &gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_all_point_orbits" << endl;
		}
	S.init(this);
	S.init_generators(gens);
	S.compute_all_point_orbits(verbose_level - 1);
	if (f_v) {
		cout << "action::compute_all_point_orbits done" << endl;
		}
}

INT action::depth_in_stab_chain(INT *Elt)
// the index of the first moved base point
{
	INT i, j, b;
	
	for (i = 0; i < base_len; i++) {
		b = base[i];
		j = element_image_of(b, Elt, 0);
		if (j != b)
			return i;
		}
	return base_len;
}

void action::strong_generators_at_depth(INT depth,
		vector_ge &gen)
// all strong generators that leave base points 0,..., depth - 1 fix
{
	INT i, j, l, n;
	
	l = Strong_gens->gens->len;
	gen.init(this);
	gen.allocate(l);
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
	sims *S, INT *sequence, INT len, INT verbose_level)
// S points to len + 1 many sims objects
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i;
	
	if (f_v) {
		cout << "action::compute_point_stabilizer_chain for sequence ";
		INT_vec_print(cout, sequence, len);
		cout << endl;
		}
	for (i = 0; i <= len; i++) {
		S[i].init(this);
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
		INT_vec_print(cout, sequence, len);
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

INT action::compute_orbit_of_point(vector_ge &strong_generators,
		INT pt, INT *orbit, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	schreier Schreier;
	INT len, i, f;
	
	if (f_v) {
		cout << "action::compute_orbit_of_point: "
				"computing orbit of point " << pt << endl;
		}
	Schreier.init(this);
	Schreier.init_generators(strong_generators);
	Schreier.compute_point_orbit(pt, 0);
	f = Schreier.orbit_first[0];
	len = Schreier.orbit_len[0];
	for (i = 0; i < len; i++) {
		orbit[i] = Schreier.orbit[f + i];
		}
	return len;
}

INT action::compute_orbit_of_point_generators_by_handle(INT nb_gen, 
	INT *gen_handle, INT pt, INT *orbit, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	vector_ge gens;
	INT i;
	
	gens.init(this);
	gens.allocate(nb_gen);
	for (i = 0; i < nb_gen; i++) {
		element_retrieve(gen_handle[i], gens.ith(i), 0);
		}
	return compute_orbit_of_point(gens, pt, orbit, verbose_level);
}


INT action::least_image_of_point(vector_ge &strong_generators,
	INT pt, INT *transporter, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	schreier Schreier;
	INT len, image, pos, i;
	
	if (f_v) {
		cout << "action::least_image_of_point: "
				"computing least image of " << pt << endl;
		}
	Schreier.init(this);
	Schreier.init_generators(strong_generators);
	Schreier.compute_point_orbit(pt, 0);
	len = Schreier.orbit_len[0];
	image = INT_vec_minimum(Schreier.orbit, len);
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

INT action::least_image_of_point_generators_by_handle(
	INT nb_gen, INT *gen_handle,
	INT pt, INT *transporter, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	vector_ge gens;
	INT i;
	
	if (nb_gen == 0) {
		element_one(transporter, 0);
		return pt;
		}
	gens.init(this);
	gens.allocate(nb_gen);
	for (i = 0; i < nb_gen; i++) {
		element_retrieve(gen_handle[i], gens.ith(i), 0);
		}
	return least_image_of_point(gens, pt, transporter, verbose_level);
}

void action::all_point_orbits(schreier &Schreier, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::all_point_orbits" << endl;
		}
	Schreier.init(this);
	if (!f_has_strong_generators) {
		cout << "action::all_point_orbits !f_has_strong_generators" << endl;
		exit(1);
		}
	Schreier.init_generators(*Strong_gens->gens /* *strong_generators */);
	Schreier.compute_all_point_orbits(verbose_level);
}

void action::compute_stabilizer_orbits(partitionstack *&Staborbits,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT i;
	vector_ge gen;
	
	if (f_v) {
		cout << "action::compute_stabilizer_orbits" << endl;
		cout << "base_len = " << base_len << endl;
		for (i = 0; i < base_len; i++) {
			cout << i << " : " << base[i] << " : " << transversal_length[i];
			INT_vec_print(cout, orbit[i], transversal_length[i]);
			cout << endl;
			}
		cout << "degree = " << degree << endl;
		}
	Staborbits = NEW_OBJECTS(partitionstack, base_len);
		// where is this freed???

	for (i = 0; i < base_len; i++) {
		strong_generators_at_depth(i, gen);
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


INT action::check_if_in_set_stabilizer(INT *Elt,
		INT size, INT *set, INT verbose_level)
{
	INT i, a, b, idx;
	INT *ordered_set;
	INT f_v = (verbose_level >= 1);
	
	ordered_set = NEW_INT(size);
	for (i = 0; i < size; i++) {
		ordered_set[i] = set[i];
		}
	INT_vec_sort(size, ordered_set);
	for (i = 0; i < size; i++) {
		a = ordered_set[i];
		b = element_image_of(a, Elt, 0);
		if (!INT_vec_search(ordered_set, size, b, idx)) {
			if (f_v) {
				cout << "action::check_if_in_set_stabilizer fails" << endl;
				cout << "set: ";
				INT_vec_print(cout, set, size);
				cout << endl;
				cout << "ordered_set: ";
				INT_vec_print(cout, ordered_set, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
				}
			FREE_INT(ordered_set);
			return FALSE;
			}
		}
	FREE_INT(ordered_set);
	return TRUE;
	
}

INT action::check_if_transporter_for_set(INT *Elt,
		INT size, INT *set1, INT *set2, INT verbose_level)
{
	INT i, a, b, idx;
	INT *ordered_set2;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 4);
	
	if (f_vv) {
		cout << "action::check_if_transporter_for_set "
				"size=" << size << endl;
		INT_vec_print(cout, set1, size);
		cout << endl;
		INT_vec_print(cout, set2, size);
		cout << endl;
		element_print(Elt, cout);
		cout << endl;
		}
	ordered_set2 = NEW_INT(size);
	for (i = 0; i < size; i++) {
		ordered_set2[i] = set2[i];
		}
	INT_vec_sort(size, ordered_set2);
	if (f_vv) {
		cout << "sorted target set:" << endl;
		INT_vec_print(cout, ordered_set2, size);
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
		if (!INT_vec_search(ordered_set2, size, b, idx)) {
			if (f_v) {
				cout << "action::check_if_transporter_for_set fails" << endl;
				cout << "set1   : ";
				INT_vec_print(cout, set1, size);
				cout << endl;
				cout << "set2   : ";
				INT_vec_print(cout, set2, size);
				cout << endl;
				cout << "ordered: ";
				INT_vec_print(cout, ordered_set2, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
				}
			FREE_INT(ordered_set2);
			return FALSE;
			}
		}
	FREE_INT(ordered_set2);
	return TRUE;
	
}

void action::compute_set_orbit(vector_ge &gens,
	INT size, INT *set,
	INT &nb_sets, INT **&Sets, INT **&Transporter,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *image_set;
	INT **New_Sets;
	INT **New_Transporter;
	INT nb_finished, allocated_nb_sets;
	INT new_allocated_nb_sets, nb_gens, i, j, h;
	
	if (f_v) {
		cout << "action::compute_set_orbit: ";
		INT_vec_print(cout, set, size);
		cout << endl;
		}
	nb_gens = gens.len;
	
	allocated_nb_sets = 100;
	Sets = NEW_PINT(allocated_nb_sets);
	Transporter = NEW_PINT(allocated_nb_sets);
	nb_sets = 0;

	image_set = NEW_INT(size);
	Sets[0] = NEW_INT(size);
	for (i = 0; i < size; i++) {
		Sets[0][i] = set[i];
		}
	INT_vec_sort(size, Sets[0]);
	
	Transporter[0] = NEW_INT(elt_size_in_INT);
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
				INT_vec_print(cout, image_set, size);
				cout << endl;
				}
			for (j = 0; j < nb_sets; j++) {
				if (INT_vec_compare(Sets[j], image_set, size) == 0)
					break;
				}
			if (j < nb_sets) {
				continue;
				}
			// n e w set found:
			if (f_v) {
				cout << "n e w set " << nb_sets << ":";
				INT_vec_print(cout, image_set, size);
				cout << endl;
				}
			Sets[nb_sets] = image_set;
			image_set = NEW_INT(size);
			Transporter[nb_sets] = NEW_INT(elt_size_in_INT);
			element_mult(Transporter[nb_finished],
					gens.ith(i), Transporter[nb_sets], 0);
			nb_sets++;
			if (nb_sets == allocated_nb_sets) {
				new_allocated_nb_sets = allocated_nb_sets + 100;
				cout << "reallocating to size "
						<< new_allocated_nb_sets << endl;
				New_Sets = NEW_PINT(new_allocated_nb_sets);
				New_Transporter = NEW_PINT(new_allocated_nb_sets);
				for (h = 0; h < nb_sets; h++) {
					New_Sets[h] = Sets[h];
					New_Transporter[h] = Transporter[h];
					}
				FREE_PINT(Sets);
				FREE_PINT(Transporter);
				Sets = New_Sets;
				Transporter = New_Transporter;
				allocated_nb_sets = new_allocated_nb_sets;
				}
			} // next i
		 nb_finished++;
		}
	FREE_INT(image_set);
	if (f_v) {
		cout << "action::compute_set_orbit "
				"found an orbit of size " << nb_sets << endl;
		for (i = 0; i < nb_sets; i++) {
			cout << i << " : ";
			INT_vec_print(cout, Sets[i], size);
			cout << endl;
			element_print(Transporter[i], cout);
			}
		}
}

void action::delete_set_orbit(INT nb_sets, INT **Sets, INT **Transporter)
{
	INT i;
	
	for (i = 0; i < nb_sets; i++) {
		FREE_INT(Sets[i]);
		FREE_INT(Transporter[i]);
		}
	FREE_PINT(Sets);
	FREE_PINT(Transporter);
}

void action::compute_minimal_set(vector_ge &gens, INT size, INT *set, 
	INT *minimal_set, INT *transporter, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT **Sets;
	INT **Transporter;
	INT nb_sets, i;
	INT min_set;
	
	if (f_v) {
		cout << "action::compute_minimal_set" << endl;
		}
	

	compute_set_orbit(gens, size, set, 
		nb_sets, Sets, Transporter, verbose_level);
	
	min_set = 0;
	for (i = 1; i < nb_sets; i++) {
		if (INT_vec_compare(Sets[i], Sets[min_set], size) < 0) {
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
	INT base_len, INT *the_base, INT level,
	vector_ge &gens, vector_ge &subset_of_gens,
	INT verbose_level)
{
	INT nb_generators_found;
	INT *gen_idx;
	INT nb_gens, i, j, bj, bj_image;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "action::find_strong_generators_at_level "
				"level=" << level << " base: ";
		INT_vec_print(cout, the_base, base_len);
		cout << endl;
		}
	nb_gens = gens.len;
	gen_idx = NEW_INT(gens.len);
	
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
	subset_of_gens.init(this);
	subset_of_gens.allocate(nb_generators_found);
	for (i = 0; i < nb_generators_found; i++) {
		j = gen_idx[i];
		element_move(gens.ith(j), subset_of_gens.ith(i), 0);
		}
	FREE_INT(gen_idx);
	if (f_v) {
		cout << "action::find_strong_generators_at_level found " 
			<< nb_generators_found << " strong generators" << endl;
		if (f_vv) {
			subset_of_gens.print(cout);
			cout << endl;
			}
		}
}

void action::compute_strong_generators_from_sims(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
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
		INT *Elt, INT *data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *base_image;
	INT i, a;
	
	if (f_v) {
		cout << "action::make_element_from_permutation_representation" << endl;
		}
	base_image = NEW_INT(base_len);
	for (i = 0; i < base_len; i++) {
		a = base[i];
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

	FREE_INT(base_image);
	if (f_v) {
		cout << "action::make_element_from_permutation_representation done"
				<< endl;
		}
}

void action::make_element_from_base_image(INT *Elt,
		INT *data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 2);
	INT *base_image;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	sims *S;
#if 1
	INT offset = 0;
	INT f_do_it_anyway_even_for_big_degree = TRUE;
	INT f_print_cycles_of_length_one = FALSE;
#endif

	INT i, j, yi, z, b, c, b_pt;

	if (f_v) {
		cout << "action::make_element_from_base_image" << endl;
		cout << "base images: ";
		INT_vec_print(cout, data, base_len);
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
	base_image = NEW_INT(base_len);
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	Elt3 = NEW_INT(elt_size_in_INT);
	Elt4 = NEW_INT(elt_size_in_INT);
	for (j = 0; j < base_len; j++) {
		base_image[j] = data[j];
		}
	element_one(Elt3, 0);
	
	for (i = 0; i < base_len; i++) {
		element_invert(Elt3, Elt4, 0);
		b_pt = base[i];
		yi = base_image[i];
		z = element_image_of(yi, Elt4, 0);
		j = S->orbit_inv[i][z];
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
		S->coset_rep(i, j, 0);
		if (f_vv) {
			cout << "cosetrep=" << endl;
			element_print_quick(S->cosetrep, cout);
			element_print_base_images(S->cosetrep);
			element_print_as_permutation_with_offset(S->cosetrep, cout, 
				offset, f_do_it_anyway_even_for_big_degree, 
				f_print_cycles_of_length_one, 0/*verbose_level*/);
			}
		element_mult(S->cosetrep, Elt3, Elt4, 0);
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
					"fatal: element_image_of(b_pt, Elt3, 0) != yi" << endl;
			exit(1);
			}
		}
	element_move(Elt3, Elt, 0);
	for (i = 0; i < base_len; i++) {
		yi = data[i];
		b = element_image_of(base[i], Elt, 0);
		if (yi != b) {
			cout << "action::make_element_from_base_image fatal: yi != b"
					<< endl;
			cout << "i=" << i << endl;
			cout << "base[i]=" << base[i] << endl;
			cout << "yi=" << yi << endl;
			cout << "b=" << b << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "action::make_element_from_base_image created element:" << endl;
		element_print_quick(Elt, cout);
		}
	FREE_INT(base_image);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(Elt4);
}

void action::make_element_2x2(INT *Elt, INT a0, INT a1, INT a2, INT a3)
{
	INT data[4];
	
	data[0] = a0;
	data[1] = a1;
	data[2] = a2;
	data[3] = a3;
	make_element(Elt, data, 0);
}

void action::make_element(INT *Elt, INT *data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
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
		//PA->A2->make_element(Elt + PA->A1->elt_size_in_INT, 
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
	INT nb_auts, INT *aut_data,
	sims &S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h, i, coset;
	INT *Elt1, *Elt2;
	longinteger_object go;
	
	if (f_v) {
		cout << "action::build_up_automorphism_group_from_aut_data "
				"action=" << label << " nb_auts=" << nb_auts << endl;
		}
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	S.init(this);
	S.init_trivial_group(verbose_level - 1);
	for (h = 0; h < nb_auts; h++) {
		if (f_v) {
			cout << "aut_data[" << h << "]=";
			INT_vec_print(cout, aut_data + h * base_len, base_len);
			cout << endl;
			}
		for (i = 0; i < base_len; i++) {
			coset = aut_data[h * base_len + i];
			//image_point = Sims->orbit[i][coset];
			Sims->path[i] = coset;
				//Sims->orbit_inv[i][aut_data[h * base_len + i]];
			}
		if (f_v) {
			cout << "path=";
			INT_vec_print(cout, Sims->path, base_len);
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
	FREE_INT(Elt1);
	FREE_INT(Elt2);
}

void action::element_power_INT_in_place(INT *Elt,
		INT n, INT verbose_level)
{
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	
	Elt2 = NEW_INT(elt_size_in_INT);
	Elt3 = NEW_INT(elt_size_in_INT);
	Elt4 = NEW_INT(elt_size_in_INT);
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
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(Elt4);
}

void action::word_in_ab(INT *Elt1, INT *Elt2, INT *Elt3,
		const BYTE *word, INT verbose_level)
{
	INT *Elt4;
	INT *Elt5;
	INT l, i;
	

	Elt4 = NEW_INT(elt_size_in_INT);
	Elt5 = NEW_INT(elt_size_in_INT);
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
	
	FREE_INT(Elt4);
	FREE_INT(Elt5);
}

void action::init_group_from_generators(
	INT *group_generator_data, INT group_generator_size,
	INT f_group_order_target, const BYTE *group_order_target, 
	vector_ge *gens, strong_generators *&Strong_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object go, cur_go;
	sims S;
	INT *Elt;
	INT nb_gens, i;
	INT nb_times = 200;

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
	S.init(this);
	Elt = NEW_INT(elt_size_in_INT);
	nb_gens = group_generator_size / make_element_size;
	if (nb_gens * make_element_size != group_generator_size) {
		cout << "action::init_group_from_generators fatal: "
				"group_generator_size is not divisible by make_element_size"
				<< endl;
		cout << "make_element_size=" << make_element_size << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		exit(1);
		}
	gens->init(this);
	gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "parsing generator " << i << ":" << endl;
			}
		INT_vec_print(cout, group_generator_data + 
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

	FREE_INT(Elt);
}

void action::init_group_from_generators_by_base_images(
	INT *group_generator_data, INT group_generator_size, 
	INT f_group_order_target, const BYTE *group_order_target, 
	vector_ge *gens, strong_generators *&Strong_gens_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object go, cur_go;
	sims S;
	INT *Elt;
	INT nb_gens, i;
	INT nb_times = 200;

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
	S.init(this);
	Elt = NEW_INT(elt_size_in_INT);
	nb_gens = group_generator_size / base_len;
	if (f_v) {
		cout << "nb_gens=" << nb_gens << endl;
		cout << "base_len=" << base_len << endl;
		}
	if (nb_gens * base_len != group_generator_size) {
		cout << "action::init_group_from_generators_by_base_images fatal: "
				"group_generator_size is not divisible by base_len" << endl;
		cout << "base_len=" << base_len << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		exit(1);
		}
	gens->init(this);
	gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "parsing generator " << i << ":" << endl;
			}
		INT_vec_print(cout, group_generator_data + 
			i * base_len, base_len);
		cout << endl;
		make_element_from_base_image(Elt, 
			group_generator_data + i * base_len, verbose_level - 2);
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

	FREE_INT(Elt);
}

void action::print_symmetry_group_type(ostream &ost)
{
	action_print_symmetry_group_type(ost, type_G);
	if (f_has_subaction) {
		ost << "->";
		subaction->print_symmetry_group_type(ost);
		}
	//else {
		//ost << "no subaction";
		//}
	
}

void action::print_info()
{
	cout << "ACTION " << label << " degree=" << degree << " of type ";
	print_symmetry_group_type(cout);
	cout << endl;
	cout << "low_level_point_size=" << low_level_point_size;
	cout << ", f_has_sims=" << f_has_sims;
	cout << ", f_has_strong_generators=" << f_has_strong_generators;
	cout << endl;

	if (f_is_linear) {
		cout << "linear of dimension " << dimension << endl;
		}

	if (base_len) {
		cout << "base: ";
		INT_vec_print(cout, base, base_len);
		cout << endl;
		}
	if (f_has_sims) {
		cout << "has sims" << endl;
		longinteger_object go;

		Sims->group_order(go);
		cout << "Order " << go << " = ";
		INT_vec_print(cout, Sims->orbit_len, base_len);
		cout << endl;
		}
	cout << endl;

}

void action::print_base()
{
	cout << "action " << label << " has base ";
	INT_vec_print(cout, base, base_len);
	cout << endl;
}

void action::group_order(longinteger_object &go)
{
	longinteger_domain D;
	
	D.multiply_up(go, transversal_length, base_len);
	
}

void action::print_group_order(ostream &ost)
{
	longinteger_object go;
	group_order(go);
	cout << go;
}

void action::print_group_order_long(ostream &ost)
{
	INT i;
	
	longinteger_object go;
	group_order(go);
	cout << go << " =";
	for (i = 0; i < base_len; i++) {
		cout << " " << transversal_length[i];
		}
}

void action::print_vector(vector_ge &v)
{
	INT i, l;
	
	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : " << endl;
		element_print_quick(v.ith(i), cout);
		cout << endl;
		}
}

void action::print_vector_as_permutation(vector_ge &v)
{
	INT i, l;
	
	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		element_print_as_permutation(v.ith(i), cout);
		cout << endl;
		}
}




void action::element_print_base_images(INT *Elt)
{
	element_print_base_images(Elt, cout);
}

void action::element_print_base_images(INT *Elt, ostream &ost)
{
	element_print_base_images_verbose(Elt, cout, 0);
}

void action::element_print_base_images_verbose(
		INT *Elt, ostream &ost, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *base_images;
	
	if (f_v) {
		cout << "action::element_print_base_images_verbose" << endl;
		}
	base_images = NEW_INT(base_len);
	element_base_images_verbose(Elt, base_images, verbose_level - 1);
	ost << "base images: ";
	INT_vec_print(ost, base_images, base_len);
	FREE_INT(base_images);
}

void action::element_base_images(INT *Elt, INT *base_images)
{
	element_base_images_verbose(Elt, base_images, 0);
}

void action::element_base_images_verbose(
		INT *Elt, INT *base_images, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, bi;
	
	if (f_v) {
		cout << "action::element_base_images_verbose" << endl;
		}
	for (i = 0; i < base_len; i++) {
		bi = base[i];
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

void action::minimize_base_images(INT level,
		sims *S, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *base_images1;
	INT *base_images2;
	INT *Elt1, *Elt2;
	INT i, j, /*bi,*/ oj, j0 = 0, image0 = 0, image;


	if (f_v) {
		cout << "action::minimize_base_images" << endl;
		cout << "level=" << level << endl;
		}
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	base_images1 = NEW_INT(base_len);
	base_images2 = NEW_INT(base_len);
	
	element_move(Elt, Elt1, 0);
	for (i = level; i < base_len; i++) {
		element_base_images(Elt1, base_images1);
		//bi = base[i];
		if (f_vv) {
			cout << "level " << i << " S->orbit_len[i]="
					<< S->orbit_len[i] << endl;
			}
		for (j = 0; j < S->orbit_len[i]; j++) {
			oj = S->orbit[i][j];
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
					<< S->orbit_len[i] << " j0=" << j0 << endl;
			}
		S->coset_rep(i, j0, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "cosetrep=" << endl;
			element_print_quick(S->cosetrep, cout);
			if (degree < 500) {
				element_print_as_permutation(S->cosetrep, cout);
				cout << endl;
				}
			}
		element_mult(S->cosetrep, Elt1, Elt2, 0);
		element_move(Elt2, Elt1, 0);
		element_base_images(Elt1, base_images2);
		if (f_vv) {
			cout << "level " << i << " j0=" << j0 << endl;
			cout << "before: ";
			INT_vec_print(cout, base_images1, base_len);
			cout << endl;
			cout << "after : ";
			INT_vec_print(cout, base_images2, base_len);
			cout << endl;
			}
		}

	element_move(Elt1, Elt, 0);
	
	FREE_INT(base_images1);
	FREE_INT(base_images2);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
}


void action::element_conjugate_bvab(INT *Elt_A,
		INT *Elt_B, INT *Elt_C, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT *Elt1, *Elt2;


	if (f_v) {
		cout << "action::element_conjugate_bvab" << endl;
		}
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	if (f_v) {
		cout << "action::element_conjugate_bvab A=" << endl;
		element_print_quick(Elt_A, cout);
		cout << "action::element_conjugate_bvab B=" << endl;
		element_print_quick(Elt_B, cout);
		}

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt1, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt_B, Elt_C, 0);
	if (f_v) {
		cout << "action::element_conjugate_bvab C=B^-1 * A * B" << endl;
		element_print_quick(Elt_C, cout);
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_v) {
		cout << "action::element_conjugate_bvab done" << endl;
		}
}

void action::element_conjugate_babv(INT *Elt_A,
		INT *Elt_B, INT *Elt_C, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT *Elt1, *Elt2;


	if (f_v) {
		cout << "action::element_conjugate_babv" << endl;
		}
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt_B, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt1, Elt_C, 0);
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
}

void action::element_commutator_abavbv(INT *Elt_A,
		INT *Elt_B, INT *Elt_C, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT *Elt1, *Elt2, *Elt3, *Elt4;


	if (f_v) {
		cout << "action::element_commutator_abavbv" << endl;
		}
	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	Elt3 = NEW_INT(elt_size_in_INT);
	Elt4 = NEW_INT(elt_size_in_INT);

	element_invert(Elt_A, Elt1, 0);
	element_invert(Elt_B, Elt2, 0);
	element_mult(Elt_A, Elt_B, Elt3, 0);
	element_mult(Elt3, Elt1, Elt4, 0);
	element_mult(Elt4, Elt2, Elt_C, 0);
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(Elt4);
}

void action::read_representatives(BYTE *fname,
		INT *&Reps, INT &nb_reps, INT &size, INT verbose_level)
{
	INT f_casenumbers = FALSE;
	INT nb_cases;
	INT *Set_sizes;
	INT **Sets;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii; 
	INT *Casenumbers;
	INT i, j;
	

	cout << "action::read_file_and_print_representatives "
			"reading file " << fname << endl;
	
	read_and_parse_data_file_fancy(fname, 
		f_casenumbers, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level*/);
	nb_reps = nb_cases;
	size = Set_sizes[0];
	Reps = NEW_INT(nb_cases * size);
	for (i = 0; i < nb_cases; i++) {
		for (j = 0; j < size; j++) {
			Reps[i * size + j] = Sets[i][j];
			}
		}
	free_data_fancy(nb_cases, 
		Set_sizes, Sets, 
		Ago_ascii, Aut_ascii, 
		Casenumbers);
}

void action::read_representatives_and_strong_generators(
	BYTE *fname, INT *&Reps,
	BYTE **&Aut_ascii, INT &nb_reps, INT &size, INT verbose_level)
//INT **&Tl, vector_ge **gens, 
{
	INT f_casenumbers = FALSE;
	INT nb_cases;
	INT *Set_sizes;
	INT **Sets;
	BYTE **Ago_ascii;
	//BYTE **Aut_ascii; 
	INT *Casenumbers;
	INT i, j;
	

	cout << "action::read_file_and_print_representatives "
			"reading file " << fname << endl;
	
	read_and_parse_data_file_fancy(fname, 
		f_casenumbers, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level*/);
	nb_reps = nb_cases;
	size = Set_sizes[0];
	Reps = NEW_INT(nb_cases * size);
	for (i = 0; i < nb_cases; i++) {
		for (j = 0; j < size; j++) {
			Reps[i * size + j] = Sets[i][j];
			}
		}
	free_data_fancy(nb_cases, 
		Set_sizes, Sets, 
		Ago_ascii, NULL /*Aut_ascii*/, 
		Casenumbers);
}

void action::read_file_and_print_representatives(
		BYTE *fname, INT f_print_stabilizer_generators)
{
	INT f_casenumbers = FALSE;
	INT nb_cases;
	INT *Set_sizes;
	INT **Sets;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii; 
	INT *Casenumbers;
	INT i;
	

	cout << "action::read_file_and_print_representatives reading file "
			<< fname << endl;
	
	read_and_parse_data_file_fancy(fname, 
		f_casenumbers, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level*/);
	for (i = 0; i < nb_cases; i++) {
		cout << "Orbit " << i << " representative ";
		INT_vec_print(cout, Sets[i], Set_sizes[i]);
		cout << endl;

		group *G;
		vector_ge *gens;
		INT *tl;

		G = NEW_OBJECT(group);
		G->init(this);
		G->init_ascii_coding_to_sims(Aut_ascii[i]);
		

		longinteger_object go;

		G->S->group_order(go);

		gens = NEW_OBJECT(vector_ge);
		tl = NEW_INT(base_len);
		G->S->extract_strong_generators_in_order(*gens, tl,
				0 /* verbose_level */);
		cout << "Stabilizer has order " << go << " tl=";
		INT_vec_print(cout, tl, base_len);
		cout << endl;

		if (f_print_stabilizer_generators) {
			cout << "The stabilizer is generated by:" << endl;
			gens->print(cout);
			}

		FREE_OBJECT(G);
		FREE_OBJECT(gens);
		FREE_INT(tl);

		}
	free_data_fancy(nb_cases, 
		Set_sizes, Sets, 
		Ago_ascii, Aut_ascii, 
		Casenumbers);

}

void action::read_set_and_stabilizer(const BYTE *fname, 
	INT no, INT *&set, INT &set_sz, sims *&stab, 
	strong_generators *&Strong_gens, 
	INT &nb_cases, 
	INT verbose_level)
{
	INT f_v = (verbose_level  >= 1);
	INT f_vv = (verbose_level  >= 2);
	INT f_casenumbers = FALSE;
	//INT nb_cases;
	INT *Set_sizes;
	INT **Sets;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii; 
	INT *Casenumbers;
	group *G;
	INT i;
	

	if (f_v) {
		cout << "action::read_set_and_stabilizer reading file " << fname
				<< " no=" << no << endl;
		}
	
	read_and_parse_data_file_fancy(fname, 
		f_casenumbers, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 1);
		// GALOIS/util.C

	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"after read_and_parse_data_file_fancy" << endl;
		cout << "Aut_ascii[no]=" << Aut_ascii[no] << endl;
		cout << "Set_sizes[no]=" << Set_sizes[no] << endl;
		}
	
	set_sz = Set_sizes[no];
	set = NEW_INT(set_sz);
	for (i = 0; i < set_sz; i ++) {
		set[i] = Sets[no][i];
		}


	G = NEW_OBJECT(group);
	G->init(this);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"before G->init_ascii_coding_to_sims" << endl;
		}
	G->init_ascii_coding_to_sims(Aut_ascii[no]);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"after G->init_ascii_coding_to_sims" << endl;
		}
		
	stab = G->S;
	G->S = NULL;
	G->f_has_sims = FALSE;

	longinteger_object go;

	stab->group_order(go);

	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(stab, 0);

	if (f_vv) {
		cout << "action::read_set_and_stabilizer Group order=" << go << endl;
		}

	FREE_OBJECT(G);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer after  FREE_OBJECT  G" << endl;
		}
	free_data_fancy(nb_cases, 
		Set_sizes, Sets, 
		Ago_ascii, Aut_ascii, 
		Casenumbers);
	if (f_v) {
		cout << "action::read_set_and_stabilizer done" << endl;
		}

}

void action::get_generators_from_ascii_coding(
		BYTE *ascii_coding, vector_ge *&gens, INT *&tl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go;
	group *G;

	if (f_v) {
		cout << "action::get_generators_from_ascii_coding" << endl;
		}
	G = NEW_OBJECT(group);
	G->init(this);
	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"before G->init_ascii_coding_to_sims" << endl;
		}
	G->init_ascii_coding_to_sims(ascii_coding);
	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"after G->init_ascii_coding_to_sims" << endl;
		}
		

	G->S->group_order(go);

	gens = NEW_OBJECT(vector_ge);
	tl = NEW_INT(base_len);
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


void action::lexorder_test(INT *set, INT set_sz,
	INT &set_sz_after_test,
	vector_ge *gens, INT max_starter,
	INT verbose_level)
{
	INT f_v = (verbose_level  >= 1);
	INT f_v5 = FALSE; //(verbose_level  >= 1);
	schreier *Sch;
	INT i, /*loc,*/ orb, first, a, a0;

	if (f_v) {
		cout << "action::lexorder_test" << endl;
		}

	Sch = NEW_OBJECT(schreier);

	if (f_v) {
		cout << "action::lexorder_test computing orbits in action "
				"of degree " << degree << ", max_starter="
				<< max_starter << endl;
		}
	Sch->init(this);
	Sch->init_generators(*gens);

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
		cout << "action::lexorder_test max_starter=" << max_starter << endl;
		}
	set_sz_after_test = 0;
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		if (FALSE) {
			cout << "action::lexorder_test  Looking at point " << a << endl;
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
		vector_ge *gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_orbits_on_points" << endl;
		}
	Sch = NEW_OBJECT(schreier);
	Sch->init(this);
	Sch->init_generators(*gens);
	Sch->compute_all_point_orbits(verbose_level);
	//Sch.print_and_list_orbits(cout);
	if (f_v) {
		cout << "action::compute_orbits_on_points done, we found "
				<< Sch->nb_orbits << " orbits" << endl;
		}
}

void action::stabilizer_of_dual_hyperoval_representative(INT k, INT n, INT no,
		vector_ge *&gens, const BYTE *&stab_order,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *data, nb_gens, data_size;
	INT i;

	if (f_v) {
		cout << "action::stabilizer_of_dual_hyperoval_representative" << endl;
		}
	DH_stab_gens(k, n, no, data, nb_gens, data_size, stab_order);

	gens = NEW_OBJECT(vector_ge);
	gens->init(this);
	gens->allocate(nb_gens);
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

void action::stabilizer_of_translation_plane_representative(
		INT q, INT k, INT no,
		vector_ge *&gens, const BYTE *&stab_order,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *data, nb_gens, data_size;
	INT i;

	if (f_v) {
		cout << "action::stabilizer_of_translation_plane_representative"
				<< endl;
		}
	Spread_stab_gens(q, k, no, data, nb_gens, data_size, stab_order);

	gens = NEW_OBJECT(vector_ge);
	gens->init(this);
	gens->allocate(nb_gens);
	if (f_vv) {
		cout << "action::stabilizer_of_translation_plane_representative "
				"creating stabilizer generators:" << endl;
		}
	for (i = 0; i < nb_gens; i++) {
		make_element(gens->ith(i), data + i * data_size, 0 /*verbose_level*/);
		}
	
	if (f_v) {
		cout << "action::stabilizer_of_translation_plane_representative done"
				<< endl;
		}
}

void action::normalizer_using_MAGMA(const BYTE *prefix,
		sims *G, sims *H, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname_magma[1000];
	BYTE fname_output[1000];
	BYTE cmd[1000];

	if (f_v) {
		cout << "action::normalizer_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%snormalizer.magma", prefix);
	sprintf(fname_output, "%snormalizer.txt", prefix);

	INT n;

	strong_generators *G_gen;
	strong_generators *H_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	H_gen = NEW_OBJECT(strong_generators);
	H_gen->init_from_sims(H, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::normalizer_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);
	
	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;


	fp << "H := sub< G |" << endl;
	H_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;
	
	fp << "N := Normalizer(G, H);" << endl;
	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #N; printf \"\\n\";" << endl;
	fp << "printf \"%o\", #Generators(N); printf \"\\n\";" << endl;
	fp << "for h := 1 to #Generators(N) do for i := 1 to "
			<< n << " do printf \"%o\", i^N.h; printf \", \"; "
			"if i mod 25 eq 0 then printf \"\n\"; end if; "
			"end for; printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}
	sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
	cout << "executing normalizer command in MAGMA" << endl;
	system(cmd);

	cout << "normalizer command in MAGMA has finished" << endl;
		
	if (f_v) {
		cout << "action::normalizer_using_MAGMA done" << endl;
		}
}

void action::conjugacy_classes_using_MAGMA(const BYTE *prefix,
		sims *G, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname_magma[1000];
	BYTE fname_output[1000];
	BYTE cmd[1000];

	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%sconjugacy_classes.magma", prefix);
	sprintf(fname_output, "%sconjugacy_classes.txt", prefix);

	INT n;

	strong_generators *G_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);
	
	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;


	fp << "C := ConjugacyClasses(G);" << endl;

	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #C; printf \"\\n\";" << endl;
	fp << "for h := 1 to #C do  printf \"%o\", C[h][1]; printf \" \";"
			"printf \"%o\", C[h][2]; printf \" \";   for i := 1 to "
			<< n << " do printf \"%o\", i^C[h][3]; printf \" \"; end for; "
			"printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}
	sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
	cout << "executing ConjugacyClasses command in MAGMA" << endl;
	system(cmd);

	cout << "command ConjugacyClasses in MAGMA has finished" << endl;
	
	FREE_OBJECT(G_gen);
	
	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA done" << endl;
		}
}

void action::centralizer_using_MAGMA(const BYTE *prefix,
		sims *G, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname_magma[1000];
	BYTE fname_output[1000];
	BYTE cmd[1000];

	if (f_v) {
		cout << "action::centralizer_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%scentralizer.magma", prefix);
	sprintf(fname_output, "%scentralizer.txt", prefix);

	INT n;

	strong_generators *G_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::centralizer_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);
	
	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;

	fp << "h := G ! ";
	element_print_as_permutation_with_offset(Elt, fp, 
			1 /* offset */, TRUE /* f_do_it_anyway_even_for_big_degree */, 
			FALSE /* f_print_cycles_of_length_one */, 0 /* verbose_level */);
	fp << ";" << endl;

	fp << "C := Centralizer(G, h);" << endl;

	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #C; printf \"\\n\";" << endl;
	fp << "printf \"%o\", #Generators(C); printf \"\\n\";" << endl;
	fp << "for h := 1 to #Generators(C) do for i := 1 to "
			<< n << " do printf \"%o\", i^C.h; printf \" \"; end for;"
			" printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}
	cout << "Written file " << fname_magma
			<< " of size " << file_size(fname_magma) << endl;
	
	sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
	cout << "executing centralizer command in MAGMA" << endl;
	system(cmd);

	cout << "command centralizer in MAGMA has finished" << endl;
	
	FREE_OBJECT(G_gen);
	
	if (f_v) {
		cout << "action::centralizer_using_MAGMA done" << endl;
		}
}

void action::point_stabilizer_any_point(INT &pt, 
	schreier *&Sch, sims *&Stab, strong_generators *&stab_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::point_stabilizer_any_point" << endl;
		}
	
	INT f; //, len;
	longinteger_object go;
	
	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"computing all point orbits:" << endl;
		}
	Sch = Strong_gens->orbits_on_points_schreier(this, 0 /* verbose_level */);
	//compute_all_point_orbits(Sch, *Strong_gens->gens, 0 /* verbose_level */);
	cout << "computing all point orbits done, found "
			<< Sch->nb_orbits << " orbits" << endl;


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action::point_stabilizer_any_point orbit rep = "
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
	INT &pt, 
	schreier *&Sch, sims *&Stab, strong_generators *&stab_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group" << endl;
		}
	
	INT f; //, len;
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
		INT line_rk, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::make_element_which_moves_a_line_in_PG3q" << endl;
		}

	INT M[4 * 4];
	INT N[4 * 4 + 1]; // + 1 if f_semilinear
	INT base_cols[4];
	INT r, c, i, j;

	//INT_vec_zero(M, 16);
	Gr->unrank_INT_here(M, line_rk, 0 /*verbose_level*/);
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
		cout << "action::make_element_which_moves_a_line_in_PG3q done" << endl;
		}
}

void action::list_elements_as_permutations_vertically(vector_ge *gens,
		ostream &ost)
{
	INT i, j, a, len;

	len = gens->len;
	for (j = 0; j < len; j++) {
		ost << " & \\alpha_{" << j << "}";
	}
	ost << "\\\\" << endl;
	for (i = 0; i < degree; i++) {
		ost << setw(3) << i;
		for (j = 0; j < len; j++) {
			a = element_image_of(i, gens->ith(j), 0 /* verbose_level */);
			ost << " & " << setw(3) << a;
		}
		ost << "\\\\" << endl;
	}
}



