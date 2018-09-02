// schreier.C
//
// Anton Betten
// December 9, 2003

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

schreier::schreier()
{
	A = NULL;
	nb_images = 0;
	images = NULL;
	orbit = NULL;
	orbit_inv = NULL;
	prev = NULL;
	label = NULL;
	orbit_first = NULL;
	orbit_len = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	schreier_gen = NULL;
	schreier_gen1 = NULL;
	cosetrep = NULL;
	cosetrep_tmp = NULL;
	f_print_function = FALSE;
	print_function = NULL;
	print_function_data = NULL;
	nb_orbits = 0;
}

schreier::schreier(action *A)
{
	init(A);
};

schreier::~schreier()
{
	//cout << "in ~schreier()" << endl;
	freeself();
	//cout << "~schreier() finished" << endl;
};

void schreier::freeself()
{
	//cout << "deleting A" << endl;
	if (A) {
		//cout << "deleting orbit" << endl;
		FREE_INT(orbit);
		//cout << "deleting orbit_inv" << endl;
		FREE_INT(orbit_inv);
		//cout << "deleting prev" << endl;
		FREE_INT(prev);
		//cout << "deleting label" << endl;
		FREE_INT(label);
		//cout << "deleting orbit_first" << endl;
		FREE_INT(orbit_first);
		//cout << "deleting orbit_len" << endl;
		FREE_INT(orbit_len);
		//cout << "deleting Elt1" << endl;
		FREE_INT(Elt1);
		//cout << "deleting Elt2" << endl;
		FREE_INT(Elt2);
		//cout << "deleting Elt3" << endl;
		FREE_INT(Elt3);
		//cout << "deleting schreier_gen" << endl;
		FREE_INT(schreier_gen);
		//cout << "deleting schreier_gen1" << endl;
		FREE_INT(schreier_gen1);
		//cout << "deleting cosetrep" << endl;
		FREE_INT(cosetrep);
		//cout << "deleting cosetrep_tmp" << endl;
		FREE_INT(cosetrep_tmp);
		//cout << "A = NULL" << endl;
		A = NULL;
		}
	//cout << "deleting images" << endl;
	delete_images();
}

void schreier::delete_images()
{
	INT i;
	
	if (images) {
		for (i = 0; i < nb_images; i++) {
			FREE_INT(images[i]);
			}
		FREE_PINT(images);
		images = NULL;
		nb_images = 0;
		}
}

void schreier::init_images(INT nb_images, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
	if (f_v) {
		cout << "schreier::init_images" << endl;
		}
	if (A == NULL) {
		cout << "schreier::init_images() action is NULL" << endl;
		exit(1);
		}
	delete_images();
	schreier::nb_images = nb_images;
	images = NEW_PINT(nb_images);
	for (i = 0; i < nb_images; i++) {
		images[i] = NEW_INT(2 * A->degree);
		for (j = 0; j < 2 * A->degree; j++) {
			images[i][j] = -1;
			}
		}
	if (f_v) {
		cout << "schreier::init_images done" << endl;
		}
}

void schreier::images_append()
{
	INT **new_images = NEW_PINT(nb_images + 1);
	INT i, j;
	
	new_images[nb_images] = NEW_INT(2 * A->degree);
	for (j = 0; j < 2 * A->degree; j++) {
		new_images[nb_images][j] = -1;
		}
	for (i = 0; i < nb_images; i++) {
		new_images[i] = images[i];
		}
	FREE_PINT(images);
	images = new_images;
	nb_images++;
}

void schreier::init(action *A)
{
	schreier::A = A;
	orbit = NEW_INT(A->degree);
	orbit_inv = NEW_INT(A->degree);
	prev = NEW_INT(A->degree);
	label = NEW_INT(A->degree);
	//orbit_no = NEW_INT(A->degree);
	orbit_first = NEW_INT(A->degree + 1);
	orbit_len = NEW_INT(A->degree);
	gens.init(A);
	gens_inv.init(A);
	initialize_tables();
	init2();
}

void schreier::init2()
{
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	schreier_gen = NEW_INT(A->elt_size_in_INT);
	schreier_gen1 = NEW_INT(A->elt_size_in_INT);
	cosetrep = NEW_INT(A->elt_size_in_INT);
	cosetrep_tmp = NEW_INT(A->elt_size_in_INT);
}

void schreier::initialize_tables()
{
	INT i;
	
	nb_orbits = 0;
	perm_identity(orbit, A->degree);
	perm_identity(orbit_inv, A->degree);
	orbit_first[0] = 0;
	for (i = 0; i < A->degree; i++) {
		prev[i] = -1;
		label[i] = -1;
		//orbit_no[i] = -1;
		}
}

void schreier::init_single_generator(INT *elt)
{
	init_generators(1, elt);
}

void schreier::init_generators(vector_ge &generators)
{
	if (generators.len) {
		init_generators(generators.len,
				generators.ith(0));
		}
	else {
		init_generators(generators.len, NULL);
		}
}

void schreier::init_generators(INT nb, INT *elt)
// elt must point to nb * A->elt_size_in_INT
// INT's that are
// group elements in INT format
{
	INT i;
	
	gens.allocate(nb);
	gens_inv.allocate(nb);
	for (i = 0; i < nb; i++) {
		//cout << "schreier::init_generators i = " << i << endl;
		gens.copy_in(i, elt + i * A->elt_size_in_INT);
		A->element_invert(elt + i * A->elt_size_in_INT,
				gens_inv.ith(i), 0);
		}
	init_images(nb, 0 /* verbose_level */);	
}

void schreier::init_generators_by_hdl(INT nb_gen, 
	INT *gen_hdl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "schreier::init_generators_by_hdl" << endl;
		cout << "nb_gen = " << nb_gen << endl;
		cout << "degree = " << A->degree << endl;
		}
	gens.allocate(nb_gen);
	gens_inv.allocate(nb_gen);
	for (i = 0; i < nb_gen; i++) {
		//cout << "schreier::init_generators_by_hdl "
		// "i = " << i << endl;
		A->element_retrieve(gen_hdl[i], gens.ith(i), 0);
		
		//cout << "schreier::init_generators_by_hdl "
		// "generator i = " << i << ":" << endl;
		//A->element_print_quick(gens.ith(i), cout);

		A->element_invert(gens.ith(i), gens_inv.ith(i), 0);
		}
	if (f_vv) {
		cout << "schreier::init_generators_by_hdl "
				"generators:" << endl;
		gens.print(cout);
		}
	if (f_v) {
		cout << "schreier::init_generators_by_hdl "
				"before init_images()" << endl;
		}
	init_images(nb_gen, verbose_level);	
	if (f_v) {
		cout << "schreier::init_generators_by_hdl "
				"done" << endl;
		}
}

INT schreier::get_image(INT i, INT gen_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a;
	
	if (f_v) {
		cout << "schreier::get_image computing image of point "
				<< i << " under generator " << gen_idx << endl;
		}
	if (images == NULL) {
		a = A->element_image_of(i,
				gens.ith(gen_idx), verbose_level - 2);
		//cout << "schreier::get_image"
		// "images == NULL" << endl;
		//exit(1);
		}
	else {
		a = images[gen_idx][i];
		if (a == -1) {
			a = A->element_image_of(i,
					gens.ith(gen_idx), verbose_level - 2);
			if (f_v) {
				cout << "schreier::get_image image of "
						"i=" << i << " is " << a << endl;
				}
			images[gen_idx][i] = a;
			images[gen_idx][A->degree + a] = i;
			}
		}
	if (f_v) {
		cout << "schreier::get_image image of point "
				<< i << " under generator " << gen_idx << " is " << a << endl;
		}
	return a;
}

void schreier::print_orbit_lengths(ostream &ost)
{
	INT i, f, l, m;
	INT *orbit_len_sorted;
	INT *sorting_perm;
	INT *sorting_perm_inv;
	INT nb_types;
	INT *type_first;
	INT *type_len;
	
	INT_vec_classify(nb_orbits, orbit_len, orbit_len_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);

	ost << nb_orbits << " orbits: " << endl;
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		m = orbit_len_sorted[f];
		if (l > 1) {
			cout << l << " \\times ";
			}
		cout << m;
		if (i < nb_types - 1)
			cout << ", ";
		}
	ost << endl;
	FREE_INT(orbit_len_sorted);
	FREE_INT(sorting_perm);
	FREE_INT(sorting_perm_inv);
	FREE_INT(type_first);
	FREE_INT(type_len);
	
}

void schreier::print_orbit_length_distribution(ostream &ost)
{
	INT *val, *mult, len;	
	
	INT_vec_distribution(orbit_len, nb_orbits, val, mult, len);
	INT_distribution_print(ost, val, mult, len);
	ost << endl;
	
	FREE_INT(val);
	FREE_INT(mult);
}

void schreier::print_orbit_reps(ostream &ost)
{
	INT i, c, r;
	
	ost << nb_orbits << " orbits" << endl;
	ost << "orbits of a group with " << gens.len
			<< " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i] : rep" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << setw(3) << i << " : " << setw(6)
				<< orbit_first[i] << " : " << setw(6) << orbit_len[i];
		c = orbit_first[i];
		r = orbit[c];
		ost << " : " << setw(6) << r << endl;
		//<< " : ";
		//print_orbit(ost, i);
		//ost << endl;
		}
	ost << endl;
}


void schreier::print(ostream &ost)
{
	INT i;
	
	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << i << " : " << orbit_first[i] << " : "
				<< orbit_len[i] << endl;
		//<< " : ";
		//print_orbit(ost, i);
		//ost << endl;
		}
	ost << endl;
}

void schreier::print_and_list_orbits_and_stabilizer(ostream &ost, 
	action *default_action, longinteger_object &go, 
	void (*print_point)(ostream &ost, INT pt, void *data), 
	void *data)
{
	INT i;
	
	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {

		sims *Stab;
		strong_generators *SG;

		ost << "Orbit " << i << " / " << nb_orbits
				<< " : of length " << orbit_len[i];
		ost << " is:" << endl;
		print_orbit(ost, i);
		ost << endl;
		ost << "Which is:" << endl;
		print_orbit_using_callback(ost, i, print_point, data);
		//ost << endl;
		ost << "The stabilizer of the element "
				<< orbit[orbit_first[i]] << " is:" << endl;
		point_stabilizer(default_action, go, Stab, i, 0 /* verbose_level */);

		SG = NEW_OBJECT(strong_generators);

		SG->init_from_sims(Stab, 0 /* verbose_level*/);
		SG->print_generators_ost(ost);
		FREE_OBJECT(SG);
		FREE_OBJECT(Stab);
		}
	ost << endl;
}

void schreier::print_and_list_orbits(ostream &ost)
{
	INT i;
	
	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << " Orbit " << i << " / " << nb_orbits
				<< " : " << orbit_first[i] << " : " << orbit_len[i];
		ost << " : ";
		print_orbit(ost, i);
		ost << endl;
		}
	ost << endl;
}

void schreier::print_and_list_orbits_tex(ostream &ost)
{
	INT i;
	
	ost << nb_orbits << " orbits:\\\\" << endl;
	ost << "orbits under a group with " << gens.len
			<< " generators acting on a set of size "
			<< A->degree << ":\\\\" << endl;
	//ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		print_and_list_orbit_tex(i, ost);
		}
	ost << endl;
}

void schreier::print_and_list_orbit_tex(INT i, ostream &ost)
{
	ost << " Orbit " << i << " / " << nb_orbits << " : ";
	print_orbit_tex(ost, i);
	ost << " of length " << orbit_len[i] << "\\\\" << endl;
}

void schreier::print_and_list_orbit_and_stabilizer_tex(INT i, 
	action *default_action, 
	longinteger_object &full_group_order, ostream &ost)
{
	ost << " Orbit " << i << " / " << nb_orbits << " : ";
	print_orbit_tex(ost, i);
	ost << " of length " << orbit_len[i] << "\\\\" << endl;

	strong_generators *gens;

	gens = generators_for_stabilizer_of_orbit_rep(default_action, 
		full_group_order, i, 0 /*verbose_level */);
	
	gens->print_generators_tex(ost);

	FREE_OBJECT(gens);
}

void schreier::print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
	INT i, action *default_action, 
	strong_generators *gens, ostream &ost)
{
	longinteger_object full_group_order;

	gens->group_order(full_group_order);
	
	ost << " Orbit " << i << " / " << nb_orbits << " : ";
	print_orbit_tex(ost, i);
	ost << " of length " << orbit_len[i] << "\\\\" << endl;

	strong_generators *gens_stab;

	gens_stab = generators_for_stabilizer_of_orbit_rep(
		default_action,
		full_group_order, i, 0 /*verbose_level */);
	
	gens_stab->print_generators_tex(ost);

	INT *Subgroup_elements_by_index;
	INT sz_subgroup;

	sz_subgroup = gens_stab->group_order_as_INT();
	
	if (sz_subgroup < 20) {
		gens->list_of_elements_of_subgroup(gens_stab, 
			Subgroup_elements_by_index, sz_subgroup,
			0 /* verbose_level */);

		INT_vec_heapsort(Subgroup_elements_by_index,
				sz_subgroup);

		ost << "The subgroup consists of the following "
				<< sz_subgroup << " elements:" << endl;
		ost << "$$" << endl;
		INT_vec_print_as_matrix(ost,
				Subgroup_elements_by_index, sz_subgroup,
				10 /* width */, TRUE /* f_tex */);
		ost << "$$" << endl;

		FREE_INT(Subgroup_elements_by_index);

		}

	FREE_OBJECT(gens_stab);
}

void schreier::print_and_list_orbits_sorted_by_length_tex(
		ostream &ost)
{
	print_and_list_orbits_sorted_by_length(ost, TRUE);
}

void schreier::print_and_list_orbits_sorted_by_length(
		ostream &ost)
{
	print_and_list_orbits_sorted_by_length(ost, FALSE);
}

void schreier::print_and_list_orbits_sorted_by_length(
	ostream &ost, INT f_tex)
{
	INT i, h;
	INT *Len;
	INT *Perm;
	INT *Perm_inv;

	Len = NEW_INT(nb_orbits);
	Perm = NEW_INT(nb_orbits);
	Perm_inv = NEW_INT(nb_orbits);
	INT_vec_copy(orbit_len, Len, nb_orbits);
	INT_vec_sorting_permutation(Len, nb_orbits,
			Perm, Perm_inv, TRUE /*f_increasingly*/);
	
	ost << "There are " << nb_orbits
			<< " orbits under a group with "
			<< gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
		}
	else {
		ost << endl;
		}
	ost << "Orbit lengths: ";
	INT_vec_print(ost, orbit_len, nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
		}
	else {
		ost << endl;
		}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
		}
	for (h = 0; h < nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			print_and_list_orbit_tex(i, ost);
			}
		else {
			ost << " Orbit " << h << " / " << nb_orbits
					<< " is " << i << " : " << orbit_len[i];
			ost << " : ";
			print_orbit(ost, i);
			ost << endl;
			}
		}
	ost << endl;

	FREE_INT(Len);
	FREE_INT(Perm);
	FREE_INT(Perm_inv);
}

void schreier::print_and_list_orbits_and_stabilizer_sorted_by_length(
	ostream &ost, INT f_tex, 
	action *default_action,
	longinteger_object &full_group_order)
{
	INT i, h;
	INT *Len;
	INT *Perm;
	INT *Perm_inv;

	Len = NEW_INT(nb_orbits);
	Perm = NEW_INT(nb_orbits);
	Perm_inv = NEW_INT(nb_orbits);
	INT_vec_copy(orbit_len, Len, nb_orbits);
	INT_vec_sorting_permutation(Len, nb_orbits,
			Perm, Perm_inv, TRUE /*f_increasingly*/);
	
	ost << "There are " << nb_orbits << " orbits under a group with "
			<< gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
		}
	else {
		ost << endl;
		}
	ost << "Orbit lengths: ";
	INT_vec_print(ost, orbit_len, nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
		}
	else {
		ost << endl;
		}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
		}
	for (h = 0; h < nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			print_and_list_orbit_and_stabilizer_tex(i,
					default_action, full_group_order, ost);
			}
		else {
			ost << " Orbit " << h << " / " << nb_orbits
					<< " is " << i << " : " << orbit_len[i];
			ost << " : ";
			print_orbit(ost, i);
			ost << endl;
			}
		}
	ost << endl;

	FREE_INT(Len);
	FREE_INT(Perm);
	FREE_INT(Perm_inv);
}

void schreier::print_and_list_orbits_and_stabilizer_sorted_by_length_and_list_stabilizer_elements(
	ostream &ost, INT f_tex, 
	action *default_action, 
	strong_generators *gens_full_group)
{
	INT i, h;
	INT *Len;
	INT *Perm;
	INT *Perm_inv;
	longinteger_object full_group_order;

	gens_full_group->group_order(full_group_order);
	Len = NEW_INT(nb_orbits);
	Perm = NEW_INT(nb_orbits);
	Perm_inv = NEW_INT(nb_orbits);
	INT_vec_copy(orbit_len, Len, nb_orbits);
	INT_vec_sorting_permutation(Len, nb_orbits,
			Perm, Perm_inv, TRUE /*f_increasingly*/);
	
	ost << "There are " << nb_orbits << " orbits under a group with "
			<< gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
		}
	else {
		ost << endl;
		}
	ost << "Orbit lengths: ";
	INT_vec_print(ost, orbit_len, nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
		}
	else {
		ost << endl;
		}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
		}
	for (h = 0; h < nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			//print_and_list_orbit_and_stabilizer_tex(
			// i, default_action, full_group_order, ost);
			print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
				i, default_action,
				gens_full_group, ost);
			}
		else {
			ost << " Orbit " << h << " / " << nb_orbits
					<< " is " << i << " : " << orbit_len[i];
			ost << " : ";
			print_orbit(ost, i);
			ost << endl;
			}
		}
	ost << endl;

	FREE_INT(Len);
	FREE_INT(Perm);
	FREE_INT(Perm_inv);
}

void schreier::print_and_list_orbits_of_given_length(
	ostream &ost, INT len)
{
	INT i;

	
	ost << "Orbits of length " << len << ":" << endl;
	cout << "i : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_len[i] != len) {
			continue;
			}
		ost << " Orbit " << i << " / "
				<< nb_orbits << " : " << orbit_len[i];
		ost << " : ";
		print_orbit(ost, i);
		ost << endl;
		}
	ost << endl;
}

void schreier::print_and_list_orbits_using_labels(
		ostream &ost, INT *labels)
{
	INT i;
	
	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << i << " : " << orbit_first[i]
			<< " : " << orbit_len[i];
		ost << " : ";
		print_orbit_using_labels(ost, i, labels);
		ost << endl;
		}
	ost << endl;
}

void schreier::print_tables(ostream &ost, 
	INT f_with_cosetrep)
{
    INT i;
    int w; //  j, k;
	
#if 0
	ost << gens.len << " generators:" << endl;
	for (i = 0; i < A->degree; i++) {
		ost << i;
		for (j = 0; j < gens.len; j++) {
			k = A->element_image_of(i, gens.ith(j), FALSE);
			ost << " : " << k;
			}
		ost << endl;
		}
	ost << endl;
#endif
	w = (int) INT_log10(A->degree) + 1;
	ost << "i : orbit[i] : orbit_inv[i] : prev[i] : label[i]";
	if (f_with_cosetrep)
		ost << " : coset_rep";
	ost << endl;
	for (i = 0; i < A->degree; i++) {
		coset_rep(i);
		//coset_rep_inv(i);
		ost << setw(w) << i << " : " << " : " 
			<< setw(w) << orbit[i] << " : "
			<< setw(w) << orbit_inv[i] << " : "
			<< setw(w) << prev[i] << " : "
			<< setw(w) << label[i];
		if (f_with_cosetrep) {
			ost << " : ";
			//A->element_print(Elt1, cout);
			A->element_print_as_permutation(cosetrep, ost);
			ost << endl;
			A->element_print_quick(cosetrep, ost);
			}
		ost << endl;
		}
	ost << endl;
}

void schreier::print_tables_latex(ostream &ost, 
	INT f_with_cosetrep)
{
    INT i;
    int w; //  j, k;
	
#if 0
	ost << gens.len << " generators:" << endl;
	for (i = 0; i < A->degree; i++) {
		ost << i;
		for (j = 0; j < gens.len; j++) {
			k = A->element_image_of(i, gens.ith(j), FALSE);
			ost << " : " << k;
			}
		ost << endl;
		}
	ost << endl;
#endif
	w = (int) INT_log10(A->degree) + 1;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|" << endl;
	if (f_with_cosetrep) {
		ost << "c|";
		}
	ost << "}" << endl;
	ost << "\\hline" << endl;
	ost << "i & orbit & orbitinv & prev & label";
	if (f_with_cosetrep) {
		ost << "& cosetrep";
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < A->degree; i++) {
		coset_rep(i);
		//coset_rep_inv(i);
		ost << i << " & " 
			<< setw(w) << orbit[i] << " & "
			<< setw(w) << orbit_inv[i] << " & "
			<< setw(w) << prev[i] << " & "
			<< setw(w) << label[i];
		if (f_with_cosetrep) {
			ost << " & ";
			//A->element_print(Elt1, cout);
			//A->element_print_as_permutation(cosetrep, ost);
			//ost << endl;
			A->element_print_latex(cosetrep, ost);
			}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;

		if (((i + 1) % 10) == 0) {
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|c|c|" << endl;
			if (f_with_cosetrep) {
				ost << "c|";
				}
			ost << "}" << endl;
			ost << "\\hline" << endl;
			ost << "i & orbit & orbitinv & prev & label";
			if (f_with_cosetrep) {
				ost << "& cosetrep";
				}
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
			}
		}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << endl;
}

void schreier::print_generators()
{
	INT j;
	
	cout << gens.len << " generators in action "
			<< A->label << " of degree " << A->degree << ":" << endl;
	for (j = 0; j < gens.len; j++) {
		cout << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		A->element_print_quick(gens.ith(j), cout);
		//A->element_print_as_permutation(gens.ith(j), cout);
		if (j < gens.len - 1) {
			cout << ", " << endl;
			}
		}
}

void schreier::print_generators_with_permutations()
{
	INT j;
	
	cout << gens.len << " generators in action "
			<< A->label << " of degree " << A->degree << ":" << endl;
	for (j = 0; j < gens.len; j++) {
		cout << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		A->element_print_quick(gens.ith(j), cout);
		A->element_print_as_permutation(gens.ith(j), cout);
		cout << endl;
		if (j < gens.len - 1) {
			cout << ", " << endl;
			}
		}
}

void schreier::print_orbit(INT orbit_no)
{
	print_orbit(cout, orbit_no);
}

void schreier::print_orbit_using_labels(
		INT orbit_no, INT *labels)
{
	print_orbit_using_labels(cout, orbit_no, labels);
}

void schreier::print_orbit(ostream &ost, INT orbit_no)
{
	INT i, first, len;
	INT *v;
	
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_INT(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
		}
	//INT_vec_print(ost, v, len);
	//INT_vec_heapsort(v, len);
	INT_vec_print_fully(ost, v, len);
	
	FREE_INT(v);
}

void schreier::print_orbit_tex(ostream &ost, INT orbit_no)
{
	INT i, first, len;
	INT *v;
	
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_INT(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
		}
	//INT_vec_print(ost, v, len);
	//INT_vec_heapsort(v, len);
	//INT_vec_print_fully(ost, v, len);
	INT_set_print_tex(ost, v, len);
	
	FREE_INT(v);
}

void schreier::print_orbit_using_labels(ostream &ost, 
	INT orbit_no, INT *labels)
{
	INT i, first, len;
	INT *v;
	
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_INT(len);
	for (i = 0; i < len; i++) {
		v[i] = labels[orbit[first + i]];
		}
	//INT_vec_print(ost, v, len);
	INT_vec_heapsort(v, len);
	INT_vec_print_fully(ost, v, len);
	
	FREE_INT(v);
}

void schreier::print_orbit_using_callback(ostream &ost, 
	INT orbit_no, 
	void (*print_point)(ostream &ost, INT pt, void *data), 
	void *data)
{
	INT i, first, len;
	
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	for (i = 0; i < len; i++) {
		ost << orbit[first + i] << " which is " << endl;
		(*print_point)(ost, orbit[first + i], data);
		}
}

void schreier::print_orbit_type(INT f_backwards)
{
	classify C;

	C.init(orbit_len, nb_orbits, FALSE, 0);
	C.print_naked(f_backwards);
}

void schreier::list_all_orbits_tex(ostream &ost)
{
	INT i, j, f, l, a;

	ost << "$";
	for (i = 0; i < nb_orbits; i++) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			a = orbit[f + j];
			ost << a;
			if (j < l - 1) {
				ost << ", ";
				}
			}
		if (i < nb_orbits - 1) {
			ost << " \\mid ";
			}
		}
	ost << "$";
}

void schreier::print_orbit_through_labels(ostream &ost, 
	INT orbit_no, INT *point_labels)
{
	INT i, first, len;
	INT *v;
	
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_INT(len);
	for (i = 0; i < len; i++) {
		v[i] = point_labels[orbit[first + i]];
		}
	INT_vec_heapsort(v, len);
	INT_vec_print_fully(ost, v, len);
	FREE_INT(v);
}

void schreier::print_orbit_sorted(ostream &ost, INT orbit_no)
{
	INT i, len;
	INT *v;
	
	len = orbit_first[orbit_no + 1] - orbit_first[orbit_no];
	v = NEW_INT(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[orbit_first[orbit_no] + i];
		}
	INT_vec_sort(len, v);
	
	ost << "{ ";
	for (i = 0; i < len; i++) {
		if (f_print_function) {
			ost << v[i] << "=";
			(*print_function)(ost, v[i], print_function_data);
			}
		else {
			ost << v[i];
			}
		if (i < len - 1)
			ost << ", ";
		}
	ost << " }";
	FREE_INT(v);
}

void schreier::print_orbit(INT cur, INT last)
{
	INT i;
	
	for (i = 0; i < A->degree; i++) {
		if (i == cur) 
			cout << ">";
		if (i == last)
			cout << ">";
		cout << i << " : " << orbit[i]
			<< " : " << orbit_inv[i] << endl;
		}
	cout << endl;
}

void schreier::swap_points(INT i, INT j)
{
	INT pi, pj;
	
	pi = orbit[i];
	pj = orbit[j];
	orbit[i] = pj;
	orbit[j] = pi;
	orbit_inv[pi] = j;
	orbit_inv[pj] = i;
}

void schreier::move_point_here(INT here, INT pt)
{
	INT a, loc;
	if (orbit[here] == pt)
		return;
	a = orbit[here];
	loc = orbit_inv[pt];
	orbit[here] = pt;
	orbit[loc] = a;
	orbit_inv[a] = loc;
	orbit_inv[pt] = here;
}

INT schreier::orbit_representative(INT pt)
{
	INT j;
	
	while (TRUE) {
		j = orbit_inv[pt];
		if (prev[j] == -1)
			return pt;
		pt = prev[j];
		}
}

INT schreier::depth_in_tree(INT j)
// j is a coset, not a point
{
	if (prev[j] == -1) {
		return 0;
		}
	else {
		return depth_in_tree(orbit_inv[prev[j]]) + 1;
		}
}

void schreier::transporter_from_orbit_rep_to_point(INT pt, 
	INT &orbit_idx, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT pos;

	if (f_v) {
		cout << "schreier::transporter_from_orbit_"
				"rep_to_point" << endl;
		}
	pos = orbit_inv[pt];
	orbit_idx = orbit_number(pt); //orbit_no[pos];
	//cout << "lies in orbit " << orbit_idx << endl;
	coset_rep(pos);
	A->element_move(cosetrep, Elt, 0);
	if (f_v) {
		cout << "schreier::transporter_from_orbit_"
				"rep_to_point done" << endl;
		}
}

void schreier::transporter_from_point_to_orbit_rep(INT pt, 
	INT &orbit_idx, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT pos;

	if (f_v) {
		cout << "schreier::transporter_from_point_"
				"to_orbit_rep" << endl;
		}
	pos = orbit_inv[pt];
	orbit_idx = orbit_number(pt); //orbit_no[pos];
	//cout << "lies in orbit " << orbit_idx << endl;
	coset_rep_with_verbosity(pos, verbose_level - 1);
	A->element_invert(cosetrep, Elt, 0);
	//A->element_move(cosetrep, Elt, 0);
	if (f_v) {
		cout << "schreier::transporter_from_point_to_orbit_rep "
				"done" << endl;
		}
}

void schreier::coset_rep(INT j)
// j is a coset, not a point
// result is in cosetrep
// determines an element in the group
// that moves the orbit representative
// to the j-th point in the orbit.
{
	INT *gen;
	
	if (prev[j] != -1) {
		coset_rep(orbit_inv[prev[j]]);
		gen = gens.ith(label[j]);
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		}
	else {
		A->element_one(cosetrep, 0);
		}
}

void schreier::coset_rep_with_verbosity(INT j, INT verbose_level)
// j is a coset, not a point
// result is in cosetrep
// determines an element in the group
// that moves the orbit representative
// to the j-th point in the orbit.
{
	INT f_v = (verbose_level >= 1);
	INT *gen;
	
	if (f_v) {
		cout << "schreier::coset_rep_with_verbosity j="
				<< j << " orbit[j]=" << orbit[j] << endl;
		}
	if (prev[j] != -1) {
		if (f_v) {
			cout << "schreier::coset_rep_with_verbosity j=" << j
					<< " label[j]=" << label[j]
					<< " orbit_inv[prev[j]]="
					<< orbit_inv[prev[j]] << endl;
			}
		coset_rep_with_verbosity(orbit_inv[prev[j]], verbose_level);
		gen = gens.ith(label[j]);
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		}
	else {
		A->element_one(cosetrep, 0);
		}
	if (f_v) {
		cout << "schreier::coset_rep_with_verbosity "
				"j=" << j << " done" << endl;
		}
}

void schreier::coset_rep_inv(INT j)
// j is a coset, not a point
// result is in cosetrep
{
	INT *gen;
	
	if (prev[j] != -1) {
		coset_rep_inv(orbit_inv[prev[j]]);
		gen = gens_inv.ith(label[j]);
		A->element_mult(gen, cosetrep, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		}
	else {
		A->element_one(cosetrep, 0);
		}
}

void schreier::get_schreier_vector(INT *&sv, 
	INT f_trivial_group, INT f_compact)
{
	if (f_compact) {
		get_schreier_vector_compact(sv, f_trivial_group);
		}
	else {
		get_schreier_vector_ordinary(sv);
		}
}

void schreier::get_schreier_vector_compact(INT *&sv, 
	INT f_trivial_group)
// allocated and creates array sv[size] using NEW_INT
// where size is n + 1 if  f_trivial_group is TRUE
// and size is 3 * n + 1 otherwise
// Here, n is the combined size of all orbits counted by nb_orbits
// sv[0] is equal to n
// sv + 1 is the array point_list of size [n],
// listing the point in increasing order
// Unless f_trivial_group, sv + 1 + n is the array prev[n] and 
// sv + 1 + 2 * n is the array label[n] 
{
	INT i, j, k, f, ff, l, p, pr, la, n = 0;
	INT *point_list;
	
	for (k = 0; k < nb_orbits; k++) {
		n += orbit_len[k];
		}
	point_list = NEW_INT(n);
	
	ff = 0;
	for (k = 0; k < nb_orbits; k++) {
		f = orbit_first[k];
		l = orbit_len[k];
		for (j = 0; j < l; j++) {
			i = f + j;
			p = orbit[i];
			point_list[ff + j] = p;
			}
		ff += l;
		}
	if (ff != n) {
		cout << "schreier::get_schreier_vector_compact "
				"ff != n" << endl;
		exit(1);
		}
	INT_vec_heapsort(point_list, n);
	
	
	if (f_trivial_group) {
		sv = NEW_INT(n + 1);
		}
	else {
		sv = NEW_INT(3 * n + 1);
		}
	sv[0] = n;
	for (i = 0; i < n; i++) {
		sv[1 + i] = point_list[i];
		}
	if (!f_trivial_group) {
		for (i = 0; i < n; i++) {
			p = point_list[i];
			j = orbit_inv[p];
			pr = prev[j];
			la = label[j];
			sv[1 + n + i] = pr;
			sv[1 + 2 * n + i] = la;
			}
		}
	FREE_INT(point_list);
}

void schreier::get_schreier_vector_ordinary(INT *&sv)
// allocates and creates array sv[2 * A->degree] using NEW_INT
// sv[i * 2 + 0] is prev[i]
// sv[i * 2 + 1] is label[i]
{
	INT i, j;
	
	sv = NEW_INT(2 * A->degree);
	for (i = 0; i < A->degree; i++) {
		j = orbit_inv[i];
		if (prev[j] != -1) {
			sv[i * 2 + 0] = prev[j];
			sv[i * 2 + 1] = label[j];
			//cout << "label[" << i << "] = " << label[j] << endl;
			}
		else {
			sv[i * 2 + 0] = -1;
			sv[i * 2 + 1] = -1;
			}
		}
}

void schreier::extend_orbit(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT cur, total0, total, cur_pt;
	INT gen_first, i, next_pt, next_pt_loc;
	
	if (f_v) {
		cout << "extend_orbit() extending orbit "
				<< nb_orbits - 1 << " of length "
			<< orbit_len[nb_orbits - 1] << endl;
		}

	gens.append(elt);
	A->element_invert(elt, A->Elt1, FALSE);
	gens_inv.append(A->Elt1);
	images_append();
	
	cur = orbit_first[nb_orbits - 1];
	total = total0 = orbit_first[nb_orbits];
	while (cur < total) {
		cur_pt = orbit[cur];
		if (FALSE) {
			cout << "schreier::extend_orbit "
					"applying generator to " << cur_pt << endl;
			}
#if 0
		if (cur < total0)
			gen_first = gens.len - 1;
		else 
			gen_first = 0;
#endif
		gen_first = 0;
		for (i = gen_first; i < gens.len; i++) {
			next_pt = get_image(cur_pt, i, 0/*verbose_level - 3*/);
				// A->element_image_of(cur_pt, gens.ith(i), FALSE);
			next_pt_loc = orbit_inv[next_pt];
			if (FALSE) {
				cout << "schreier::extend_orbit generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
				}
			if (next_pt_loc < total)
				continue;
			if (FALSE) {
				cout << "schreier::extend_orbit n e w pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
				}
			swap_points(total, next_pt_loc);
			prev[total] = cur_pt;
			label[total] = i;
			//orbit_no[total] = nb_orbits - 1;
			total++;
			if (FALSE) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	orbit_first[nb_orbits] = total;
	orbit_len[nb_orbits - 1] = total - orbit_first[nb_orbits - 1];
	//orbit_first[nb_orbits + 1] = A->degree;
	//orbit_len[nb_orbits] = A->degree - total;
	if (f_v) {
		cout << "schreier::extend_orbit orbit extended to length "
				<< orbit_len[nb_orbits - 1] << endl;
		}
	if (FALSE) {
		cout << "{ ";
		for (i = orbit_first[nb_orbits - 1];
				i < orbit_first[nb_orbits]; i++) {
			cout << orbit[i];
			if (i < orbit_first[nb_orbits] - 1)
				cout << ", ";
			}
		cout << " }" << endl;
		}
}

void schreier::compute_all_point_orbits(INT verbose_level)
{
	INT pt, pt_loc, cur, pt0;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits "
				"verbose_level=" << verbose_level << endl;
		}
	if (A->degree > ONE_MILLION) {
		f_vv = FALSE;
		}
	initialize_tables();
	for (pt0 = 0, pt = 0; pt < A->degree; pt++) {
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		if (f_vv) {
			cout << "schreier::compute_all_point_orbits pt = "
					<< pt << " / " << A->degree
					<< " nb_orbits=" << nb_orbits
					<< " cur=" << cur
					<< ", computing orbit" << endl;
			}
		if (A->degree > ONE_MILLION && (pt - pt0) > 50000) {
			cout << "schreier::compute_all_point_orbits pt = "
					<< pt << " / " << A->degree
					<< " nb_orbits=" << nb_orbits
					<< " cur=" << cur
					<< ", computing orbit" << endl;
			pt0 = pt;
			}
		compute_point_orbit(pt, verbose_level - 2);
		}
	if (f_v) {
		cout << "schreier::compute_all_point_orbits found "
				<< nb_orbits << " orbits" << endl;
		classify Cl;

		Cl.init(orbit_len, nb_orbits, FALSE, 0);
		cout << "The distribution of orbit lengths is: ";
		Cl.print(FALSE);
		}
}

void schreier::compute_all_point_orbits_with_prefered_reps(
	INT *prefered_reps, INT nb_prefered_reps, 
	INT verbose_level)
{
	INT i, pt, pt_loc, cur;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_"
				"with_prefered_reps" << endl;
		}
	initialize_tables();
	for (i = 0; i < nb_prefered_reps; i++) {
		pt = prefered_reps[i];
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		compute_point_orbit(pt, verbose_level - 1);
		}
	for (pt = 0; pt < A->degree; pt++) {
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		compute_point_orbit(pt, verbose_level - 1);
		}
	if (f_v) {
		cout << "found " << nb_orbits << " orbit";
		if (nb_orbits != 1)
			cout << "s";
		cout << " on points" << endl;
		}
}


void schreier::compute_all_point_orbits_with_preferred_labels(
	INT *preferred_labels, INT verbose_level)
{
	INT pt, pt_loc, cur, a, i;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT *labels, *perm, *perm_inv;
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_"
				"preferred_labels" << endl;
		//cout << "preferred_labels :";
		//INT_vec_print(cout, preferred_labels, A->degree);
		//cout << endl;
		cout << "A->degree = " << A->degree << endl;
		}
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_"
				"preferred_labels allocating tables" << endl;
		}
	initialize_tables();
	labels = NEW_INT(A->degree);
	perm = NEW_INT(A->degree);
	perm_inv = NEW_INT(A->degree);
	for (i = 0; i < A->degree; i++) {
		labels[i] = preferred_labels[i];
		}
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_"
				"with_preferred_labels allocating tables done, "
				"sorting" << endl;
		}
	INT_vec_sorting_permutation(labels, A->degree,
			perm, perm_inv, TRUE /* f_increasingly */);

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_"
				"with_preferred_labels sorting done" << endl;
		}
	
	for (a = 0; a < A->degree; a++) {
		pt = perm_inv[a];
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		// now we need to make sure that the point pt
		// is moved to position cur:
		// actually this is not needed as the
		// function compute_point_orbit does this, too.
		swap_points(cur, pt_loc);
		
		if (f_v) {
			cout << "schreier::compute_all_point_orbits_with_"
					"preferred_labels computing orbit of point "
					<< pt << " = " << a << " / " << A->degree << endl;
			}
		compute_point_orbit(pt, verbose_level - 2);
		if (f_v) {
			cout << "schreier::compute_all_point_orbits_with_"
					"preferred_labels computing orbit of point "
					<< pt << " done, found an orbit of length "
					<< orbit_len[nb_orbits - 1]
					<< " nb_orbits = " << nb_orbits << endl;
			}
		}
	if (f_v) {
		cout << "found " << nb_orbits << " orbit";
		if (nb_orbits != 1)
			cout << "s";
		cout << " on points" << endl;
		}
	FREE_INT(labels);
	FREE_INT(perm);
	FREE_INT(perm_inv);
}

void schreier::compute_all_orbits_on_invariant_subset(
	INT len, INT *subset, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, f;
	
	if (f_v) {
		cout << "schreier::compute_all_orbits_on_invariant_subset" << endl;
		cout << "computing orbits on a set of size " << len << endl;
		}
	initialize_tables();
	for (i = 0; i < len; i++) {
		move_point_here(i, subset[i]);
		}
	while (TRUE) {
		f = orbit_first[nb_orbits];
		if (f >= len)
			break;
		compute_point_orbit(orbit[f], 0 /* verbose_level */);
		}
	if (f > len) {
		cout << "schreier::compute_all_orbits_on_invariant_subset "
				"the set is not G-invariant" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "found " << nb_orbits << " orbits" << endl;
		print_orbit_length_distribution(cout);
		}
}

INT schreier::sum_up_orbit_lengths()
{
	INT i, l, N;
	
		N = 0;
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		N += l;
		}
	return N;
}

void schreier::compute_point_orbit(INT pt, INT verbose_level)
{
	INT pt_loc, cur, cur_pt, total, i, next_pt;
	INT next_pt_loc, total1, cur1;
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; // (verbose_level >= 5);
	//INT f_vvv = FALSE; //(verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::compute_point_orbit" << endl;
		cout << "computing orbit of point " << pt
				<< " in action " << A->label << endl;
		}
	pt_loc = orbit_inv[pt];
	cur = orbit_first[nb_orbits];
	if (pt_loc < cur) {
		cout << "schreier::compute_point_orbit "
				"i < orbit_first[nb_orbits]" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "schreier::compute_point_orbit "
				"computing orbit of pt " << pt << endl;
		}
	if (pt_loc > orbit_first[nb_orbits]) {
		swap_points(orbit_first[nb_orbits], pt_loc);
		}
	//orbit_no[orbit_first[nb_orbits]] = nb_orbits;
	total = cur + 1;
	while (cur < total) {
		cur_pt = orbit[cur];
		if (f_vv) {
			cout << "schreier::compute_point_orbit cur="
					<< cur << " total=" << total
					<< " applying generators to " << cur_pt << endl;
			}
		for (i = 0; i < gens.len; i++) {
			if (f_vv) {
				cout << "schreier::compute_point_orbit "
						"applying generator "
						<< i << " to point " << cur_pt << endl;
				}
			next_pt = get_image(cur_pt, i,
				0 /*verbose_level - 5*/); // !!
				// A->element_image_of(cur_pt, gens.ith(i), FALSE);
			next_pt_loc = orbit_inv[next_pt];
			if (f_vv) {
				cout << "schreier::compute_point_orbit generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
				}
			if (next_pt_loc < total)
				continue;
			if (f_vv) {
				cout << "schreier::compute_point_orbit n e w pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
				}
			swap_points(total, next_pt_loc);
			prev[total] = cur_pt;
			label[total] = i;
			//orbit_no[total] = nb_orbits;
			total++;
			total1 = total - orbit_first[nb_orbits];
			cur1 = cur - orbit_first[nb_orbits];
			if ((total1 % 10000) == 0 ||
					(cur1 > 0 && (cur1 % 10000) == 0)) {
				cout << "schreier::compute_point_orbit degree = "
						<< A->degree << " length = " << total1
					<< " processed = " << cur1 << " nb_orbits="
					<< nb_orbits << " cur_pt=" << cur_pt << " next_pt="
					<< next_pt << " orbit_first[nb_orbits]="
					<< orbit_first[nb_orbits] << endl;
				}
			if (FALSE) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	orbit_first[nb_orbits + 1] = total;
	orbit_len[nb_orbits] = total - orbit_first[nb_orbits];
	//orbit_first[nb_orbits + 2] = A->degree;
	//orbit_len[nb_orbits + 1] = A->degree - total;
	if (f_v) {
		cout << "found orbit of length " << orbit_len[nb_orbits]
				<< " total length " << total
				<< " degree=" << A->degree << endl;
		}
	if (FALSE) {
		cout << "{ ";
		for (i = orbit_first[nb_orbits];
				i < orbit_first[nb_orbits + 1]; i++) {
			cout << orbit[i];
			if (i < orbit_first[nb_orbits + 1] - 1)
				cout << ", ";
			}
		cout << " }" << endl;
		}
	if (FALSE) {
		cout << "coset reps:" << endl;
		for (i = orbit_first[nb_orbits];
				i < orbit_first[nb_orbits + 1]; i++) {
			cout << i << " : " << endl;
			coset_rep(i);
			A->element_print(cosetrep, cout);
			cout << "image = " << orbit[i] << " = "
					<< A->element_image_of(pt, cosetrep, 0) << endl;
			cout << endl;
			
			}
		}
	nb_orbits++;
}

void schreier::non_trivial_random_schreier_generator(
	action *A_original, INT verbose_level)
// computes non trivial random Schreier generator into schreier_gen
// non-trivial is with respect to A_original
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = FALSE; //(verbose_level >= 3);
	INT f_v4 = FALSE; //(verbose_level >= 4);
	INT cnt = 0;
	
	if (f_v) {
		cout << "schreier::non_trivial_random_schreier_generator "
				"verbose_level=" << verbose_level << endl;
		}
	while (TRUE) {
		if (f_v) {
			cout << "schreier::non_trivial_random_schreier_generator "
					"calling random_schreier_generator "
					"(cnt=" << cnt << ")" << endl;
			}
		random_schreier_generator(verbose_level - 1);
		cnt++;
		if (!A_original->element_is_one(schreier_gen, verbose_level - 5)) {
			if (f_vv) {
				cout << "schreier::non_trivial_random_schreier_generator "
						"found a non-trivial random Schreier generator in "
						<< cnt << " trials" << endl;
				}
			if (f_vvv) {
				A->element_print(schreier_gen, cout);
				cout << endl;
				}
			return;
			}
		else {
			if (f_v4) {
				A->element_print(schreier_gen, cout);
				cout << endl;
				}
			if (f_vv) {
				cout << "schreier::non_trivial_random_schreier_generator "
						"the element is the identity in action "
						<< A_original->label << ", trying again" << endl;
				}
			}
		}
}

void schreier::random_schreier_generator_ith_orbit(
	INT orbit_no, INT verbose_level)
{
	INT first, len, r1, r2, pt, pt2, pt2_coset;
	INT *gen;
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 2);
	INT f_vvv = FALSE; //(verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::random_schreier_generator_ith_orbit, "
				"orbit " << orbit_no << endl;
		}
	if (f_vvv) {
		cout << "generators are:" << endl;
		gens.print(cout);
		}
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	pt = orbit[first];
	if (f_vv) {
		cout << "pt=" << pt << endl;
		cout << "orbit_first[orbit_no]=" << orbit_first[orbit_no] << endl;
		cout << "orbit_len[orbit_no]=" << orbit_len[orbit_no] << endl;
		cout << "gens.len=" << gens.len << endl;
		}
	
	// get a random coset:
	r1 = random_integer(orbit_len[orbit_no]);
	if (f_vv) {
		cout << "r1=" << r1 << endl;
		}
	//pt1 = orbit[r1];
	coset_rep(orbit_first[orbit_no] + r1);
	// coset rep now in cosetrep
	if (f_vvv) {
		cout << "cosetrep " << orbit_first[orbit_no] + r1 << endl;
		A->element_print_quick(cosetrep, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(cosetrep, cout);
			cout << endl;
			}
		}
		
	// get a random generator:
	r2 = random_integer(gens.len);
	if (f_vv) {
		cout << "r2=" << r2 << endl;
		}
	gen = gens.ith(r2);
	if (f_vvv) {
		cout << "generator " << r2 << endl;
		A->element_print(gen, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(gen, cout);
			cout << endl;
			}
		}
	if (f_vv) {
		cout << "random coset " << r1
				<< ", random generator " << r2 << endl;
		}
	
	A->element_mult(cosetrep, gen, schreier_gen1, 0);
	if (f_vvv) {
		cout << "cosetrep * generator " << endl;
		A->element_print_quick(schreier_gen1, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(schreier_gen1, cout);
			cout << endl;
			}
		}
	pt2 = A->element_image_of(pt, schreier_gen1, 0);
	if (f_vv) {
		//cout << "pt2=" << pt2 << endl;
		cout << "maps " << pt << " to " << pt2 << endl;
		}
	pt2_coset = orbit_inv[pt2];
	if (f_vv) {
		cout << "pt2_coset=" << pt2_coset << endl;
		}
	if (pt2_coset < first) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"pt2_coset < first" << endl;
		exit(1);
		}
	if (pt2_coset >= first + len) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"pt2_coset >= first + len" << endl;
		exit(1);
		}
	
	coset_rep_inv(pt2_coset);
	// coset rep now in cosetrep
	if (f_vvv) {
		cout << "cosetrep (inverse) " << pt2_coset << endl;
		A->element_print_quick(cosetrep, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(cosetrep, cout);
			cout << endl;
			}
		}
	
	A->element_mult(schreier_gen1, cosetrep, schreier_gen, 0);
	if (A->element_image_of(pt, schreier_gen, 0) != pt) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"fatal: schreier generator does not stabilize pt" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"done" << endl;
		}
	if (f_vvv) {
		A->element_print_quick(schreier_gen, cout);
		cout << endl;
		if (A->degree < 100) {
			A->element_print_as_permutation(cosetrep, cout);
			cout << endl;
			}
		}
}

void schreier::random_schreier_generator(INT verbose_level)
// computes random Schreier generator into schreier_gen
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT r1, r2, pt, pt2, pt2b, pt2_coset;
	INT *gen;
	INT pt1, pt1b;
	
	if (f_v) {
		cout << "schreier::random_schreier_generator orbit_len = " 
			<< orbit_len[0] << " nb generators = "
			<< gens.len << " in action " << A->label << endl;
		}
	pt = orbit[0];
	if (f_vv) {
		cout << "pt=" << pt << endl;
		}
	
	// get a random coset:
	r1 = random_integer(orbit_len[0]);
	pt1 = orbit[r1];
	
	coset_rep(r1);
	// coset rep now in cosetrep
	pt1b = A->element_image_of(pt, cosetrep, 0);
	if (f_vv) {
		cout << "random coset " << r1 << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "image of pt under cosetrep = " << pt1b << endl;
		}
	if (pt1b != pt1) {
		cout << "schreier::random_schreier_generator fatal: "
				"cosetrep does not work" << endl;
		cout << "pt=" << pt << endl;
		cout << "random coset " << r1 << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "image of pt under cosetrep = " << pt1b << endl;
		A->element_image_of(pt, cosetrep, 10);	
		exit(1);
		}
	
	// get a random generator:
	r2 = random_integer(gens.len);
	gen = gens.ith(r2);
	if (f_vv) {
		cout << "random coset " << r1 << ", "
				"random generator " << r2 << endl;
		cout << "generator:" << endl;
		A->element_print_quick(gen, cout);
		cout << "image of pt1 under generator = pt2 = "
				<< A->element_image_of(pt1, gen, 0) << endl;
		}
	pt2b = A->element_image_of(pt1, gen, 0);
	
	A->element_mult(cosetrep, gen, schreier_gen1, 0);
	if (f_vv) {
		cout << "cosetrep * gen=" << endl;
		A->element_print_quick(schreier_gen1, cout);
		}
	pt2 = A->element_image_of(pt, schreier_gen1, 0);
	if (f_vv) {
		cout << "image of pt under cosetrep*gen = " << pt2 << endl;
		}
	if (pt2 != pt2b) {
		cout << "schreier::random_schreier_generator "
				"something is wrong! " << endl;
		cout << "pt2=" << pt2 << " = image of pt "
				"under cosetrep * gen" << endl;
		cout << "pt2b=" << pt2b << " = image of pt1 "
				"under gen" << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "generator:" << endl;
		A->element_print_quick(gen, cout);
		cout << "cosetrep * gen=" << endl;
		A->element_print_quick(schreier_gen1, cout);
		cout << "pt=" << pt << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "pt1b=" << pt1b << endl;
		cout << "pt2=" << pt2 << endl;
		cout << "pt2b=" << pt2b << endl;

		cout << "repeat 1" << endl;
		cout << "repeating pt1b = A->element_image_of(pt, "
				"cosetrep, 0):" << endl;
		pt1b = A->element_image_of(pt, cosetrep, verbose_level + 3);
		cout << "pt1b = " << pt1b << endl;

		cout << "repeat 2" << endl;
		cout << "repeating pt2b = A->element_image_of(pt1, "
				"gen, 0):" << endl;
		pt2b = A->element_image_of(pt1, gen, verbose_level + 3);

		cout << "repeat 3" << endl;
		cout << "repeating pt2 = A->element_image_of(pt, "
				"schreier_gen1, 0):" << endl;
		pt2 = A->element_image_of(pt, schreier_gen1, verbose_level + 3);


		exit(1);
	}
	//cout << "maps " << pt << " to " << pt2 << endl;
	pt2_coset = orbit_inv[pt2];
	
	coset_rep_inv(pt2_coset);
	// coset rep now in cosetrep
	if (f_vv) {
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "image of pt2 under cosetrep = "
				<< A->element_image_of(pt2, cosetrep, 0) << endl;
		}
	
	A->element_mult(schreier_gen1, cosetrep, schreier_gen, 0);
	if (f_vv) {
		cout << "schreier_gen=cosetrep*gen*cosetrep:" << endl;
		A->element_print_quick(schreier_gen, cout);
		cout << "image of pt under schreier_gen = "
				<< A->element_image_of(pt, schreier_gen, 0) << endl;
		}
	if (A->element_image_of(pt, schreier_gen, 0) != pt) {
		cout << "schreier::random_schreier_generator() "
				"fatal: schreier generator does not stabilize pt" << endl;
		exit(1);
		}
	if (FALSE) {
		cout << "random Schreier generator:" << endl;
		A->element_print(schreier_gen, cout);
		cout << endl;
		}
}

void schreier::trace_back(INT *path, INT i, INT &j)
{
	INT ii = orbit_inv[i];
	
	if (prev[ii] == -1) {
		if (path) {
			path[0] = i;
			}
		j = 1;
		}
	else {
		trace_back(path, prev[ii], j);
		if (path) {
			path[j] = i;
			}
		j++;
		}
}

void schreier::print_tree(INT orbit_no)
{
	INT *path;
	INT i, j, l;
	
	path = NEW_INT(A->degree);
	i = orbit_first[orbit_no];
	while (i < orbit_first[orbit_no + 1]) {
		trace_back(path, orbit[i], l);
		// now l is the distance from the root
		cout << l;
		for (j = 0; j < l; j++) {
			cout << " " << path[j];
			}
		cout << " 0 ";
		if (label[i] != -1) {
			cout << " $s_{" << label[i] << "}$";
			}
		cout << endl;
		i++;
		}
	FREE_INT(path);
}

void schreier::draw_forest(const char *fname_mask, 
	INT xmax, INT ymax, 
	INT f_circletext, INT rad, 
	INT f_embedded, INT f_sideways, 
	double scale, double line_width, 
	INT f_has_point_labels, INT *point_labels, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	char fname[1000];
	INT i;

	if (f_v) {
		cout << "schreier::draw_forest" << endl;
		}
	for (i = 0; i < nb_orbits; i++) {
		sprintf(fname, fname_mask, i);

		if (f_v) {
			cout << "schreier::draw_forest drawing orbit "
					<< i << " / " << nb_orbits << endl;
			}
		draw_tree(fname, 
			i /* orbit_no */, xmax, ymax, 
			f_circletext, rad, 
			f_embedded, f_sideways, 
			scale, line_width, 
			f_has_point_labels, point_labels, 
			verbose_level);
		}
	if (f_v) {
		cout << "schreier::draw_forest done" << endl;
		}
}

void schreier::draw_tree(const char *fname, 
	INT orbit_no, INT xmax, INT ymax, 
	INT f_circletext, INT rad, 
	INT f_embedded, INT f_sideways, 
	double scale, double line_width, 
	INT f_has_point_labels, INT *point_labels, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *path;
	INT *weight;
	INT *placement_x;
	INT i, j, last, max_depth = 0;


	if (f_v) {
		cout << "schreier::draw_tree" << endl;
		}
	path = NEW_INT(A->degree);
	weight = NEW_INT(A->degree);
	placement_x = NEW_INT(A->degree);
		
	i = orbit_first[orbit_no];
	last = orbit_first[orbit_no + 1];
	
	for (j = 0; j < A->degree; j++) {
		weight[j] = 0;
		placement_x[j] = 0;
		}
	subtree_calc_weight(weight, max_depth, i, last);
	if (f_vv) {
		cout << "the weights: " << endl;
		for (j = i; j < last; j++) {
			cout << j << " : " << weight[j] << " : " << endl;
			}
		cout << endl;
		cout << "max_depth = " << max_depth << endl;
		}
	subtree_place(weight, placement_x, 0, 1000000, i, last);
	if (f_vv) {
		for (j = i; j < last; j++) {
			cout << j << " : " << placement_x[j] << endl;
			}
		cout << endl;
		}
	if (orbit_len[orbit_no] > 100) {
		f_circletext = FALSE;
		}
	draw_tree2(fname, xmax, ymax, f_circletext, 
		weight, placement_x, max_depth, i, last, rad, 
		f_embedded, f_sideways, 
		scale, line_width, 
		f_has_point_labels, point_labels, 
		verbose_level - 2);
	
	FREE_INT(path);
	FREE_INT(weight);
	FREE_INT(placement_x);
	if (f_v) {
		cout << "schreier::draw_tree done" << endl;
		}
}

static void calc_y_coordinate(INT &y, INT l, INT max_depth)
{
	INT dy;
	
	dy = (INT)((double)1000000 / (double)max_depth);
	y = (INT)(dy * ((double)l + 0.5));
	y = 1000000 - y;
}

void schreier::draw_tree2(const char *fname, 
	INT xmax, INT ymax, INT f_circletext, 
	INT *weight, INT *placement_x, 
	INT max_depth, INT i, INT last, INT rad, 
	INT f_embedded, INT f_sideways, 
	double scale, double line_width, 
	INT f_has_point_labels, INT *point_labels, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT x_min = 0, x_max = 1000000;
	INT y_min = 0, y_max = 1000000;
	INT factor_1000 = 1000;
	char fname_full[1000];

	if (f_v) {
		cout << "schreier::draw_tree2" << endl;
		}
	sprintf(fname_full, "%s.mp", fname);
	mp_graphics G(fname_full, x_min, y_min, x_max, y_max, 
		f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	G.set_parameters(scale, line_width);
	
	G.header();
	G.begin_figure(factor_1000);
	
	INT x = 500000, y;
	calc_y_coordinate(y, 0, max_depth);
	

	subtree_draw_lines(G, f_circletext, x, y, weight, 
		placement_x, max_depth, i, last, verbose_level);

	subtree_draw_vertices(G, f_circletext, x, y, weight, 
		placement_x, max_depth, i, last, rad, 
		f_has_point_labels, point_labels, 
		verbose_level);

	INT j, L, l, N;
	double avg;
	
	N = last - i;
	L = 0;
	for (j = i; j < last; j++) {
		trace_back(NULL, orbit[j], l);
		L += l;
		}
	avg = (double) L / (double)N;
	x = 500000;
	calc_y_coordinate(y, max_depth + 1, max_depth);
	char str[1000];
	INT nb_gens;
	double H; // entropy

	nb_gens = gens.len;
	if (nb_gens) {
		H = log(N) / log(nb_gens);
		}
	else {
		H = 0.;
		}
	sprintf(str, "N=%ld, avg=%lf,  gens=%ld, H=%lf", N, avg, nb_gens, H);
	//G.aligned_text(x, y, "", str);
	

#if 0
	if (f_circletext) {
		G.circle_text(x, y, rad, "$\\emptyset$");
		}
	else {
		G.circle_text(x, y, rad, "");
		//G.circle(x, y, rad);
		}
#endif

	G.end_figure();
	//print_and_list_orbits_tex(ostream &ost)
	G.footer();
	if (f_v) {
		cout << "schreier::draw_tree2 done" << endl;
		}
}

void schreier::subtree_draw_lines(mp_graphics &G, 
	INT f_circletext, INT parent_x, INT parent_y, INT *weight, 
	INT *placement_x, INT max_depth, INT i, INT last, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT pt = orbit[i];
	INT x, y, l, ii;
	INT Px[3], Py[3];
	
	if (f_v) {
		cout << "schreier::subtree_draw_lines" << endl;
		}
	trace_back(NULL, pt, l);
	// l is 1 if pt is the root.
	x = placement_x[pt];
	calc_y_coordinate(y, l, max_depth);

	//G.circle(x, y, 2000);
	Px[0] = parent_x;
	Py[0] = parent_y;
	Px[1] = x;
	Py[1] = y;
	Px[2] = (Px[0] + Px[1]) >> 1;
	Py[2] = (Py[0] + Py[1]) >> 1;
	//cout << "schreier::subtree_draw_lines "
	// << parent_x << "," << parent_y << " - "
	// << x << "," << y << endl;


#if 0
	INT y1;
	calc_y_coordinate(y1, 0, max_depth);
	if (parent_x == 500000 && parent_y == y1) {
		}
	else {
		G.polygon2(Px, Py, 0, 1);
		}
#endif
	if (l > 1) {
		char str[1000];
		// if pt is not the root node:
		G.polygon2(Px, Py, 0, 1);
		sprintf(str, "$\\alpha_{%ld}$", label[i]);
		G.aligned_text(Px[2], Py[2], "", str);
		}
	
	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			subtree_draw_lines(G, f_circletext,
					x, y, weight, placement_x,
					max_depth, ii, last, verbose_level);
			}
		}

	if (f_v) {
		cout << "schreier::subtree_draw_lines done" << endl;
		}
}

void schreier::subtree_draw_vertices(mp_graphics &G, 
	INT f_circletext, INT parent_x, INT parent_y, INT *weight, 
	INT *placement_x, INT max_depth, INT i, INT last, INT rad, 
	INT f_has_point_labels, INT *point_labels, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT pt = orbit[i];
	INT x, y, l, ii;
	//INT Px[2], Py[2];
	char str[1000];
	
	if (f_v) {
		cout << "schreier::subtree_draw_vertices" << endl;
		}
	trace_back(NULL, pt, l);
	x = placement_x[pt];
	calc_y_coordinate(y, l, max_depth);

#if 0
	Px[0] = parent_x;
	Py[0] = parent_y;
	Px[1] = x;
	Py[1] = y;
	//cout << "schreier::subtree_draw_vertices "
	// << parent_x << "," << parent_y << " - " << x << "," << y << endl;
	//G.polygon2(Px, Py, 0, 1);
#endif

	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			subtree_draw_vertices(G, f_circletext,
				x, y, weight, placement_x,
				max_depth, ii, last, rad,
				f_has_point_labels, point_labels, 
				verbose_level);
			}
		}
#if 0
	if (pt == 169303 || pt == 91479) {
		G.circle(x, y, 4 * rad);
		}
#endif
	if (f_has_point_labels) {
		sprintf(str, "%ld", point_labels[pt]);
		}
	else {
		sprintf(str, "%ld", pt);
		}
	if (f_circletext) {
		G.circle_text(x, y, rad, str);
		}
	else {
		G.circle_text(x, y, rad, "");
		//G.circle(x, y, rad);
		//G.aligned_text(Px, Py, 1, "tl", str);
		}
	if (f_v) {
		cout << "schreier::subtree_draw_vertices done" << endl;
		}
}

void schreier::subtree_place(INT *weight, INT *placement_x, 
	INT left, INT right, INT i, INT last)
{
	INT pt = orbit[i];
	INT ii, l, w, w0, w1, lft, rgt, width;
	double dx;
	
	placement_x[pt] = (left + right) >> 1;
	w = weight[pt];
	width = right - left;
	dx = width / (double) (w - 1);
		// the node itself counts for the weight, so we subtract one
	w0 = 0;
	
	trace_back(NULL, pt, l);
	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			w1 = weight[orbit[ii]];
			lft = left + (INT)((double)w0 * dx);
			rgt = left + (INT)((double)(w0 + w1) * dx);
			subtree_place(weight, placement_x, lft, rgt, ii, last);
			w0 += w1;
			}
		}
}

INT schreier::subtree_calc_weight(INT *weight, 
	INT &max_depth, INT i, INT last)
{
	INT pt = orbit[i];
	INT ii, l, w = 1, w1;
	
	trace_back(NULL, pt, l);
	if (l > max_depth)
		max_depth = l;
	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			w1 = subtree_calc_weight(weight, max_depth, ii, last);
			w += w1;
			}
		}
	weight[pt] = w;
	return w;
}

INT schreier::subtree_depth_first(ostream &ost,
		INT *path, INT i, INT last)
{
	INT pt = orbit[i];
	INT ii, l, w = 1, w1;
	
	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
		
			
			trace_back(path, orbit[ii], l);
			// now l is the distance from the root
			print_path(ost, path, l);

			w1 = subtree_depth_first(ost, path, ii, last);
			w += w1;
			}
		}
	return w;
}

void schreier::print_path(ostream &ost, INT *path, INT l)
{
	INT j;
	
	ost << l;
	for (j = 0; j < l; j++) {
		ost << " " << path[j];
		}
	ost << endl;
}

void schreier::intersection_vector(INT *set,
		INT len, INT *intersection_cnt)
{
	INT i, pt, /*pt_loc,*/ o;
	
	for (i = 0; i < nb_orbits; i++) {
		intersection_cnt[i] = 0;
		}
	for (i = 0; i < len; i++) {
		pt = set[i];
		//pt_loc = orbit_inv[pt];
		o = orbit_number(pt); // orbit_no[pt_loc];
		intersection_cnt[o]++;
		}
}

void schreier::orbits_on_invariant_subset_fast(
	INT len, INT *subset, INT verbose_level)
{
	INT i, p, j;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast "
			"computing orbits on invariant subset "
			"of size " << len << " in action ";
		A->print_info();
		}
	
	for (i = 0; i < len; i++) {
		p = subset[i];
		j = orbit_inv[p];
		if (j >= orbit_first[nb_orbits]) {
			if (f_vvv) {
				cout << "computing orbit no " << nb_orbits << endl;
				}
			compute_point_orbit(p, 0);
			}
		}
#if 0
	if (orbit_first[nb_orbits] != len) {
		cout << "schreier::orbits_on_invariant_subset_"
				"fast orbit_first[nb_orbits] != len" << endl;
		cout << "orbit_first[nb_orbits] = "
				<< orbit_first[nb_orbits] << endl;
		cout << "len = " << len << endl;
		cout << "subset:" << endl;
		INT_vec_print(cout, subset, len);
		cout << endl;
		print_tables(cout, FALSE);
		exit(1);
		}
#endif
	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast "
			"found " << nb_orbits
			<< " orbits on the invariant subset of size " << len << endl;
		}
}

void schreier::orbits_on_invariant_subset(INT len, INT *subset, 
	INT &nb_orbits_on_subset, 
	INT *&orbit_perm, INT *&orbit_perm_inv)
{
	INT i, j, a, pos;
	
	compute_all_point_orbits(0);
	nb_orbits_on_subset = 0;
	orbit_perm = NEW_INT(nb_orbits);
	orbit_perm_inv = NEW_INT(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		orbit_perm_inv[i] = -1;
		}
	for (i = 0; i < nb_orbits; i++) {
		j = orbit_first[i];
		a = orbit[j];
		for (pos = 0; pos < len; pos++) {
			if (subset[pos] == a) {
				orbit_perm[nb_orbits_on_subset] = i;
				orbit_perm_inv[i] = nb_orbits_on_subset;
				nb_orbits_on_subset++;
				break;
				}
			}
		}
	j = nb_orbits_on_subset;
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_perm_inv[i] == -1) {
			orbit_perm[j] = i;
			orbit_perm_inv[i] = j;
			j++;
			}
		}	
}

void schreier::get_orbit_partition_of_points_and_lines(
	partitionstack &S, INT verbose_level)
{
	INT first_column_element, pos, first_column_orbit, i, j, f, l, a;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::get_orbit_partition_"
				"of_points_and_lines" << endl;
		}
	first_column_element = S.startCell[1];
	if (f_v) {
		cout << "first_column_element = "
				<< first_column_element << endl;
		}
	pos = orbit_inv[first_column_element];
	first_column_orbit = orbit_number(first_column_element);
	
	for (i = first_column_orbit - 1; i > 0; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
	for (i = nb_orbits - 1; i > first_column_orbit; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
}

void schreier::get_orbit_partition(partitionstack &S, 
	INT verbose_level)
{
	INT pos, i, j, f, l, a;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::get_orbit_partition" << endl;
		}
	for (i = nb_orbits - 1; i > 0; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
}

strong_generators *schreier::generators_for_stabilizer_of_arbitrary_point_and_transversal(
	action *default_action, 
	longinteger_object &full_group_order, INT pt, vector_ge *&cosets, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	strong_generators *gens0;
	strong_generators *gens;
	INT orbit_index;
	INT orbit_index1;
	INT *transporter;
	INT *transporter1;
	INT i, fst, len;

	if (f_v) {
		cout << "schreier::generators_for_stabilizer_of_"
				"arbitrary_point_and_transversal" << endl;
		}
	
	cosets = NEW_OBJECT(vector_ge);
	cosets->init(A);
	transporter = NEW_INT(A->elt_size_in_INT);
	transporter1 = NEW_INT(A->elt_size_in_INT);
	
	orbit_index = orbit_number(pt);

	gens0 = generators_for_stabilizer_of_orbit_rep(default_action, 
		full_group_order, orbit_index, 0 /* verbose_level */);

	fst = orbit_first[orbit_index];
	len = orbit_len[orbit_index];
	cosets->allocate(len);

	transporter_from_point_to_orbit_rep(pt,
			orbit_index1, transporter, 0 /* verbose_level */);

	if (orbit_index1 != orbit_index) {
		cout << "schreier::generators_for_stabilizer_of_"
				"arbitrary_point_and_transversal "
				"orbit_index1 != orbit_index" << endl;
		exit(1);
		}
	
	gens = NEW_OBJECT(strong_generators);

	
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_of_"
				"arbitrary_point_and_transversal before "
				"gens->init_generators_for_the_conjugate_group_aGav" << endl;
		}
	gens->init_generators_for_the_conjugate_group_aGav(gens0, 
		transporter, verbose_level);
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_of_"
				"arbitrary_point_and_transversal after "
				"gens->init_generators_for_the_conjugate_group_aGav" << endl;
		}

	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point_and_transversal computing "
				"coset representatives" << endl;
		}
	for (i = 0; i < len; i++) {
		transporter_from_orbit_rep_to_point(orbit[fst + i],
				orbit_index1, transporter1, 0 /* verbose_level */);
		A->element_mult(transporter, transporter1, cosets->ith(i), 0);
		}
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point_and_transversal computing "
				"coset representatives done" << endl;
		}

	FREE_INT(transporter);
	FREE_INT(transporter1);
	
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point_and_transversal done" << endl;
		}
	FREE_OBJECT(gens0);
	return gens;
}

strong_generators *schreier::generators_for_stabilizer_of_arbitrary_point(
	action *default_action, 
	longinteger_object &full_group_order, INT pt, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	strong_generators *gens0;
	strong_generators *gens;
	INT orbit_index;
	INT orbit_index1;
	INT *transporter;

	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point" << endl;
		}
	
	transporter = NEW_INT(A->elt_size_in_INT);
	
	orbit_index = orbit_number(pt);

	gens0 = generators_for_stabilizer_of_orbit_rep(default_action, 
		full_group_order, orbit_index, 0 /* verbose_level */);

	transporter_from_point_to_orbit_rep(pt,
			orbit_index1, transporter, 0 /* verbose_level */);

	if (orbit_index1 != orbit_index) {
		cout << "schreier::generators_for_stabilizer_of_"
				"arbitrary_point orbit_index1 != orbit_index" << endl;
		exit(1);
		}
	
	gens = NEW_OBJECT(strong_generators);

	
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point before gens->init_generators_"
				"for_the_conjugate_group_aGav" << endl;
		}
	gens->init_generators_for_the_conjugate_group_aGav(gens0, 
		transporter, verbose_level);
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point after gens->init_generators_"
				"for_the_conjugate_group_aGav" << endl;
		}

	FREE_INT(transporter);
	
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_arbitrary_point done" << endl;
		}
	FREE_OBJECT(gens0);
	return gens;
}


strong_generators *schreier::generators_for_stabilizer_of_orbit_rep(
	action *default_action, 
	longinteger_object &full_group_order, INT orbit_idx, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	strong_generators *gens;
	sims *Stab;

	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_orbit_rep" << endl;
		}


	point_stabilizer(default_action, full_group_order, 
		Stab, orbit_idx, verbose_level);

	longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_orbit_rep found a stabilizer group "
				"of order " << stab_order << endl;
		}
	
	gens = NEW_OBJECT(strong_generators);
	gens->init(A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "schreier::generators_for_stabilizer_"
				"of_orbit_rep done" << endl;
		}
	return gens;
}

void schreier::point_stabilizer(action *default_action, 
	longinteger_object &go, 
	sims *&Stab, INT orbit_no, 
	INT verbose_level)
// this function allocates a sims structure into Stab.
{
	Stab = NEW_OBJECT(sims);
	longinteger_object cur_go, target_go;
	longinteger_domain D;
	INT len, r, cnt = 0, f_added, *p_gen, drop_out_level, image;
	INT *residue;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_v4 = (verbose_level >= 4);
	//INT f_v5 = (verbose_level >= 5);
	
	
	if (f_v) {
		cout << "schreier::point_stabilizer "
				"computing stabilizer of representative of orbit "
			<< orbit_no << " inside a group of "
					"order " << go << " in action ";
		default_action->print_info();
		cout << endl;
		}
	residue = NEW_INT(default_action->elt_size_in_INT);
	len = orbit_len[orbit_no];
	D.integral_division_by_INT(go, len, target_go, r);
	if (r) {	
		cout << "schreier::point_stabilizer "
				"orbit length does not divide group order" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "expecting group of order " << target_go << endl;
		}
	
	Stab->init(default_action);
	Stab->init_trivial_group(verbose_level - 1);
	while (TRUE) {
		Stab->group_order(cur_go);
		if (D.compare(cur_go, target_go) == 0) {
			break;
			}
		if (cnt % 2 || Stab->nb_gen[0] == 0) {
			random_schreier_generator_ith_orbit(orbit_no,
					0 /* verbose_level */);
			p_gen = schreier_gen;
			if (f_vvv) {
				cout << "random Schreier generator from the orbit:" << endl;
				default_action->element_print(p_gen, cout);
				}
			}
		else {
			Stab->random_schreier_generator(0 /* verbose_level */);
			p_gen = Stab->schreier_gen;
			if (f_v4) {
				cout << "random schreier generator from sims:" << endl;
				default_action->element_print(p_gen, cout);
				}
			}



		if (Stab->strip(p_gen, residue,
				drop_out_level, image, 0 /*verbose_level - 3*/)) {
			if (f_vvv) {
				cout << "element strips through" << endl;
				if (f_v4) {
					cout << "residue:" << endl;
					A->element_print(residue, cout);
					cout << endl;
					}
				}
			f_added = FALSE;
			}
		else {
			f_added = TRUE;
			if (f_vvv) {
				cout << "element needs to be inserted at level = " 
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					A->element_print(residue, cout);
					cout  << endl;
					}
				}
			Stab->add_generator_at_level(residue,
					drop_out_level, verbose_level - 4);
			}
		Stab->group_order(cur_go);
		if ((f_vv && f_added) || f_vvv) {
			cout << "iteration " << cnt
				<< " the n e w group order is " << cur_go
				<< " expecting a group of order "
				<< target_go << endl;
			}
		cnt++;
		}
	FREE_INT(residue);
	if (f_v) {
		cout << "schreier::point_stabilizer finished" << endl;
		}
}

void schreier::get_orbit(INT orbit_idx, INT *set, INT &len, 
	INT verbose_level)
{
	INT f, i;

	f = orbit_first[orbit_idx];
	len = orbit_len[orbit_idx];
	for (i = 0; i < len; i++) {
		set[i] = orbit[f + i];
		}
}

void schreier::compute_orbit_statistic(INT *set, INT set_size, 
	INT *orbit_count, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a, /*j,*/ o;

	if (f_v) {
		cout << "schreier::compute_orbit_statistic" << endl;
		}
	INT_vec_zero(orbit_count, nb_orbits);
#if 0
	for (i = 0; i < nb_orbits; i++) {
		orbit_count[i] = 0;
		}
#endif
	for (i = 0; i < set_size; i++) {
		a = set[i];
		//j = orbit_inv[a];
		o = orbit_number(a); //orbit_no[j];
		orbit_count[o]++;
		}
	if (f_v) {
		cout << "schreier::compute_orbit_statistic done" << endl;
		}
}

void schreier::test_sv(action *A, 
	INT *hdl_strong_generators, INT *sv, 
	INT f_trivial_group, INT f_compact, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *crep, *Elt1, *Elt2, *Elt3;
	
	crep = NEW_INT(A->elt_size_in_INT);
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	INT k, i, j, pt, pt0;
	INT f_check_image = FALSE;
	
	if (f_v) {
		cout << "testing the schreier vector" << endl;
		}
	for (k = 0; k < nb_orbits; k++) {
		for (j = 0; j < orbit_len[k]; j++) {
			i = orbit_first[k] + j;
			pt = orbit[i];
			coset_rep_inv(i);
			schreier_vector_coset_rep_inv(
				A, sv, hdl_strong_generators, pt, pt0,
				crep, Elt1, Elt2, Elt3, 
				f_trivial_group, f_compact,
				f_check_image, verbose_level - 4);
			A->element_invert(crep, Elt1, 0);
			A->element_mult(cosetrep, Elt1, Elt2, 0);
			if (!A->element_is_one(Elt2, 0)) {
				cout << "schreier::test_sv() test fails" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "sv test passed" << endl;
		}
	FREE_INT(crep);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
}

void schreier::write_to_memory_object(
		memory_object *m, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "schreier::write_to_memory_object" << endl;
		}
	m->write_int(0); // indicator
	m->write_int(1); // version
	m->write_int(A->degree);
	m->write_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		m->write_int(orbit_first[i]);
		m->write_int(orbit_len[i]);
		}
	for (i = 0; i < A->degree; i++) {
		m->write_int(orbit[i]);
		m->write_int(prev[i]);
		m->write_int(label[i]);
		//m->write_int(orbit_no[i]);
		}
	gens.write_to_memory_object(m, verbose_level - 1);
	gens_inv.write_to_memory_object(m, verbose_level - 1);
	if (f_v) {
		cout << "schreier::write_to_memory_object done" << endl;
		}
}

void schreier::read_from_memory_object(
		memory_object *m, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, deg, dummy, a, version;

	if (f_v) {
		cout << "schreier::read_from_memory_object" << endl;
		}
	init2();
	m->read_int(&a);
	if (a == 0) {
		m->read_int(&version);
		m->read_int(&deg);
		}
	else {
		version = 0;
		deg = a;
		}
	m->read_int(&nb_orbits);
	if (deg != A->degree) {
		cout << "schreier::read_from_memory_object "
				"deg != A->degree" << endl;
		}
	orbit_first = NEW_INT(nb_orbits);
	orbit_len = NEW_INT(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		m->read_int(&orbit_first[i]);
		m->read_int(&orbit_len[i]);
		}
	orbit = NEW_INT(A->degree);
	orbit_inv = NEW_INT(A->degree);
	prev = NEW_INT(A->degree);
	label = NEW_INT(A->degree);
	//orbit_no = NEW_INT(A->degree);
	for (i = 0; i < A->degree; i++) {
		m->read_int(&orbit[i]);
		m->read_int(&prev[i]);
		m->read_int(&label[i]);
		m->read_int(&dummy);
		if (version == 0) {
			m->read_int(&dummy);
			}
		//m->read_int(&orbit_no[i]);
		}
	perm_inverse(orbit, orbit_inv, A->degree);
	gens.init(A);
	gens.read_from_memory_object(m, verbose_level - 1);
	gens_inv.init(A);
	gens_inv.read_from_memory_object(m, verbose_level - 1);
	if (f_v) {
		cout << "schreier::read_from_memory_object done" << endl;
		}
}

void schreier::write_file(char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	memory_object M;
	
	if (f_v) {
		cout << "schreier::write_file" << endl;
		}
	M.alloc(1024 /* length */, verbose_level - 1);
	M.used_length = 0;
	M.cur_pointer = 0;
	write_to_memory_object(&M, verbose_level - 1);
	M.write_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "schreier::write_file done" << endl;
		}
}

void schreier::read_file(const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	memory_object M;
	
	if (f_v) {
		cout << "schreier::read_file reading file "
				<< fname << " of size " << file_size(fname) << endl;
		}
	M.read_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "schreier::read_file read file " << fname << endl;
		}
	M.cur_pointer = 0;
	read_from_memory_object(&M, verbose_level - 1);
	if (f_v) {
		cout << "schreier::read_file done" << endl;
		}
}

void schreier::write_to_file_binary(ofstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a = 0, version = 1;

	if (f_v) {
		cout << "schreier::write_to_file_binary" << endl;
		}
	fp.write((char *) &a, sizeof(INT));
	fp.write((char *) &version, sizeof(INT));
	fp.write((char *) &A->degree, sizeof(INT));
	fp.write((char *) &nb_orbits, sizeof(INT));
	for (i = 0; i < nb_orbits; i++) {
		fp.write((char *) &orbit_first[i], sizeof(INT));
		fp.write((char *) &orbit_len[i], sizeof(INT));
		}
	for (i = 0; i < A->degree; i++) {
		fp.write((char *) &orbit[i], sizeof(INT));
		fp.write((char *) &prev[i], sizeof(INT));
		fp.write((char *) &label[i], sizeof(INT));
		//fp.write((char *) &orbit_no[i], sizeof(INT));
		}
	gens.write_to_file_binary(fp, verbose_level - 1);
	gens_inv.write_to_file_binary(fp, verbose_level - 1);
	if (f_v) {
		cout << "schreier::write_to_file_binary done" << endl;
		}
}

void schreier::read_from_file_binary(ifstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, deg, dummy, a, version;

	if (f_v) {
		cout << "schreier::read_from_file_binary" << endl;
		}
	init2();
	fp.read((char *) &a, sizeof(INT));
	if (a == 0) {
		fp.read((char *) &version, sizeof(INT));
		fp.read((char *) &deg, sizeof(INT));
		}
	else {
		version = 0;
		deg = a;
		}
	//fp.read((char *) &deg, sizeof(INT));
	fp.read((char *) &nb_orbits, sizeof(INT));
	if (deg != A->degree) {
		cout << "schreier::read_from_file_binary "
				"deg != A->degree" << endl;
		}
	orbit_first = NEW_INT(nb_orbits);
	orbit_len = NEW_INT(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		fp.read((char *) &orbit_first[i], sizeof(INT));
		fp.read((char *) &orbit_len[i], sizeof(INT));
		}
	orbit = NEW_INT(A->degree);
	orbit_inv = NEW_INT(A->degree);
	prev = NEW_INT(A->degree);
	label = NEW_INT(A->degree);
	//orbit_no = NEW_INT(A->degree);
	for (i = 0; i < A->degree; i++) {
		fp.read((char *) &orbit[i], sizeof(INT));
		fp.read((char *) &prev[i], sizeof(INT));
		fp.read((char *) &label[i], sizeof(INT));
		if (version == 0) {
			fp.read((char *) &dummy, sizeof(INT));
			//fp.read((char *) &orbit_no[i], sizeof(INT));
			}
		}
	perm_inverse(orbit, orbit_inv, A->degree);
	
	gens.init(A);
	gens.read_from_file_binary(fp, verbose_level - 1);
	gens_inv.init(A);
	gens_inv.read_from_file_binary(fp, verbose_level - 1);
	if (f_v) {
		cout << "schreier::read_from_file_binary done" << endl;
		}
}


void schreier::write_file_binary(char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::write_file_binary" << endl;
		}
	{
		ofstream fp(fname, ios::binary);

		write_to_file_binary(fp, verbose_level - 1);
	}
	cout << "schreier::write_file_binary Written file "
			<< fname << " of size " << file_size(fname) << endl;
	if (f_v) {
		cout << "schreier::write_file_binary done" << endl;
		}
}

void schreier::read_file_binary(const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::read_file_binary reading file "
				<< fname << " of size " << file_size(fname) << endl;
		}
	cout << "schreier::read_file_binary Reading file "
			<< fname << " of size " << file_size(fname) << endl;
	{
		ifstream fp(fname, ios::binary);

		read_from_file_binary(fp, verbose_level - 1);
	}
	if (f_v) {
		cout << "schreier::read_file_binary done" << endl;
		}
}

void schreier::orbits_as_set_of_sets(
		set_of_sets *&S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Sz;
	INT i, j, a, f, l;
	
	if (f_v) {
		cout << "schreier::orbits_as_set_of_sets" << endl;
		}
	S = NEW_OBJECT(set_of_sets);
	Sz = NEW_INT(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		Sz[i] = l;
		}
	
	S->init_basic(A->degree /* underlying_set_size */,
			nb_orbits, Sz, 0 /* verbose_level */);
	for (i = 0; i < nb_orbits; i++) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			a = orbit[f + j];
			S->Sets[i][j] = a;
			}
		}
	FREE_INT(Sz);
	if (f_v) {
		cout << "schreier::orbits_as_set_of_sets done" << endl;
		}
}

void schreier::get_orbit_reps(INT *&Reps,
		INT &nb_reps, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a, f;
	
	if (f_v) {
		cout << "schreier::get_orbit_reps" << endl;
		}
	nb_reps = nb_orbits;
	Reps = NEW_INT(nb_reps);
	for (i = 0; i < nb_reps; i++) {
		f = orbit_first[i];
		a = orbit[f];
		Reps[i] = a;
		}
	if (f_v) {
		cout << "schreier::get_orbit_reps done" << endl;
		}
}

INT schreier::find_shortest_orbit_if_unique(INT &idx)
{
	INT l_min = 0, l, i;
	INT idx_min = -1;
	INT f_is_unique = TRUE;
	
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		if (idx_min == -1) {
			l_min = l;
			idx_min = i;
			f_is_unique = TRUE;
			}
		else if (l < l_min) {
			l_min = l;
			idx_min = i;
			f_is_unique = TRUE;
			}
		else if (l_min == l) {
			f_is_unique = FALSE;
			}
		}
	idx = idx_min;
	return f_is_unique;
}

void schreier::elements_in_orbit_of(INT pt, 
	INT *orb, INT &nb, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT idx, f;

	if (f_v) {
		cout << "schreier::elements_in_orbit_of" << endl;
		}
	idx = orbit_number(pt);
	f = orbit_first[idx];
	nb = orbit_len[idx];
	INT_vec_copy(orbit + f, orb, nb);
	if (f_v) {
		cout << "schreier::elements_in_orbit_of done" << endl;
		}
}

void schreier::get_orbit_lengths_once_each(
	INT *&orbit_lengths, INT &nb_orbit_lengths)
{
	INT *val, *mult, len;	
	
	INT_vec_distribution(orbit_len, nb_orbits, val, mult, len);
	//INT_distribution_print(ost, val, mult, len);
	//ost << endl;
	
	nb_orbit_lengths = len;

	orbit_lengths = NEW_INT(nb_orbit_lengths);

	INT_vec_copy(val, orbit_lengths, nb_orbit_lengths);

	FREE_INT(val);
	FREE_INT(mult);
}


INT schreier::orbit_number(INT pt)
{
	INT pos;
	INT idx;

	pos = orbit_inv[pt];
	if (INT_vec_search(orbit_first, nb_orbits, pos, idx)) {
		;
		}
	else {
		if (idx == 0) {
			cout << "schreier::orbit_number idx == 0" << endl;
			exit(1);
			}
		idx--;
		}
	if (orbit_first[idx] <= pos &&
			pos < orbit_first[idx] + orbit_len[idx]) {
		return idx;
		}
	else {
		cout << "schreier::orbit_number something is wrong, "
				"perhaps the orbit of the point has not yet "
				"been computed" << endl;
		exit(1);
		}
}

void schreier::latex(const char *fname)
{
	INT f_with_cosetrep = TRUE;
	
	{
	ofstream fp(fname);

	latex_head_easy(fp);

	print_and_list_orbits_tex(fp);

	print_tables_latex(fp, f_with_cosetrep);

	latex_foot(fp);
	}
}

void schreier::get_orbit_decomposition_scheme_of_graph(
	INT *Adj, INT n, INT *&Decomp_scheme, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT I, J;
	INT f1, l1;
	INT f2, l2;
	INT i, j, r, r0, a, b;

	if (f_v) {
		cout << "schreier::get_orbit_decomposition_"
				"scheme_of_graph" << endl;
		}
	Decomp_scheme = NEW_INT(nb_orbits * nb_orbits);
	INT_vec_zero(Decomp_scheme, nb_orbits * nb_orbits);
	for (I = 0; I < nb_orbits; I++) {
		f1 = orbit_first[I];
		l1 = orbit_len[I];
		if (FALSE) {
			cout << "I = " << I << " f1 = " << f1
					<< " l1 = " << l1 << endl;
			}
		for (J = 0; J < nb_orbits; J++) {
			r0 = 0;
			f2 = orbit_first[J];
			l2 = orbit_len[J];
			if (FALSE) {
				cout << "J = " << J << " f2 = " << f2
						<< " l2 = " << l2 << endl;
				}
			for (i = 0; i < l1; i++) {
				a = orbit[f1 + i];
				r = 0;
				for (j = 0; j < l2; j++) {
					b = orbit[f2 + j];
					if (Adj[a * n + b]) {
						r++;
						}
					}
				if (i == 0) {
					r0 = r;
					}
				else {
					if (r0 != r) {
						cout << "schreier::get_orbit_decomposition_"
								"scheme_of_graph not tactical" << endl;
						cout << "I=" << I << endl;
						cout << "J=" << J << endl;
						cout << "r0=" << r0 << endl;
						cout << "r=" << r << endl;
						exit(1); 
						}
					}
				}
			if (FALSE) {
				cout << "I = " << I << " J = " << J << " r = " << r0 << endl;
				}
			Decomp_scheme[I * nb_orbits + J] = r0;
			}
		}
	if (f_v) {
		cout << "Decomp_scheme = " << endl;
		INT_matrix_print(Decomp_scheme, nb_orbits, nb_orbits);
		}
	if (f_v) {
		cout << "schreier::get_orbit_decomposition_"
				"scheme_of_graph done" << endl;
		}
}

void schreier::list_elements_as_permutations_vertically(ostream &ost)
{
	A->list_elements_as_permutations_vertically(&gens, ost);
}


