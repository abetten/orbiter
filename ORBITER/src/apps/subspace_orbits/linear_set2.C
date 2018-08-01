// linear_set2.C
// 
// Anton Betten
// July 15, 2014
//
//
//

#include "linear_set.h"


void linear_set::construct_semifield(INT orbit_for_W, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);

	if (f_v) {
		cout << "linear_set::construct_semifield orbit_for_W=" << orbit_for_W << endl;
		}

	INT *set1;
	INT *set2;
	INT dimU, dimW;
	INT *Basis1;
	INT *Basis2;
	INT *BasisU;
	INT *BasisW;
	INT i;

	set1 = NEW_INT(secondary_level);
	set2 = NEW_INT(secondary_depth);
	dimU = secondary_level + 1;
	dimW = secondary_depth;
	Basis1 = NEW_INT(secondary_level * n);
	Basis2 = NEW_INT(secondary_depth * n);
	BasisU = NEW_INT(dimU * n1);
	BasisW = NEW_INT(dimW * n1);

	INT_vec_zero(BasisU, dimU * n1);
	INT_vec_zero(BasisW, dimW * n1);

	Gen->get_set_by_level(secondary_level, secondary_orbit_at_level, set1);
	for (i = 0; i < secondary_level; i++) {
		PG_element_unrank_modified(*Fq, Basis1 + i * n, 1, n, set1[i]);
		}
	for (i = 0; i < secondary_level; i++) {
		PG_element_unrank_modified(*Fq, BasisU + i * n1, 1, n, set1[i]);
		}
	BasisU[secondary_level * n1 + n] = 1; // the vector v
	if (f_vv) {
		cout << "set1: ";
		INT_vec_print(cout, set1, secondary_level);
		cout << endl;
		cout << "Basis1:" << endl;
		INT_matrix_print(Basis1, secondary_level, n);
		cout << "BasisU:" << endl;
		INT_matrix_print(BasisU, dimU, n1);
		}


	Gen2->get_set_by_level(secondary_depth, orbit_for_W, set2);
	for (i = 0; i < secondary_depth; i++) {
		PG_element_unrank_modified(*Fq, Basis2 + i * n, 1, n, set2[i]);
		}
	for (i = 0; i < secondary_depth; i++) {
		PG_element_unrank_modified(*Fq, BasisW + i * n1, 1, n, set2[i]);
		}

	if (f_vv) {
		cout << "set2: ";
		INT_vec_print(cout, set2, secondary_depth);
		cout << endl;
		cout << "Basis2:" << endl;
		INT_matrix_print(Basis2, secondary_depth, n);
		cout << "BasisW:" << endl;
		INT_matrix_print(BasisW, dimW, n1);
		}


	INT *large_linear_set;
	INT large_linear_set_sz;
	INT *small_linear_set;
	INT small_linear_set_sz;
	INT *small_linear_set_W;
	INT small_linear_set_W_sz;
	
	D1->compute_linear_set(BasisU, dimU, 
		large_linear_set, large_linear_set_sz, 
		0 /*verbose_level*/);

	if (f_vv) {
		cout << "The large linear set of size " << large_linear_set_sz << " is ";
		INT_vec_print(cout, large_linear_set, large_linear_set_sz);
		cout << endl;
		D1->print_linear_set_tex(large_linear_set, large_linear_set_sz);
		cout << endl;
		}

	D->compute_linear_set(Basis1, secondary_level, 
		small_linear_set, small_linear_set_sz, 
		0 /*verbose_level*/);
	if (f_vv) {
		cout << "The small linear set of size " << small_linear_set_sz << " is ";
		INT_vec_print(cout, small_linear_set, small_linear_set_sz);
		cout << endl;
		D->print_linear_set_tex(small_linear_set, small_linear_set_sz);
		cout << endl;
		}


	D->compute_linear_set(Basis2, secondary_depth, 
		small_linear_set_W, small_linear_set_W_sz, 
		0 /*verbose_level*/);
	if (f_vv) {
		cout << "The small linear set for W of size " << small_linear_set_W_sz << " is ";
		INT_vec_print(cout, small_linear_set_W, small_linear_set_W_sz);
		cout << endl;
		D->print_linear_set_tex(small_linear_set_W, small_linear_set_W_sz);
		cout << endl;
		}

	INT *is_deleted;
	INT a, b, idx;

	for (i = 0; i < small_linear_set_sz; i++) {
		a = small_linear_set[i];
		b = spread_embedding[a];
		small_linear_set[i] = b;
		}
	if (f_vv) {
		cout << "After embedding, the small linear set of size " << small_linear_set_sz << " is ";
		INT_vec_print(cout, small_linear_set, small_linear_set_sz);
		cout << endl;
		D1->print_linear_set_tex(small_linear_set, small_linear_set_sz);
		cout << endl;
		}


	is_deleted = NEW_INT(large_linear_set_sz);
	for (i = 0; i < large_linear_set_sz; i++) {
		is_deleted[i] = FALSE;
		}

	for (i = 0; i < small_linear_set_sz; i++) {
		a = small_linear_set[i];
		if (!INT_vec_search(large_linear_set, large_linear_set_sz, a, idx)) {
			cout << "Cannot find embedded spread element in large linear set, something is wrong" << endl;
			exit(1);
			}
		is_deleted[idx] = TRUE;
		}
	
	INT *linear_set;
	INT linear_set_sz;
	INT j;

	linear_set_sz = 0;
	for (i = 0; i < large_linear_set_sz; i++) {
		if (!is_deleted[i]) {
			linear_set_sz++;
			}
		}
	linear_set = NEW_INT(linear_set_sz);
	j = 0;
	for (i = 0; i < large_linear_set_sz; i++) {
		if (!is_deleted[i]) {
			linear_set[j++] = large_linear_set[i];
			}
		}
	if (f_vv) {
		cout << "The linear set of size " << linear_set_sz << " is ";
		INT_vec_print(cout, linear_set, linear_set_sz);
		cout << endl;
		D1->print_linear_set_tex(linear_set, linear_set_sz);
		cout << endl;
		}


	INT *base_cols;
	INT *kernel_cols;
	INT *Spread_element_basis;
	INT *Basis_elt;
	INT *Basis_infinity;
	INT h;
	INT *v1, *v2;
	INT n2;

	n2 = n1 - dimW;
	Spread_element_basis = NEW_INT(D1->spread_element_size);
	Basis_infinity = NEW_INT(s * n2);
	Basis_elt = NEW_INT(dimW * n2);
	base_cols = NEW_INT(n1);
	kernel_cols = NEW_INT(n1);
	if (Fq->Gauss_simple(BasisW, dimW, n1, base_cols, 0/* verbose_level*/) != dimW) {
		cout << "BasisW does not have the correct rank" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "BasisW:" << endl;
		INT_matrix_print(BasisW, dimW, n1);
		cout << "base_cols:";
		INT_vec_print(cout, base_cols, dimW);
		cout << endl;
		}

	Fq->kernel_columns(n1, dimW, base_cols, kernel_cols);
	if (f_vv) {
		cout << "kernel_cols:";
		INT_vec_print(cout, kernel_cols, n2);
		cout << endl;
		}



	INT_vec_zero(Basis_infinity, s * n2);
	for (i = 0; i < s; i++) {
		//a = kernel_cols[i] - s;
		Basis_infinity[i * n2 + i] = 1;
		}
	if (f_vv) {
		cout << "Basis element infinity:" << endl;
		INT_matrix_print(Basis_infinity, s, n2);
		}
	

	INT nb_components;
	INT **Components;
	INT *Spread_set;

	nb_components = linear_set_sz + 1;
	Components = NEW_PINT(nb_components);
	Spread_set = NEW_INT(linear_set_sz * s * s);


	Components[0] = NEW_INT(s * n2);
	INT_vec_copy(Basis_infinity, Components[0], s * n2);

	for (h = 0; h < linear_set_sz; h++) {
		if (f_v3) {
			cout << "spread element " << h << " / " << linear_set_sz << ":" << endl;
			}
		a = linear_set[h];
		INT_vec_copy(D1->Spread_elements + a * D1->spread_element_size, Spread_element_basis, D1->spread_element_size);
		if (f_v3) {
			cout << "Spread element " << a << " is:" << endl;
			INT_matrix_print(Spread_element_basis, s, n1);
			}
		
		for (i = 0; i < dimW; i++) {
			a = base_cols[i];
			v1 = BasisW + i * n1;
			for (j = 0; j < s; j++) {
				v2 = Spread_element_basis + j * n1;
				if (v2[a]) {
					Fq->Gauss_step(v1, v2, n1, a, 0 /* verbose_level*/);
					}
				}
			}
		if (f_v3) {
			cout << "Basis after reduction mod W:" << endl;
			INT_matrix_print(Spread_element_basis, s, n1);
			}

		for (i = 0; i < dimW; i++) {
			for (j = 0; j < n2; j++) {
				a = kernel_cols[j];
				Basis_elt[i * n2 + j] = Spread_element_basis[i * n1 + a];
				}
			}

		if (f_v3) {
			cout << "Basis element:" << endl;
			INT_matrix_print(Basis_elt, s, n2);
			}

		Fq->Gauss_easy(Basis_elt, s, n2);

		if (f_v3) {
			cout << "Basis element after RREF:" << endl;
			INT_matrix_print(Basis_elt, s, n2);
			}

		for (i = 0; i < s; i++) {
			for (j = 0; j < s; j++) {
				a = Basis_elt[i * n2 + s + j];
				Spread_set[h * s * s + i * s + j] = a;
				}
			}

		Components[h + 1] = NEW_INT(s * n2);
		INT_vec_copy(Basis_elt, Components[h + 1], s * n2);
		}
	
	if (f_v3) {
		cout << "The components are:" << endl;
		for (h = 0; h < linear_set_sz + 1; h++) {
			cout << "Component " << h << " / " << linear_set_sz << ":" << endl;
			INT_matrix_print(Components[h], s, n2);
			}
		}

	INT h2;

	h2 = 0;
	for (h = 0; h < linear_set_sz + 1; h++) {
		if (h == 1) {
			continue;
			}
		for (i = 0; i < s; i++) {
			for (j = 0; j < s; j++) {
				a = Components[h][i * n2 + s + j];
				Spread_set[h2 * s * s + i * s + j] = a;
				}
			}

		h2++;
		}

	INT h1, k3;
	INT *Intersection;
	
	Intersection = NEW_INT(n2 * n2);
	for (h1 = 0; h1 < nb_components; h1++) {
		for (h2 = h1 + 1; h2 < nb_components; h2++) {
			Fq->intersect_subspaces(n2, s, Components[h1], s, Components[h2], 
				k3, Intersection, 0 /* verbose_level */);
			if (k3) {
				cout << "Components " << h1 << " and " << h2 << " intersect non-trivially!" << endl;
				cout << "Component " << h1 << " / " << nb_components << ":" << endl;
				INT_matrix_print(Components[h1], s, n2);
				cout << "Component " << h2 << " / " << nb_components << ":" << endl;
				INT_matrix_print(Components[h2], s, n2);
				}
			}
		}
	if (f_vv) {
		cout << "The components are disjoint!" << endl;
		}


	INT rk;
	
	if (f_v3) {
		cout << "The spread_set is:" << endl;
		INT_matrix_print(Spread_set, linear_set_sz, s * s);
		}
	rk = Fq->Gauss_easy(Spread_set, linear_set_sz, s * s);
	if (f_v) {
		cout << "rank = " << rk << endl;
		}
	if (f_v3) {
		cout << "The spread_set basis is:" << endl;
		INT_matrix_print(Spread_set, rk, s * s);
		for (h = 0; h < rk; h++) {
			cout << "basis elt " << h << " / " << rk << ":" << endl;
			INT_matrix_print(Spread_set + h * s * s, s, s);
			}
		}



	if (f_v3) {
		cout << "opening grassmann:" << endl;
		}
	grassmann *Grass;
	Grass = new grassmann;
	Grass->init(n2, s, Fq, 0 /*verbose_level*/);

	INT *spread_elements_numeric;

	spread_elements_numeric = NEW_INT(nb_components);
	for (h = 0; h < nb_components; h++) {
		spread_elements_numeric[h] = Grass->rank_INT_here(Components[h], 0);
		}
	
	if (f_vv) {
		cout << "spread elements numeric:" << endl;
		for (i = 0; i < nb_components; i++) {
			cout << setw(3) << i << " : " << spread_elements_numeric[i] << endl;
			}
		}

	if (f_identify) {
		cout << "linear_set::construct_semifield before T->identify" << endl;

		if (nb_components != order + 1) {
			cout << "nb_components != order + 1" << endl;
			exit(1);
			}

		INT *transporter;
		INT f_implicit_fusion = FALSE;
		INT final_node;

		transporter = NEW_INT(T->gen->A->elt_size_in_INT);

		recognize(
			T->gen, 
			spread_elements_numeric, nb_components, 
			transporter, f_implicit_fusion, 
			final_node, 0 /*verbose_level*/);
		//T->identify(spread_elements_numeric, nb_components, verbose_level);

		longinteger_object go;
		INT lvl;
		INT orbit_at_lvl;

		lvl = order + 1;
		orbit_at_lvl = final_node - T->gen->first_oracle_node_at_level[lvl];

		T->gen->get_stabilizer_order(lvl, orbit_at_lvl, go);

		cout << "linear_set::construct_semifield after recognize" << endl;
		cout << "final_node=" << final_node << " which is isomorphism type " << orbit_at_lvl << " with stabilizer order " << go << endl;
		cout << "transporter=" << endl;
		T->gen->A->element_print_quick(transporter, cout);

		FREE_INT(transporter);
		}

#if 0
	andre_construction *Andre;

	Andre = new andre_construction;
	
	cout << "Creating the projective plane using the Andre construction:" << endl;
	Andre->init(Fq, s, spread_elements_numeric, 0 /*verbose_level*/);
	cout << "Done creating the projective plane using the Andre construction." << endl;
#endif

	
	FREE_INT(large_linear_set);
	FREE_INT(small_linear_set);
	FREE_INT(linear_set);
	FREE_INT(small_linear_set_W);
	FREE_INT(set1);
	FREE_INT(set2);
	FREE_INT(Basis1);
	FREE_INT(Basis2);
	FREE_INT(BasisU);
	FREE_INT(BasisW);
}


