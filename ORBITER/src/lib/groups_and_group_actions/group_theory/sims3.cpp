// sims3.C
//
// Anton Betten
// November 22, 2016

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

void sims::subgroup_make_characteristic_vector(
		sims *Sub, INT *C, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT go, go_sub;
	INT i, j;


	if (f_v) {
		cout << "sims::subgroup_make_characteristic_vector" << endl;
		}

	go = group_order_INT();
	INT_vec_zero(C, go);
	go_sub = Sub->group_order_INT();
	for (i = 0; i < go_sub; i++) {
		Sub->element_unrank_INT(i, Elt1);
		j = element_rank_INT(Elt1);
		C[j] = TRUE;
		}
	if (f_v) {
		cout << "The characteristic vector of the "
				"subgroup of order " << go_sub << " is:" << endl;
		INT_vec_print_as_table(cout, C, go, 25);
		}
	if (f_v) {
		cout << "sims::subgroup_make_characteristic_vector done" << endl;
		}
}

void sims::normalizer_based_on_characteristic_vector(INT *C_sub, 
	INT *Gen_idx, INT nb_gens, INT *N, INT &N_go, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT go;
	INT i, j, a;
	vector_ge *gens;


	if (f_v) {
		cout << "sims::normalizer_based_on_characteristic_vector" << endl;
		}

	gens = new vector_ge;
	gens->init(A);
	gens->allocate(nb_gens);
	for (j = 0; j < nb_gens; j++) {
		a = Gen_idx[j];
		element_unrank_INT(a, gens->ith(j));
		}

	go = group_order_INT();
	INT_vec_zero(N, go);

	N_go = 0;
	for (i = 0; i < go; i++) {
		element_unrank_INT(i, Elt1);
		A->element_invert(Elt1, Elt2, 0);
		for (j = 0; j < nb_gens; j++) {
			A->element_mult(Elt2, gens->ith(j), Elt3, 0);
			A->element_mult(Elt3, Elt1, Elt4, 0);
			a = element_rank_INT(Elt4);
			if (!C_sub[a]) {
				break;
				}
			}
		if (j == nb_gens) {
			N[i]++;
			N_go++;
			}
		}
	delete gens;
	if (f_v) {
		cout << "sims::normalizer_based_on_characteristic_vector done" << endl;
		}
}

void sims::order_structure_relative_to_subgroup(INT *C_sub, 
	INT *Order, INT *Residue, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT go;
	INT i, j, a;


	if (f_v) {
		cout << "sims::order_structure_relative_to_subgroup" << endl;
		}


	go = group_order_INT();
	for (i = 0; i < go; i++) {
		element_unrank_INT(i, Elt1);
		A->element_move(Elt1, Elt2, 0);
		for (j = 1; ; j++) {
			a = element_rank_INT(Elt2);
			if (C_sub[a]) {
				break;
				}
			A->element_mult(Elt2, Elt1, Elt3, 0);
			A->element_move(Elt3, Elt2, 0);
			}
		Order[i] = j;
		Residue[i] = a;
#if 0
		if ((j % 2) == 0) {
			cout << "element " << i << " has relative order " << j << endl;
			cout << "element:" << endl;
			A->element_print(Elt1, cout);
			}
#endif
		}

	if (f_v) {
		cout << "sims::order_structure_relative_to_subgroup done" << endl;
		}
}

