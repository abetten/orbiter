// sims3.cpp
//
// Anton Betten
// November 22, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


void sims::subgroup_make_characteristic_vector(
		sims *Sub, int *C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int go, go_sub;
	long int i, j;


	if (f_v) {
		cout << "sims::subgroup_make_characteristic_vector" << endl;
		}

	go = group_order_lint();
	Int_vec_zero(C, go);
	go_sub = Sub->group_order_lint();
	for (i = 0; i < go_sub; i++) {
		Sub->element_unrank_lint(i, Elt1);
		j = element_rank_lint(Elt1);
		C[j] = true;
		}
	if (f_v) {
		cout << "The characteristic vector of the "
				"subgroup of order " << go_sub << " is:" << endl;
		other::orbiter_kernel_system::Orbiter->Int_vec->print_as_table(cout, C, go, 25);
		}
	if (f_v) {
		cout << "sims::subgroup_make_characteristic_vector done" << endl;
		}
}

void sims::normalizer_based_on_characteristic_vector(
		int *C_sub,
	int *Gen_idx, int nb_gens, int *N, long int &N_go,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int go;
	long int i, j, a;
	data_structures_groups::vector_ge *gens;


	if (f_v) {
		cout << "sims::normalizer_based_on_characteristic_vector" << endl;
		}

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	for (j = 0; j < nb_gens; j++) {
		a = Gen_idx[j];
		element_unrank_lint(a, gens->ith(j));
		}

	go = group_order_lint();
	Int_vec_zero(N, go);

	N_go = 0;
	for (i = 0; i < go; i++) {
		element_unrank_lint(i, Elt1);
		A->Group_element->element_invert(Elt1, Elt2, 0);
		for (j = 0; j < nb_gens; j++) {
			A->Group_element->element_mult(Elt2, gens->ith(j), Elt3, 0);
			A->Group_element->element_mult(Elt3, Elt1, Elt4, 0);
			a = element_rank_lint(Elt4);
			if (!C_sub[a]) {
				break;
				}
			}
		if (j == nb_gens) {
			N[i]++;
			N_go++;
			}
		}
	FREE_OBJECT(gens);
	if (f_v) {
		cout << "sims::normalizer_based_on_characteristic_vector done" << endl;
		}
}

void sims::order_structure_relative_to_subgroup(
		int *C_sub,
	int *Order, int *Residue, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int go;
	long int i, j, a;


	if (f_v) {
		cout << "sims::order_structure_relative_to_subgroup" << endl;
		}


	go = group_order_lint();
	for (i = 0; i < go; i++) {
		element_unrank_lint(i, Elt1);
		A->Group_element->element_move(Elt1, Elt2, 0);
		for (j = 1; ; j++) {
			a = element_rank_lint(Elt2);
			if (C_sub[a]) {
				break;
				}
			A->Group_element->element_mult(Elt2, Elt1, Elt3, 0);
			A->Group_element->element_move(Elt3, Elt2, 0);
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

}}}

