/*
 * ug_5_3_quaternion.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter;



int main()
{

	orbiter::layer5_applications::user_interface::orbiter_top_level_session Orbiter;

	int verbose_level = 2;
	int q = 3;
	int f_without_tables = false;
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);

	F->finite_field_init_small_order(q,
			f_without_tables,
			true /* f_compute_related_fields */,
			verbose_level);

	int gens[] = { 1,1,1,2, 2,1,1,1, 0,2,1,0 };


	actions::action *A;
	data_structures_groups::vector_ge *nice_gens;
	data_structures_groups::vector_ge *subgroup_gens;

	A = NEW_OBJECT(actions::action);

	A->Known_groups->init_general_linear_group(2, F,
			false /*f_semilinear */, true /* f_basis */, false /* f_init_sims */,
			nice_gens,
			verbose_level);


	groups::strong_generators *Gens;

	Gens = NEW_OBJECT(groups::strong_generators);

	ring_theory::longinteger_object target_go;

	target_go.create(8);

	Gens->init_from_data_with_target_go(A,
			gens,
			4, 3,
			target_go,
			subgroup_gens,
			verbose_level);

	groups::sims *S;

	S = Gens->create_sims(verbose_level);

	long int go;
	int i, j, k;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	go = S->group_order_lint();
	for (i = 0; i < go; i++) {
		S->element_unrank_lint(i, Elt1);
		for (j = 0; j < go; j++) {
			S->element_unrank_lint(j, Elt2);
			A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
			k = S->element_rank_lint(Elt3);
			cout << k;
			if (j < go - 1) {
				cout << "\t";
			}
		}
		cout << endl;
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_OBJECT(Gens);
	FREE_OBJECT(S);
	FREE_OBJECT(A);
	FREE_OBJECT(F);

}




