// factor_group.C
// 
// Anton Betten
// started:     03/10/2009
// last change: 03/12/2009
//
// 
//
//

#include "orbiter.h"


void create_factor_group(action *A, sims *S, INT goi, 
	INT size_subgroup, INT *subgroup, factor_group *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1;
	INT i;
	
	if (f_v) {
		cout << "create_factor_group" << endl;
		}
	Elt1 = new INT[A->elt_size_in_INT];
	
	F->goi = goi;
	F->A = A;
	F->S = S;
	F->size_subgroup = size_subgroup;
	F->subgroup = subgroup;
	
	F->all_cosets = new INT[goi];
	
	
	if (f_v) {
		cout << "create_factor_group" << endl;
		cout << "the subgroup is:" << endl;
		for (i = 0; i < size_subgroup; i++) {
			cout << i << " element has rank " << subgroup[i] << endl;
			S->element_unrank_INT(subgroup[i], Elt1);
			A->print(cout, Elt1);
			//A->print_as_permutation(cout, Elt1);
			cout << endl;
			}
		cout << endl << endl;
		}
	
	if (f_v) {
		cout << "create_factor_group before S->all_cosets" << endl;
		}
	S->all_cosets(subgroup, size_subgroup, F->all_cosets, verbose_level);
	F->nb_cosets = goi / size_subgroup;
	
	F->ByRightMultiplication = new action;
	F->FactorGroup = new action;
	F->FactorGroupConjugated = new action;
		
	if (f_v) {
		cout << "create_factor_group before induced_action_by_right_multiplication" << endl;
		}
	F->ByRightMultiplication->induced_action_by_right_multiplication(FALSE /* f_basis */, S, S, FALSE, verbose_level);

#if 0
	for (u = 0; u < SG->len; u++) {
		for (i = 0; i < goi; i++) {
			j = ByRightMultiplication.image_of(SG->ith(u), i);
			if (j != perm[u * goi + i]) {
				cout << "problem in action by right multiplication" << endl;
				exit(1);
				}
			}
		}
	cout << "action by right multiplication works" << endl << endl;
#endif



	if (f_v) {
		cout << "create_factor_group before induced_action_on_sets" << endl;
		}
	F->FactorGroup->induced_action_on_sets(*F->ByRightMultiplication, 
		S, F->nb_cosets, size_subgroup, F->all_cosets, TRUE, verbose_level);

	if (f_v) {
		cout << "create_factor_group after induced_action_on_sets" << endl;
		longinteger_object go;
		F->FactorGroup->Sims->group_order(go);
		cout << "induced group has order " << go << endl;
		}



#if 0
	//INT goi_factor_group;
	longinteger_object go;
	
	F->FactorGroup->Sims->group_order(go);
	F->goi_factor_group = go.as_INT();
	if (f_v) {
		cout << "create_factor_group: goi_factor_group = " << F->goi_factor_group << ":" << endl;
		cout << "create_factor_group: computing the regular representation of degree " << F->goi_factor_group << ":" << endl;
		}
		
#if 0
	for (i = 0; i < SG->len; i++) {
		F->FactorGroup->print_as_permutation(cout, SG->ith(i));
		cout << endl;
		}
	cout << endl;
#endif

	if (f_v) {
		cout << "create_factor_group: before induced_action_by_right_multiplication:" << endl;
		}
	F->FactorGroupConjugated->induced_action_by_right_multiplication(F->FactorGroup->Sims, 
		F->FactorGroup->Sims, FALSE, verbose_level);
	if (f_v) {
		cout << "create_factor_group: after induced_action_by_right_multiplication:" << endl;
		}
#endif


	delete [] Elt1;
}

