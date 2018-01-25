// sims2.C
//
// Anton Betten
// January 11, 2009

#include "galois.h"
#include "action.h"

void choose_random_generator_derived_group(sims *G, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6;
	action *A;
	
	if (f_v) {
		cout << "choose_random_generator_derived_group" << endl;
		}
	A = G->A;
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	Elt4 = NEW_INT(A->elt_size_in_INT);
	Elt5 = NEW_INT(A->elt_size_in_INT);
	Elt6 = NEW_INT(A->elt_size_in_INT);
	
	G->random_element(Elt1, verbose_level - 1);
	G->random_element(Elt2, verbose_level - 1);
	A->invert(Elt1, Elt3);
	A->invert(Elt2, Elt4);
	A->mult(Elt3, Elt4, Elt5);
	A->mult(Elt1, Elt2, Elt6);
	A->mult(Elt5, Elt6, Elt);
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(Elt4);
	FREE_INT(Elt5);
	FREE_INT(Elt6);
}

void sims::build_up_subgroup_random_process(sims *G, 
	void (*choose_random_generator_for_subgroup)(sims *G, INT *Elt, INT verbose_level), 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	//INT f_vvvv = (verbose_level >= 10);
	longinteger_object go, G_order, quo, rem;
	INT drop_out_level, image, cnt, f_added;
	action *GA;
	
	GA = A;
	
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process" << endl;
		}
	G->group_order(G_order);
	group_order(go);
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process(): old group order is " << G_order << endl;
		cout << "the group is in action " << G->A->label << " with base_length = " << G->A->base_len 
			<< " and degree " << G->A->degree << endl;
		cout << "the image action has base_length = " << GA->base_len 
			<< " and degree " << GA->degree << endl;
		cout << "current action " << GA->label << endl;
		cout << "current group order = " << go << endl;
		}
	cnt = 0;
	while (cnt < 200) {
	
		if (f_vv) {
			cout << "iteration " << cnt << endl;
			}
#if 0
		if (cnt > 1000) {
			cout << "sims::build_up_group_random_process() cnt > 1000, something seems to be wrong" << endl;
			test_if_subgroup(G, 2);
			exit(1);
			}
#endif
		if (FALSE) {
			G->A->print_base();
			G->print_orbit_len();
			}
		if ((cnt % 2) == 0) {
			if (f_vvv) {
				cout << "choosing random schreier generator" << endl;
				}
			random_schreier_generator(0/*verbose_level - 3*/);
			A->element_move(schreier_gen, GA->Elt1, 0);
			if (FALSE) {
				cout << "random element chosen:" << endl;
				A->element_print(GA->Elt1, cout);
				cout << endl;
				}
			}
		else if ((cnt % 2) == 1){
			if (f_vvv) {
				cout << "choosing random element in the group by which we extend" << endl;
				}
			(*choose_random_generator_for_subgroup)(G, GA->Elt1, verbose_level - 1);
			if (FALSE) {
				cout << "random element chosen" << endl;
				}
			if (FALSE) {
				GA->element_print(GA->Elt1, cout);
				cout << endl;
				}
			}
		if (strip(GA->Elt1, GA->Elt2, drop_out_level, image, 0/*verbose_level*/)) {
			if (f_vvv) {
				cout << "element strips through" << endl;
				if (FALSE) {
					cout << "residue = " << endl;
					GA->element_print(GA->Elt2, cout);
					cout << endl;
					}
				}
			f_added = FALSE;
			closure_group(100, verbose_level - 2);
			}
		else {
			f_added = TRUE;
			if (f_v) {
				cout << "element needs to be inserted at level = " 
					<< drop_out_level << " with image " << image << endl;
				if (TRUE) {
					GA->element_print(GA->Elt2, cout);
					cout  << endl;
					}
				}
			add_generator_at_level(GA->Elt2, drop_out_level, 0/*verbose_level - 3*/);
			}
		
		group_order(go);
		if ((f_v && f_added) || f_vv) {
			cout << "new group order is " << go << " : ";
			print_transversal_lengths();
			}
		cnt++;
		}
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process finished: found a group of order " << go << endl;
		print_transversal_lengths();
		}
}

