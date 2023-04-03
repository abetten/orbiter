// sims2.cpp
//
// Anton Betten
// January 11, 2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



void choose_random_generator_derived_group(
		sims *G,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6;
	actions::action *A;
	
	if (f_v) {
		cout << "choose_random_generator_derived_group" << endl;
		}
	A = G->A;
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	Elt5 = NEW_int(A->elt_size_in_int);
	Elt6 = NEW_int(A->elt_size_in_int);
	
	G->random_element(Elt1, verbose_level - 1);
	G->random_element(Elt2, verbose_level - 1);
	A->Group_element->invert(Elt1, Elt3);
	A->Group_element->invert(Elt2, Elt4);
	A->Group_element->mult(Elt3, Elt4, Elt5);
	A->Group_element->mult(Elt1, Elt2, Elt6);
	A->Group_element->mult(Elt5, Elt6, Elt);
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(Elt5);
	FREE_int(Elt6);
}

void sims::build_up_subgroup_random_process(
		sims *G,
	void (*choose_random_generator_for_subgroup)(sims *G,
			int *Elt, int verbose_level),
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	//int f_vvvv = (verbose_level >= 10);
	ring_theory::longinteger_object go, G_order, quo, rem;
	int drop_out_level, image, cnt, f_added;
	actions::action *GA;
	
	GA = A;
	
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process" << endl;
		}
	G->group_order(G_order);
	group_order(go);
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process(): "
				"old group order is " << G_order << endl;
		cout << "the group is in action " << G->A->label
				<< " with base_length = " << G->A->base_len()
			<< " and degree " << G->A->degree << endl;
		cout << "the image action has base_length = " << GA->base_len()
			<< " and degree " << GA->degree << endl;
		cout << "current action " << GA->label << endl;
		cout << "current group order = " << go << endl;
		}
	cnt = 0;
	while (cnt < 200) {
	
		if (f_vv) {
			cout << "sims::build_up_subgroup_random_process iteration " << cnt << endl;
			}
#if 0
		if (cnt > 1000) {
			cout << "sims::build_up_group_random_process "
					"cnt > 1000, something seems to be wrong" << endl;
			test_if_subgroup(G, 2);
			exit(1);
			}
#endif
		if (false) {
			G->A->print_base();
			G->print_orbit_len();
			}
		if ((cnt % 2) == 0) {
			if (f_vvv) {
				cout << "sims::build_up_subgroup_random_process choosing random schreier generator" << endl;
				}
			random_schreier_generator(GA->Elt1, 0/*verbose_level - 3*/);
			//A->element_move(schreier_gen, GA->Elt1, 0);
			if (false) {
				cout << "sims::build_up_subgroup_random_process random element chosen:" << endl;
				A->Group_element->element_print(GA->Elt1, cout);
				cout << endl;
				}
			}
		else if ((cnt % 2) == 1){
			if (f_vvv) {
				cout << "sims::build_up_subgroup_random_process choosing random element in the "
						"group by which we extend" << endl;
				}
			(*choose_random_generator_for_subgroup)(G,
					GA->Elt1, verbose_level - 1);
			if (false) {
				cout << "sims::build_up_subgroup_random_process random element chosen" << endl;
				}
			if (false) {
				GA->Group_element->element_print(GA->Elt1, cout);
				cout << endl;
				}
			}
		if (strip(GA->Elt1, GA->Elt2, drop_out_level,
				image, 0/*verbose_level*/)) {
			if (f_vvv) {
				cout << "sims::build_up_subgroup_random_process element strips through" << endl;
				if (false) {
					cout << "sims::build_up_subgroup_random_process residue = " << endl;
					GA->Group_element->element_print(GA->Elt2, cout);
					cout << endl;
					}
				}
			f_added = false;
			closure_group(100, verbose_level - 2);
			}
		else {
			f_added = true;
			if (f_v) {
				cout << "sims::build_up_subgroup_random_process element needs to be inserted at level = "
					<< drop_out_level << " with image "
					<< image << endl;
				if (true) {
					GA->Group_element->element_print(GA->Elt2, cout);
					cout  << endl;
					}
				}
			add_generator_at_level(GA->Elt2,
					drop_out_level, 0/*verbose_level - 3*/);
			}
		
		group_order(go);
		if ((f_v && f_added) || f_vv) {
			cout << "sims::build_up_subgroup_random_process new group order is " << go << " : ";
			print_transversal_lengths();
			}
		cnt++;
		}
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process "
				"finished: found a group of order " << go << endl;
		print_transversal_lengths();
		}
}

}}}

