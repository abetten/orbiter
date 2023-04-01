// action_induce.cpp
//
// Anton Betten
// 1/1/2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


int action::least_moved_point_at_level(int level, int verbose_level)
{
	return Sims->least_moved_point_at_level(level, verbose_level);
}

void action::lex_least_base_in_place(
		groups::sims *old_Sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *set;
	long int *old_base;
	int i, lmp, old_base_len;

	if (f_v) {
		cout << "action::lex_least_base_in_place action "
				<< label << " base=";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << endl;
		print_info();
		//cout << "the generators are:" << endl;
		//Sims->print_generators();
	}

	set = NEW_lint(degree);
	old_base = NEW_lint(base_len());
	old_base_len = base_len();
	for (i = 0; i < base_len(); i++) {
		old_base[i] = base_i(i);
	}



	for (i = 0; i < base_len(); i++) {
		set[i] = base_i(i);
		if (f_v) {
			cout << "action::lex_least_base_in_place "
					"i=" << i << " computing the least moved point" << endl;
		}
		lmp = least_moved_point_at_level(i, verbose_level - 2);
		if (f_v) {
			cout << "action::lex_least_base_in_place "
					"i=" << i << " the least moved point is " << lmp << endl;
		}
		if (lmp >= 0 && lmp < base_i(i)) {
			if (f_v) {
				cout << "action::lex_least_base_in_place "
						"i=" << i << " least moved point = " << lmp
					<< " less than base point " << base_i(i) << endl;
				cout << "doing a base change:" << endl;
			}
			set[i] = lmp;
			base_change_in_place(i + 1, set, old_Sims, verbose_level);
			if (f_v) {
				cout << "action::lex_least_base_in_place "
						"after base_change_in_place: action:" << endl;
				print_info();
			}
 		}
	}
	if (f_v) {
		cout << "action::lex_least_base_in_place "
				"done, action " << label << " base=";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << endl;
		print_info();
		//cout << "the generators are:" << endl;
		//Sims->print_generators();
		int f_changed = FALSE;

		if (old_base_len != base_len()) {
			f_changed = TRUE;
		}
		if (!f_changed) {
			for (i = 0; i < base_len(); i++) {
				if (old_base[i] != base_i(i)) {
					f_changed = TRUE;
					break;
				}
			}
		}
		if (f_changed) {
			cout << "The base has changed !!!" << endl;
			cout << "old base: ";
			Lint_vec_print(cout, old_base, old_base_len);
			cout << endl;
			cout << "new base: ";
			//int_vec_print(cout, Stabilizer_chain->base, base_len());
			cout << endl;
		}
	}
	FREE_lint(old_base);
	FREE_lint(set);
}

void action::lex_least_base(
		action *old_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *set;
	action *old_A;
	int i, lmp;

	if (f_v) {
		cout << "action::lex_least_base action "
				<< old_action->label << " base=";
		//int_vec_print(cout, old_action->Stabilizer_chain->base, old_action->base_len());
		cout << endl;
	}
#if 0
	if (!f_has_sims) {
		cout << "action::lex_least_base fatal: does not have sims" << endl;
		exit(1);
	}
#endif


	if (f_v) {
		//cout << "the generators are:" << endl;
		//old_action->Sims->print_generators();
	}
	//A = NEW_OBJECT(action);

	set = NEW_lint(old_action->degree);

	old_A = old_action;

	if (!old_action->f_has_sims) {
		cout << "action::lex_least_base does not have Sims" << endl;
		exit(1);
	}

	for (i = 0; i < old_A->base_len(); i++) {
		set[i] = old_A->base_i(i);
		if (f_v) {
			cout << "action::lex_least_base "
					"calling least_moved_point_at_level " << i << endl;
		}
		lmp = old_A->least_moved_point_at_level(i, verbose_level - 2);
		if (lmp < old_A->base_i(i)) {
			if (f_v) {
				cout << "action::lex_least_base least moved point = " << lmp
					<< " less than base point " << old_A->base_i(i) << endl;
				cout << "doing a base change:" << endl;
			}
			set[i] = lmp;
			//A = NEW_OBJECT(action);

			action *A;

			A = old_A->Induced_action->base_change(
					i + 1,
					set,
					old_action->Sims,
					verbose_level - 2);
			old_A = A;
		}
	}
	old_A->Induced_action->base_change(
			old_A->base_len(),
			old_A->get_base(),
			old_action->Sims,
			verbose_level - 1);

	FREE_lint(set);
	if (f_v) {
		cout << "action::lex_least_base action " << label << " base=";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << endl;
	}
}

int action::test_if_lex_least_base(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *AA;
	int i;

	if (f_v) {
		cout << "action::test_if_lex_least_base:" << endl;
		print_info();
	}

	AA = NEW_OBJECT(action);

	AA->lex_least_base(this, verbose_level);
	for (i = 0; i < base_len(); i++) {
		if (AA->base_len() >= i) {
			if (base_i(i) > AA->base_i(i)) {
				cout << "action::test_if_lex_least_base "
						"returns FALSE" << endl;
				cout << "base[i]=" << base_i(i) << endl;
				cout << "AA->base[i]=" << AA->base_i(i) << endl;
				FREE_OBJECT(AA);
				return FALSE;
			}
		}
	}
	FREE_OBJECT(AA);
	return TRUE;
}

void action::base_change_in_place(
		int size, long int *set, groups::sims *old_Sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	int i;

	if (f_v) {
		cout << "action::base_change_in_place" << endl;
	}
	//A = NEW_OBJECT(action);

	//action_global AG;


	if (f_v) {
		cout << "action::base_change_in_place "
				"before Induced_action->base_change" << endl;
	}
	A = Induced_action->base_change(size, set, old_Sims, verbose_level);
	if (f_v) {
		cout << "action::base_change_in_place "
				"after Induced_action->base_change" << endl;
	}
	Stabilizer_chain->free_base_data();
	if (f_v) {
		cout << "action::base_change_in_place "
				"after free_base_data" << endl;
	}
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(
			this, A->base_len(), verbose_level);
	//allocate_base_data(A->base_len);
	if (f_v) {
		cout << "action::base_change_in_place "
				"after allocate_base_data"
				<< endl;
	}
	set_base_len(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		base_i(i) = A->base_i(i);
	}
	if (f_v) {
		cout << "action::base_change_in_place after copying base" << endl;
	}


	A->Sims->A = this;
	A->Sims->gens.A = this;
	A->Sims->gens_inv.A = this;
		// not to forget: the sims also has an action pointer in it
		// and this one has to be changed to the old action
	if (f_v) {
		cout << "action::base_change_in_place "
				"after changing action pointer in A->Sims" << endl;
	}

	if (f_has_sims) {
		if (f_v) {
			cout << "action::base_change_in_place "
					"before FREE_OBJECT Sims" << endl;
			cout << "Sims=" << Sims << endl;
		}
		FREE_OBJECT(Sims);
		if (f_v) {
			cout << "action::base_change_in_place "
					"after FREE_OBJECT Sims" << endl;
		}
		Sims = NULL;
		f_has_sims = FALSE;
	}

	if (f_v) {
		cout << "action::base_change_in_place after deleting sims" << endl;
	}

	if (f_v) {
		cout << "action::base_change_in_place before init_sims_only" << endl;
	}
	init_sims_only(A->Sims, verbose_level);
	if (f_v) {
		cout << "action::base_change_in_place after init_sims_only" << endl;
	}

	if (f_has_strong_generators) {
		f_has_strong_generators = FALSE;
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
	}

	A->f_has_sims = FALSE;
	A->Sims = NULL;

	if (f_v) {
		cout << "action::base_change_in_place "
				"before FREE_OBJECT(A)" << endl;
	}
	FREE_OBJECT(A);
	if (f_v) {
		cout << "action::base_change_in_place "
				"after FREE_OBJECT(A)" << endl;
	}

	compute_strong_generators_from_sims(verbose_level - 3);

	if (f_v) {
		cout << "action::base_change_in_place finished, created action"
				<< endl;
		print_info();
		//cout << "generators are:" << endl;
		//Sims->print_generators();
		//cout << "Sims:" << endl;
		//Sims->print(3);
	}
	if (f_v) {
		cout << "action::base_change_in_place done" << endl;
	}
}


void action::create_orbits_on_subset_using_restricted_action(
		action *&A_by_restriction,
		groups::schreier *&Orbits, groups::sims *S,
		int size, long int *set,
		std::string &label_of_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_induce = FALSE;

	if (f_v) {
		cout << "action::create_orbits_on_subset_using_restricted_action" << endl;
	}
	A_by_restriction = Induced_action->create_induced_action_by_restriction(
			S,
			size, set,
			label_of_set,
			f_induce,
			verbose_level - 1);
	Orbits = NEW_OBJECT(groups::schreier);

	A_by_restriction->compute_all_point_orbits(*Orbits,
			S->gens, verbose_level - 2);
	if (f_v) {
		cout << "action::create_orbits_on_subset_using_restricted_action "
				"done" << endl;
	}
}

void action::create_orbits_on_sets_using_action_on_sets(
		action *&A_on_sets,
		groups::schreier *&Orbits, groups::sims *S,
		int nb_sets, int set_size, long int *sets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_induce = FALSE;

	if (f_v) {
		cout << "action::create_orbits_on_sets_using_action_on_sets" << endl;
	}

	A_on_sets = Induced_action->create_induced_action_on_sets(
			nb_sets, set_size, sets,
			verbose_level);

	Orbits = NEW_OBJECT(groups::schreier);

	A_on_sets->compute_all_point_orbits(*Orbits, S->gens, verbose_level - 2);
	if (f_v) {
		cout << "action::create_orbits_on_sets_using_action_on_sets "
				"done" << endl;
	}
}




int action::choose_next_base_point_default_method(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b;

	if (f_v) {
		cout << "action::choose_next_base_point_default_method" << endl;
		cout << "calling Group_element->find_non_fixed_point" << endl;
	}
	b = Group_element->find_non_fixed_point(Elt, verbose_level - 1);
	if (b == -1) {
		if (f_v) {
			cout << "action::choose_next_base_point_default_method "
					"cannot find another base point" << endl;
		}
		return -1;
	}
	if (f_v) {
		cout << "action::choose_next_base_point_default_method current base: ";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << " choosing next base point to be " << b << endl;
	}
	return b;
}

void action::generators_to_strong_generators(
	int f_target_go,
	ring_theory::longinteger_object &target_go,
	data_structures_groups::vector_ge *gens,
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::generators_to_strong_generators" << endl;
		if (f_target_go) {
			cout << "induced_action::generators_to_strong_generators "
					"trying to create a group of order " << target_go << endl;
		}
	}

	groups::sims *S;

	if (f_v) {
		cout << "action::generators_to_strong_generators "
				"before create_sims_from_generators_randomized" << endl;
	}

	S = create_sims_from_generators_randomized(
		gens, f_target_go,
		target_go, verbose_level - 2);

	if (f_v) {
		cout << "action::generators_to_strong_generators "
				"after create_sims_from_generators_randomized" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "action::generators_to_strong_generators "
				"before Strong_gens->init_from_sims" << endl;
	}
	Strong_gens->init_from_sims(S, verbose_level - 5);

	FREE_OBJECT(S);

	if (f_v) {
		cout << "action::generators_to_strong_generators done" << endl;
	}
}




}}}

