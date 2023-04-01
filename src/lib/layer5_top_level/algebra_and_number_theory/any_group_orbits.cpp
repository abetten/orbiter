/*
 * any_group_orbits.cpp
 *
 *  Created on: May 22, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


void any_group::orbits_on_subsets(
		poset_classification::poset_classification_control *Control,
		poset_classification::poset_classification *&PC,
		int subset_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_subsets subset_size=" << subset_size << endl;
	}
	poset_classification::poset_with_group_action *Poset;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);

	if (f_v) {
		cout << "any_group::orbits_on_subsets control=" << endl;
		Control->print();
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets label=" << label << endl;
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets A_base=" << endl;
		A_base->print_info();
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets A=" << endl;
		A->print_info();
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets group order" << endl;

		ring_theory::longinteger_object go;

		Subgroup_gens->group_order(go);

		cout << go << endl;
	}


	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"before Poset->init_subset_lattice" << endl;
	}
	Poset->init_subset_lattice(A_base, A,
			Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			subset_size,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"before orbits_on_poset_post_processing" << endl;
	}
	orbits_on_poset_post_processing(
			PC, subset_size,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"after orbits_on_poset_post_processing" << endl;
	}


	if (f_v) {
		cout << "any_group::orbits_on_subsets done" << endl;
	}
}


void any_group::orbits_on_poset_post_processing(
		poset_classification::poset_classification *PC,
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_poset_post_processing" << endl;
	}



#if 0
	if (Descr->f_test_if_geometric) {
		int d = Descr->test_if_geometric_depth;

		//for (depth = 0; depth <= orbits_on_subsets_size; depth++) {

		cout << "Orbits on subsets of size " << d << ":" << endl;
		PC->list_all_orbits_at_level(d,
				FALSE /* f_has_print_function */,
				NULL /* void (*print_function)(std::ostream &ost, int len, int *S, void *data)*/,
				NULL /* void *print_function_data*/,
				TRUE /* f_show_orbit_decomposition */,
				TRUE /* f_show_stab */,
				FALSE /* f_save_stab */,
				TRUE /* f_show_whole_orbit*/);
		int nb_orbits, orbit_idx;

		nb_orbits = PC->nb_orbits_at_level(d);
		for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

			int orbit_length;
			long int *Orbit;

			cout << "before PC->get_whole_orbit depth " << d
					<< " orbit " << orbit_idx
					<< " / " << nb_orbits << ":" << endl;
			PC->get_whole_orbit(
					d, orbit_idx,
					Orbit, orbit_length, verbose_level);
			cout << "depth " << d << " orbit " << orbit_idx
					<< " / " << nb_orbits << " has length "
					<< orbit_length << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit, orbit_length, d);

			action *Aut;
			longinteger_object ago;
			nauty_interface_with_group Nauty;

			Aut = Nauty.create_automorphism_group_of_block_system(
				A2->degree /* nb_points */,
				orbit_length /* nb_blocks */,
				depth /* block_size */, Orbit,
				verbose_level);
			Aut->group_order(ago);
			cout << "The automorphism group of the set system "
					"has order " << ago << endl;

			FREE_OBJECT(Aut);
			FREE_lint(Orbit);
		}
		if (nb_orbits == 2) {
			cout << "the number of orbits at depth " << depth
					<< " is two, we will try create_automorphism_"
					"group_of_collection_of_two_block_systems" << endl;
			long int *Orbit1;
			int orbit_length1;
			long int *Orbit2;
			int orbit_length2;

			cout << "before PC->get_whole_orbit depth " << d
					<< " orbit " << orbit_idx
					<< " / " << nb_orbits << ":" << endl;
			PC->get_whole_orbit(
					depth, 0 /* orbit_idx*/,
					Orbit1, orbit_length1, verbose_level);
			cout << "depth " << d << " orbit " << 0
					<< " / " << nb_orbits << " has length "
					<< orbit_length1 << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit1, orbit_length1, d);

			PC->get_whole_orbit(
					depth, 1 /* orbit_idx*/,
					Orbit2, orbit_length2, verbose_level);
			cout << "depth " << d << " orbit " << 1
					<< " / " << nb_orbits << " has length "
					<< orbit_length2 << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit2, orbit_length2, d);

			action *Aut;
			longinteger_object ago;
			nauty_interface_with_group Nauty;

			Aut = Nauty.create_automorphism_group_of_collection_of_two_block_systems(
				A2->degree /* nb_points */,
				orbit_length1 /* nb_blocks */,
				depth /* block_size */, Orbit1,
				orbit_length2 /* nb_blocks */,
				depth /* block_size */, Orbit2,
				verbose_level);
			Aut->group_order(ago);
			cout << "The automorphism group of the collection of two set systems "
					"has order " << ago << endl;

			FREE_OBJECT(Aut);
			FREE_lint(Orbit1);
			FREE_lint(Orbit2);

		} // if nb_orbits == 2
	} // if (f_test_if_geometric)
#endif



	if (f_v) {
		cout << "any_group::orbits_on_poset_post_processing done" << endl;
	}
}








#if 0
void any_group::do_conjugacy_class_of_element(
		std::string &elt_label, std::string &elt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element" << endl;
	}


	int *data, sz;

	Int_vec_scan(elt_text, data, sz);

	if (f_v) {
		cout << "computing conjugacy class of ";
		Int_vec_print(cout, data, sz);
		cout << endl;
	}


	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	A->make_element(Elt, data, 0 /* verbose_level */);

	if (!A->f_has_sims) {
		if (f_v) {
			cout << "any_group::do_conjugacy_class_of_element "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			groups::sims *S;

			S = LG->Strong_gens->create_sims(verbose_level);

			if (f_v) {
				cout << "any_group::do_conjugacy_class_of_element "
						"before init_sims" << endl;
			}
			A->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "any_group::do_conjugacy_class_of_element "
						"after init_sims" << endl;
			}
		}

	}
	groups::sims *S;

	S = A->Sims;

	long int the_set[1];
	int set_size = 1;

	the_set[0] = S->element_rank_lint(Elt);

	if (f_v) {
		cout << "computing conjugacy class of " << endl;
		A->element_print_latex(Elt, cout);
		cout << "which is the set ";
		Lint_vec_print(cout, the_set, set_size);
		cout << endl;
	}


	actions::action A_conj;
	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element "
				"before A_conj.induced_action_by_conjugation" << endl;
	}
	A_conj.induced_action_by_conjugation(S, S,
			FALSE /* f_ownership */, FALSE /* f_basis */,
			verbose_level);
	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element "
				"created action by conjugation" << endl;
	}



	//schreier Classes;
	//Classes.init(&A_conj, verbose_level - 2);
	//Classes.init_generators(*A1->Strong_gens->gens, verbose_level - 2);
	//cout << "Computing orbits:" << endl;
	//Classes.compute_all_point_orbits(1 /*verbose_level - 1*/);
	//cout << "found " << Classes.nb_orbits << " conjugacy classes" << endl;




	algebra_global_with_action Algebra;

	long int *Table;
	int orbit_length;

	Algebra.compute_orbit_of_set(
			the_set, set_size,
			A, &A_conj,
			LG->Strong_gens->gens,
			elt_label,
			LG->label,
			Table,
			orbit_length,
			verbose_level);


	// write as txt file:

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname.assign(elt_label);
	fname.append("_orbit_under_");
	fname.append(LG->label);
	fname.append("_elements_coded.csv");

	if (f_v) {
		cout << "Writing table to file " << fname << endl;
	}
	{
		ofstream ost(fname);
		int i;

		// header line:
		ost << "ROW";
		for (int j = 0; j < A->make_element_size; j++) {
			ost << ",C" << j;
		}
		ost << endl;

		for (i = 0; i < orbit_length; i++) {

			ost << i;
			S->element_unrank_lint(Table[i], Elt);

			for (int j = 0; j < A->make_element_size; j++) {
				ost << "," << Elt[j];
			}
			ost << endl;
		}
		ost << "END" << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	FREE_int(Elt);
	FREE_int(data);
	FREE_lint(Table);

	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element done" << endl;
	}
}
#endif

void any_group::do_orbits_on_group_elements_under_conjugation(
		std::string &fname_group_elements_coded,
		std::string &fname_transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation" << endl;
	}




	if (!A->f_has_sims) {
		if (f_v) {
			cout << "any_group::do_orbits_on_group_elements_under_conjugation "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			//sims *S;

			A->Known_groups->create_sims(verbose_level);

#if 0
			if (f_v) {
				cout << "any_group::do_orbits_on_group_elements_under_conjugation before init_sims" << endl;
			}
			A2->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "any_group::do_orbits_on_group_elements_under_conjugation after init_sims" << endl;
			}
#endif
		}

	}





	groups::sims *S;

	S = A->Sims;

	if (f_v) {
		cout << "the group has order " << S->group_order_lint() << endl;
	}
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "computing the element ranks:" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	long int *the_ranks;
	data_structures_groups::vector_ge *Transporter;
	int m, n;
	int i;

	{
		int *M;
		Fio.int_matrix_read_csv(fname_group_elements_coded,
				M, m, n, 0 /*verbose_level*/);
		if (f_v) {
			cout << "read a set of size " << m << endl;
		}
		the_ranks = NEW_lint(m);
		for (i = 0; i < m; i++) {

			if (FALSE) {
				cout << i << " : ";
				Int_vec_print(cout, M + i * n, n);
				cout << endl;
			}

			LG->A_linear->Group_element->make_element(Elt, M + i * n, 0 /* verbose_level */);
			if (FALSE) {
				cout << "computing rank of " << endl;
				LG->A_linear->Group_element->element_print_latex(Elt, cout);
			}

			the_ranks[i] = S->element_rank_lint(Elt);
			if (FALSE) {
				cout << i << " : " << the_ranks[i] << endl;
			}
		}

		FREE_int(M);
	}

	Transporter = NEW_OBJECT(data_structures_groups::vector_ge);
	Transporter->init(S->A, 0);
	{
		int *M;
		Fio.int_matrix_read_csv(fname_transporter,
				M, m, n, 0 /*verbose_level*/);
		if (f_v) {
			cout << "read a set of size " << m << endl;
		}
		Transporter->allocate(m, 0);
		for (i = 0; i < m; i++) {

			if (FALSE) {
				cout << i << " : ";
				Int_vec_print(cout, M + i * n, n);
				cout << endl;
			}

			LG->A_linear->Group_element->make_element(Transporter->ith(i), M + i * n, 0 /* verbose_level */);
			if (FALSE) {
				cout << "computing rank of " << endl;
				LG->A_linear->Group_element->element_print_latex(Elt, cout);
			}

		}

		FREE_int(M);
	}




	if (f_v) {
		cout << "computing conjugacy classes on the set " << endl;
		Lint_vec_print(cout, the_ranks, m);
		cout << endl;
	}

	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation "
				"before Algebra.orbits_under_conjugation" << endl;
	}
	Algebra.orbits_under_conjugation(
			the_ranks, m, S,
			LG->Strong_gens,
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation "
				"after Algebra.orbits_under_conjugation" << endl;
	}




	FREE_int(Elt);

	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation done" << endl;
	}
}


}}}

