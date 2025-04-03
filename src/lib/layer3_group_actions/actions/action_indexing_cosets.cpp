// action_indexing_cosets.cpp
//
// Anton Betten
// July 8, 2003
//
// moved here from action.cpp: June 6, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


void action::coset_unrank(
		groups::sims *G,
		groups::sims *U,
		long int rank, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, base_idx = 0, base_pt, rank0, nb, len, k, elt_k;
	int rem_int;
	int *Elt_gk, *Elt1, *Elt2;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object G0_order, G_order;
	algebra::ring_theory::longinteger_object U_order, index, rem, a, b, c, d, Uk_order;
	groups::schreier G_orb, U_orb;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "action::coset_unrank rank=" << rank << endl;
		cout << "in action:" << endl;
		print_info();
		cout << "verbose_level=" << verbose_level << endl;
	}
	Elt_gk = NEW_int(elt_size_in_int);
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	
	G->group_order(G_order);

	U->group_order(U_order);

	D.integral_division(
			G_order, U_order, index, rem, 0);

	if (f_v) {
		cout << "The full group has order " << G_order << endl;
		cout << "The subgroup has order " << U_order << endl;
		cout << "The index is " << index << endl;
	}
	
#if 0
	if (!test_if_lex_least_base(verbose_level)) {
		cout << "base is not lexleast" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "the base is lexleast" << endl;
	}
#endif



	if (f_v) {
		G->print_transversal_lengths();
		U->print_transversal_lengths();
	}

	G_orb.init(this, verbose_level - 2);
	//G_orb.initialize_tables(); // not needed, already done in init
	G_orb.Generators_and_images->init_generators(G->gens, verbose_level - 2);

		// G_orb is used to determine representatives of the double cosets
	
	U_orb.init(this, verbose_level - 2);
	//U_orb.initialize_tables(); // not needed, already done in init
	U_orb.Generators_and_images->init_generators(U->gens, verbose_level - 2);
	
	for (i = 0; i < base_len(); i++) {
		if (G->get_orbit_length(i) > 1 /*U->orbit_len[i]*/) {
			base_idx = i;
			break;
		}
	}
	if (i == base_len()) {
		if (f_v) {
			cout << "the groups are equal" << endl;
		}
		if (rank) {
			cout << "the groups are equal but rank is not zero" << endl;
			exit(1);
		}
#if 0
		G->element_unrank_int(rank, Elt);
		if (f_v) {
			cout << "the element with rank " << rank << " is:" << endl;
			element_print_quick(Elt, cout);
		}
#endif
		Group_element->element_one(Elt, 0);
		goto done;
	}
	base_pt = base_i(base_idx);
	if (f_v) {
		cout << "base_idx = " << base_idx << endl;
		cout << "base_pt = " << base_pt << endl;
		cout << "G->orbit_len[base_idx]=" << G->get_orbit_length(base_idx) << endl;
		cout << "U->orbit_len[base_idx]=" << U->get_orbit_length(base_idx) << endl;
	}

	D.integral_division_by_int(G_order, G->get_orbit_length(base_idx), G0_order, rem_int);

	if (f_v) {
		cout << "G0_order=" << G0_order << endl;
	}

	int *orbit;
	int orbit_len;
	
	orbit_len = G->get_orbit_length(base_idx);
	

	// orbit is the G-orbit of base_pt

	orbit = NEW_int(orbit_len);
	for (int t = 0; t < orbit_len; t++) {
		orbit[t] = G->get_orbit(base_idx, t);
	}
	//int_vec_copy(G->orbit[base_idx], orbit, orbit_len);
	Sorting.int_vec_heapsort(orbit, orbit_len);

	if (f_v) {
		cout << "orbit of length " << orbit_len << ":";
		Int_vec_print(cout, orbit, orbit_len);
		cout << endl;
	}

	int nb_U_orbits_on_subset;

	{
		int print_interval = 10000;

		G_orb.compute_point_orbit(base_pt, print_interval, 0 /* verbose_level - 2*/);
	}
	if (f_v) {
		cout << "orbit of base_pt under G has length " << G_orb.Forest->orbit_len[0] << endl;
	}

	if (G_orb.Forest->orbit_len[0] != orbit_len) {
		cout << "action::coset_unrank G_orb.Forest->orbit_len[0] != orbit_len" << endl;
		exit(1);
	}

	U_orb.orbits_on_invariant_subset_fast(orbit_len, orbit, verbose_level - 2);
	nb_U_orbits_on_subset = U_orb.Forest->nb_orbits;
	if (f_v) {
		cout << "U-orbits: ";
		U_orb.Forest->print_orbit_length_distribution(cout);
		cout << endl;
		cout << "in order:" << endl;
		U_orb.Forest->print_orbit_lengths(cout);
		cout << endl;
		U_orb.Forest->print_and_list_orbits(cout);
	}
	
	rank0 = 0;
	for (k = 0; k < nb_U_orbits_on_subset; k++) {
		len = U_orb.Forest->orbit_len[k];
		b.create(len);
		D.mult(G0_order, b, c);
		D.integral_division(c, U_order, d, rem, 0);
		if (!rem.is_zero()) {
			cout << "action::coset_unrank: remainder is not zero, "
					"something is wrong" << endl;
			exit(1);
		}
		nb = d.as_int();

			// nb = length of k-th U-orbit * |G0| / |U|

		elt_k = U_orb.Forest->orbit[U_orb.Forest->orbit_first[k]];
		if (f_v) {
			cout << "double coset k=" << k << " elt_k=" << elt_k
					<< " nb=" << nb << endl;
		}
		if (rank0 + nb > rank) {
			if (f_v) {
				cout << "we are in double coset " << k << endl;
				cout << "reduced rank is " << rank - rank0 << endl;
			}


			if (f_v) {
				G_orb.Forest->print_and_list_orbits(cout);
			}
			//G->coset_rep(base_idx, G->orbit_inv[base_idx][elt_k], 0/* verbose_level*/);
			G_orb.Generators_and_images->coset_rep(G_orb.Forest->orbit_inv[elt_k], 0 /* verbose_level */);
			Group_element->element_move(G_orb.Generators_and_images->cosetrep, Elt_gk, 0);

			if (f_v) {
				cout << "gk (before)=" << endl;
				Group_element->element_print_quick(Elt_gk, cout);
				Group_element->element_print_as_permutation(Elt_gk, cout);
			}

			Group_element->minimize_base_images(base_idx + 1, G, Elt_gk, verbose_level);
			if (f_v) {
				cout << "gk (after)=" << endl;
				Group_element->element_print_quick(Elt_gk, cout);
				Group_element->element_print_as_permutation(Elt_gk, cout);
			}

			if (Group_element->element_image_of(base_pt, Elt_gk, 0) != elt_k) {
				cout << "image of base point under gk is not as expected!" << endl;
				cout << "base_pt=" << base_pt << endl;
				cout << "elt_k=" << elt_k << endl;
				cout << "image=" << Group_element->element_image_of(base_pt, Elt_gk, 0) << endl;
				exit(1);
			}
			groups::sims *Gk = NULL;
			groups::sims *Uk = NULL;
			int print_interval = 10000;

			G_orb.Forest->initialize_tables(verbose_level - 2);
			G_orb.Generators_and_images->init_generators(G->gens, verbose_level - 2);
				// this is redundant as the generators for G are already in G_orb
				// in fact, it might be a memory leak
	
			G_orb.compute_point_orbit(elt_k, print_interval, 0 /* verbose_level - 2*/);
				// we recompute the orbit, but this time with elt_k as 
				// orbit representative, so that we can compute 
				// the stabilizer of elt_k in G

			if (f_v) {
				cout << "orbit of elt_k under G has length " << G_orb.Forest->orbit_len[0] << endl;
			}
			G_orb.point_stabilizer(this, G_order, Gk, 0, 0/*verbose_level - 2*/);
			
			//D.integral_division_by_int(U_order, len, Uk_order, rem_int);
			//cout << "expecting stabilizer of " << k << "-th point in U to have order " << Uk_order << endl;
			U_orb.point_stabilizer(this, U_order, Uk, k, 0/*verbose_level - 2*/);

			if (f_v) {
				cout << "Gk transversal lengths:" << endl;
				Gk->print_transversal_lengths();
				cout << "Uk transversal lengths:" << endl;
				Uk->print_transversal_lengths();
			}

			if (f_v) {
				cout << "recursing" << endl;
			}
			coset_unrank(Gk, Uk, rank - rank0, Elt1, verbose_level);
			if (f_v) {
				cout << "recursion done" << endl;
				cout << "Elt1=" << endl;
				Group_element->element_print_quick(Elt1, cout);

				cout << "Elt_gk=" << endl;
				Group_element->element_print_quick(Elt_gk, cout);



			}

			Group_element->element_mult(Elt_gk, Elt1, Elt2, 0);
			if (f_v) {
				cout << "Elt_gk * Elt1=" << endl;
				Group_element->element_print_quick(Elt2, cout);
			}
			Group_element->element_move(Elt2, Elt, 0);
			
			delete Gk;
			delete Uk; 
			goto done2;
		}
		rank0 += nb;
	}

done2:
	FREE_int(orbit);

done:

	FREE_int(Elt_gk);
	FREE_int(Elt1);
	FREE_int(Elt2);
	
}

long int action::coset_rank(
		groups::sims *G, groups::sims *U, int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rank = 0;
	long int i, base_idx = 0, base_pt, rank1, nb, len, k, kk, elt_k, im;
	int rem_int;
	int *Elt_gk, *Elt1, *Elt2, *Elt3, *Elt_u;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object G0_order, G_order, U_order, index, rem, a, b, c, d, Uk_order;
	groups::schreier G_orb, U_orb;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "##################################" << endl;
		cout << "action::coset_rank element" << endl;
		Group_element->element_print_quick(Elt, cout);
		Group_element->element_print_base_images(Elt, cout);
		cout << endl;
		cout << "in action:" << endl;
		print_info();
	}
	Elt_gk = NEW_int(elt_size_in_int);
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt_u = NEW_int(elt_size_in_int);
	
	G->group_order(G_order);

	U->group_order(U_order);

	D.integral_division(G_order, U_order, index, rem, 0);

	if (f_v) {
		cout << "The full group has order " << G_order << endl;
		cout << "The subgroup has order " << U_order << endl;
		cout << "The index is " << index << endl;
	}
	
#if 0
	if (!test_if_lex_least_base(0/*verbose_level*/)) {
		cout << "base is not lexleast" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "the base is lexleast" << endl;
	}

	if (f_v) {
		G->print_transversal_lengths();
		U->print_transversal_lengths();
	}

	G_orb.init(this, verbose_level - 2);
	//G_orb.initialize_tables();
	G_orb.Generators_and_images->init_generators(G->gens, verbose_level - 2);
	
	U_orb.init(this, verbose_level - 2);
	//U_orb.initialize_tables();
	U_orb.Generators_and_images->init_generators(U->gens, verbose_level - 2);
	
	for (i = 0; i < base_len(); i++) {
		if (G->get_orbit_length(i) > 1) {
			base_idx = i;
			break;
		}
	}
	if (i == base_len()) {
		if (f_v) {
			cout << "the groups are equal" << endl;
		}
#if 0
		G->element_unrank_int(rank, Elt);
		if (f_v) {
			cout << "the element with rank " << rank << " is:" << endl;
			element_print_quick(Elt, cout);
		}
#endif
		//element_one(Elt, 0);
		goto done;
	}
	base_pt = base_i(base_idx);
	if (f_v) {
		cout << "base_idx = " << base_idx << endl;
		cout << "base_pt = " << base_pt << endl;
		cout << "G->orbit_len[base_idx]=" << G->get_orbit_length(base_idx) << endl;
		cout << "U->orbit_len[base_idx]=" << U->get_orbit_length(base_idx) << endl;
	}

	D.integral_division_by_int(G_order, G->get_orbit_length(base_idx), G0_order, rem_int);

	if (f_v) {
		cout << "G0_order=" << G0_order << endl;
	}

	int *orbit;
	int orbit_len;
	
	orbit_len = G->get_orbit_length(base_idx);
	

	orbit = NEW_int(orbit_len);
	for (int t = 0; t < orbit_len; t++) {
		orbit[t] = G->get_orbit(base_idx, t);
	}
	//int_vec_copy(G->orbit[base_idx], orbit, orbit_len);
	Sorting.int_vec_heapsort(orbit, orbit_len);

	if (f_v) {
		cout << "G-orbit of length " << orbit_len << ":";
		Int_vec_print(cout, orbit, orbit_len);
		cout << endl;
	}

	//int nb_U_orbits_on_subset;

	{
		int print_interval = 10000;

		G_orb.compute_point_orbit(base_pt, print_interval, 0 /* verbose_level - 2*/);
	}
	if (f_v) {
		cout << "orbit of base_pt under G has length " << G_orb.Forest->orbit_len[0] << endl;
		cout << "G-orbits: ";
		G_orb.Forest->print_and_list_orbits(cout);
	}

	U_orb.orbits_on_invariant_subset_fast(orbit_len, orbit, verbose_level - 2);
	//nb_U_orbits_on_subset = U_orb.nb_orbits;
	if (f_v) {
		cout << "U-orbits: ";
		U_orb.Forest->print_orbit_length_distribution(cout);
		cout << endl;
		cout << "in order:" << endl;
		U_orb.Forest->print_orbit_lengths(cout);
		cout << endl;
		U_orb.Forest->print_and_list_orbits(cout);
	}
	
	Group_element->element_move(Elt, Elt1, 0);
	im = Group_element->element_image_of(base_pt, Elt1, 0);
	if (f_v) {
		cout << "image of base point " << base_pt << " is " << im << endl;
	}
	k = U_orb.Forest->orbit_number(im); //U_orb.orbit_no[U_orb.orbit_inv[im]];
	if (f_v) {
		cout << "Which lies in orbit " << k << endl;
	}
	for (kk = 0; kk < k; kk++) {
		len = U_orb.Forest->orbit_len[kk];
		b.create(len);
		D.mult(G0_order, b, c);
		D.integral_division(c, U_order, d, rem, 0);
		if (!rem.is_zero()) {
			cout << "action::coset_rank: remainder is not zero, something is wrong" << endl;
			exit(1);
		}
		nb = d.as_int();
		rank += nb;
	}
	if (f_v) {
		cout << "after going through the previous double cosets, rank=" << rank << endl;
	}
	len = U_orb.Forest->orbit_len[k];
	b.create(len);
	D.mult(G0_order, b, c);
	D.integral_division(c, U_order, d, rem, 0);
	if (!rem.is_zero()) {
		cout << "action::coset_rank: remainder is not zero, something is wrong" << endl;
		exit(1);
	}
	nb = d.as_int();
	elt_k = U_orb.Forest->orbit[U_orb.Forest->orbit_first[k]];
	if (f_v) {
		cout << "elt_k=" << elt_k << endl;
	}


	G_orb.Generators_and_images->coset_rep(
			G_orb.Forest->orbit_inv[elt_k], 0 /* verbose_level */);
	Group_element->element_move(
			G_orb.Generators_and_images->cosetrep, Elt_gk, 0);

	if (Group_element->element_image_of(base_pt, Elt_gk, 0) != elt_k) {
		cout << "image of base point under gk is not as expected!" << endl;
		cout << "base_pt=" << base_pt << endl;
		cout << "elt_k=" << elt_k << endl;
		cout << "image=" << Group_element->element_image_of(base_pt, Elt_gk, 0) << endl;
		cout << "gk (before minimizing base images)=" << endl;
		Group_element->element_print_quick(Elt_gk, cout);
		Group_element->element_print_base_images(Elt_gk, cout);
		cout << endl;
		Group_element->element_print_as_permutation(Elt_gk, cout);
		exit(1);
	}
	if (f_v) {
		cout << "gk (before minimizing base images)=" << endl;
		Group_element->element_print_quick(Elt_gk, cout);
		//element_print_base_images(Elt_gk, cout);
		//cout << endl;
		Group_element->element_print_as_permutation(Elt_gk, cout);
	}

	Group_element->minimize_base_images(base_idx + 1, G, Elt_gk, 0/*verbose_level*/);
	if (f_v) {
		cout << "gk (after minimizing base images)=" << endl;
		Group_element->element_print_quick(Elt_gk, cout);
		//element_print_base_images(Elt_gk, cout);
		//cout << endl;
		Group_element->element_print_as_permutation(Elt_gk, cout);
	}

	if (Group_element->element_image_of(base_pt, Elt_gk, 0) != elt_k) {
		cout << "image of base point under gk is not as expected!" << endl;
		cout << "base_pt=" << base_pt << endl;
		cout << "elt_k=" << elt_k << endl;
		cout << "image=" << Group_element->element_image_of(base_pt, Elt_gk, 0) << endl;
		cout << "gk (after minimizing base images)=" << endl;
		Group_element->element_print_quick(Elt_gk, cout);
		Group_element->element_print_as_permutation(Elt_gk, cout);
		exit(1);
	}
	{
		groups::sims *Gk = NULL;
		groups::sims *Uk = NULL;

		int print_interval = 10000;

		G_orb.Forest->initialize_tables(verbose_level - 2);
		G_orb.Generators_and_images->init_generators(G->gens, verbose_level - 2);
		G_orb.compute_point_orbit(elt_k, print_interval,
				0 /* verbose_level - 2*/);

		if (f_v) {
			cout << "orbit of elt_k under G has length " << G_orb.Forest->orbit_len[0] << endl;
		}
		G_orb.point_stabilizer(this, G_order, Gk, 0, 0/*verbose_level - 2*/);

		//D.integral_division_by_int(U_order, len, Uk_order, rem_int);
		//cout << "expecting stabilizer of " << k << "-th point in U to have order " << Uk_order << endl;
		U_orb.point_stabilizer(this, U_order, Uk, k, 0/*verbose_level - 2*/);

		if (f_v) {
			cout << "Gk transversal lengths:" << endl;
			Gk->print_transversal_lengths();
			cout << "Uk transversal lengths:" << endl;
			Uk->print_transversal_lengths();
		}
	
		if (f_v) {
			cout << "Elt_gk=" << endl;
			Group_element->element_print_quick(Elt_gk, cout);
		}
		Group_element->element_invert(Elt_gk, Elt3, 0);
		if (f_v) {
			cout << "we are now going to divide off Elt_gk from the left." << endl;
			cout << "Elt_gk^-1=" << endl;
			Group_element->element_print_quick(Elt3, cout);
		}
		
		Group_element->element_mult(Elt3, Elt1, Elt2, 0);
		if (f_v) {
			cout << "Elt_gk^-1 * Elt =" << endl;
			Group_element->element_print_quick(Elt2, cout);
			//element_print_base_images(Elt2, cout);
			//cout << endl;
		}


		int im;

		im = Group_element->element_image_of(elt_k, Elt2, 0);
		if (im != elt_k) {
			if (f_v) {
				cout << "image of elt_k = " << elt_k << " is " << im << endl;
				cout << "we are now dividing off an element of U "
						"from the right so that elt_k is fixed" << endl;
			}

			U_orb.Generators_and_images->coset_rep_inv(
					U_orb.Forest->orbit_inv[im], 0 /* verbose_level */);
			Group_element->element_move(
					U_orb.Generators_and_images->cosetrep, Elt_u, 0);
			if (f_v) {
				cout << "Elt_u =" << endl;
				Group_element->element_print_quick(Elt_u, cout);
				cout << "moves " << im << " to " << elt_k << endl;
			}
			if (Group_element->element_image_of(im, Elt_u, 0) != elt_k) {
				cout << "image of " << im << " is "
						<< Group_element->element_image_of(im, Elt_u, 0)
						<< " but not " << elt_k << " fatal" << endl;
				exit(1);
			}
			Group_element->element_mult(Elt2, Elt_u, Elt3, 0);
			Group_element->element_move(Elt3, Elt2, 0);
			if (f_v) {
				cout << "after multiplying Elt_u:" << endl;
				Group_element->element_print_quick(Elt2, cout);
			}
		}

		if (f_v) {
			cout << "recursing" << endl;
		}
	
		rank1 = coset_rank(Gk, Uk, Elt2, verbose_level);
		if (f_v) {
			cout << "recursion done, rank1=" << rank1 << endl;
		}
		rank += rank1;
		if (f_v) {
			cout << "rank=" << rank << endl;
		}

		delete Gk;
		delete Uk;
	}

	FREE_int(orbit);

done:

	FREE_int(Elt_gk);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt_u);
	return rank;	
}

}}}


