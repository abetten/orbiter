/*
 * sims_group_theory.cpp
 *
 *  Created on: Aug 24, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


void sims::random_element(
		int *elt, int verbose_level)
// compute a random element among the group
// elements represented by the chain
// (chooses random cosets along the stabilizer chain)
{
	int f_v = (verbose_level >= 1);
	int i;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "sims::random_element" << endl;
		cout << "sims::random_element orbit_len=";
		Int_vec_print(cout, orbit_len, A->base_len());
		cout << endl;
		//cout << "transversals:" << endl;
		//print_transversals();
		}
	for (i = 0; i < A->base_len(); i++) {
		path[i] = Os.random_integer(orbit_len[i]);
		}
	if (f_v) {
		cout << "sims::random_element" << endl;
		cout << "path=";
		Int_vec_print(cout, path, A->base_len());
		cout << endl;
		}
	element_from_path(elt, verbose_level /*- 1 */);
	if (f_v) {
		cout << "sims::random_element done" << endl;
		}
}

void sims::random_element_of_order(
		int *elt,
		int order, int verbose_level)
{
	int f_v = (verbose_level >=1);
	int f_vv = (verbose_level >=2);
	int o, n, cnt;

	if (f_v) {
		cout << "sims::random_element_of_order " << order << endl;
		}
	cnt = 0;
	while (true) {
		cnt++;
		random_element(elt, verbose_level - 1);
		o = A->Group_element->element_order(elt);
		if ((o % order) == 0) {
			break;
			}
		}
	if (f_v) {
		cout << "sims::random_element_of_order " << o
				<< " found with " << cnt << " trials" << endl;
		}
	if (f_vv) {
		A->Group_element->element_print_quick(elt, cout);
		}
	n = o / order;
	if (f_v) {
		cout << "sims::random_element_of_order we will raise to the "
				<< n << "-th power" << endl;
		}
	A->Group_element->element_power_int_in_place(elt, n, 0);
	if (f_vv) {
		A->Group_element->element_print_quick(elt, cout);
		}
}

void sims::random_elements_of_order(
		data_structures_groups::vector_ge *elts,
		int *orders, int nb, int verbose_level)
{
	int i;
	int f_v = (verbose_level >=1);

	if (f_v) {
		cout << "sims::random_elements_of_order" << endl;
		}

	elts->init(A, verbose_level - 2);
	elts->allocate(nb, verbose_level - 2);
	for (i = 0; i < nb; i++) {
		random_element_of_order(elts->ith(i),
				orders[i], verbose_level);
		}
	if (f_v) {
		cout << "sims::random_elements_of_order done" << endl;
		}
}

void sims::transitive_extension(
		schreier &O,
		data_structures_groups::vector_ge &SG,
		int *tl, int verbose_level)
{
	transitive_extension_tolerant(O, SG, tl, false, verbose_level);
}

int sims::transitive_extension_tolerant(
		schreier &O,
		data_structures_groups::vector_ge &SG,
		int *tl,
	int f_tolerant, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go, ol, ego, cur_ego, rgo, rem;
	int orbit_len, j;
	ring_theory::longinteger_domain D;
	orbiter_kernel_system::os_interface Os;

	orbit_len = O.orbit_len[0];
	if (f_v) {
		cout << "sims::transitive_extension_tolerant "
				"computing transitive extension" << endl;
		cout << "f_tolerant=" << f_tolerant << endl;
		}
	group_order(go);
	ol.create(orbit_len, __FILE__, __LINE__);
	D.mult(go, ol, ego);
	if (f_v) {
		cout << "sims::transitive_extension_tolerant "
				"group order " << go << ", orbit length "
				<< orbit_len << ", current group order " << ego << endl;
		}
	group_order(cur_ego);

	//if (f_vv) {
		//print(0);
		//}
	while (D.compare_unsigned(cur_ego, ego) != 0) {

		// we do not enter the while loop if orbit_len is 1,
		// hence the following makes sense:
		// we want non trivial generators, hence we want j non zero.
		if (D.compare_unsigned(cur_ego, ego) > 0) {
			cout << "sims::transitive_extension_tolerant fatal: "
					"group order overshoots target" << endl;
			cout << "current group order = " << cur_ego << endl;
			cout << "target group order = " << ego << endl;
			if (f_tolerant) {
				cout << "we are tolerant, so we return false" << endl;
				return false;
				}
			cout << "we are not tolerant, so we exit" << endl;
			exit(1);
			}

		while (true) {
			j = Os.random_integer(orbit_len);
			if (j)
				break;
			}

		O.coset_rep(j, 0 /* verbose_level */);

		random_element(Elt2, verbose_level - 1);

		A->Group_element->element_mult(O.cosetrep, Elt2, Elt3, false);

		if (f_vv) {
			cout << "sims::transitive_extension_tolerant "
					"choosing random coset " << j << ", random element ";
			Int_vec_print(cout, path, A->base_len());
			cout << endl;
			//A->element_print(Elt3, cout);
			//cout << endl;
			}

		if (!strip_and_add(Elt3, Elt1 /* residue */, 0/*verbose_level - 1*/)) {
			continue;
			}


		group_order(cur_ego);
		if (f_v) {
			cout << "sims::transitive_extension_tolerant "
					"found an extension of order " << cur_ego
					<< " of " << ego
				<< " with " << gens.len
				<< " strong generators" << endl;
			D.integral_division(ego, cur_ego, rgo, rem, 0);
			cout << "remaining factor: " << rgo
					<< " remainder " << rem << endl;
			}


		}
	if (f_v) {
		cout << "sims::transitive_extension_tolerant "
				"extracting strong generators" << endl;
		}
	extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	return true;
}

void sims::transitive_extension_using_coset_representatives_extract_generators(
	int *coset_reps, int nb_cosets,
	data_structures_groups::vector_ge &SG,
	int *tl,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::transitive_extension_using_coset_"
				"representatives_extract_generators" << endl;
		}
	transitive_extension_using_coset_representatives(
		coset_reps, nb_cosets,
		verbose_level);
	extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_"
				"representatives_extract_generators done" << endl;
		}
}


void sims::transitive_extension_using_coset_representatives(
	int *coset_reps, int nb_cosets,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go, ol, ego, cur_ego, rgo, rem;
	int orbit_len, j;
	ring_theory::longinteger_domain D;
	orbiter_kernel_system::os_interface Os;

	orbit_len = nb_cosets;
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_"
				"representatives computing transitive extension" << endl;
		}
	group_order(go);
	ol.create(orbit_len, __FILE__, __LINE__);
	D.mult(go, ol, ego);
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_"
				"representatives group order " << go
				<< ", orbit length " << orbit_len
				<< ", current group order " << ego << endl;
		}
	group_order(cur_ego);

	//if (f_vv) {
		//print(0);
		//}
	while (D.compare_unsigned(cur_ego, ego) != 0) {

		// we do not enter the while loop if orbit_len is 1,
		// hence the following makes sense:
		// we want non trivial generators, hence we want j non zero.
		if (D.compare_unsigned(cur_ego, ego) > 0) {
			cout << "sims::transitive_extension_using_coset_"
					"representatives fatal: group order "
					"overshoots target" << endl;
			cout << "current group order = " << cur_ego << endl;
			cout << "target group order = " << ego << endl;
			cout << "we are not tolerant, so we exit" << endl;
			exit(1);
			}

		while (true) {
			j = Os.random_integer(orbit_len);
			if (j) {
				break;
				}
			}

		random_element(Elt2, verbose_level - 1);

		A->Group_element->element_mult(coset_reps + j * A->elt_size_in_int,
				Elt2, Elt3, 0);

		if (f_vv) {
			cout << "sims::transitive_extension_using_coset_"
					"representatives choosing random coset "
					<< j << ", random element ";
			Int_vec_print(cout, path, A->base_len());
			cout << endl;
			//A->element_print(Elt3, cout);
			//cout << endl;
			}

		if (!strip_and_add(Elt3, Elt1 /* residue */,
				0/*verbose_level - 1*/)) {
			continue;
			}


		group_order(cur_ego);
		if (f_v) {
			cout << "sims::transitive_extension_using_coset_"
					"representatives found an extension of order "
					<< cur_ego << " of " << ego
				<< " with " << gens.len << " strong generators" << endl;
			D.integral_division(ego, cur_ego, rgo, rem, 0);
			cout << "remaining factor: " << rgo
					<< " remainder " << rem << endl;
			}


		}
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_"
				"representatives done" << endl;
		}
}

void sims::transitive_extension_using_generators(
	int *Elt_gens, int nb_gens, int subgroup_index,
	data_structures_groups::vector_ge &SG,
	int *tl,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go, ol, ego, cur_ego, rgo, rem;
	int j;
	ring_theory::longinteger_domain D;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "sims::transitive_extension_using_generators "
				"computing transitive extension" << endl;
		}
	group_order(go);
	ol.create(subgroup_index, __FILE__, __LINE__);
	D.mult(go, ol, ego);
	if (f_v) {
		cout << "sims::transitive_extension_using_generators "
				"group order " << go << ", subgroup_index "
				<< subgroup_index << ", current group order " << ego << endl;
		}
	group_order(cur_ego);

	//if (f_vv) {
		//print(0);
		//}
	while (D.compare_unsigned(cur_ego, ego) != 0) {

		// we do not enter the while loop if orbit_len is 1,
		// hence the following makes sense:
		// we want non trivial generators, hence we want j non zero.
		if (D.compare_unsigned(cur_ego, ego) > 0) {
			cout << "sims::transitive_extension_using_generators "
					"fatal: group order overshoots target" << endl;
			cout << "current group order = " << cur_ego << endl;
			cout << "target group order = " << ego << endl;
			cout << "we are not tolerant, so we exit" << endl;
			exit(1);
			}

		j = Os.random_integer(nb_gens);

		random_element(Elt2, verbose_level - 1);

		A->Group_element->element_mult(Elt_gens + j * A->elt_size_in_int, Elt2, Elt3, 0);

		if (f_vv) {
			cout << "sims::transitive_extension_using_generators "
					"choosing random coset " << j << ", random element ";
			Int_vec_print(cout, path, A->base_len());
			cout << endl;
			//A->element_print(Elt3, cout);
			//cout << endl;
			}

		if (!strip_and_add(Elt3, Elt1 /* residue */, verbose_level - 1)) {
			continue;
			}


		group_order(cur_ego);
		if (f_v) {
			cout << "sims::transitive_extension_using_generators "
					"found an extension of order " << cur_ego
					<< " of " << ego
				<< " with " << gens.len << " strong generators" << endl;
			D.integral_division(ego, cur_ego, rgo, rem, 0);
			cout << "remaining factor: " << rgo
					<< " remainder " << rem << endl;
			}


		}
	if (f_v) {
		cout << "sims::transitive_extension_using_generators "
				"extracting strong generators" << endl;
		}
	extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	//return true;
}


void sims::point_stabilizer_stabchain_with_action(
		actions::action *A2,
		sims &S, int pt, int verbose_level)
// first computes the orbit of the point pt in action A2
// under the generators
// that are stored at present (using a temporary schreier object),
// then sifts random schreier generators into S
{
	schreier O;
	ring_theory::longinteger_object go, stab_order, cur_stab_order, rgo, rem;
	int orbit_len, r, cnt = 0, image; // d
	ring_theory::longinteger_domain D;
	int *Elt;

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"computing stabilizer of point "
			<< pt << " in action " << A2->label
			<< " verbose_level=" << verbose_level << endl;
		cout << "internal action: " << A->label << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action group order = " << go << endl;
	}

	O.init(A2, verbose_level - 2);

	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action before O.init_generators" << endl;
	}
	O.init_generators(gens, verbose_level - 2);
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action after O.init_generators" << endl;
	}

	if (f_vvv && A2->degree < 150) {
		O.print_generators();
		O.print_generators_with_permutations();
		int j;
		for (j = 0; j < O.gens.len; j++) {
			cout << "generator " << j << ":" << endl;
			//A->element_print(gens.ith(j), cout);
			//A->element_print_quick(gens.ith(j), cout);
			A->Group_element->element_print_as_permutation(O.gens.ith(j), cout);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"computing point orbit" << endl;
	}
	O.compute_point_orbit(pt, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"computing point orbit done" << endl;
	}


	orbit_len = O.orbit_len[0];
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"found orbit of length " << orbit_len << endl;
	}

	if (f_vvv && A2->degree < 150) {
		O.print(cout);
	}
	D.integral_division_by_int(go, orbit_len, stab_order, r);
	if (r != 0) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"orbit_len does not divide group order" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"group_order = " << go << " orbit_len = "
				<< orbit_len << " target stab_order = "
				<< stab_order << endl;
	}
	if (stab_order.is_one()) {
		if (f_v) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"stabilizer is trivial, finished" << endl;
		}
		S.init(A, verbose_level - 2);
		S.init_trivial_group(verbose_level - 1);
#if 0
		for (i = 0; i < A->base_len; i++) {
			tl[i] = 1;
		}
#endif
		return;
	}


	data_structures_groups::vector_ge stab_gens;

	stab_gens.init(A, verbose_level);

	//stab_gens.append(O.schreier_gen);

	//sims S;
	//int drop_out_level, image;
	//int *p_schreier_gen;

	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"before S.init" << endl;
	}
	S.init(A, verbose_level - 2);
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"before S.init_generators" << endl;
	}
	S.init_generators(stab_gens, verbose_level - 2);
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"after S.init_generators" << endl;
	}
	S.compute_base_orbits(verbose_level - 1);
	if (false) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"generators:" << endl;
		S.gens.print(cout);
	}

	S.group_order(cur_stab_order);
	if (f_vv) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"before the loop, stabilizer has order "
				<< cur_stab_order << " of " << stab_order << endl;
		cout << "sims::point_stabilizer_stabchain_with_action "
				"creating the stabilizer using random generators" << endl;
	}

	while (D.compare_unsigned(cur_stab_order, stab_order) != 0) {


		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"loop iteration " << cnt
					<< " cur_stab_order=" << cur_stab_order
					<< " stab_order=" << stab_order << endl;
		}

		if (cnt % 2 || nb_gen[0] == 0) {
			if (f_vv) {
				cout << "sims::point_stabilizer_stabchain_with_action "
						"creating random generator no " << cnt + 1
						<< " using the Schreier vector" << endl;
			}
			//O.non_trivial_random_schreier_generator(A2, Elt, verbose_level - 1);
			// A Betten 9/1/2019
			// this may get stuck in a forever loop, therefore we do this:
			O.random_schreier_generator(Elt, verbose_level - 1);
			//p_schreier_gen = O.schreier_gen;
		}
		else {
			if (f_vv) {
				cout << "sims::point_stabilizer_stabchain_with_action "
						"creating random generator no " << cnt + 1
						<< " using the Sims chain" << endl;
			}
			S.random_schreier_generator(Elt, verbose_level - 1);
			//p_schreier_gen = Elt; //S.schreier_gen;
		}
		cnt++;
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"random generator no " << cnt << endl;
			A->Group_element->element_print_quick(Elt, cout);
			cout << endl;
			cout << "sims::point_stabilizer_stabchain_with_action "
					"random generator no " << cnt
					<< " as permutation in natural action:" << endl;
			A->Group_element->element_print_as_permutation(Elt, cout);
			cout << endl;
			cout << "sims::point_stabilizer_stabchain_with_action "
					"random generator no " << cnt
					<< " as permutation in chosen action:" << endl;
			A2->Group_element->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
		image = A2->Group_element->element_image_of(pt, Elt,
				0 /* verbose_level */);
		if (image != pt) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"image is not equal to pt" << endl;
			cout << "pt=" << pt << endl;
			cout << "image=" << image << endl;
			exit(1);
		}
		if (f_vvv) {
			A->Group_element->element_print_quick(Elt, cout);
			if (A2->degree < 150) {
				A2->Group_element->element_print_as_permutation(Elt, cout);
				cout << endl;
			}
		}

		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"random generator no " << cnt
					<< " before strip_and_add" << endl;
		}
		if (!S.strip_and_add(Elt,
				Elt1 /* residue */, verbose_level - 3)) {
			if (f_vvv) {
				cout << "sims::point_stabilizer_stabchain_with_action "
						"strip_and_add returns false" << endl;
			}
			//continue;
		}
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"random generator no " << cnt
					<< " before strip_and_add" << endl;
		}

		S.group_order(cur_stab_order);
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"group order " << go << endl;
			cout << "orbit length " << orbit_len << endl;
			cout << "current stab_order = " << cur_stab_order
				<< " / " << stab_order
				<< " with " << S.gens.len
				<< " strong generators" << endl;
		}

		int cmp;

		cmp = D.compare_unsigned(cur_stab_order, stab_order);
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"compare yields " << cmp << endl;
		}
		if (cmp > 0) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"overshooting the target group order" << endl;
			cout << "current stab_order = " << cur_stab_order
					<< " / " << stab_order << endl;
			exit(1);
		}
		D.integral_division(stab_order, cur_stab_order, rgo, rem, 0);
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"remaining factor: " << rgo
					<< " remainder " << rem << endl;
		}

		if (D.compare_unsigned(cur_stab_order, stab_order) == 1) {
			cout << "sims::point_stabilizer_stabchain_with_action "
					"group order " << go << endl;
			cout << "orbit length " << orbit_len << endl;
			cout << "current stab_order = " << cur_stab_order
				<< " / " << stab_order
				<< " with " << S.gens.len
				<< " strong generators" << endl;
			D.integral_division(stab_order,
					cur_stab_order, rgo, rem, 0);
			cout << "remaining factor: " << rgo
					<< " remainder " << rem << endl;
			cout << "the current stabilizer is:" << endl;
			S.print_transversals();
			cout << "sims::point_stabilizer_stabchain_with_action "
					"computing stabilizer of point " << pt
					<< " in action " << A2->label
					<< " verbose_level=" << verbose_level << endl;
			cout << "internal action: " << A->label << endl;
			cout << "The orbit of point " << pt << " is:" << endl;
			O.print_and_list_orbits(cout);
			//O.print_tables(cout, true /* f_with_cosetrep */);
			cout << "sims::point_stabilizer_stabchain_with_action "
					"cur_stab_order > stab_order, error" << endl;
			exit(1);
		}

	}
	FREE_int(Elt);
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action "
				"found a stabilizer of order " << cur_stab_order
				<< " of " << stab_order
			<< " with " << S.gens.len
			<< " strong generators" << endl;
	}
}

void sims::point_stabilizer(
		data_structures_groups::vector_ge &SG,
		int *tl, int pt, int verbose_level)
// computes strong generating set for the stabilizer of point pt
{
	int f_v = (verbose_level >= 1);
	sims S;

	if (f_v) {
		cout << "sims::point_stabilizer" << endl;
	}
	point_stabilizer_stabchain_with_action(A,
			S, pt, verbose_level);
	S.extract_strong_generators_in_order(SG, tl,
			verbose_level - 2);
	if (f_v) {
		cout << "sims::point_stabilizer done" << endl;
	}
}

void sims::point_stabilizer_with_action(
		actions::action *A2,
		data_structures_groups::vector_ge &SG,
		int *tl, int pt,
		int verbose_level)
// computes strong generating set for
// the stabilizer of point pt in action A2
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	sims S;

	if (f_v) {
		cout << "sims::point_stabilizer_with_action "
				"pt=" << pt << endl;
		cout << "sims::point_stabilizer_with_action "
				"action = " << A2->label << endl;
		cout << "sims::point_stabilizer_with_action "
				"internal action = " << A->label << endl;
	}
	if (f_v) {
		cout << "sims::point_stabilizer_with_action "
				"before point_stabilizer_stabchain_with_action" << endl;
	}
	point_stabilizer_stabchain_with_action(A2, S, pt, verbose_level);
	if (f_v) {
		cout << "sims::point_stabilizer_with_action "
				"after point_stabilizer_stabchain_with_action" << endl;
	}
	if (f_v) {
		cout << "sims::point_stabilizer_with_action "
				"before extract_strong_generators_in_order" << endl;
	}
	S.extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	if (f_v) {
		cout << "sims::point_stabilizer_with_action done" << endl;
	}
}

void sims::conjugate(
		actions::action *A,
	sims *old_G, int *Elt,
	int f_overshooting_OK,
	int verbose_level)
// Elt * g * Elt^{-1} where g is in old_G
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 4);
	//int f_vvv = (verbose_level >= 3);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go, target_go, quo, rem;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5;
	int cnt, drop_out_level, image, f_added, c;

	if (f_v) {
		cout << "sims::conjugate "
				"f_overshooting_OK=" << f_overshooting_OK << endl;
	}
	if (f_v) {
		cout << "action = " << A->label << endl;
	}
	if (false) {
		cout << "transporter = " << endl;
		A->Group_element->print(cout, Elt);
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	Elt5 = NEW_int(A->elt_size_in_int);
	init(A, verbose_level - 2);
	init_trivial_group(verbose_level - 1);
	group_order(go);
	old_G->group_order(target_go);
	A->Group_element->invert(Elt, Elt2);
	cnt = 0;
	while (true) {

		if (f_vv) {
			cout << "sims::conjugate iteration " << cnt << endl;
		}
		if (cnt > 500) {
			cout << "sims::conjugate cnt > 1000, "
					"something seems to be wrong" << endl;
			exit(1);
		}
		if ((cnt % 2) == 0) {
			if (f_vv) {
				cout << "sims::conjugate choosing random schreier generator" << endl;
			}
			random_schreier_generator(Elt1, verbose_level - 3);
			A->Group_element->element_move(Elt1, A->Elt1, false);
			if (false) {
				cout << "sims::conjugate random element chosen:" << endl;
				A->Group_element->element_print(A->Elt1, cout);
				cout << endl;
			}
			A->Group_element->move(A->Elt1, Elt4);
		}
		else if ((cnt % 2) == 1){
			if (f_vv) {
				cout << "sims::conjugate choosing random element in the group "
						"by which we extend" << endl;
			}
			old_G->random_element(A->Elt1, verbose_level - 1);
			if (false) {
				cout << "sims::conjugate random element chosen, path = ";
				Int_vec_print(cout, old_G->path, old_G->A->base_len());
				cout << endl;
			}
			if (false) {
				A->Group_element->element_print(A->Elt1, cout);
				cout << endl;
			}
			A->Group_element->mult(Elt, A->Elt1, Elt3);
			A->Group_element->mult(Elt3, Elt2, Elt4);
			if (f_vv) {
				cout << "sims::conjugate conjugated" << endl;
			}
			if (false) {
				A->Group_element->element_print(Elt4, cout);
				cout << endl;
			}
		}
		if (strip(Elt4, A->Elt2, drop_out_level, image,
				verbose_level - 3)) {
			if (f_vv) {
				cout << "sims::conjugate element strips through, "
						"residue = " << endl;
				if (false) {
					A->Group_element->element_print_quick(A->Elt2, cout);
					cout << endl;
				}
			}
			f_added = false;
		}
		else {
			f_added = true;
			if (f_vv) {
				cout << "sims::conjugate element needs to be inserted at level = "
					<< drop_out_level << " with image "
					<< image << endl;
				if (false) {
					A->Group_element->element_print(A->Elt2, cout);
					cout  << endl;
				}
			}
			add_generator_at_level(A->Elt2, drop_out_level,
					verbose_level - 3);
		}

		group_order(go);
		if ((f_v && f_added) || f_vv) {
			cout << "sims::conjugate current group order is " << go << endl;
		}
		if (f_vv) {
			print_transversal_lengths();
		}
		c = D.compare(target_go, go);
		cnt++;
		if (c == 0) {
			if (f_v) {
				cout << "sims::conjugate reached the full group after "
						<< cnt << " iterations" << endl;
			}
			break;
		}
		if (c < 0) {
			if (true) {
				cout << "sims::conjugate overshooting the expected "
						"group after " << cnt << " iterations" << endl;
				cout << "current group order is " << go
						<< " target_go=" << target_go << endl;
			}
			if (f_overshooting_OK) {
				break;
			}
			else {
				exit(1);
			}
		}
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(Elt5);
	if (f_v) {
		cout << "sims::conjugate done" << endl;
	}
}

int sims::test_if_in_set_stabilizer(
		actions::action *A,
		long int *set, int size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go, a;
	long int goi, i, ret;
	int *Elt1;

	if (f_v) {
		cout << "sims::test_if_in_set_stabilizer "
				"action = " << A->label << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "testing group of order " << goi << endl;
		}
	ret = true;
	for (i = 0; i < goi; i++) {
		a.create(i, __FILE__, __LINE__);
		element_unrank(a, Elt1);
		if (A->Group_element->check_if_in_set_stabilizer(Elt1,
				size, set, verbose_level)) {
			if (f_vv) {
				cout << "element " << i
						<< " strips through, residue = " << endl;
				}
			}
		else {
			cout << "element " << i
					<< " does not stabilize the set" << endl;
			A->Group_element->element_print(Elt1, cout);
			cout << endl;
			ret = false;
			break;
			}
		}
	FREE_int(Elt1);
	return ret;
}

int sims::test_if_subgroup(
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go, a, b;
	int goi, i, ret, drop_out_level, image;
	int *Elt1, *Elt2;

	if (f_v) {
		cout << "sims::test_if_subgroup" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	old_G->group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "testing group of order " << goi << endl;
		}
	ret = true;
	for (i = 0; i < goi; i++) {
		a.create(i, __FILE__, __LINE__);
		old_G->element_unrank(a, Elt1);
		if (strip(Elt1, Elt2, drop_out_level, image,
				verbose_level - 3)) {
			a.create(i, __FILE__, __LINE__);
			old_G->element_unrank(a, Elt1);
			element_rank(b, Elt1);
			if (f_vv) {
				cout << "element " << i
						<< " strips through, rank " << b << endl;
				}
			}
		else {
			cout << "element " << i << " is not contained" << endl;
			old_G->element_unrank(a, Elt1);
			A->Group_element->element_print(Elt1, cout);
			cout << endl;
			ret = false;
			break;
			}
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
	return ret;
}

int sims::find_element_with_exactly_n_fixpoints_in_given_action(
		int *Elt, int nb_fixpoints,
		actions::action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	long int i, order = 0;
	long int goi;
	int *cycle_type;

	if (f_v) {
		cout << "sims::find_element_with_exactly_n_fixpoints_in_given_action" << endl;
	}
	cycle_type = NEW_int(A_given->degree);
	group_order(go);
	goi = go.as_lint();
	for (i = 0; i < goi; i++) {
		element_unrank_lint(i, Elt);
		order = A_given->Group_element->element_order_and_cycle_type(Elt, cycle_type);
		if (cycle_type[0] == nb_fixpoints) {
			if (f_v) {
				cout << "sims::find_element_with_exactly_n_fixpoints_in_given_action "
						"found an element of order " << order
						<< " and with exactly " << nb_fixpoints << " fixpoints" << endl;
				cout << "Elt=" << endl;
				A->Group_element->element_print(Elt, cout);
			}
			break;
		}
	}
	if (i == goi) {
		cout << "sims::find_element_with_exactly_n_fixpoints_in_given_action "
				"could not find a suitable element" << endl;
		exit(1);
	}
	FREE_int(cycle_type);
	if (f_v) {
		cout << "sims::find_element_with_exactly_n_fixpoints_in_given_action done" << endl;
	}
	return order;
}

void sims::table_of_group_elements_in_data_form(
		int *&Table, int &len, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	ring_theory::longinteger_object go;
	long int i;

	if (f_v) {
		cout << "sims::table_of_group_elements_in_data_form" << endl;
		}
	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	len = go.as_lint();
	sz = A->make_element_size;
	Table = NEW_int(len * sz);
	for (i = 0; i < len; i++) {
		element_unrank_lint(i, Elt);
		Int_vec_copy(Elt, Table + i * sz, sz);
		}
	FREE_int(Elt);
	if (f_v) {
		cout << "sims::table_of_group_elements_in_data_form done" << endl;
		}
}

void sims::regular_representation(
		int *Elt,
		int *perm, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	long int goi, i, j;
	int *Elt1;
	int *Elt2;
	combinatorics::combinatorics_domain Combi;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	group_order(go);
	goi = go.as_lint();
	for (i = 0; i < goi; i++) {
		element_unrank_lint(i, Elt1);
		A->Group_element->mult(Elt1, Elt, Elt2);
		j = element_rank_lint(Elt2);
		perm[i] = j;
		}
	if (f_v) {
		cout << "sims::regular_representation of" << endl;
		A->Group_element->print(cout, Elt);
		cout << endl;
		cout << "is:" << endl;
		Combi.perm_print(cout, perm, goi);
		cout << endl;
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
}

void sims::element_ranks_subgroup(
		sims *subgroup,
		int *element_ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	long int goi;
	long int i, j;
	int *Elt1;

	subgroup->group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "sims::element_ranks_subgroup subgroup of order "
				<< goi << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	for (i = 0; i < goi; i++) {
		subgroup->element_unrank_lint(i, Elt1);
		j = element_rank_lint(Elt1);
		element_ranks[i] = j;
		}
	FREE_int(Elt1);
}

void sims::center(
		data_structures_groups::vector_ge &gens,
		int *center_element_ranks, int &nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	data_structures_groups::vector_ge gens_inv;
	long int goi, i, j, k, len;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	if (f_v) {
		cout << "sims::center" << endl;
		}
	len = gens.len;
	gens_inv.init(A, verbose_level - 2);
	gens_inv.allocate(len, verbose_level - 2);
	for (i = 0; i < len; i++) {
		A->Group_element->invert(gens.ith(i), gens_inv.ith(i));
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	nb_elements = 0;
	group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "sims::center computing the center "
				"of a group of order " << goi << endl;
		}
	for (i = 0; i < goi; i++) {
		element_unrank_lint(i, Elt1);
		for (j = 0; j < len; j++) {
			A->Group_element->mult(gens_inv.ith(j), Elt1, Elt2);
			A->Group_element->mult(Elt2, gens.ith(j), Elt3);
			k = element_rank_lint(Elt3);
			if (k != i)
				break;
			}
		if (j == len) {
			center_element_ranks[nb_elements++] = i;
			}
		}
	if (f_v) {
		cout << "sims::center center is of order "
				<< nb_elements << ":" << endl;
		Int_vec_print(cout, center_element_ranks, nb_elements);
		cout << endl;
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "sims::center done" << endl;
		}
}

void sims::all_cosets(
		int *subset, int size,
		long int *all_cosets, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	long int goi, i, j, k, nb_cosets, cnt;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *f_taken;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "sims::all_cosets" << endl;
		cout << "action " << A->label << endl;
		cout << "subset of order " << size << endl;
		cout << "group of order " << goi << endl;
		}
	nb_cosets = goi / size;
	if (size * nb_cosets != goi) {
		cout << "sims::all_cosets size * nb_cosets != goi" << endl;
		}
	f_taken = NEW_int(goi);
	for (i = 0; i < goi; i++) {
		f_taken[i] = false;
		}
	cnt = 0;
	for (i = 0; i < goi; i++) {
		if (f_taken[i])
			continue;
		element_unrank_lint(i, Elt1);
		for (j = 0; j < size; j++) {
			element_unrank_lint(subset[j], Elt2);
			A->Group_element->mult(Elt2, Elt1, Elt3); // we need right cosets!!!
			k = element_rank_lint(Elt3);
			if (f_taken[k]) {
				cout << "sims::all_cosets error: f_taken[k]" << endl;
				exit(1);
				}
			all_cosets[cnt * size + j] = k;
			f_taken[k] = true;
			}
		cnt++;
		}
	if (cnt != nb_cosets) {
		cout << "sims::all_cosets cnt != nb_cosets" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "sims::all_cosets finished" << endl;
		orbiter_kernel_system::Orbiter->Lint_vec->matrix_print_width(cout,
				all_cosets, nb_cosets, size, size, 2);
		cout << endl;
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(f_taken);
}

void sims::find_standard_generators_int(
		int ord_a, int ord_b,
		int ord_ab, int &a, int &b, int &nb_trials,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_trials1, o;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	if (f_v) {
		cout << "sims::find_standard_generators_int "
				"ord_a=" << ord_a
				<< " ord_b=" << ord_b
				<< " ord_ab=" << ord_ab << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	while (true) {

		a = find_element_of_given_order_int(ord_a,
				nb_trials1, verbose_level - 1);

		nb_trials += nb_trials1;

		b = find_element_of_given_order_int(ord_b,
				nb_trials1, verbose_level - 1);

		nb_trials += nb_trials1;

		element_unrank_lint(a, Elt1);
		element_unrank_lint(b, Elt2);

		A->Group_element->mult(Elt1, Elt2, Elt3);

		o = A->Group_element->element_order(Elt3);

		if (o == ord_ab) {
			break;
		}
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "sims::find_standard_generators_int "
				"found a=" << a << " b=" << b
				<< " nb_trials=" << nb_trials << endl;
	}
}

long int sims::find_element_of_given_order_int(
		int ord,
		int &nb_trials, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	int o, d, goi;
	int *Elt1;
	long int a;

	nb_trials = 0;
	group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "sims::find_element_of_given_order_int" << endl;
		cout << "action " << A->label << endl;
		cout << "group of order " << goi << endl;
		cout << "looking for an element of order " << ord << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	while (true) {
		nb_trials++;
		if (f_v) {
			cout << "sims::find_element_of_given_order_int "
					"before random_element" << endl;
		}
		random_element(Elt1, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "sims::find_element_of_given_order_int :"
					"after random_element" << endl;
		}
		o = A->Group_element->element_order(Elt1);
		if (f_v) {
			cout << "sims::find_element_of_given_order_int "
					"random_element has order " << o << endl;
		}
		if (o % ord == 0) {
			if (f_v) {
				cout << "sims::find_element_of_given_order_int "
						"the order is divisible by " << ord
						<< " which is good" << endl;
			}
			break;
		}
	}
	d = o / ord;
	if (f_v) {
		cout << "sims::find_element_of_given_order_int "
				"raising to the power " << d << endl;
	}
	A->Group_element->element_power_int_in_place(Elt1, d, verbose_level - 1);
	if (f_v) {
		cout << "sims::find_element_of_given_order_int "
				"after raising to the power " << d << endl;
	}
	a = element_rank_lint(Elt1);
	FREE_int(Elt1);
	return a;
}

int sims::find_element_of_given_order_int(
		int *Elt,
		int ord, int &nb_trials, int max_trials,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 4);
	ring_theory::longinteger_object go;
	int o, d, goi;
	int *Elt1;

	nb_trials = 0;
	group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "sims::find_element_of_given_order_int" << endl;
		cout << "action " << A->label << endl;
		cout << "group of order " << goi << endl;
		cout << "looking for an element of order " << ord << endl;
		cout << "max_trials = " << max_trials << endl;
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	o = 0;
	while (nb_trials < max_trials) {
		nb_trials++;
		if (f_vv) {
			cout << "sims::find_element_of_given_order_int "
					"before random_element" << endl;
		}
		random_element(Elt1, 0 /*verbose_level - 1*/);
		if (f_vv) {
			cout << "sims::find_element_of_given_order_int "
					"after random_element" << endl;
		}
		o = A->Group_element->element_order(Elt1);
		if (f_vv) {
			cout << "sims::find_element_of_given_order_int "
					"random_element has order " << o << endl;
		}
		if (o % ord == 0) {
			if (f_vv) {
				cout << "sims::find_element_of_given_order_int "
						"the order is divisible by " << ord
						<< " which is good" << endl;
			}
			break;
		}
	}
	if (nb_trials == max_trials) {
		FREE_int(Elt1);
		if (f_v) {
			cout << "sims::find_element_of_given_order_int "
					"unsuccessful" << endl;
		}
		return false;
	}
	d = o / ord;
	if (f_v) {
		cout << "sims::find_element_of_given_order_int "
				"raising to the power " << d << endl;
	}
	A->Group_element->element_power_int_in_place(Elt1, d, verbose_level - 1);
	if (f_v) {
		cout << "sims::find_element_of_given_order_int "
				"after raising to the power " << d << endl;
	}
	A->Group_element->element_move(Elt1, Elt, 0);
	FREE_int(Elt1);
	if (f_v) {
		cout << "sims::find_element_of_given_order_int done" << endl;
	}
	return true;
}

void sims::find_element_of_prime_power_order(
		int p,
		int *Elt, int &e, int &nb_trials,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	int o;

	nb_trials = 0;
	group_order(go);
	if (f_v) {
		cout << "sims::find_element_of_prime_power_order" << endl;
		cout << "action " << A->label << endl;
		cout << "group of order " << go << endl;
		cout << "prime " << p << endl;
	}
	while (true) {
		nb_trials++;
		random_element(Elt, 0 /*verbose_level - 1*/);
		o = A->Group_element->element_order(Elt);
		e = 0;
		while (o % p == 0) {
			e++;
			o = o / p;
		}
		if (e) {
			break;
		}
	}
	A->Group_element->element_power_int_in_place(Elt, o, verbose_level - 1);
	if (f_v) {
		cout << "sims::find_element_of_prime_power_order done, "
				"e=" << e << " nb_trials=" << nb_trials << endl;
	}
}

void sims::evaluate_word_int(
		int word_len,
		int *word, int *Elt, int verbose_level)
{
	int *Elt1;
	int *Elt2;
	long int i, j;


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	A->Group_element->one(Elt);
	for (i = 0; i < word_len; i++) {
		j = word[i];
		element_unrank_lint(j, Elt1);
		A->Group_element->mult(Elt1, Elt, Elt2);
		A->Group_element->move(Elt2, Elt);
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
}

void sims::sylow_subgroup(
		int p, sims *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go, go1, go_P, go_P1;
	int i, e, e1, c, nb_trials;

	if (f_v) {
		cout << "sims::sylow_subgroup" << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	group_order(go);
	if (f_v) {
		cout << "sims::sylow_subgroup "
				"the group has order "
				<< go << endl;
	}
	e = D.multiplicity_of_p(go, go1, p);
	if (f_v) {
		cout << "sims::sylow_subgroup the prime "
				<< p << " divides exactly " << e << " times" << endl;
	}
	go_P.create_power(p, e);
	if (f_v) {
		cout << "sims::sylow_subgroup trying to find a subgroup "
				"of order " << go_P << endl;
	}

	P->init(A, verbose_level - 2);
	P->init_trivial_group(verbose_level - 1);

	P->group_order(go_P1);
	while (true) {

		c = D.compare(go_P1, go_P);

		if (c == 0) {
			break;
		}

		if (c > 0) {
			cout << "sims::sylow_subgroup "
					"overshooting the group order" << endl;
			exit(1);
		}

		find_element_of_prime_power_order(
				p, Elt1, e1,
				nb_trials, 0 /* verbose_level */);

		for (i = 0; i < e1; i++) {
			if (P->is_normalizing(Elt1,
					0 /* verbose_level */)) {

				if (P->strip_and_add(Elt1, Elt2 /* residue */,
						0 /* verbose_level */)) {
					P->group_order(go_P1);
					if (f_v) {
						cout << "sims::sylow_subgroup "
								"the order of the "
								"subgroup has increased to " << go_P1 << endl;
					}
				}
				break;
			}
			A->Group_element->element_power_int_in_place(Elt1, p,
					0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "sims::sylow_subgroup found a " << p
				<< "-Sylow subgroup of order " << go_P1 << endl;
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "sims::sylow_subgroup done" << endl;
	}
}

int sims::is_normalizing(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = false;
	int i;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int drop_out_level, image;

	if (f_v) {
		cout << "sims::is_normalizing" << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);

	for (i = 0; i < nb_gen[0]; i++) {
		A->Group_element->element_invert(Elt, Elt1, 0);
		A->Group_element->element_move(gens.ith(i), Elt2, 0);
		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
		A->Group_element->element_mult(Elt3, Elt, Elt4, 0);
		if (!strip(Elt4, Elt3 /* residue */, drop_out_level,
				image, 0 /* verbose_level */)) {
			if (f_v) {
				cout << "sims::is_normalizing the element "
						"does not normalize generator "
						<< i << " / " << nb_gen[0] << endl;
			}
			break;
		}
	}
	if (i == nb_gen[0]) {
		if (f_v) {
			cout << "sims::is_normalizing the element "
					"normalizes all " << nb_gen[0]
					<< " generators" << endl;
		}
		ret = true;
	}
	else {
		ret = false;
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	if (f_v) {
		cout << "sims::is_normalizing done" << endl;
	}
	return ret;
}

void sims::create_Cayley_graph(
		data_structures_groups::vector_ge *gens,
		int *&Adj, long int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h, j;
	ring_theory::longinteger_object go;
	int *Elt1;
	int *Elt2;

	if (f_v) {
		cout << "sims::create_Cayley_graph" << endl;
	}
	group_order(go);
	n = go.as_lint();
	if (f_v) {
		cout << "sims::create_Cayley_graph "
				"Computing the adjacency matrix of a graph with "
				<< n << " vertices" << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Adj = NEW_int(n * n);
	Int_vec_zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		element_unrank_lint(i, Elt1);
		//cout << "i=" << i << endl;
		for (h = 0; h < gens->len; h++) {
			A->Group_element->element_mult(Elt1, gens->ith(h), Elt2, 0);
#if 0
			cout << "i=" << i << " h=" << h << endl;
			cout << "Elt1=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "g_h=" << endl;
			A->element_print_quick(gens->ith(h), cout);
			cout << "Elt2=" << endl;
			A->element_print_quick(Elt2, cout);
#endif
			j = element_rank_lint(Elt2);
			Adj[i * n + j] = Adj[j * n + i] = 1;
#if 0
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
			}
#endif
		}
	}

#if 0
	cout << "The adjacency matrix of a graph with "
			<< n << " vertices has been computed" << endl;
	//int_matrix_print(Adj, goi, goi);
#endif

	FREE_int(Elt1);
	FREE_int(Elt2);


	if (f_v) {
		cout << "sims::create_Cayley_graph done" << endl;
	}
}

void sims::create_group_table(
		int *&Table, long int &n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	ring_theory::longinteger_object go;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	if (f_v) {
		cout << "sims::create_group_table" << endl;
	}
	group_order(go);
	n = go.as_int();
	if (f_v) {
		cout << "sims::create_group_table "
				"Computing the table of a group of order "
				<< n << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Table = NEW_int(n * n);
	Int_vec_zero(Table, n * n);

	for (i = 0; i < n; i++) {

		element_unrank_lint(i, Elt1);
		//cout << "i=" << i << endl;

		for (j = 0; j < n; j++) {

			element_unrank_lint(j, Elt2);
			A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
#if 0
			cout << "i=" << i << " j=" << j << endl;
			cout << "Elt_i=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "Elt_j=" << endl;
			A->element_print_quick(Elt2, cout);
			cout << "Elt3=" << endl;
			A->element_print_quick(Elt3, cout);
#endif
			k = element_rank_lint(Elt3);
			Table[i * n + j] = k;
		}
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "sims::create_group_table done" << endl;
	}
}

void sims::compute_conjugacy_classes(
		actions::action *&Aconj,
		induced_actions::action_by_conjugation *&ABC, schreier *&Sch,
	strong_generators *&SG, int &nb_classes,
	int *&class_size, int *&class_rep,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f;

	if (f_v) {
		cout << "sims::compute_conjugacy_classes" << endl;
	}
	//Aconj = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "sims::compute_conjugacy_classes "
				"before Aconj->induced_action_by_conjugation" << endl;
	}

	Aconj = A->Induced_action->create_induced_action_by_conjugation(
		this,
		false /* f_ownership */,
		false /* f_basis */, NULL,
		verbose_level - 1);

#if 0
	action *induced_action::create_induced_action_by_conjugation(
			groups::sims *Base_group, int f_ownership,
			int f_basis, groups::sims *old_G,
			int verbose_level)
#endif


	if (f_v) {
		cout << "sims::compute_conjugacy_classes "
				"after Aconj->induced_action_by_conjugation" << endl;
	}

	ABC = Aconj->G.ABC;


	Sch = NEW_OBJECT(schreier);

	Sch->init(Aconj, verbose_level - 2);


	SG = NEW_OBJECT(strong_generators);

	SG->init_from_sims(this, 0);


	Sch->init_generators(*SG->gens, verbose_level - 2);

	if (f_v) {
		cout << "sims::compute_conjugacy_classes "
				"Computing conjugacy classes:" << endl;
	}
	Sch->compute_all_point_orbits(verbose_level);


	nb_classes = Sch->nb_orbits;

	class_size = NEW_int(nb_classes);
	class_rep = NEW_int(nb_classes);

	for (i = 0; i < nb_classes; i++) {
		class_size[i] = Sch->orbit_len[i];
		f = Sch->orbit_first[i];
		class_rep[i] = Sch->orbit[f];
	}

	if (f_v) {
		cout << "class size : ";
		Int_vec_print(cout, class_size, nb_classes);
		cout << endl;
		cout << "class rep : ";
		Int_vec_print(cout, class_rep, nb_classes);
		cout << endl;
	}


	if (f_v) {
		cout << "sims::compute_conjugacy_classes done" << endl;
	}

}

void sims::compute_all_powers(
		int elt_idx, int n, int *power_elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	if (f_v) {
		cout << "sims::compute_all_powers" << endl;
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	element_unrank_lint(elt_idx, Elt1);
	A->Group_element->element_move(Elt1, Elt2, 0);
	power_elt[0] = elt_idx;

	for (i = 2; i <= n; i++) {
		A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
		a = element_rank_lint(Elt3);
		power_elt[i - 1] = a;
		A->Group_element->element_move(Elt3, Elt1, 0);
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "sims::create_group_table done" << endl;
	}
}

long int sims::mult_by_rank(
		long int rk_a, long int rk_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rk_c;

	if (f_v) {
		cout << "sims::mult_by_rank" << endl;
		}
	element_unrank_lint(rk_a, Elt1);
	element_unrank_lint(rk_b, Elt2);
	A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);
	rk_c = element_rank_lint(Elt3);
	return rk_c;
}

long int sims::mult_by_rank(
		long int rk_a, long int rk_b)
{
	int rk_c;

	rk_c = mult_by_rank(rk_a, rk_b, 0);
	return rk_c;
}

long int sims::invert_by_rank(
		long int rk_a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rk_b;

	if (f_v) {
		cout << "sims::invert_by_rank" << endl;
		}
	element_unrank_lint(rk_a, Elt1);
	A->Group_element->element_invert(Elt1, Elt2, 0);
	rk_b = element_rank_lint(Elt2);
	return rk_b;
}

long int sims::conjugate_by_rank(
		long int rk_a, long int rk_b,
		int verbose_level)
// computes b^{-1} * a * b
{
	int f_v = (verbose_level >= 1);
	long int rk_c;

	if (f_v) {
		cout << "sims::conjugate_by_rank" << endl;
		}
	element_unrank_lint(rk_a, Elt1); // Elt1 = a
	element_unrank_lint(rk_b, Elt2); // Elt2 = b
	A->Group_element->element_invert(Elt2, Elt3, 0); // Elt3 = b^{-1}
	A->Group_element->element_mult(Elt3, Elt1, Elt4, 0);
	A->Group_element->element_mult(Elt4, Elt2, Elt3, 0);
	rk_c = element_rank_lint(Elt3);
	return rk_c;
}

long int sims::conjugate_by_rank_b_bv_given(
		long int rk_a,
		int *Elt_b, int *Elt_bv, int verbose_level)
// comutes b^{-1} * a * b
{
	int f_v = (verbose_level >= 1);
	long int rk_c;

	if (f_v) {
		cout << "sims::conjugate_by_rank_b_bv_given" << endl;
		}
	element_unrank_lint(rk_a, Elt1); // Elt1 = a
	A->Group_element->element_mult(Elt_bv, Elt1, Elt4, 0);
	A->Group_element->element_mult(Elt4, Elt_b, Elt3, 0);
	rk_c = element_rank_lint(Elt3);
	return rk_c;
}

#if 0
int sims::identify_group(char *path_t144,
		char *discreta_home, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int group_idx;
	int h, i, j, *Elt;
	longinteger_object go;
	const char *fname = "group_generators.txt";

	if (f_v) {
		cout << "sims::identify_group" << endl;
		}
	group_order(go);
	{
	ofstream f(fname);

		// generators start from one

	f << gens.len << " " << A->degree << endl;
	for (h = 0; h < gens.len; h++) {
		Elt = gens.ith(h);
		for (i = 0; i < A->degree; i++) {
			j = A->element_image_of(i, Elt, 0);
			f << j + 1 << " ";
			}
		f << endl;
		}
	}
	if (f_v) {
		cout << "sims::identify_group written file "
				<< fname << " of size " << file_size(fname) << endl;
		}
	char cmd[2000];

	snprintf(cmd, sizeof(cmd), "%s/t144.out -discreta_home %s "
			"group_generators.txt >log.tmp",
			path_t144, discreta_home);

	if (f_v) {
		cout << "sims::identify_group calling '"
				<< cmd << "'" << endl;
		}

	system(cmd);

	{
	ifstream f("result.txt");
	f >> group_idx;
	}
	if (f_v) {
		cout << "sims::identify_group: the group is "
				"isomorphic to group " << go << "#"
				<< group_idx << endl;
		}
	return group_idx;
}
#endif


void sims::zuppo_list(
		int *Zuppos, int &nb_zuppos, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int goi;
	ring_theory::longinteger_object go;
	int rk, o, i, j;
	int *Elt1;
	int *Elt2;
	int *f_done;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "sims::zuppo_list" << endl;
		}
	group_order(go);
	cout << "go=" << go << endl;
	goi = go.as_int();
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	f_done = NEW_int(goi);
	Int_vec_zero(f_done, goi);
	if (f_v) {
		cout << "sims::zuppo_list group of order " << goi << endl;
		}
	nb_zuppos = 0;
	for (rk = 0; rk < goi; rk++) {
		//cout << "element " << rk << " / " << goi << endl;
		if (f_done[rk]) {
			continue;
			}
		element_unrank_lint(rk, Elt1, 0 /*verbose_level*/);
		//cout << "element created" << endl;
		o = A->Group_element->element_order(Elt1);
		//cout << "element order = " << o << endl;
		if (o == 1) {
			continue;
			}
		if (!NT.is_prime_power(o)) {
			continue;
			}
		if (f_v) {
			cout << "sims::zuppo_list element " << rk << " / " << goi << " has order "
					<< o << " which is a prime power; "
					"nb_zuppos = " << nb_zuppos << endl;
		}
		Zuppos[nb_zuppos++] = rk;
		f_done[rk] = true;
		for (i = 1; i < o; i++) {
			if (NT.gcd_lint(i, o) == 1) {
				A->Group_element->element_move(Elt1, Elt2, 0);
				A->Group_element->element_power_int_in_place(Elt2,
						i, 0 /* verbose_level*/);
				j = element_rank_lint(Elt2);
				f_done[j] = true;
				}
			}
		}
	if (f_v) {
		cout << "sims::zuppo_list We found " << nb_zuppos << " zuppo elements" << endl;
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(f_done);
	if (f_v) {
		cout << "sims::zuppo_list done" << endl;
		}
}

void sims::dimino(
	int *subgroup, int subgroup_sz, int *gens, int &nb_gens,
	int *cosets,
	int new_gen,
	int *group, int &group_sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, c, idx, new_coset_rep, nb_cosets;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "sims::dimino new_gen = " << new_gen << endl;
		}
	Int_vec_copy(subgroup, group, subgroup_sz);
	Sorting.int_vec_heapsort(group, subgroup_sz);
	group_sz = subgroup_sz;

	cosets[0] = 0;
	nb_cosets = 1;
	gens[nb_gens++] = new_gen;
	for (i = 0; i < nb_cosets; i++) {
		for (j = 0; j < nb_gens; j++) {
			if (f_vv) {
				cout << "sims::dimino coset rep " << i << " = " << cosets[i] << endl;
				cout << "sims::dimino generator " << j << " = " << gens[j] << endl;
				}

			c = mult_by_rank(cosets[i], gens[j]);
			if (f_vv) {
				cout << "sims::dimino coset rep " << i << " times generator "
						<< j << " is " << c << endl;
				}
			if (Sorting.int_vec_search(group, group_sz, c, idx)) {
				if (f_vv) {
					cout << "sims::dimino already there" << endl;
					}
				continue;
				}
			if (f_vv) {
				cout << "sims::dimino new coset rep" << endl;
				}
			new_coset_rep = c;

			for (k = 0; k < subgroup_sz; k++) {
				c = mult_by_rank(subgroup[k], new_coset_rep);
				group[group_sz++] = c;
				}
			Sorting.int_vec_heapsort(group, group_sz);
			if (f_vv) {
				cout << "sims::dimino new group size = " << group_sz << endl;
				}
			cosets[nb_cosets++] = new_coset_rep;
			}
		}
	if (f_vv) {
		cout << "sims::dimino, the group order has been updated to " << group_sz << endl;
		}

	if (f_v) {
		cout << "sims::dimino done" << endl;
		}
}

void sims::Cayley_graph(
		int *&Adj, int &sz,
		data_structures_groups::vector_ge *gens_S,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::Cayley_graph" << endl;
	}
	int *Elt1, *Elt2;
	long int i, j;
	int h;
	int nb_S;


	nb_S = gens_S->len;
	sz = group_order_lint();

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Adj = NEW_int(sz * sz);

	Int_vec_zero(Adj, sz * sz);

	if (f_v) {
		cout << "Computing the Cayley graph:" << endl;
	}

	for (i = 0; i < sz; i++) {

		element_unrank_lint(i, Elt1);
		//cout << "i=" << i << endl;

		for (h = 0; h < nb_S; h++) {

			A->Group_element->element_mult(Elt1, gens_S->ith(h), Elt2, 0);
#if 0
			cout << "i=" << i << " h=" << h << endl;
			cout << "Elt1=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "g_h=" << endl;
			A->element_print_quick(gens->ith(h), cout);
			cout << "Elt2=" << endl;
			A->element_print_quick(Elt2, cout);
#endif
			j = element_rank_lint(Elt2);
			Adj[i * sz + j] = Adj[j * sz + i] = 1;

			if (i == 0) {
				if (f_v) {
					cout << "edge " << i << " " << j << endl;
				}
			}
		}
	}
	if (f_v) {
		cout << "sims::Cayley_graph done" << endl;
	}

}


}}}


