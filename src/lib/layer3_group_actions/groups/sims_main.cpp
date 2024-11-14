/*
 * sims_main.cpp
 *
 *  Created on: Sep 1, 2019
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace groups {



void sims::compute_base_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "sims::compute_base_orbits" << endl;
	}
	if (f_vv) {
		cout << "sims::compute_base_orbits "
				"base_len=" << A->base_len() << endl;
	}
	for (i = A->base_len() - 1; i >= 0; i--) {
		if (false) {
			cout << "sims::compute_base_orbits "
					"level " << i << endl;
		}
		compute_base_orbit(i, 0/*verbose_level - 1*/);
		if (f_vv) {
			cout << "sims::compute_base_orbits level " << i
				<< " base point " << A->base_i(i)
				<< " orbit length " << orbit_len[i] << endl;
		}
	}
	if (f_v) {
		cout << "sims::compute_base_orbits done, orbit_len=";
		Int_vec_print(cout, orbit_len, A->base_len());
		cout << endl;
	}
}

void sims::compute_base_orbits_known_length(
		int *tl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "sims::compute_base_orbits_known_length: ";
		cout << "action " << A->label << " tl=";
		Int_vec_print(cout, tl, A->base_len());
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	for (i = A->base_len() - 1; i >= 0; i--) {
		if (f_v) {
			cout << "sims::compute_base_orbits_known_length "
					"computing level " << i << endl;
		}
		compute_base_orbit_known_length(i, tl[i], verbose_level);
		if (f_v) {
			cout << "sims::compute_base_orbits_known_length "
					"level " << i
				<< " base point " << A->base_i(i)
				<< " orbit length " << orbit_len[i]
				<< " has been computed" << endl;
		}
		if (orbit_len[i] != tl[i]) {
			cout << "sims::compute_base_orbits_known_length "
					"orbit_len[i] != tl[i]" << endl;
			cout << "orbit_len[i]=" << orbit_len[i] << endl;
			cout << "tl[i]=" << tl[i] << endl;
			print_generators_at_level_or_below(i);
			exit(1);
		}
		}
	if (f_v) {
		cout << "sims::compute_base_orbits_known_length done" << endl;
	}
}

void sims::extend_base_orbit(
		int new_gen_idx, int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, cur, cur_pt, total, total0;
	int next_pt, next_pt_loc, gen_idx, nbg;

	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl
				<< " verbose_level = " << verbose_level << endl;
	}
	cur = 0;
	total = total0 = orbit_len[lvl];
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl
				<< " orbit_len[lvl] = " << orbit_len[lvl] << endl;
	}
	nbg = nb_gen[lvl];
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl
				<< " nbg = " << nbg << endl;
	}
	if (orbit[lvl] == NULL) {
		cout << "sims::extend_base_orbit orbit[lvl] == NULL" << endl;
		exit(1);
	}
	while (cur < total) {
		cur_pt = orbit[lvl][cur];
		if (f_vvv) {
			cout << "sims::extend_base_orbit: cur=" << cur
					<< " total = " << total << " cur_pt=" << cur_pt << endl;
		}
		for (i = 0; i < nbg; i++) {
			if (f_vvv) {
				cout << "sims::extend_base_orbit: "
						"applying generator " << i << " / " << nbg << endl;
			}
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			if (f_vvv) {
				cout << "sims::extend_base_orbit: "
						"next_pt = " << next_pt << endl;
			}
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (f_vvv) {
				cout << "sims::extend_base_orbit: "
						"next_pt_loc = " << next_pt_loc << endl;
			}
			if (f_vvv) {
				cout << "sims::extend_base_orbit "
						"generator " << gen_idx << " maps "
						<< cur_pt << " to " << next_pt << endl;
			}
			if (next_pt_loc < total) {
				continue;
			}
			if (f_vvv) {
				cout << "sims::extend_base_orbit "
						"additional pt " << next_pt << " reached from "
						<< cur_pt << " under generator "
						<< i << endl;
			}
			swap_points(lvl, total, next_pt_loc);
			prev[lvl][total] = cur_pt;
			label[lvl][total] = gen_idx;
			total++;
			if (f_vvv) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				//print_orbit(cur, total - 1);
			}
		}
		cur++;
	}
	orbit_len[lvl] = total;
	if (f_v) {
		cout << "sims::extend_base_orbit "
				<< lvl << " finished" << endl;
		cout << lvl << "-th base point " << A->base_i(lvl)
			<< " orbit extended to length " << orbit_len[lvl];
		if (false) {
			cout << " { ";
			for (i = 0; i < orbit_len[lvl]; i++) {
				cout << orbit[lvl][i];
				if (i < orbit_len[lvl] - 1)
					cout << ", ";
			}
			cout << " }" << endl;
		}
		else {
			cout << endl;
		}
	}
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << " done" << endl;
	}
}

void sims::compute_base_orbit(
		int lvl, int verbose_level)
// applies all generators at the given level to compute
// the corresponding basic orbit.
// the generators are the first nb_gen[lvl] in the generator arry
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int pt, pt_loc, cur, cur_pt, i, next_pt, next_pt_loc, gen_idx;

	pt = A->base_i(lvl);
	pt_loc = orbit_inv[lvl][pt];
	if (f_v) {
		cout << "sims::compute_base_orbit: "
				"computing orbit of " << lvl << "-th base point "
				<< pt << " with " << nb_gen[lvl] << " generators" << endl;
	}
	if (pt_loc > 0) {
		swap_points(lvl, 0, pt_loc);
	}
	cur = 0;
	orbit_len[lvl] = 1;
	while (cur < orbit_len[lvl]) {
		cur_pt = orbit[lvl][cur];
		if (false) {
			cout << "sims::compute_base_orbit "
					"applying generator to " << cur_pt << endl;
		}
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (false) {
				cout << "sims::compute_base_orbit "
						"generator " << i << " maps " << cur_pt
						<< " to " << next_pt << endl;
			}
			if (next_pt_loc < orbit_len[lvl]) {
				continue;
			}
			if (false) {
				cout << "additional pt " << next_pt << " reached from "
						<< cur_pt << " under generator "
						<< i << endl;
			}
			swap_points(lvl, orbit_len[lvl], next_pt_loc);
			prev[lvl][orbit_len[lvl]] = cur_pt;
			label[lvl][orbit_len[lvl]] = gen_idx;
			orbit_len[lvl]++;
			if (false) {
				cout << "sims::compute_base_orbit "
						"cur = " << cur << endl;
				cout << "sims::compute_base_orbit "
						"orbit_len[lvl] = " << orbit_len[lvl] << endl;
				//print_orbit(cur, total - 1);
			}
		}
		cur++;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit finished, "
			<< lvl << "-th base orbit of length "
			<< orbit_len[lvl] << endl;
	}
	if (false) {
		cout << "{ ";
		for (i = 0; i < orbit_len[lvl]; i++) {
			cout << orbit[lvl][i];
			if (i < orbit_len[lvl] - 1)
				cout << ", ";
		}
		cout << " }" << endl;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit done" << endl;
	}
}

void sims::compute_base_orbit_known_length(
		int lvl,
		int target_length, int verbose_level)
// applies all generators at the given level to compute
// the corresponding basic orbit.
// the generators are the first nb_gen[lvl] in the generator array
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;
	//int f_vvv = (verbose_level >= 3);
	//int f_v10 = false; // (verbose_level >= 10);
	int pt, pt_loc, cur, cur_pt, i, next_pt, next_pt_loc, gen_idx;
	double progress;

	pt = A->base_i(lvl);
	pt_loc = orbit_inv[lvl][pt];
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length: "
				"computing orbit of " << lvl
			<< "-th base point " << pt
			<< " target_length = " << target_length
			<< " nb_gens=" << nb_gen[lvl] << endl;
	}
	if (target_length > 1000000) {
		f_vv = true;
	}
	if (false) {
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			cout << "sims::compute_base_orbit_known_length "
					"generator " << i << ":" << endl;
			A->Group_element->element_print_quick(gens.ith(gen_idx), cout);
		}
	}
	if (pt_loc > 0) {
		swap_points(lvl, 0, pt_loc);
	}
	cur = 0;
	orbit_len[lvl] = 1;
	while (cur < orbit_len[lvl] && orbit_len[lvl] < target_length) {
		cur_pt = orbit[lvl][cur];
		if (f_vv) {
			if (target_length) {
				progress = (double) cur / (double) target_length;
			}
			else {
				progress = 0.;
			}
			if (cur % ((1 << 21) - 1) == 0) {
				cout << "sims::compute_base_orbit_known_length "
						"lvl=" << lvl << " cur=" << cur
						<< " orbit_len[lvl]=" << orbit_len[lvl]
						<< " target_length=" << target_length
						<< " progress=" << progress * 100 << "%" << endl;
			}
		}
		if (false) {
			cout << "sims::compute_base_orbit_known_length "
					"applying " << nb_gen[lvl] << " generators to "
					<< cur_pt << " orbit_len[lvl]=" << orbit_len[lvl]
					<< " target_length=" << target_length << endl;
		}
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (false) {
				cout << "sims::compute_base_orbit_known_length "
						"generator " << i << " maps " << cur_pt
						<< " to " << next_pt << endl;
			}
			if (next_pt_loc < orbit_len[lvl]) {
				continue;
			}
			if (false) {
				cout << "sims::compute_base_orbit_known_length "
						"additional pt " << next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
			}
			swap_points(lvl, orbit_len[lvl], next_pt_loc);
			prev[lvl][orbit_len[lvl]] = cur_pt;
			label[lvl][orbit_len[lvl]] = gen_idx;
			orbit_len[lvl]++;
			if (false) {
				cout << "sims::compute_base_orbit_known_length "
						"cur = " << cur << endl;
				cout << "sims::compute_base_orbit_known_length "
						"orbit_len[lvl] = " << orbit_len[lvl] << endl;
				//print_orbit(cur, total - 1);
			}
		}
		cur++;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length finished, "
				<< lvl << "-th base orbit of length "
				<< orbit_len[lvl] << endl;
	}
	if (false) {
		cout << "{ ";
		for (i = 0; i < orbit_len[lvl]; i++) {
			cout << orbit[lvl][i];
			if (i < orbit_len[lvl] - 1) {
				cout << ", ";
			}
		}
		cout << " }" << endl;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length done" << endl;
	}
}

int sims::strip_and_add(
		int *elt, int *residue, int verbose_level)
// returns true if something was added,
// false if element stripped through
{
	int drop_out_level, image;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "sims::strip_and_add" << endl;
	}
	if (strip(elt, residue, drop_out_level,
			image, 0 /*verbose_level*/)) {
		if (f_v) {
			cout << "sims::strip_and_add "
					"element strips to the identity, finished" << endl;
		}
		return false;
	}
	if (f_v) {
		cout << "sims::strip_and_add after strip, "
				"drop_out_level = " << drop_out_level
				<< " image = " << image << endl;
	}
	if (false) {
		cout << "sims::strip_and_add residue = " << endl;
		A->Group_element->element_print_quick(residue, cout);
		//A->element_print_as_permutation(residue, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "sims::strip_and_add calling add_generator_at_level "
				<< drop_out_level << endl;
	}
	add_generator_at_level(residue, drop_out_level, verbose_level);
	//add_generator_at_level_only(residue, drop_out_level, verbose_level);
	// !!! this was add_generator_at_level previously

	if (false) {
		cout << "sims::strip_and_add increased set of generators:" << endl;
		gens.print(cout);
		gens.print_as_permutation(cout);
	}
	if (f_v) {
		cout << "sims::strip_and_add finished, final group order is ";
		print_group_order(cout);
		cout << endl;
	}
	if (f_v) {
		cout << "sims::strip_and_add done" << endl;
	}
	return true;
}

int sims::strip(
		int *elt, int *residue,
		int &drop_out_level, int &image,
		int verbose_level)
// returns true if the element sifts through
// returns fals if we drop out, which means at a certain level,
//   the image of the base point is not known to belong to the basic orbit
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, bi, j, j_coset;



	if (f_v) {
		cout << "sims::strip" << endl;
		cout << "sims::strip my_base_len=" << my_base_len << endl;
	}
	if (A == NULL) {
		cout << "sims::strip A==NULL" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "sims::strip A=" << A->label << endl;
		cout << "sims::strip A->base_len()=" << A->base_len() << endl;
		cout << "sims::strip my_base_len=" << my_base_len << endl;
	}

	if (my_base_len != A->base_len()) {
		cout << "sims::strip my_base_len != A->base_len()" << endl;
		cout << "sims::strip A=" << A->label << endl;
		cout << "sims::strip my_base_len=" << A->base_len() << endl;
		cout << "sims::strip my_base_len=" << my_base_len << endl;
		exit(1);
	}

	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = true;
	int f_print_cycles_of_length_one = true;

	if (f_vv) {
		A->Group_element->element_print_quick(elt, cout);
		cout << endl;

		A->Group_element->element_print_for_make_element(elt, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
				elt, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}
	A->Group_element->element_move(elt, strip1, 0);

	for (i = 0; i < my_base_len; i++) {
		if (f_v) {
			cout << "sims::strip level " << i
					<< " / " << my_base_len << endl;
			//A->element_print(strip1, cout);
			//cout << endl;
		}
		bi = A->base_i(i);
		if (f_vv) {
			cout << "sims::strip computing image of " << i
					<< "-th base element bi = " << bi << endl;
		}
		j = A->Group_element->element_image_of(
				bi, strip1, 0 /*verbose_level - 2*/);
		if (f_v) {
			cout << "sims::strip level " << i
					<< " / " << my_base_len
					<< " base point " << bi
					<< " gets mapped to " << j << endl;
		}
		if (f_v) {
			cout << "sims::strip level " << i << " / "
					<< A->base_len()
					<< " before get_orbit_inv j=" << j << endl;
		}
		j_coset = get_orbit_inv(i, j);
		if (f_v) {
			cout << "sims::strip j_coset " << j_coset << endl;
		}
		if (j_coset >= orbit_len[i]) {
			if (f_v) {
				cout << "sims::strip not in the orbit, "
						"dropping out" << endl;
			}
			image = j;
			drop_out_level = i;
			A->Group_element->element_move(strip1, residue, 0);
			if (f_v) {
				cout << "sims::strip returns false, "
						"drop_out_level=" << drop_out_level << endl;
			}
			return false;
		}
		else {
			if (f_v) {
				cout << "sims::strip computing representative "
						"of coset " << j_coset << endl;
				cout << "sims::strip before coset_rep_inv" << endl;
			}

			coset_rep_inv(eltrk3, i, j_coset, verbose_level - 2);


			if (false) {
				cout << "sims::strip representative "
						"of coset " << j_coset << " is " << endl;
				A->Group_element->element_print(eltrk3, cout);
				cout << endl;

				A->Group_element->element_print_for_make_element(eltrk3, cout);
				cout << endl;

				A->Group_element->element_print_as_permutation_with_offset(
						eltrk3, cout,
					offset, f_do_it_anyway_even_for_big_degree,
					f_print_cycles_of_length_one,
					0/*verbose_level*/);
				cout << endl;

			}
			if (false) {
				cout << "sims::strip before element_mult, "
						"strip1=" << endl;
				A->Group_element->element_print(strip1, cout);
				cout << endl;
			}
			if (false) {
				cout << "sims::strip before element_mult, "
						"cosetrep=" << endl;
				A->Group_element->element_print(eltrk3, cout);
				cout << endl;
			}
			A->Group_element->element_mult(strip1, eltrk3, strip2, 0 /*verboe_level*/);
			if (false) {
				cout << "sims::strip before element_move" << endl;
			}
			A->Group_element->element_move(strip2, strip1, 0);
			if (false) {
				cout << "sims::strip after dividing off, "
						"we have strip1= " << endl;
				A->Group_element->element_print(strip1, cout);
				cout << endl;
			}
		}
	}
	if (f_v) {
		cout << "sims::strip after loop" << endl;
	}
	A->Group_element->element_move(strip1, residue, 0);
	if (f_v) {
		cout << "sims::strip returns true" << endl;
	}
	return true;
}

void sims::add_generator_at_level(
		int *elt,
		int lvl, int verbose_level)
// add the generator to the array of generators and then extends the
// basic orbits 0,..,lvl using extend_base_orbit
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "sims::add_generator_at_level adding generator at "
				"level " << lvl
				<< " verbose_level = " << verbose_level<< endl;
		print_generator_depth_and_perm();
		if (false) {
			A->Group_element->element_print_quick(elt, cout);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "sims::add_generator_at_level "
				"before add_generator" << endl;
	}
	add_generator(elt, verbose_level);
	if (f_v) {
		cout << "sims::add_generator_at_level "
				"after add_generator" << endl;
		print_generator_depth_and_perm();
	}
	for (i = lvl; i >= 0; i--) {
		if (f_v) {
			cout << "sims::add_generator_at_level "
				<< lvl << " calling extend_base_orbit " << i << endl;
		}
		extend_base_orbit(gens.len - 1, i, verbose_level - 1);
	}
	if (f_v) {
		cout << "sims::add_generator_at_level done" << endl;
	}
}

void sims::add_generator_at_level_only(
		int *elt,
		int lvl, int verbose_level)
// add the generator to the array of generators and then extends the
// basic orbit lvl using extend_base_orbit
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "sims::add_generator_at_level_only "
				"level " << lvl << endl;
		if (f_vvv) {
			A->Group_element->element_print(elt, cout);
			cout << endl;
		}
	}
	add_generator(elt, verbose_level);
	if (f_vvv) {
		print_generator_depth_and_perm();
	}
	extend_base_orbit(gens.len - 1, lvl, verbose_level - 1);
	if (f_v) {
		cout << "sims::add_generator_at_level_only "
				"level " << lvl << " done" << endl;
	}
}

void sims::build_up_group_random_process_no_kernel(
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go, go1;
	sims K;

	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel" << endl;
	}
	old_G->group_order(go);
	if (f_v) {
		cout << "target group order = " << go << endl;
	}
	K.init(A, verbose_level - 2);
	K.init_trivial_group(verbose_level - 1);
	K.group_order(go1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel "
				"kernel group order " << go1 << endl;
	}
	init_trivial_group(verbose_level - 1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel "
				"before build_up_group_random_process" << endl;
	}
	build_up_group_random_process(
			&K, old_G, go,
		false /* f_override_chose_next_base_point */,
		NULL /* choose_next_base_point_method */,
		verbose_level - 1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel "
				"after build_up_group_random_process" << endl;
	}
}

void sims::extend_group_random_process_no_kernel(
		sims *extending_by_G,
		ring_theory::longinteger_object &target_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//longinteger_object go, go1, go2;
	//longinteger_domain D;
	sims K;

	if (f_v) {
		cout << "sims::extend_group_random_process_no_kernel" << endl;
	}
	//group_order(go);
	//extending_by_G->group_order(go1);
	//D.mult(go, go1, go2);
	if (f_v) {
		cout << "target group order = " << target_go << endl;
	}

	K.init(A, verbose_level - 2);
	K.init_trivial_group(verbose_level - 1);
	build_up_group_random_process(
		&K,
		extending_by_G,
		target_go,
		false /* f_override_chose_next_base_point */,
		NULL /* choose_next_base_point_method */,
		verbose_level + 3);
	if (f_v) {
		cout << "sims::extend_group_random_process_no_kernel done" << endl;
	}
}


int sims::closure_group(
		int nb_times, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	//int f_vvv = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 6);
	int i, f_extended = false;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	ring_theory::longinteger_object old_go, go, go1;

	if (f_v) {
		cout << "sims::closure_group" << endl;
	}
	if (f_vv) {
		print_transversal_lengths();
		cout << "verbose_level=" << verbose_level << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	group_order(old_go);
	if (f_vv) {
		cout << "sims::closure_group for group of order "
				<< old_go << endl;
	}
	if (old_go.is_one()) {
		FREE_int(Elt1);
		FREE_int(Elt2);
		if (f_v) {
			cout << "sims::closure_group finishing with false "
					"because the old group order is one" << endl;
		}
		return false;
	}
	group_order(go);
	for (i = 0; i < nb_times; i++) {
		if (f_vv) {
			cout << "sims::closure_group loop " << i << " / "
					<< nb_times << " go=" << go << endl;
		}
		if (f_v3) {
			cout << "sims::closure_group "
					"before random_schreier_generator" << endl;
		}
		random_schreier_generator(Elt3, verbose_level /*- 4*/);
		if (f_v3) {
			cout << "sims::closure_group "
					"after random_schreier_generator" << endl;
		}
		group_order(go);
		A->Group_element->element_move(Elt3, Elt2, 0);
		if (strip_and_add(Elt2, Elt1 /* residue */,
				verbose_level - 3)) {
			group_order(go1);
			if (f_vv) {
				cout << "closure_group: iteration " << i
						<< " the group has been extended, old order "
					<< go << " extended group order " << go1 << endl;
				print_transversal_lengths();
			}
			if (f_v3) {
				cout << "original element:" << endl;
				A->Group_element->element_print_quick(Elt3, cout);
				cout << "additional generator:" << endl;
				A->Group_element->element_print_quick(Elt2, cout);
			}
			f_extended = true;
		}
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_extended) {
		if (f_v) {
			cout << "sims::closure_group group order extended from "
					<< old_go << " to " << go1 << endl;
			if (f_vv) {
				print_transversal_lengths();
			}
		}
	}
	else {
		if (f_vv) {
			cout << "sims::closure_group group order stays at "
					<< old_go << endl;
		}
	}
	return f_extended;
}



}}}

